"""Verify the fix: order=1 should now contain the flow."""
import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator
from immrax.taylor.algorithms.base import tx_tm_eval
from scipy.integrate import solve_ivp
import numpy as np

jax.config.update("jax_enable_x64", True)


def vdp_np(t, x):
    mu = 1.0
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


class VDP(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x):
        mu = 1.0
        return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = VDP()
t0 = 0.0
tf = 7.0
dt = 0.05
delta = 1e-4
eps = 1e-2

# Reference trajectories (high accuracy)
edge_x0s = [[-2.1, 0.0], [-1.9, 0.0], [-2.0, 0.0]]
t_eval = np.arange(t0, tf + dt, dt)
ref_sols = []
for x0 in edge_x0s:
    sol = solve_ivp(vdp_np, [t0, tf], x0, t_eval=t_eval,
                    rtol=1e-13, atol=1e-14, method='DOP853')
    ref_sols.append(sol)

for order in [1, 3]:
    print(f"\n{'='*60}")
    print(f"ORDER = {order}")
    print(f"{'='*60}")

    ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
    tmx = irx.taylor_model_identity(ix0, order=order)

    fpg = BasicTMFlowpipeGenerator(sys)
    fp = fpg.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=5, delta=delta, eps=eps)
    print(f"nsteps: {fp.nsteps}, success: {fp.success}")

    # Check pointwise containment at each step
    t_order = fp._per_leaf_order[0]
    all_ok = True
    min_margin = float('inf')

    for step in range(min(int(fp.nsteps), 200)):
        t_end = float(fp.times[step])
        if jnp.isnan(t_end):
            break

        spatial = tx_tm_eval(fp[step], t_end, t_order)
        ref_idx = np.argmin(np.abs(t_eval - t_end))

        for i, x0 in enumerate(edge_x0s):
            actual = jnp.array(ref_sols[i].y[:, ref_idx])
            tm_eval = spatial.evaluate(actual[:1] if False else jnp.array(x0))
            # Pointwise check
            in_tm = bool(jnp.all(actual >= tm_eval.lower) & jnp.all(actual <= tm_eval.upper))
            if not in_tm:
                margin = float(jnp.min(jnp.minimum(actual - tm_eval.lower, tm_eval.upper - actual)))
                if all_ok:
                    print(f"  FIRST FAIL: step {step} (t={t_end:.4f}), x0={x0}")
                    print(f"    actual={actual}")
                    print(f"    tm_eval={tm_eval}")
                    print(f"    margin={margin:.6e}")
                all_ok = False
            else:
                margin = float(jnp.min(jnp.minimum(actual - tm_eval.lower, tm_eval.upper - actual)))
                min_margin = min(min_margin, margin)

    if all_ok:
        print(f"  All steps PASS! min_margin={min_margin:.6e}")
    else:
        print(f"  Some steps FAILED")

    # Check hull containment specifically at some steps
    print(f"\n  Hull containment check at selected steps:")
    for step in [0, 1, 5, 10, 20, 50, 100, int(fp.nsteps)-1]:
        if step >= int(fp.nsteps):
            continue
        t_end = float(fp.times[step])
        if jnp.isnan(t_end):
            break
        spatial = tx_tm_eval(fp[step], t_end, t_order)
        hull = spatial.interval_hull()
        ref_idx = np.argmin(np.abs(t_eval - t_end))

        step_ok = True
        for i, x0 in enumerate(edge_x0s):
            actual = jnp.array(ref_sols[i].y[:, ref_idx])
            in_hull = bool(jnp.all(actual >= hull.lower) & jnp.all(actual <= hull.upper))
            if not in_hull:
                step_ok = False

        rem_width = spatial.remainder.upper - spatial.remainder.lower
        print(f"    step {step} (t={t_end:.4f}): hull_ok={step_ok}, rem_width={rem_width}")
