"""Diagnose order=1 flowpipe containment failure."""
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

# Compute actual trajectories from edge initial conditions using scipy
edge_x0s = [[-2.1, 0.0], [-1.9, 0.0], [-2.0, 0.0]]
t_eval = np.arange(t0, tf + dt, dt)

ref_sols = []
for x0 in edge_x0s:
    sol = solve_ivp(vdp_np, [t0, tf], x0, t_eval=t_eval, rtol=1e-12, atol=1e-14)
    ref_sols.append(sol)
    print(f"x0={x0}: sol.success={sol.success}, final={sol.y[:,-1]}")

# Test with order=1 vs order=3
for order in [1, 3]:
    print(f"\n{'='*60}")
    print(f"ORDER = {order}")
    print(f"{'='*60}")

    ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
    tmx = irx.taylor_model_identity(ix0, order=order)

    fpg = BasicTMFlowpipeGenerator(sys)
    fp = fpg.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=5, delta=delta, eps=eps)
    print(f"nsteps: {fp.nsteps}, success: {fp.success}")

    # Check containment at each step
    first_fail = None
    t_order = fp._per_leaf_order[0]

    for step in range(min(int(fp.nsteps), 200)):
        t_end = float(fp.times[step])
        if jnp.isnan(t_end):
            break

        spatial_end = tx_tm_eval(fp[step], t_end, t_order)
        hull = spatial_end.interval_hull()

        # Find closest reference time index
        ref_idx = np.argmin(np.abs(t_eval - t_end))

        # Check each edge IC
        for i, x0 in enumerate(edge_x0s):
            actual = jnp.array(ref_sols[i].y[:, ref_idx])
            in_hull = bool(jnp.all(actual >= hull.lower) & jnp.all(actual <= hull.upper))
            if not in_hull:
                if first_fail is None:
                    first_fail = step
                    margin_lo = actual - hull.lower
                    margin_hi = hull.upper - actual
                    print(f"\n  FIRST FAIL at step {step} (t={t_end:.4f}):")
                    print(f"    x0={x0} -> actual={actual}")
                    print(f"    hull: {hull.lower} to {hull.upper}")
                    print(f"    remainder: {spatial_end.remainder}")
                    print(f"    margin_lo={margin_lo}")
                    print(f"    margin_hi={margin_hi}")
                break

        if first_fail is not None and step > first_fail + 5:
            break

    if first_fail is None:
        print(f"  All {int(fp.nsteps)} steps passed containment check!")

    # Detailed view around the failure
    if first_fail is not None:
        print(f"\n  Steps around failure ({max(0,first_fail-3)} to {first_fail+3}):")
        for step in range(max(0, first_fail - 3), min(int(fp.nsteps), first_fail + 4)):
            t_end = float(fp.times[step])
            spatial_end = tx_tm_eval(fp[step], t_end, t_order)
            hull = spatial_end.interval_hull()
            hull_width = hull.upper - hull.lower
            rem_width = spatial_end.remainder.upper - spatial_end.remainder.lower
            ref_idx = np.argmin(np.abs(t_eval - t_end))

            print(f"    step {step}: t={t_end:.4f}")
            print(f"      hull_w={hull_width}, rem_w={rem_width}")
            for i, x0 in enumerate(edge_x0s):
                actual = jnp.array(ref_sols[i].y[:, ref_idx])
                margin = jnp.minimum(actual - hull.lower, hull.upper - actual)
                in_hull = bool(jnp.all(actual >= hull.lower) & jnp.all(actual <= hull.upper))
                print(f"      x0={x0}: actual={actual}, min_margin={float(jnp.min(margin)):.6e}, ok={in_hull}")
