"""Check pointwise TM evaluation vs reference at failing steps."""
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
tf = 2.0  # shorter horizon
dt = 0.05
delta = 1e-4
eps = 1e-2
t_order = 5

# Reference: solve with very tight tolerances
edge_x0s = [[-2.1, 0.0], [-1.9, 0.0], [-2.0, 0.0]]
t_eval = np.arange(t0, tf + dt, dt)
ref_sols = []
for x0 in edge_x0s:
    sol = solve_ivp(vdp_np, [t0, tf], x0, t_eval=t_eval,
                    rtol=1e-14, atol=1e-14, method='DOP853')
    ref_sols.append(sol)

ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=1)
fpg = BasicTMFlowpipeGenerator(sys)
fp = fpg.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=t_order, delta=delta, eps=eps)

print(f"nsteps: {fp.nsteps}, success: {fp.success}")

# For each step, check POINTWISE TM evaluation at x0=[-1.9, 0] and x0=[-2.1, 0]
x_test_points = [
    jnp.array([-2.1, 0.0]),  # lower edge
    jnp.array([-1.9, 0.0]),  # upper edge
]

for step in range(min(int(fp.nsteps), 15)):
    t_end = float(fp.times[step])
    if jnp.isnan(t_end):
        break

    spatial = tx_tm_eval(fp[step], t_end, t_order)

    ref_idx = np.argmin(np.abs(t_eval - t_end))

    print(f"\nStep {step} (t={t_end:.4f}):")
    print(f"  remainder: {spatial.remainder}")

    for xi, x_test in enumerate(x_test_points):
        # Evaluate TM at this specific point
        tm_eval = spatial.evaluate(x_test)
        tm_poly = spatial.evaluate_polynomial(x_test)

        # Reference
        actual = jnp.array(ref_sols[xi].y[:, ref_idx])

        # Pointwise containment
        in_tm = bool(jnp.all(actual >= tm_eval.lower) & jnp.all(actual <= tm_eval.upper))

        # Error: actual - polynomial center
        poly_error = actual - tm_poly

        print(f"  x0={x_test}:")
        print(f"    poly(x0)     = {tm_poly}")
        print(f"    actual       = {actual}")
        print(f"    poly_error   = {poly_error}")
        print(f"    tm_eval      = {tm_eval}")
        print(f"    |poly_error| vs |rem|: {jnp.abs(poly_error)} vs {jnp.maximum(jnp.abs(spatial.remainder.lower), jnp.abs(spatial.remainder.upper))}")
        print(f"    pointwise ok = {in_tm}")
