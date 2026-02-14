"""Verify shrink wrapping preserves interval hull at each step."""

import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator, BungerTMFlowpipeGenerator
from immrax.taylor.algorithms.base import tx_tm_eval
from immrax.taylor.algorithms.bunger import BungerTMFlowpipeGenerator as BTFG


class VDP(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x):
        mu = 1.0
        return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = VDP()
t0 = 0.0
tf = 1.0  # shorter horizon
ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=2)
dt = 0.05

# Run Basic (no shrink wrap) to get reference tubes
fpg_basic = BasicTMFlowpipeGenerator(sys)

@jax.jit
def gen_basic(t0, tf, tmx, dt, delta, eps):
    return fpg_basic.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps)

fp_basic = gen_basic(t0, tf, tmx, dt, 1e-4, 1e-2)

# Extract spatial TMs from Basic and test shrink wrapping
print("=== Shrink wrap hull preservation test ===\n")
for step_i in range(min(10, int(fp_basic.nsteps))):
    tube = fp_basic[step_i]
    t_end = float(fp_basic.times[step_i])
    spatial = tx_tm_eval(tube, t_end, fp_basic._per_leaf_order[0])

    original_hull = spatial.interval_hull()
    wrapped = BTFG._shrink_wrap(spatial)
    wrapped_hull = wrapped.interval_hull()

    # Check: wrapped hull should contain original hull
    lo_ok = jnp.all(wrapped_hull.lower <= original_hull.lower + 1e-12)
    hi_ok = jnp.all(wrapped_hull.upper >= original_hull.upper - 1e-12)
    ok = lo_ok and hi_ok

    if not ok:
        gap_lo = original_hull.lower - wrapped_hull.lower  # should be ≥ 0
        gap_hi = wrapped_hull.upper - original_hull.upper  # should be ≥ 0
        print(f"Step {step_i} (t={t_end:.3f}): FAIL")
        print(f"  original hull = [{original_hull.lower}, {original_hull.upper}]")
        print(f"  wrapped hull  = [{wrapped_hull.lower}, {wrapped_hull.upper}]")
        print(f"  gap_lo (should be ≥0) = {gap_lo}")
        print(f"  gap_hi (should be ≥0) = {gap_hi}")
        print(f"  remainder before = [{spatial.remainder.lower}, {spatial.remainder.upper}]")
        print(f"  remainder after  = [{wrapped.remainder.lower}, {wrapped.remainder.upper}]")

        # Debug: show intermediate values
        poly_bounds = spatial._bound_polynomial()
        c0 = spatial.constant_term
        nonconst_lo = poly_bounds.lower - c0
        nonconst_hi = poly_bounds.upper - c0
        poly_radius = (nonconst_hi - nonconst_lo) / 2
        nonconst_center = (nonconst_lo + nonconst_hi) / 2
        print(f"  c0 = {c0}")
        print(f"  nonconst_lo = {nonconst_lo}")
        print(f"  nonconst_hi = {nonconst_hi}")
        print(f"  poly_radius = {poly_radius}")
        print(f"  nonconst_center = {nonconst_center}")
        print(f"  rem_pert = {spatial.remainder.pert}")
        print(f"  rem_center = {spatial.remainder.center}")
    else:
        margin_lo = jnp.min(original_hull.lower - wrapped_hull.lower)
        margin_hi = jnp.min(wrapped_hull.upper - original_hull.upper)
        print(f"Step {step_i} (t={t_end:.3f}): OK  margin_lo={float(margin_lo):.2e}, margin_hi={float(margin_hi):.2e}")
