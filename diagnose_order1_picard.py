"""Check Picard polynomial difference at the failing step."""
import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import (
    BasicTMFlowpipeGenerator, pjetm, pjet,
    TaylorModel, taylor_model_identity, tm_integrate_variable,
)
from immrax.taylor.algorithms.base import tx_tm_eval, tps_to_tx
from immrax.utils import prolongation, check_containment

jax.config.update("jax_enable_x64", True)


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
t_order = 5

ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=1)

fpg = BasicTMFlowpipeGenerator(sys)
fp = fpg.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=t_order, delta=delta, eps=eps)

print(f"nsteps: {fp.nsteps}, success: {fp.success}")

# Look at tubes for steps 0 and 1
for step_idx in range(3):
    print(f"\n{'='*60}")
    print(f"STEP {step_idx}")
    print(f"{'='*60}")

    tube = fp[step_idx]
    t_end = float(fp.times[step_idx])
    print(f"t_end = {t_end}")
    print(f"tube domain: {tube.domain}")
    print(f"tube center: {tube.center}")
    print(f"tube remainder: {tube.remainder}")
    print(f"tube per_leaf_order: {tube._per_leaf_order}")
    print(f"tube coeffs shape: {tube.coeffs.shape}")

    # Manually apply the Picard operator
    tm_ic = tx_tm_eval(tube, tube.domain[0].lower, t_order)
    tm_id = taylor_model_identity(tube.domain, order=tube._per_leaf_order)
    f_tm_tx = pjetm(sys.f)(tm_id[0:1], tube)
    tm_int = tm_integrate_variable(f_tm_tx, var_idx=0, keep_order=True)

    con_sh = tm_ic.coeffs.shape
    K_coeffs = tm_int.coeffs.at[:con_sh[0], :con_sh[1]].add(tm_ic.coeffs)
    K_remainder = tm_int.remainder

    # Polynomial difference
    poly_diff = K_coeffs - tube.coeffs
    print(f"\n  Picard K(Y) polynomial diff (K.poly - Y.poly):")
    print(f"    max |diff|: {float(jnp.max(jnp.abs(poly_diff))):.6e}")
    print(f"    diff:\n{poly_diff}")

    # Bound poly diff over domain
    # The polynomial difference is itself a polynomial in the same variables
    # We need to bound it over the domain
    from immrax.taylor.taylor_model import _bound_monomials_over_domain
    mono_bounds = _bound_monomials_over_domain(
        tube.multiindices, tube.shifted_domain, max(tube._per_leaf_order)
    )

    # poly_diff has shape (n_out, n_monomials)
    # mono_bounds has shape (n_monomials,) intervals
    # Bound: sum_j poly_diff[i,j] * mono_bounds[j]
    d_pos = jnp.maximum(poly_diff, 0.0)
    d_neg = jnp.minimum(poly_diff, 0.0)
    poly_diff_lower = jnp.sum(d_pos * mono_bounds.lower + d_neg * mono_bounds.upper, axis=-1)
    poly_diff_upper = jnp.sum(d_pos * mono_bounds.upper + d_neg * mono_bounds.lower, axis=-1)
    poly_diff_bound = irx.interval(poly_diff_lower, poly_diff_upper)

    print(f"\n  Poly diff bound over domain: {poly_diff_bound}")
    print(f"  K(Y) remainder: {K_remainder}")
    print(f"  Y remainder: {tube.remainder}")

    # Total K(Y) - Y.poly bound
    total_K_minus_Y = irx.interval(
        K_remainder.lower + poly_diff_lower,
        K_remainder.upper + poly_diff_upper,
    )
    print(f"\n  Total K(Y) - Y.poly (poly_diff + K.rem): {total_K_minus_Y}")

    # Check containment
    rem_only = check_containment(K_remainder, tube.remainder)
    total_check = check_containment(total_K_minus_Y, tube.remainder)
    print(f"\n  Remainder-only check: {rem_only}")
    print(f"  Total (poly_diff + rem) check: {total_check}")

    # Evaluate at the end time to see spatial TM
    spatial = tx_tm_eval(tube, t_end, t_order)
    hull = spatial.interval_hull()
    print(f"\n  Spatial hull at t_end={t_end}: {hull}")
