"""Verify the IC remainder hypothesis: the Picard check should be
E_prev + int_rem ⊆ E, not just int_rem ⊆ E."""
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
dt = 0.05
delta = 1e-4
eps = 1e-2
t_order = 5

ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=1)

fpg = BasicTMFlowpipeGenerator(sys)
fpg._initialize(t0, 7.0, tmx, dt, t_order=t_order, delta=delta, eps=eps)

# Manually run step 0
print("="*60)
print("STEP 0")
print("="*60)
tmi = tmx  # initial TM
E_prev_0 = tmi.remainder
print(f"E_prev (step 0 IC remainder): {E_prev_0}")

dt_actual, tmf, poly0, contractive0 = fpg._step(jnp.float64(t0), tmi, jnp.float64(dt))
print(f"contractive: {contractive0}")
print(f"tube remainder E: {poly0.remainder}")
print(f"E - E_prev: {irx.interval(poly0.remainder.lower - E_prev_0.lower, poly0.remainder.upper - E_prev_0.upper)}")

# Check: Picard remainder + E_prev vs E
picard_0 = fpg._picard(poly0)
print(f"Picard remainder: {picard_0.remainder}")

wrong_check = check_containment(picard_0.remainder, poly0.remainder)
correct_total = irx.interval(
    picard_0.remainder.lower + E_prev_0.lower,
    picard_0.remainder.upper + E_prev_0.upper,
)
correct_check = check_containment(correct_total, poly0.remainder)
print(f"Wrong check (int_rem ⊆ E): {wrong_check}")
print(f"Correct check (E_prev + int_rem ⊆ E): {correct_check}")
print(f"  correct total: {correct_total}")
print(f"  tube remainder: {poly0.remainder}")

# Step 1
print("\n" + "="*60)
print("STEP 1")
print("="*60)
tmi1 = tmf  # spatial TM at t=0.05
E_prev_1 = tmi1.remainder
print(f"E_prev (step 1 IC remainder): {E_prev_1}")

dt_actual1, tmf1, poly1, contractive1 = fpg._step(jnp.float64(t0 + dt), tmi1, jnp.float64(dt))
print(f"contractive: {contractive1}")
print(f"tube remainder E: {poly1.remainder}")

margin = irx.interval(
    poly1.remainder.lower - E_prev_1.lower,
    poly1.remainder.upper - E_prev_1.upper,
)
print(f"E - E_prev (margin): {margin}")

picard_1 = fpg._picard(poly1)
print(f"Picard remainder: {picard_1.remainder}")

wrong_check = check_containment(picard_1.remainder, poly1.remainder)
correct_total = irx.interval(
    picard_1.remainder.lower + E_prev_1.lower,
    picard_1.remainder.upper + E_prev_1.upper,
)
correct_check = check_containment(correct_total, poly1.remainder)
print(f"Wrong check (int_rem ⊆ E): {wrong_check}")
print(f"Correct check (E_prev + int_rem ⊆ E): {correct_check}")
print(f"  correct total: {correct_total}")
print(f"  tube remainder: {poly1.remainder}")

# How many extra inflations would the correct check need?
test_poly = TaylorModel(
    poly1.coeffs, poly1.exponents, E_prev_1,  # start with E_prev
    poly1.flat_domain, poly1.flat_center,
    _input_pytree=poly1._input_pytree,
    _output_pytree=poly1._output_pytree,
    _per_leaf_order=poly1._per_leaf_order,
)

def _inflate(rem):
    return rem * irx.icentpert(1.0, eps) + irx.icentpert(0.0, delta)

print("\nInflation iterations needed for correct check:")
for i in range(20):
    test_poly.remainder = _inflate(test_poly.remainder)
    picard_i = fpg._picard(test_poly)
    total_i = irx.interval(
        picard_i.remainder.lower + E_prev_1.lower,
        picard_i.remainder.upper + E_prev_1.upper,
    )
    ok = check_containment(total_i, test_poly.remainder)
    print(f"  iter {i}: E={test_poly.remainder}, correct_check={ok}")
    if ok == 1:
        print(f"  Needed {i+1} inflations for correct check (vs 1 for wrong check)")
        break
