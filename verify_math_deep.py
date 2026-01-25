
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import lax
from immrax.inclusion import interval, icentpert
from immrax.generator.sets.taylor_model import TaylorModel
from immrax.inclusion.tm import nattm, tm_inclusion_registry

print("DEBUG: Checking registry keys...")
print(f"asin_p in registry: {lax.asin_p in tm_inclusion_registry}")
# print keys to see what's there
# for k in tm_inclusion_registry:
#     print(f"  {k} ({k.name})")

def check_containment(f_tm, f_ref, domain_min, domain_max, num_samples=1000):
    """
    Checks if TM output contains the true function values over the domain.
    Returns: (passed, violation_max)
    """
    # Sample points
    xs = np.linspace(domain_min, domain_max, num_samples)
    
    # Create TM for x on the domain
    center = (domain_min + domain_max) / 2.0
    radius = (domain_max - domain_min) / 2.0
    
    # x = c + r*u, u in [-1, 1]
    # coeffs: c, r
    d = 1
    order = 4 # Use high order to check if implemented
    from immrax.generator.sets.taylor_model import _get_canonical_exponents
    exponents = _get_canonical_exponents(d, order)
    coeffs = jnp.zeros((1, exponents.shape[1]))
    # Constant term (idx 0 usually)
    coeffs = coeffs.at[:, 0].set(center)
    # Linear term (idx 1 usually)
    coeffs = coeffs.at[:, 1].set(radius)
    
    tm_input = TaylorModel(coeffs, exponents, interval(jnp.array([0.]), jnp.array([0.])), 
                           jnp.array([center]), jnp.array([radius]), _static_order=order)
    
    # Compute TM
    tm_out = f_tm(tm_input)
    
    # Check containment manually
    max_violation = 0.0
    passed = True
    
    for x in xs:
        # Evaluate true function
        y_true = f_ref(x)
        
        # Evaluate TM bounds at x
        # TM evaluate returns interval
        y_interval = tm_out.evaluate(jnp.array([x]))
        
        lower = y_interval.lower[0]
        upper = y_interval.upper[0]
        
        if y_true < lower - 1e-6 or y_true > upper + 1e-6:
            passed = False
            violation = max(lower - y_true, y_true - upper)
            max_violation = max(max_violation, violation)
            # print(f"Violation at x={x}: true={y_true}, interval=[{lower}, {upper}]")
            
    return passed, max_violation, tm_out

def verify_log_math():
    print("\n--- Verifying Log Math ---")
    # Test log(x) for x in [0.9, 1.1] -> u in [-0.1, 0.1]
    # Remainder should be tight.
    # Test log(x) for x in [0.1, 1.9] -> u in [-0.9, 0.9]
    # Near singularity 0.
    
    def test_range(low, high):
        print(f"Testing log(x) on [{low}, {high}]")
        f_ref = lambda x: np.log(x)
        f_tm = nattm(jnp.log)
        passed, viol, tm = check_containment(f_tm, f_ref, low, high)
        print(f"Containment: {'PASS' if passed else 'FAIL'}, Max Violation: {viol:.2e}")
        print(f"TM Remainder width: {tm.remainder.width[0]:.2e}")
        if not passed:
            print("!!! log bound failed !!!")
            
    test_range(0.9, 1.1)
    test_range(0.1, 1.0) # Dangerous range for Taylor series if not careful

def verify_asin_order():
    print("\n--- Verifying Asin Order ---")
    # Check if higher order coeffs are generated
    # arcsin(x) = x + x^3/6 + ...
    # if order 3, we expect x^3 coeff to be non-zero (approx 1/6)
    
    f_tm = nattm(jnp.arcsin, max_order=3)
    
    # Input: x on [-0.5, 0.5]
    center = 0.0
    radius = 0.5
    d=1
    order=3
    from immrax.generator.sets.taylor_model import _get_canonical_exponents
    exponents = _get_canonical_exponents(d, order)
    coeffs = jnp.zeros((1, exponents.shape[1]))
    coeffs = coeffs.at[:, 1].set(radius) # x = 0.5 u
    
    tm_in = TaylorModel(coeffs, exponents, interval(jnp.array([0.]), jnp.array([0.])), 
                        jnp.array([center]), jnp.array([radius]), _static_order=order)
    
    tm_out = f_tm(tm_in)
    
    # Expected: arcsin(0.5 u) = 0.5 u + (0.5 u)^3 / 6 + ...
    # = 0.5 u + 0.02083 u^3
    
    # Coeffs layout: [0]=const, [1]=u, [2]=u^2, [3]=u^3 (assuming 1D canonical order)
    
    print(f"Coeffs: {tm_out.coeffs}")
    
    u3_coeff = tm_out.coeffs[0, 3]
    expected_u3 = (0.5**3) / 6.0
    print(f"Expected u^3 coeff: {expected_u3}")
    print(f"Actual u^3 coeff: {u3_coeff}")
    
    if abs(u3_coeff) < 1e-6 and abs(expected_u3) > 1e-4:
        print("FAIL: Asin missing higher order terms!")
    elif abs(u3_coeff - expected_u3) < 1e-4:
        print("PASS: Asin matches higher order terms.")
    else:
        print("FAIL: Asin coefficients mismatch (wrong value).")

def verify_atan_order():
    print("\n--- Verifying Atan Order ---")
    # arctan(x) = x - x^3/3 + ...
    f_tm = nattm(jnp.arctan, max_order=3)
    
    center = 0.0
    radius = 0.5
    order = 3
    from immrax.generator.sets.taylor_model import _get_canonical_exponents
    d=1
    exponents = _get_canonical_exponents(d, order)
    coeffs = jnp.zeros((1, exponents.shape[1]))
    coeffs = coeffs.at[:, 1].set(radius)
    
    tm_in = TaylorModel(coeffs, exponents, interval(jnp.array([0.]), jnp.array([0.])), 
                        jnp.array([center]), jnp.array([radius]), _static_order=order)
    
    tm_out = f_tm(tm_in)
    
    # Expected: 0.5 u - (0.5 u)^3 / 3
    u3_coeff = tm_out.coeffs[0, 3]
    expected_u3 = -(0.5**3) / 3.0
    print(f"Expected u^3 coeff: {expected_u3}")
    print(f"Actual u^3 coeff: {u3_coeff}")
    
    if abs(u3_coeff) < 1e-6 and abs(expected_u3) > 1e-4:
        print("FAIL: Atan missing higher order terms!")

if __name__ == "__main__":
    verify_log_math()
    verify_asin_order()
    verify_atan_order()
