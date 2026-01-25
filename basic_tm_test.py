
import jax
import jax.numpy as jnp
from immrax.inclusion import interval, icentpert
from immrax.generator.sets.taylor_model import TaylorModel, taylor_model
from immrax.inclusion.tm import nattm

def verify_tm():
    print("Verifying tm.py implementation...")

    # 1. Setup a simple Taylor Model
    # 1D domain [-1, 1], center 0, radius 1
    # TM = x (identity)
    # coeffs: [0, 1] for order 1, or [0, 1, 0...] for higher
    
    # We need to construct a valid TM manually first
    # Let's use a helper if available, or build raw
    
    # Assume 1d input, order 2
    d = 1
    order = 2
    domain_center = jnp.array([0.0])
    domain_radius = jnp.array([1.0])
    
    # x = 0 + 1*u + 0*u^2 ...
    # coeffs shape: (1, num_monomials)
    # num_monomials for d=1, order=2 is 3: [0], [1], [2]
    # coeffs: [0, 1, 0]
    
    # Construct TM for 'x'
    # Use nattm on identity function to get it? 
    # No, nattm expects TM inputs.
    
    # Let's look at how to create a TM. 
    # tm.py uses _tm_from_interval or manual init.
    # We can try to use a simple creation if possible.
    
    # Let's create a TM representing 'x' on [-1, 1]
    # x(u) = 0 + 1*u
    # normalized u is in [-1, 1], so x = 0 + 1*u maps to [-1, 1] if radius is 1.
    
    from immrax.generator.sets.taylor_model import _get_canonical_exponents
    exponents = _get_canonical_exponents(d, order)
    # exponents: [[0], [1], [2]]
    
    coeffs = jnp.array([[0.0, 1.0, 0.0]]) # 1*u
    remainder = interval(jnp.array([0.0]), jnp.array([0.0]))
    
    tm_x = TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=order)
    
    print("Created TM x:", tm_x)
    

    # ... (previous setup)
    
    print("Created TM x:", tm_x)
    
    # helper to print coeffs
    def print_coeffs(name, tm):
        print(f"{name} coeffs:\n{tm.coeffs}")
        print(f"{name} remainder: {tm.remainder}")

    # 2. Test Identity
    f_id = lambda x: x
    f_tm = nattm(f_id)
    res_id = f_tm(tm_x)
    print_coeffs("Identity", res_id)
    # Check identity: should be same as input
    assert jnp.allclose(res_id.coeffs, tm_x.coeffs), "Identity failed"
    
    # 3. Test Square
    f_sq = lambda x: x*x
    f_tm_sq = nattm(f_sq)
    res_sq = f_tm_sq(tm_x)
    print_coeffs("Square", res_sq)
    
    # Expected: x^2 corresponds to monomial [2]
    # exponents for d=1, order=2 are [[0], [1], [2]]
    # index 2 is x^2
    expected_sq = jnp.array([[0.0, 0.0, 1.0]])
    assert jnp.allclose(res_sq.coeffs, expected_sq), f"Square failed. Got {res_sq.coeffs}"
    
    # 4. Test Sin
    f_sin = lambda x: jnp.sin(x)
    f_tm_sin = nattm(f_sin)
    res_sin = f_tm_sin(tm_x)
    print_coeffs("Sin", res_sin)
    
    # sin(x) = x - x^3/6 ... but order is 2, so sin(x) ~ x
    # coeff for x (idx 1) should be 1.0
    # coeff for x^2 (idx 2) should be 0.0 (since sin is odd)
    # remainder should capture higher order terms
    assert jnp.allclose(res_sin.coeffs[:, 1], 1.0), "Sin linear term failed"
    assert jnp.allclose(res_sin.coeffs[:, 2], 0.0), "Sin quadratic term failed"
    
    # 5. Test Composite
    def f_comp(x):
        # y = x^2 + 1
        # z = sin(y) = sin(1 + x^2)
        # approx sin(1) + cos(1) * x^2 ...
        y = x * x + 1
        z = jnp.sin(y)
        return z
        
    f_tm_comp = nattm(f_comp)
    res_comp = f_tm_comp(tm_x)
    print_coeffs("Composite", res_comp)
    
    # Check constant term ~ sin(1)
    const_idx = 0
    expected_const = jnp.sin(1.0)
    assert jnp.allclose(res_comp.coeffs[:, 0], expected_const), "Composite const term failed"
    
    # Check x^2 term ~ cos(1) * 1
    # d/dx sin(1+x^2) = cos(1+x^2) * 2x -> at x=0 is 0?
    # Wait, Taylor expansion calculation:
    # f(x) = sin(1 + x^2)
    # f(0) = sin(1)
    # f'(x) = cos(1 + x^2) * 2x -> f'(0) = 0
    # f''(x) = -sin(1 + x^2) * (2x)^2 + cos(1 + x^2) * 2 -> f''(0) = 2 * cos(1)
    # Taylor: f(0) + f'(0)x + f''(0)/2 x^2
    #       = sin(1) + 0*x + cos(1)*x^2
    
    expected_x2 = jnp.cos(1.0)
    assert jnp.allclose(res_comp.coeffs[:, 2], expected_x2), f"Composite x^2 term failed. Got {res_comp.coeffs[:, 2]}, expected {expected_x2}"

    print("Verification complete - Numerical checks passed.")

if __name__ == "__main__":
    verify_tm()
