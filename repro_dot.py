
import jax
import jax.numpy as jnp
from jax import lax
import collections
# Mock TaylorModel
class TaylorModel:
    def __init__(self, coeffs, exponents, remainder, domain_center, domain_radius, _static_order):
        self.coeffs = coeffs
        self.exponents = exponents
        self.remainder = remainder
        self.domain_center = domain_center
        self.domain_radius = domain_radius
        self._static_order = _static_order
# Mock Interval
class Interval:
    def __init__(self, lower, upper=None):
        if upper is None:
            upper = lower
        self.lower = lower
        self.upper = upper
        
def interval(lower, upper=None):
    return Interval(lower, upper)
# Mock inclusion_registry
# We need to simulate the interval dot general
def mock_interval_dot_general(lhs, rhs, dimension_numbers):
    #lhs is interval(arr, arr) -> Point interval
    #rhs is interval(rem) -> Error interval
    # simple implementation: |A| * |R|
    # But since lhs is thin, it is A * R.
    # interval(A) * interval(R_low, R_high)
    # This is complex to implement fully, but we just want to ensure the function runs.
    # Let's just return a dummy interval with correct shape.
    
    # Calculate shape
    # We can use lax.dot_general on the shapes
    l_shape = lhs.lower.shape
    r_shape = rhs.lower.shape
    # We can just run dot_general on the lower bounds to get the shape
    res_lower = lax.dot_general(lhs.lower, rhs.lower, dimension_numbers)
    return Interval(jnp.zeros_like(res_lower), jnp.zeros_like(res_lower))
mock_registry = {
    lax.dot_general_p: mock_interval_dot_general
}
# The target function to test
def _tm_dot_general_array(arr, tm, dim_nums):
    """Handle dot_general(Array, TM) — the primary Array-TM implementation.
    The monomial axis (last axis of tm.coeffs) is on the rhs, so it
    naturally ends up at the end of the dot_general output.
    """
    # Mocking the import
    # from immrax.inclusion.nif import inclusion_registry
    inclusion_registry = mock_registry
    new_coeffs = lax.dot_general(arr, tm.coeffs, dimension_numbers=dim_nums)
    new_remainder = inclusion_registry[lax.dot_general_p](
        interval(arr, arr), tm.remainder, dimension_numbers=dim_nums
    )
    return TaylorModel(
        new_coeffs, tm.exponents, new_remainder,
        tm.domain_center, tm.domain_radius, _static_order=tm._static_order,
    )
def test_dot_general():
    print("Testing _tm_dot_general_array with mocks...")
    
    # A: (2, 2)
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    # TM B: shape (2,)
    # coeffs: (2, 3). 3 monomials.
    coeffs = jnp.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    
    # remainder: (2,)
    rem = interval(jnp.zeros(2), jnp.ones(2))
    
    # dummies
    exponents = jnp.zeros((3, 2)) # dummy
    center = jnp.zeros(2)
    radius = jnp.ones(2)
    
    tm = TaylorModel(coeffs, exponents, rem, center, radius, _static_order=2)
    
    # dim_nums for dot(A, B): 
    # contraction: (1,), (0,)  -> Contract dim 1 of A with dim 0 of B
    # batch: (), ()
    dim_nums = (((1,), (0,)), ((), ()))
    
    # Call the function
    res_tm = _tm_dot_general_array(A, tm, dim_nums)
    
    print("Result coeffs shape:", res_tm.coeffs.shape)
    print("Result coeffs values:\n", res_tm.coeffs)
    
    expected_coeffs = jnp.array([
        [5.0, 50.0, 500.0],
        [11.0, 110.0, 1100.0]
    ])
    
    diff = jnp.max(jnp.abs(res_tm.coeffs - expected_coeffs))
    print(f"Max difference: {diff}")
    
    if diff < 1e-5:
        print("SUCCESS: Coefficients match expected values.")
    else:
        print("FAILURE: Coefficients do not match.")
if __name__ == "__main__":
    test_dot_general()
