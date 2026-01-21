
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import random

def check_integral_approximation():
    print("--- Checking Integral of Exponential Approximation ---")
    # Exact: \int_0^t e^{A \tau} d\tau = A^{-1} (e^{At} - I)
    # Approx in code: t*I + 0.5*t^2*A
    
    key = random.PRNGKey(0)
    n = 5
    A = random.normal(key, (n, n))
    dt = 0.1
    
    # Exact calculation using block matrix expm
    # exp([ [A, I], [0, 0] ] * t) = [ [e^At, \int e^At], [0, I] ]
    M = jnp.zeros((2*n, 2*n))
    M = M.at[:n, :n].set(A)
    M = M.at[:n, n:].set(jnp.eye(n))
    
    expM = jax.scipy.linalg.expm(M * dt)
    exact_int_Phi = expM[:n, n:]
    
    # Code approximation
    approx_int_Phi = dt * jnp.eye(n) + 0.5 * (dt**2) * A
    
    diff = jnp.linalg.norm(exact_int_Phi - approx_int_Phi)
    rel_diff = diff / jnp.linalg.norm(exact_int_Phi)
    
    print(f"A norm: {jnp.linalg.norm(A)}")
    print(f"dt: {dt}")
    print(f"Difference L2 norm: {diff}")
    print(f"Relative difference: {rel_diff}")
    
    # Try with larger A or dt
    dt_large = 1.0
    expM_large = jax.scipy.linalg.expm(M * dt_large)
    exact_large = expM_large[:n, n:]
    approx_large = dt_large * jnp.eye(n) + 0.5 * (dt_large**2) * A
    print(f"Large dt ({dt_large}) Relative Diff: {jnp.linalg.norm(exact_large - approx_large) / jnp.linalg.norm(exact_large)}")
    
    if rel_diff > 1e-3:
        print("CONCLUSION: Approximation is valid only for very small dt or A. It is only 2nd order.")
    else:
        print("CONCLUSION: Approximation might be acceptable for small steps.")

def check_hessian_bound():
    print("\n--- Checking Hessian Bound Logic ---")
    # Code logic for bound of 0.5 * (x-c)^T H (x-c):
    # h_max = max(abs(H_entries))
    # dx_sq_sum = sum( (x-c)^2 )
    # bound = 0.5 * h_max * dx_sq_sum
    
    # We want to check if |0.5 * v^T H v| <= 0.5 * h_max * ||v||^2
    # This is equivalent to checking if |v^T H v| <= h_max * ||v||^2
    # For v = [1, 1, ...], ||v||^2 = n.
    # If H is all ones, v^T H v = n^2.
    # Bound says: 1 * n = n.
    # Actual n^2. So bound is unsafe by factor of n.
    
    n = 10
    v = jnp.ones(n)
    H = jnp.ones((n, n))
    
    quad_form = jnp.abs(v @ H @ v) # n^2 = 100
    
    h_max = jnp.max(jnp.abs(H)) # 1
    v_sq_sum = jnp.sum(v**2) # n = 10
    
    bound = h_max * v_sq_sum # 10
    
    print(f"Dimension n: {n}")
    print(f"H: all ones")
    print(f"v: all ones")
    print(f"Actual Quadratic Form |v^T H v|: {quad_form}")
    print(f"Code Bound (h_max * ||v||^2): {bound}")
    
    if bound < quad_form:
        print("CONCLUSION: Bound is UNSAFE (Underestimation).")
    else:
        print("CONCLUSION: Bound is safe.")

if __name__ == "__main__":
    check_integral_approximation()
    check_hessian_bound()
