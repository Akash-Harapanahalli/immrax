
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import random

def check_integral_approximation():
    print("--- Checking Integral of Exponential Approximation ---")
    
    key = random.PRNGKey(0)
    n = 5
    A = random.normal(key, (n, n))
    dt = 0.5 # Larger dt to test accuracy
    
    # Exact calculation using block matrix expm
    M = jnp.zeros((2*n, 2*n))
    M = M.at[:n, :n].set(A * dt)
    M = M.at[:n, n:].set(jnp.eye(n) * dt)
    
    expM = jax.scipy.linalg.expm(M)
    exact_int_Phi = expM[:n, n:]
    
    # Code now should match this exactly. 
    # This test verifies that the proposed "exact" method is indeed exact/consistent.
    
    # Block matrix method IS the fix, so comparing it to itself is trivial but validates the logic.
    # Let's compare to the old approximation to show improvement.
    approx_int_Phi = dt * jnp.eye(n) + 0.5 * (dt**2) * A
    
    diff = jnp.linalg.norm(exact_int_Phi - approx_int_Phi)
    rel_diff = diff / jnp.linalg.norm(exact_int_Phi)
    
    print(f"Old approx relative error (dt={dt}): {rel_diff:.4f}")
    if rel_diff > 0.01:
         print("Old approximation was indeed inaccurate.")
    else:
         print("Old approximation was surprisingly okay.")

def check_hessian_bound_safety():
    print("\n--- Checking Hessian Bound Logic Safety ---")
    
    n = 10
    v = jnp.ones(n)
    H = jnp.ones((n, n))
    
    # Actual quadratic form: v^T H v = 1 * n * n = 100
    quad_form = jnp.abs(v @ H @ v)
    
    h_max = jnp.max(jnp.abs(H))
    
    # Old faulty logic: 0.5 * h_max * sum(v^2)
    old_bound = 0.5 * h_max * jnp.sum(v**2) # 0.5 * 1 * 10 = 5
    # Actual error is 0.5 * 100 = 50. Old bound 5 << 50. UNSAFE.
    
    # New logic: 0.5 * h_max * (sum(|v|))^2
    new_bound = 0.5 * h_max * (jnp.sum(jnp.abs(v)))**2 # 0.5 * 1 * 100 = 50
    
    print(f"Actual Quadratic Error: {0.5 * quad_form}")
    print(f"Old Bound: {old_bound} (UNSAFE)" if old_bound < 0.5 * quad_form else f"Old Bound: {old_bound}")
    print(f"New Bound: {new_bound} (SAFE)" if new_bound >= 0.5 * quad_form else f"New Bound: {new_bound} (UNSAFE)")
    
    assert new_bound >= 0.5 * quad_form - 1e-5, "New bound failed safety check!"

def check_polytope_sign():
    print("\n--- Checking Polytope Conversion Sign ---")
    # k vertices
    # sum(v_i) = 2 - k logic
    k = 3
    # If we have 3 vertices, v1, v2, v3 in [-1, 1].
    # w1, w2, w3 in [0, 1], sum(w) = 1.
    # w = (v+1)/2
    # (v1+1)/2 + ... = 1
    # sum(v) + k = 2 => sum(v) = 2 - k.
    # For k=3, sum(v) = -1.
    
    # Previous code: sum(v) = k - 2 = 1.
    # Check bounds:
    # If v = [-1, -1, 1], sum = -1. w = [0, 0, 1]. sum(w)=1. Valid.
    # If v = [1/3, 1/3, 1/3], sum = 1. w = [2/3, ...]. sum(w)=2. Invalid.
    
    # So sum(v) = 2-k is correct.
    print(f"For k={k}, target sum(v) should be {2.0 - k}")
    print("Logic verified mathematically.")

if __name__ == "__main__":
    check_integral_approximation()
    check_hessian_bound_safety()
    check_polytope_sign()
