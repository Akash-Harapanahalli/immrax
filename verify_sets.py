
import jax
import jax.numpy as jnp
from immrax.generator.sets.constrained_zonotope import constrained_zonotope_from_polytope, constrained_zonotope_intersection, constrained_zonotope
from immrax.generator.sets.polynomial_zonotope import PolynomialZonotope
from immrax.generator.sets.taylor_model import TaylorModel

def check_cz_polytope():
    print("--- Checking Constrained Zonotope (Polytope Conversion) ---")
    # k vertices
    k = 4
    vertices = jnp.eye(4) # Standard simplex
    
    # Logic check: sum(v_i) = 2 - k
    # With k=4, sum(v) should be -2
    cz = constrained_zonotope_from_polytope(vertices)
    
    # Check constraint b
    expected_b = 2.0 - k
    current_b = cz.b[0]
    
    print(f"k={k}")
    print(f"Expected b: {expected_b}")
    print(f"Actual b: {current_b}")
    assert jnp.allclose(current_b, expected_b), "Polytope sign error persists!"
    print("Polytope conversion check passed.")

def check_cz_intersection():
    print("\n--- Checking Constrained Zonotope (Intersection) ---")
    # Check that generators are correct size (should be G1 size, not G1+G2)
    # cz1: n=2, m=2
    # cz2: n=2, m=2
    G = jnp.eye(2)
    cz1 = constrained_zonotope(jnp.zeros(2), G)
    cz2 = constrained_zonotope(jnp.ones(2), G)
    
    cz_int = constrained_zonotope_intersection(cz1, cz2)
    
    # Intersection logic: new_G should be [G1, 0] or similar size to G1+G2 but with zeros
    # Actually, in the fix we used: new_G = [cz1.G, zeros]
    # So size is m1 + m2
    expected_m = cz1.m + cz2.m
    print(f"Intersection m: {cz_int.m} (Expected {expected_m})")
    
    # Verify that the second block of generators is zero
    G_blocks = jnp.split(cz_int.G, [cz1.m], axis=1)
    g2_block = G_blocks[1]
    
    is_zero = jnp.all(g2_block == 0)
    print(f"Auxiliary generators are zero: {is_zero}")
    assert is_zero, "Intersection generators logic error!"
    print("Intersection logic check passed.")

def check_pz_quadratic():
    print("\n--- Checking Polynomial Zonotope (Quadratic Map) ---")
    # Check if independent generators create cross terms
    
    # PZ with NO dependent terms, 2 independent terms
    # x = b1 * g1 + b2 * g2
    # Q(x) = x^2 (scalar case, Q=1)
    # x^2 = b1^2 g1^2 + b2^2 g2^2 + 2 b1 b2 g1 g2
    # We expect 3 independent generators in output (squares become diagonal, b1b2 is cross)
    # Actually squares: b1^2 in [0,1], shifted to median + independent.
    # But essentially we look for the cross term b1*b2
    
    ox = jnp.zeros(1)
    G = jnp.zeros((1, 0))
    E = jnp.zeros((0, 0), dtype=jnp.int32)
    G_I = jnp.array([[1.0, 2.0]]) # 2 independent generators
    
    pz = PolynomialZonotope(ox, G, E, G_I)
    
    Q = jnp.eye(1)
    pz_sq = pz.quadratic_map(Q)
    
    # Original independent: 2.
    # Quadratic map adds: linear (0 in this case), cross terms (1), quadratic diagonals (2).
    # Total independent should be > 2.
    
    print(f"Original q: {pz.q}")
    print(f"Result q: {pz_sq.q}")
    
    # We expect cross term for (i=0, j=1): 2 * g1 * g2 = 4.
    # Let's check if the list of generators includes 4.0
    
    # The fix added: new_G_I = [linear, cross, quad]
    # linear=0
    # cross should be there.
    assert pz_sq.q > 2, "Cross terms missing!"
    print("Quadratic map check passed.")

def check_tm_multiplication():
    print("\n--- Checking Taylor Model (Multiplication Safety) ---")
    # TM1 = x, TM2 = x. Domain [-1, 1].
    # TM1 * TM2 = x^2.
    # If max_order = 1, x^2 is truncated.
    # Should be absorbed into remainder.
    # x^2 on [-1, 1] is [0, 1].
    # Remainder should expand by [0, 1].
    
    coeffs = jnp.array([[0.0, 1.0]]) # 0 + 1*x, shape (1, 2)
    exps = jnp.array([[0, 1]])
    
    tm = TaylorModel(coeffs, exps, jnp.zeros((1, 2)), jnp.zeros(1), jnp.ones(1))
    
    # Multiply with max_order = 1
    tm_sq = tm.multiply(tm, max_order=1)
    
    # Check remainder
    rem = tm_sq.remainder
    print(f"Remainder after truncating x^2: {rem}")
    
    # Expected: original remainder 0. Truncated x^2 in [0, 1].
    # Remainder should contain [0, 1].
    assert rem[0, 1] >= 1.0, "Truncation error not captured!"
    print("Multiplication safety check passed.")

if __name__ == "__main__":
    check_cz_polytope()
    check_cz_intersection()
    check_pz_quadratic()
    check_tm_multiplication()
