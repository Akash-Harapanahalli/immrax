
import jax.numpy as jnp
import jax
from immrax.generator.sets.zonotope import Zonotope
from immrax.generator.sets.polynomial_zonotope import PolynomialZonotope
from immrax.generator.sets.constrained_zonotope import ConstrainedZonotope

def test_structure():
    print("Testing Set Structure Addition...")
    
    # 1. Setup
    n = 2
    ox = jnp.zeros(n)
    G = jnp.eye(n)
    
    # Create PZ
    # G dependent, G_I independent
    # PZ with 1 dependent generator
    E = jnp.array([[1], [1]]) # p=2, h=1 ? No E is (p, h)
    # Let p=1, h=1
    E = jnp.array([[1]])
    G_pz = jnp.array([[1.], [0.5]])
    PZ = PolynomialZonotope(ox, G_pz, E)
    print(f"Initial PZ: {PZ}")
    
    # Create CZ
    # G=I, A=[1, -1], b=[0] -> v1 - v2 = 0 -> v1=v2
    A = jnp.array([[1., -1.]])
    b = jnp.array([0.])
    CZ = ConstrainedZonotope(ox, G, A, b)
    print(f"Initial CZ: {CZ}")
    
    # Create Zonotope (Error)
    G_z = jnp.diag(jnp.array([0.1, 0.1]))
    Z = Zonotope(ox, G_z)
    print(f"Zonotope (Error): {Z}")
    
    # 2. Test Addition
    print("\n--- Testing PZ + Z ---")
    PZ_new = PZ + Z
    print(f"Result Type: {type(PZ_new)}")
    print(f"Result: {PZ_new}")
    
    if not isinstance(PZ_new, PolynomialZonotope):
        print("FAIL: Result is not PolynomialZonotope")
    else:
        # Check generators
        # Original G_I was 0. New G_I should be G_z (2 cols)
        if PZ_new.q == 2:
            print("SUCCESS: Separated into Independent Generators (q=2)")
        else:
            print(f"FAIL: Expected q=2, got {PZ_new.q}")

    print("\n--- Testing CZ + Z ---")
    CZ_new = CZ + Z
    print(f"Result Type: {type(CZ_new)}")
    print(f"Result: {CZ_new}")
    
    if not isinstance(CZ_new, ConstrainedZonotope):
        print("FAIL: Result is not ConstrainedZonotope")
    else:
        # Check constraints
        # Original A was (1, 2). New A should be (1, 4) with zeros
        if CZ_new.A.shape == (1, 4):
             # Check if last 2 cols are zero
             last_cols = CZ_new.A[:, 2:]
             if jnp.all(last_cols == 0):
                 print("SUCCESS: Zonotope generators added as unconstrained (zeros in A)")
             else:
                 print(f"FAIL: Expected zeros in A, got {last_cols}")
        else:
            print(f"FAIL: Expected A shape (1, 4), got {CZ_new.A.shape}")

    # 3. Test reduce_order
    print("\n--- Testing Reduce Order ---")
    # Reduce CZ to original size (m=2)
    CZ_reduced = CZ_new.reduce_order(1.0) # order = (m-p)/n? No, target_order is (m-p)/n usually?
    # CZ.reduce_order param is target_order.
    # target_m = target_order * n + p
    # Try to reduce back to 2 generators (target_m=2).
    # target_order = (2 - 1)/2 = 0.5
    
    # But let's just try target_order=1.0 -> target_m = 2 + 1 = 3
    CZ_reduced = CZ_new.reduce_order(1.0)
    print(f"Reduced CZ: {CZ_reduced}")
    print(f"Reduced A shape: {CZ_reduced.A.shape}")
    
    # Check if we kept the constrained generators (which were large=1.0) vs Z generators (small=0.1)
    # Original G: [[1, 0], [0, 1]]. Norms: 1, 1.
    # Z G: [[0.1, 0], [0, 0.1]]. Norms: 0.1, 0.1.
    # Should keep original.
    # But target_m = 3. So keep 2 largest + box of rest?
    # Or keep 3 largest?
    # reduced A should have kept the non-zero cols if we kept G
    print(f"Reduced A: {CZ_reduced.A}")

if __name__ == "__main__":
    test_structure()
