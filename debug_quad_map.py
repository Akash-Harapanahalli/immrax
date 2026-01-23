
import jax
import jax.numpy as jnp
from immrax.generator.sets.polynomial_zonotope import PolynomialZonotope
import time

def test_quad_map():
    print("Testing PolynomialZonotope quadratic_map JIT...")
    
    n = 2
    h = 5
    q = 5
    
    ox = jnp.zeros(n)
    G = jnp.eye(n, h)
    E = jnp.zeros((1, h), dtype=jnp.int32)
    G_I = jnp.eye(n, q)
    
    pz = PolynomialZonotope(ox, G, E, G_I)
    
    # Quadratic form
    Q = jnp.eye(n)
    
    @jax.jit
    def do_map(pz):
        return pz.quadratic_map(Q)
        
    print("Compiling...")
    start = time.time()
    res = do_map(pz)
    end = time.time()
    print(f"Compilation finished in {end - start:.4f}s")
    print(f"Result shape: G={res.G.shape}, G_I={res.G_I.shape}")
    
    # Test with larger size
    print("\nTesting larger size (h=20, q=20)...")
    h = 20
    q = 20
    G = jnp.eye(n, h)
    E = jnp.zeros((1, h), dtype=jnp.int32)
    G_I = jnp.eye(n, q)
    pz = PolynomialZonotope(ox, G, E, G_I)
    
    @jax.jit
    def do_map_large(pz):
        return pz.quadratic_map(Q)
        
    print("Compiling large...")
    start = time.time()
    res = do_map_large(pz)
    end = time.time()
    print(f"Compilation finished in {end - start:.4f}s")
    print(f"Result shape: G={res.G.shape}, G_I={res.G_I.shape}")

if __name__ == "__main__":
    try:
        test_quad_map()
    except Exception as e:
        import traceback
        traceback.print_exc()
