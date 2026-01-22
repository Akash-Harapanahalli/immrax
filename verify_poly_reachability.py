
import jax
import jax.numpy as jnp
from immrax.system import System
from immrax.generator.reachability import AlthoffGirardReachability, LohnerReachability
from immrax.generator.sets.zonotope import Zonotope
from immrax.generator.sets.constrained_zonotope import constrained_zonotope
from immrax.generator.sets.polynomial_zonotope import PolynomialZonotope

class HarmOsc(System):
    def __init__(self):
        super().__init__(2)
    def f(self, t, x):
        return jnp.array([x[1], -x[0]])

def test_reachability():
    sys = HarmOsc()
    dt = 0.01
    t0 = 0.0
    steps = 10
    
    # 0. Initial Set (Zonotope)
    c = jnp.array([1.0, 0.0])
    G = jnp.diag(jnp.array([0.1, 0.1]))
    Z0 = Zonotope(c, G)
    
    print("--- Testing Zonotope Reachability ---")
    alg_z = AlthoffGirardReachability(sys, dt, target_order=10)
    res_z = alg_z.compute_reach_sets(t0, steps, Z0)
    print(f"Computed {len(res_z)} steps.")
    print(f"Final Center: {res_z.sets[-1].ox}")
    
    # 1. Constrained Zonotope
    print("\n--- Testing ConstrainedZonotope Reachability ---")
    CZ0 = constrained_zonotope(c, G) # Helper converts Z to CZ
    alg_cz = AlthoffGirardReachability(sys, dt, target_order=10)
    res_cz = alg_cz.compute_reach_sets(t0, steps, CZ0)
    print(f"Computed {len(res_cz)} steps.")
    print(f"Final Center: {res_cz.sets[-1].ox}")
    
    # 2. Polynomial Zonotope
    print("\n--- Testing PolynomialZonotope Reachability ---")
    # Manually create PZ from Z logic
    # PZ = c + G * eps + G_I * beta
    # Here just G * independent
    PZ0 = PolynomialZonotope(c, jnp.zeros((2,0)), jnp.zeros((0,0), int), G)
    alg_pz = AlthoffGirardReachability(sys, dt, target_order=10)
    res_pz = alg_pz.compute_reach_sets(t0, steps, PZ0)
    print(f"Computed {len(res_pz)} steps.")
    print(f"Final Center: {res_pz.sets[-1].ox}")
    
    # Check consistency
    diff = jnp.linalg.norm(res_z.sets[-1].ox - res_cz.sets[-1].ox)
    print(f"\nDifference Z vs CZ centers: {diff}")
    assert diff < 1e-6, "Z and CZ should match for linear dynamics/linearization center"

if __name__ == "__main__":
    test_reachability()
