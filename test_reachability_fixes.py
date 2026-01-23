
import jax
import jax.numpy as jnp
from immrax import System
from immrax.generator.algorithms import LohnerReachability, TaylorGirardReachability
from immrax.generator.sets import Zonotope, zonotope_from_interval
from immrax.inclusion import Interval
import time

jax.config.update("jax_enable_x64", True)

class VanderPol(System):
    def __init__(self):
        super().__init__('continuous', 2)
        
    def f(self, t, x):
        return jnp.array([
            x[1],
            (1.0 - x[0]**2)*x[1] - x[0]
        ])

def test_reachability_fixes():
    sys = VanderPol()
    dt = 0.01
    t_end = 0.5
    num_steps = int(t_end / dt)
    
    # Initial set: Box [1.0, 1.1] x [1.0, 1.1]
    # Represented as Zonotope
    center = jnp.array([1.05, 1.05])
    radius = jnp.array([0.05, 0.05])
    z0 = zonotope_from_interval(Interval(center - radius, center + radius))
    
    print("Testing LohnerReachability with QR and Spatial Remainder...")
    lohner = LohnerReachability(sys, dt, taylor_order=3)
    start = time.time()
    res_lohner = lohner.compute_reach_sets(0.0, num_steps, z0)
    print(f"Lohner Time: {time.time() - start:.4f}s")
    
    final_z_lohner = res_lohner.sets[-1]
    print(f"Lohner Final Generators: {final_z_lohner.generators.shape}")
    print(f"Lohner Final Interval Hull: {final_z_lohner.interval_hull()}")
    
    # Check bounding box finite
    assert jnp.all(jnp.isfinite(final_z_lohner.center))
    
    print("\nTesting TaylorGirardReachability with Spatial Remainder...")
    tg = TaylorGirardReachability(sys, dt, target_order=10, taylor_order=3)
    start = time.time()
    res_tg = tg.compute_reach_sets(0.0, num_steps, z0)
    print(f"TaylorGirard Time: {time.time() - start:.4f}s")
    
    final_z_tg = res_tg.sets[-1]
    print(f"TG Final Interval Hull: {final_z_tg.interval_hull()}")
    
    assert jnp.all(jnp.isfinite(final_z_tg.center))

    print("\nVerification Passed!")

if __name__ == "__main__":
    test_reachability_fixes()
