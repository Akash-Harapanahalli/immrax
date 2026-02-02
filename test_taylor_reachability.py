
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from immrax import System
from immrax.generator.algorithms import TaylorModelReachability
from immrax.generator.sets import taylor_model_identity, TaylorModel, taylor_model_concatenate
from immrax.inclusion import Interval
import time

jax.config.update("jax_enable_x64", True)

class VanderPol(System):
    def __init__(self):
        super().__init__('continuous', 2)

    def f(self, t, x):
        # x0' = x1
        # x1' = (1-x0^2)x1 - x0
        dx0 = x[1]
        dx1 = (1.0 - x[0]**2)*x[1] - x[0]

        # Handle TaylorModel inputs specially
        if isinstance(dx0, TaylorModel):
            return taylor_model_concatenate([dx0, dx1])
        return jnp.array([dx0, dx1])

def test_reachability():
    sys = VanderPol()
    dt = 0.01
    t_end = 0.5
    num_steps = int(t_end / dt)
    
    # Initial set: Box [1.9, 2.1] x [-0.1, 0.1] at x0 approx 2, x1 approx 0
    # Actually standard VDP cycle is around origin. Let's start near limit cycle?
    # Or just start near [1, 1].
    
    # Let's try starting at [1., 1.] with radius [0.1, 0.1]
    center = jnp.array([1., 1.])
    radius = jnp.array([0.1, 0.1])
    
    # Create initial Taylor Model from interval
    # We use identity polynomial (order 1) to represent the initial set variables
    # x = c + r*u, u in [-1, 1]
    tm0 = taylor_model_identity(
        Interval(center - radius, center + radius),
        order=2 # Polynomial order for state dependence
    )
    
    algo = TaylorModelReachability(sys, dt, taylor_order=3)
    
    print(f"Running TaylorModelReachability for {num_steps} steps...")
    start_time = time.time()
    reach_sets = algo.compute_reach_sets(0.0, num_steps, tm0)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f}s")
    
    # Basic check: do solutions blow up?
    final_tm = reach_sets.sets[-1]
    final_hull = final_tm.to_interval()
    
    print(f"Final Time: {reach_sets.ts[-1]}")
    print(f"Final Set Hull: {final_hull}")
    
    assert jnp.all(jnp.isfinite(final_hull.lower))
    assert jnp.all(jnp.isfinite(final_hull.upper))
    
    # Check that it's not surprisingly huge (sanity check for VDP in short time)
    # Expected: x0 increases, x1 decreases? 
    # At (1,1): x0'=1, x1' = (0)*1 - 1 = -1.
    # So x0 should go to 1.5, x1 to 0.5 roughly after 0.5s.
    
    mid_state = jnp.array([1.5, 0.5])
    print(f"Expected approx center: {mid_state}")
    
    # Center check
    res_center = (final_hull.lower + final_hull.upper)/2
    dist = jnp.linalg.norm(res_center - mid_state)
    print(f"Distance from expected center: {dist}")
    
    # Tolerance generous due to nonlinearity over 0.5s
    assert dist < 0.2
    
    print("Verification Passed!")

if __name__ == "__main__":
    test_reachability()
