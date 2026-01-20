"""Example: Plotting reachable sets for a pendulum system.
Demonstrates Method-Level JIT compilation with static loop bounds.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from functools import partial

# Adjust imports to match your package structure
from immrax.system import System
from immrax.generator import Zonotope
from immrax.generator.reachability import (
    LohnerReachability,
    AlthoffGirardReachability,
    AlthoffTaylorReachability,
)


# --- Define the Pendulum System ---

class PendulumSystem(System):
    """Simple pendulum."""
    def __init__(self, damping: float = 0.1):
        super().__init__("continuous", 2)
        self.b = damping

    def f(self, t, x):
        theta, omega = x[0], x[1]
        return jnp.array([omega, -jnp.sin(theta) - self.b * omega])

    def tree_flatten(self):
        return ((self.b,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(damping=children[0])


# --- Benchmarking Logic ---

def benchmark_full_method(name, generator_cls, sys, t0, num_steps, dt, Z0, **kwargs):
    """JIT compiles the entire compute_reach_sets method using static num_steps."""
    print(f"\n--- Benchmarking {name} ---")
    
    # Initialize generator
    gen = generator_cls(sys, dt, **kwargs)
    
    # Define the JIT-able function
    # num_steps is arg 1. We mark it static so JAX unrolls/scans correctly.
    @partial(jax.jit, static_argnums=(1,))
    def run_all(t0, num_steps, Z0):
        return gen.compute_reach_sets(t0, num_steps, Z0)

    # 1. Compilation (Warmup)
    print("  Compiling (entire horizon)...", end="", flush=True)
    start_time = time.perf_counter()
    
    # Run once to trigger JIT
    res_warmup = run_all(t0, num_steps, Z0)
    res_warmup.center_stack.block_until_ready()
    
    compile_time = time.perf_counter() - start_time
    print(f" Done. (Total warm-up: {compile_time:.4f} s)")

    # 2. Execution (Cached)
    print("  Running execution...", end="", flush=True)
    start_time = time.perf_counter()
    
    res = run_all(t0, num_steps, Z0)
    res.center_stack.block_until_ready()
    
    exec_time = time.perf_counter() - start_time
    print(f" Done. (Time: {exec_time:.4f} s)")
    
    actual_steps = len(res)
    print(f"  > Steps: {actual_steps}")
    print(f"  > Throughput: {actual_steps / exec_time:.1f} steps/sec")

    return res


# --- Plotting Utilities ---

def plot_reachtube(ax, tube, color='blue', alpha=0.3, stride=1, label=None):
    patches = []
    centers = tube.center_stack[::stride]
    gens = tube.gen_stack[::stride]
    
    for c, G in zip(centers, gens):
        Z = Zonotope(c, G)
        G_arr = np.array(Z.G)
        c_arr = np.array(Z.ox)
        
        # 1. Filter small generators
        norms = np.linalg.norm(G_arr, axis=0)
        G_arr = G_arr[:, norms > 1e-9]
        
        if G_arr.shape[1] > 0:
            # 2. ALIGN GENERATORS (The Fix)
            # Ensure all generators point into the right half-plane (x > 0)
            # If x == 0, ensure y > 0.
            # This makes all angles lie in [-pi/2, pi/2]
            
            # Find columns where the first component is negative
            neg_x = G_arr[0, :] < 0
            # Or if x is zero, where y is negative
            neg_y = (G_arr[0, :] == 0) & (G_arr[1, :] < 0)
            
            # Flip those generators
            flip_mask = neg_x | neg_y
            G_arr[:, flip_mask] *= -1
            
            # 3. Sort by angle
            angles = np.arctan2(G_arr[1, :], G_arr[0, :])
            idx = np.argsort(angles)
            G_sorted = G_arr[:, idx]
            
            # 4. Trace Boundary
            # Start at the "bottom-left" extreme: center minus sum of all aligned generators
            current = c_arr - np.sum(G_sorted, axis=1)
            verts = [current.copy()]
            
            # Walk "forward" (add 2*g)
            for i in range(G_sorted.shape[1]):
                current += 2 * G_sorted[:, i]
                verts.append(current.copy())
            
            # Walk "backward" (subtract 2*g) - completes the loop
            for i in range(G_sorted.shape[1]):
                current -= 2 * G_sorted[:, i]
                verts.append(current.copy())
                
            patches.append(Polygon(np.array(verts[:-1]), closed=True))

    p = PatchCollection(patches, facecolor=color, alpha=alpha, edgecolor=(0,0,0,0.5), linewidth=0.5)
    ax.add_collection(p)
    if label: ax.add_patch(Polygon([[0,0]], color=color, alpha=alpha, label=label))
    if len(centers) > 0: ax.autoscale_view()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)


def plot_vmapped_trajectories(ax, sys, Z0, n_samples=50, t0=0.0, tf=3.0, dt=0.02, **kwargs):
    key = jax.random.PRNGKey(0)
    vs = jax.random.uniform(key, (n_samples, Z0.m), minval=-1., maxval=1.)
    x0s = Z0.ox + jax.vmap(lambda v: Z0.G @ v)(vs)
    
    # # Calculate exact steps to match main loop
    # num_steps = int((tf - t0) / dt)

    # def step_rk4(x, t):
    #     k1 = sys.f(t, x)
    #     k2 = sys.f(t + dt/2, x + dt/2 * k1)
    #     k3 = sys.f(t + dt/2, x + dt/2 * k2)
    #     k4 = sys.f(t + dt, x + dt * k3)
    #     return x + dt/6*(k1 + 2*k2 + 2*k3 + k4), None

    def sim(x0):
        # Scan with static time grid
        # times = t0 + jnp.arange(num_steps) * dt
        # _, xs = jax.lax.scan(step_rk4, x0, times)
        # return xs
        return sys.compute_trajectory(t0, tf, x0, dt=dt)

    trajs = jax.vmap(sim)(x0s) 
    trajs = trajs.to_convenience()
    
    for i in range(n_samples):
        ax.plot(trajs.ys[i, :, 0], trajs.ys[i, :, 1], **kwargs)

# --- Main ---

def main():
    plt.style.use('bmh')
    
    pendulum = PendulumSystem(damping=0.2)
    Z0 = Zonotope(jnp.array([0.5, 0.0]), jnp.diag(jnp.array([0.05, 0.05])))

    t0, tf = 0.0, 10.0
    dt = 0.02
    
    # Explicitly calculate num_steps as a standard int
    num_steps = int((tf - t0) / dt)
    
    print(f"System: Pendulum (damping=0.2)")
    print(f"Horizon: [{t0}, {tf}], dt={dt}, Steps={num_steps}")

    # Run Benchmarks
    tube_lohner = benchmark_full_method("Lohner", LohnerReachability, pendulum, t0, num_steps, dt, Z0)
    tube_ag = benchmark_full_method("Althoff-Girard", AlthoffGirardReachability, pendulum, t0, num_steps, dt, Z0)
    tube_at = benchmark_full_method("Althoff-Taylor", AlthoffTaylorReachability, pendulum, t0, num_steps, dt, Z0)

    # Plotting
    print("\nPlotting...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    traj_opts = {'color': 'k', 'lw': 0.5, 'alpha': 0.5}

    names = ["Lohner", "Althoff-Girard", "Althoff-Taylor", "Comparison"]
    tubes = [tube_lohner, tube_ag, tube_at]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        axes[i].set_title(names[i])
        plot_reachtube(axes[i], tubes[i], color=colors[i], stride=10)
        plot_vmapped_trajectories(axes[i], pendulum, Z0, t0=t0, tf=tf, dt=dt, **traj_opts)

    axes[3].set_title("Comparison")
    for i, t in enumerate(tubes):
        plot_reachtube(axes[3], t, color=colors[i], alpha=0.15, stride=10, label=names[i])
    plot_vmapped_trajectories(axes[3], pendulum, Z0, t0=t0, tf=tf, dt=dt, **traj_opts)
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel('Theta')
        ax.set_ylabel('Omega')

    plt.tight_layout()
    plt.savefig('jit_benchmark.png')
    plt.show()

if __name__ == "__main__":
    main()