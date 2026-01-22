"""Test script for reachability algorithms with different set representations.

Tests AlthoffGirardReachability with Zonotope, ConstrainedZonotope, and
PolynomialZonotope on nonlinear systems. Measures JIT compilation and execution times.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial

jax.config.update('jax_enable_x64', True)

from immrax.system import System
from immrax.generator import AlthoffGirardReachability
from immrax.generator.sets import (
    Zonotope, zonotope_from_interval,
    ConstrainedZonotope, constrained_zonotope_from_zonotope,
    PolynomialZonotope, polynomial_zonotope_from_interval,
)
from immrax.inclusion import interval


# --- Nonlinear Systems ---

class VanDerPol(System):
    """Van der Pol oscillator."""
    def __init__(self, mu: float = 1.0):
        super().__init__("continuous", 2)
        self.mu = mu

    def f(self, t, x, *args):
        return jnp.array([
            x[1],
            self.mu * (1 - x[0]**2) * x[1] - x[0]
        ])


class Pendulum(System):
    """Simple pendulum with damping."""
    def __init__(self, g: float = 9.81, l: float = 1.0, b: float = 0.5):
        super().__init__("continuous", 2)
        self.g = g
        self.l = l
        self.b = b

    def f(self, t, x, *args):
        return jnp.array([
            x[1],
            -self.g / self.l * jnp.sin(x[0]) - self.b * x[1]
        ])


# --- Benchmark function ---

def benchmark_reachability(algo, set0, num_steps: int, set_name: str):
    """Benchmark reachability computation with JIT compilation timing."""

    # Create a JIT-compiled version of compute_reach_sets
    # We need to wrap it to handle the static num_steps argument
    @partial(jax.jit, static_argnums=(1,))
    def jit_compute(set0, num_steps):
        return algo.compute_reach_sets(0.0, num_steps, set0, f_args=())

    print(f"  {set_name}:")

    # Compilation run (first call triggers JIT compilation)
    print(f"    Compiling...", end=" ", flush=True)
    compile_start = time.perf_counter()
    result = jit_compute(set0, num_steps)
    # Block until computation is done
    result.ts.block_until_ready()
    compile_time = time.perf_counter() - compile_start
    print(f"done ({compile_time:.3f}s)")

    # Execution run (JIT-compiled, should be fast)
    print(f"    Executing...", end=" ", flush=True)
    exec_start = time.perf_counter()
    result = jit_compute(set0, num_steps)
    result.ts.block_until_ready()
    exec_time = time.perf_counter() - exec_start
    print(f"done ({exec_time:.4f}s)")

    # Get final hull width
    final_set = result[num_steps]
    hull = final_set.interval_hull()
    hull_width = np.array(hull.width)

    print(f"    Final hull width: {hull_width}")
    print(f"    Speedup: {compile_time / exec_time:.1f}x")

    return {
        'result': result,
        'compile_time': compile_time,
        'exec_time': exec_time,
        'hull_width': hull_width,
    }


# --- Plotting ---

def plot_reach_sets(ax, reach_result, color, label, alpha=0.4):
    """Plot reachable sets as interval hulls."""
    n_sets = len(reach_result.ts)

    for i in range(n_sets):
        s = reach_result[i]
        hull = s.interval_hull()
        lower = np.array(hull.lower)
        upper = np.array(hull.upper)

        # Skip invalid hulls
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            continue

        # Fade alpha over time
        a = alpha * (0.3 + 0.7 * (1 - i / n_sets))

        rect = plt.Rectangle(
            lower, upper[0] - lower[0], upper[1] - lower[1],
            facecolor=color, alpha=a, edgecolor=color, linewidth=0.3,
            label=label if i == 0 else None
        )
        ax.add_patch(rect)


def run_benchmarks():
    """Run benchmarks comparing set representations."""
    print("=" * 70)
    print("REACHABILITY BENCHMARKS")
    print("Comparing: Zonotope vs ConstrainedZonotope vs PolynomialZonotope")
    print("Using: AlthoffGirardReachability with JIT compilation")
    print("=" * 70)

    dt = 0.05
    num_steps = 40
    target_order = 2.0

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Van der Pol ---
    print("\n" + "-" * 70)
    print("[1/2] Van der Pol Oscillator (mu=0.5)")
    print("-" * 70)

    vdp = VanDerPol(mu=0.5)
    iv_vdp = interval(jnp.array([0.9, 0.0]), jnp.array([1.1, 0.2]))

    z0 = zonotope_from_interval(iv_vdp)
    cz0 = constrained_zonotope_from_zonotope(z0)
    pz0 = polynomial_zonotope_from_interval(iv_vdp)

    algo_vdp = AlthoffGirardReachability(vdp, dt=dt, target_order=target_order)

    results['vdp_z'] = benchmark_reachability(algo_vdp, z0, num_steps, "Zonotope")
    results['vdp_cz'] = benchmark_reachability(algo_vdp, cz0, num_steps, "ConstrainedZonotope")
    results['vdp_pz'] = benchmark_reachability(algo_vdp, pz0, num_steps, "PolynomialZonotope")

    ax = axes[0]
    plot_reach_sets(ax, results['vdp_z']['result'], 'blue', 'Zonotope')
    plot_reach_sets(ax, results['vdp_cz']['result'], 'green', 'ConstrainedZonotope')
    plot_reach_sets(ax, results['vdp_pz']['result'], 'red', 'PolynomialZonotope')
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Van der Pol (mu=0.5)\ndt={dt}, steps={num_steps}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Pendulum ---
    print("\n" + "-" * 70)
    print("[2/2] Damped Pendulum (b=0.5)")
    print("-" * 70)

    pendulum = Pendulum(b=0.5)
    iv_pend = interval(jnp.array([0.2, -0.1]), jnp.array([0.3, 0.1]))

    z0_p = zonotope_from_interval(iv_pend)
    cz0_p = constrained_zonotope_from_zonotope(z0_p)
    pz0_p = polynomial_zonotope_from_interval(iv_pend)

    dt_pend = 0.02
    num_steps_pend = 50
    algo_pend = AlthoffGirardReachability(pendulum, dt=dt_pend, target_order=target_order)

    results['pend_z'] = benchmark_reachability(algo_pend, z0_p, num_steps_pend, "Zonotope")
    results['pend_cz'] = benchmark_reachability(algo_pend, cz0_p, num_steps_pend, "ConstrainedZonotope")
    results['pend_pz'] = benchmark_reachability(algo_pend, pz0_p, num_steps_pend, "PolynomialZonotope")

    ax = axes[1]
    plot_reach_sets(ax, results['pend_z']['result'], 'blue', 'Zonotope')
    plot_reach_sets(ax, results['pend_cz']['result'], 'green', 'ConstrainedZonotope')
    plot_reach_sets(ax, results['pend_pz']['result'], 'red', 'PolynomialZonotope')
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('theta')
    ax.set_ylabel('omega')
    ax.set_title(f'Damped Pendulum (b=0.5)\ndt={dt_pend}, steps={num_steps_pend}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reachability_benchmark.png', dpi=150, bbox_inches='tight')

    # --- Summary ---
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"{'System':<12} {'Set Type':<22} {'Compile (s)':<12} {'Execute (s)':<12} {'Speedup':<10}")
    print("-" * 70)

    for key in ['vdp_z', 'vdp_cz', 'vdp_pz', 'pend_z', 'pend_cz', 'pend_pz']:
        r = results[key]
        sys_name = 'VanDerPol' if 'vdp' in key else 'Pendulum'
        set_type = key.split('_')[1].upper()
        if set_type == 'Z':
            set_type = 'Zonotope'
        elif set_type == 'CZ':
            set_type = 'ConstrainedZonotope'
        else:
            set_type = 'PolynomialZonotope'

        speedup = r['compile_time'] / r['exec_time']
        print(f"{sys_name:<12} {set_type:<22} {r['compile_time']:<12.3f} {r['exec_time']:<12.4f} {speedup:<10.1f}x")

    print("=" * 70)
    print("Saved plot to: reachability_benchmark.png")
    print("=" * 70)

    plt.show()

    return results


if __name__ == "__main__":
    results = run_benchmarks()
