"""Direct comparison of old vs new reachability implementations."""

import jax
import jax.numpy as jnp
import time
from functools import partial

jax.config.update('jax_enable_x64', True)

from immrax.system import System
from immrax.generator.old_reachability import AlthoffGirardReachability as OldAlthoffGirard
from immrax.generator.reachability import AlthoffGirardReachability as NewAlthoffGirard
from immrax.generator.sets import Zonotope, zonotope_from_interval
from immrax.inclusion import interval


class VanDerPol(System):
    def __init__(self, mu: float = 1.0):
        super().__init__("continuous", 2)
        self.mu = mu

    def f(self, t, x, *args):
        return jnp.array([
            x[1],
            self.mu * (1 - x[0]**2) * x[1] - x[0]
        ])


def benchmark(name, algo, z0, num_steps):
    @partial(jax.jit, static_argnums=(1,))
    def compute(set0, num_steps):
        return algo.compute_reach_sets(0.0, num_steps, set0, f_args=())

    print(f"\n{name}:")

    # Compilation
    start = time.perf_counter()
    result = compute(z0, num_steps)
    result.ts.block_until_ready()
    compile_time = time.perf_counter() - start
    print(f"  Compile: {compile_time:.4f}s")

    # Multiple execution runs
    exec_times = []
    for _ in range(10):
        start = time.perf_counter()
        result = compute(z0, num_steps)
        result.ts.block_until_ready()
        exec_times.append(time.perf_counter() - start)

    avg_exec = sum(exec_times) / len(exec_times)
    print(f"  Execute (avg of 10): {avg_exec*1000:.4f}ms")

    return compile_time, avg_exec


if __name__ == "__main__":
    print("=" * 60)
    print("OLD vs NEW Reachability Comparison (Zonotope only)")
    print("=" * 60)

    vdp = VanDerPol(mu=0.5)
    iv = interval(jnp.array([0.9, 0.0]), jnp.array([1.1, 0.2]))
    z0 = zonotope_from_interval(iv)

    dt = 0.05
    num_steps = 40

    # Old uses max_generators, new uses target_order
    # max_generators=4 roughly corresponds to target_order=2.0 for n=2
    old_algo = OldAlthoffGirard(vdp, dt=dt, max_generators=4)
    new_algo = NewAlthoffGirard(vdp, dt=dt, target_order=2.0)

    old_compile, old_exec = benchmark("OLD (ZonotopeReachSets)", old_algo, z0, num_steps)
    new_compile, new_exec = benchmark("NEW (GenericReachSets)", new_algo, z0, num_steps)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Compile time: OLD={old_compile:.4f}s, NEW={new_compile:.4f}s, ratio={new_compile/old_compile:.2f}x")
    print(f"Execute time: OLD={old_exec*1000:.4f}ms, NEW={new_exec*1000:.4f}ms, ratio={new_exec/old_exec:.2f}x")
