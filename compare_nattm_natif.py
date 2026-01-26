
import jax
import jax.numpy as jnp
from immrax.inclusion import interval, nif
import time
from immrax.taylor import nattm, taylor_model_from_interval
from immrax.taylor.taylor_model import TaylorModel

def benchmark(fn, input_val, name=""):
    """Benchmark JIT compilation and execution."""
    print(f"  Benchmarking {name}...")
    
    # Wrap to ensure we JIT the function call
    jitted_fn = jax.jit(fn)
    
    # 1. Compilation + First Run
    try:
        start_time = time.perf_counter()
        out = jitted_fn(input_val)
        # Block until ready
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, out)
        end_time = time.perf_counter()
        first_run_time = end_time - start_time
    except Exception as e:
        print(f"    JIT Failed: {e}")
        # Fallback to eager execution
        start_time = time.perf_counter()
        out = fn(input_val)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, out)
        end_time = time.perf_counter()
        return 0.0, end_time - start_time, out

    # 2. Optimized Run
    start_time = time.perf_counter()
    out = jitted_fn(input_val)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, out)
    end_time = time.perf_counter()
    opt_run_time = end_time - start_time
    
    compile_time = first_run_time - opt_run_time
    # Avoid negative compile time noise
    if compile_time < 0: compile_time = 0.0
    
    return compile_time, opt_run_time, out

def run_comparison():
    print("=== Comparing nattm (Taylor Models) vs natif (Interval Arithmetic) ===\n")
    print("Metrics: Width (tightness) and Runtime (JIT Compile / Post-JIT Exec)\n")

    
    def f1(x):
        # High nonlinearity, standard wrapping effect candidate
        return jnp.sin(x) * jnp.exp(0.5 * x)

    def f2(x):
        # Composition causing dependency problem in intervals
        return x - jnp.sin(x) 
    
    def f3(x):
        # Oscillatory
        return jnp.sin(x) + jnp.cos(3*x)
        
    def f4(x):
        # Rational / singularities nearby
        # Domain [0.1, 0.5]
        return 1.0 / (1.0 + x**2)

    test_cases = [
        ("f(x) = sin(x) * exp(0.5x)", f1, interval(jnp.array([-0.5]), jnp.array([0.5]))),
        ("f(x) = x - sin(x)", f2, interval(jnp.array([-1.0]), jnp.array([1.0]))),
        ("f(x) = sin(x) + cos(3x)", f3, interval(jnp.array([0.0]), jnp.array([0.5]))),
        ("f(x) = 1 / (1 + x^2)", f4, interval(jnp.array([0.1]), jnp.array([0.5]))),
    ]

    for name, func, iv in test_cases:
        print(f"--- Function: {name} ---")
        print(f"Input Interval: {iv}")
        
        # 1. Interval Arithmetic (natif)
        f_nif = nif.natif(func)
        ct_nif, et_nif, iv_out = benchmark(f_nif, iv, "natif")
        width_iv = iv_out.width[0]
        
        # 2. Taylor Model (nattm) - Order 2
        order2 = 2
        tm_in2 = taylor_model_from_interval(iv, order=order2)
        f_tm2 = nattm(func, max_order=order2)
        ct_tm2, et_tm2, tm_out2 = benchmark(f_tm2, tm_in2, f"nattm(k={order2})")
        hull2 = tm_out2.interval_hull()
        width_tm2 = hull2.width[0]
        
        # 3. Taylor Model (nattm) - Order 4
        order4 = 4
        tm_in4 = taylor_model_from_interval(iv, order=order4)
        f_tm4 = nattm(func, max_order=order4)
        ct_tm4, et_tm4, tm_out4 = benchmark(f_tm4, tm_in4, f"nattm(k={order4})")
        hull4 = tm_out4.interval_hull()
        width_tm4 = hull4.width[0]
        
        # 4. Improvement Metrics
        imp2 = (width_iv - width_tm2) / width_iv * 100
        imp4 = (width_iv - width_tm4) / width_iv * 100
        
        print(f"  natif:       Width={width_iv:.6f}  Time: {ct_nif*1000:.1f}ms (jit) / {et_nif*1000:.3f}ms (run)")
        print(f"  nattm(k=2):  Width={width_tm2:.6f}  Time: {ct_tm2*1000:.1f}ms (jit) / {et_tm2*1000:.3f}ms (run)  ({imp2:.1f}% tighter)")
        print(f"  nattm(k=4):  Width={width_tm4:.6f}  Time: {ct_tm4*1000:.1f}ms (jit) / {et_tm4*1000:.3f}ms (run)  ({imp4:.1f}% tighter)")
        print()


if __name__ == "__main__":
    run_comparison()
