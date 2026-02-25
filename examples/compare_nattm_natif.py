"""Compare pjetm (Taylor Model) vs natif (Interval) for function bounding.

This script demonstrates the tighter bounds achievable with Taylor models
compared to pure interval arithmetic for various nonlinear functions,
including vector input/output functions.
"""

import jax
import jax.numpy as jnp
import numpy as np

import immrax as irx
from immrax.inclusion import interval, icentpert, natif
from immrax.taylor.taylor_model import taylor_model_identity
from immrax.taylor.pjetm import pjetm
from immrax.utils import run_times


def compute_true_range(f, iv, num_samples=10000):
    """Compute true range by dense sampling (for scalar output)."""
    n = iv.lower.shape[0]
    if n == 1:
        xs = np.linspace(float(iv.lower[0]), float(iv.upper[0]), num_samples)
        ys = np.array([f(jnp.array([x])) for x in xs])
    else:
        # Grid sampling for multi-dimensional input
        samples_per_dim = int(num_samples ** (1/n)) + 1
        grids = [np.linspace(float(iv.lower[i]), float(iv.upper[i]), samples_per_dim)
                 for i in range(n)]
        mesh = np.meshgrid(*grids)
        points = np.stack([m.flatten() for m in mesh], axis=1)
        ys = np.array([f(jnp.array(p)) for p in points])

    if ys.ndim == 1:
        return np.min(ys), np.max(ys)
    else:
        return np.min(ys, axis=0), np.max(ys, axis=0)


def width(iv):
    """Compute width of an interval."""
    return iv.upper - iv.lower


def to_scalar(x):
    """Convert array to scalar float, handling 0D and 1D arrays."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        return float(x)
    else:
        return float(x[0])


def compare_scalar(name, f, iv, tm_order=4, num_runs=100):
    """Compare natif vs pjetm for scalar output functions."""
    # Create inputs
    tm = taylor_model_identity(iv, order=tm_order)

    # Compile and warm up
    natif_f = jax.jit(natif(f))
    nattm_f = jax.jit(lambda tm: pjetm(f, max_order=tm_order)(tm).interval_hull())

    # Warm-up runs
    iv_result = natif_f(iv)
    tm_result = nattm_f(tm)

    # Timed runs
    _, natif_times = run_times(num_runs, natif_f, iv)
    _, nattm_times = run_times(num_runs, nattm_f, tm)

    # True range
    true_min, true_max = compute_true_range(f, iv)
    true_width = true_max - true_min

    iv_width = to_scalar(width(iv_result))
    tm_width = to_scalar(width(tm_result))

    return {
        'name': name,
        'input_dim': iv.lower.shape[0],
        'output_dim': 1,
        'true_width': true_width,
        'natif_width': iv_width,
        'nattm_width': tm_width,
        'natif_overest': iv_width / true_width if true_width > 1e-10 else float('inf'),
        'nattm_overest': tm_width / true_width if true_width > 1e-10 else float('inf'),
        'improvement': iv_width / tm_width if tm_width > 1e-10 else float('inf'),
        'natif_time_ms': float(jnp.mean(natif_times)) * 1000,
        'nattm_time_ms': float(jnp.mean(nattm_times)) * 1000,
        'natif_bounds': (to_scalar(iv_result.lower), to_scalar(iv_result.upper)),
        'nattm_bounds': (to_scalar(tm_result.lower), to_scalar(tm_result.upper)),
        'true_bounds': (true_min, true_max),
    }


def compare_vector(name, f, iv, tm_order=4, num_runs=100):
    """Compare natif vs pjetm for vector output functions."""
    tm = taylor_model_identity(iv, order=tm_order)

    # Compile and warm up
    natif_f = jax.jit(natif(f))
    nattm_f = jax.jit(lambda tm: pjetm(f, max_order=tm_order)(tm).interval_hull())

    iv_result = natif_f(iv)
    tm_result = nattm_f(tm)

    # Timed runs
    _, natif_times = run_times(num_runs, natif_f, iv)
    _, nattm_times = run_times(num_runs, nattm_f, tm)

    # True range (vector)
    true_min, true_max = compute_true_range(f, iv)
    true_widths = true_max - true_min

    iv_widths = width(iv_result)
    tm_widths = width(tm_result)

    # Average metrics across output dimensions
    avg_improvement = float(jnp.mean(iv_widths / jnp.maximum(tm_widths, 1e-10)))

    return {
        'name': name,
        'input_dim': iv.lower.shape[0],
        'output_dim': iv_result.lower.shape[0],
        'true_widths': true_widths,
        'natif_widths': iv_widths,
        'nattm_widths': tm_widths,
        'avg_improvement': avg_improvement,
        'natif_time_ms': float(jnp.mean(natif_times)) * 1000,
        'nattm_time_ms': float(jnp.mean(nattm_times)) * 1000,
        'natif_bounds': (iv_result.lower, iv_result.upper),
        'nattm_bounds': (tm_result.lower, tm_result.upper),
        'true_bounds': (true_min, true_max),
    }


def print_table(results, is_vector=False):
    """Print results in a formatted table."""
    if is_vector:
        print(f"{'Function':<25} {'In->Out':<10} {'Avg Improv':<12} {'natif (ms)':<12} {'pjetm (ms)':<12}")
        print("-" * 75)
        for r in results:
            dims = f"{r['input_dim']}->{r['output_dim']}"
            print(f"{r['name']:<25} {dims:<10} {r['avg_improvement']:<12.2f}x "
                  f"{r['natif_time_ms']:<12.3f} {r['nattm_time_ms']:<12.3f}")
    else:
        print(f"{'Function':<25} {'True Width':<12} {'natif Width':<12} {'pjetm Width':<12} {'Improv':<10} {'natif (ms)':<10} {'pjetm (ms)':<10}")
        print("-" * 100)
        for r in results:
            print(f"{r['name']:<25} {r['true_width']:<12.6f} {r['natif_width']:<12.6f} "
                  f"{r['nattm_width']:<12.6f} {r['improvement']:<10.2f}x "
                  f"{r['natif_time_ms']:<10.3f} {r['nattm_time_ms']:<10.3f}")


def main():
    print("=" * 100)
    print("Comparison: pjetm (Taylor Model) vs natif (Interval) for Function Bounding")
    print("=" * 100)

    num_runs = 100
    print(f"\nTiming averaged over {num_runs} runs (after JIT compilation)")

    # =========================================================================
    # Scalar Input -> Scalar Output Functions
    # =========================================================================
    print("\n" + "=" * 100)
    print("SCALAR INPUT -> SCALAR OUTPUT")
    print("=" * 100)

    scalar_tests = [
        # Basic transcendentals
        ("sin(x)", lambda x: jnp.sin(x[0]), icentpert(jnp.array([0.5]), jnp.array([0.3])), 4),
        ("cos(x)", lambda x: jnp.cos(x[0]), icentpert(jnp.array([0.5]), jnp.array([0.3])), 4),
        ("exp(x)", lambda x: jnp.exp(x[0]), icentpert(jnp.array([0.5]), jnp.array([0.3])), 4),
        ("log(x)", lambda x: jnp.log(x[0]), icentpert(jnp.array([1.0]), jnp.array([0.3])), 4),

        # Polynomials
        ("x^2", lambda x: x[0]**2, icentpert(jnp.array([1.0]), jnp.array([0.5])), 4),
        ("x^3", lambda x: x[0]**3, icentpert(jnp.array([1.0]), jnp.array([0.5])), 4),
        ("x^2 - x", lambda x: x[0]**2 - x[0], icentpert(jnp.array([1.0]), jnp.array([0.5])), 4),

        # Compositions
        ("sin(cos(x))", lambda x: jnp.sin(jnp.cos(x[0])), icentpert(jnp.array([0.5]), jnp.array([0.3])), 5),
        ("exp(-x^2)", lambda x: jnp.exp(-x[0]**2), icentpert(jnp.array([0.0]), jnp.array([0.5])), 5),
        ("sin(exp(x))", lambda x: jnp.sin(jnp.exp(x[0])), icentpert(jnp.array([0.0]), jnp.array([0.3])), 5),
        ("log(1+x^2)", lambda x: jnp.log(1 + x[0]**2), icentpert(jnp.array([1.0]), jnp.array([0.5])), 4),

        # Dependency problem examples
        ("x - x (=0)", lambda x: x[0] - x[0], icentpert(jnp.array([1.0]), jnp.array([1.0])), 4),
        ("x * (1-x)", lambda x: x[0] * (1 - x[0]), icentpert(jnp.array([0.5]), jnp.array([0.3])), 4),
        ("(x-1)*(x+1)", lambda x: (x[0]-1)*(x[0]+1), icentpert(jnp.array([0.5]), jnp.array([0.4])), 4),
        ("sin(x) - x", lambda x: jnp.sin(x[0]) - x[0], icentpert(jnp.array([0.]), jnp.array([1.])), 4),
    ]

    scalar_results = []
    for name, f, iv, order in scalar_tests:
        try:
            result = compare_scalar(name, f, iv, tm_order=order, num_runs=num_runs)
            scalar_results.append(result)
        except Exception as e:
            print(f"Error with {name}: {e}")

    print_table(scalar_results, is_vector=False)

    # =========================================================================
    # Vector Input -> Scalar Output Functions
    # =========================================================================
    print("\n" + "=" * 100)
    print("VECTOR INPUT -> SCALAR OUTPUT")
    print("=" * 100)

    vec_scalar_tests = [
        # 2D -> 1D
        ("x1*x2", lambda x: x[0]*x[1],
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.2, 0.2])), 3),
        ("x1^2 + x2^2", lambda x: x[0]**2 + x[1]**2,
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.3, 0.3])), 3),
        ("sin(x1)*cos(x2)", lambda x: jnp.sin(x[0])*jnp.cos(x[1]),
         icentpert(jnp.array([0.5, 0.5]), jnp.array([0.2, 0.2])), 3),
        ("exp(-(x1^2+x2^2))", lambda x: jnp.exp(-(x[0]**2 + x[1]**2)),
         icentpert(jnp.array([0.0, 0.0]), jnp.array([0.3, 0.3])), 3),
        ("x1*x2 - x1 - x2", lambda x: x[0]*x[1] - x[0] - x[1],
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.3, 0.3])), 3),

        # 3D -> 1D
        ("x1*x2*x3", lambda x: x[0]*x[1]*x[2],
         icentpert(jnp.array([1.0, 1.0, 1.0]), jnp.array([0.2, 0.2, 0.2])), 2),
        ("norm(x)^2", lambda x: x[0]**2 + x[1]**2 + x[2]**2,
         icentpert(jnp.array([1.0, 1.0, 1.0]), jnp.array([0.2, 0.2, 0.2])), 2),
    ]

    vec_scalar_results = []
    for name, f, iv, order in vec_scalar_tests:
        try:
            result = compare_scalar(name, f, iv, tm_order=order, num_runs=num_runs)
            vec_scalar_results.append(result)
        except Exception as e:
            print(f"Error with {name}: {e}")

    print_table(vec_scalar_results, is_vector=False)


    # =========================================================================
    # Complex Dependency & Stress Tests
    # =========================================================================
    print("\n" + "=" * 100)
    print("COMPLEX DEPENDENCY & STRESS TESTS")
    print("=" * 100)

    complex_tests = [
        # 1. Rosenbrock-like (coupled): (1-x)^2 + 100*(y - x^2)^2
        # Intervals fail hard on x^2 dependency
        ("Rosenbrock (2D)", lambda x: (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2,
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.1, 0.1])), 4),

        # 2. Trigonometric Identity: sin^2(x) + cos^2(x) - 1
        # Should be exactly 0 (or close to it). Intervals will be [-1, 1] worst case.
        ("sin^2 + cos^2 - 1", lambda x: jnp.sin(x[0])**2 + jnp.cos(x[0])**2 - 1.0,
         icentpert(jnp.array([0.5]), jnp.array([1.0])), 6),

        # 3. Rational with Repeated Variables: x / (x^2 + 1)
        # Standard example where dependency matters.
        ("x / (x^2 + 1)", lambda x: x[0] / (x[0]**2 + 1.0),
         icentpert(jnp.array([1.0]), jnp.array([0.5])), 4),

        # 4. Determinant of 2x2: x1*x4 - x2*x3
        # Direct dependency test
        ("Det(2x2)", lambda x: x[0]*x[3] - x[1]*x[2],
         icentpert(jnp.array([1., 0., 0., 1.]), jnp.array([0.2, 0.2, 0.2, 0.2])), 3),

        # 5. Coupled ODE-like step (Runge-Kutta style sub-stage)
        # k1 = f(x), k2 = f(x + 0.5*h*k1) .. lots of nesting
        ("RK4 Step Term: x + h*(-x)", lambda x: x[0] + 0.1 * (-x[0] + 0.1 * (-x[0])),
         icentpert(jnp.array([1.0]), jnp.array([0.5])), 3),
         
        # 6. High-order interaction
        # (x+y)^5 - (x-y)^5
        ("(x+y)^5 - (x-y)^5", lambda x: (x[0] + x[1])**5 - (x[0] - x[1])**5,
         icentpert(jnp.array([0.0, 0.0]), jnp.array([0.1, 0.1])), 5),
    ]

    complex_results = []
    for name, f, iv, order in complex_tests:
        try:
            result = compare_scalar(name, f, iv, tm_order=order, num_runs=num_runs)
            complex_results.append(result)
        except Exception as e:
            print(f"Error with {name}: {e}")
            import traceback
            traceback.print_exc()

    print_table(complex_results, is_vector=False)

    # =========================================================================
    # Vector Input -> Vector Output Functions
    # =========================================================================
    print("\n" + "=" * 100)
    print("VECTOR INPUT -> VECTOR OUTPUT")
    print("=" * 100)

    # Define vector functions using natural jnp.array([...]) syntax
    def linear_2d(x):
        return jnp.array([-x[0] + 0.5*x[1], 0.5*x[0] - x[1]])

    def rotation_2d(x):
        return jnp.array([0.8*x[0] - 0.6*x[1], 0.6*x[0] + 0.8*x[1]])

    def vanderpol_rhs(x):
        # x' = [x1, (1-x0^2)*x1 - x0]
        return jnp.array([x[1], (1 - x[0]**2) * x[1] - x[0]])

    def lotka_volterra_rhs(x):
        # x' = [x0*(1-x1), x1*(x0-1)]
        return jnp.array([x[0] * (1 - x[1]), x[1] * (x[0] - 1)])

    def pendulum_rhs(x):
        # x' = [x1, -sin(x0) - 0.1*x1]
        return jnp.array([x[1], -jnp.sin(x[0]) - 0.1 * x[1]])

    def henon_map(x):
        # y = [1 - 1.4*x0^2 + x1, 0.3*x0]
        return jnp.array([1 - 1.4 * x[0]**2 + x[1], 0.3 * x[0]])

    def lorenz_rhs(x):
        # x' = [10*(x1-x0), x0*(28-x2)-x1, x0*x1 - 8/3*x2]
        return jnp.array([
            10 * (x[1] - x[0]),
            x[0] * (28 - x[2]) - x[1],
            x[0] * x[1] - 8/3 * x[2]
        ])

    def breaking(x):
        return jnp.array([x[0]**4 + x[1]**4 - 2*x[0]**2*x[1]**2, jnp.sin(x[0]) - x[0]])

    vec_vec_tests = [
        # 2D -> 2D (dynamical systems style)
        ("Linear: Ax", linear_2d,
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.2, 0.2])), 3),

        ("Rotation", rotation_2d,
         icentpert(jnp.array([1.0, 0.0]), jnp.array([0.2, 0.2])), 3),

        ("Van der Pol RHS", vanderpol_rhs,
         icentpert(jnp.array([0.5, 0.5]), jnp.array([0.2, 0.2])), 3),

        ("Lotka-Volterra RHS", lotka_volterra_rhs,
         icentpert(jnp.array([1.0, 1.0]), jnp.array([0.2, 0.2])), 3),

        ("Pendulum RHS", pendulum_rhs,
         icentpert(jnp.array([0.5, 0.0]), jnp.array([0.2, 0.2])), 3),

        # Polynomial maps
        ("Henon map", henon_map,
         icentpert(jnp.array([0.0, 0.0]), jnp.array([0.2, 0.2])), 3),

        # 3D -> 3D
        ("Lorenz RHS", lorenz_rhs,
         icentpert(jnp.array([1.0, 1.0, 1.0]), jnp.array([0.1, 0.1, 0.1])), 2),

        ("Breaking", breaking,
         icentpert(jnp.array([0., 0.]), jnp.array([1., 1.])), 4),
    ]

    vec_vec_results = []
    for name, f, iv, order in vec_vec_tests:
        try:
            result = compare_vector(name, f, iv, tm_order=order, num_runs=num_runs)
            vec_vec_results.append(result)
        except Exception as e:
            print(f"Error with {name}: {e}")

    print_table(vec_vec_results, is_vector=True)

    # Detailed output for vector functions
    print("\nDetailed bounds for vector functions:")
    print("-" * 80)
    for r in vec_vec_results:
        print(f"\n{r['name']} ({r['input_dim']}D -> {r['output_dim']}D):")
        for i in range(r['output_dim']):
            natif_w = float(r['natif_widths'][i])
            nattm_w = float(r['nattm_widths'][i])
            true_w = float(r['true_widths'][i])
            improv = natif_w / nattm_w if nattm_w > 1e-10 else float('inf')
            print(f"  dim {i}: true={true_w:.4f}, natif={natif_w:.4f}, pjetm={nattm_w:.4f}, improv={improv:.2f}x")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    all_scalar = scalar_results + vec_scalar_results
    improvements = [r['improvement'] for r in all_scalar if r['improvement'] < 1000]

    if improvements:
        print(f"\nScalar output functions ({len(improvements)} tests):")
        print(f"  Average width improvement: {np.mean(improvements):.2f}x")
        print(f"  Median width improvement:  {np.median(improvements):.2f}x")
        print(f"  Max improvement:           {np.max(improvements):.2f}x")

    vec_improvements = [r['avg_improvement'] for r in vec_vec_results if r['avg_improvement'] < 1000]
    if vec_improvements:
        print(f"\nVector output functions ({len(vec_improvements)} tests):")
        print(f"  Average width improvement: {np.mean(vec_improvements):.2f}x")
        print(f"  Median width improvement:  {np.median(vec_improvements):.2f}x")
        print(f"  Max improvement:           {np.max(vec_improvements):.2f}x")

    # Timing comparison
    natif_times = [r['natif_time_ms'] for r in all_scalar + vec_vec_results]
    nattm_times = [r['nattm_time_ms'] for r in all_scalar + vec_vec_results]

    print(f"\nTiming (all tests):")
    print(f"  Average natif time: {np.mean(natif_times):.3f} ms")
    print(f"  Average pjetm time: {np.mean(nattm_times):.3f} ms")
    print(f"  pjetm/natif ratio:  {np.mean(nattm_times)/np.mean(natif_times):.2f}x slower")

    print("\n" + "=" * 100)
    print("NOTES")
    print("=" * 100)
    print("""
- 'Improvement' = natif_width / nattm_width (higher is better for pjetm)
- Taylor models provide tighter bounds by tracking polynomial dependencies
- The 'x - x' case shows the dependency problem: intervals give non-zero width,
  Taylor models correctly give zero (since x - x = 0 algebraically)
- pjetm is slower than natif due to polynomial coefficient propagation
- For dynamical systems, tighter bounds compound over time, making the
  computational overhead worthwhile for reachability analysis
""")


if __name__ == "__main__":
    main()
