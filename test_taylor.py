import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from immrax.generator.sets.taylor_model import taylor_model_from_function, TaylorModel
from immrax.inclusion import interval

def test_taylor_from_function():
    # Test function: f(x) = x0^2 + 2*x0*x1 + x1^3 + 5
    # Center at [0, 0]
    def poly(x):
        return x[0]**2 + 2*x[0]*x[1] + x[1]**3 + 5.0

    center = jnp.array([0., 0.])
    radius = jnp.array([1., 1.])
    max_order = 3
    
    # Expected coefficients at [0,0]:
    # Constant: 5
    # x0: 0
    # x1: 0
    # x0^2: 1
    # x0*x1: 2
    # x1^2: 0
    # x1^3: 1
    # others: 0
    
    tm = taylor_model_from_function(poly, center, radius, max_order)

    print("Coefficients:")
    print(tm.coeffs)
    print("Exponents:")
    print(tm.exponents)
    print("Remainder interval:")
    print(f"  {tm.remainder}")
    print("Interval hull:")
    print(f"  {tm.interval_hull()}")
    
    # Check values
    # We iterate and check
    def get_coeff(powers):
        # find column in exponents
        matches = jnp.all(tm.exponents == jnp.array(powers)[:, None], axis=0)
        idx = jnp.argmax(matches)
        if not jnp.any(matches): return 0.0
        return tm.coeffs[0, idx]
        
    assert jnp.abs(get_coeff([0, 0]) - 5.0) < 1e-5
    assert jnp.abs(get_coeff([2, 0]) - 1.0) < 1e-5
    assert jnp.abs(get_coeff([1, 1]) - 2.0) < 1e-5
    assert jnp.abs(get_coeff([0, 3]) - 1.0) < 1e-5
    
    print("\nLocal Polynomial Test Passed!")
    
    # Test shifted center: x based at [1, 1]
    # f(1+dx, 1+dy) = (1+dx)^2 + 2(1+dx)(1+dy) + (1+dy)^3 + 5
    # = (1 + 2dx + dx^2) + (2 + 2dx + 2dy + 2dxdy) + (1 + 3dy + 3dy^2 + dy^3) + 5
    # = 9 + 4dx + 5dy + dx^2 + 2dxdy + 3dy^2 + dy^3
    
    center2 = jnp.array([1., 1.])
    tm2 = taylor_model_from_function(poly, center2, radius, max_order)
    
    def get_coeff2(powers):
        matches = jnp.all(tm2.exponents == jnp.array(powers)[:, None], axis=0)
        idx = jnp.argmax(matches)
        if not jnp.any(matches): return 0.0
        return tm2.coeffs[0, idx]
        
    print("\nShifted Center Coefficients:")
    print(tm2.coeffs)
    print("Shifted Remainder interval:")
    print(f"  {tm2.remainder}")
    print("Shifted Interval hull:")
    print(f"  {tm2.interval_hull()}")
    
    assert jnp.abs(get_coeff2([0, 0]) - 9.0) < 1e-5
    assert jnp.abs(get_coeff2([1, 0]) - 4.0) < 1e-5
    assert jnp.abs(get_coeff2([0, 1]) - 5.0) < 1e-5
    assert jnp.abs(get_coeff2([2, 0]) - 1.0) < 1e-5
    assert jnp.abs(get_coeff2([1, 1]) - 2.0) < 1e-5
    assert jnp.abs(get_coeff2([0, 2]) - 3.0) < 1e-5
    assert jnp.abs(get_coeff2([0, 3]) - 1.0) < 1e-5
    
    print("\nShifted Polynomial Test Passed!")

    # Test sine function (higher order)
    def f_sin(x): return jnp.sin(x[0])
    
    tm_sin = taylor_model_from_function(f_sin, jnp.array([0.]), jnp.array([1.]), 5)
    # sin(x) = x - x^3/6 + x^5/120
    
    def get_coeff_sin(p):
        matches = jnp.all(tm_sin.exponents == jnp.array([p])[:, None], axis=0)
        idx = jnp.argmax(matches)
        return tm_sin.coeffs[0, idx]

    print("\nSine Coefficients:")
    print(tm_sin.coeffs)
    print("Sine Remainder interval:")
    print(f"  {tm_sin.remainder}")
    print("Sine Interval hull:")
    print(f"  {tm_sin.interval_hull()}")
    
    assert jnp.abs(get_coeff_sin(1) - 1.0) < 1e-5
    assert jnp.abs(get_coeff_sin(3) + 1/6) < 1e-5
    assert jnp.abs(get_coeff_sin(5) - 1/120) < 1e-5
    
    print("\nSine Test Passed!")

def plot_taylor_1d(f, tm: TaylorModel, title="Taylor Model vs Function"):
    """Plot a 1D Taylor model against the original function."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Domain
    center = tm.domain_center[0]
    radius = tm.domain_radius[0]
    x = np.linspace(center - radius, center + radius, 200)

    # Evaluate function
    f_vals = np.array([f(jnp.array([xi])) for xi in x])

    # Evaluate Taylor polynomial
    poly_vals = np.array([tm.evaluate_polynomial(jnp.array([xi]))[0] for xi in x])

    # Get interval hull
    hull = tm.interval_hull()

    # Plot 1: Function vs Taylor polynomial
    ax1.plot(x, f_vals, 'b-', linewidth=2, label='True function')
    ax1.plot(x, poly_vals, 'r--', linewidth=2, label='Taylor polynomial')
    ax1.axhline(hull.lower[0], color='g', linestyle=':', alpha=0.7, label='Interval hull')
    ax1.axhline(hull.upper[0], color='g', linestyle=':', alpha=0.7)
    ax1.fill_between(x, hull.lower[0], hull.upper[0], alpha=0.1, color='g')
    ax1.axvline(center, color='gray', linestyle='--', alpha=0.5, label='Center')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(f'{title}\nOrder {tm.order}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    error = f_vals - poly_vals
    ax2.plot(x, error, 'k-', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(x, tm.remainder.lower[0], tm.remainder.upper[0],
                     alpha=0.3, color='orange', label='Remainder bound')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Error (f - polynomial)')
    ax2.set_title('Approximation Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_taylor_2d(f, tm: TaylorModel, title="Taylor Model vs Function", resolution=50):
    """Plot a 2D Taylor model against the original function."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Domain
    cx, cy = tm.domain_center
    rx, ry = tm.domain_radius
    x = np.linspace(cx - rx, cx + rx, resolution)
    y = np.linspace(cy - ry, cy + ry, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function and Taylor polynomial
    f_vals = np.zeros_like(X)
    poly_vals = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            pt = jnp.array([X[i, j], Y[i, j]])
            f_vals[i, j] = f(pt)
            poly_vals[i, j] = tm.evaluate_polynomial(pt)[0]

    # Plot 1: True function
    im1 = axes[0].contourf(X, Y, f_vals, levels=20, cmap='viridis')
    axes[0].set_xlabel('x₀')
    axes[0].set_ylabel('x₁')
    axes[0].set_title('True Function')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Taylor polynomial
    im2 = axes[1].contourf(X, Y, poly_vals, levels=20, cmap='viridis')
    axes[1].set_xlabel('x₀')
    axes[1].set_ylabel('x₁')
    axes[1].set_title(f'Taylor Polynomial (order {tm.order})')
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Error
    error = f_vals - poly_vals
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='RdBu_r')
    axes[2].set_xlabel('x₀')
    axes[2].set_ylabel('x₁')
    axes[2].set_title('Error (f - polynomial)')
    plt.colorbar(im3, ax=axes[2])

    # Add interval hull info
    hull = tm.interval_hull()
    fig.suptitle(f'{title}\nInterval Hull: [{hull.lower[0]:.3f}, {hull.upper[0]:.3f}]',
                 fontsize=12)

    plt.tight_layout()
    return fig


def plot_taylor_comparison():
    """Compare Taylor models of different orders."""
    print("\n" + "="*60)
    print("Plotting Taylor Model Comparisons")
    print("="*60)

    # 1D Example: sin(x)
    print("\n1. Plotting sin(x) Taylor approximations...")
    def f_sin(x):
        return jnp.sin(x[0])

    center = jnp.array([0.])
    radius = jnp.array([np.pi])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, order in enumerate([1, 3, 5, 7]):
        ax = axes[idx // 2, idx % 2]
        tm = taylor_model_from_function(f_sin, center, radius, order)

        x = np.linspace(-np.pi, np.pi, 200)
        f_vals = np.sin(x)
        poly_vals = np.array([tm.evaluate_polynomial(jnp.array([xi]))[0] for xi in x])
        hull = tm.interval_hull()

        ax.plot(x, f_vals, 'b-', linewidth=2, label='sin(x)')
        ax.plot(x, poly_vals, 'r--', linewidth=2, label=f'Taylor (order {order})')
        ax.fill_between(x, hull.lower[0], hull.upper[0], alpha=0.1, color='g',
                        label=f'Hull: [{hull.lower[0]:.2f}, {hull.upper[0]:.2f}]')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Order {order}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    plt.suptitle('Taylor Approximations of sin(x) on [-π, π]', fontsize=14)
    plt.tight_layout()
    plt.savefig('taylor_sin_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: taylor_sin_comparison.png")

    # 2D Example: x^2 + sin(y)
    print("\n2. Plotting 2D function x₀² + sin(x₁)...")
    def f_2d(x):
        return x[0]**2 + jnp.sin(x[1])

    center_2d = jnp.array([0., 0.])
    radius_2d = jnp.array([1.5, np.pi/2])

    fig = plot_taylor_2d(f_2d,
                         taylor_model_from_function(f_2d, center_2d, radius_2d, 4),
                         title="f(x) = x₀² + sin(x₁)")
    plt.savefig('taylor_2d_example.png', dpi=150, bbox_inches='tight')
    print("   Saved: taylor_2d_example.png")

    # 1D Example with detailed view: exp(x)
    print("\n3. Plotting exp(x) Taylor approximation...")
    def f_exp(x):
        return jnp.exp(x[0])

    center_exp = jnp.array([0.])
    radius_exp = jnp.array([2.])

    tm_exp = taylor_model_from_function(f_exp, center_exp, radius_exp, 5)
    fig = plot_taylor_1d(f_exp, tm_exp, title="exp(x) Taylor Approximation")
    plt.savefig('taylor_exp_example.png', dpi=150, bbox_inches='tight')
    print("   Saved: taylor_exp_example.png")

    # Show the polynomial vs original
    print("\n4. Plotting polynomial test function...")
    def poly(x):
        return x[0]**2 + 2*x[0]*x[1] + x[1]**3 + 5.0

    tm_poly = taylor_model_from_function(poly, jnp.array([0., 0.]), jnp.array([1., 1.]), 3)
    fig = plot_taylor_2d(poly, tm_poly, title="f(x) = x₀² + 2x₀x₁ + x₁³ + 5")
    plt.savefig('taylor_poly_example.png', dpi=150, bbox_inches='tight')
    print("   Saved: taylor_poly_example.png")

    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)

    # plt.show()  # Commented out to avoid blocking


if __name__ == "__main__":
    test_taylor_from_function()
    plot_taylor_comparison()
