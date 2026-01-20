"""Example: Plotting reachable sets for a pendulum system.

This script demonstrates zonotope-based reachability analysis using
both Lohner's algorithm and the Althoff-Girard algorithm.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from immrax.system import System
from immrax.generator import (
    Zonotope,
    lohner_reachtube,
    althoff_girard_reachtube,
)


# --- Define the Pendulum System ---


class PendulumSystem(System):
    """Simple pendulum: dθ/dt = ω, dω/dt = -sin(θ) - b*ω

    State: x = [θ, ω] where θ is angle and ω is angular velocity
    """

    def __init__(self, damping: float = 0.1):
        super().__init__("continuous", 2)
        self.b = damping

    def f(self, t, x):
        theta, omega = x[0], x[1]
        dtheta = omega
        domega = -jnp.sin(theta) - self.b * omega
        return jnp.array([dtheta, domega])


# --- Plotting Utilities ---


def zonotope_vertices(Z: Zonotope) -> np.ndarray:
    """Compute vertices of a 2D zonotope for plotting.

    Uses the standard algorithm: sort generators by angle and traverse.
    """
    c = np.array(Z.ox)
    G = np.array(Z.G)
    n, m = G.shape

    if n != 2:
        raise ValueError("Can only plot 2D zonotopes")

    if m == 0:
        return c.reshape(1, 2)

    # Remove zero generators
    norms = np.linalg.norm(G, axis=0)
    nonzero_mask = norms > 1e-10
    G = G[:, nonzero_mask]
    m = G.shape[1]

    if m == 0:
        return c.reshape(1, 2)

    # Sort generators by angle
    angles = np.arctan2(G[1, :], G[0, :])
    sorted_idx = np.argsort(angles)
    G_sorted = G[:, sorted_idx]

    # Traverse the zonotope boundary
    # Start from center - sum of all generators
    vertex = c - np.sum(G_sorted, axis=1)
    vertices = [vertex.copy()]

    # Add each generator twice (forward and backward traversal)
    for i in range(m):
        vertex = vertex + 2 * G_sorted[:, i]
        vertices.append(vertex.copy())

    for i in range(m):
        vertex = vertex - 2 * G_sorted[:, i]
        vertices.append(vertex.copy())

    return np.array(vertices[:-1])  # Last vertex equals first


def plot_zonotope(ax, Z: Zonotope, **kwargs):
    """Plot a 2D zonotope as a filled polygon."""
    vertices = zonotope_vertices(Z)
    polygon = Polygon(vertices, **kwargs)
    ax.add_patch(polygon)
    return polygon


def plot_reachtube(ax, tube, color='blue', alpha=0.3, label=None):
    """Plot a reachable tube as a sequence of zonotopes."""
    for i in range(len(tube)):
        Z = tube[i]
        kwargs = {'facecolor': color, 'edgecolor': color, 'alpha': alpha, 'linewidth': 0.5}
        if i == 0 and label:
            kwargs['label'] = label
        plot_zonotope(ax, Z, **kwargs)


def plot_trajectories(ax, sys, Z0, n_samples=10, t0=0.0, tf=1.0, dt=0.01, **kwargs):
    """Plot sample trajectories from the initial zonotope."""
    import jax

    key = jax.random.PRNGKey(0)

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        v = jax.random.uniform(subkey, shape=(Z0.m,), minval=-1, maxval=1)
        x0 = Z0.ox + Z0.G @ v

        traj = sys.compute_trajectory(t0, tf, x0, dt=dt)
        ts = np.array(traj.ts)
        ys = np.array(traj.ys)

        # Filter valid times
        valid = np.isfinite(ts)
        ts, ys = ts[valid], ys[valid]

        ax.plot(ys[:, 0], ys[:, 1], **kwargs)


# --- Main Example ---


def main():
    # Create pendulum system
    pendulum = PendulumSystem(damping=0.1)

    # Initial zonotope: small region around (θ=0.2, ω=0)
    center = jnp.array([0.2, 0.0])
    generators = jnp.array([
        [0.0001, 0.0],   # Small uncertainty in θ
        [0.0, 0.0001],   # Small uncertainty in ω
    ])
    Z0 = Zonotope(center, generators)

    # Time parameters
    t0, tf = 0.0, 5.
    dt = 0.05

    print("Computing reachable tubes...")
    print(f"  Initial set: center={center}, generators shape={generators.shape}")
    print(f"  Time: [{t0}, {tf}], dt={dt}")

    # Compute reachable tubes with both algorithms
    print("\n  Running Lohner's algorithm...")
    tube_lohner = lohner_reachtube(
        pendulum, t0, tf, Z0, dt,
        max_generators=8,
        taylor_order=2,
    )
    print(f"    Done. {len(tube_lohner)} time steps.")

    print("\n  Running Althoff-Girard algorithm...")
    tube_ag = althoff_girard_reachtube(
        pendulum, t0, tf, Z0, dt,
        max_generators=8,
    )
    print(f"    Done. {len(tube_ag)} time steps.")

    # --- Plot Results ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Lohner's algorithm
    ax1 = axes[0]
    ax1.set_title("Lohner's Algorithm")
    plot_reachtube(ax1, tube_lohner, color='blue', alpha=0.4, label='Reachable set')
    plot_trajectories(ax1, pendulum, Z0, n_samples=20, t0=t0, tf=tf,
                      color='black', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel(r'$\theta$ (angle)')
    ax1.set_ylabel(r'$\omega$ (angular velocity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='datalim')

    # Plot 2: Althoff-Girard algorithm
    ax2 = axes[1]
    ax2.set_title("Althoff-Girard Algorithm")
    plot_reachtube(ax2, tube_ag, color='red', alpha=0.4, label='Reachable set')
    plot_trajectories(ax2, pendulum, Z0, n_samples=20, t0=t0, tf=tf,
                      color='black', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel(r'$\theta$ (angle)')
    ax2.set_ylabel(r'$\omega$ (angular velocity)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='datalim')

    # Plot 3: Comparison (overlay)
    ax3 = axes[2]
    ax3.set_title("Comparison")
    plot_reachtube(ax3, tube_lohner, color='blue', alpha=0.3, label='Lohner')
    plot_reachtube(ax3, tube_ag, color='red', alpha=0.3, label='Althoff-Girard')
    plot_trajectories(ax3, pendulum, Z0, n_samples=10, t0=t0, tf=tf,
                      color='black', linewidth=0.5, alpha=0.7, label='Trajectories')
    ax3.set_xlabel(r'$\theta$ (angle)')
    ax3.set_ylabel(r'$\omega$ (angular velocity)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.savefig('reachable_sets.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to 'reachable_sets.png'")
    plt.show()

    # --- Print some statistics ---
    print("\n--- Statistics ---")
    print(f"Lohner final zonotope: {tube_lohner[-1].m} generators")
    print(f"Althoff-Girard final zonotope: {tube_ag[-1].m} generators")

    # Compare volumes (using interval hull as proxy)
    hull_lohner = tube_lohner[-1].interval_hull()
    hull_ag = tube_ag[-1].interval_hull()
    vol_lohner = np.prod(np.array(hull_lohner.upper - hull_lohner.lower))
    vol_ag = np.prod(np.array(hull_ag.upper - hull_ag.lower))
    print(f"Final interval hull volume - Lohner: {vol_lohner:.4f}, Althoff-Girard: {vol_ag:.4f}")


if __name__ == "__main__":
    main()
