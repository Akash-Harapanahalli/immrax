"""Tests for zonotope-based reachability algorithms."""

import jax
import jax.numpy as jnp
import pytest

from immrax.system import System
from immrax.generator import (
    Zonotope,
    zonotope,
    zonotope_from_interval,
    LohnerReachability,
    AlthoffGirardReachability,
    lohner_reachtube,
    althoff_girard_reachtube,
)
from immrax.inclusion import Interval


# --- Test Systems ---


class LinearSystem(System):
    """Simple linear system: dx/dt = A @ x"""

    def __init__(self, A):
        super().__init__("continuous", A.shape[0])
        self.A = A

    def f(self, t, x):
        return self.A @ x


class VanDerPolSystem(System):
    """Van der Pol oscillator: a classic nonlinear test system."""

    def __init__(self, mu=1.0):
        super().__init__("continuous", 2)
        self.mu = mu

    def f(self, t, x):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1
        return jnp.array([dx1, dx2])


class SimpleNonlinearSystem(System):
    """Simple nonlinear system: dx/dt = -x + x^2"""

    def __init__(self):
        super().__init__("continuous", 1)

    def f(self, t, x):
        return -x + 0.1 * x**2


class PendulumSystem(System):
    """Pendulum system: dθ/dt = ω, dω/dt = -sin(θ)"""

    def __init__(self):
        super().__init__("continuous", 2)

    def f(self, t, x):
        theta, omega = x[0], x[1]
        return jnp.array([omega, -jnp.sin(theta)])


# --- Fixtures ---


@pytest.fixture
def linear_system():
    """A simple stable linear system."""
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    return LinearSystem(A)


@pytest.fixture
def vanderpol_system():
    """Van der Pol oscillator."""
    return VanDerPolSystem(mu=0.5)


@pytest.fixture
def simple_nonlinear():
    """Simple 1D nonlinear system."""
    return SimpleNonlinearSystem()


@pytest.fixture
def pendulum_system():
    """Pendulum system."""
    return PendulumSystem()


@pytest.fixture
def initial_zonotope_2d():
    """Initial zonotope in R^2."""
    ox = jnp.array([1.0, 0.5])
    G = jnp.array([[0.1, 0.0], [0.0, 0.1]])
    return Zonotope(ox, G)


@pytest.fixture
def initial_zonotope_1d():
    """Initial zonotope in R^1."""
    ox = jnp.array([0.5])
    G = jnp.array([[0.1]])
    return Zonotope(ox, G)


# --- Test Lohner Algorithm ---


class TestLohnerReachability:
    def test_initialization(self, linear_system):
        reach = LohnerReachability(linear_system, dt=0.1)
        assert reach.sys == linear_system
        assert reach.dt == 0.1

    def test_rejects_discrete_systems(self):
        class DiscreteSystem(System):
            def __init__(self):
                super().__init__("discrete", 2)

            def f(self, t, x):
                return x

        with pytest.raises(ValueError, match="continuous"):
            LohnerReachability(DiscreteSystem(), dt=0.1)

    def test_single_step_linear(self, linear_system, initial_zonotope_2d):
        reach = LohnerReachability(linear_system, dt=0.1, taylor_order=2)
        Z_next = reach.step(0.0, initial_zonotope_2d)

        # For linear system, center should evolve as exp(A*dt) @ c
        # Approximate: c + dt * A @ c
        expected_center = initial_zonotope_2d.ox + 0.1 * linear_system.A @ initial_zonotope_2d.ox
        assert jnp.allclose(Z_next.ox, expected_center, atol=0.01)

    def test_single_step_nonlinear(self, simple_nonlinear, initial_zonotope_1d):
        reach = LohnerReachability(simple_nonlinear, dt=0.05, taylor_order=2)
        Z_next = reach.step(0.0, initial_zonotope_1d)

        # Check that result is a valid zonotope
        assert Z_next.n == 1
        assert Z_next.m > 0

    def test_generator_reduction(self, linear_system, initial_zonotope_2d):
        """Test that generator reduction keeps bounded number of generators."""
        reach = LohnerReachability(linear_system, dt=0.1, max_generators=4)

        Z = initial_zonotope_2d
        for _ in range(10):
            Z = reach.step(0.0, Z)

        # Should not exceed max_generators
        assert Z.m <= 4

    def test_compute_reachtube_linear(self, linear_system, initial_zonotope_2d):
        tube = lohner_reachtube(
            linear_system,
            t0=0.0,
            tf=1.0,
            Z0=initial_zonotope_2d,
            dt=0.1,
        )

        # Check tube structure
        assert len(tube) == 11  # 0.0, 0.1, ..., 1.0
        assert jnp.allclose(tube.times[0], 0.0)
        assert jnp.allclose(tube.times[-1], 1.0)

        # Check that first zonotope matches initial
        Z_first = tube[0]
        assert jnp.allclose(Z_first.ox, initial_zonotope_2d.ox)

    def test_compute_reachtube_nonlinear(self, vanderpol_system, initial_zonotope_2d):
        tube = lohner_reachtube(
            vanderpol_system,
            t0=0.0,
            tf=0.5,
            Z0=initial_zonotope_2d,
            dt=0.05,
            max_generators=6,
        )

        assert len(tube) == 11
        # Zonotopes should remain valid
        for i in range(len(tube)):
            Z = tube[i]
            assert Z.n == 2
            assert Z.m > 0


# --- Test Althoff-Girard Algorithm ---


class TestAlthoffGirardReachability:
    def test_initialization(self, linear_system):
        reach = AlthoffGirardReachability(linear_system, dt=0.1)
        assert reach.sys == linear_system
        assert reach.dt == 0.1

    def test_single_step_linear(self, linear_system, initial_zonotope_2d):
        reach = AlthoffGirardReachability(linear_system, dt=0.1)
        Z_next = reach.step(0.0, initial_zonotope_2d)

        # For linear system, should be similar to exact propagation
        expected_center = initial_zonotope_2d.ox + 0.1 * linear_system.A @ initial_zonotope_2d.ox
        assert jnp.allclose(Z_next.ox, expected_center, atol=0.02)

    def test_single_step_nonlinear(self, simple_nonlinear, initial_zonotope_1d):
        reach = AlthoffGirardReachability(simple_nonlinear, dt=0.05)
        Z_next = reach.step(0.0, initial_zonotope_1d)

        # Check that result is valid
        assert Z_next.n == 1
        assert Z_next.m > 0

    def test_generator_reduction(self, linear_system, initial_zonotope_2d):
        """Test Girard's reduction method."""
        reach = AlthoffGirardReachability(linear_system, dt=0.1, max_generators=4)

        Z = initial_zonotope_2d
        for _ in range(10):
            Z = reach.step(0.0, Z)

        assert Z.m <= 4

    def test_compute_reachtube_linear(self, linear_system, initial_zonotope_2d):
        tube = althoff_girard_reachtube(
            linear_system,
            t0=0.0,
            tf=1.0,
            Z0=initial_zonotope_2d,
            dt=0.1,
        )

        assert len(tube) == 11
        assert jnp.allclose(tube.times[0], 0.0)
        assert jnp.allclose(tube.times[-1], 1.0)

    def test_compute_reachtube_pendulum(self, pendulum_system, initial_zonotope_2d):
        # Start near equilibrium
        Z0 = Zonotope(jnp.array([0.1, 0.0]), jnp.array([[0.05, 0.0], [0.0, 0.05]]))

        tube = althoff_girard_reachtube(
            pendulum_system,
            t0=0.0,
            tf=0.5,
            Z0=Z0,
            dt=0.05,
            max_generators=6,
        )

        assert len(tube) == 11
        for i in range(len(tube)):
            Z = tube[i]
            assert Z.n == 2


# --- Test Overapproximation Property ---


class TestOverapproximation:
    """Test that the reachable tube actually contains trajectories."""

    def _interpolate_trajectory(self, ts, ys, t):
        """Linear interpolation of trajectory at time t."""
        # Find the index of the closest time step
        idx = jnp.searchsorted(ts, t)
        idx = jnp.clip(idx, 1, len(ts) - 1)
        t0, t1 = ts[idx - 1], ts[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        alpha = (t - t0) / (t1 - t0 + 1e-10)
        return y0 + alpha * (y1 - y0)

    def test_lohner_contains_trajectory(self, linear_system, initial_zonotope_2d):
        """Check that sampled trajectories stay within the tube."""
        tube = lohner_reachtube(
            linear_system,
            t0=0.0,
            tf=1.0,
            Z0=initial_zonotope_2d,
            dt=0.1,
        )

        # Sample initial conditions from zonotope
        key = jax.random.PRNGKey(42)
        n_samples = 5

        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            # Random point in initial zonotope
            v = jax.random.uniform(subkey, shape=(initial_zonotope_2d.m,), minval=-1, maxval=1)
            x0 = initial_zonotope_2d.ox + initial_zonotope_2d.G @ v

            # Compute trajectory
            raw_traj = linear_system.compute_trajectory(0.0, 1.0, x0, dt=0.01)
            ts = raw_traj.ts
            ys = raw_traj.ys

            # Check containment at tube time points
            for j, t in enumerate(tube.times):
                x_t = self._interpolate_trajectory(ts, ys, float(t))
                Z_t = tube[j]
                hull = Z_t.interval_hull()
                # Point should be within interval hull (necessary condition)
                assert jnp.all(x_t >= hull.lower - 1e-3), f"Trajectory escaped at t={t}"
                assert jnp.all(x_t <= hull.upper + 1e-3), f"Trajectory escaped at t={t}"

    def test_althoff_girard_contains_trajectory(self, linear_system, initial_zonotope_2d):
        """Check that sampled trajectories stay within the tube."""
        tube = althoff_girard_reachtube(
            linear_system,
            t0=0.0,
            tf=1.0,
            Z0=initial_zonotope_2d,
            dt=0.1,
        )

        key = jax.random.PRNGKey(123)
        n_samples = 5

        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            v = jax.random.uniform(subkey, shape=(initial_zonotope_2d.m,), minval=-1, maxval=1)
            x0 = initial_zonotope_2d.ox + initial_zonotope_2d.G @ v

            raw_traj = linear_system.compute_trajectory(0.0, 1.0, x0, dt=0.01)
            ts = raw_traj.ts
            ys = raw_traj.ys

            for j, t in enumerate(tube.times):
                x_t = self._interpolate_trajectory(ts, ys, float(t))
                Z_t = tube[j]
                hull = Z_t.interval_hull()
                assert jnp.all(x_t >= hull.lower - 1e-3), f"Trajectory escaped at t={t}"
                assert jnp.all(x_t <= hull.upper + 1e-3), f"Trajectory escaped at t={t}"


# --- Test ZonotopeReachTube ---


class TestZonotopeReachTube:
    def test_indexing(self, linear_system, initial_zonotope_2d):
        tube = lohner_reachtube(linear_system, 0.0, 0.5, initial_zonotope_2d, 0.1)

        Z0 = tube[0]
        assert isinstance(Z0, Zonotope)
        assert Z0.n == 2

    def test_at_time(self, linear_system, initial_zonotope_2d):
        tube = lohner_reachtube(linear_system, 0.0, 1.0, initial_zonotope_2d, 0.1)

        Z_mid = tube.at_time(0.55)
        assert isinstance(Z_mid, Zonotope)

    def test_len(self, linear_system, initial_zonotope_2d):
        tube = lohner_reachtube(linear_system, 0.0, 1.0, initial_zonotope_2d, 0.1)
        assert len(tube) == 11
