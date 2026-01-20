import jax
import jax.numpy as jnp
import pytest

from immrax.generator.sets.zonotope import (
    Zonotope,
    zonotope,
    zonotope_from_interval,
    zonotope_concatenate,
)
from immrax.inclusion.interval import Interval


# --- Fixtures ---


@pytest.fixture
def simple_zonotope():
    """A simple 2D zonotope with 2 generators."""
    ox = jnp.array([1.0, 2.0])
    G = jnp.array([[1.0, 0.5], [0.0, 1.0]])
    return Zonotope(ox, G)


@pytest.fixture
def unit_box_zonotope():
    """A unit box centered at origin (interval [-1,1]^2 as zonotope)."""
    ox = jnp.array([0.0, 0.0])
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return Zonotope(ox, G)


# --- Test Properties ---


class TestZonotopeProperties:
    def test_dimensions(self, simple_zonotope):
        assert simple_zonotope.n == 2
        assert simple_zonotope.m == 2

    def test_order(self, simple_zonotope):
        assert simple_zonotope.order == 1.0

    def test_shape(self, simple_zonotope):
        assert simple_zonotope.shape == (2,)

    def test_center(self, simple_zonotope):
        assert jnp.allclose(simple_zonotope.center, jnp.array([1.0, 2.0]))

    def test_generators(self, simple_zonotope):
        expected_G = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        assert jnp.allclose(simple_zonotope.generators, expected_G)


# --- Test Pytree ---


class TestZonotopePytree:
    def test_tree_flatten_unflatten(self, simple_zonotope):
        children, aux = simple_zonotope.tree_flatten()
        reconstructed = Zonotope.tree_unflatten(aux, children)
        assert jnp.allclose(reconstructed.ox, simple_zonotope.ox)
        assert jnp.allclose(reconstructed.G, simple_zonotope.G)

    def test_jit_compatible(self, simple_zonotope):
        @jax.jit
        def get_center(z):
            return z.center

        result = get_center(simple_zonotope)
        assert jnp.allclose(result, simple_zonotope.center)

    def test_vmap_over_operations(self, simple_zonotope):
        """Test that zonotope operations work inside vmap."""
        matrices = jnp.stack([jnp.eye(2), 2 * jnp.eye(2), 3 * jnp.eye(2)])

        @jax.vmap
        def apply_matrix(A):
            return (A @ simple_zonotope).center

        results = apply_matrix(matrices)
        assert results.shape == (3, 2)
        assert jnp.allclose(results[0], simple_zonotope.center)
        assert jnp.allclose(results[1], 2 * simple_zonotope.center)


# --- Test Set Operations ---


class TestMinkowskiSum:
    def test_minkowski_sum_same_dimension(self, simple_zonotope):
        result = simple_zonotope + simple_zonotope
        assert result.n == 2
        assert result.m == 4  # Generators concatenated
        assert jnp.allclose(result.center, 2 * simple_zonotope.center)

    def test_minkowski_sum_different_zonotopes(self, simple_zonotope, unit_box_zonotope):
        result = simple_zonotope + unit_box_zonotope
        assert result.n == 2
        assert result.m == 4
        assert jnp.allclose(result.center, simple_zonotope.center)

    def test_minkowski_sum_type_error(self, simple_zonotope):
        with pytest.raises(TypeError):
            simple_zonotope + jnp.array([1.0, 2.0])


class TestLinearMapping:
    def test_left_multiply(self, unit_box_zonotope):
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        result = A @ unit_box_zonotope
        assert jnp.allclose(result.center, jnp.array([0.0, 0.0]))
        expected_G = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        assert jnp.allclose(result.G, expected_G)

    def test_projection(self, simple_zonotope):
        # Project to first coordinate
        P = jnp.array([[1.0, 0.0]])
        result = P @ simple_zonotope
        assert result.n == 1
        assert result.m == 2

    def test_identity_mapping(self, simple_zonotope):
        I = jnp.eye(2)
        result = I @ simple_zonotope
        assert jnp.allclose(result.center, simple_zonotope.center)
        assert jnp.allclose(result.G, simple_zonotope.G)


class TestScalarMultiplication:
    def test_scalar_multiply(self, simple_zonotope):
        result = 2.0 * simple_zonotope
        assert jnp.allclose(result.center, 2.0 * simple_zonotope.center)
        assert jnp.allclose(result.G, 2.0 * simple_zonotope.G)

    def test_scalar_multiply_right(self, simple_zonotope):
        result = simple_zonotope * 2.0
        assert jnp.allclose(result.center, 2.0 * simple_zonotope.center)

    def test_negation(self, simple_zonotope):
        result = -simple_zonotope
        assert jnp.allclose(result.center, -simple_zonotope.center)
        assert jnp.allclose(result.G, -simple_zonotope.G)


class TestSubtraction:
    def test_subtraction(self, simple_zonotope, unit_box_zonotope):
        result = simple_zonotope - unit_box_zonotope
        assert result.n == 2
        assert result.m == 4
        assert jnp.allclose(result.center, simple_zonotope.center)


# --- Test Conversion Methods ---


class TestIntervalHull:
    def test_unit_box_hull(self, unit_box_zonotope):
        hull = unit_box_zonotope.interval_hull()
        assert jnp.allclose(hull.lower, jnp.array([-1.0, -1.0]))
        assert jnp.allclose(hull.upper, jnp.array([1.0, 1.0]))

    def test_simple_zonotope_hull(self, simple_zonotope):
        hull = simple_zonotope.interval_hull()
        # center = [1, 2], G = [[1, 0.5], [0, 1]]
        # radius = [1.5, 1.0]
        assert jnp.allclose(hull.lower, jnp.array([-0.5, 1.0]))
        assert jnp.allclose(hull.upper, jnp.array([2.5, 3.0]))


class TestContains:
    def test_center_contained(self, simple_zonotope):
        assert simple_zonotope.contains(simple_zonotope.center)

    def test_interior_point_contained(self, unit_box_zonotope):
        # Interior points should be contained
        assert unit_box_zonotope.contains(jnp.array([0.5, 0.5]))
        assert unit_box_zonotope.contains(jnp.array([-0.5, -0.5]))

    def test_outside_not_contained(self, unit_box_zonotope):
        # Point clearly outside
        assert not unit_box_zonotope.contains(jnp.array([2.0, 0.0]))


# --- Test Helper Functions ---


class TestZonotopeHelper:
    def test_zonotope_with_generators(self):
        z = zonotope(jnp.array([1.0, 2.0]), jnp.array([[1.0], [0.0]]))
        assert z.n == 2
        assert z.m == 1

    def test_zonotope_point(self):
        z = zonotope(jnp.array([1.0, 2.0]))
        assert z.n == 2
        assert z.m == 0


class TestZonotopeFromInterval:
    def test_unit_interval(self):
        interval = Interval(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
        z = zonotope_from_interval(interval)
        assert z.n == 2
        assert z.m == 2
        assert jnp.allclose(z.center, jnp.array([0.0, 0.0]))
        # Generators should be diagonal
        assert jnp.allclose(z.G, jnp.eye(2))

    def test_asymmetric_interval(self):
        interval = Interval(jnp.array([0.0, 1.0]), jnp.array([2.0, 5.0]))
        z = zonotope_from_interval(interval)
        assert jnp.allclose(z.center, jnp.array([1.0, 3.0]))
        assert jnp.allclose(z.G, jnp.diag(jnp.array([1.0, 2.0])))


class TestZonotopeConcatenate:
    def test_concatenate_two(self, unit_box_zonotope):
        z1 = unit_box_zonotope
        z2 = Zonotope(jnp.array([5.0]), jnp.array([[0.5]]))
        result = zonotope_concatenate([z1, z2])
        assert result.n == 3
        assert result.m == 3
        assert jnp.allclose(result.center, jnp.array([0.0, 0.0, 5.0]))
