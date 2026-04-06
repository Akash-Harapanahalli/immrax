import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from tests.utils import validate_overapproximation_nd
from immrax.parametric import *

# --- Helper Classes ---


# --- Fixtures for test data ---

@pytest.fixture(
    params=[
        LinfNormotope(jnp.array([0., 0.]), jnp.eye(2), jnp.array([1.])),
        L1Normotope(jnp.array([0., 0.]),   jnp.eye(2), jnp.array([1.])),
        L2Normotope(jnp.array([0., 0.]),   jnp.eye(2), jnp.array([1.])),
    ]
)
def normotope_type(request):
    """Parametrized fixture for evaluation points."""
    return request.param

# --- Test Functions ---

def test_normotope_contains(normotope_type: Normotope):
    # Tests contains function for the three types of normotopes
    assert normotope_type.contains(jnp.array([0.5, 0.5])) == True
    assert normotope_type.contains(jnp.array([1., 0.])) == True
    assert normotope_type.contains(jnp.array([1.1, 1.1])) == False

def test_polytope_contains():
    # Testing Polytopes
    x = 0.5 * jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0], [-1.0, 2.0], [2.0, -1.0]])
    b = jnp.array([1.0, 1.0, 1.0])
    # polytope = irx.Polytope(jnp.array([0., 0.]), A, jnp.hstack((b,b)))
    polytope = irx.Polytope.from_Hpolytope(A, b)
    assert polytope.contains(x) == True

def test_ellipsoid_contains():
    # Ellipsoid and L2Normotope should be mathematically identical, so sample random points and check if they are contained within both
    ellipsoid = Ellipsoid(jnp.array([0., 0.]), jnp.eye(2), jnp.array([1.]))
    l2n = L2Normotope(jnp.array([0., 0.]), jnp.eye(2), jnp.array([1.]))

    num_samples = 100
    sample_pts = jax.random.uniform(
                jax.random.PRNGKey(0),
                (num_samples,2),
                minval= -1.,
                maxval= 1.
            )
    
    for i in range(num_samples):
        assert ellipsoid.contains(sample_pts[i, :]) == l2n.contains(sample_pts[i, :])

def test_ellipsoid_projection():
    # This tests get_pojection_mtx
    P = jnp.diag(jnp.array([1., 1., 0.5]))
    M = Ellipsoid.get_projection_mtx(P, 0, 1)
    assert jnp.allclose(M, jnp.eye(2))
    pass

