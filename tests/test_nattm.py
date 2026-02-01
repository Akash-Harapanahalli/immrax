"""Tests for the nattm (Natural Taylor Model Function) interpreter."""

import jax
import jax.numpy as jnp
import pytest

from immrax.inclusion import Interval, interval
from immrax.taylor import (
    TaylorModel,
    taylor_model,
    taylor_model_from_function,
    nattm,
    _get_canonical_exponents,
)


# --- Helpers ---

def make_identity_tm(center, radius, order=2):
    """Create a TM representing the identity function x on a 1D domain.

    p(u) = center + radius * u, remainder = 0, where u in [-1, 1].
    """
    center = jnp.atleast_1d(jnp.asarray(center, dtype=jnp.float32))
    radius = jnp.atleast_1d(jnp.asarray(radius, dtype=jnp.float32))
    d = center.shape[0]
    n = center.shape[0]
    exponents = _get_canonical_exponents(d, order)
    m = exponents.shape[1]
    coeffs = jnp.zeros((n, m), dtype=jnp.float32)
    # Set constant term (exponent [0,...,0])
    const_idx = int(jnp.argmin(jnp.sum(exponents, axis=0)))
    coeffs = coeffs.at[:, const_idx].set(center)
    # Set linear terms: for variable i, exponent e_i
    for i in range(d):
        ei = jnp.zeros(d, dtype=jnp.int32).at[i].set(1)
        for j in range(m):
            if jnp.all(exponents[:, j] == ei):
                coeffs = coeffs.at[i, j].set(radius[i])
                break
    remainder = interval(jnp.zeros(n, dtype=jnp.float32))
    return TaylorModel(coeffs, exponents, remainder, center, radius, _static_order=order)


def tm_hull_contains_samples(f, tm, n_samples=200, atol=1e-5):
    """Check that the TM's interval hull contains sampled function values."""
    hull = tm.interval_hull()
    d = tm.d
    key = jax.random.PRNGKey(0)
    samples = jax.random.uniform(key, (n_samples, d),
                                  minval=tm.domain_center - tm.domain_radius,
                                  maxval=tm.domain_center + tm.domain_radius)
    for i in range(n_samples):
        val = f(samples[i])
        assert jnp.all(val >= hull.lower - atol), (
            f"Sample {i}: {val} < hull.lower {hull.lower}"
        )
        assert jnp.all(val <= hull.upper + atol), (
            f"Sample {i}: {val} > hull.upper {hull.upper}"
        )


# --- Univariate transcendental functions ---

@pytest.mark.parametrize("f,center", [
    (jnp.exp, 0.0),
    (jnp.exp, 1.0),
    (jnp.log, 1.0),
    (jnp.log, 2.0),
    (jnp.sin, 0.0),
    (jnp.cos, 0.0),
    (jnp.tanh, 0.0),
    (jnp.sqrt, 1.0),
])
def test_univariate_runs(f, center):
    """nattm of a univariate function should produce a valid TM."""
    radius = 0.3
    tm_x = make_identity_tm(center, radius, order=3)
    tm_f = nattm(f)(tm_x)
    hull = tm_f.interval_hull()
    assert jnp.all(hull.upper >= hull.lower), "Invalid interval hull"


@pytest.mark.parametrize("f,center", [
    pytest.param(jnp.tan, 0.0, id="tan"),
    pytest.param(jnp.arcsin, 0.0, id="arcsin"),
    pytest.param(jnp.arctan, 0.0, id="arctan"),
])
def test_univariate_no_jet_rule(f, center):
    """Primitives without jet rules should raise KeyError (known limitation)."""
    radius = 0.3
    tm_x = make_identity_tm(center, radius, order=3)
    with pytest.raises(KeyError):
        nattm(f)(tm_x)


# --- Arithmetic ---

def test_add():
    f = lambda x: x[0:1] + x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_sub():
    f = lambda x: x[0:1] - x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_mul():
    f = lambda x: x[0:1] * x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = make_identity_tm(center, radius, order=3)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_neg():
    f = lambda x: -x
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = make_identity_tm(center, radius)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_scalar_mul():
    f = lambda x: 3.0 * x
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = make_identity_tm(center, radius)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_integer_pow():
    f = lambda x: x ** 3
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = make_identity_tm(center, radius, order=3)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Structural operations ---

def test_reshape():
    def f(x):
        y = x * 2.0
        return y.reshape(1, -1).reshape(-1)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_broadcast():
    def f(x):
        return x * jnp.ones(2)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_slice():
    def f(x):
        return x[0:1] * 2.0

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_concatenate():
    def f(x):
        a = x[0:1]
        b = x[1:2]
        return jnp.concatenate([a * 2, b + 1])

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Passthrough operations ---

def test_passthrough_copy():
    """Passthrough ops should preserve TM structure."""
    def f(x):
        return x + 0.0  # triggers copy/add

    center = jnp.array([1.0])
    radius = jnp.array([0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    hull = tm_f.interval_hull()
    assert jnp.all(hull.upper >= hull.lower)


# --- Dot products / matmul ---

def test_matvec():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    def f(x):
        return A @ x

    center = jnp.array([1.0, 0.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Reductions ---

def test_reduce_sum():
    def f(x):
        return jnp.sum(x, keepdims=True)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Compositions ---

def test_polynomial_composition():
    """A purely polynomial composition should be exact (up to truncation)."""
    def f(x):
        return x ** 2 + 2 * x + 1

    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = make_identity_tm(center, radius, order=3)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_linear_system():
    """Matrix multiply + addition (linear system)."""
    A = jnp.array([[1.0, 2.0], [1.0, 1.0]])
    b = jnp.array([0.5, -0.5])

    def f(x):
        return A @ x + b

    center = jnp.array([0.0, 0.0])
    radius = jnp.array([1.0, 1.0])
    tm_x = make_identity_tm(center, radius, order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)
