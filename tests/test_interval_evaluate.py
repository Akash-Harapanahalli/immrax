"""Tests for TaylorPolynomial.interval_evaluate.

For each test polynomial, we verify that for many randomly sampled points
x inside the input box ix, the true value p.evaluate(x) lies within the
bounds returned by p.interval_evaluate(ix).
"""

import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.taylor import taylor_polynomial_identity, pjet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_contains_samples(p, ix, n_samples=200, atol=1e-5, method='horner'):
    """Check that p.interval_evaluate(ix, method) contains p.evaluate(x) for random x in ix."""
    key = jax.random.PRNGKey(42)
    samples = jax.random.uniform(
        key, (n_samples, p.d), minval=ix.lower, maxval=ix.upper
    )
    bound = p.interval_evaluate(ix, method=method)
    for i in range(n_samples):
        val = p.evaluate(samples[i])
        assert jnp.all(val >= bound.lower - atol), (
            f"Sample {i}: {val} < lower bound {bound.lower}"
        )
        assert jnp.all(val <= bound.upper + atol), (
            f"Sample {i}: {val} > upper bound {bound.upper}"
        )


# ---------------------------------------------------------------------------
# 1-D polynomials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_1d_identity(method):
    """p(x) = x, centered at 1, evaluated over [0.5, 1.5]."""
    center = jnp.array([1.0])
    p = taylor_polynomial_identity(center, order=1)
    ix = irx.icentpert(center, 0.5)
    assert_contains_samples(p, ix, method=method)


@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_1d_quadratic(method):
    """p(x) = x^2, centered at 0, evaluated over [-0.5, 0.5]."""
    center = jnp.array([0.0])
    x = taylor_polynomial_identity(center, order=2)
    p = pjet(lambda t: t ** 2)(x)
    ix = irx.icentpert(center, 0.5)
    assert_contains_samples(p, ix, method=method)


@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_1d_cubic(method):
    """p(x) = x^3 - x, centered at 0, evaluated over [-0.7, 0.7]."""
    center = jnp.array([0.0])
    x = taylor_polynomial_identity(center, order=3)
    p = pjet(lambda t: t ** 3 - t)(x)
    ix = irx.icentpert(center, 0.7)
    assert_contains_samples(p, ix, method=method)


@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_1d_off_center(method):
    """p(x) = (x-2)^2 + 1, centered at 2, evaluated over [1.5, 2.5]."""
    center = jnp.array([2.0])
    x = taylor_polynomial_identity(center, order=2)
    p = pjet(lambda t: t ** 2 + 1.0)(x)
    ix = irx.icentpert(center, 0.5)
    assert_contains_samples(p, ix, method=method)


# ---------------------------------------------------------------------------
# 2-D polynomials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_2d_linear(method):
    """p(x, y) = x + y, centered at (0, 0), evaluated over [-0.5, 0.5]^2."""
    center = jnp.array([0.0, 0.0])
    xy = taylor_polynomial_identity(center, order=1)
    p = pjet(lambda t: t[0:1] + t[1:2])(xy)
    ix = irx.icentpert(center, 0.5)
    assert_contains_samples(p, ix, method=method)


@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_2d_quadratic(method):
    """p(x, y) = x^2 + xy + y^2, centered at (0, 0), evaluated over [-0.5, 0.5]^2."""
    center = jnp.array([0.0, 0.0])
    xy = taylor_polynomial_identity(center, order=2)
    p = pjet(lambda t: t[0:1] ** 2 + t[0:1] * t[1:2] + t[1:2] ** 2)(xy)
    ix = irx.icentpert(center, 0.5)
    assert_contains_samples(p, ix, method=method)


@pytest.mark.parametrize("method", ["horner", "vmap", "cumprod"])
def test_2d_off_center(method):
    """p(x, y) = x*y, centered at (1, 1), evaluated over [0.7, 1.3]^2."""
    center = jnp.array([1.0, 1.0])
    xy = taylor_polynomial_identity(center, order=2)
    p = pjet(lambda t: t[0:1] * t[1:2])(xy)
    ix = irx.icentpert(center, 0.3)
    assert_contains_samples(p, ix, method=method)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_interval():
    """interval_evaluate should always return an Interval."""
    center = jnp.array([0.0])
    p = taylor_polynomial_identity(center, order=2)
    ix = irx.icentpert(center, 0.5)
    result = p.interval_evaluate(ix)
    assert isinstance(result, irx.Interval)


def test_invalid_method_raises():
    """interval_evaluate with an unknown method should raise ValueError."""
    center = jnp.array([0.0])
    p = taylor_polynomial_identity(center, order=1)
    ix = irx.icentpert(center, 0.5)
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        p.interval_evaluate(ix, method='bad')


# ---------------------------------------------------------------------------
# Agreement across methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("center,radius", [
    (jnp.array([0.0]), 0.5),
    (jnp.array([1.0, 2.0]), 0.3),
])
def test_horner_standard_agree(center, radius):
    """All methods should produce valid overapproximations that overlap."""
    p = taylor_polynomial_identity(center, order=2)
    ix = irx.icentpert(center, radius)
    h = p.interval_evaluate(ix, method='horner')
    s = p.interval_evaluate(ix, method='cumprod')
    # Both must be valid intervals
    assert jnp.all(h.upper >= h.lower)
    assert jnp.all(s.upper >= s.lower)
    # Their intersection must be non-empty (they bound the same function)
    assert jnp.all(jnp.minimum(h.upper, s.upper) >= jnp.maximum(h.lower, s.lower))
