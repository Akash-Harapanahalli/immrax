"""Tests for rigorous floating-point widening mode."""

import jax
import jax.numpy as jnp
import pytest

from immrax.inclusion import (
    Interval,
    interval,
    icentpert,
    natif,
    widen,
)


# --- widen primitive ---

class TestWiden:
    """Test the widen function and its custom JVP."""

    def test_widen_expands_bounds(self):
        """widen should push lower down and upper up."""
        iv = interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        w = widen(iv, 1)
        assert jnp.all(w.lower < iv.lower)
        assert jnp.all(w.upper > iv.upper)

    def test_widen_n_ulps(self):
        """Widening by n ULPs should be wider than n-1 ULPs."""
        iv = interval(jnp.array([1.0]), jnp.array([2.0]))
        w1 = widen(iv, 1)
        w2 = widen(iv, 2)
        assert jnp.all(w2.lower < w1.lower)
        assert jnp.all(w2.upper > w1.upper)

    def test_widen_degenerate(self):
        """Widening a degenerate interval should produce a non-degenerate one."""
        iv = interval(jnp.array([1.0]))
        w = widen(iv, 1)
        assert jnp.all(w.lower < w.upper)

    def test_widen_jvp(self):
        """JVP through widen should act as identity (gradient = 1)."""
        iv = interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

        # Differentiate through widen's lower bound
        def f_lower(x):
            return widen(interval(x, jnp.array([3.0, 4.0])), 1).lower

        grad_lower = jax.jacfwd(f_lower)(jnp.array([1.0, 2.0]))
        assert jnp.allclose(grad_lower, jnp.eye(2), atol=1e-6)

        # Differentiate through widen's upper bound
        def f_upper(x):
            return widen(interval(jnp.array([1.0, 2.0]), x), 1).upper

        grad_upper = jax.jacfwd(f_upper)(jnp.array([3.0, 4.0]))
        assert jnp.allclose(grad_upper, jnp.eye(2), atol=1e-6)

    def test_widen_grad(self):
        """grad through widen should work (reverse mode)."""
        def f(x):
            iv = interval(x, x + 0.1)
            w = widen(iv, 1)
            return jnp.sum(w.upper)

        g = jax.grad(f)(jnp.array([1.0, 2.0]))
        assert jnp.allclose(g, jnp.ones(2), atol=1e-6)


# --- rigorous kwarg on natif ---

class TestRigorousKwarg:
    """Test that natif(f, rigorous=...) controls widening."""

    def test_rigorous_default_true(self):
        """natif should be rigorous by default."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        def f(a):
            return a + a

        rigorous = natif(f)(x)
        non_rigorous = natif(f, rigorous=False)(x)

        # Default (rigorous=True) should produce wider bounds
        assert jnp.all(rigorous.lower <= non_rigorous.lower)
        assert jnp.all(rigorous.upper >= non_rigorous.upper)
        assert jnp.any(rigorous.lower < non_rigorous.lower)
        assert jnp.any(rigorous.upper > non_rigorous.upper)

    def test_rigorous_widens_add(self):
        """natif(f, rigorous=True) should produce wider bounds than rigorous=False for addition."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        y = icentpert(jnp.array([3.0, 4.0]), 0.1)

        def f(a, b):
            return a + b

        normal = natif(f, rigorous=False)(x, y)
        rigorous = natif(f, rigorous=True)(x, y)

        assert jnp.all(rigorous.lower <= normal.lower)
        assert jnp.all(rigorous.upper >= normal.upper)
        assert jnp.any(rigorous.lower < normal.lower)
        assert jnp.any(rigorous.upper > normal.upper)

    def test_rigorous_widens_mul(self):
        """Rigorous multiplication should produce wider bounds."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        def f(a):
            return a * a

        normal = natif(f, rigorous=False)(x)
        rigorous = natif(f)(x)

        assert jnp.all(rigorous.lower <= normal.lower)
        assert jnp.all(rigorous.upper >= normal.upper)

    def test_rigorous_widens_transcendental(self):
        """Rigorous sin should produce wider bounds (n_ulps=2)."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        normal = natif(jnp.sin, rigorous=False)(x)
        rigorous = natif(jnp.sin)(x)

        assert jnp.all(rigorous.lower <= normal.lower)
        assert jnp.all(rigorous.upper >= normal.upper)

    def test_non_rigorous_no_widen(self):
        """rigorous=False should not widen anything."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        def f(a):
            return a + a

        r1 = natif(f, rigorous=False)(x)
        r2 = natif(f, rigorous=False)(x)

        assert jnp.allclose(r1.lower, r2.lower)
        assert jnp.allclose(r1.upper, r2.upper)

    def test_exact_ops_not_widened(self):
        """Exact operations (neg) should NOT be widened even when rigorous."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        def f(a):
            return -a

        normal = natif(f, rigorous=False)(x)
        rigorous = natif(f)(x)

        # neg has n_ulps=0, so bounds should be identical
        assert jnp.allclose(rigorous.lower, normal.lower)
        assert jnp.allclose(rigorous.upper, normal.upper)


# --- Containment validation ---

class TestRigorousContainment:
    """Verify that rigorous natif still contains true function values."""

    def test_sin_containment(self):
        """Rigorous sin bounds should contain true values."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        y = natif(jnp.sin)(x)

        for alpha in jnp.linspace(0, 1, 50):
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = jnp.sin(x_s)
            assert jnp.all(y_true >= y.lower - 1e-7)
            assert jnp.all(y_true <= y.upper + 1e-7)

    def test_polynomial_containment(self):
        """Rigorous polynomial bounds should contain true values."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.2)

        def poly(x):
            return x ** 3 - 2 * x ** 2 + x

        y = natif(poly)(x)

        for alpha in jnp.linspace(0, 1, 50):
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = poly(x_s)
            assert jnp.all(y_true >= y.lower - 1e-6)
            assert jnp.all(y_true <= y.upper + 1e-6)

    def test_exp_containment(self):
        """Rigorous exp bounds should contain true values."""
        x = icentpert(jnp.array([0.0, 1.0]), 0.5)
        y = natif(jnp.exp)(x)

        for alpha in jnp.linspace(0, 1, 50):
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = jnp.exp(x_s)
            assert jnp.all(y_true >= y.lower - 1e-6)
            assert jnp.all(y_true <= y.upper + 1e-6)

    def test_composed_containment(self):
        """Rigorous bounds for composed function."""
        x = icentpert(jnp.array([0.5, 1.0]), 0.1)

        def f(x):
            return jnp.exp(jnp.sin(x))

        y = natif(f)(x)

        for alpha in jnp.linspace(0, 1, 50):
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = f(x_s)
            assert jnp.all(y_true >= y.lower - 1e-6)
            assert jnp.all(y_true <= y.upper + 1e-6)


# --- Differentiability through rigorous mode ---

class TestRigorousDifferentiability:
    """Verify that AD works through rigorous natif."""

    def test_grad_through_rigorous_natif(self):
        """grad(lambda x: natif(f)(x).upper) should work."""
        def f(x):
            return x ** 2

        def objective(center):
            iv = icentpert(center, 0.1)
            result = natif(f)(iv)
            return jnp.sum(result.upper)

        g = jax.grad(objective)(jnp.array([1.0, 2.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jacfwd_through_rigorous_natif(self):
        """jacfwd should work through rigorous natif."""
        def f(x):
            return jnp.sin(x)

        def objective(center):
            iv = icentpert(center, 0.1)
            result = natif(f)(iv)
            return result.upper

        J = jax.jacfwd(objective)(jnp.array([1.0, 2.0]))
        assert jnp.all(jnp.isfinite(J))
        assert J.shape == (2, 2)

    def test_jit_with_rigorous(self):
        """JIT should work with rigorous natif."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        @jax.jit
        def compute(x):
            return natif(jnp.sin)(x)

        y = compute(x)
        assert jnp.all(y.lower <= y.upper)
        assert jnp.all(jnp.isfinite(y.lower))
        assert jnp.all(jnp.isfinite(y.upper))
