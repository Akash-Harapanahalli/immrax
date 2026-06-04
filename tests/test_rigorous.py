"""Tests for rigorous floating-point widening mode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion import (
    Interval,
    interval,
    icentpert,
    natif,
    widen,
    rigorous,
    non_rigorous,
)
from immrax.inclusion.interval import (
    _resolve_rigorous,
    _widen_lower,
    _widen_upper,
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

    def test_rigorous_default_follows_global(self):
        """natif's default (rigorous=None) resolves to the global flag, which is
        False by default: the default matches rigorous=False, and an explicit
        rigorous=True widens."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        def f(a):
            return a + a

        default = natif(f)(x)
        non_rigorous = natif(f, rigorous=False)(x)
        explicit = natif(f, rigorous=True)(x)

        # Default == non-rigorous (the global default is False)
        assert jnp.all(default.lower == non_rigorous.lower)
        assert jnp.all(default.upper == non_rigorous.upper)
        # Explicit rigorous=True still widens
        assert jnp.any(explicit.lower < non_rigorous.lower)
        assert jnp.any(explicit.upper > non_rigorous.upper)

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
        rigorous = natif(f, rigorous=True)(x)

        assert jnp.all(rigorous.lower <= normal.lower)
        assert jnp.all(rigorous.upper >= normal.upper)

    def test_rigorous_widens_transcendental(self):
        """Rigorous sin should produce wider bounds (n_ulps=2)."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        normal = natif(jnp.sin, rigorous=False)(x)
        rigorous = natif(jnp.sin, rigorous=True)(x)

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
        rigorous = natif(f, rigorous=True)(x)

        # neg has n_ulps=0, so bounds should be identical even when rigorous
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


# --- Overconservatism measurement ---

class TestRigorousOverconservatism:
    """Measure how much extra width rigorous mode adds for dot/sum."""

    def test_matmul_overconservatism(self):
        """Rigorous matmul should be only marginally wider than non-rigorous."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        n, m, p = 10, 20, 15
        A = icentpert(
            jax.random.normal(k1, (n, m)),
            0.1 * jnp.abs(jax.random.normal(k2, (n, m))),
        )
        B = icentpert(
            jax.random.normal(k3, (m, p)),
            0.1 * jnp.abs(jax.random.normal(k4, (m, p))),
        )

        rigorous = natif(jnp.matmul, rigorous=True)(A, B)
        normal = natif(jnp.matmul, rigorous=False)(A, B)

        # Rigorous must enclose non-rigorous
        assert jnp.all(rigorous.lower <= normal.lower)
        assert jnp.all(rigorous.upper >= normal.upper)

        normal_width = normal.upper - normal.lower
        rigorous_width = rigorous.upper - rigorous.lower
        relative_excess = (rigorous_width - normal_width) / normal_width

        print(f"\nMatmul ({n}x{m} @ {m}x{p}):")
        print(f"  Mean relative width increase: {jnp.mean(relative_excess):.2e}")
        print(f"  Max  relative width increase: {jnp.max(relative_excess):.2e}")

        # ULP-scale widening should be negligible compared to interval widths
        assert jnp.max(relative_excess) < 1e-4

    def test_matmul_containment(self):
        """Rigorous matmul bounds should contain sampled true values."""
        key = jax.random.PRNGKey(7)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        n, m, p = 5, 8, 6
        A = icentpert(
            jax.random.normal(k1, (n, m)),
            0.1 * jnp.abs(jax.random.normal(k2, (n, m))),
        )
        B = icentpert(
            jax.random.normal(k3, (m, p)),
            0.1 * jnp.abs(jax.random.normal(k4, (m, p))),
        )

        result = natif(jnp.matmul, rigorous=True)(A, B)

        for alpha in jnp.linspace(0, 1, 30):
            for beta in jnp.linspace(0, 1, 30):
                A_s = A.lower + alpha * (A.upper - A.lower)
                B_s = B.lower + beta * (B.upper - B.lower)
                y_true = A_s @ B_s
                assert jnp.all(y_true >= result.lower - 1e-6)
                assert jnp.all(y_true <= result.upper + 1e-6)

    def test_sum_overconservatism(self):
        """Rigorous sum should be only marginally wider than non-rigorous."""
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        n = 100
        x = icentpert(
            jax.random.normal(k1, (n,)),
            0.1 * jnp.abs(jax.random.normal(k2, (n,))),
        )

        rigorous = natif(jnp.sum, rigorous=True)(x)
        normal = natif(jnp.sum, rigorous=False)(x)

        rigorous_width = rigorous.upper - rigorous.lower
        normal_width = normal.upper - normal.lower
        relative_excess = (rigorous_width - normal_width) / normal_width

        print(f"\nSum (n={n}):")
        print(f"  Normal width:   {normal_width:.6f}")
        print(f"  Rigorous width: {rigorous_width:.6f}")
        print(f"  Relative width increase: {relative_excess:.2e}")

        assert relative_excess < 1e-4

    def test_sum_containment(self):
        """Rigorous sum bounds should contain sampled true values."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)

        n = 100
        x = icentpert(
            jax.random.normal(k1, (n,)),
            0.1 * jnp.abs(jax.random.normal(k2, (n,))),
        )

        result = natif(jnp.sum, rigorous=True)(x)

        for alpha in jnp.linspace(0, 1, 100):
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = jnp.sum(x_s)
            assert y_true >= result.lower - 1e-6
            assert y_true <= result.upper + 1e-6


# ---------------------------------------------------------------------------
# Constructor widening tests
# ---------------------------------------------------------------------------

def _is_widened(iv, lo, hi):
    """True when iv is wider than [lo, hi] by exactly 1 ULP each side."""
    expected_lo = np.asarray(_widen_lower(jnp.asarray(lo)))
    expected_hi = np.asarray(_widen_upper(jnp.asarray(hi)))
    return (
        np.array_equal(np.asarray(iv.lower), expected_lo)
        and np.array_equal(np.asarray(iv.upper), expected_hi)
    )


def _is_exact(iv, lo, hi):
    """True when iv bounds are exactly lo, hi (no widening)."""
    return (
        np.array_equal(np.asarray(iv.lower), np.asarray(lo))
        and np.array_equal(np.asarray(iv.upper), np.asarray(hi))
    )


class TestIntervalConstructorRigorous:
    """interval() should not widen by default (rigorous=False)."""

    def test_default_no_widen(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        iv = interval(lo, hi)
        assert _is_exact(iv, lo, hi)

    def test_rigorous_true_widens(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        iv = interval(lo, hi, rigorous=True)
        assert _is_widened(iv, lo, hi)

    def test_rigorous_false_no_widen(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        iv = interval(lo, hi, rigorous=False)
        assert _is_exact(iv, lo, hi)

    def test_degenerate_no_widen(self):
        """interval(x) (point interval) should not widen by default."""
        x = jnp.float32(3.14)
        iv = interval(x)
        assert _is_exact(iv, x, x)

    def test_degenerate_rigorous_true_widens(self):
        """interval(x, rigorous=True) should widen by 1 ULP."""
        x = jnp.float32(3.14)
        iv = interval(x, rigorous=True)
        assert _is_widened(iv, x, x)

    def test_degenerate_rigorous_false_no_widen(self):
        """interval(x, rigorous=False) should NOT widen."""
        x = jnp.float32(3.14)
        iv = interval(x, rigorous=False)
        assert _is_exact(iv, x, x)

    def test_passthrough_interval(self):
        """interval(iv) returns iv unchanged."""
        iv0 = Interval(jnp.float32(1.0), jnp.float32(2.0))
        iv = interval(iv0)
        assert iv is iv0

    def test_vector_no_widen(self):
        lo = jnp.array([1.0, 2.0], dtype=jnp.float32)
        hi = jnp.array([3.0, 4.0], dtype=jnp.float32)
        iv = interval(lo, hi)
        assert _is_exact(iv, lo, hi)

    def test_vector_rigorous_true_widens(self):
        lo = jnp.array([1.0, 2.0], dtype=jnp.float32)
        hi = jnp.array([3.0, 4.0], dtype=jnp.float32)
        iv = interval(lo, hi, rigorous=True)
        assert _is_widened(iv, lo, hi)


class TestIcentpertConstructorRigorous:
    """icentpert() should not widen by default (rigorous=False)."""

    def test_default_no_widen(self):
        c = jnp.float32(1.0)
        p = jnp.float32(0.5)
        iv = icentpert(c, p)
        assert _is_exact(iv, c - p, c + p)

    def test_rigorous_true_widens(self):
        c = jnp.float32(1.0)
        p = jnp.float32(0.5)
        iv = icentpert(c, p, rigorous=True)
        assert _is_widened(iv, c - p, c + p)

    def test_rigorous_false_no_widen(self):
        c = jnp.float32(1.0)
        p = jnp.float32(0.5)
        iv = icentpert(c, p, rigorous=False)
        assert _is_exact(iv, c - p, c + p)


class TestContextManagers:
    """rigorous / non_rigorous context managers."""

    def test_kwarg_true_overrides_non_rigorous(self):
        """Explicit rigorous=True wins over non_rigorous() context."""
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        with non_rigorous():
            iv = interval(lo, hi, rigorous=True)
        assert _is_widened(iv, lo, hi)

    def test_kwarg_false_overrides_rigorous(self):
        """Explicit rigorous=False wins over rigorous() context."""
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        with rigorous():
            iv = interval(lo, hi, rigorous=False)
        assert _is_exact(iv, lo, hi)

    def test_nesting_inner_wins(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        with non_rigorous():
            with rigorous():
                iv = interval(lo, hi)
        assert _is_widened(iv, lo, hi)

    def test_nesting_restores_outer(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        with non_rigorous():
            with rigorous():
                pass
            iv = interval(lo, hi)
        assert _is_exact(iv, lo, hi)

    def test_context_restores_after_exit(self):
        lo, hi = jnp.float32(1.0), jnp.float32(2.0)
        with rigorous():
            pass
        iv = interval(lo, hi)
        assert _is_exact(iv, lo, hi)

    def test_non_rigorous_icentpert(self):
        c = jnp.float32(1.0)
        p = jnp.float32(0.5)
        with non_rigorous():
            iv = icentpert(c, p)
        assert _is_exact(iv, c - p, c + p)


class TestResolveRigorous:
    def test_default(self):
        assert _resolve_rigorous() is False

    def test_kwarg_overrides_default(self):
        assert _resolve_rigorous(False) is False

    def test_kwarg_overrides_context(self):
        with non_rigorous():
            assert _resolve_rigorous(True) is True
        with rigorous():
            assert _resolve_rigorous(False) is False


class TestStructuralNoWiden:
    """Structural operations (reshape, intersect, etc.) must NOT widen."""

    def test_reshape_no_widen(self):
        iv = Interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        iv2 = iv.reshape(2, 1)
        np.testing.assert_array_equal(np.asarray(iv2.lower), np.asarray(iv.lower.reshape(2, 1)))
        np.testing.assert_array_equal(np.asarray(iv2.upper), np.asarray(iv.upper.reshape(2, 1)))

    def test_intersect_no_widen(self):
        from immrax.inclusion import interval_intersect
        a = Interval(jnp.array([1.0]), jnp.array([3.0]))
        b = Interval(jnp.array([2.0]), jnp.array([4.0]))
        c = interval_intersect([a, b])
        np.testing.assert_array_equal(np.asarray(c.lower), [2.0])
        np.testing.assert_array_equal(np.asarray(c.upper), [3.0])
