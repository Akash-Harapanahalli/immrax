"""Tests for hmjacM: Higher-order Mixed Jacobian M tensors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import immrax as irx
from immrax.inclusion import hmjacM, mjacM, interval, icentpert, taylor_bounds


# --- Test functions ---

def quadratic(x):
    """f(x) = x^2, element-wise."""
    return x ** 2


def cubic(x):
    """f(x) = x^3, element-wise."""
    return x ** 3


def sin_fn(x):
    """f(x) = sin(x), element-wise."""
    return jnp.sin(x)


def multi_arg_fn(t, x):
    """f(t, x) = t * x + x^2"""
    return t * x + x ** 2


def coupled_fn(x):
    """f(x) = [x0*x1, x0^2 + x1]"""
    return jnp.array([x[0] * x[1], x[0] ** 2 + x[1]])


# --- Order 1: consistency with mjacM ---

class TestOrder1Consistency:
    """Verify that hmjacM(f, order=1) matches mjacM(f)."""

    def test_single_arg_scalar(self):
        """Single 1D argument."""
        x = icentpert(jnp.array([1.0]), 0.1)
        M_orig = mjacM(quadratic)(x)
        derivs, M_new = hmjacM(quadratic, order=1)(x)

        assert len(M_orig) == len(M_new)
        for orig, new in zip(M_orig, M_new):
            assert orig.shape == new.shape
            assert jnp.allclose(orig.lower, new.lower, atol=1e-6)
            assert jnp.allclose(orig.upper, new.upper, atol=1e-6)

        # derivs should contain [g(zc)] = [f(center)]
        assert len(derivs) == 1
        assert jnp.allclose(derivs[0], quadratic(x.center), atol=1e-6)

    def test_single_arg_vector(self):
        """Single 2D argument."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        M_orig = mjacM(coupled_fn)(x)
        derivs, M_new = hmjacM(coupled_fn, order=1)(x)

        assert len(M_orig) == len(M_new)
        for orig, new in zip(M_orig, M_new):
            assert orig.shape == new.shape
            assert jnp.allclose(orig.lower, new.lower, atol=1e-6)
            assert jnp.allclose(orig.upper, new.upper, atol=1e-6)

        assert len(derivs) == 1
        assert jnp.allclose(derivs[0], coupled_fn(x.center), atol=1e-6)

    def test_multi_arg(self):
        """Two arguments: f(t, x)."""
        t = icentpert(jnp.array([0.5]), 0.05)
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        center = ((t.lower + t.upper) / 2, (x.lower + x.upper) / 2)

        M_orig = mjacM(multi_arg_fn)(t, x, center=center)
        derivs, M_new = hmjacM(multi_arg_fn, order=1)(t, x, center=center)

        assert len(M_orig) == len(M_new)
        for orig, new in zip(M_orig, M_new):
            assert orig.shape == new.shape
            assert jnp.allclose(orig.lower, new.lower, atol=1e-6)
            assert jnp.allclose(orig.upper, new.upper, atol=1e-6)

        assert len(derivs) == 1
        assert jnp.allclose(derivs[0], multi_arg_fn(*center), atol=1e-6)


# --- Order 2: shape and correctness ---

class TestOrder2:
    """Test second-order hmjacM."""

    def test_shape_single_arg(self):
        """Check output tensor shape for order=2, single argument."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M2 = hmjacM(coupled_fn, order=2)(x)

        # One argument group
        assert len(M2) == 1
        block = M2[0]
        # Shape should be (m, n, n) = (2, 2, 2)
        assert block.shape == (2, 2, 2), f"Expected (2,2,2), got {block.shape}"

    def test_shape_multi_arg(self):
        """Check output tensor shape for order=2, multi-argument."""
        t = icentpert(jnp.array([0.5]), 0.05)
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M2 = hmjacM(multi_arg_fn, order=2)(t, x)

        # Two argument groups
        assert len(M2) == 2
        # Block for t: (m, n, n_t) = (2, 3, 1)
        assert M2[0].shape == (2, 3, 1), f"Expected (2,3,1), got {M2[0].shape}"
        # Block for x: (m, n, n_x) = (2, 3, 2)
        assert M2[1].shape == (2, 3, 2), f"Expected (2,3,2), got {M2[1].shape}"

    def test_bounds_valid(self):
        """Check that interval bounds are valid (lower <= upper)."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M2 = hmjacM(coupled_fn, order=2)(x)
        block = M2[0]
        assert jnp.all(block.lower <= block.upper)

    def test_contains_true_hessian(self):
        """Check that [D^2 f] contains the true Hessian at several sample points."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M2 = hmjacM(coupled_fn, order=2)(x)
        block = M2[0]  # shape (2, 2, 2) = Interval Hessian

        # Compute true Hessian at several points in the interval
        hess_fn = jax.jacfwd(jax.jacfwd(coupled_fn))
        for alpha in jnp.linspace(0, 1, 20):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            H_true = hess_fn(x_sample)  # shape (2, 2, 2)
            assert jnp.all(H_true >= block.lower - 1e-6), \
                f"Hessian below lower bound at alpha={alpha}"
            assert jnp.all(H_true <= block.upper + 1e-6), \
                f"Hessian above upper bound at alpha={alpha}"

    def test_quadratic_exact(self):
        """For f(x) = x^2, the Hessian is exactly diag(2), independent of x."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.2)

        def f(x):
            return x ** 2

        derivs, M2 = hmjacM(f, order=2)(x)
        block = M2[0]  # shape (2, 2, 2)

        # True Hessian of x^2 is diag(2, 2) (diagonal in the last two axes)
        H_true = jax.jacfwd(jax.jacfwd(f))(x.center)
        assert jnp.allclose(block.lower, H_true, atol=1e-5)
        assert jnp.allclose(block.upper, H_true, atol=1e-5)

    def test_derivs_order2(self):
        """Check that derivs contains [g(zc), Dg(zc)] for order=2."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M2 = hmjacM(coupled_fn, order=2)(x)

        # derivs[0] = g(zc) = f(center), shape (2,)
        assert jnp.allclose(derivs[0], coupled_fn(x.center), atol=1e-6)
        # derivs[1] = Dg(zc) = Jacobian at center, shape (2, 2)
        J_true = jax.jacfwd(coupled_fn)(x.center)
        assert jnp.allclose(derivs[1], J_true, atol=1e-6)


# --- Order 3: shape and basic validation ---

class TestOrder3:
    """Test third-order hmjacM."""

    def test_shape(self):
        """Check output tensor shape for order=3."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M3 = hmjacM(cubic, order=3)(x)

        assert len(M3) == 1
        block = M3[0]
        # Shape: (m, n, n, n) = (2, 2, 2, 2)
        assert block.shape == (2, 2, 2, 2), f"Expected (2,2,2,2), got {block.shape}"

    def test_contains_true_derivative(self):
        """Check that [D^3 f] contains the true 3rd derivative at sample points."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M3 = hmjacM(cubic, order=3)(x)
        block = M3[0]

        d3f = jax.jacfwd(jax.jacfwd(jax.jacfwd(cubic)))
        for alpha in jnp.linspace(0, 1, 20):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            D3_true = d3f(x_sample)
            assert jnp.all(D3_true >= block.lower - 1e-5)
            assert jnp.all(D3_true <= block.upper + 1e-5)

    def test_cubic_exact(self):
        """For f(x) = x^3, D^3 f = diag(6), constant."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.2)
        derivs, M3 = hmjacM(cubic, order=3)(x)
        block = M3[0]

        D3_true = jax.jacfwd(jax.jacfwd(jax.jacfwd(cubic)))(x.center)
        assert jnp.allclose(block.lower, D3_true, atol=1e-5)
        assert jnp.allclose(block.upper, D3_true, atol=1e-5)

    def test_derivs_order3(self):
        """Check that derivs contains [g(zc), Dg(zc), D^2g(zc)] for order=3."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M3 = hmjacM(cubic, order=3)(x)

        # derivs[0] = g(zc), shape (2,)
        assert jnp.allclose(derivs[0], cubic(x.center), atol=1e-6)
        # derivs[1] = Dg(zc), shape (2, 2)
        J_true = jax.jacfwd(cubic)(x.center)
        assert jnp.allclose(derivs[1], J_true, atol=1e-6)
        # derivs[2] = D^2 g(zc), shape (2, 2, 2)
        H_true = jax.jacfwd(jax.jacfwd(cubic))(x.center)
        assert jnp.allclose(derivs[2], H_true, atol=1e-6)


# --- Taylor inclusion validation ---

class TestTaylorInclusion:
    """Validate the Taylor inclusion property:
    f(x) - T_{k-1}(x; x') in (1/k!) [D^k f] * (x - x')^{otimes k}
    """

    def test_order1_inclusion(self):
        """Order 1: f(x) - f(x') should be bounded by [M](x - x')."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        center = x.center

        derivs, M1 = hmjacM(sin_fn, order=1)(x)
        M_block = M1[0]  # shape (2, 2)

        # derivs[0] should be f(center)
        assert jnp.allclose(derivs[0], sin_fn(center), atol=1e-6)

        for alpha in jnp.linspace(0, 1, 50):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            residual = sin_fn(x_sample) - derivs[0]
            dx = x_sample - center

            # residual should be in [M_block] @ dx
            bound = interval(M_block) @ interval(dx)
            assert jnp.all(residual >= bound.lower - 1e-6)
            assert jnp.all(residual <= bound.upper + 1e-6)

    def test_order2_inclusion(self):
        """Order 2: f(x) - f(x') - Df(x')(x-x') should be bounded by (1/2)[D^2 f](x-x')^2."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        center = x.center

        derivs, M2 = hmjacM(sin_fn, order=2)(x)
        H_block = M2[0]  # shape (2, 2, 2)

        # Use derivs directly instead of recomputing
        f_center = derivs[0]  # g(zc) = f(center)
        Df_center = derivs[1]  # Dg(zc) = Jacobian at center, shape (2, 2)

        for alpha in jnp.linspace(0, 1, 50):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            dx = x_sample - center
            # Residual after first-order Taylor subtraction
            residual = sin_fn(x_sample) - f_center - Df_center @ dx

            # Bound: (1/2) * H_block contracted with dx twice
            Hdx = interval(H_block) @ interval(dx)  # (2, 2)
            bound = (interval(Hdx) @ interval(dx)) * 0.5

            assert jnp.all(residual >= bound.lower - 1e-5), \
                f"Order-2 inclusion violated at alpha={alpha}"
            assert jnp.all(residual <= bound.upper + 1e-5), \
                f"Order-2 inclusion violated at alpha={alpha}"


# --- Edge cases ---

class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_order_zero_raises(self):
        """order=0 should raise ValueError."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            hmjacM(quadratic, order=0)

    def test_single_dimension(self):
        """1D input, 1D output."""
        x = icentpert(jnp.array([1.0]), 0.1)
        derivs1, M1 = hmjacM(sin_fn, order=1)(x)
        assert M1[0].shape == (1, 1)

        derivs2, M2 = hmjacM(sin_fn, order=2)(x)
        assert M2[0].shape == (1, 1, 1)


# --- taylor_bounds ---

class TestTaylorBounds:
    """Test the taylor_bounds function."""

    def test_order1_contains_true_values(self):
        """Order-1 Taylor bound should contain f(x) for all x in the interval."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M = hmjacM(sin_fn, order=1)(x)
        bound = taylor_bounds(derivs, M, x.center)

        for alpha in jnp.linspace(0, 1, 50):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            y_true = sin_fn(x_sample)
            y_bound = bound(interval(x_sample))
            assert jnp.all(y_true >= y_bound.lower - 1e-6)
            assert jnp.all(y_true <= y_bound.upper + 1e-6)

    def test_order2_contains_true_values(self):
        """Order-2 Taylor bound should contain f(x) for all x in the interval."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M = hmjacM(sin_fn, order=2)(x)
        bound = taylor_bounds(derivs, M, x.center)

        for alpha in jnp.linspace(0, 1, 50):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            y_true = sin_fn(x_sample)
            y_bound = bound(interval(x_sample))
            assert jnp.all(y_true >= y_bound.lower - 1e-6)
            assert jnp.all(y_true <= y_bound.upper + 1e-6)

    def test_order3_contains_true_values(self):
        """Order-3 Taylor bound should contain f(x) for all x in the interval."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M = hmjacM(cubic, order=3)(x)
        bound = taylor_bounds(derivs, M, x.center)

        for alpha in jnp.linspace(0, 1, 50):
            x_sample = x.lower + alpha * (x.upper - x.lower)
            y_true = cubic(x_sample)
            y_bound = bound(interval(x_sample))
            assert jnp.all(y_true >= y_bound.lower - 1e-6)
            assert jnp.all(y_true <= y_bound.upper + 1e-6)

    def test_interval_input(self):
        """taylor_bounds should accept interval inputs and return valid bounds."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M = hmjacM(sin_fn, order=2)(x)
        bound = taylor_bounds(derivs, M, x.center)

        y_bound = bound(x)
        assert jnp.all(y_bound.lower <= y_bound.upper)

        # Should contain f at center
        y_center = sin_fn(x.center)
        assert jnp.all(y_center >= y_bound.lower - 1e-6)
        assert jnp.all(y_center <= y_bound.upper + 1e-6)

    def test_higher_order_tighter(self):
        """Higher order should give tighter bounds (for smooth functions)."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)

        derivs1, M1 = hmjacM(sin_fn, order=1)(x)
        bound1 = taylor_bounds(derivs1, M1, x.center)
        y1 = bound1(x)
        width1 = jnp.sum(y1.upper - y1.lower)

        derivs2, M2 = hmjacM(sin_fn, order=2)(x)
        bound2 = taylor_bounds(derivs2, M2, x.center)
        y2 = bound2(x)
        width2 = jnp.sum(y2.upper - y2.lower)

        assert width2 < width1, f"Order 2 ({width2}) not tighter than order 1 ({width1})"

    def test_multi_arg(self):
        """taylor_bounds should work with multi-argument functions."""
        t = icentpert(jnp.array([0.5]), 0.05)
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        center = (t.center, x.center)

        derivs, M = hmjacM(multi_arg_fn, order=2)(t, x, center=center)
        bound = taylor_bounds(derivs, M, center)

        # Test with separate arguments (no concatenation needed)
        for alpha in jnp.linspace(0, 1, 50):
            t_s = t.lower + alpha * (t.upper - t.lower)
            x_s = x.lower + alpha * (x.upper - x.lower)
            y_true = multi_arg_fn(t_s, x_s)
            y_bound = bound(interval(t_s), interval(x_s))
            assert jnp.all(y_true >= y_bound.lower - 1e-5)
            assert jnp.all(y_true <= y_bound.upper + 1e-5)

    def test_quadratic_exact_at_center(self):
        """At the center point, the bound should be tight (equal to f(center))."""
        x = icentpert(jnp.array([1.0, 2.0]), 0.1)
        derivs, M = hmjacM(quadratic, order=2)(x)
        bound = taylor_bounds(derivs, M, x.center)

        y_bound = bound(interval(x.center))
        y_true = quadratic(x.center)
        assert jnp.allclose(y_bound.lower, y_true, atol=1e-6)
        assert jnp.allclose(y_bound.upper, y_true, atol=1e-6)
