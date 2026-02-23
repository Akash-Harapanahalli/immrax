"""Tests for mdit (Mixed Derivative Interval Tensor transform).

The key invariant under test:

    For all x ∈ ix:  f(x) ∈ poly(x) + M.contract(x − xc, scale=True)

where poly(x) = Σ_{k=0}^{p} T_k.contract(x − xc, scale=True) is the
degree-p Taylor polynomial and M = result[p+1] is the interval-valued MDIT
that bounds the remainder.
"""

import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.inclusion import (
    isinterval,
    MultiIndex,
    taylor_approx,
    mdit,
)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def f_sin(x):
    return jnp.sin(x)


def f_exp(x):
    return jnp.exp(x)


def f_cubic(x):
    """Degree-3 polynomial; 4th derivative is identically 0."""
    return x**3


def f_sin_cos_2d(x):
    return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1])])


def f_quadratic_2d(x):
    return jnp.array([x[0] ** 2 + x[1] ** 2])


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3])
def test_output_length(p):
    """mdit should return p+2 tensors: T_0 .. T_p then M."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    result = mdit(f_sin, p)(ix, xc)
    assert len(result) == p + 2


@pytest.mark.parametrize("p", [1, 2, 3])
def test_tensor_orders(p):
    """T_k should have .p == k; the MDIT M should have .p == p+1."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    result = mdit(f_sin, p)(ix, xc)
    for k in range(p + 1):
        assert result[k].p == k, f"T_{k}.p == {result[k].p}, expected {k}"
    assert result[p + 1].p == p + 1


@pytest.mark.parametrize("p", [1, 2, 3])
def test_mdit_entries_are_intervals(p):
    """Every stored entry of the MDIT M should be an Interval."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    M = mdit(f_sin, p)(ix, xc)[p + 1]
    assert len(M) > 0, "MDIT has no entries"
    for alpha, val in M.items():
        assert isinterval(val), f"M[{alpha}] is not an Interval"


@pytest.mark.parametrize("p", [1, 2, 3])
def test_mdit_intervals_valid(p):
    """All MDIT interval entries must satisfy lower <= upper."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    M = mdit(f_sin, p)(ix, xc)[p + 1]
    for alpha, val in M.items():
        assert jnp.all(val.lower <= val.upper), (
            f"Invalid interval at {alpha}: [{val.lower}, {val.upper}]"
        )


# ---------------------------------------------------------------------------
# Derivative correctness
# ---------------------------------------------------------------------------


def test_order0_equals_function_at_center():
    """T_0 should store f(xc)."""
    xc = jnp.array([1.0])
    ix = irx.icentpert(xc, 0.3)
    result = mdit(f_sin, 2)(ix, xc)
    val = result[0][MultiIndex(0)]
    assert jnp.allclose(val, jnp.sin(xc), atol=1e-6)


def test_order1_matches_gradient_1d():
    """T_1[e_0] should match the gradient of f at xc for a 1-D function."""
    xc = jnp.array([1.0])
    ix = irx.icentpert(xc, 0.3)
    result = mdit(f_exp, 2)(ix, xc)
    T1_val = result[1][MultiIndex(1)]
    grad_val = jax.grad(lambda x: f_exp(x)[0])(xc)
    assert jnp.allclose(T1_val, grad_val, atol=1e-5)


def test_order1_matches_jacobian_2d():
    """T_1[e_i] should match ∂f/∂x_i at xc for a 2-D function."""
    xc = jnp.array([1.0, 0.5])
    ix = irx.icentpert(xc, 0.2)
    result = mdit(f_sin_cos_2d, 2)(ix, xc)
    T1 = result[1]
    J = jax.jacobian(f_sin_cos_2d)(xc)  # shape (1, 2)
    assert jnp.allclose(T1[MultiIndex(1, 0)], J[:, 0], atol=1e-5)
    assert jnp.allclose(T1[MultiIndex(0, 1)], J[:, 1], atol=1e-5)


def test_sin_known_derivatives_at_zero():
    """sin at xc=0: k-th derivative should be 0, 1, 0, -1, 0."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    p = 4
    result = mdit(f_sin, p)(ix, xc)
    expected = [0.0, 1.0, 0.0, -1.0, 0.0]
    for k in range(p + 1):
        val = float(result[k][MultiIndex(k)].squeeze())
        assert abs(val - expected[k]) < 1e-5, (
            f"T_{k}[({k},)] = {val:.6f}, expected {expected[k]}"
        )


def test_exp_all_derivatives_equal_one():
    """exp at xc=0: every order derivative equals 1."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    p = 3
    result = mdit(f_exp, p)(ix, xc)
    for k in range(p + 1):
        val = float(result[k][MultiIndex(k)].squeeze())
        assert abs(val - 1.0) < 1e-5, f"T_{k}[({k},)] = {val:.6f}, expected 1.0"


# ---------------------------------------------------------------------------
# Remainder enclosure:  f(x) ∈ poly(x) + M.contract(x − xc, scale=True)
# ---------------------------------------------------------------------------


def _check_remainder_enclosure(f, result, p, xc, ix, n_samples=200, atol=1e-5):
    """Sample x from ix uniformly and verify the Taylor + MDIT bound contains f(x)."""
    key = jax.random.PRNGKey(42)
    samples = irx.utils.gen_ics(ix, n_samples, key=key)
    poly_tensors = result[:-1]
    M = result[-1]

    for i in range(n_samples):
        x = samples[i]
        dx = x - xc
        poly = irx.taylor_approx(poly_tensors, xc, x)
        rem = M.contract(dx, scale=True)
        rem = irx.interval(rem)
        fint = poly + rem
        fx = f(x)

        assert jnp.all(fx >= fint.lower - atol), (
            f"Sample {i}: f(x)={fx} below {fint.lower=}"
        )
        assert jnp.all(fx <= fint.upper + atol), (
            f"Sample {i}: f(x)={fx} above {fint.upper=}"
        )


@pytest.mark.parametrize("p", [1, 2, 3])
def test_enclosure_sin_1d(p):
    """MDIT remainder bound must enclose sin(x) on a symmetric interval."""
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 0.5)
    result = mdit(f_sin, p)(ix, xc)
    _check_remainder_enclosure(f_sin, result, p, xc, ix)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_enclosure_exp_1d(p):
    """MDIT remainder bound must enclose exp(x) on a shifted interval."""
    xc = jnp.array([1.0])
    ix = irx.icentpert(xc, 0.3)
    result = mdit(f_exp, p)(ix, xc)
    _check_remainder_enclosure(f_exp, result, p, xc, ix)


@pytest.mark.parametrize("p", [2, 3])
def test_enclosure_sin_cos_2d(p):
    """MDIT remainder bound must enclose sin(x₀)cos(x₁) in 2D."""
    xc = jnp.array([0.0, 0.0])
    ix = irx.icentpert(xc, 0.3)
    result = mdit(f_sin_cos_2d, p)(ix, xc)
    # 2D computations accumulate more floating-point error; use a relaxed atol.
    _check_remainder_enclosure(
        f_sin_cos_2d, result, p, xc, ix, n_samples=100, atol=1e-4
    )


def test_enclosure_quadratic_2d():
    """Degree-2 poly with p=2: Taylor poly is exact; remainder must still contain f(x)."""
    xc = jnp.array([1.0, 1.0])
    ix = irx.icentpert(xc, 0.5)
    p = 2
    result = mdit(f_quadratic_2d, p)(ix, xc)
    _check_remainder_enclosure(f_quadratic_2d, result, p, xc, ix, n_samples=100)


def test_cubic_remainder_near_zero():
    """For f(x)=x³ and p=3, the 4th derivative is 0 so M should be near-zero.

    When JAX simplifies the constant-zero derivative, M entries may be stored
    as real arrays rather than Intervals; both representations are accepted.
    """
    xc = jnp.array([0.0])
    ix = irx.icentpert(xc, 1.0)
    p = 3
    M = mdit(f_cubic, p)(ix, xc)[p + 1]
    m4 = M[MultiIndex(4)]
    if isinterval(m4):
        lower = float(m4.lower.squeeze())
        upper = float(m4.upper.squeeze())
    else:
        lower = upper = float(m4.squeeze())
    assert abs(lower) < 1e-4 and abs(upper) < 1e-4, (
        f"Expected M[(4,)] ≈ 0 for x³, got [{lower:.2e}, {upper:.2e}]"
    )


# ---------------------------------------------------------------------------
# taylor_approx consistency
# ---------------------------------------------------------------------------


def test_taylor_approx_agrees_with_manual_contraction():
    """taylor_approx(T_0..T_p, xc, x) must agree with manual per-tensor contraction."""
    xc = jnp.array([0.5])
    ix = irx.icentpert(xc, 0.3)
    p = 3
    result = mdit(f_sin, p)(ix, xc)
    x_test = jnp.array([0.6])
    dx = x_test - xc
    poly_ta = taylor_approx(result[: p + 1], xc, x_test)
    poly_manual = sum(t.contract(dx, scale=True) for t in result[: p + 1])
    assert jnp.allclose(poly_ta, poly_manual, atol=1e-6)
