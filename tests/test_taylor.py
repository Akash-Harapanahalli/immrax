"""Tests for the nattm (Natural Taylor Model Function) interpreter."""

import jax
import jax.numpy as jnp
import pytest

from immrax.inclusion import icentpert, interval
from immrax.taylor import (
    taylor_model_from_function,
    taylor_model_identity,
    nattm,
)
from immrax.taylor.taylor_model import integrate_variable


def tm_hull_contains_samples(f, tm, n_samples=200, atol=1e-5):
    """Check that the TM's interval hull contains sampled function values."""
    hull = tm.interval_hull()
    d = tm.d
    key = jax.random.PRNGKey(0)
    samples = jax.random.uniform(key, (n_samples, d),
                                  minval=tm.domain.lower,
                                  maxval=tm.domain.upper)

    for i in range(n_samples):
        val = f(samples[i])
        tm_val = tm(samples[i])
        assert jnp.all(val >= hull.lower - atol), (
            f"Sample {i}: {val} < hull.lower {hull.lower}"
        )
        assert jnp.all(val <= hull.upper + atol), (
            f"Sample {i}: {val} > hull.upper {hull.upper}"
        )
        assert jnp.all(val >= tm_val.lower - atol), (
            f"Sample {i}: {val} < tm_val.lower {tm_val.lower}"
        )
        assert jnp.all(val <= tm_val.upper + atol), (
            f"Sample {i}: {val} > tm_val.upper {tm_val.upper}"
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
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    tm_f = nattm(f)(tm_x)
    hull = tm_f.interval_hull()
    assert jnp.all(hull.upper >= hull.lower), "Invalid interval hull"
    tm_hull_contains_samples(f, tm_f)


@pytest.mark.parametrize("f,center", [
    pytest.param(jnp.tan, 0.0, id="tan"),
    pytest.param(jnp.arcsin, 0.0, id="arcsin"),
    pytest.param(jnp.arctan, 0.0, id="arctan"),
])
def test_univariate_no_jet_rule(f, center):
    """Primitives without jet rules should raise KeyError (known limitation)."""
    radius = 0.3
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    with pytest.raises(KeyError):
        nattm(f)(tm_x)


# --- Arithmetic ---

def test_add():
    f = lambda x: x[0:1] + x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_sub():
    f = lambda x: x[0:1] - x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_mul():
    f = lambda x: x[0:1] * x[1:2]
    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_neg():
    f = lambda x: -x
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = taylor_model_identity(icentpert(center, radius))
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_scalar_mul():
    f = lambda x: 3.0 * x
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = taylor_model_identity(icentpert(center, radius))
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_integer_pow():
    f = lambda x: x ** 3
    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Structural operations ---

def test_reshape():
    def f(x):
        y = x * 2.0
        return y.reshape(1, -1).reshape(-1)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_broadcast():
    def f(x):
        return x * jnp.ones(2)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_slice():
    def f(x):
        return x[0:1] * 2.0

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_concatenate():
    def f(x):
        a = x[0:1]
        b = x[1:2]
        return jnp.concatenate([a * 2, b + 1])

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


def test_split():
    """jnp.split then recombine with arithmetic."""
    def f(x):
        a, b = jnp.split(x, 2)
        return a * b

    center = jnp.array([1.0, 2.0, 3.0, 4.0])
    radius = jnp.array([0.3, 0.3, 0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (2,)
    tm_hull_contains_samples(f, tm_f)


def test_split_uneven():
    """jnp.split with uneven sizes."""
    def f(x):
        a, b = jnp.split(x, [1])
        return jnp.concatenate([b, a])

    center = jnp.array([1.0, 2.0, 3.0])
    radius = jnp.array([0.3, 0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


# --- Passthrough operations ---

def test_passthrough_copy():
    """Passthrough ops should preserve TM structure."""
    def f(x):
        return x + 0.0  # triggers copy/add

    center = jnp.array([1.0])
    radius = jnp.array([0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
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
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Reductions ---

def test_reduce_sum():
    def f(x):
        return jnp.sum(x, keepdims=True)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Compositions ---

def test_polynomial_composition():
    """A purely polynomial composition should be exact (up to truncation)."""
    def f(x):
        return x ** 2 + 2 * x + 1

    center = jnp.array([1.0])
    radius = jnp.array([0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
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
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    tm_hull_contains_samples(f, tm_f)


# --- Integration ---

def test_integrate_variable_multidimensional():
    """Integrate x0 * x1^2 w.r.t. x0 from 0 and verify soundness.

    The exact integral is  x0^2 * x1^2 / 2.
    We test both keep_order=True (truncates into remainder) and
    keep_order=False (keeps the higher-order polynomial).
    """
    center = jnp.array([0.0, 0.0])
    radius = jnp.array([1.0, 1.0])
    domain = icentpert(center, radius)

    # f(x) = x0 * x1^2
    f = lambda x: jnp.array([x[0] * x[1] ** 2])
    tm = taylor_model_from_function(f, domain, max_order=3)

    # Exact integral: g(x) = x0^2 * x1^2 / 2
    g = lambda x: jnp.array([x[0] ** 2 * x[1] ** 2 / 2.0])

    # --- keep_order=False: polynomial order increases, exact result kept ---
    itm_full = integrate_variable(tm, var_idx=0, start=0.0, keep_order=False)
    assert itm_full.order[0] == tm.order[0] + 1

    # Polynomial should match the exact integral at sampled points
    key = jax.random.PRNGKey(42)
    samples = jax.random.uniform(key, (200, 2), minval=domain.lower, maxval=domain.upper)
    for i in range(200):
        poly_val = itm_full.evaluate_polynomial(samples[i])
        exact_val = g(samples[i])
        assert jnp.allclose(poly_val, exact_val, atol=1e-4), (
            f"keep_order=False mismatch at {samples[i]}: {poly_val} vs {exact_val}"
        )

    # --- keep_order=True: same order, high-degree terms bounded in remainder ---
    itm_reduced = integrate_variable(tm, var_idx=0, start=0.0, keep_order=True)
    assert itm_reduced.order == tm.order

    # The interval hull must still contain the exact integral values (soundness)
    hull = itm_reduced.interval_hull()
    for i in range(200):
        exact_val = g(samples[i])
        assert jnp.all(exact_val >= hull.lower - 1e-5), (
            f"Below hull at {samples[i]}: {exact_val} < {hull.lower}"
        )
        assert jnp.all(exact_val <= hull.upper + 1e-5), (
            f"Above hull at {samples[i]}: {exact_val} > {hull.upper}"
        )

    # --- start=None should use center (here 0.0), same as start=0.0 ---
    itm_none = integrate_variable(tm, var_idx=0, start=None, keep_order=False)
    assert jnp.allclose(itm_none.coeffs, itm_full.coeffs, atol=1e-6)


# --- Scatter operations ---

def test_scatter_set_into_zeros():
    """scatter (at[].set) placing TM values into a zeros array."""
    def f(x):
        y = jnp.zeros(4)
        return y.at[1].set(x[0]).at[3].set(x[1])

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (4,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_set_with_array_indices():
    """scatter (at[].set) using an array of indices."""
    def f(x):
        y = jnp.zeros(5)
        return y.at[jnp.array([0, 2, 4])].set(x)

    center = jnp.array([1.0, 2.0, 3.0])
    radius = jnp.array([0.3, 0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (5,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_add():
    """scatter_add (at[].add) adding TM values to a constant array."""
    def f(x):
        y = jnp.ones(3)
        return y.at[0].add(x[0]).at[2].add(x[1])

    center = jnp.array([0.0, 0.0])
    radius = jnp.array([1.0, 1.0])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_add_duplicate_indices():
    """scatter_add with duplicate indices accumulates contributions."""
    def f(x):
        y = jnp.zeros(2)
        return y.at[jnp.array([0, 0])].add(x)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (2,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_set_2d():
    """scatter (at[].set) on a 2D array — row-level scatter."""
    def f(x):
        y = jnp.zeros((3, 2))
        return y.at[1].set(x)

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3, 2)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_in_composition():
    """scatter inside a larger computation — scatter then transform."""
    def f(x):
        y = jnp.zeros(3)
        y = y.at[0].set(x[0]).at[1].set(x[1]).at[2].set(x[0] + x[1])
        return y ** 2

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_max():
    """scatter_max (at[].max) — element-wise max at scatter indices."""
    def f(x):
        y = jnp.zeros(3)
        return y.at[0].max(x[0]).at[2].max(x[1])

    center = jnp.array([1.0, -1.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


def test_scatter_min():
    """scatter_min (at[].min) — element-wise min at scatter indices."""
    def f(x):
        y = jnp.ones(3)
        return y.at[1].min(x[0]).at[2].min(x[1])

    center = jnp.array([0.5, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


# --- Select operations ---

def test_select_n_where():
    """jnp.where with constant predicate selecting between TM and array."""
    def f(x):
        return jnp.where(jnp.array([True, False]), x, jnp.zeros(2))

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.5, 0.5])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (2,)
    tm_hull_contains_samples(f, tm_f)


def test_select_n_three_way():
    """lax.select_n with 3 cases mixing TM and constant arrays."""
    from jax import lax

    def f(x):
        pred = jnp.array([0, 1, 2])
        return lax.select_n(pred, jnp.zeros(3), x[0] * jnp.ones(3), x[1] * jnp.ones(3))

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=2)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (3,)
    tm_hull_contains_samples(f, tm_f)


def test_select_n_in_composition():
    """select_n inside a larger computation — select then square."""
    def f(x):
        # Use constant predicate to select, then apply polynomial
        selected = jnp.where(jnp.array([True, False]), x, jnp.ones(2))
        return selected ** 2

    center = jnp.array([1.0, 2.0])
    radius = jnp.array([0.3, 0.3])
    tm_x = taylor_model_identity(icentpert(center, radius), order=3)
    tm_f = nattm(f)(tm_x)
    assert tm_f._output_shape == (2,)
    tm_hull_contains_samples(f, tm_f)


# --- Multi-arg ---

def test_multiarg():
    def f(t,x,w) :
        # A time decaying oscillator with unknown forcing w
        return jnp.array([
            jnp.exp(-t) * x[1],
            jnp.exp(-t) * (-x[0]) + w[0]
        ])

    it0 = interval([0.], [1.])
    itx = icentpert([1., 0.], 0.1)
    itw = icentpert([0.], 0.05)

    # center for expansion
    center = [it0.lower, itx.center, itw.center]
    order = [4, 2, 1]

    tm_joint = taylor_model_identity([it0, itx, itw], center, order)

    tm_f = nattm(f)(tm_joint)
    print(tm_f)
