"""Rigorous floating-point widening of interval bounds."""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from immrax.inclusion import (
    Interval,
    icentpert,
    interval,
    interval_intersect,
    natif,
    non_rigorous,
    rigorous,
    widen,
)
from immrax.inclusion.interval import _resolve_rigorous


def _widened(iv, lo, hi):
    return bool(jnp.all(iv.lower < lo) and jnp.all(iv.upper > hi))


def _exact(iv, lo, hi):
    return bool(jnp.all(iv.lower == lo) and jnp.all(iv.upper == hi))


def test_widen():
    iv = interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    w1, w2 = widen(iv, 1), widen(iv, 2)
    assert _widened(w1, iv.lower, iv.upper) and _widened(w2, w1.lower, w1.upper)
    assert jnp.all(widen(interval(jnp.array([1.0])), 1).lower < 1.0)
    # differentiates as the identity
    assert jnp.allclose(
        jax.jacfwd(lambda x: widen(interval(x, x + 1), 1).lower)(iv.lower), jnp.eye(2)
    )
    assert jnp.allclose(
        jax.grad(lambda x: jnp.sum(widen(interval(x, x + 0.1), 1).upper))(iv.lower), 1.0
    )


def test_mode_precedence():
    """kwarg > innermost context > global default (False)."""
    assert _resolve_rigorous() is False
    with non_rigorous():
        assert _resolve_rigorous(True) is True
        with rigorous():
            assert _resolve_rigorous() is True
        assert _resolve_rigorous() is False
    with rigorous():
        assert _resolve_rigorous(False) is False
    assert _resolve_rigorous() is False


def test_constructors():
    lo, hi, x = jnp.float32(1.0), jnp.float32(2.0), jnp.float32(3.14)
    assert _exact(interval(lo, hi), lo, hi) and _widened(
        interval(lo, hi, rigorous=True), lo, hi
    )
    assert _exact(interval(x), x, x) and _widened(interval(x, rigorous=True), x, x)
    assert _widened(icentpert(lo, 0.5, rigorous=True), 0.5, 1.5)
    with rigorous():
        assert _widened(interval(lo, hi), lo, hi) and _exact(
            interval(lo, hi, rigorous=False), lo, hi
        )
    iv = Interval(lo, hi)
    assert interval(iv) is iv


def test_structural_ops_exact():
    iv = Interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    with rigorous():
        assert _exact(iv.reshape(2, 1), iv.lower.reshape(2, 1), iv.upper.reshape(2, 1))
        c = interval_intersect(
            [
                Interval(jnp.array([1.0]), jnp.array([3.0])),
                Interval(jnp.array([2.0]), jnp.array([4.0])),
            ]
        )
        assert _exact(c, 2.0, 3.0)
        x = icentpert(jnp.array([1.0, 2.0]), 0.1, rigorous=False)
        assert _exact(natif(lambda a: -a)(x), -x.upper, -x.lower)  # neg is exact


A = onp.random.default_rng(1).normal(size=(5, 8)).astype(onp.float32)
# (name, f) with f written against a numpy-like module ``m``, so the float64
# reference is plain numpy
FNS = [
    ("add", lambda m, a: a + a),
    ("mul", lambda m, a: a * a),
    ("sin", lambda m, a: m.sin(a)),
    ("poly", lambda m, a: a**3 - 2 * a**2 + a),
    ("exp-composed", lambda m, a: m.exp(m.sin(a)) * a),
    ("matmul", lambda m, a: A @ a),
    ("sum", lambda m, a: m.sum(a * a)),
]


@pytest.mark.parametrize("name,fm", FNS, ids=[n for n, _ in FNS])
def test_natif_rigorous(name, fm):
    """Rigorous bounds enclose the plain ones, contain float64 truth, and add only ULP-scale width."""
    f = lambda a: fm(jnp, a)
    x = icentpert(jnp.linspace(0.3, 1.7, 8, dtype=jnp.float32), 0.1)
    plain, rig = natif(f, rigorous=False)(x), natif(f, rigorous=True)(x)
    assert jnp.all(rig.lower <= plain.lower) and jnp.all(rig.upper >= plain.upper)
    assert jnp.any(rig.lower < plain.lower) or jnp.any(rig.upper > plain.upper)
    w = plain.upper - plain.lower
    assert jnp.max((rig.upper - rig.lower - w) / jnp.maximum(w, 1e-30)) < 1e-4
    xs = (
        onp.random.default_rng(0)
        .uniform(onp.asarray(x.lower), onp.asarray(x.upper), (200, 8))
        .astype(onp.float64)
    )
    ys = onp.stack([onp.asarray(fm(onp, x), onp.float64) for x in xs])
    assert onp.all(ys >= onp.asarray(rig.lower, onp.float64)) and onp.all(
        ys <= onp.asarray(rig.upper, onp.float64)
    )


def test_rigorous_differentiable_and_jittable():
    f = natif(jnp.sin, rigorous=True)
    x = jnp.array([1.0, 2.0])
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda c: jnp.sum(f(icentpert(c, 0.1)).upper))(x))
    )
    assert jnp.all(jnp.isfinite(jax.jacfwd(lambda c: f(icentpert(c, 0.1)).lower)(x)))
    out = jax.jit(lambda c: f(icentpert(c, 0.1)))(x)
    assert jnp.all(out.lower < jnp.sin(x)) and jnp.all(out.upper > jnp.sin(x))
