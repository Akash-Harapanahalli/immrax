from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import immrax as irx
from immrax.parametric.sets.chebyshev_polytope import (
    affine_change,
    cheb_coeffs,
    cheb_eval,
    cheb_indices,
    _grid,
    signed_bound,
)


class VDP(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([x[1], (1 - x[0] ** 2) * x[1] - x[0]])


def test_affine_change_exact():
    mu, M = jnp.array([0.1, -0.2]), jnp.array([[0.5, 0.1], [-0.2, 0.7]])
    S, exps = affine_change(mu, M, 3), cheb_indices(2, 3)
    w = jax.random.uniform(jax.random.PRNGKey(0), (50, 2), minval=-1, maxval=1)
    lhs = jax.vmap(lambda w: cheb_eval(mu + M @ w, exps))(w)
    rhs = jax.vmap(lambda w: S @ cheb_eval(w, exps))(w)
    assert jnp.allclose(lhs, rhs, atol=1e-5)


def test_signed_bound_contains():
    q, G = 6, 4
    p = lambda w: jnp.sin(3 * w[0]) * w[1] ** 2 + w[0] * w[1]
    C = cheb_coeffs(jax.vmap(p)(jnp.asarray(_grid(2, q)))[:, None], 2, q)
    pc, dev = signed_bound(C, G)
    w = onp.random.default_rng(0).uniform(-1, 1, (4000, 2))
    cell = (onp.clip(((w + 1) / 2 * G).astype(int), 0, G - 1) * [G, 1]).sum(1)
    pw = onp.asarray(
        jax.vmap(lambda w: cheb_eval(w, cheb_indices(2, q - 1)))(jnp.asarray(w))
    )
    vals = (
        pw @ onp.asarray(C[tuple(cheb_indices(2, q - 1).T)])[:, 0]
    )  # the interpolant itself
    assert onp.all(
        onp.abs(vals - onp.asarray(pc)[cell, 0]) <= onp.asarray(dev)[cell, 0] + 1e-9
    )


def _reach(bound, m, tf, **kw):
    emb = irx.ChebyshevPolytopeEmbedding(VDP(), bound=bound, **kw)
    ix0 = irx.icentpert(jnp.array([1.4, 2.4]), jnp.array([0.15, 0.05]))
    pt0 = irx.ChebyshevPolytope.from_interval(ix0, m=m)
    emb._initialize(pt0)
    ys = emb.compute_reachset(0.0, tf, pt0, dt=0.1, solver="symplectic_heun").ys[0]
    f = VDP().f
    F = lambda x: x + 0.05 * (f(0, x) + f(0, x + 0.1 * f(0, x)))
    X = ix0.lower + (ix0.upper - ix0.lower) * jax.random.uniform(
        jax.random.PRNGKey(1), (2000, 2)
    )
    worst = -jnp.inf
    for k in range(ys.ox.shape[0]):
        pt = irx.ChebyshevPolytope(ys.ox[k], ys.H[k], ys.alpha[k], ys.lo[k], ys.hi[k])
        worst = max(worst, float(jnp.max(jax.vmap(pt.g)(X))))
        X = jax.vmap(F)(X)
    return emb, pt, worst


def test_cells_sound():
    _, _, worst = _reach("cells", 3, 2.0, G=16)
    assert worst <= 1e-9


def test_sos_sound():
    pytest.importorskip("mosek")
    emb, pt, worst = _reach("sos", 2, 1.0)
    assert worst <= 1e-9
    C = jnp.eye(2)
    assert jnp.all(emb.support(pt, C) >= pt.ox - 1e-9)
