import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from typing import ClassVar

import immrax as irx
from immrax.parametric.sets.polynomial_normotope import (
    duplication_matrix,
    _lift_data,
    _degree_from_shape,
    degree_sizes,
    lift_state,
    lift_operator,
    assemble_alpha,
    block_inverse,
    alpha_inv_top,
)

NM = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2)]


def _p(z, n, m):
    E = _lift_data(n, m)[0]
    return jnp.concatenate([lift_state(z, Ek) for Ek in E])


@pytest.mark.parametrize("n,m", NM)
def test_lift_is_generator(n, m):
    """L(M) p(z) == d/dt p(z) under z_dot = M z (pointwise identity)."""
    key = jax.random.PRNGKey(n * 10 + m)
    M = jax.random.normal(key, (n, n))
    z = jax.random.normal(jax.random.fold_in(key, 1), (n,))
    _, D, Dp = _lift_data(n, m)
    lhs = lift_operator(M, n, m, D, Dp) @ _p(z, n, m)
    rhs = jax.jacobian(lambda zz: _p(zz, n, m))(z) @ (M @ z)
    assert jnp.max(jnp.abs(lhs - rhs)) < 1e-4  # float32 op-order roundoff


@pytest.mark.parametrize("n,m", NM)
def test_duplication_pinv(n, m):
    """D_k^+ D_k = I and D_k^+ = diag(1/c) D_k^T."""
    for k in range(2, m + 1):
        Dk, Dkp = duplication_matrix(n, k)
        Nk = Dk.shape[1]
        assert onp.max(onp.abs(Dkp @ Dk - onp.eye(Nk))) < 1e-10
        assert onp.allclose(Dkp, onp.linalg.pinv(Dk))


@pytest.mark.parametrize("n,m", NM)
def test_degree_roundtrip(n, m):
    P = sum(degree_sizes(n, m))
    assert _degree_from_shape(n, P) == m


def _structured(n, m, seed=0, y=1.3):
    Ns = degree_sizes(n, m)
    P = sum(Ns)
    k = jax.random.PRNGKey(seed)
    a1 = jnp.eye(n) * 6.0 + 0.2 * jax.random.normal(jax.random.fold_in(k, 0), (n, n))
    rest = 0.3 * jax.random.normal(jax.random.fold_in(k, 1), (n, P - n))
    H = jnp.concatenate([a1, rest], axis=1)
    Bs = tuple(
        jnp.eye(Nk) * 4.0 + 0.2 * jax.random.normal(jax.random.fold_in(k, 2 + i), (Nk, Nk))
        for i, Nk in enumerate(Ns[1:])
    )
    ox = jnp.arange(1.0, n + 1.0)
    return irx.L2PolynomialNormotope(ox, H, Bs, y)


@pytest.mark.parametrize("n,m", NM)
def test_block_inverse(n, m):
    pnt = _structured(n, m)
    alpha = assemble_alpha(pnt.H, pnt.Bs, n)
    assert jnp.max(jnp.abs(block_inverse(pnt.H, pnt.Bs, n) - jnp.linalg.inv(alpha))) < 1e-4
    assert jnp.max(jnp.abs(alpha_inv_top(pnt.H, pnt.Bs, n) - jnp.linalg.inv(alpha)[:n])) < 1e-4
    # det = det(alpha_1) * prod det(B_k)
    det = jnp.linalg.det(pnt.alpha_1)
    for B in pnt.Bs:
        det = det * jnp.linalg.det(B)
    assert jnp.abs(det - jnp.linalg.det(alpha)) < 1e-3 * (1 + jnp.abs(det))


@pytest.mark.parametrize("n,m", NM)
def test_g_matches_dense(n, m):
    pnt = _structured(n, m)
    alpha = assemble_alpha(pnt.H, pnt.Bs, n)
    z = 0.05 * jax.random.normal(jax.random.PRNGKey(7), (n,))
    g_dense = jnp.linalg.norm(alpha @ _p(z, n, m))
    assert jnp.abs(pnt.g(pnt.ox + z) - g_dense) < 1e-4


def test_iover_sound_grid():
    """Every grid point inside the (nonconvex) set lies inside iover()."""
    n, m = 2, 2
    pnt = _structured(n, m, seed=4, y=1.0)
    ib = pnt.iover()
    gx = onp.linspace(float(pnt.ox[0]) - 1.0, float(pnt.ox[0]) + 1.0, 400)
    gy = onp.linspace(float(pnt.ox[1]) - 1.0, float(pnt.ox[1]) + 1.0, 400)
    XX, YY = onp.meshgrid(gx, gy)
    pts = jnp.asarray(onp.stack([XX.ravel(), YY.ravel()], 1))
    g = onp.asarray(jax.vmap(pnt.g)(pts))
    inset = onp.asarray(pts)[g <= float(pnt.y)]
    lo, up = onp.asarray(ib.lower), onp.asarray(ib.upper)
    assert inset.shape[0] > 0
    assert (((inset < lo - 1e-9) | (inset > up + 1e-9)).any(axis=1)).sum() == 0


@pytest.mark.parametrize("n,m", NM)
def test_pytree_roundtrip(n, m):
    pnt = _structured(n, m)
    leaves, tree = jax.tree_util.tree_flatten(pnt)
    pnt2 = jax.tree_util.tree_unflatten(tree, leaves)
    assert pnt2.m == m and pnt2.n == n and pnt2.P == sum(degree_sizes(n, m))
    z = 0.03 * jax.random.normal(jax.random.PRNGKey(1), (n,))
    assert jnp.abs(pnt2.g(pnt.ox + z) - pnt.g(pnt.ox + z)) < 1e-12


def test_vec_unvec_roundtrip():
    n, m = 2, 3
    pnt = _structured(n, m)
    pnt2 = irx.L2PolynomialNormotope.unvec(pnt.vec(), n, m)
    assert jnp.allclose(pnt2.H, pnt.H) and jnp.allclose(pnt2.Bs[0], pnt.Bs[0])
    assert jnp.allclose(pnt2.Bs[1], pnt.Bs[1]) and jnp.allclose(pnt2.y, pnt.y)


class _Linear(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([[0.0, 1.0], [-1.0, -0.3]]) @ x


@pytest.mark.parametrize("mode", ["block", "flow"])
def test_linear_exactness(mode):
    """For a linear system the gauge is invariant: y stays constant (c == 0)."""
    pnt = _structured(2, 2, seed=2, y=1.3)
    emb = irx.PolynomialNormotopeEmbedding(_Linear(), inverse_mode=mode)
    emb._initialize(pnt)
    sol = emb.compute_reachset(0.0, 2.0, pnt, dt=0.002, solver="euler")
    y = onp.asarray(sol.ys[0].y)
    nf = int(onp.sum(onp.isfinite(onp.asarray(sol.ts))))
    assert onp.max(onp.abs(y[:nf] - 1.3)) < 5e-3


class _RevVDP(irx.System):
    xlen: ClassVar[int] = 2
    mu: float = 1.0

    def f(self, t, x):
        return jnp.array([-x[1], -self.mu * (1 - x[0] ** 2) * x[1] + x[0]])


@pytest.mark.parametrize("mode", ["block", "flow"])
def test_embedding_sound_reverse_vdp(mode):
    """Monte-Carlo trajectories stay inside the flowed set at every frame."""
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    box = irx.icentpert(ox, jnp.array([0.08, 0.08]))
    nt0 = irx.L2Normotope.from_interval(box)
    a1 = nt0.alpha
    H = jnp.concatenate([a1, 0.2 * float(jnp.linalg.norm(a1)) / jnp.sqrt(3) *
                         jax.random.normal(jax.random.PRNGKey(5), (2, 3))], axis=1)
    B2 = jnp.eye(3) * float(jnp.linalg.norm(a1))
    samp = jnp.asarray(onp.asarray(nt0.sample_boundary(jax.random.PRNGKey(9), 300)))
    pr = irx.L2PolynomialNormotope(ox, H, (B2,), 1.0)
    y0 = float(jnp.max(jax.vmap(pr.g)(samp)))
    pr = irx.L2PolynomialNormotope(ox, H, (B2,), y0)

    dt, tf = 0.005, 0.6

    @jax.jit
    def roll(x):
        def st(c, _):
            return c + dt * sys.f(0.0, c), c + dt * sys.f(0.0, c)
        _, xs = jax.lax.scan(st, x, None, length=int(tf / dt))
        return jnp.vstack([x, xs])

    mc = onp.asarray(jax.vmap(roll)(samp))
    emb = irx.PolynomialNormotopeEmbedding(sys, inverse_mode=mode)
    emb._initialize(pr)
    sol = emb.compute_reachset(0.0, tf, pr, dt=dt, solver="euler")
    ts = onp.asarray(sol.ts)
    nf = int(onp.sum(onp.isfinite(ts)))
    worst = -9.0
    for k in range(nf):
        pk = irx.L2PolynomialNormotope(
            jnp.asarray(sol.ys[0].ox)[k],
            jnp.asarray(sol.ys[0].H)[k],
            (jnp.asarray(sol.ys[0].Bs[0])[k],),
            jnp.asarray(sol.ys[0].y)[k],
        )
        idx = min(int(round(ts[k] / dt)), mc.shape[1] - 1)
        v = float(jnp.max(jax.vmap(lambda x: pk.g(x) - pk.y)(jnp.asarray(mc[:, idx, :]))))
        if onp.isfinite(v):
            worst = max(worst, v)
    assert worst <= 1e-6
