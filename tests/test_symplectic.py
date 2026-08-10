import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from typing import ClassVar
from immutabledict import immutabledict

import immrax as irx
from immrax.parametric.embedding import ReachsetSolution

A_LIN = onp.array([[-0.15, -1.0], [1.0, -0.15]])


class _Linear(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.asarray(A_LIN) @ x


class _RevVDP(irx.System):
    xlen: ClassVar[int] = 2
    mu: float = 1.0

    def f(self, t, x):
        return jnp.array([-x[1], -self.mu * (1 - x[0] ** 2) * x[1] + x[0]])


def _lin_polytope(extra_rows=False):
    box = irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
    pt = irx.Polytope.from_interval(box)
    if extra_rows:
        pt = pt.add_rows(jnp.array([[1.0, 1.0], [-1.0, 1.0]]), jnp.eye(2))
    return pt


def _aux0(alpha):
    alpha = onp.asarray(alpha)
    u = onp.linalg.svd(alpha, full_matrices=True)[0]
    return jnp.asarray(onp.linalg.pinv(alpha)), jnp.asarray(u[:, alpha.shape[1] :].T)


def test_adjoint_pairing_linear():
    """Symplectic alpha-step is the exact discrete adjoint of the Euler state map."""
    pt0 = _lin_polytope()
    alpha_p0, N0 = _aux0(pt0.alpha)
    emb = irx.AdjointEmbedding(_Linear(), alpha_p0, N0)
    dt, tf = 0.01, 4.0
    N = int(round(tf / dt))

    sol = emb.compute_reachset(0.0, tf, pt0, dt=dt)  # default: symplectic
    assert isinstance(sol, ReachsetSolution)
    assert sol.ts.shape == (N + 1,)
    assert int(jnp.isfinite(sol.ts).sum()) == N + 1
    alpha = onp.asarray(sol.ys[0].alpha)
    alpha_p = onp.asarray(sol.ys[1][0])
    assert alpha.shape[0] == N + 1

    B = onp.eye(2) + dt * A_LIN
    Bk = onp.eye(2)
    res_sym = 0.0
    for k in range(N + 1):
        res_sym = max(res_sym, onp.max(onp.abs(alpha[k] @ Bk - onp.asarray(pt0.alpha))))
        assert onp.max(onp.abs(alpha_p[k] @ alpha[k] - onp.eye(2))) < 1e-4
        Bk = Bk @ B

    sole = emb.compute_reachset(0.0, tf, pt0, dt=dt, solver="euler")
    ts_e = onp.asarray(sole.ts)
    nf = int(onp.isfinite(ts_e).sum())
    alpha_e = onp.asarray(sole.ys[0].alpha)
    res_eul = 0.0
    for k in range(nf):
        kk = int(round(ts_e[k] / dt))
        res_eul = max(
            res_eul,
            onp.max(
                onp.abs(
                    alpha_e[k] @ onp.linalg.matrix_power(B, kk) - onp.asarray(pt0.alpha)
                )
            ),
        )

    assert res_sym < 1e-3
    assert res_sym < 0.01 * res_eul


def test_jit_transparent():
    pt0 = _lin_polytope()
    alpha_p0, N0 = _aux0(pt0.alpha)
    emb = irx.AdjointEmbedding(_Linear(), alpha_p0, N0)
    sol = jax.jit(lambda p: emb.compute_reachset(0.0, 0.2, p, dt=0.01))(pt0)
    assert sol.ts.shape == (21,)
    assert bool(jnp.all(jnp.isfinite(sol.ys[0].y)))


def test_normotope_sound_reverse_vdp():
    """Monte-Carlo boundary trajectories stay inside the flowed normotope."""
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    nt0 = irx.L2Normotope.from_interval(irx.icentpert(ox, jnp.array([0.08, 0.08])))
    samp = nt0.sample_boundary(jax.random.PRNGKey(9), 300)
    dt, tf = 0.005, 0.6
    N = int(round(tf / dt))

    @jax.jit
    def roll(x):
        def st(c, _):
            return c + dt * sys.f(0.0, c), c + dt * sys.f(0.0, c)

        _, xs = jax.lax.scan(st, x, None, length=N)
        return jnp.vstack([x, xs])

    mc = onp.asarray(jax.vmap(roll)(samp))
    emb = irx.NormotopeEmbedding(sys)
    emb._initialize(nt0)  # resolve gsc outside jit (existing pattern)
    sol = emb.compute_reachset(0.0, tf, nt0, dt=dt)  # default: symplectic
    worst = -9.0
    for k in range(N + 1):
        pk = irx.L2Normotope(
            jnp.asarray(sol.ys[0].ox)[k],
            jnp.asarray(sol.ys[0].alpha)[k],
            jnp.asarray(sol.ys[0].y)[k],
        )
        v = float(jnp.max(jax.vmap(lambda x: pk.g(x) - pk.y)(jnp.asarray(mc[:, k, :]))))
        worst = max(worst, v)
    assert worst <= 1e-4


def test_hypercontrol_invariants():
    """U != 0 path: alpha_p @ alpha = I and N @ alpha = 0 preserved exactly."""
    pt0 = _lin_polytope(extra_rows=True)
    alpha_p0, N0 = _aux0(pt0.alpha)
    emb = irx.AdjointEmbedding(_Linear(), alpha_p0, N0)
    U0 = 0.3 * onp.asarray(
        jax.random.normal(jax.random.PRNGKey(3), pt0.alpha.shape)
    )
    # f_kwargs is static under filter_jit; pass U as a hashable nested tuple.
    U_static = tuple(map(tuple, U0.tolist()))
    sol = emb.compute_reachset(
        0.0, 1.0, pt0, dt=0.01, f_kwargs=immutabledict({"U": U_static})
    )
    alpha = onp.asarray(sol.ys[0].alpha)
    alpha_p = onp.asarray(sol.ys[1][0])
    Ns = onp.asarray(sol.ys[1][1])
    for k in range(alpha.shape[0]):
        assert onp.max(onp.abs(alpha_p[k] @ alpha[k] - onp.eye(2))) < 1e-4
        assert onp.max(onp.abs(Ns[k] @ alpha[k])) < 1e-4


def _small_pnt():
    from immrax.parametric.sets.polynomial_normotope import degree_sizes

    n, m = 2, 2
    Ns = degree_sizes(n, m)
    k = jax.random.PRNGKey(0)
    H = jnp.concatenate(
        [jnp.eye(n) * 6.0, 0.3 * jax.random.normal(k, (n, sum(Ns) - n))], axis=1
    )
    Bs = (jnp.eye(Ns[1]) * 4.0,)
    return irx.L2PolynomialNormotope(jnp.array([1.0, 2.0]), H, Bs, 1.3)


def test_symplectic_unsupported_raises():
    pnt = _small_pnt()
    emb = irx.PolynomialNormotopeEmbedding(_Linear())
    emb._initialize(pnt)
    with pytest.raises(NotImplementedError, match="tsit5"):
        emb.compute_reachset(0.0, 0.1, pnt, dt=0.01)


def test_reach_ilqr_symplectic():
    """Symplectic ReachiLQR rollout: full horizon survives, cost decreases."""
    nt0 = irx.L2Normotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
    )
    emb = irx.NormotopeEmbedding(_Linear())
    ilqr = irx.ReachiLQR(
        emb, nt0,
        lambda pt: pt.y,
        lambda t, pt, U: 0.0,
        0.0, 1.0, 50,
    )  # default solver: symplectic
    Us = ilqr.initial_controls()
    l_traj, K_traj = ilqr.initial_gains()
    res = ilqr.iterate(Us, l_traj, K_traj, 10.0)
    assert int(res.ifinal) == 50
    c0 = float(res.cost_final)
    assert onp.isfinite(c0)
    iover = res.iover
    for _ in range(5):
        res = ilqr.iterate(
            res.Us_new, res.l_traj, res.K_traj, 10.0, iover=iover
        )
        iover = res.iover
    assert int(res.ifinal) == 50
    assert onp.isfinite(float(res.cost_final))
    assert float(res.cost_final) < c0


def test_reach_ilqr_symplectic_rejects_unsupported():
    pnt = _small_pnt()
    emb = irx.PolynomialNormotopeEmbedding(_Linear())
    with pytest.raises(NotImplementedError, match="_symplectic_step"):
        # default solver is symplectic; unsupported embeddings must fail fast
        irx.ReachiLQR(
            emb, pnt,
            lambda pt: pt.y,
            lambda t, pt, U: 0.0,
            0.0, 1.0, 10,
        )
