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

    def rownorm(M):
        return M / onp.linalg.norm(M, axis=1, keepdims=True)

    # The step row-normalizes alpha each step (exact set identity), so the
    # pairing alpha_k B^k = alpha_0 holds row-wise up to positive scale.
    B = onp.eye(2) + dt * A_LIN
    Bk = onp.eye(2)
    res_sym = 0.0
    for k in range(N + 1):
        res_sym = max(
            res_sym,
            onp.max(onp.abs(rownorm(alpha[k] @ Bk) - rownorm(onp.asarray(pt0.alpha)))),
        )
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
                    rownorm(alpha_e[k] @ onp.linalg.matrix_power(B, kk))
                    - rownorm(onp.asarray(pt0.alpha))
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
    U0 = 0.3 * onp.asarray(jax.random.normal(jax.random.PRNGKey(3), pt0.alpha.shape))
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


def test_reach_ilqr_symplectic():
    """Symplectic ReachiLQR rollout: full horizon survives, cost decreases."""
    nt0 = irx.L2Normotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
    )
    emb = irx.NormotopeEmbedding(_Linear())
    ilqr = irx.ReachiLQR(
        emb,
        nt0,
        lambda pt: pt.y,
        lambda t, pt, U: 0.0,
        0.0,
        1.0,
        50,
    )  # default solver: symplectic
    Us = ilqr.initial_controls()
    l_traj, K_traj = ilqr.initial_gains()
    res = ilqr.iterate(Us, l_traj, K_traj, 10.0)
    assert int(res.ifinal) == 50
    c0 = float(res.cost_final)
    assert onp.isfinite(c0)
    iover = res.iover
    for _ in range(5):
        res = ilqr.iterate(res.Us_new, res.l_traj, res.K_traj, 10.0, iover=iover)
        iover = res.iover
    assert int(res.ifinal) == 50
    assert onp.isfinite(float(res.cost_final))
    assert float(res.cost_final) < c0


# ---- symplectic partitioned RK (Heun / RK4) ----

from immrax.parametric.symplectic_rk import (  # noqa: E402
    SOLVERS,
    TABLEAUS,
    get_tableau,
    symplectic_partner,
)

RK_SOLVERS = ["symplectic_heun", "symplectic_rk4"]


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_partner_symplecticity(name):
    A, b, _ = get_tableau(name)
    Ah = symplectic_partner(A, b)
    res = b[:, None] * Ah + (b[:, None] * A).T - onp.outer(b, b)
    assert onp.max(onp.abs(res)) < 1e-14


def _rk_map(f, x, h, tab):
    A, b, _ = get_tableau(tab)
    ks = []
    for i in range(len(b)):
        xi = x
        for j in range(i):
            xi = xi + h * A[i, j] * ks[j]
        ks.append(f(xi))
    return x + h * sum(b[i] * ks[i] for i in range(len(b)))


@pytest.mark.parametrize("solver", RK_SOLVERS)
def test_rk_adjoint_pairing_linear(solver):
    """alpha_{k+1} Phi = alpha_k, Phi the RK state map of the linear system."""
    pt0 = _lin_polytope()
    alpha_p0, N0 = _aux0(pt0.alpha)
    emb = irx.AdjointEmbedding(_Linear(), alpha_p0, N0)
    dt, tf = 0.05, 4.0
    N = int(round(tf / dt))
    sol = emb.compute_reachset(0.0, tf, pt0, dt=dt, solver=solver)
    alpha = onp.asarray(sol.ys[0].alpha)
    alpha_p = onp.asarray(sol.ys[1][0])

    Phi = onp.asarray(
        jax.jacfwd(lambda x: _rk_map(lambda z: A_LIN @ z, x, dt, SOLVERS[solver]))(
            jnp.zeros(2)
        )
    )

    def rownorm(M):
        return M / onp.linalg.norm(M, axis=1, keepdims=True)

    Pk = onp.eye(2)
    for k in range(N + 1):
        assert (
            onp.max(onp.abs(rownorm(alpha[k] @ Pk) - rownorm(onp.asarray(pt0.alpha))))
            < 1e-4
        )
        assert onp.max(onp.abs(alpha_p[k] @ alpha[k] - onp.eye(2))) < 1e-4
        Pk = Pk @ Phi


@pytest.mark.parametrize("solver", RK_SOLVERS)
def test_rk_hypercontrol_invariants(solver):
    pt0 = _lin_polytope(extra_rows=True)
    alpha_p0, N0 = _aux0(pt0.alpha)
    emb = irx.AdjointEmbedding(_RevVDP(), alpha_p0, N0)
    U0 = 0.3 * onp.asarray(jax.random.normal(jax.random.PRNGKey(3), pt0.alpha.shape))
    U_static = tuple(map(tuple, U0.tolist()))
    sol = emb.compute_reachset(
        0.0, 1.0, pt0, dt=0.02, solver=solver, f_kwargs=immutabledict({"U": U_static})
    )
    alpha = onp.asarray(sol.ys[0].alpha)
    alpha_p = onp.asarray(sol.ys[1][0])
    Ns = onp.asarray(sol.ys[1][1])
    for k in range(alpha.shape[0]):
        assert onp.max(onp.abs(alpha_p[k] @ alpha[k] - onp.eye(2))) < 1e-4
        assert onp.max(onp.abs(Ns[k] @ alpha[k])) < 1e-4


def _rk_samples(sys, ox, r, dt, N, tab, sub=1):
    key = jax.random.PRNGKey(0)
    samp = ox + r * jax.random.uniform(key, (500, 2), minval=-1, maxval=1)
    samp = jnp.vstack([samp, ox + r * jnp.sign(jax.random.normal(key, (200, 2)))])

    def roll(x):
        def st(c, _):
            for _ in range(sub):
                c = _rk_map(lambda z: sys.f(0.0, z), c, dt / sub, tab)
            return c, c

        _, xs = jax.lax.scan(st, x, None, length=N)
        return jnp.vstack([x, xs])

    return jax.jit(jax.vmap(roll))(samp)


def _poly_violation(ys, mc):
    worst = -jnp.inf
    for k in range(mc.shape[1]):
        a, o, y = ys.alpha[k], ys.ox[k], ys.y[k]
        K = len(y) // 2
        z = (mc[:, k, :] - o) @ a.T
        worst = max(worst, float(jnp.max(-y[:K] - z)), float(jnp.max(z - y[K:])))
    return worst


@pytest.mark.parametrize("solver", RK_SOLVERS)
@pytest.mark.parametrize("kind", ["adjoint", "stacked", "linf", "l2"])
def test_rk_sound_reverse_vdp(solver, kind):
    """Samples of the matching discrete RK map (and, for RK4, the fine-resolved
    flow) stay inside the flowed set."""
    sys = _RevVDP()
    ox, r = jnp.array([1.4, 2.3]), jnp.array([0.08, 0.08])
    box = irx.icentpert(ox, r)
    dt, tf = 0.02, 0.6
    N = int(round(tf / dt))
    mcs = [_rk_samples(sys, ox, r, dt, N, SOLVERS[solver])]
    if solver == "symplectic_rk4":
        mcs.append(_rk_samples(sys, ox, r, dt, N, "rk4", sub=20))

    if kind in ("adjoint", "stacked"):
        if kind == "adjoint":
            pt0 = irx.Polytope.from_interval(box).add_rows(
                jnp.array([[1.0, 1.0], [-1.0, 1.0]]), jnp.eye(2)
            )
            emb = irx.AdjointEmbedding(sys, *_aux0(pt0.alpha))
        else:
            pt0 = irx.Polytope.stacked_from_interval(box)
            emb = irx.parametric.StackedAdjointEmbedding(sys, 2)
        sol = emb.compute_reachset(0.0, tf, pt0, dt=dt, solver=solver)
        for mc in mcs:
            assert _poly_violation(sol.ys[0], mc) <= 1e-5
    else:
        cls = irx.LinfNormotope if kind == "linf" else irx.L2Normotope
        nt0 = cls.from_interval(box)
        emb = irx.NormotopeEmbedding(sys)
        emb._initialize(nt0)
        sol = emb.compute_reachset(0.0, tf, nt0, dt=dt, solver=solver)
        for mc in mcs:
            for k in range(N + 1):
                pk = cls(sol.ys[0].ox[k], sol.ys[0].alpha[k], sol.ys[0].y[k])
                assert float(jnp.max(jax.vmap(pk.g)(mc[:, k, :]) - pk.y)) <= 1e-5


def test_rk_reach_ilqr():
    nt0 = irx.L2Normotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
    )
    emb = irx.NormotopeEmbedding(_Linear())
    ilqr = irx.ReachiLQR(
        emb,
        nt0,
        lambda pt: pt.y,
        lambda t, pt, U: 0.0,
        0.0,
        1.0,
        20,
        solver="symplectic_rk4",
    )
    res = ilqr.iterate(ilqr.initial_controls(), *ilqr.initial_gains(), 10.0)
    assert int(res.ifinal) == 20
    assert onp.isfinite(float(res.cost_final))


class _NoSymplectic(irx.ParametricEmbedding):
    """An embedding with only continuous dynamics."""

    def _initialize(self, pt0):
        return None

    def _dynamics(self, t, state, *args):
        return jax.tree.map(jnp.zeros_like, state)


def test_unsupported_embedding_raises():
    nt0 = irx.L2Normotope.from_interval(irx.icentpert(jnp.zeros(2), 0.1))
    emb = _NoSymplectic(_Linear())
    for solver in ["symplectic", "symplectic_rk4"]:
        with pytest.raises(NotImplementedError, match="tsit5"):
            emb.compute_reachset(0.0, 0.1, nt0, dt=0.01, solver=solver)
    with pytest.raises(NotImplementedError, match="_symplectic_step"):
        irx.ReachiLQR(emb, nt0, lambda pt: pt.y, lambda t, pt, U: 0.0, 0.0, 1.0, 10)
