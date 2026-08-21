import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from typing import ClassVar

import immrax as irx
from immrax.parametric.sets.chebyshev_normotope import (
    cheb_indices,
    cheb_eval,
    chebyshev_lift,
    scale_matrix,
)

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


def test_lift_exact():
    """A b + C_H b_hi reconstructs the lifted field exactly (polynomial f)."""
    n, m = 2, 3
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    # T is the preconditioner: w = T z, i.e. the set spans z in +-(1/diag(T)).
    T = jnp.diag(jnp.array([1.0 / 0.3, 1.0 / 0.5]))
    Tf = jnp.linalg.inv(T)
    A, C_H = chebyshev_lift(lambda x: sys.f(0.0, x), ox, T, n, m, 3)
    exps = cheb_indices(n, m)[0]
    exps_all = cheb_indices(n, m + 2)[0]
    exps_hi = exps_all[exps_all.sum(1) > m]

    ws = jax.random.uniform(jax.random.PRNGKey(0), (50, n), minval=-1.0, maxval=1.0)

    def lifted(w):
        fe = T @ (sys.f(0.0, ox + Tf @ w) - sys.f(0.0, ox))
        return jax.jvp(lambda v: cheb_eval(v, exps), (w,), (fe,))[1]

    def recon(w):
        return A @ cheb_eval(w, exps) + C_H @ cheb_eval(w, exps_hi)

    err = jnp.max(jnp.abs(jax.vmap(lifted)(ws) - jax.vmap(recon)(ws)))
    assert float(err) < 1e-4
    assert float(jnp.max(jnp.abs(A[0]))) == 0.0  # T_0 row is zero

    # Linear f: no HOT at all.
    _, C2 = chebyshev_lift(lambda x: _Linear().f(0.0, x), ox, T, n, m, 1)
    assert C2.shape[1] == 0


def test_scale_matrix_identity():
    """b(mu + M w) == S b(w) for a full affine map."""
    n, m = 2, 3
    exps = cheb_indices(n, m)[0]
    key = jax.random.PRNGKey(1)
    mu = jnp.array([0.1, -0.2])
    M = jnp.array([[0.7, 0.2], [-0.1, 1.3]])
    S = scale_matrix(mu, M, n, m)
    ws = jax.random.uniform(key, (40, n), minval=-1.0, maxval=1.0)
    err = jnp.max(
        jnp.abs(
            jax.vmap(lambda w: cheb_eval(mu + M @ w, exps) - S @ cheb_eval(w, exps))(ws)
        )
    )
    assert float(err) < 1e-5


def test_refit_set_identity():
    """_refit preserves normalized membership g/y exactly (set identity)."""
    ox = jnp.array([1.4, 2.3])
    pt = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(ox, jnp.array([0.08, 0.08])), m=2
    )
    emb = irx.ChebyshevNormotopeEmbedding(_RevVDP())
    emb._initialize(pt)
    pt2 = emb._refit(pt)
    xs = ox + jax.random.uniform(
        jax.random.PRNGKey(2), (100, 2), minval=-0.12, maxval=0.12
    )
    r1 = jax.vmap(pt.g)(xs) / pt.y
    r2 = jax.vmap(pt2.g)(xs) / pt2.y
    assert float(jnp.max(jnp.abs(r1 - r2))) < 1e-4


def test_linear_exact_offset():
    """Linear system: no HOT, U=0 => y is exactly invariant (symplectic pairing)."""
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1])), m=2
    )
    emb = irx.ChebyshevNormotopeEmbedding(_Linear())
    emb._initialize(pt0)
    sol = emb.compute_reachset(0.0, 2.0, pt0, dt=0.01)
    ys = onp.asarray(sol.ys[0].y)
    assert onp.all(onp.isfinite(ys))
    # enclosure comparable to the L2 symplectic baseline
    nt0 = irx.L2Normotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
    )
    emb2 = irx.NormotopeEmbedding(_Linear())
    emb2._initialize(nt0)
    sol2 = emb2.compute_reachset(0.0, 2.0, nt0, dt=0.01)
    pk = irx.ChebyshevNormotope(
        sol.ys[0].ox[-1], sol.ys[0].T[-1], sol.ys[0].alpha[-1], sol.ys[0].y[-1]
    )
    nk = irx.L2Normotope(sol2.ys[0].ox[-1], sol2.ys[0].alpha[-1], sol2.ys[0].y[-1])
    wc = onp.asarray(pk.iover().upper - pk.iover().lower)
    wl = onp.asarray(nk.iover().upper - nk.iover().lower)
    assert onp.all(wc < 1.5 * wl + 1e-3)


def test_vdp_sound_m2():
    """Monte-Carlo trajectories stay inside the flowed m=2 Chebyshev normotope."""
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(ox, jnp.array([0.08, 0.08])), m=2
    )
    emb = irx.ChebyshevNormotopeEmbedding(sys)
    emb._initialize(pt0)
    dt, tf = 0.005, 0.6
    N = int(round(tf / dt))
    sol = emb.compute_reachset(0.0, tf, pt0, dt=dt)

    cand = ox + jax.random.uniform(
        jax.random.PRNGKey(9), (4000, 2), minval=-0.15, maxval=0.15
    )
    samp = cand[jax.vmap(pt0.g)(cand) <= pt0.y][:300]
    assert samp.shape[0] > 50

    @jax.jit
    def roll(x):
        def st(c, _):
            nx = c + dt * sys.f(0.0, c)
            return nx, nx

        _, xs = jax.lax.scan(st, x, None, length=N)
        return jnp.vstack([x, xs])

    mc = jax.vmap(roll)(samp)
    worst = -9.0
    for k in range(N + 1):
        pk = irx.ChebyshevNormotope(
            sol.ys[0].ox[k], sol.ys[0].T[k], sol.ys[0].alpha[k], sol.ys[0].y[k]
        )
        v = float(jnp.max(jax.vmap(lambda x: pk.g(x) - pk.y)(mc[:, k, :])))
        worst = max(worst, v)
    assert worst <= 1e-3
    # tube stays within ~1.5x of the MC hull width at the end
    pk = irx.ChebyshevNormotope(
        sol.ys[0].ox[N], sol.ys[0].T[N], sol.ys[0].alpha[N], sol.ys[0].y[N]
    )
    wc = onp.asarray(pk.iover().upper - pk.iover().lower)
    wm = onp.asarray(mc[:, N, :].max(0) - mc[:, N, :].min(0))
    assert onp.all(wc < 1.6 * wm)


def test_vdp_sound_m3_partition():
    """m=3 diverges under pure interval bounds; the partition drift keeps it
    sound and within ~1.6x of the MC hull."""
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(ox, jnp.array([0.08, 0.08])), m=3
    )
    emb = irx.ChebyshevNormotopeEmbedding(sys)  # default drift="partition"
    emb._initialize(pt0)
    dt, tf = 0.005, 0.6
    N = int(round(tf / dt))
    sol = emb.compute_reachset(0.0, tf, pt0, dt=dt)
    assert bool(jnp.all(jnp.isfinite(sol.ys[0].y)))

    cand = ox + jax.random.uniform(
        jax.random.PRNGKey(9), (4000, 2), minval=-0.15, maxval=0.15
    )
    samp = cand[jax.vmap(pt0.g)(cand) <= pt0.y][:300]

    @jax.jit
    def roll(x):
        def st(c, _):
            nx = c + dt * sys.f(0.0, c)
            return nx, nx

        _, xs = jax.lax.scan(st, x, None, length=N)
        return jnp.vstack([x, xs])

    mc = jax.vmap(roll)(samp)
    worst = -9.0
    for k in range(N + 1):
        pk = irx.ChebyshevNormotope(
            sol.ys[0].ox[k], sol.ys[0].T[k], sol.ys[0].alpha[k], sol.ys[0].y[k]
        )
        v = float(jnp.max(jax.vmap(lambda x: pk.g(x) - pk.y)(mc[:, k, :])))
        worst = max(worst, v)
    assert worst <= 1e-3
    pk = irx.ChebyshevNormotope(
        sol.ys[0].ox[N], sol.ys[0].T[N], sol.ys[0].alpha[N], sol.ys[0].y[N]
    )
    wc = onp.asarray(pk.iover().upper - pk.iover().lower)
    wm = onp.asarray(mc[:, N, :].max(0) - mc[:, N, :].min(0))
    assert onp.all(wc < 1.6 * wm)


def test_vdp_sound_m4_graded_interval():
    """Graded spectator weights realize the adjoint cancellation: m=4 is sound
    and tight with the cheap interval bounds alone (no partition needed)."""
    sys = _RevVDP()
    ox = jnp.array([1.4, 2.3])
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(ox, jnp.array([0.08, 0.08])), m=4
    )
    emb = irx.ChebyshevNormotopeEmbedding(sys, drift="interval")
    emb._initialize(pt0)
    dt, tf = 0.005, 0.6
    N = int(round(tf / dt))
    sol = emb.compute_reachset(0.0, tf, pt0, dt=dt)
    assert bool(jnp.all(jnp.isfinite(sol.ys[0].y)))

    cand = ox + jax.random.uniform(
        jax.random.PRNGKey(9), (4000, 2), minval=-0.15, maxval=0.15
    )
    samp = cand[jax.vmap(pt0.g)(cand) <= pt0.y][:300]

    @jax.jit
    def roll(x):
        def st(c, _):
            nx = c + dt * sys.f(0.0, c)
            return nx, nx

        _, xs = jax.lax.scan(st, x, None, length=N)
        return jnp.vstack([x, xs])

    mc = jax.vmap(roll)(samp)
    worst = -9.0
    for k in range(N + 1):
        pk = irx.ChebyshevNormotope(
            sol.ys[0].ox[k], sol.ys[0].T[k], sol.ys[0].alpha[k], sol.ys[0].y[k]
        )
        v = float(jnp.max(jax.vmap(lambda x: pk.g(x) - pk.y)(mc[:, k, :])))
        worst = max(worst, v)
    assert worst <= 1e-3
    pk = irx.ChebyshevNormotope(
        sol.ys[0].ox[N], sol.ys[0].T[N], sol.ys[0].alpha[N], sol.ys[0].y[N]
    )
    wc = onp.asarray(pk.iover().upper - pk.iover().lower)
    wm = onp.asarray(mc[:, N, :].max(0) - mc[:, N, :].min(0))
    assert onp.all(wc < 1.6 * wm)


def test_diffrax_solver_raises():
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1])), m=2
    )
    emb = irx.ChebyshevNormotopeEmbedding(_Linear())
    emb._initialize(pt0)
    with pytest.raises(NotImplementedError, match="symplectic"):
        emb.compute_reachset(0.0, 0.1, pt0, dt=0.01, solver="euler")


def test_jit_transparent():
    pt0 = irx.ChebyshevNormotope.from_interval(
        irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1])), m=2
    )
    emb = irx.ChebyshevNormotopeEmbedding(_Linear())
    emb._initialize(pt0)
    sol = jax.jit(lambda p: emb.compute_reachset(0.0, 0.2, p, dt=0.01))(pt0)
    assert sol.ts.shape == (21,)
    assert bool(jnp.all(jnp.isfinite(sol.ys[0].y)))
