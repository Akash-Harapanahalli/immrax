r"""Chebyshev polytope: the adjoint embedding lifted to a basis.

.. math::
    S = \{x : \underline{y} \le \alpha\, \psi(x) \le \overline{y}\},
    \qquad \psi(x) = b(H(x - \mathring{x})),

a polytope in the lifted coordinates ``psi``. ``b`` stacks the tensor Chebyshev
products of total degree ``<= m`` and ``(ox, H)`` only rescales the lift so the set
sits in the unit box, where Chebyshev is well conditioned: changing it is the exact
change of basis ``alpha <- alpha S``.

With ``d/dt psi = L psi + e`` (``L`` the lifted generator, ``e`` what it leaves
out), the embedding is the polytope adjoint embedding in lifted coordinates,

.. math::
    \dot\alpha = -\alpha L + U, \qquad
    \dot{\overline{y}}_i = \sup_{x \in S,\ \alpha_i \psi = \overline{y}_i} U_i \psi + \alpha_i e

(and likewise for ``ly``). Rows with ``U = 0`` are transported by the lifted
adjoint and absorb only ``e``; rows with ``U = alpha L`` stay fixed. The first
``n`` rows are fixed identity rows on the coordinates (the unit box), and the
gauge ``(ox, H)`` is reset every step so they read ``-1 <= z_i <= 1``.

A step lifts the step map ``F`` exactly (Chebyshev interpolation), transports
the rows by its adjoint, bounds the offsets and the image's box, and re-gauges.
Two bounds are available: ``"cells"`` (a ``G**n`` partition of the box; offsets
move by the residual) and ``"sos"`` (validated SOS over the set, MOSEK; offsets
are the sup/inf of the transported rows).
"""

import functools
from math import comb

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike

from ...inclusion import Interval, i2centpert, interval
from ..embedding import ParametricEmbedding
from ..parametope import Parametope
from ..symplectic_rk import get_tableau


# ---------------------------------------------------------------- basis


@functools.lru_cache(maxsize=None)
def cheb_indices(n: int, m: int) -> onp.ndarray:
    """Multi-indices of total degree ``<= m``, graded (``T_0`` first, then the
    ``n`` linear terms in order)."""

    def rec(n, k):
        if n == 1:
            return [[k]]
        return [[f] + r for f in range(k, -1, -1) for r in rec(n - 1, k - f)]

    return onp.array([e for k in range(m + 1) for e in rec(n, k)], dtype=int)


def _degree(n: int, P: int) -> int:
    m = 0
    while comb(n + m, m) < P:
        m += 1
    if comb(n + m, m) != P:
        raise ValueError(f"{P} is not a Chebyshev basis size for n={n}")
    return m


def cheb_eval(z: Array, exps) -> Array:
    """``b(z)`` via the three-term recurrence."""
    kmax = int(exps.max())
    Ts = [jnp.ones_like(z), z]
    for _ in range(2, kmax + 1):
        Ts.append(2.0 * z * Ts[-1] - Ts[-2])
    Ts = jnp.stack(Ts[: kmax + 1])
    return jnp.prod(Ts[onp.asarray(exps), onp.arange(exps.shape[1])], axis=1)


def _nodes(q: int) -> onp.ndarray:
    return onp.cos(onp.pi * (2 * onp.arange(q) + 1) / (2 * q))


def _dct(q: int) -> onp.ndarray:
    """``coeffs = D @ values-at-nodes``, exact for degree ``< q``."""
    D = (2.0 / q) * onp.cos(onp.arange(q)[:, None] * onp.arccos(_nodes(q))[None, :])
    D[0] /= 2.0
    return D


@functools.lru_cache(maxsize=None)
def _grid(n: int, q: int) -> onp.ndarray:
    mesh = onp.meshgrid(*([_nodes(q)] * n), indexing="ij")
    return onp.stack([g.ravel() for g in mesh], axis=1)


def cheb_coeffs(vals: Array, n: int, q: int) -> Array:
    """Values on ``_grid(n, q)`` ``(q**n, ...)`` -> coefficient tensor ``(q,)*n + (...)``."""
    D = jnp.asarray(_dct(q), dtype=vals.dtype)
    C = vals.reshape((q,) * n + vals.shape[1:])
    for a in range(n):
        C = jnp.moveaxis(jnp.tensordot(D, C, axes=(1, a)), 0, a)
    return C


def _at(C: Array, exps) -> Array:
    return C[tuple(onp.asarray(exps).T)]


def affine_change(mu: Array, M: Array, m: int) -> Array:
    """``S`` with ``b(mu + M z) = S b(z)`` (exact: affine maps preserve degree)."""
    n = mu.shape[0]
    exps = cheb_indices(n, m)
    grid = jnp.asarray(_grid(n, m + 1), dtype=M.dtype)
    vals = jax.vmap(lambda z: cheb_eval(mu + M @ z, exps))(grid)
    return _at(cheb_coeffs(vals, n, m + 1), exps).T


# ---------------------------------------------------------------- cells


@functools.lru_cache(maxsize=None)
def _subdivide(q: int, G: int) -> onp.ndarray:
    """``R[j]``: 1-D coefficients on ``[-1, 1]`` -> coefficients on cell ``j``."""
    e = onp.linspace(-1.0, 1.0, G + 1)
    R = []
    for a, b in zip(e[:-1], e[1:]):
        x = a + (b - a) * (_nodes(q) + 1.0) / 2.0
        R.append(_dct(q) @ onp.cos(onp.arange(q)[None, :] * onp.arccos(x)[:, None]))
    return onp.stack(R)


def cells(C: Array, G: int) -> Array:
    """Coefficient tensor ``(q,)*n + (k,)`` -> per-cell coefficients ``(G**n, q**n, k)``."""
    n, q = C.ndim - 1, C.shape[0]
    R = jnp.asarray(_subdivide(q, G), dtype=C.dtype)
    for a in range(n):
        C = jnp.moveaxis(
            jnp.tensordot(R, C, axes=(2, 2 * a)), (0, 1), (2 * a, 2 * a + 1)
        )
    C = jnp.transpose(C, [*range(0, 2 * n, 2), *range(1, 2 * n, 2), 2 * n])
    return C.reshape(G**n, q**n, -1)


def _radius(c: Array) -> Array:
    """``sum |c_beta|`` over the non-constant coefficients of each cell."""
    return jnp.sum(jnp.abs(c[:, 1:]), 1)


@functools.lru_cache(maxsize=None)
def _cell_tables(q: int, G: int):
    """Per cell ``j`` and degree ``k``: ``T_k`` at the centre, ``max |T_k - T_k(centre)|``,
    and ``max |T_k|`` (exact: endpoints and the interior extrema ``cos(i pi / k)``)."""
    e = onp.linspace(-1.0, 1.0, G + 1)
    T = lambda k, p: onp.cos(k * onp.arccos(onp.clip(p, -1, 1)))
    Tc, D, M = (onp.zeros((G, q)) for _ in range(3))
    for j in range(G):
        a, b = e[j], e[j + 1]
        for k in range(q):
            ext = [onp.cos(i * onp.pi / k) for i in range(k + 1)] if k else []
            pts = [a, b] + [p for p in ext if a <= p <= b]
            Tc[j, k] = T(k, (a + b) / 2)
            D[j, k] = max(abs(T(k, p) - Tc[j, k]) for p in pts)
            M[j, k] = max(abs(T(k, p)) for p in pts)
    return Tc, D, M


def _contract(C: Array, tabs) -> Array:
    """``sum_beta C_beta prod_a tabs[a][j_a, beta_a]`` -> ``(G**n, k)``."""
    for a in range(C.ndim - 1):
        C = jnp.moveaxis(jnp.tensordot(tabs[a], C, axes=(1, a)), 0, a)
    return C.reshape(-1, C.shape[-1])


def signed_bound(C: Array, G: int):
    """Per cell: ``p(centre)`` and ``d`` with ``|p - p(centre)| <= d`` on the cell.
    Separable (never forms per-cell coefficients): telescope the product
    ``prod_a T(w_a) - prod_a T(c_a)`` one axis at a time."""
    n, q = C.ndim - 1, C.shape[0]
    Tc, D, M = (jnp.asarray(t, C.dtype) for t in _cell_tables(q, G))
    A = jnp.abs(C)
    dev = sum(
        _contract(A, [jnp.abs(Tc)] * a + [D] + [M] * (n - a - 1)) for a in range(n)
    )
    return _contract(C, [Tc] * n), dev


# ------------------------------------------------------------ set class


@register_pytree_node_class
class ChebyshevPolytope(Parametope):
    r"""``{x : ly <= alpha psi(x) <= uy}``, ``psi(x) = b(H (x - ox))``, ``y = (-ly, uy)``
    stacked as in :class:`Polytope`. ``alpha`` is ``(rows, P)``; its first ``n``
    rows are the identity on the coordinates ``z = H (x - ox)`` with offsets ``-+1``."""

    def __init__(self, ox: ArrayLike, H: ArrayLike, alpha: ArrayLike, y: ArrayLike):
        self.ox, self.H, self.alpha, self.y = ox, H, alpha, y

    def tree_flatten(self):
        return (self.ox, self.H, self.alpha, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_parametope(cls, pt: Parametope):
        raise TypeError("construct directly as ChebyshevPolytope(ox, H, alpha, y)")

    @property
    def ly(self):
        return -self.y[: len(self.y) // 2]

    @property
    def uy(self):
        return self.y[len(self.y) // 2 :]

    @property
    def iy(self) -> Interval:
        return interval(self.ly, self.uy)

    @property
    def n(self) -> int:
        return int(self.ox.shape[0])

    @property
    def m(self) -> int:
        return _degree(self.n, int(self.alpha.shape[1]))

    def psi(self, x: Array) -> Array:
        return cheb_eval(self.H @ (x - self.ox), cheb_indices(self.n, self.m))

    def g(self, x: Array) -> Array:
        """``<= 0`` iff ``x`` is in the set."""
        u = self.alpha @ self.psi(x)
        return jnp.max(jnp.maximum(u - self.uy, self.ly - u))

    def contains(self, x: Array, **kwargs) -> bool:
        return self.g(x) <= 0.0

    def iover(self) -> Interval:
        """Interval hull of the unit box (the fixed rows)."""
        r = jnp.sum(jnp.abs(jnp.linalg.inv(self.H)), 1)
        return interval(self.ox - r, self.ox + r)

    @classmethod
    def from_interval(cls, itvl, m: int = 2) -> "ChebyshevPolytope":
        """The box itself: fixed rows on the coordinates, and every non-constant
        basis function as a transported row, all in ``[-1, 1]``."""
        cent, pert = i2centpert(interval(itvl))
        n = cent.shape[0]
        I = jnp.eye(cheb_indices(n, m).shape[0], dtype=pert.dtype)
        alpha = jnp.concatenate((I[1 : n + 1], I[1:]))
        return cls(
            cent, jnp.diag(1.0 / pert), alpha, jnp.ones(2 * alpha.shape[0], pert.dtype)
        )


# ------------------------------------------------------------ embedding


class ChebyshevPolytopeEmbedding(ParametricEmbedding):
    r"""Lifted adjoint embedding of a :class:`ChebyshevPolytope` (see the module
    docstring), discretized by the explicit RK map of ``f`` that ``solver`` names
    (``"symplectic"``: Euler, ``"symplectic_heun"``: Heun).

    ``f`` must be polynomial; its degree is probed in :meth:`_initialize` (call once
    outside ``jit``) unless given as ``deg_f``. ``G``: cells per axis
    (``bound="cells"``); ``Gfit``: samples per axis of the set-projected lift
    (``bound="sos"``).
    """

    bound: str = eqx.field(static=True, default="cells")
    deg_f: int = eqx.field(static=True, default=None)
    G: int = eqx.field(static=True, default=32)
    Gfit: int = eqx.field(static=True, default=16)

    def _initialize(self, pt0: ChebyshevPolytope) -> ArrayLike:
        if not isinstance(pt0, ChebyshevPolytope):
            raise ValueError(f"{pt0=} is not a ChebyshevPolytope")
        if self.bound not in ("cells", "sos"):
            raise ValueError(f"bound must be 'cells' or 'sos', got {self.bound!r}")
        if self.deg_f is None:
            object.__setattr__(self, "deg_f", _probe_degree(self.sys.f, pt0.n))
        return None

    def _dynamics(self, t, state, *args):
        raise NotImplementedError(
            f"{type(self).__name__} is a discrete-step method; use solver='symplectic' "
            "or 'symplectic_heun'."
        )

    def _step_map(self, t, dt, tableau):
        """The explicit RK map of ``tableau`` and its polynomial degree."""
        if self.sys.evolution == "discrete":
            return (lambda x: self.sys.f(t, x)), self.deg_f
        A, b, c = get_tableau(tableau)
        deg = []
        for i in range(len(b)):  # stage i evaluates f at x + dt sum_j A_ij k_j
            deg.append(self.deg_f * max([1] + [deg[j] for j in range(i) if A[i, j]]))

        def F(x):
            ks = []
            for i in range(len(b)):
                xi = x + dt * sum(A[i, j] * ks[j] for j in range(i) if A[i, j])
                ks.append(self.sys.f(t + c[i] * dt, xi))
            return x + dt * sum(bi * k for bi, k in zip(b, ks) if bi)

        return F, max(deg)

    def _symplectic_step(self, t, dt, state, *args, tableau="euler", **kwargs):
        pt, aux = state
        return self.step(*self._step_map(t, dt, tableau), pt), aux

    def step(self, F, dF: int, pt: ChebyshevPolytope) -> ChebyshevPolytope:
        """Lift ``F``, transport the rows by its adjoint, bound offsets and the
        image's box (``self.bound``), re-gauge."""
        n, m = pt.n, pt.m
        exps = cheb_indices(n, m)
        q = max(m * dF, 2 * m) + 1  # interpolation exact up to the degree of b(phi)
        Hi, Fo = jnp.linalg.inv(pt.H), F(pt.ox)
        phi = lambda z: pt.H @ (F(pt.ox + Hi @ z) - Fo)  # the step in z coordinates
        grid = jnp.asarray(_grid(n, q), dtype=pt.alpha.dtype)
        Bphi = jax.vmap(lambda z: cheb_eval(phi(z), exps))(grid)
        Cphi = cheb_coeffs(Bphi, n, q)
        P1 = Cphi[..., 1 : n + 1]  # coefficients of phi itself
        J = _at(P1, exps[1 : n + 1]).T  # linear part of phi
        U = jnp.linalg.svd(Hi @ J)[0]  # principal axes of the image
        Cp = P1 @ (U.T @ Hi).T  # image coordinates p = U^T (F(x) - F(ox))
        bounds = self._cells if self.bound == "cells" else self._sos
        aT, ly, uy, pl, pu = bounds(pt, phi, Bphi, Cphi, Cp, q, dF)

        # re-gauge: x = Fo + U (c + diag(d) z'); transported rows rewritten exactly,
        # fixed rows reset to the identity in [-1, 1]
        c, d = (pl + pu) / 2.0, jnp.maximum((pu - pl) / 2.0, 1e-12)
        aT = aT @ affine_change(pt.H @ (U @ c), pt.H @ (U * d), m)
        alpha = jnp.concatenate((jnp.eye(exps.shape[0], dtype=aT.dtype)[1 : n + 1], aT))
        one = jnp.ones(n, ly.dtype)
        y = jnp.concatenate((one, -ly, one, uy))
        return ChebyshevPolytope(Fo + U @ c, jnp.linalg.inv(U * d), alpha, y)

    def _cells(self, pt, phi, Bphi, Cphi, Cp, q, dF):
        """Box-projected lift; offsets move by the residual, bounded over the cells
        that can meet the set."""
        n, m, G = pt.n, pt.m, self.G
        exps = cheb_indices(n, m)
        aT = pt.alpha[n:]
        an = jnp.linalg.solve(_at(Cphi, exps), aT.T).T  # an K = alpha
        B = jax.vmap(lambda z: cheb_eval(z, exps))(
            jnp.asarray(_grid(n, q), dtype=aT.dtype)
        )

        # near cells: every row's range on the cell meets [ly, uy]
        cu = cells(cheb_coeffs(B @ pt.alpha.T, n, q)[(slice(0, m + 1),) * n], G)
        ru, tol = _radius(cu), 1e-9 * (jnp.abs(pt.uy) + jnp.abs(pt.ly))
        near = jnp.all(
            (cu[:, 0] - ru <= pt.uy + tol) & (cu[:, 0] + ru >= pt.ly - tol), 1
        )
        sup = lambda v: jnp.max(jnp.where(near[:, None], v, -jnp.inf), 0)

        rc, rd = signed_bound(cheb_coeffs(Bphi @ an.T - B @ aT.T, n, q), G)
        ly, uy = pt.ly[n:] - sup(rd - rc), pt.uy[n:] + sup(rc + rd)
        cp = cells(Cp[(slice(0, min(dF + 1, q)),) * n], G)
        rp = _radius(cp)
        return an, ly, uy, -sup(rp - cp[:, 0]), sup(cp[:, 0] + rp)

    def _sos(self, pt, phi, Bphi, Cphi, Cp, q, dF):
        """Set-projected lift; offsets and image box are SOS sup/inf over the set."""
        n, m, D = pt.n, pt.m, 2 * pt.m
        exps = cheb_indices(n, m)
        P, aT = exps.shape[0], pt.alpha[n:]

        # lift: least squares of b(phi) onto span b over samples of the set
        zf = jnp.asarray(_fit_grid(n, self.Gfit), dtype=aT.dtype)
        Bf = jax.vmap(lambda z: cheb_eval(z, exps))(zf)
        Bpf = jax.vmap(lambda z: cheb_eval(phi(z), exps))(zf)
        uf = Bf @ pt.alpha.T
        wts = jnp.all((uf >= pt.ly) & (uf <= pt.uy), 1).astype(Bf.dtype)[:, None]
        lam = 1e-6 * jnp.maximum(jnp.sum(wts), 1.0)
        K = jnp.linalg.solve(
            (Bf * wts).T @ Bf + lam * jnp.eye(P),
            (Bf * wts).T @ Bpf + lam * _at(Cphi, exps),
        ).T
        an = jnp.linalg.solve(K.T, aT.T).T

        # objectives: transported rows and image coordinates; degree <= D to SOS,
        # the rest bounded on the box
        Cobj = jnp.concatenate([cheb_coeffs(Bphi @ an.T, n, q), Cp], -1)
        tot = onp.indices((q,) * n).sum(0)
        high = jnp.where((tot > D)[..., None], Cobj, 0.0)
        tail = jnp.abs(high).reshape(-1, Cobj.shape[-1]).sum(0)
        head = _at(Cobj, cheb_indices(n, D))
        up, dn = _sos_bounds(aT, pt.ly[n:], pt.uy[n:], head, n, m)
        hi, lo = up + tail, -dn - tail
        k = aT.shape[0]
        return an, lo[:k], hi[:k], lo[k:], hi[k:]

    def support(self, pt: ChebyshevPolytope, c: Array) -> Array:
        """Validated SOS bound on ``sup_x c_k . x`` over the set, for each row ``c_k``."""
        n, m = pt.n, pt.m
        head = jnp.zeros((cheb_indices(n, 2 * m).shape[0], c.shape[0]), pt.alpha.dtype)
        Hi = jnp.linalg.inv(pt.H)
        head = (
            head.at[0].set(c @ pt.ox).at[1 : n + 1].set((c @ Hi).T)
        )  # c.x = c.ox + c H^-1 z
        return _sos_bounds(pt.alpha[n:], pt.ly[n:], pt.uy[n:], head, n, m)[0]


@functools.lru_cache(maxsize=None)
def _fit_grid(n: int, G: int) -> onp.ndarray:
    g = onp.linspace(-1.0, 1.0, G)
    return onp.stack([a.ravel() for a in onp.meshgrid(*([g] * n), indexing="ij")], 1)


def _sos_bounds(alpha, ly, uy, head, n, m):
    """Validated ``(sup p_k, sup -p_k)`` over the set for the columns ``p_k`` of ``head``."""
    from ._chebyshev_sos import solver

    Q = head.shape[1]
    out = jax.ShapeDtypeStruct((2, Q), head.dtype)
    cb = lambda a, ly, uy, p: onp.asarray(
        solver(n, m, a.shape[0]).solve(a, ly, uy, p), dtype=p.dtype
    )
    up_dn = jax.pure_callback(cb, out, alpha, ly, uy, head)
    return up_dn[0], up_dn[1]


def _probe_degree(f, n: int, kmax: int = 12) -> int:
    """Degree of ``x -> f(0, x)`` along random lines (Chebyshev interpolation)."""
    rng = onp.random.default_rng(0)
    q = kmax + 2
    s = _nodes(q)
    deg = 0
    for _ in range(3):
        x0, v = rng.normal(size=n), rng.normal(size=n)
        vals = onp.stack([onp.asarray(f(0.0, jnp.asarray(x0 + si * v))) for si in s])
        c = onp.abs(_dct(q) @ vals).max(1)
        tol = 100 * onp.finfo(vals.dtype).eps * max(c.max(), 1.0)
        nz = onp.nonzero(c > tol)[0]
        deg = max(deg, int(nz.max()) if nz.size else 0)
    if deg > kmax:
        raise ValueError(
            f"f appears non-polynomial; pass deg_f to {ChebyshevPolytopeEmbedding.__name__}."
        )
    return max(deg, 1)
