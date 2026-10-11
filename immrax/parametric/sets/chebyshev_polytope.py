r"""Chebyshev polytope

.. math::
    \{x : \|w\|_\infty \le 1,\ \ \ell_i \le \alpha_i \cdot b(w) \le u_i\},
    \qquad w = H (x - \mathring{x}),

a polytope in the lifted coordinates ``b(w)`` intersected with the chart box.
``b`` stacks the tensor Chebyshev products :math:`T_\beta(w)` of total degree
``<= m`` and ``alpha`` is square and invertible.

One step under a polynomial map ``F`` with ``phi(w) = H (F(ox + H^{-1} w) - F(ox))``:

1. lift: ``K`` = the projection of ``b(phi)`` onto ``span b``;
2. adjoint: ``alpha+ = alpha K^{-1}``, so the faces move with the dynamics;
3. offsets: bound the transported faces over the set;
4. chart: a box on the principal axes of the image, bounded over the set.

Two bounds are available. ``"cells"`` bounds over a ``G**n`` partition of the
chart and updates offsets by the residual ``alpha+ b(phi) - alpha b`` (small, so
the cover's slack is small). ``"sos"`` bounds over the set itself with validated
degree-``2m`` SOS certificates (MOSEK) and sets offsets to the sup/inf of the
transported face directly. Polynomials are computed exactly by interpolation at
Chebyshev nodes.
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


def cheb_eval(w: Array, exps) -> Array:
    """``b(w)`` via the three-term recurrence."""
    kmax = int(exps.max())
    Ts = [jnp.ones_like(w), w]
    for _ in range(2, kmax + 1):
        Ts.append(2.0 * w * Ts[-1] - Ts[-2])
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
    """``S`` with ``b(mu + M w) = S b(w)`` (exact: affine maps preserve degree)."""
    n = mu.shape[0]
    exps = cheb_indices(n, m)
    grid = jnp.asarray(_grid(n, m + 1), dtype=M.dtype)
    vals = jax.vmap(lambda w: cheb_eval(mu + M @ w, exps))(grid)
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
    r"""``{x : ||w||_inf <= 1, lo <= alpha b(w) <= hi}``, ``w = H (x - ox)``."""

    def __init__(
        self,
        ox: ArrayLike,
        H: ArrayLike,
        alpha: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
    ):
        self.ox, self.H, self.alpha, self.lo, self.hi = ox, H, alpha, lo, hi

    def tree_flatten(self):
        return (self.ox, self.H, self.alpha, self.lo, self.hi), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_parametope(cls, pt: Parametope):
        raise TypeError("construct directly as ChebyshevPolytope(ox, H, alpha, lo, hi)")

    @property
    def n(self) -> int:
        return int(self.ox.shape[0])

    @property
    def m(self) -> int:
        return _degree(self.n, int(self.alpha.shape[0]))

    def g(self, x: Array) -> Array:
        """``<= 0`` iff ``x`` is in the set."""
        w = self.H @ (x - self.ox)
        u = self.alpha @ cheb_eval(w, cheb_indices(self.n, self.m))
        return jnp.maximum(
            jnp.max(jnp.maximum(u - self.hi, self.lo - u)), jnp.max(jnp.abs(w)) - 1.0
        )

    def contains(self, x: Array, **kwargs) -> bool:
        return self.g(x) <= 0.0

    def iover(self) -> Interval:
        """Interval hull of the chart box."""
        r = jnp.sum(jnp.abs(jnp.linalg.inv(self.H)), 1)
        return interval(self.ox - r, self.ox + r)

    @classmethod
    def from_interval(cls, itvl, m: int = 2) -> "ChebyshevPolytope":
        """The box itself: chart = the box, faces ``T_beta(w) in [-1, 1]``."""
        cent, pert = i2centpert(interval(itvl))
        P = cheb_indices(cent.shape[0], m).shape[0]
        one = jnp.ones(P, pert.dtype)
        return cls(cent, jnp.diag(1.0 / pert), jnp.eye(P, dtype=pert.dtype), -one, one)


# ------------------------------------------------------------ embedding


class ChebyshevPolytopeEmbedding(ParametricEmbedding):
    r"""Step a :class:`ChebyshevPolytope` through the explicit RK map of ``f`` chosen by
    ``solver`` (``"symplectic"``: Euler, ``"symplectic_heun"``: Heun). See the module
    docstring for the step and the two bounds.

    ``f`` must be polynomial; its degree is probed in :meth:`_initialize` (call once
    outside ``jit``) unless given as ``deg_f``. ``G`` is the number of cells per axis
    (``bound="cells"``); ``Gfit`` the samples per axis of the set-projected lift
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
        F, dF = self._step_map(t, dt, tableau)
        step = self.step_cells if self.bound == "cells" else self.step_sos
        return step(F, dF, pt), aux

    def step_cells(self, F, dF: int, pt: ChebyshevPolytope) -> ChebyshevPolytope:
        """Box-projected lift; offsets += residual bound over the near cells."""
        n, m = pt.n, pt.m
        exps, q = cheb_indices(n, m), m * dF + 1
        grid = jnp.asarray(_grid(n, q), dtype=pt.alpha.dtype)
        Hi, Fo = jnp.linalg.inv(pt.H), F(pt.ox)
        phi = lambda w: pt.H @ (F(pt.ox + Hi @ w) - Fo)
        B = jax.vmap(lambda w: cheb_eval(w, exps))(grid)
        Bphi = jax.vmap(lambda w: cheb_eval(phi(w), exps))(grid)
        Cphi = cheb_coeffs(Bphi, n, q)
        an = jnp.linalg.solve(_at(Cphi, exps), pt.alpha.T).T  # an K = alpha
        u = B @ pt.alpha.T

        # cells where every face can hold
        cu = cells(cheb_coeffs(u, n, q)[(slice(0, m + 1),) * n], self.G)
        ru, tol = _radius(cu), 1e-9 * (jnp.abs(pt.hi) + jnp.abs(pt.lo))
        near = jnp.all(
            (cu[:, 0] - ru <= pt.hi + tol) & (cu[:, 0] + ru >= pt.lo - tol), 1
        )
        sup = lambda v: jnp.max(jnp.where(near[:, None], v, -jnp.inf), 0)

        rc, rd = signed_bound(cheb_coeffs(Bphi @ an.T - u, n, q), self.G)
        hi, lo = pt.hi + sup(rc + rd), pt.lo - sup(rd - rc)

        # chart: principal axes of the linear part, box over the near cells
        P1 = Cphi[..., 1 : n + 1]  # coefficients of phi
        U = jnp.linalg.svd(Hi @ _at(P1, exps[1 : n + 1]).T)[0]
        cz = cells(P1[(slice(0, dF + 1),) * n] @ (U.T @ Hi).T, self.G)
        rz = _radius(cz)
        zlo = jnp.min(jnp.where(near[:, None], cz[:, 0] - rz, jnp.inf), 0)
        zhi = jnp.max(jnp.where(near[:, None], cz[:, 0] + rz, -jnp.inf), 0)
        return _rechart(pt, Fo, U, an, lo, hi, zlo, zhi)

    def step_sos(self, F, dF: int, pt: ChebyshevPolytope) -> ChebyshevPolytope:
        """Set-projected lift; offsets = SOS sup/inf of the transported faces."""
        n, m = pt.n, pt.m
        D = 2 * m
        exps, q = cheb_indices(n, m), max(m * dF, D) + 1
        P = exps.shape[0]
        grid = jnp.asarray(_grid(n, q), dtype=pt.alpha.dtype)
        Hi, Fo = jnp.linalg.inv(pt.H), F(pt.ox)
        phi = lambda w: pt.H @ (F(pt.ox + Hi @ w) - Fo)
        Bphi = jax.vmap(lambda w: cheb_eval(phi(w), exps))(grid)
        Cphi = cheb_coeffs(Bphi, n, q)

        # lift: least squares of b(phi) onto span b over samples of the set
        wf = jnp.asarray(_fit_grid(n, self.Gfit), dtype=pt.alpha.dtype)
        Bf = jax.vmap(lambda w: cheb_eval(w, exps))(wf)
        Bpf = jax.vmap(lambda w: cheb_eval(phi(w), exps))(wf)
        uf = Bf @ pt.alpha.T
        wts = jnp.all((uf >= pt.lo) & (uf <= pt.hi), 1).astype(Bf.dtype)[:, None]
        lam = 1e-6 * jnp.maximum(jnp.sum(wts), 1.0)
        K = jnp.linalg.solve(
            (Bf * wts).T @ Bf + lam * jnp.eye(P),
            (Bf * wts).T @ Bpf + lam * _at(Cphi, exps),
        ).T
        an = jnp.linalg.solve(K.T, pt.alpha.T).T

        # objectives: transported faces and chart axes; degree <= D to SOS, rest bounded on the box
        P1 = Cphi[..., 1 : n + 1]  # coefficients of phi
        U = jnp.linalg.svd(Hi @ _at(P1, exps[1 : n + 1]).T)[0]
        Cobj = jnp.concatenate([cheb_coeffs(Bphi @ an.T, n, q), P1 @ (U.T @ Hi).T], -1)
        tot = onp.indices((q,) * n).sum(0)
        tail = jnp.sum(
            jnp.abs(jnp.where((tot > D)[..., None], Cobj, 0.0)).reshape(-1, P + n), 0
        )
        head = _at(Cobj, cheb_indices(n, D))
        up, dn = _sos_bounds(pt.alpha, pt.lo, pt.hi, head, n, m)
        hi, lo = up[:P] + tail[:P], -dn[:P] - tail[:P]
        zhi, zlo = up[P:] + tail[P:], -dn[P:] - tail[P:]
        return _rechart(pt, Fo, U, an, lo, hi, zlo, zhi)

    def support(self, pt: ChebyshevPolytope, c: Array) -> Array:
        """Validated SOS bound on ``sup_x c_k . x`` over the set, for each row ``c_k``."""
        n, m = pt.n, pt.m
        exps = cheb_indices(n, 2 * m)
        Hi = jnp.linalg.inv(pt.H)
        head = jnp.zeros((exps.shape[0], c.shape[0]), pt.alpha.dtype)
        head = (
            head.at[0].set(c @ pt.ox).at[1 : n + 1].set((c @ Hi).T)
        )  # c.x = c.ox + c H^-1 w
        return _sos_bounds(pt.alpha, pt.lo, pt.hi, head, n, m)[0]


def _rechart(pt, Fo, U, an, lo, hi, zlo, zhi) -> ChebyshevPolytope:
    """New chart ``x = Fo + U (c + diag(d) w)``; faces rewritten exactly in it.
    The constant face stays exactly ``1 in [-1, 1]``."""
    c, d = (zlo + zhi) / 2.0, jnp.maximum((zhi - zlo) / 2.0, 1e-12)
    an = an @ affine_change(pt.H @ (U @ c), pt.H @ (U * d), pt.m)
    e0 = jnp.zeros(an.shape[0], an.dtype).at[0].set(1.0)
    an, lo, hi = an.at[0].set(e0), lo.at[0].set(-1.0), hi.at[0].set(1.0)
    return ChebyshevPolytope(Fo + U @ c, jnp.linalg.inv(U * d), an, lo, hi)


@functools.lru_cache(maxsize=None)
def _fit_grid(n: int, G: int) -> onp.ndarray:
    g = onp.linspace(-1.0, 1.0, G)
    return onp.stack([a.ravel() for a in onp.meshgrid(*([g] * n), indexing="ij")], 1)


def _sos_bounds(alpha, lo, hi, head, n, m):
    """Validated ``(sup p_k, sup -p_k)`` over the set for the columns ``p_k`` of ``head``."""
    from ._chebyshev_sos import solver

    Q = head.shape[1]
    out = jax.ShapeDtypeStruct((2, Q), head.dtype)
    cb = lambda a, l, h, p: onp.asarray(solver(n, m).solve(a, l, h, p), dtype=p.dtype)
    up_dn = jax.pure_callback(cb, out, alpha, lo, hi, head)
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
