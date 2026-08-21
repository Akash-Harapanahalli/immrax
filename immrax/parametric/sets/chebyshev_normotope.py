r"""Chebyshev polynomial normotope: the nonconvex set

.. math::
    \{\ox + z : \|\alpha\, b(T z)\|_2^2 \le y\},

where ``b(w)`` stacks the multivariate Chebyshev products
:math:`T_\beta(w) = \prod_i T_{\beta_i}(w_i)` of total degree :math:`\le m`
(**including** the constant :math:`T_0`). The shaping map is the composite
``alpha . b . T``: the ``n x n`` preconditioner ``T`` maps the deviation
``z`` into scaled coordinates ``w = T z``, and is maintained by discrete
refitting so the set always lives in ``w \in [-1,1]^n`` with a unit-ball
linear marginal (well-conditioned basis).

Why Chebyshev: every basis term satisfies :math:`|T_\beta(w)| \le 1` on the
fitted box, so interval bounds on the offset drift are coefficient abs-sums --
degree-independent and well conditioned, unlike monomial interval analysis.
The level ``y`` is the **squared** norm (unlike the other normotopes), which
removes the ``1/(2y)`` singularity from the radius dynamics.

The embedding integrates the Chebyshev--Carleman adjoint symplectically
(``alpha_{k+1}(I + dt A_k) = alpha_k + dt U_k``) with the frame frozen within a
step; refitting is an exact discrete change of basis between steps. See
:class:`ChebyshevNormotopeEmbedding`.
"""

import functools
from math import comb

import numpy as onp
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Float

from taylax import FullTensorPolynomial, pjet

from ...inclusion import Interval, interval, natif
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from .polynomial_normotope import monomial_exponents


# ----------------------------------------------------------------- basis tables

@functools.lru_cache(maxsize=None)
def cheb_indices(n: int, m: int):
    """``(exps, lin_idx)``: total-degree ``<= m`` multi-indices incl. the constant.

    ``exps`` is ``(|b|, n)`` graded (degree 0, 1, ..., m); ``lin_idx`` gives the
    rows of the pure linear terms ``T_1(w_i)`` (so ``w_i = b(w)[lin_idx[i]]``).
    """
    exps = onp.concatenate([monomial_exponents(n, k) for k in range(m + 1)], axis=0)
    eye = onp.eye(n, dtype=int)
    lin_idx = onp.array(
        [int(onp.where((exps == eye[i]).all(1))[0][0]) for i in range(n)]
    )
    return exps, lin_idx


def _cheb_size(n: int, m: int) -> int:
    """``|b| = C(n+m, m)`` (total degree ``<= m`` incl. constant)."""
    return comb(n + m, m)


def _degree_from_size(n: int, P: int) -> int:
    m = 0
    while _cheb_size(n, m) < P:
        m += 1
    if _cheb_size(n, m) != P:
        raise ValueError(f"|b| = {P} is not a valid Chebyshev lift size for n={n}")
    return m


def cheb_eval(w: Array, exps) -> Array:
    """``b(w)``: products of ``T_k`` via the recurrence (differentiable)."""
    if exps.shape[0] == 0:
        return jnp.zeros(0, dtype=w.dtype)
    kmax = int(exps.max())
    Ts = [jnp.ones_like(w), w]
    for _ in range(2, kmax + 1):
        Ts.append(2.0 * w * Ts[-1] - Ts[-2])
    T = jnp.stack(Ts[: kmax + 1])  # (kmax+1, n)
    return jnp.prod(T[onp.asarray(exps), onp.arange(exps.shape[1])[None, :]], axis=1)


# ------------------------------------------------------- interpolation machinery

def _dct_matrix(q: int) -> onp.ndarray:
    """``D`` with ``c = D g``: Chebyshev coefficients from values at the
    Chebyshev--Gauss nodes ``x_j = cos(pi (2j+1) / (2q))`` (exact for deg < q)."""
    j = onp.arange(q)
    k = onp.arange(q)[:, None]
    D = (2.0 / q) * onp.cos(k * onp.pi * (2 * j + 1) / (2 * q))
    D[0] /= 2.0
    return D


@functools.lru_cache(maxsize=None)
def _grid_data(n: int, q: int):
    """Tensor Chebyshev--Gauss grid ``(q**n, n)`` and the per-axis DCT matrix."""
    nodes = onp.cos(onp.pi * (2 * onp.arange(q) + 1) / (2 * q))
    mesh = onp.meshgrid(*([nodes] * n), indexing="ij")
    grid = onp.stack([g.ravel() for g in mesh], axis=1)
    return grid, _dct_matrix(q)


def _tensor_coeffs(vals: Array, D: Array, n: int, q: int) -> Array:
    """Per-axis DCT of ``vals`` (``(q**n, P)``) -> flat coefficients ``(q**n, P)``."""
    P = vals.shape[-1]
    C = vals.reshape((q,) * n + (P,))
    for a in range(n):
        C = jnp.moveaxis(jnp.tensordot(D, C, axes=(1, a)), 0, a)
    return C.reshape(q**n, P)


@functools.lru_cache(maxsize=None)
def _lift_static(n: int, m: int, deg_f: int):
    """Static data for the Chebyshev--Carleman lift at ``(n, m, deg_f)``.

    The lifted field ``Db(w) T^{-1} f_err(T w)`` has total (and per-axis) degree
    ``<= m + deg_f - 1``, so ``q = m + deg_f`` Gauss nodes per axis interpolate
    it exactly. Rows split into ``|beta| <= m`` (the lift ``A``) and
    ``m < |beta| <= m + deg_f - 1`` (the HOT coefficients ``C_H``).
    """
    q = m + max(deg_f, 1)
    grid, D = _grid_data(n, q)
    exps_low = cheb_indices(n, m)[0]
    exps_all = cheb_indices(n, max(m + deg_f - 1, m))[0]
    exps_hi = exps_all[exps_all.sum(1) > m]
    strides = q ** onp.arange(n - 1, -1, -1)
    return q, grid, D, exps_low @ strides, exps_hi @ strides


def chebyshev_lift(f, ox: Array, T: Array, n: int, m: int, deg_f: int):
    """``(A, C_H)``: Chebyshev coefficients of the lifted error field at ``(ox, T)``.

    With ``w = T z``, ``d/dt b(w) = A b(w) + C_H b_hi(w)`` exactly for
    polynomial ``f`` of degree ``<= deg_f``; ``b_hi`` are the total-degree
    ``m+1 .. m+deg_f-1`` terms (each in ``[-1,1]`` on the box).
    """
    q, grid, D, idx_low, idx_hi = _lift_static(n, m, deg_f)
    exps = cheb_indices(n, m)[0]
    grid = jnp.asarray(grid, dtype=ox.dtype)
    D = jnp.asarray(D, dtype=ox.dtype)
    fox = f(ox)
    Tf = jnp.linalg.inv(T)  # z = T^{-1} w on the grid

    def lifted_field(w):
        fe = T @ (f(ox + Tf @ w) - fox)  # w_dot = T f_err
        return jax.jvp(lambda v: cheb_eval(v, exps), (w,), (fe,))[1]

    vals = jax.vmap(lifted_field)(grid)  # (q**n, |b|)
    flat = _tensor_coeffs(vals, D, n, q)
    return flat[onp.asarray(idx_low)].T, flat[onp.asarray(idx_hi)].T


def scale_matrix(mu: Array, M: Array, n: int, m: int) -> Array:
    """``S`` with ``b(mu + M w) = S b(w)`` (exact: composition with an affine map
    preserves total degree, and total degree ``<= m`` implies per-axis degree
    ``<= m``, resolved by the ``m+1``-point grid). ``mu, M`` may be traced."""
    q = m + 1
    grid, D = _grid_data(n, q)
    exps = cheb_indices(n, m)[0]
    strides = q ** onp.arange(n - 1, -1, -1)
    idx = onp.asarray(exps @ strides)
    grid = jnp.asarray(grid, dtype=M.dtype)
    D = jnp.asarray(D, dtype=M.dtype)
    vals = jax.vmap(lambda w: cheb_eval(mu + M @ w, exps))(grid)  # (q**n, |b|)
    return _tensor_coeffs(vals, D, n, q)[idx].T


def _monomial_cheb_coeffs(p: int) -> onp.ndarray:
    """``c`` with ``w**p = sum_k c[k] T_k(w)`` (exact; via ``w T_k = (T_{k+1}+T_{|k-1|})/2``)."""
    c = onp.zeros(p + 1)
    c[0] = 1.0
    for step in range(p):
        nc = onp.zeros(p + 1)
        for k in range(step + 1):
            v = c[k]
            if v == 0.0:
                continue
            if k == 0:
                nc[1] += v
            else:
                nc[k + 1] += v / 2
                nc[k - 1] += v / 2
        c = nc
    return c


def _even_axis_bound(alpha: Array, y, n: int, m: int) -> Array:
    """Per-axis bound ``|w_i| <= s_i`` recovered through the pure ``T_2(w_i)``
    slots: on the variety ``w_i^2 = (v + 1)/2`` with ``v`` the slot value, and
    the lifted ellipsoid on the ``b_0 = 1`` slice bounds ``v`` from above. Sound
    for any ``alpha``; only informative when those slots carry real weight
    (e.g. the ``shape_deg >= 2`` initializations). Requires ``m >= 2``."""
    exps = cheb_indices(n, m)[0]
    eye = onp.eye(n, dtype=int)
    idx2 = onp.array(
        [int(onp.where((exps == 2 * eye[i]).all(1))[0][0]) for i in range(n)]
    )
    ainv = jnp.linalg.inv(alpha)
    a = ainv[0, :]
    na2 = jnp.dot(a, a)
    srad = jnp.sqrt(jnp.maximum(y - 1.0 / na2, 0.0))
    C = ainv[idx2, :]
    cen = C @ a / na2
    rad = jnp.linalg.norm(C - jnp.outer(cen, a), axis=1) * srad
    return jnp.sqrt(jnp.maximum((1.0 + cen + rad) / 2.0, 0.0))


def _slice_frame(alpha: Array, y, n: int, m: int):
    """Exact linear marginal of the lifted ellipsoid on the ``v_0 = 1`` slice.

    With ``u = alpha v``, ``v_0 = a^T u`` and ``w_i = c_i^T u`` for rows
    ``a, c_i`` of ``alpha^{-1}``; on ``{||u||^2 <= y, a^T u = 1}`` the linear
    coordinates trace the ellipsoid ``{mu + Cp xi : ||xi|| <= s}`` with
    ``mu = C a/||a||^2``, ``Cp = C - mu a^T``, ``s^2 = y - 1/||a||^2``.
    Returns ``(mu, E)`` with shape matrix ``E = s^2 Cp Cp^T`` (``n x n`` PSD).
    """
    ainv = jnp.linalg.inv(alpha)
    lin_idx = onp.asarray(cheb_indices(n, m)[1])
    a = ainv[0, :]
    C = ainv[lin_idx, :]
    na2 = jnp.dot(a, a)
    mu = C @ a / na2
    Cp = C - jnp.outer(mu, a)
    s2 = jnp.maximum(y - 1.0 / na2, 0.0)
    return mu, s2 * (Cp @ Cp.T)


def _cheb_eval_safe(w, E, n: int):
    """``b(w)`` via scalar recurrences and Python loops -- natif-safe (no gather)."""
    if E.shape[0] == 0:
        return jnp.zeros(0)
    kmax = int(E.max())
    one = jnp.ones(())
    tabs = []
    for i in range(n):
        ts = [one, w[i]]
        for _ in range(2, kmax + 1):
            ts.append(2.0 * w[i] * ts[-1] - ts[-2])
        tabs.append(ts)
    cols = []
    for r in range(E.shape[0]):
        t = one
        for i in range(n):
            e = int(E[r, i])
            if e:
                t = t * tabs[i][e]
        cols.append(t)
    return jnp.stack(cols)


def _partition_level_bound(alpha, W, Hc, y, n: int, m: int, deg_f: int, G: int):
    r"""Boundary-partition bound of the drift scalar on the level set (the
    :func:`~immrax.parametric.sets.gram_normotope._level_set_drift` pattern).

    ``E >= sup_{V = y} N`` with ``V = ||alpha b||^2`` and the telescoped
    increment ``N = 2 (alpha b)^T (W b + Hc b_hi)``, bounded over a static
    ``G**n`` grid of the fitted box ``[-1,1]^n``: only cells whose ``natif(V)``
    straddles ``y`` contribute (they cover the boundary); each contributes the
    Lagrangian-projected centered mean-value upper bound (``where``-masked, so
    shapes are static -> JIT/vmap/autodiff clean).
    """
    exps = cheb_indices(n, m)[0]
    exps_all = cheb_indices(n, max(m + deg_f - 1, m))[0]
    exps_hi = exps_all[exps_all.sum(1) > m]

    def Vfun(w):
        return jnp.sum((alpha @ _cheb_eval_safe(w, exps, n)) ** 2)

    def Nfun(w):
        b = _cheb_eval_safe(w, exps, n)
        return 2.0 * jnp.dot(alpha @ b, W @ b + Hc @ _cheb_eval_safe(w, exps_hi, n))

    nV = natif(Vfun)
    nJN = natif(jax.jacfwd(Nfun))
    nJV = natif(jax.jacfwd(Vfun))
    gN = jax.grad(Nfun)
    gV = jax.grad(Vfun)

    hw = 1.0 / G
    axes = onp.linspace(-1.0 + hw, 1.0 - hw, G)
    WC = onp.stack(
        [g.ravel() for g in onp.meshgrid(*([axes] * n), indexing="ij")], axis=1
    )

    def cell(wc):
        bx = interval(wc - hw, wc + hw)
        Vb = nV(bx)
        keep = (Vb.lower <= y) & (y <= Vb.upper)
        gNc = gN(wc)
        gVc = gV(wc)
        lam = jnp.dot(gNc, gVc) / (jnp.dot(gVc, gVc) + 1e-30)
        gc = Nfun(wc) - lam * (Vfun(wc) - y)  # = N(wc) on the level set
        JN = nJN(bx)
        JV = nJV(bx)
        lo = jnp.minimum(lam * JV.lower, lam * JV.upper)
        hi = jnp.maximum(lam * JV.lower, lam * JV.upper)
        ub = gc + jnp.sum(
            jnp.maximum(jnp.abs(JN.lower - hi), jnp.abs(JN.upper - lo)) * hw
        )
        return jnp.where(keep, ub, -jnp.inf)

    return jnp.max(jax.vmap(cell)(jnp.asarray(WC, dtype=alpha.dtype)))


@functools.lru_cache(maxsize=None)
def _scalar_static(n: int, m: int, deg_f: int):
    """Grid/DCT data for interpolating the drift scalar (degree ``<= 2m + deg_f - 1``)."""
    q = 2 * m + max(deg_f, 1)
    grid, D = _grid_data(n, q)
    exps_low = cheb_indices(n, m)[0]
    exps_all = cheb_indices(n, max(m + deg_f - 1, m))[0]
    return q, grid, D, exps_low, exps_all[exps_all.sum(1) > m]


def _scalar_boxbound(alpha, W, Hc, n: int, m: int, deg_f: int):
    """Box bound of ``s(w) = 2 (alpha b)^T (W b + Hc b_hi)`` by Chebyshev
    interpolation + coefficient abs-sum (constant term kept exact). Near-tight
    for the box range since all product cancellations happen in the coefficients."""
    q, grid, D, exps, exps_hi = _scalar_static(n, m, deg_f)
    grid = jnp.asarray(grid, dtype=alpha.dtype)
    D = jnp.asarray(D, dtype=alpha.dtype)

    def sfun(w):
        b = cheb_eval(w, exps)
        return 2.0 * jnp.dot(alpha @ b, W @ b + Hc @ cheb_eval(w, exps_hi))

    c = _tensor_coeffs(jax.vmap(sfun)(grid)[:, None], D, n, q)[:, 0]
    return c[0] + jnp.sum(jnp.abs(c)) - jnp.abs(c[0])


def _isqrtm_psd(E: Array, tiny: float = 1e-12) -> Array:
    """Symmetric PSD inverse square root via ``eigh`` (eigenvalues clamped at ``tiny``)."""
    lam, V = jnp.linalg.eigh((E + E.T) / 2.0)
    return (V / jnp.sqrt(jnp.maximum(lam, tiny))) @ V.T


# ------------------------------------------------------------------- set class

@register_pytree_node_class
class ChebyshevNormotope(Parametope):
    r"""Chebyshev-basis polynomial normotope ``{ox + z : ||alpha b(T z)||_2^2 <= y}``.

    Children are ``(ox, T, alpha, y)``: the shaping map is the composite
    ``alpha . b . T``, where the ``n x n`` preconditioner ``T`` maps the
    deviation ``z`` into scaled coordinates ``w = T z`` (analogous to the
    shaping matrix ``H`` of a linear normotope). ``y`` is the **squared** level
    (``g`` returns the squared norm, so ``contains`` still reads ``g <= y``).
    ``alpha`` is dense ``(|b|, |b|)`` and must be invertible for :meth:`iover`.
    """

    def __init__(self, ox: ArrayLike, T: ArrayLike, alpha: ArrayLike, y: Float):
        self.ox = ox
        self.T = T
        self.alpha = alpha
        self.y = y

    def tree_flatten(self):
        return ((self.ox, self.T, self.alpha, self.y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_parametope(cls, pt: Parametope):
        raise TypeError(
            "ChebyshevNormotope carries a preconditioner T; construct it "
            "directly as ChebyshevNormotope(ox, T, alpha, y)."
        )

    # -- structure --
    @property
    def n(self) -> int:
        return int(self.ox.shape[0])

    @property
    def P(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def m(self) -> int:
        return _degree_from_size(self.n, self.P)

    # -- geometry --
    def b(self, w: Array) -> Array:
        return cheb_eval(w, cheb_indices(self.n, self.m)[0])

    def g(self, x: Array) -> Float:
        """The **squared** level ``||alpha b(T (x - ox))||_2^2``."""
        return jnp.sum((self.alpha @ self.b(self.T @ (x - self.ox))) ** 2)

    def iover(self) -> Interval:
        """Support box of the linear marginal ellipsoid on the ``b_0 = 1`` slice,
        intersected (for ``m >= 2``) with the box recovered through the pure
        ``T_2(w_i)`` slots -- the route that sees ``shape_deg >= 2`` tightness."""
        mu, E = _slice_frame(self.alpha, self.y, self.n, self.m)
        Tf = jnp.linalg.inv(self.T)
        r = jnp.sqrt(jnp.maximum(jnp.diag(Tf @ E @ Tf.T), 0.0))
        lo = self.ox + Tf @ mu - r
        hi = self.ox + Tf @ mu + r
        if self.m >= 2:
            s = _even_axis_bound(self.alpha, self.y, self.n, self.m)
            re = jnp.abs(Tf) @ s  # w-box [-s, s] mapped to a z-box around ox
            lo = jnp.maximum(lo, self.ox - re)
            hi = jnp.minimum(hi, self.ox + re)
            hi = jnp.maximum(hi, lo)
        return interval(lo, hi)

    @classmethod
    def from_interval(
        cls, itvl, m: int = 1, eps: float = 0.3, gamma: float = 0.2,
        shape_deg: int = 1,
    ) -> "ChebyshevNormotope":
        """Lift an interval's linear data to a degree-``m`` set.

        The box determines only the linear rows of ``alpha`` (its circumscribed
        ellipsoid: the unit ball in ``w = T z``, ``T = diag(1/pert)/sqrt(n)``).
        The higher-degree rows are not determined by the set; they are seeded at
        ``eps * gamma**(deg-1)`` (compensated in ``y`` -- an ``O(eps^2)``
        inflation) so that ``alpha`` is invertible. The degree-scaled seeding
        restores the ``L^k`` grading that the unit-box normalization strips
        from the basis: the truncation residual reaching degree-``k`` rows is
        ``O(L^{m+1-k})``, so seeding them at ``O(gamma^{k-1})`` prices the
        truncation error at ``gamma^{2(m-1)}`` and raising ``m`` behaves like
        raising a Taylor-model order (at ``cond(alpha) ~ gamma^{-(m-1)}``).
        Equal seeding (``gamma=1``) prices the top-degree error at ``O(1)`` and
        makes higher ``m`` worse. Choose ``gamma`` near the effective expansion
        parameter (~ set scale x field curvature).

        ``shape_deg = p > 1`` spends degree on initial tightness instead: active
        rows ``w_i^p`` give the ``2p``-norm ball ``{sum w_i^{2p} <= n}``, whose
        per-axis inflation over the box is ``n^(1/2p)`` instead of ``sqrt(n)``
        (n=2: 1.19x at p=2 vs 1.41x). Drift-free iff ``p <= m - deg_f + 1``
        (those rows' dynamics stay inside the carried basis).

        .. warning:: ``shape_deg > 1`` is currently STATIC-USE ONLY (tight
           initial sets, ROA-style certificates). Flowing it diverges: the
           per-step exact-fit refit needs an extent estimator with zero slack
           at its fixed point, and every recovery route for even-carried
           extent has percent-level slack; since a frame rescale by ``s``
           multiplies degree-``k`` rows by ``s**k``, that slack pumps the
           row-scale ratios geometrically and the gauge runs away. Needs a
           deadband refit (tolerate ``set inside (1+delta) box``, rescale only
           on threshold crossings) before it can be flowed.
        """
        from ...inclusion import i2centpert
        from ...inclusion import interval as _interval

        cent, pert = i2centpert(_interval(itvl))
        n = cent.shape[0]
        exps, lin_idx = cheb_indices(n, m)
        deg = exps.sum(1)
        p = int(shape_deg)
        if not 1 <= p <= m:
            raise ValueError(f"shape_deg={p} must be in [1, {m}]")
        d = eps * gamma ** onp.maximum(deg - 1, 0).astype(float)
        if p == 1:
            # circumscribed ellipsoid: unit ball in w, active linear rows
            d[onp.asarray(lin_idx)] = 1.0
            T = jnp.diag(1.0 / pert) / jnp.sqrt(float(n))
            y = 1.0 + float(onp.sum(d**2)) - float(n)
            return cls(cent, T, jnp.diag(jnp.asarray(d, dtype=pert.dtype)), jnp.asarray(y))
        # 2p-norm ball {sum_i w_i^{2p} <= n}: per-axis inflation n^(1/2p) instead
        # of sqrt(n). Active rows w_i^p live at degree p; drift-free whenever
        # p <= m - deg_f + 1 (their dynamics stay inside the carried basis).
        A0 = onp.diag(d)
        ccoef = _monomial_cheb_coeffs(p)
        eye = onp.eye(n, dtype=int)
        seed_sq = float(onp.sum(d**2))
        for i in range(n):
            row = onp.zeros(exps.shape[0])
            for k in range(p + 1):
                j = int(onp.where((exps == k * eye[i]).all(1))[0][0])
                row[j] = ccoef[k]
            jtop = int(onp.where((exps == p * eye[i]).all(1))[0][0])
            seed_sq -= float(d[jtop] ** 2)
            A0[jtop, :] = row
        T = jnp.diag(1.0 / pert)  # box faces at |w_i| = 1
        y = float(n) + seed_sq  # V(corner) <= n + sum of seed contributions
        return cls(cent, T, jnp.asarray(A0, dtype=pert.dtype), jnp.asarray(y))


# ------------------------------------------------------------------- embedding

class ChebyshevNormotopeEmbedding(ParametricEmbedding):
    r"""Symplectic-only embedding onto :class:`ChebyshevNormotope` dynamics.

    Per step (frame frozen within the step; see :meth:`_symplectic_step`):

    .. math::
        \alpha_{k+1}(I + h A_k) &= \alpha_k + h U_k, \\
        \ox_{k+1} &= \ox_k + h f(t_k, \ox_k),

    with ``A_k`` the Chebyshev--Carleman lift at ``(ox_k, T_k)``. Along the
    lifted Euler step ``b_+ = (I + h A) b + h H`` the implicit adjoint
    telescopes exactly -- ``alpha_{k+1} b_+ = alpha_k b + h (U b + alpha_{k+1}
    C_H b_hi)`` -- so only the hypercontrol and the genuine HOT drive the
    offset: ``y_{k+1}`` bounds ``||alpha_k b + h G||^2`` by the min of a
    Cauchy--Schwarz bound (``||alpha b|| <= sqrt(y)``) and a bilinear abs-sum
    bound (``|T_beta| <= 1``). Each step starts with an exact discrete refit
    mapping the set's linear marginal ellipsoid to the unit ball (recenter +
    full affine frame change, so the basis tracks the set's orientation) plus a
    norm renormalization -- both set identities -- keeping the basis
    well-conditioned.

    Exact for polynomial ``f`` of degree ``deg_f`` (auto-detected in
    :meth:`_initialize` via a ``pjet`` probe if ``None``; call ``_initialize``
    once outside jit, like the ``gsc`` pattern). Diffrax solvers are unsupported:
    refitting is inherently a discrete step.

    Parameters
    ----------
    drift : {"partition", "interval"}
        ``"partition"`` (default) additionally bounds the drift on the level set
        ``{||alpha b||^2 = y}`` via a ``G**n`` boundary-cell grid of the fitted
        box (:func:`_partition_level_bound`), closing the box-vs-set gap the
        pure interval bounds pay (necessary for ``m >= 3``). ``"interval"``
        uses only the coefficient abs-sum bounds (cheaper, viable for
        ``m <= 2``).
    G : int
        Per-axis grid resolution for the partition drift (static).
    """

    deg_f: int = eqx.field(static=True, default=None)
    drift: str = eqx.field(static=True, default="partition")
    G: int = eqx.field(static=True, default=24)

    def _initialize(self, pt0: ChebyshevNormotope) -> ArrayLike:
        if not isinstance(pt0, ChebyshevNormotope):
            raise ValueError(f"{pt0=} is not a ChebyshevNormotope")
        if self.deg_f is None:
            object.__setattr__(self, "deg_f", _probe_degree(self.sys.f, pt0.n))
        return None

    def hypercontrol_shape(self, pt0: ChebyshevNormotope) -> tuple:
        P = int(pt0.alpha.shape[0])
        return (P, P)

    def _dynamics(self, t, state, *args):
        raise NotImplementedError(
            "ChebyshevNormotopeEmbedding is symplectic-only (refitting is a "
            "discrete change of basis); use solver='symplectic'."
        )

    def _refit(self, pt: ChebyshevNormotope) -> ChebyshevNormotope:
        """Exact set identity: recenter and rotate by the linear marginal
        ellipsoid's axes, then scale each axis by the tightest available sound
        extent bound -- the ellipsoid radius or (for ``m >= 2``) the even-slot
        recovery, whichever is smaller -- and normalize ``||alpha||_F``.

        Taking the min per axis is what makes the refit a stable fixed point
        for sets whose extent is carried by even-degree rows (``shape_deg >=
        2``): there the ellipsoid radius is loose, and scaling by it first and
        correcting after does not compose to identity."""
        n, m, P = pt.n, pt.m, pt.P
        mu, E = _slice_frame(pt.alpha, pt.y, n, m)
        lam, V = jnp.linalg.eigh((E + E.T) / 2.0)
        # rotation + recenter: old w = mu + V new w (pure rotation, no scaling)
        alpha = pt.alpha @ scale_matrix(mu, V, n, m)
        Tnew = V.T @ pt.T
        ox = pt.ox + jnp.linalg.solve(pt.T, mu)
        r_ell = jnp.sqrt(jnp.maximum(lam, 0.0))
        if m >= 2:
            # per-axis extent through the pure T_2 slots (in the rotated frame);
            # informative for even-carried shape, huge (inactive) otherwise
            r = jnp.minimum(r_ell, _even_axis_bound(alpha, pt.y, n, m))
        else:
            r = r_ell
        r = jnp.maximum(r, 1e-12)
        alpha = alpha @ scale_matrix(jnp.zeros(n, alpha.dtype), jnp.diag(r), n, m)
        Tnew = jnp.diag(1.0 / r) @ Tnew
        c = jnp.linalg.norm(alpha) / jnp.sqrt(float(P))
        return ChebyshevNormotope(ox, Tnew, alpha / c, pt.y / c**2)

    def _symplectic_step(self, t, dt, state, *args, U=None, adjoint=True):
        pt, _aux = state
        pt = self._refit(pt)
        n, m = pt.n, pt.m
        ox, T, alpha, y = pt.ox, pt.T, pt.alpha, pt.y

        f = lambda x: self.sys.f(t, x)
        A, C_H = chebyshev_lift(f, ox, T, n, m, int(self.deg_f))

        Ut = jnp.zeros_like(alpha) if U is None else U.reshape(alpha.shape)
        if adjoint:
            B = jnp.eye(A.shape[0], dtype=alpha.dtype) + dt * A
            alpha_next = jnp.linalg.solve(B.T, (alpha + dt * Ut).T).T
            W = Ut  # alpha_next (I + dt A) = alpha + dt Ut, exactly
        else:
            alpha_next = alpha + dt * Ut
            W = Ut + alpha_next @ A

        # Exact telescoping along the lifted Euler step b+ = (I + dt A) b + dt H:
        #   alpha_next b+ = alpha b + dt G,   G = W b + alpha_next C_H b_hi,
        # so the adjoint transport cancels and only U and the HOT drive y.
        Hc = alpha_next @ C_H
        g = jnp.linalg.norm(jnp.sum(jnp.abs(W), axis=1) + jnp.sum(jnp.abs(Hc), axis=1))
        sqy = jnp.sqrt(jnp.maximum(y, 0.0))
        # V+ = ||alpha b + dt G||^2 = V + 2 dt (alpha b)^T G + dt^2 ||G||^2,
        # bounded three ways (min is sound):
        y_cs = (sqy + dt * g) ** 2  # Cauchy-Schwarz: ||alpha b|| <= sqrt(y)
        sym = alpha.T @ W
        y_bl = (
            y
            + dt * (jnp.sum(jnp.abs(sym + sym.T)) + 2.0 * jnp.sum(jnp.abs(alpha.T @ Hc)))
            + dt**2 * g**2
        )  # bilinear: every basis term in [-1, 1] on the fitted box
        # Scalar interpolation: expand 2 (alpha b)^T G in Chebyshev and abs-sum
        # the coefficients -- near-tight for the box range (products kept exact).
        E_sc = _scalar_boxbound(alpha, W, Hc, n, m, int(self.deg_f))
        y_sc = y + dt * E_sc + dt**2 * g**2
        y_next = jnp.minimum(jnp.minimum(y_cs, y_bl), y_sc)

        # Boundary-partition drift on the level set (Nagumo-style, like the Gram
        # embedding): closes the box-vs-set gap the pure interval bounds pay.
        if self.drift == "partition" and (C_H.shape[1] > 0 or U is not None):
            E_pt = _partition_level_bound(
                alpha, W, Hc, y, n, m, int(self.deg_f), self.G
            )
            y_pt = jnp.where(
                jnp.isfinite(E_pt), y + dt * E_pt + dt**2 * g**2, jnp.inf
            )
            y_next = jnp.minimum(y_next, y_pt)

        pt_next = ChebyshevNormotope(ox + dt * f(ox), T, alpha_next, y_next)
        return pt_next, None


def _probe_degree(f, n: int, kmax: int = 8) -> int:
    """Polynomial degree of ``x -> f(0, x)`` via a taylax ``pjet`` probe."""
    e = pjet(lambda x: f(0.0, x))(
        FullTensorPolynomial.identity(jnp.zeros(n), order=kmax)
    )
    deg = 1
    for k in range(1, kmax + 1):
        if onp.max(onp.abs(onp.asarray(e.get_order(k).coeffs))) > 1e-6:
            deg = k
    if deg == kmax:
        raise ValueError(
            f"f appears non-polynomial (nonzero Taylor order {kmax}); pass "
            "deg_f explicitly to ChebyshevNormotopeEmbedding."
        )
    return deg
