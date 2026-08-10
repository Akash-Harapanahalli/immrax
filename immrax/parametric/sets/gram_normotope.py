r"""Gram (SOS) polynomial normotope: the nonconvex set

.. math::
    \{\ox + z : \|R\,p(z)\| \le y\} \;=\; \{\ox + z : p(z)^\T P\, p(z) \le y^2\},
    \qquad P = R^\T R,

carried in **dense square-root** form ``R`` (``|p|x|p|``) rather than the
structured factor of :class:`PolynomialNormotope`. ``p(z)`` stacks the reduced
degree-:math:`1\dots m` monomials of :math:`z` in **taylax order** so the
Carleman lift ``A`` (used by the embedding) and ``p`` share a basis.

The point of the dense factor: the Loewner-monotone control of the
forward-invariance theory lives on the Gram ``P``, and a dense ``R`` realizes the
exact ``P``-flow ``\dot P = -A^\T P - P A - (U')^\T U'`` via

.. math::
    \dot R = -R\,A - \tfrac12 R^{-\T}(U')^\T U',

with ``P = R^\T R \succ 0`` structural (no PD assertion) and the in-loop drift
reusing the factor-form level-set bound verbatim (``\alpha := R``). See
:class:`GramNormotopeEmbedding`.
"""

import functools

import numpy as onp
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Float
from typing import Callable

from taylax import FullTensorPolynomial, pjet

from ...inclusion import Interval, interval, natif, icentpert
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from .polynomial_normotope import _degree_from_shape


# ----------------------------------------------------------------- lift basis

@functools.lru_cache(maxsize=None)
def _gram_basis(n: int, m: int):
    """``(exps, lin_idx)`` for ``(n, m)`` in **taylax order**.

    ``exps`` is ``(|p|, n)`` integer exponents of the degree-``1..m`` monomials
    (taylax order, matching :func:`pjet`'s ``get_order``). ``lin_idx`` gives the
    rows of the ``n`` linear monomials ``e_i`` (so ``z_i = p(z)[lin_idx[i]]``),
    used to project to the first ``n`` coordinates.
    """
    blocks = []
    for k in range(1, m + 1):
        e = pjet(lambda z: jnp.sum(z**0 * 0.0) + jnp.sum(z))(
            FullTensorPolynomial.identity(jnp.zeros(n), order=k)
        )
        blocks.append(onp.asarray(e.get_order(k).multiindices.to_numpy()).T)
    exps = onp.concatenate(blocks, axis=0).astype(int)  # (|p|, n)
    eye = onp.eye(n, dtype=int)
    lin_idx = onp.array(
        [int(onp.where((exps[:n] == eye[i]).all(1))[0][0]) for i in range(n)]
    )
    return exps, lin_idx


def _lift(z: Array, exps) -> Array:
    """``p(z)`` for static integer ``exps``; uses ``integer_pow`` (natif-safe)."""
    cols = []
    for k in range(exps.shape[0]):
        t = jnp.ones((), z.dtype)
        for i in range(exps.shape[1]):
            e = int(exps[k, i])
            if e:
                t = t * z[i] ** e
        cols.append(t)
    return jnp.stack(cols)


def carleman_A(f: Callable, ox: Array, exps, m: int) -> Array:
    r"""Full block-upper-triangular Carleman lift ``A`` of the error field at ``ox``.

    ``A p(z)`` is the degree-``<= m`` part of the lifted field ``Dp(z) f_err(z)``
    (``f_err(z) = f(ox+z) - f(ox)``); the escape ``Dp f_err - A p`` is then the
    pure degree-``m+1`` residual. Same basis as :func:`_lift`.
    """
    n = ox.shape[0]
    pf = lambda z: _lift(z, exps)

    def fe(z):
        return f(ox + z) - f(ox)

    def lifted_field(z):
        return jax.jvp(pf, (z,), (fe(z),))[1]  # Dp(z) @ fe(z)

    e = pjet(lifted_field)(FullTensorPolynomial.identity(jnp.zeros(n), order=m))
    return jnp.concatenate([e.get_order(i).coeffs for i in range(1, m + 1)], axis=1)


# ------------------------------------------------------------------- set class

@register_pytree_node_class
class GramNormotope(Parametope):
    r"""Dense-factor L2 polynomial normotope ``{ox + z : ||R p(z)||_2 <= y}``.

    Children are ``(ox, R, y)`` (``R`` stored as ``alpha``); ``P = R^T R`` is the
    SOS Gram matrix. Monomial basis is taylax-ordered (see :func:`_gram_basis`).
    """

    def __init__(self, ox: ArrayLike, R: ArrayLike, y: Float):
        super().__init__(ox, R, y)

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "GramNormotope":
        return cls(pt.ox, pt.alpha, pt.y)

    # -- structure --
    @property
    def R(self) -> Array:
        return self.alpha

    @property
    def n(self) -> int:
        return int(self.ox.shape[0])

    @property
    def P(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def m(self) -> int:
        return _degree_from_shape(self.n, self.P)

    @property
    def Pmat(self) -> Array:
        """The SOS Gram matrix ``P = R^T R``."""
        return self.alpha.T @ self.alpha

    @property
    def P11(self) -> Array:
        """Top-left ``n x n`` Gram block ``(R^T R)[:n, :n]`` (the convex shadow)."""
        n = self.n
        return self.Pmat[:n, :n]

    # -- geometry --
    def _exps(self):
        return _gram_basis(self.n, self.m)[0]

    def p(self, z: Array) -> Array:
        return _lift(z, self._exps())

    def g(self, x: Array) -> Float:
        return jnp.linalg.norm(self.alpha @ self.p(x - self.ox))

    def iover(self) -> Interval:
        """Tight L2 enclosure: per-coordinate support of the projected ellipsoid.

        For any factor ``R`` and ``z`` in the set, ``z = (R^{-1} R p(z))[lin]`` and
        ``||R p(z)|| <= y``, so ``z`` lies in the box ``C {||w|| <= y}`` with ``C``
        the linear-monomial rows of ``R^{-1}``.
        """
        n = self.n
        _, lin_idx = _gram_basis(n, self.m)
        C = jnp.linalg.inv(self.alpha)[jnp.asarray(lin_idx), :]
        return icentpert(self.ox, self.y * jnp.sqrt(jnp.sum(C**2, axis=1)))

    def log_volume_linear(self) -> Float:
        """Log-volume of the convex (degree-1) shadow up to an additive constant:
        ``n log y - 1/2 logdet P11``."""
        sign, logdet = jnp.linalg.slogdet(self.P11)
        return self.n * jnp.log(self.y) - 0.5 * logdet


# --------------------------------------------------------------------- drift

def _level_set_drift(f, ox, R, A, y, Up, exps, lin_idx, G):
    r"""In-loop tight level-set drift ``ydot`` for the Gram embedding.

    ``ydot = max_{||R p(z)|| = y} (N_ctrl(z)) / y`` with

        N_ctrl(z) = <R p, R (Dp f_err - A p)> - 1/2 ||U' p||^2,   V(z) = ||R p||^2,

    bounded over a fixed ``G**n`` grid of the ``iover`` z-box: only cells whose
    ``natif(V)`` straddles ``y^2`` contribute; each contributes the
    Lagrangian-projected centered mean-value upper bound (``where``-masked, so the
    shape is static -> JIT/vmap/autodiff clean). Sound: the boundary
    ``{V = y^2}`` is covered by the straddling cells.
    """
    n = ox.shape[0]
    y2 = y * y

    def pf(z):
        return _lift(z, exps)

    def fe(z):
        return f(ox + z) - f(ox)

    def Nfun(z):
        Rp = R @ pf(z)
        esc = jax.jvp(pf, (z,), (fe(z),))[1] - A @ pf(z)  # Dp f_err - A p
        ctrl = 0.5 * jnp.sum((Up @ pf(z)) ** 2)
        return jnp.dot(Rp, R @ esc) - ctrl

    def Vfun(z):
        return jnp.sum((R @ pf(z)) ** 2)

    nN = natif(Nfun)
    nV = natif(Vfun)
    nJN = natif(jax.jacfwd(Nfun))
    nJV = natif(jax.jacfwd(Vfun))
    gN = jax.grad(Nfun)
    gV = jax.grad(Vfun)

    # iover z-box half-widths (linear-monomial rows of R^{-1}).
    C = jnp.linalg.inv(R)[jnp.asarray(lin_idx), :]
    zr = y * jnp.sqrt(jnp.sum(C**2, axis=1))
    hw = zr / G
    axes = [jnp.linspace(-zr[i] + hw[i], zr[i] - hw[i], G) for i in range(n)]
    ZC = jnp.stack([a.ravel() for a in jnp.meshgrid(*axes, indexing="ij")], axis=1)

    def cell(zc):
        bx = interval(zc - hw, zc + hw)
        Vb = nV(bx)
        keep = (Vb.lower <= y2) & (y2 <= Vb.upper)
        gNc = gN(zc)
        gVc = gV(zc)
        lam = jnp.dot(gNc, gVc) / (jnp.dot(gVc, gVc) + 1e-30)
        gc = Nfun(zc) - lam * (Vfun(zc) - y2)  # = N_ctrl(zc) on the level set
        JN = nJN(bx)
        JV = nJV(bx)
        lo = jnp.minimum(lam * JV.lower, lam * JV.upper)
        hi = jnp.maximum(lam * JV.lower, lam * JV.upper)
        ub = gc + jnp.sum(jnp.maximum(jnp.abs(JN.lower - hi), jnp.abs(JN.upper - lo)) * hw)
        return jnp.where(keep, ub, -jnp.inf)

    return jnp.max(jax.vmap(cell)(ZC)) / y


# --------------------------------------------------------------------- embedding

class GramNormotopeEmbedding(ParametricEmbedding):
    r"""Embedding onto dense-factor :class:`GramNormotope` dynamics.

    .. math::
        \dot\ox &= f(t, \ox), \\
        \dot R  &= -R\,A - \tfrac12 R^{-\T}(U')^\T U', \\
        \dot y  &= \max_{\|R p(z)\| = y}
                   \frac{\langle Rp, R(Dp\,f_{\mathrm{err}} - A p)\rangle
                         - \tfrac12\|U' p(z)\|^2}{y},

    where ``A`` is the full Carleman lift (:func:`carleman_A`) and ``ydot`` is the
    in-loop tight level-set drift (:func:`_level_set_drift`) over a ``G**n`` grid.
    The control ``U'`` (shape ``(|p|, |p|)``) realizes ``\dot P = -A^\T P - P A -
    (U')^\T U'`` on the Gram ``P = R^\T R`` -- Loewner-NSD by construction.

    Parameters
    ----------
    G : int
        Per-axis grid resolution for the drift bound (static). Larger ``G`` is
        tighter and slower; the whole point is an offline, accurate drift.
    """

    G: int = eqx.field(static=True, default=24)

    def _initialize(self, pt0: GramNormotope) -> ArrayLike:
        if not isinstance(pt0, GramNormotope):
            raise ValueError(f"{pt0=} is not a GramNormotope")
        return None

    def hypercontrol_shape(self, pt0: GramNormotope) -> tuple:
        P = int(pt0.alpha.shape[0])
        return (P, P)

    def _dynamics(self, t, state, *args, U=None, adjoint=True, ix=None):
        pt, _aux = state
        n, m = pt.n, pt.m
        exps, lin_idx = _gram_basis(n, m)
        R, y = pt.alpha, pt.y

        f = lambda x: self.sys.f(t, x)
        A = carleman_A(f, pt.ox, exps, m)

        Up = jnp.zeros_like(R) if U is None else U
        if adjoint:
            R_dot = -R @ A - 0.5 * jnp.linalg.inv(R).T @ (Up.T @ Up)
        else:
            R_dot = -0.5 * jnp.linalg.inv(R).T @ (Up.T @ Up)

        y_dot = _level_set_drift(f, pt.ox, R, A, y, Up, exps, lin_idx, self.G) * y

        pt_dot = pt.__class__(self.sys.f(t, pt.ox), R_dot, y_dot)
        return pt_dot, None
