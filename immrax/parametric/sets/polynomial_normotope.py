r"""Polynomial Normotopes: nonconvex sets from a monomial lifting of a Normotope.

A :class:`PolynomialNormotope` is the set

.. math::
    \{\ox + z : \|\alpha\, p(z)\| \le y\},

where :math:`p(z) = [z;\, p_2(z);\, \dots;\, p_m(z)]` stacks the reduced
degree-:math:`k` monomials of :math:`z` (no constant term) and the shaping matrix
:math:`\alpha \in \R^{|p|\times|p|}` has the **structured** form

.. math::
    \alpha = \begin{bmatrix} \alpha_1 & \alpha_2 & \cdots & \alpha_m \\
                             0 & B_2 & & \\ & & \ddots & \\ 0 & & & B_m \end{bmatrix}

(dense first :math:`n` rows :math:`H = [\alpha_1\mid\cdots\mid\alpha_m]` carrying
the nonconvex geometry, plus block-diagonal coercive regularizers
:math:`B_k`). It is block-upper-triangular, so it is invertible -- and the set
compact -- iff :math:`\alpha_1` and each :math:`B_k` are invertible, and the
inverse is obtained from one :math:`n\times n` and the :math:`B_k` inversions.

This is a Normotope on the lifted space :math:`\R^{|p|}` projected to the first
:math:`n` coordinates; the structure is closed under the adjoint embedding flow.
"""

import functools
import itertools
from math import comb, factorial

import numpy as onp
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import equinox as eqx
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Float
from typing import Callable

from ...inclusion import Interval, interval, icentpert, mjacM
from ...utils import get_sparse_corners
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from .normotope import L2Normotope, L1Normotope, LinfNormotope


# --------------------------------------------------------------------- lift math

def monomial_exponents(n: int, k: int) -> onp.ndarray:
    """(N_k, n) integer exponents of all degree-k monomials in n vars.

    Graded basis, first variable most significant: (2,0) > (1,1) > (0,2).
    """

    def rec(n, k):
        if n == 1:
            return [[k]]
        out = []
        for first in range(k, -1, -1):
            for rest in rec(n - 1, k - first):
                out.append([first] + rest)
        return out

    return onp.asarray(rec(n, k), dtype=onp.int64)


def _multinomial(exp_vec, k: int) -> int:
    r = factorial(k)
    for e in exp_vec:
        r //= factorial(int(e))
    return r


def duplication_matrix(n: int, k: int):
    """``(D_k, D_k^+)`` with ``z^{kron k} = D_k p_k(z)``.

    ``D_k`` is ``(n**k, N_k)`` 0/1; its columns are orthogonal, so
    ``D_k^+ = diag(1/c_beta) D_k^T`` with ``c_beta = k!/prod(beta_i!)`` (no SVD).
    """
    E = monomial_exponents(n, k)
    col = {tuple(int(x) for x in e): j for j, e in enumerate(E)}
    Dk = onp.zeros((n**k, E.shape[0]))
    for row, tup in enumerate(itertools.product(range(n), repeat=k)):
        e = [0] * n
        for i in tup:
            e[i] += 1
        Dk[row, col[tuple(e)]] = 1.0
    c = onp.asarray([_multinomial(e, k) for e in E], dtype=float)  # = diag(D_k^T D_k)
    Dkp = (1.0 / c)[:, None] * Dk.T
    return Dk, Dkp


@functools.lru_cache(maxsize=None)
def _lift_data(n: int, m: int):
    """Static lift constants for ``(n, m)``: per-degree exponents and ``(D_k, D_k^+)``.

    Index ``k-1`` holds degree ``k`` (``D``/``Dp`` at index 0 are unused: the
    degree-1 block is just ``M``). Built once on host as numpy constants.
    """
    E = tuple(monomial_exponents(n, k) for k in range(1, m + 1))
    D, Dp = [None], [None]
    for k in range(2, m + 1):
        d, dp = duplication_matrix(n, k)
        D.append(d)
        Dp.append(dp)
    return E, tuple(D), tuple(Dp)


def _degree_from_shape(n: int, P: int) -> int:
    """Recover the degree ``m`` from ``n`` and ``|p| = P`` (sum is monotone in m)."""
    total, k = 0, 0
    while total < P:
        k += 1
        total += comb(n + k - 1, k)
    if total != P:
        raise ValueError(f"|p| = {P} is not a valid lift size for n={n}")
    return k


def degree_sizes(n: int, m: int) -> list:
    """``[N_1, ..., N_m]`` with ``N_k = C(n+k-1, k)``."""
    return [comb(n + k - 1, k) for k in range(1, m + 1)]


def lift_state(z: Array, E: Array) -> Array:
    """Degree block ``p_k(z)`` for exponent matrix ``E`` (``(N_k, n)``)."""
    base = jnp.where(E == 0, 1.0, z[None, :])  # zero exponent -> factor 1 (safe 0**0)
    return jnp.prod(base**E, axis=1)


def kron_sum(M: Array, k: int, n: int) -> Array:
    r""":math:`M^{\oplus k} = \sum_j I^{\otimes(j-1)} \otimes M \otimes I^{\otimes(k-j)}`."""
    if k == 1:
        return M
    total = jnp.zeros((n**k, n**k), dtype=M.dtype)
    for j in range(1, k + 1):
        left = jnp.eye(n ** (j - 1), dtype=M.dtype)
        right = jnp.eye(n ** (k - j), dtype=M.dtype)
        total = total + jnp.kron(jnp.kron(left, M), right)
    return total


def lift_blocks(M: Array, n: int, m: int, D, Dp) -> list:
    r"""Per-degree lift blocks ``[L_1(M), ..., L_m(M)]`` with ``L_1 = M`` and
    ``L_k(M) = D_k^+ M^{\oplus k} D_k`` -- the diagonal blocks of :func:`lift_operator`."""
    blocks = [M]
    for k in range(2, m + 1):
        blocks.append(Dp[k - 1] @ kron_sum(M, k, n) @ D[k - 1])
    return blocks


def lift_operator(M: Array, n: int, m: int, D, Dp) -> Array:
    r"""Block-diagonal lift :math:`L(M) = \blkdiag(M, D_2^+ M^{\oplus2} D_2, \dots)`.

    Generator of :math:`\dot p(z)` under :math:`\dot z = Mz`. Linear in ``M`` and
    vmap-able over a ``(num_corners, n, n)`` stack of point matrices.
    """
    return jax.scipy.linalg.block_diag(*lift_blocks(M, n, m, D, Dp))


# --------------------------------------------------- structured-alpha assembly

def assemble_alpha(H: Array, Bs, n: int) -> Array:
    """Dense ``|p|x|p|`` shaping matrix from the structured components.

    ``H`` is ``(n, |p|)`` (dense first rows); ``Bs`` the block-diagonal blocks
    ``(B_2, ..., B_m)``.
    """
    P = H.shape[1]
    if not Bs:
        return H
    D = jax.scipy.linalg.block_diag(*Bs)  # (P-n, P-n)
    bottom = jnp.concatenate([jnp.zeros((P - n, n), H.dtype), D], axis=1)
    return jnp.concatenate([H, bottom], axis=0)


def alpha_inv_top(H: Array, Bs, n: int) -> Array:
    """First ``n`` rows of the analytic inverse: ``[alpha_1^{-1} | -alpha_1^{-1} R D^{-1}]``.

    These rows are all that ``iover`` (the projection to the first ``n`` coords)
    needs.
    """
    a1inv = jnp.linalg.inv(H[:, :n])
    if not Bs:
        return a1inv
    Dinv = jax.scipy.linalg.block_diag(*[jnp.linalg.inv(B) for B in Bs])
    return jnp.concatenate([a1inv, -a1inv @ H[:, n:] @ Dinv], axis=1)


def block_inverse(H: Array, Bs, n: int) -> Array:
    """Dense ``|p|x|p|`` inverse via the block-upper-triangular formula.

    Needs only the ``alpha_1`` (n x n) and ``B_k`` inversions, never a dense solve.
    """
    top = alpha_inv_top(H, Bs, n)  # (n, P)
    if not Bs:
        return top
    P = H.shape[1]
    Dinv = jax.scipy.linalg.block_diag(*[jnp.linalg.inv(B) for B in Bs])
    bottom = jnp.concatenate([jnp.zeros((P - n, n), H.dtype), Dinv], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def _vec_norm(norm_cls, vec: Array) -> Float:
    """Borrow the (state-free) vector norm of the matching Normotope subclass."""
    P = vec.shape[0]
    return norm_cls(jnp.zeros(P), jnp.eye(P), 1.0).norm(vec)


# ------------------------------------------------------------------- set classes

@register_pytree_node_class
class PolynomialNormotope(Parametope):
    r"""Structured polynomial normotope ``{ox + z : ||alpha p(z)|| <= y}``.

    Stores the structured components ``(ox, H, Bs, y)`` -- dense first rows
    ``H = [alpha_1 | ... | alpha_m]`` (``(n, |p|)``) and block-diagonal
    regularizers ``Bs = (B_2, ..., B_m)`` -- not the dense ``alpha``. A concrete
    subclass sets ``norm_cls`` (the matching :class:`Normotope` subclass) to fix
    the lifted-space norm and its log-norm ``mu``.
    """

    norm_cls: type = None  # set by subclass (e.g. L2Normotope)

    def __init__(self, ox: ArrayLike, H: ArrayLike, Bs, y: Float):
        self.ox = ox
        self.H = H
        self.Bs = tuple(Bs)
        self.y = y

    # -- pytree: children are the structured components --
    def tree_flatten(self):
        return ((self.ox, self.H, self.Bs, self.y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ox, H, Bs, y = children
        return cls(ox, H, tuple(Bs), y)

    # -- structure descriptors --
    @property
    def n(self) -> int:
        return int(self.ox.shape[0])

    @property
    def P(self) -> int:
        """Lift size ``|p|``."""
        return int(self.H.shape[1])

    @property
    def m(self) -> int:
        return _degree_from_shape(self.n, self.P)

    @property
    def alpha_1(self) -> Array:
        """The invertible linear block ``H[:, :n]``."""
        return self.H[:, : self.n]

    @property
    def alpha(self) -> Array:
        """The dense ``|p|x|p|`` shaping matrix (assembled on demand)."""
        return assemble_alpha(self.H, self.Bs, self.n)

    # -- geometry --
    def _p_blocks(self, z: Array) -> list:
        E = _lift_data(self.n, self.m)[0]
        return [lift_state(z, Ek) for Ek in E]

    def p(self, z: Array) -> Array:
        """The monomial lift ``[z; p_2(z); ...; p_m(z)]``."""
        return jnp.concatenate(self._p_blocks(z))

    def alpha_p(self, z: Array) -> Array:
        """``alpha p(z)`` in ``R^{|p|}`` (top = H p(z), then B_k p_k(z))."""
        pb = self._p_blocks(z)
        top = self.H @ jnp.concatenate(pb)
        bottom = [self.Bs[k - 2] @ pb[k - 1] for k in range(2, self.m + 1)]
        return jnp.concatenate([top, *bottom]) if bottom else top

    def g(self, x: Array) -> Float:
        return _vec_norm(self.norm_cls, self.alpha_p(x - self.ox))

    def mu(self, A: Array) -> Float:
        """Logarithmic norm of the lifted-space norm (used by the embedding)."""
        return self.norm_cls.mu(A)

    def iover(self, ainv_top: Array | None = None) -> Interval:
        """Sound interval enclosure: project the lifted normotope to the first
        ``n`` coordinates via the (analytic) top rows of ``alpha^{-1}``."""
        C = alpha_inv_top(self.H, self.Bs, self.n) if ainv_top is None else ainv_top
        W = self.norm_cls.norm_ball_iover(self.P) * self.y  # box >= {||w|| <= y}
        return interval(C) @ W + self.ox

    # -- (de)vectorization --
    def vec(self) -> Array:
        flat_B = [B.reshape(-1) for B in self.Bs]
        return jnp.concatenate([self.ox, self.H.reshape(-1), *flat_B, jnp.atleast_1d(self.y)])

    @classmethod
    def unvec(cls, vec: Array, n: int, m: int) -> "PolynomialNormotope":
        Ns = degree_sizes(n, m)
        P = sum(Ns)
        ox = vec[:n]
        off = n + n * P
        H = vec[n:off].reshape(n, P)
        Bs = []
        for Nk in Ns[1:]:
            Bs.append(vec[off : off + Nk * Nk].reshape(Nk, Nk))
            off += Nk * Nk
        return cls(ox, H, Bs, vec[-1])


@register_pytree_node_class
class L2PolynomialNormotope(PolynomialNormotope):
    norm_cls = L2Normotope

    def iover(self, ainv_top: Array | None = None) -> Interval:
        """Tight L2 enclosure: per-coordinate support of the projected ellipsoid."""
        C = alpha_inv_top(self.H, self.Bs, self.n) if ainv_top is None else ainv_top
        return icentpert(self.ox, self.y * jnp.sqrt(jnp.sum(C**2, axis=1)))


@register_pytree_node_class
class L1PolynomialNormotope(PolynomialNormotope):
    norm_cls = L1Normotope


@register_pytree_node_class
class LinfPolynomialNormotope(PolynomialNormotope):
    norm_cls = LinfNormotope


# --------------------------------------------------------------------- embedding

class PolynomialNormotopeEmbedding(ParametricEmbedding):
    r"""Embedding onto structured :class:`PolynomialNormotope` dynamics.

    .. math::
        \dot{\ox} &= f(t, \ox) \\
        \dot{\alpha} &= -\alpha\, L(Df(\ox)) + U \\
        \dot{y} &= \max_{M} \mu(\dot\alpha\,\alpha^{-1} + \alpha\, L(M)\, \alpha^{-1})\, y,

    where ``M`` ranges over the sparse corners of the base ``n x n`` interval
    Jacobian and the adjoint drift ``-alpha L(Df)`` is applied block-wise so the
    structure is preserved. The inverse is obtained analytically from the block
    decomposition (``inverse_mode="block"``, default) or flowed in the aux channel
    via ``alpha^{-1}_dot = -alpha^{-1} alpha_dot alpha^{-1}`` (``"flow"``).

    Parameters
    ----------
    gsc : Callable, optional
        Sparse-corner strategy for the base interval Jacobian; resolved in
        :meth:`_initialize` if ``None``.
    inverse_mode : {"block", "flow"}
        How ``alpha^{-1}`` is obtained.
    """

    gsc: Callable = eqx.field(static=True, default=None)
    inverse_mode: str = eqx.field(static=True, default="block")

    def _initialize(self, pnt0: PolynomialNormotope) -> ArrayLike:
        if not isinstance(pnt0, PolynomialNormotope):
            raise ValueError(f"{pnt0=} is not a PolynomialNormotope")
        if self.gsc is None:
            ix0 = pnt0.iover()
            M = mjacM(self.sys.f)(0.0, ix0, center=(jnp.zeros(1), pnt0.ox))[1]
            object.__setattr__(self, "gsc", get_sparse_corners(interval(M)))
        if self.inverse_mode == "flow":
            return block_inverse(pnt0.H, pnt0.Bs, pnt0.n)  # aux = flowed alpha^{-1}
        return None

    def hypercontrol_shape(self, pnt0: PolynomialNormotope) -> tuple:
        # Flat structured control: vec(U_H) then vec(U_{B_k}) for each block.
        return (int(pnt0.H.size + sum(B.size for B in pnt0.Bs)),)

    def _split_control(self, U: Array, pnt: PolynomialNormotope):
        n, P = pnt.n, pnt.P
        U_H = U[: n * P].reshape(n, P)
        off, U_Bs = n * P, []
        for B in pnt.Bs:
            U_Bs.append(U[off : off + B.size].reshape(B.shape))
            off += B.size
        return U_H, U_Bs

    def iover(self, state):
        pnt, aux = state
        if self.inverse_mode == "flow":
            return pnt.iover(ainv_top=aux[: pnt.n])
        return pnt.iover()

    def _dynamics(self, t, state, *args, U=None, perm=None, adjoint=True, ix=None):
        pnt, aux = state
        n, m = pnt.n, pnt.m
        _, D, Dp = _lift_data(n, m)
        Ns = degree_sizes(n, m)

        # Per-degree lifts of the center Jacobian; adjoint transports each block.
        A = jax.jacfwd(self.sys.f, 1)(0.0, pnt.ox)
        Lb = lift_blocks(A, n, m, D, Dp)  # [L_1(A), ..., L_m(A)]

        if U is not None:
            U_H, U_Bs = self._split_control(U, pnt)
        else:
            U_H = jnp.zeros_like(pnt.H)
            U_Bs = [jnp.zeros_like(B) for B in pnt.Bs]

        if adjoint:
            cols, off = [], 0
            for k, Nk in enumerate(Ns):
                cols.append(pnt.H[:, off : off + Nk] @ Lb[k])
                off += Nk
            H_dot = -jnp.concatenate(cols, axis=1) + U_H
            Bs_dot = [-(pnt.Bs[k - 2] @ Lb[k - 1]) + U_Bs[k - 2] for k in range(2, m + 1)]
        else:
            H_dot, Bs_dot = U_H, list(U_Bs)

        alpha = assemble_alpha(pnt.H, pnt.Bs, n)
        alpha_dot = assemble_alpha(H_dot, tuple(Bs_dot), n)
        ainv = aux if self.inverse_mode == "flow" else block_inverse(pnt.H, pnt.Bs, n)

        box = pnt.iover(ainv_top=ainv[:n])
        box = box if ix is None else (ix & box)
        Mx = interval(
            mjacM(self.sys.f)(t, box, center=(jnp.zeros(1), pnt.ox), permutation=perm)[1]
        )

        # Exact contraction: corner the base n x n Jacobian, lift each corner, max
        # mu over corners (mu convex, L affine => exact sup over [Mx]).
        base = alpha_dot @ ainv
        Ms = self.gsc(Mx)
        mus = jax.vmap(
            lambda Mi: pnt.mu(base + alpha @ lift_operator(Mi, n, m, D, Dp) @ ainv)
        )(Ms)
        y_dot = jnp.max(mus) * pnt.y

        pt_dot = pnt.__class__(self.sys.f(0.0, pnt.ox), H_dot, tuple(Bs_dot), y_dot)
        aux_dot = -aux @ alpha_dot @ aux if self.inverse_mode == "flow" else None
        return pt_dot, aux_dot
