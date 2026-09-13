import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from ...embedding import embed
from ...inclusion import (
    Interval,
    Permutation,
    icopy,
    i2ut,
    interval,
    mjacM,
    natif,
)
from ...neural import fastlin
from ..parametope import Parametope
from ..embedding import ParametricEmbedding


@register_pytree_node_class
class AffineParametope(Parametope):
    r"""Defines a parametope with the particular structured nonlinearity

    .. math::
        g(\alpha, x - \mathring{x}) = (-h(\alpha (x - \mathring{x})), h(\alpha (x - \mathring{x})))

    and y split into lower and upper bounds y = (ly, uy).
    """

    def h(self, z: ArrayLike):
        """Evaluates the nonlinearity h at z

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        pass

    def g(self, x: ArrayLike):
        """Evaluates the nonlinearity g at alpha, x

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        return self.h(jnp.dot(self.alpha, x - self.ox))

    def hinv(self, iy: Interval):
        """Overapproximating inverse image of the nonlinearity h

        Parameters
        ----------
        iy : ArrayLike
            _description_
        """
        pass

    def k_face(self, k: int) -> Interval:
        """Overapproximate the k-face of the hParametope"""
        pass

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope(cls, pt: "hParametope"):
        return pt


hParametope = AffineParametope


class AdjointEmbedding(ParametricEmbedding):
    def __init__(
        self,
        sys,
        alpha_p0,
        N0,
        kap: float = None,
        permutation=None,
        disable_adjoint=False,
    ):
        super().__init__(sys)
        self.Jf_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.kap = kap
        self.alpha_p0 = alpha_p0
        self.N0 = N0
        self.permutation = permutation
        self.disable_adjoint = disable_adjoint

    def _initialize(self, pt0: hParametope) -> ArrayLike:
        if not isinstance(pt0, hParametope):
            raise ValueError(f"{pt0=} is not a hParametope needed for AdjointEmbedding")

        return (self.alpha_p0, self.N0)

    def hypercontrol_shape(self, pt0: hParametope) -> tuple:
        # The hypercontrol is added to the H (alpha) dynamics.
        return pt0.alpha.shape

    def iover(self, state):
        pt, aux = state
        alpha_p, N = aux
        iz = _refine_N(pt.hinv(pt.y), N)
        return interval(alpha_p) @ iz + pt.ox

    def _dynamics(self, t, state, *args, U=None, alpha_pair=None, **kwargs):
        r"""Offset dynamics for the frame ``pt.alpha``.

        With ``A`` the Jacobian frozen at the center, ``M`` its mixed-Jacobian
        enclosure over the set, and ``U*`` the CBF plus hypercontrol forcing on
        ``alpha_dot = -alpha A + U*``, the offsets grow along

            z_dot = W alpha^+ z,   W = alpha_pair (M - A) + U*.

        ``alpha_pair`` defaults to ``pt.alpha``; :meth:`_symplectic_step` passes
        ``alpha_next``, the multiplier the exact discrete pairing requires.
        """
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y
        alpha_pair = alpha if alpha_pair is None else alpha_pair

        args_centers = tuple(arg.center for arg in args)
        centers = (jnp.array([t]), ox) + args_centers

        J = self.Jf_x(*centers)
        A = jnp.zeros_like(J) if self.disable_adjoint else J
        U0 = -alpha @ A
        Ustar = _licq_Ustar(alpha, U0, self.kap) + _hyper(U, alpha)

        if self.permutation is None:
            self.permutation = Permutation(range(sum(len(c) for c in centers)))

        big_iz = pt.hinv(y)
        MM = self.Mf(
            t,
            interval(alpha_p) @ big_iz + ox,
            *args,
            center=centers,
            permutation=self.permutation,
        )

        # Disturbance arguments, bounded about their centers. Like the state
        # residual, these pair with alpha_pair.
        dist = interval(jnp.zeros_like(ox))
        for M, arg, cent in zip(MM[2:], args, args_centers):
            dist = dist + interval(M) @ (arg - cent)
        dist = interval(alpha_pair) @ dist

        W = interval(alpha_pair) @ (MM[1] - A) + Ustar

        K = len(y) // 2
        mul = jnp.concatenate((-jnp.ones(K), jnp.ones(K)))  # sign of lower offsets
        Jh = natif(jax.jacfwd(lambda z: jnp.asarray(pt.h(z))))

        def F(t, iy, *args):
            iz = pt.hinv(i2ut(_refine_N(iy, N)) * mul)
            PH = Jh(iz)  # post first-order cancellation
            return interval(PH[len(PH) // 2 :, :]) @ (
                W @ (interval(alpha_p) @ iz) + dist
            )

        E_res = embed(F)(t, y * mul, *args) * mul

        H_dot = U0 + Ustar
        pt_dot = pt.from_parametope(hParametope(self.sys.f(*centers), H_dot, E_res))
        return pt_dot, (-alpha_p @ H_dot @ alpha_p, -N @ H_dot @ alpha_p)

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y

        centers = (jnp.array([t]), ox) + tuple(arg.center for arg in args)
        J = self.Jf_x(*centers)
        A = jnp.zeros_like(J) if self.disable_adjoint else J
        Ustar = _licq_Ustar(alpha, -alpha @ A, self.kap) + _hyper(U, alpha)

        # alpha_next (I + dt A) = alpha + dt U* is the exact discrete adjoint of
        # the Euler state map, so z_next = z + dt (alpha_next (M - A) + U*)(x - ox):
        # the residual pairs with alpha_NEXT. Pairing it with alpha instead leaks
        # dt alpha_dot M -- first order in the set size, and of either sign, so it
        # under-approximates whenever that sign is negative.
        B = jnp.eye(A.shape[0], dtype=alpha.dtype) + dt * A
        alpha_next = jnp.linalg.solve(B.T, (alpha + dt * Ustar).T).T
        forced = None if (U is None and self.kap is None) else alpha + dt * Ustar
        alpha_p_next, N_next = _step_aux(B, alpha_p, N, dt, Ustar, forced)

        pt_dot, _ = self._dynamics(
            t, state, *args, U=U, alpha_pair=alpha_next, **kwargs
        )
        y_next = y + dt * pt_dot.y

        # Row normalization: alpha_i z <= y_i and (c alpha_i) z <= c y_i are the
        # same halfspace, so this is a no-op on the set while stopping the
        # adjoint's exponential magnitude growth (and the resulting overflow).
        d = jnp.linalg.norm(alpha_next, axis=1)
        K = y_next.shape[0] // 2
        y_next = jnp.concatenate([y_next[:K] / d, y_next[K:] / d])
        return (
            pt.from_parametope(
                hParametope(ox + dt * pt_dot.ox, alpha_next / d[:, None], y_next)
            ),
            (alpha_p_next * d[None, :], N_next * d[None, :]),
        )


def _stacked_y(y, d):
    """Offsets of a stacked frame, as (block-1 interval, block-2 interval)."""
    m = y.shape[0] // 2
    lo, up = -y[:m], y[m:]
    return interval(lo[:d], up[:d]), interval(lo[d:], up[d:])


def _join_y(b1, b2):
    return jnp.concatenate([-b1.lower, -b2.lower, b1.upper, b2.upper])


def _meet(a, b):
    return interval(jnp.maximum(a.lower, b.lower), jnp.minimum(a.upper, b.upper))


class StackedAdjointEmbedding(AdjointEmbedding):
    r"""Adjoint embedding on a doubled frame :math:`\alpha = [A;\,I]`.

    Block 1 follows the adjoint; block 2 is held at the identity. Freezing it is
    a hypercontrol: the discrete pairing ``alpha_next (I + dt J) = alpha + dt
    U*`` forces ``U*_2 = J``, and block 2's offset growth becomes
    ``(I(Mx - J) + J) = Mx`` -- the plain interval bound.

    Each step the blocks refine each other, and ``_dynamics`` is handed
    ``[0 | I]`` so the hull it bounds the mean-value remainder over is their
    *meet*: as tight as the adjoint's own frame normally, capped by the
    axis-aligned bound once that frame's hull starts to run away. The cap is
    what stops the offset blowup a plain adjoint suffers on wide initial sets.

    Build the initial set with :meth:`~immrax.parametric.Polytope.stacked_from_interval`.
    """

    def __init__(self, sys, n, permutation=None, refine=True):
        # aux carries ap1 = A^-1 alone (n x n). [ap1 | 0] would also be a valid
        # left inverse of [A; I], but its zero block is stored at every step --
        # 2.4 GB of zeros at n=250.
        super().__init__(sys, jnp.eye(n), jnp.zeros((0, 2 * n)),
                         permutation=permutation)
        self.refine = refine

    def iover(self, state):
        pt, (ap1, _) = state
        b1, b2 = _stacked_y(pt.y, pt.alpha.shape[1])
        return _meet(interval(ap1) @ b1, b2) + pt.ox

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        pt, (ap1, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y
        d = alpha.shape[1]
        centers = (jnp.array([t]), ox) + tuple(a.center for a in args)
        J = self.Jf_x(*centers)
        I = jnp.eye(d, dtype=alpha.dtype)
        B = I + dt * J

        A_cur = alpha[:d]
        A_next = jnp.linalg.solve(B.T, A_cur.T).T
        ap1_next = B @ ap1      # exact inverse update; no O(d^3) solve per step
        Ustar = jnp.vstack([jnp.zeros_like(J), J])

        if self.refine:
            b1, b2 = _stacked_y(y, d)
            b2 = _meet(b2, interval(ap1) @ b1)
            b1 = _meet(b1, interval(A_cur) @ b2)
            y = _join_y(b1, b2)

        alpha_next = jnp.vstack([A_next, I])
        state_next = (
            pt.from_parametope(hParametope(ox, alpha_next, y)),
            (jnp.hstack([jnp.zeros((d, d), alpha.dtype), I]), N),
        )
        pt_dot, _ = self._dynamics(
            t, state_next, *args, U=Ustar, alpha_pair=alpha_next, **kwargs
        )
        y_next = y + dt * pt_dot.y

        nrm = jnp.concatenate([jnp.linalg.norm(A_next, axis=1),
                               jnp.ones(d, alpha.dtype)])
        alpha_next = alpha_next / nrm[:, None]
        m = y_next.shape[0] // 2
        y_next = jnp.concatenate([y_next[:m] / nrm, y_next[m:] / nrm])
        return (
            pt.from_parametope(hParametope(ox + dt * pt_dot.ox, alpha_next, y_next)),
            (ap1_next * nrm[:d][None, :], N),
        )


def _hyper(U, alpha):
    """Hypercontrol forcing on alpha_dot, reshaped to the frame."""
    return jnp.zeros_like(alpha) if U is None else jnp.asarray(U).reshape(alpha.shape)


def _step_aux(B, alpha_p, N, dt, Ustar, forced):
    """alpha_p / N updates preserving alpha_p @ alpha = I and N @ alpha = 0.

    ``forced = alpha + dt U*``, or None when U* == 0 -- then alpha_next =
    alpha @ B^-1 and the updates are exact without an inverse. Square ``forced``
    uses inv (LU): cheaper and better-behaved under the iLQR backward pass.
    """
    if forced is None:
        return B @ alpha_p, N
    Mh = (
        jnp.linalg.inv(forced)
        if forced.shape[0] == forced.shape[1]
        else jnp.linalg.pinv(forced)
    )
    return B @ Mh, N - dt * (N @ Ustar) @ Mh


def _licq_Ustar(alpha, U0, kap):
    """LICQ CBF-QP alpha forcing; zeros when kap is None."""
    if kap is None:
        return jnp.zeros_like(U0)

    U0flat = U0.reshape(-1)

    def barrier_LICQ(alpha):
        # return jax.jit(jnp.linalg.det, backend='cpu')(alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True))
        return jnp.linalg.det(
            alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True)
        )
        # return jnp.linalg.slogdet(alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True))[1]

    balpha = barrier_LICQ(alpha)
    k = kap * balpha**3

    pLfh, Lfh = jax.jvp(barrier_LICQ, (alpha,), (U0,))
    unroll = lambda v: jax.jvp(
        barrier_LICQ, (alpha,), (v.reshape(alpha.shape),)
    )
    pLgh, Lgh = jax.vmap(unroll)(jnp.eye(alpha.size))

    # Solution to QP
    return jnp.where(
        Lfh + Lgh @ U0flat + k >= 0.0,
        jnp.zeros_like(U0flat),  # constraint inactive
        -(Lfh + Lgh @ U0flat + k) * Lgh.T / (Lgh @ Lgh.T),
    ).reshape(alpha.shape)


class FastlinAdjointEmbedding(ParametricEmbedding):
    def __init__(
        self, sys, alpha_p0, N0, permutation=None, ustars=None, tt=None, kap=None,
        forward_mode: str = "ibp", iterated: bool = False,
    ):
        super().__init__(sys)
        self.Jf_x = jax.jacfwd(sys.olsystem.f, 1)
        self.Jf_u = jax.jacfwd(sys.olsystem.f, 2)
        self.Mf = mjacM(sys.olsystem.f)
        self.alpha_p0 = alpha_p0
        self.N0 = N0
        self.permutation = permutation
        self.ustars = ustars
        self.tt = tt
        self.kap = kap
        self.forward_mode = forward_mode
        self.iterated = iterated

    def _initialize(self, pt0: hParametope) -> ArrayLike:
        if not isinstance(pt0, hParametope):
            raise ValueError(f"{pt0=} is not a hParametope needed for AdjointEmbedding")

        return (self.alpha_p0, self.N0)

    def hypercontrol_shape(self, pt0: hParametope) -> tuple:
        # The hypercontrol is added to the H (alpha) dynamics.
        return pt0.alpha.shape

    def iover(self, state):
        pt, aux = state
        alpha_p, N = aux
        iz = _refine_N(pt.hinv(pt.y), N)
        return interval(alpha_p) @ iz + pt.ox

    def _fastlin_terms(self, pt, alpha_p):
        """CROWN/fastlin pass over the control lifted to z-coordinates."""
        big_iz = pt.hinv(pt.y)

        def lifted_net(z):
            return self.sys.control(alpha_p @ z)

        lifted_net.out_len = self.sys.control.out_len
        lifted_net.u = lambda t, y: lifted_net(y)

        # fastlin_res = fastlin(self.sys.control)(interval(alpha_p)@(big_iz + alpha@ox))
        fastlin_res = fastlin(
            lifted_net, iterated=self.iterated, forward_mode=self.forward_mode,
        )(big_iz + pt.alpha @ pt.ox)
        C = fastlin_res.C
        # C = jax.jacfwd(lifted_net)(alpha@ox)

        big_iu = fastlin_res(big_iz + pt.alpha @ pt.ox)
        return big_iz, fastlin_res, C, big_iu

    def _dynamics(self, t, state, *args, U=None, alpha_pair=None, **kwargs):
        r"""Offset dynamics under the CROWN/fastlin-bounded feedback.

        ``A = J_x + J_u C alpha`` is the closed-loop Jacobian frozen at the
        center and ``(M_x, M_u)`` its mixed-Jacobian enclosure. The offsets grow
        along

            z_dot = alpha_pair [(M_x - J_x) + (M_u - J_u) C alpha] alpha^+ z
                    + alpha_pair M_u (controller residual) + U* alpha^+ z.

        ``alpha_pair`` defaults to ``pt.alpha``; :meth:`_symplectic_step` passes
        ``alpha_next``, the multiplier the exact discrete pairing requires.
        """
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y
        alpha_pair = alpha if alpha_pair is None else alpha_pair

        big_iz, fastlin_res, C, big_iu = self._fastlin_terms(pt, alpha_p)
        ou = self.sys.control(ox)
        centers = (jnp.array([t]), ox, ou) + tuple(arg.center for arg in args)

        J_x, J_u = self.Jf_x(*centers), self.Jf_u(*centers)
        U0 = -alpha @ (J_x + J_u @ C @ alpha)
        Ustar = _licq_Ustar(alpha, U0, self.kap) + _hyper(U, alpha)

        if self.permutation is None:
            self.permutation = Permutation(range(sum(len(c) for c in centers)))
        MM = self.Mf(
            t,
            interval(alpha_p) @ big_iz + ox,
            big_iu,
            *args,
            center=centers,
            permutation=self.permutation,
        )
        Mx, Mu = MM[1], MM[2]
        # Controller linearization residual, in the lifted coordinates.
        ures = fastlin_res.lud + C @ alpha @ ox - ou

        K = len(y) // 2
        mul = jnp.concatenate((-jnp.ones(K), jnp.ones(K)))  # sign of lower offsets
        Jh = natif(jax.jacfwd(lambda z: jnp.asarray(pt.h(z))))

        def F(t, iy, *args):
            iy = _refine_N(iy, N)
            iz = pt.hinv(i2ut(iy) * mul)

            def _ret():
                az = interval(alpha_p) @ iz
                PH = Jh(iz)  # post first-order cancellation
                return interval(PH[len(PH) // 2 :, :]) @ (
                    interval(alpha_pair)
                    @ (
                        ((Mx - J_x) + (Mu - J_u) @ C @ alpha) @ az
                        + interval(Mu) @ ures
                    )
                    + interval(Ustar) @ az
                )

            return jax.lax.cond(
                jnp.any(iy.lower > iy.upper),
                lambda: interval(jnp.zeros_like(iz.lower)),
                _ret,
            )

        E_res = embed(F)(t, y * mul, *args) * mul

        H_dot = U0 + Ustar
        pt_dot = pt.from_parametope(
            hParametope(self.sys.olsystem.f(*centers), H_dot, E_res)
        )
        return pt_dot, (-alpha_p @ H_dot @ alpha_p, -N @ H_dot @ alpha_p)

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y

        _, _, C, _ = self._fastlin_terms(pt, alpha_p)
        ou = self.sys.control(ox)
        centers = (jnp.array([t]), ox, ou) + tuple(arg.center for arg in args)
        # C @ alpha frozen at the old frame: keeps the implicit step a linear solve.
        A = self.Jf_x(*centers) + self.Jf_u(*centers) @ C @ alpha
        Ustar = _licq_Ustar(alpha, -alpha @ A, self.kap) + _hyper(U, alpha)

        # See AdjointEmbedding._symplectic_step: the residual pairs with alpha_NEXT.
        B = jnp.eye(A.shape[0], dtype=alpha.dtype) + dt * A
        alpha_next = jnp.linalg.solve(B.T, (alpha + dt * Ustar).T).T
        forced = None if (U is None and self.kap is None) else alpha + dt * Ustar
        alpha_p_next, N_next = _step_aux(B, alpha_p, N, dt, Ustar, forced)

        pt_dot, _ = self._dynamics(
            t, state, *args, U=U, alpha_pair=alpha_next, **kwargs
        )
        return (
            pt.from_parametope(
                hParametope(ox + dt * pt_dot.ox, alpha_next, y + dt * pt_dot.y)
            ),
            (alpha_p_next, N_next),
        )


def _refine_N(y: Interval, N) -> Interval:
    """Tighten the offset interval ``y`` using the null-space rows ``N``
    (a no-op when ``N`` is empty)."""
    if len(N) > 0:
        refinements = _mat_refine_all(N, jnp.arange(len(y)), y)
        return interval(
            jnp.max(refinements.lower, axis=0),
            jnp.min(refinements.upper, axis=0),
        )
    return y


def _vec_refine(null_vector: jax.Array, var_index: jax.Array, y: Interval):
    ret = icopy(y)

    # Set up linear algebra computations for the refinement
    bounding_vars = interval(null_vector.at[var_index].set(0))
    ref_var = interval(null_vector[var_index])
    b1 = lambda: ((-bounding_vars @ ret) / ref_var) & ret[var_index]
    b2 = lambda: ret[var_index]

    # Compute refinement based on null vector, if possible
    ndb0 = jnp.abs(null_vector[var_index]) > 1e-10
    ret = jax.lax.cond(ndb0, b1, b2)

    # fix fpe problem with upper < lower
    retu = jnp.where(ret.upper >= ret.lower, ret.upper, ret.lower)
    return interval(ret.lower, retu)


_mat_refine = jax.vmap(_vec_refine, in_axes=(0, None, None), out_axes=0)
_mat_refine_all = jax.vmap(_mat_refine, in_axes=(None, 0, None), out_axes=1)
