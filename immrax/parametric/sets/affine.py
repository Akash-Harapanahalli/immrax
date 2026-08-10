from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from ...embedding import embed
from ...inclusion import (
    Interval,
    Permutation,
    icopy,
    i2ut,
    interval,
    jacM,
    mjacM,
    natif,
)
from ...neural import fastlin
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from functools import partial

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
        kap: float = 0.1,
        permutation=None,
        disable_adjoint=False,
    ):
        #  refine_factory:Callable[[ArrayLike], Callable]=partial(SampleRefinement, num_samples=10)):
        super().__init__(sys)
        self.Jf_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.Jf = jacM(sys.f)
        self.kap = kap
        self.alpha_p0 = alpha_p0
        self.N0 = N0
        self.permutation = permutation
        self.disable_adjoint = disable_adjoint

    def _initialize(self, pt0: hParametope) -> ArrayLike:
        if not isinstance(pt0, hParametope):
            raise ValueError(f"{pt0=} is not a hParametope needed for AdjointEmbedding")

        # Setup refinement
        alpha = pt0.alpha
        # _id = lambda x : x
        # self.refine = self.refine_factory(alpha).get_refine_func() if alpha.shape[0] > alpha.shape[1] else _id
        # alpha_p = jnp.linalg.pinv(alpha)
        # N = null_space(alpha.T)
        # print(N@alpha)
        # return (alpha_p, N)
        return (self.alpha_p0, self.N0)

    def hypercontrol_shape(self, pt0: hParametope) -> tuple:
        # The hypercontrol is added to the H (alpha) dynamics.
        return pt0.alpha.shape

    def iover(self, state):
        pt, aux = state
        alpha_p, N = aux
        iz = _refine_N(pt.hinv(pt.y), N)
        return interval(alpha_p) @ iz + pt.ox

    def _dynamics(
        self, t, state: Tuple[hParametope, ArrayLike], *args,
        U=None, alpha_dot=None, **kwargs,
    ):
        pt, aux = state
        ox = pt.ox

        # jax.debug.print('args={args}', args=args)

        K = len(pt.y) // 2
        # ly = -pt.y[:K] # negative for lower bound
        # uy = pt.y[K:]
        # iy = lu2i(ly, uy)
        y = pt.y
        alpha = pt.alpha
        alpha_p, N = aux

        # alpha_p = jnp.linalg.inv(alpha)

        ## Adjoint dynamics + LICQ CBF

        args_centers = tuple(arg.center for arg in args)
        centers = (jnp.array([t]), ox) + args_centers

        J = self.Jf_x(*centers)
        if not self.disable_adjoint:
            u0 = -alpha @ J
        else:
            u0 = jnp.zeros_like(-alpha @ J)

        u0flat = u0.reshape(-1)

        # CBF: Enforce pairwise independence on the rows of alpha

        PENALTY = 0

        if PENALTY == 0:
            ustar = jnp.zeros_like(u0)

        elif PENALTY == 1:

            def soft_overmax(x, eps=1e-5):
                return jnp.max(jnp.exp(x) / jnp.sum(jnp.exp(x)))

            def barrier_LICQ(alpha):
                # Normalize rows of alpha
                return jnp.linalg.det(
                    alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True)
                )
                # return jnp.linalg.slogdet(alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True))[1]

            balpha = barrier_LICQ(alpha)
            k = self.kap * balpha**3

            pLfh, Lfh = jax.jvp(barrier_LICQ, (alpha,), (u0,))
            unroll = lambda v: jax.jvp(
                barrier_LICQ, (alpha,), (v.reshape(alpha.shape),)
            )
            pLgh, Lgh = jax.vmap(unroll)(jnp.eye(alpha.size))

            # Solution to QP
            ustar = jnp.where(
                Lfh + Lgh @ u0flat + k >= 0.0,
                jnp.zeros_like(u0flat),  # constraint inactive
                -(Lfh + Lgh @ u0flat + k) * Lgh.T / (Lgh @ Lgh.T),
            ).reshape(alpha.shape)
        elif PENALTY == 2:
            ustar = jnp.zeros_like(u0)

            def soft(H):
                HHT = H @ H.T
                return jnp.sum((HHT - jnp.eye(H.shape[0])) ** 2)

            ustar = -self.kap * jax.grad(soft)(alpha)

        # Hypercontrol enters exactly like the CBF term ustar: it perturbs H_dot
        # AND must appear in the offset-growth bound below, so the enclosure stays
        # sound. Fold it into ustar so every use of ustar accounts for it.
        Uctrl = jnp.zeros_like(alpha) if U is None else jnp.asarray(U).reshape(alpha.shape)
        ustar = ustar + Uctrl

        if alpha_dot is not None:
            # Symplectic path: H_dot == alpha_dot below, and the offset bound
            # sees alpha@(Mx - J) + ustar == alpha@Mx + alpha_dot.
            ustar = alpha_dot - u0

        ## Offset Dynamics given ustar

        MJACM = True

        # For properly handling signs in lower offsets
        K_2 = len(y) // 2
        mul = jnp.concatenate((-jnp.ones(K_2), jnp.ones(K_2)))
        big_iz = pt.hinv(pt.y)
        Jh = natif(jax.jacfwd(lambda z: jnp.asarray(pt.h(z))))

        refine = lambda y: _refine_N(y, N)

        if MJACM:
            if self.permutation is None:
                lenperm = sum([len(arg) for arg in centers])
                self.permutation = Permutation(range(lenperm))

            MM = self.Mf(
                t,
                interval(alpha_p) @ big_iz + ox,
                *args,
                center=centers,
                permutation=self.permutation,
            )
            ls = []
            us = []

            # jax.debug.print('{a}, {b}, {c}', a=len(MM[2:]), b=len(args), c=len(args_centers))
            # jax.debug.print('{a}', a=len(args_centers))
            for M, arg, cent in zip(MM[2:], args, args_centers):
                term = interval(M) @ (arg - cent)
                ls.append(term.lower)
                us.append(term.upper)

            dist = interval(jnp.sum(jnp.asarray(ls)), jnp.sum(jnp.asarray(us)))
            # jax.debug.print('dist={dist}', dist=dist)

            def F(t, iy, *args):
                iy = refine(iy)
                iz = pt.hinv(i2ut(iy) * mul)

                # _, iJx = self.Jf(interval(t), interval(Hp)@iz + ox, *args)
                # MM = self.Mf(t, interval(alpha_p)@big_iz + ox, *args, \
                #             centers=(centers,), permutations=self.permutation)[0]
                Mx = MM[1]

                # empty = jnp.any(iy.lower > iy.upper)
                # empty = jnp.ones_like(iy.lower).astype(bool)
                empty = False

                def _zero():
                    return interval(jnp.zeros_like(iz.lower))

                def _ret():
                    # Post first order cancellation
                    PH = Jh(iz)
                    if not self.disable_adjoint:
                        return interval(PH[len(PH) // 2 :, :]) @ (
                            (interval(alpha) @ (Mx - J) + ustar)
                            @ (interval(alpha_p) @ iz)
                            + dist
                        )
                    else:
                        return interval(PH[len(PH) // 2 :, :]) @ (
                            (interval(alpha) @ (Mx) + ustar) @ (interval(alpha_p) @ iz)
                            + dist
                        )

                return jax.lax.cond(empty, _zero, _ret)

            E = embed(F)
        else:

            def F_second(t, iy, *args):
                iy = refine(iy)
                iz = pt.hinv(i2ut(iy) * mul)

                empty = jnp.any(iy.lower > iy.upper)

                def _zero():
                    return interval(jnp.zeros_like(iz.lower))

                def _ret():
                    def _get_second(oz, z):
                        primals = (t, alpha_p @ oz + ox)
                        series = (
                            (0.0, 0.0),
                            (alpha_p @ z, jnp.zeros_like(alpha_p @ z)),
                        )
                        _, coeffs = jet(self.sys.f, primals, series)
                        return coeffs[1]

                    res = natif(_get_second)(big_iz, iz)

                    # Post first order cancellation
                    PH = Jh(iz)
                    return interval(PH[len(PH) // 2 :, :]) @ (
                        interval(ustar) @ alpha_p @ iz + interval(alpha) @ res
                    )

                return jax.lax.cond(empty, _zero, _ret)

            E = embed(F_second)

        E_res = E(t, y * mul, *args) * mul

        # H_dot includes the hypercontrol (folded into ustar above); propagate it
        # through H+ and N so alpha_p @ alpha = I and N @ alpha = 0 are preserved.
        H_dot = u0 + ustar
        pt_dot = pt.from_parametope(hParametope(self.sys.f(*centers), H_dot, E_res))
        alpha_p_dot = -alpha_p @ H_dot @ alpha_p
        N_dot = -N @ H_dot @ alpha_p

        return (pt_dot, (alpha_p_dot, N_dot))

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y

        centers = (jnp.array([t]), ox) + tuple(arg.center for arg in args)
        J = self.Jf_x(*centers)
        A = jnp.zeros_like(J) if self.disable_adjoint else J
        # PENALTY == 0 in _dynamics => the CBF part of ustar is identically zero;
        # only the hypercontrol is explicit forcing. If PENALTY is ever
        # re-enabled there, replicate the CBF term here.
        ustar = jnp.zeros_like(alpha) if U is None else jnp.asarray(U).reshape(alpha.shape)

        B = jnp.eye(A.shape[0], dtype=alpha.dtype) + dt * A
        alpha_next = jnp.linalg.solve(B.T, (alpha + dt * ustar).T).T
        alpha_dot = (alpha_next - alpha) / dt

        pt_dot, _ = self._dynamics(t, state, *args, U=U, alpha_dot=alpha_dot, **kwargs)

        if U is None:
            # ustar == 0: alpha_next = alpha @ B^{-1}, so these are exact.
            alpha_p_next = B @ alpha_p
            N_next = N
        else:
            # Exact updates preserving alpha_p @ alpha = I and N @ alpha = 0.
            # inv (LU) when square: cheaper and better-behaved derivatives than
            # SVD-pinv under the iLQR backward pass.
            forced = alpha + dt * ustar
            Mh = (
                jnp.linalg.inv(forced)
                if forced.shape[0] == forced.shape[1]
                else jnp.linalg.pinv(forced)
            )
            alpha_p_next = B @ Mh
            N_next = N - dt * (N @ ustar) @ Mh

        pt_next = pt.from_parametope(
            hParametope(ox + dt * pt_dot.ox, alpha_next, y + dt * pt_dot.y)
        )
        return pt_next, (alpha_p_next, N_next)


def _licq_ustar(alpha, u0, kap):
    """LICQ CBF-QP alpha forcing; zeros when kap is None."""
    if kap is None:
        return jnp.zeros_like(u0)

    u0flat = u0.reshape(-1)

    def barrier_LICQ(alpha):
        # return jax.jit(jnp.linalg.det, backend='cpu')(alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True))
        return jnp.linalg.det(
            alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True)
        )
        # return jnp.linalg.slogdet(alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True))[1]

    balpha = barrier_LICQ(alpha)
    k = kap * balpha**3

    pLfh, Lfh = jax.jvp(barrier_LICQ, (alpha,), (u0,))
    unroll = lambda v: jax.jvp(
        barrier_LICQ, (alpha,), (v.reshape(alpha.shape),)
    )
    pLgh, Lgh = jax.vmap(unroll)(jnp.eye(alpha.size))

    # Solution to QP
    return jnp.where(
        Lfh + Lgh @ u0flat + k >= 0.0,
        jnp.zeros_like(u0flat),  # constraint inactive
        -(Lfh + Lgh @ u0flat + k) * Lgh.T / (Lgh @ Lgh.T),
    ).reshape(alpha.shape)


class FastlinAdjointEmbedding(ParametricEmbedding):
    def __init__(
        self, sys, alpha_p0, N0, permutation=None, ustars=None, tt=None, kap=None,
        forward_mode: str = "ibp", iterated: bool = False,
    ):
        #  refine_factory:Callable[[ArrayLike], Callable]=partial(SampleRefinement, num_samples=10)):
        super().__init__(sys)
        self.Jf_x = jax.jacfwd(sys.olsystem.f, 1)
        self.Jf_u = jax.jacfwd(sys.olsystem.f, 2)
        self.Mf = mjacM(sys.olsystem.f)
        self.Jf = jacM(sys.olsystem.f)
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

        # Setup refinement
        # alpha = pt0.alpha
        # _id = lambda x : x
        # self.refine = self.refine_factory(alpha).get_refine_func() if alpha.shape[0] > alpha.shape[1] else _id
        # alpha_p = jnp.linalg.pinv(alpha)
        # N = null_space(alpha.T)
        # print(N@alpha)
        # return (alpha_p, N)
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

    def _dynamics(
        self, t, state: Tuple[hParametope, ArrayLike], *args,
        U=None, alpha_dot=None, **kwargs,
    ):
        pt, aux = state
        ox = pt.ox

        K = len(pt.y) // 2
        # ly = -pt.y[:K] # negative for lower bound
        # uy = pt.y[K:]
        # iy = lu2i(ly, uy)
        y = pt.y
        alpha = pt.alpha
        alpha_p, N = aux

        # Global fastlin

        Jh = natif(jax.jacfwd(lambda z: jnp.asarray(pt.h(z))))
        big_iz, fastlin_res, C, big_iu = self._fastlin_terms(pt, alpha_p)
        CHox = C @ alpha @ ox
        ou = self.sys.control(ox)

        ## Adjoint dynamics + LICQ CBF

        args_centers = (arg.center for arg in args)
        centers = (jnp.array([t]), ox, ou) + tuple(args_centers)

        J_x = self.Jf_x(*centers)
        J_u = self.Jf_u(*centers)
        # u0 = -alpha@(J_x@alpha_p + J_u@C)
        u0 = -alpha @ (J_x + J_u @ C @ alpha)

        ustar = _licq_ustar(alpha, u0, self.kap)

        # Hypercontrol enters exactly like the CBF term ustar: it perturbs H_dot
        # AND must appear in the offset-growth bound below, so the enclosure stays
        # sound. Fold it into ustar so every use of ustar accounts for it.
        Uctrl = jnp.zeros_like(alpha) if U is None else jnp.asarray(U).reshape(alpha.shape)
        ustar = ustar + Uctrl

        if alpha_dot is not None:
            # Symplectic path: H_dot == alpha_dot below, and the offset bound
            # sees alpha@(Mx + Mu@C@alpha) + alpha_dot.
            ustar = alpha_dot - u0

        ## Offset Dynamics given ustar

        # For properly handling signs in lower offsets
        K_2 = len(y) // 2
        mul = jnp.concatenate((-jnp.ones(K_2), jnp.ones(K_2)))

        refine = lambda y: _refine_N(y, N)

        if self.permutation is None:
            lenperm = sum([len(arg) for arg in centers])
            self.permutation = Permutation(range(lenperm))

        MM = self.Mf(
            t,
            interval(alpha_p) @ big_iz + ox,
            big_iu,
            *args,
            center=centers,
            permutation=self.permutation,
        )

        ls = []
        us = []

        def F(t, iy, *args):
            # Bound outputs along h^{-1}([iy.lower, iy.upper])

            iy = refine(iy)
            iz = pt.hinv(i2ut(iy) * mul)

            Mx = MM[1]
            Mu = MM[2]

            empty = jnp.any(iy.lower > iy.upper)

            def _zero():
                return interval(jnp.zeros_like(iz.lower))

            def _ret():
                # Post first order cancellation
                PH = Jh(iz)
                return interval(PH[len(PH) // 2 :, :]) @ (
                    interval(alpha)
                    @ (
                        ((Mx - J_x) + (Mu - J_u) @ C @ alpha) @ alpha_p @ iz
                        + interval(Mu) @ (fastlin_res.lud + CHox - ou)
                    )
                    + interval(ustar) @ alpha_p @ iz
                )

            return jax.lax.cond(empty, _zero, _ret)

        E = embed(F)
        E_res = E(t, y * mul, *args) * mul

        # H_dot includes the hypercontrol (folded into ustar above); propagate it
        # through H+ and N so alpha_p @ alpha = I and N @ alpha = 0 are preserved.
        H_dot = u0 + ustar
        pt_dot = pt.from_parametope(
            hParametope(self.sys.olsystem.f(*centers), H_dot, E_res)
        )
        alpha_p_dot = -alpha_p @ H_dot @ alpha_p
        N_dot = -N @ H_dot @ alpha_p

        return (pt_dot, (alpha_p_dot, N_dot))

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        pt, (alpha_p, N) = state
        alpha, ox, y = pt.alpha, pt.ox, pt.y

        _big_iz, _flr, C, _big_iu = self._fastlin_terms(pt, alpha_p)
        ou = self.sys.control(ox)
        centers = (jnp.array([t]), ox, ou) + tuple(arg.center for arg in args)
        J_x = self.Jf_x(*centers)
        J_u = self.Jf_u(*centers)

        # Closed-loop adjoint matrix with C@alpha FROZEN at the old point:
        # linearizes the alpha-dependence so the implicit step stays a linear solve.
        A = J_x + J_u @ C @ alpha
        u0 = -alpha @ A
        ustar = _licq_ustar(alpha, u0, self.kap)
        if U is not None:
            ustar = ustar + jnp.asarray(U).reshape(alpha.shape)

        B = jnp.eye(A.shape[0], dtype=alpha.dtype) + dt * A
        alpha_next = jnp.linalg.solve(B.T, (alpha + dt * ustar).T).T
        alpha_dot = (alpha_next - alpha) / dt

        pt_dot, _ = self._dynamics(t, state, *args, U=U, alpha_dot=alpha_dot, **kwargs)

        if U is None and self.kap is None:
            # ustar == 0: alpha_next = alpha @ B^{-1}, so these are exact.
            alpha_p_next = B @ alpha_p
            N_next = N
        else:
            # Exact updates preserving alpha_p @ alpha = I and N @ alpha = 0.
            # inv (LU) when square: cheaper and better-behaved derivatives than
            # SVD-pinv under the iLQR backward pass.
            forced = alpha + dt * ustar
            Mh = (
                jnp.linalg.inv(forced)
                if forced.shape[0] == forced.shape[1]
                else jnp.linalg.pinv(forced)
            )
            alpha_p_next = B @ Mh
            N_next = N - dt * (N @ ustar) @ Mh

        pt_next = pt.from_parametope(
            hParametope(ox + dt * pt_dot.ox, alpha_next, y + dt * pt_dot.y)
        )
        return pt_next, (alpha_p_next, N_next)


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
