import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif, icentpert
from immrax.system import System
from immrax.utils import inv_fact, prolongation, check_containment
from .. import (
    TaylorModel,
    pjetm,
    pjet,
    tm_integrate_variable,
    taylor_model_concatenate,
    taylor_model_identity,
)
from .base import TMFlowpipeGenerator, tx_tm_eval, tps_to_tx

from typing import Tuple


class BasicTMFlowpipeGenerator(TMFlowpipeGenerator):
    """A basic implementation of a Picard-based flowpipe generator."""

    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps=4096, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""
        self.t_order = t_order
        self._prolonged_f = prolongation(self.sys.f, t_order)

        # Parameters for the eps-delta inflation
        self.eps = kwargs.get("eps", 1e-2)
        self.delta = kwargs.get("delta", 1e-2)

    def _picard(self, tm_tx: TaylorModel, ic_remainder=None, u=None) -> TaylorModel:
        """Apply the Picard operator to the TaylorModel tm_tx over the domain of the TaylorModel.

        The Picard operator is defined as:
        K : C([t0,t1], R^n) -> C([t0,t1], R^n),
        K(x, t) = x(t0) + \\int_{t0}^t f(x(s), s) ds

        Applied to a TaylorModel tm_tx, this operator uses TaylorModel arithmetic to compute:
        K(p + E) = (q + J) + \\int_{t0}^t f(p + E, s) ds

        As a special case, if the polynomial part of the TaylorModel is the Taylor series expansion
        of the flow map, the polynomial part is a fixed point (to the corresponding order), meaning
        K(p + E) = p + E'
        where E' contains all the remainders.

        Parameters
        ----------
        tm_tx : TaylorModel
            The tube TaylorModel to apply the Picard operator to.
        ic_remainder : Interval, optional
            The remainder from the initial condition (previous step's spatial
            TM remainder).  If provided, this is added to the returned
            remainder to account for IC uncertainty.  If None, the IC
            remainder from ``tx_tm_eval`` at ``t0`` is used (which equals
            the tube's own remainder — correct only when E_prev = 0).
        u : optional
            Control input.
        """
        # Initial condition
        tm_ic = tx_tm_eval(tm_tx, tm_tx.domain[0].lower, self.t_order)

        # pjetm now handles multi-arg: pass t identity and tm_tx as separate args
        tm_id = taylor_model_identity(tm_tx.domain, order=tm_tx._per_leaf_order)
        if u is not None:
            f_tm_tx = pjetm(self.sys.f)(tm_id[0:1], tm_tx, u)
        else:
            f_tm_tx = pjetm(self.sys.f)(tm_id[0:1], tm_tx)
        tm_int = tm_integrate_variable(f_tm_tx, var_idx=0, keep_order=True)

        con_sh = tm_ic.coeffs.shape

        coeffs = tm_int.coeffs.at[: con_sh[0], : con_sh[1]].add(tm_ic.coeffs)

        if ic_remainder is None:
            ic_remainder = tm_ic.remainder

        return TaylorModel(
            coeffs=coeffs,
            exponents=tm_int.exponents,
            remainder=tm_int.remainder + ic_remainder,
            flat_domain=tm_int.flat_domain,
            flat_center=tm_int.flat_center,
            _input_pytree=tm_int._input_pytree,
            _output_pytree=tm_int._output_pytree,
            _per_leaf_order=tm_int._per_leaf_order,
        )

    def _step(self, t: float, tmi: TaylorModel, dt_max: float, **kwargs):
        u = kwargs.get("u", None)

        # Step 1: Compute the Taylor expansion of the flow map to t_order, x_order

        if u is not None:
            poly_coeffs = pjet(lambda x: self._prolonged_f(t, x, u))(tmi.polynomial)
        else:
            poly_coeffs = pjet(lambda x: self._prolonged_f(t, x))(tmi.polynomial)
        poly = tps_to_tx(
            poly_coeffs,
            # tmi.remainder,
            interval(jnp.zeros_like(tmi.remainder.lower)),
            (interval(t, t + dt_max), tmi.domain),
            (t, tmi.center),
            per_leaf_order=(self.t_order,) + tmi._per_leaf_order,
        )

        # Step 2: eps-inflation to check the contraction of the Picard operator
        # rem will start at tmi.remainder

        def _remainder_inflation(rem):
            return rem * icentpert(1.0, self.eps) + icentpert(0.0, self.delta)

        def _check_picard(i, carry):
            poly, contractive = carry

            # Noop if contractive using jax.lax.cond
            poly.remainder = jax.lax.cond(
                contractive,
                lambda rem: rem,
                lambda rem: _remainder_inflation(rem),
                poly.remainder,
            )
            contractive = jax.lax.cond(
                contractive,
                lambda: contractive,
                lambda: (
                    check_containment(
                        self._picard(poly, ic_remainder=tmi.remainder, u=u).remainder,
                        poly.remainder,
                    )
                    == 1
                ),
            )

            return (poly, contractive)

        poly, contractive = jax.lax.fori_loop(0, 1000, _check_picard, (poly, False))

        return dt_max, tx_tm_eval(poly, t + dt_max, self.t_order), poly, contractive
