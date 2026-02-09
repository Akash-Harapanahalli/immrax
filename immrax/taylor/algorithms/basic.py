import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import taylor_model, TaylorModel, TaylorPolynomial, nattm, nattp
from .base import TMFlowpipeGenerator

from typing import Tuple


def tx_tm_eval(tm: TaylorModel, t: float) -> TaylorModel:
    """Evaluate a time-dependent Taylor Model at a given time.

    Parameters
    ----------
    tm : TaylorModel
        The time-dependent Taylor Model to evaluate.
    t : float
        The time at which to evaluate the Taylor Model.
    t_order : int
        The order of the Taylor expansion.

    Returns
    -------
    TaylorModel
        The spatial Taylor Model extracted from evaluation at time t.
    """

    # Since t is always the first variable, the exponent matrix is structured
    # in [0,0,0,...,1,1,1,...,t_order,t_order,t_order]
    # i.e., tm.exponents[1:, L * i : L * (i + 1)] is the same for every i

    t_order = tm.exponents[0, -1]
    L = tm.exponents.shape[1] // (t_order + 1)

    exponents = tm.exponents[1:, 0:L]
    coeffs_list = jnp.split(tm.coeffs, t_order + 1, axis=1)
    coeffs = jnp.sum(
        jnp.asarray(coeffs_list) * (t ** jnp.arange(t_order + 1))[:, None, None],
        axis=0,
    )

    return TaylorModel(
        coeffs=coeffs,
        exponents=exponents,
        remainder=tm.remainder,
        flat_domain=tm.flat_domain[1:],
        flat_center=tm.flat_center[1:],
        _domain_treedef=tm._domain_treedef.children()[1],
        _leaf_shapes=tm._leaf_shapes[1:],
        _per_leaf_order=tm._per_leaf_order[1:],
    )


def tps_to_tx(
    polys: list[TaylorPolynomial], remainder, domain, center
) -> TaylorPolynomial:
    """Convert a list of TaylorPolynomials to a single TaylorPolynomial with canonical
    coefficient structure, over the domain

    Parameters
    ----------
    polys : list[TaylorPolynomial]
        List of TaylorPolynomials to convert.

    Returns
    -------
    TaylorPolynomial
        The TaylorPolynomial with canonical coefficient structure.
    """
    t_order = len(polys) - 1
    mon_len = polys[0].exponents.shape[-1]
    coeffs = jnp.concatenate(
        [polys[i].coeffs * inv_fact(i) for i in range(t_order + 1)], axis=1
    )
    exponents_top = jnp.repeat(jnp.arange(t_order + 1), mon_len)
    exponents_bottom = jnp.concatenate([p.exponents for p in polys], axis=1)

    return taylor_model(
        coeffs=coeffs,
        exponents=jnp.vstack((exponents_top, exponents_bottom)),
        remainder=remainder,
        domain=domain,
        center=center,
    )


class BasicTMFlowpipeGenerator(TMFlowpipeGenerator):
    """A basic implementation of a Picard-based flowpipe generator."""

    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps=4096, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""
        self.t_order = t_order
        self._prolonged_f = prolongation(self.sys.f, t_order)

        # Parameters for the eps-delta inflation
        self.eps = kwargs.get("eps", 1e-2)
        self.delta = kwargs.get("delta", 1e-2)

    def _step(self, t: float, tmi: TaylorModel, dt_max: float, **kwargs):
        # Step 1: Compute the Taylor expansion of the flow map to t_order, x_order

        poly_coeffs = nattp(lambda x: self._prolonged_f(t, x))(tmi.polynomial)
        poly = tps_to_tx(
            poly_coeffs,
            tmi.remainder,
            (interval(jnp.asarray(t), jnp.asarray(t + dt_max)), tmi.domain),
            (t, tmi.center),
        )
        return poly
