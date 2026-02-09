import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import TaylorModel, TaylorPolynomial, nattm, nattp
from .base import TMFlowpipeGenerator

from typing import Tuple


def tx_tm_eval(tm: TaylorModel, t: float, t_order: int) -> TaylorModel:
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

    L = (t_order + 1) // tm.exponents.shape[0]

    exponents = tm.exponents[1:, 0:L]
    coeffs_list = [tm.coeffs[1:, L * i : L * (i + 1)] for i in range(t_order + 1)]
    coeffs = jnp.sum(
        jnp.asarray(coeffs_list) * (t ** jnp.arange(t_order + 1)),
        axis=0,
    )

    return TaylorModel(
        coeffs=coeffs,
        exponents=exponents,
        remainder=tm.remainder[1:],
        flat_domain=tm.flat_domain[1:],
        flat_center=tm.flat_center[1:],
        _domain_treedef=tm._domain_treedef.children()[1],
        _leaf_shapes=tm._leaf_shapes[1:],
        _per_leaf_order=tm._per_leaf_order[1:],
    )

def 


class BasicTMFlowpipeGenerator(TMFlowpipeGenerator):
    """A basic implementation of a Picard-based flowpipe generator."""

    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""
        self.t_order = t_order
        self._prolonged_f = prolongation(self.sys.f, t_order)

        # Parameters for the eps-delta inflation
        self.eps = kwargs.get("eps", 1e-2)
        self.delta = kwargs.get("delta", 1e-2)

    def _step(self, t: float, tmi: TaylorModel, dt_max: float, **kwargs):
        # Step 1: Compute the Taylor expansion of the flow map to t_order, x_order
        # TODO: Fix the nattp module and change center propagation

        # poly = nattm(self._prolonged_f, structured_center=True)(tmi).polynomial

        tt = jnp.arange(self.t_order + 1) * inv_fact(jnp.arange(self.t_order + 1))
        poly = nattp(get_poly)(tmi.polynomial)
