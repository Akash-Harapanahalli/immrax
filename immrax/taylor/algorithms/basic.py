import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import TaylorModel, TaylorPolynomial, nattm, nattp
from .base import TMFlowpipeGenerator

from typing import Tuple


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

        def get_poly(x):
            # Compute the prolonged f at t, x
            coeffs = self._prolonged_f(t, x)
            return jnp.sum(coeffs * tt)

        poly = nattp(get_poly)(tmi.polynomial)
