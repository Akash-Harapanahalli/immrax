"""Shared utilities for validated ODE integration algorithms."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval
from immrax.utils import inv_fact


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ValidatedSolution:
    """Result of validated ODE integration.

    Attributes
    ----------
    times : jax.Array
        Time grid ``[t_0, t_1, …, t_N]``, shape ``(maxsteps+1,)``.
        Entries beyond ``nsteps`` are padded with ``tmax``.
    flowpipe_lo : jax.Array
        Lower bounds of interval enclosures, shape ``(maxsteps+1, n)``.
    flowpipe_hi : jax.Array
        Upper bounds of interval enclosures, shape ``(maxsteps+1, n)``.
    nsteps : jax.Array
        Number of real integration steps taken (scalar int).
    success : jax.Array
        ``True`` if every step was validated (scalar bool).
    """

    def __init__(self, times, flowpipe_lo, flowpipe_hi, nsteps, success):
        self.times = times
        self.flowpipe_lo = flowpipe_lo
        self.flowpipe_hi = flowpipe_hi
        self.nsteps = nsteps
        self.success = success

    def tree_flatten(self):
        return ((self.times, self.flowpipe_lo, self.flowpipe_hi,
                 self.nsteps, self.success), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @property
    def flowpipe(self):
        """Return list of Interval objects for backwards compatibility."""
        n = int(self.nsteps) + 1
        return [interval(self.flowpipe_lo[i], self.flowpipe_hi[i])
                for i in range(n)]


jax.tree_util.register_pytree_node_class(ValidatedSolution)


# ---------------------------------------------------------------------------
# Step-size selection (JAX-traceable)
# ---------------------------------------------------------------------------

def _step_size(derivs: list, abstol, order: int):
    """Compute step size from Taylor coefficient norms.

    Uses the last two coefficients (orders ``p-1`` and ``p``) following
    the standard TaylorIntegration heuristic::

        h = min_k (abstol / ||x_k / k!||_∞)^{1/k}
    """
    h = jnp.inf
    for k in [order - 1, order]:
        if 0 < k:
            coeff_norm = jnp.max(jnp.abs(derivs[k])) * inv_fact(k)
            hk = jnp.where(coeff_norm > 0,
                           (abstol / coeff_norm) ** (1.0 / k),
                           jnp.inf)
            h = jnp.minimum(h, hk)
    return h


# ---------------------------------------------------------------------------
# Taylor polynomial evaluation
# ---------------------------------------------------------------------------

def _eval_taylor(derivs: list, h, order: int):
    """Evaluate ``Σ x^(k) h^k / k!`` via Horner's method (array output)."""
    result = derivs[order] * inv_fact(order)
    for k in range(order - 1, -1, -1):
        result = derivs[k] * inv_fact(k) + h * result
    return result


def _eval_taylor_interval(derivs: list, h_iv: Interval, order: int) -> Interval:
    """Evaluate the Taylor polynomial with an *interval* time step.

    ``derivs`` are plain arrays (evaluated at the centre); ``h_iv`` is an
    interval (typically ``[0, h]``).
    """
    result = interval(derivs[order] * inv_fact(order))
    for k in range(order - 1, -1, -1):
        coeff = interval(derivs[k] * inv_fact(k))
        result = coeff + h_iv * result
    return result


# ---------------------------------------------------------------------------
# Contractivity check (JAX-traceable, returns JAX bool)
# ---------------------------------------------------------------------------

def _is_contractive(delta: Interval, delta_x: Interval):
    """Check ``delta ⊂ int(delta_x)`` component-wise."""
    strict = (jnp.all(delta.lower > delta_x.lower) &
              jnp.all(delta.upper < delta_x.upper))
    both_zero = (jnp.all(delta.lower == 0) & jnp.all(delta.upper == 0) &
                 jnp.all(delta_x.lower == 0) & jnp.all(delta_x.upper == 0))
    return strict | both_zero
