"""Shared utilities for validated ODE integration algorithms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod

from immrax.inclusion import Interval, interval
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import TaylorModel

from typing import Tuple


@register_pytree_node_class
class TMFlowpipe:
    """Result of validated ODE integration.

    Each TaylorModel in the flowpipe has domain variables (t, x),
    where t\in [t_i, t_{i+1}] is expanded around t_i and
    x is expanded around the nominal polynomial approximation of the solution.

    Attributes
    ----------
    times : list[float]
        Time grid ``[t_0, t_1, …, t_N]``.
    tms : list[TaylorModel]
        TaylorModel at each step.  Domain variables are ``(dt, x_0)``.
    nsteps : int
        Number of real integration steps taken.
    success : bool
        ``True`` if every step was validated.
    """

    def __init__(self, times, tms, nsteps, success):
        self.times = times
        self.tms = tms
        self.nsteps = nsteps
        self.success = success

    def tree_flatten(self):
        return (self.tms,), {
            "times": self.times,
            "nsteps": self.nsteps,
            "success": self.success,
        }

    @classmethod
    def tree_unflatten(cls, aux, children):
        (tms,) = children
        return cls(aux["times"], tms, aux["nsteps"], aux["success"])


class TMFlowpipeGenerator(ABC):
    """Abstract base class for TaylorModel flowpipe generators.

    These collections of algorithms generically use the following steps to generate a validated flowpipe:

    1. Take in an initial TaylorModel (usually identity), a largest dt, and an interval [t_0, t_f]
    2. The taylor expansion of the solution is computed in (t,x) around (t_i,x_i) using either:
        a. Lie derivatives of the system vector field
        b. Iterated Picard iteration
    3. Verify the contraction of a Picard operator. If not contractive either:
        a. Increase the remainder portion of the TaylorModel
        b. Reduce the step size dt
    """

    sys: System

    def __init__(self, sys: System):
        self.sys = sys

    @abstractmethod
    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""

    @abstractmethod
    def _step(
        self, t, tmi, dt_max, **kwargs
    ) -> Tuple[float, TaylorModel, TaylorModel, bool]:
        """Take one validated step.

        Parameters
        ----------
        tmi : TaylorModel
            Time-independent TaylorModel at the start of the step.
        dt_max : float
            Maximum allowed step size.

        Returns
        -------
        dt : float
            Actual step size taken.
        tmf : TaylorModel
            Time-independent TaylorModel at the end of the step (at t = tmi + dt)
        tm_tube : TaylorModel
            Time-dependent validated TaylorModel over [tmi, tmi + dt]
        success : bool
            ``True`` if the step was validated.
        """

    def generate_flowpipe(
        self, t0, tf, tm0, dt_max, *, t_order=4, max_steps=4096, **kwargs
    ):
        """Generate a validated flowpipe for the given system.

        Parameters
        ----------
        t0 : float
            Initial time.
        tf : float
            Final time.
        tm0 : TaylorModel
            Initial TaylorModel.
        dt_max : float
            Maximum allowed step size.
        max_steps : int, optional
            Maximum number of steps, by default 4096.
        **kwargs
            Keyword arguments to pass to the step function.

        Returns
        -------
        TMFlowpipe
            Validated flowpipe.
        """

        self._initialize(
            t0, tf, tm0, dt_max, t_order=t_order, max_steps=max_steps, **kwargs
        )

        # Use jax.lax.scan to iterate through steps, noop when t_f is reached or success is False

        def scan_fn(carry, _):
            tmi, ti, success, steps_taken = carry

            noop_cond = jnp.logical_and(success, ti < tf)

            def noop():
                return (tmi, ti, success, steps_taken), (tmi, jnp.nan)

            def take_step():
                dt, tmf, tm_tube, success = self._step(ti, tmi, dt_max, **kwargs)
                return (ti + dt, tmf, success, steps_taken + 1), (ti + dt, tm_tube)

            return jax.lax.cond(noop_cond, take_step, noop)

        (final_t, final_tm, final_success, final_steps), (times, tms) = jax.lax.scan(
            scan_fn, (t0, tm0, True, 0), jnp.arange(max_steps)
        )

        return TMFlowpipe(times, tms, final_steps, final_success)
