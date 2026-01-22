"""Base classes and utilities for reachability algorithms.

This module provides the foundational classes and helper functions used by
all reachability algorithms.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.jet import jet
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ...system import System
from ...inclusion import Interval, natif
from ..sets import Zonotope


# --- Helper Functions ---

def fact(n):
    """Compute factorial using log-gamma for numerical stability."""
    return lax.exp(lax.lgamma(n + 1.))


def inv_fact(n):
    """Compute inverse factorial using log-gamma for numerical stability."""
    return lax.exp(-lax.lgamma(n + 1.))


def prolongation(f: Callable, p: int) -> Callable:
    """Generate a function that computes the Taylor coefficient series of the flow.

    Parameters
    ----------
    f : Callable
        The vector field function f(t, x, *args)
    p : int
        Order of prolongation (number of derivatives beyond the first)

    Returns
    -------
    Callable
        A function that returns the Taylor series coefficients [x, x', x'', ...]
    """
    def f_prolonged(t, x, *args):
        def _f(t, x): return f(t, x, *args)
        t_series = [t, 1.]
        x_series = [x, _f(t, x)]
        for k in range(p):
            out1, out2 = jet(_f, (t_series[0], x_series[0]), (t_series[1:], x_series[1:]))
            t_series.append(0.)
            x_series.append(out2[-1])
        return x_series
    return f_prolonged


# --- Data Structures ---

class ReachableSets(ABC):
    """Abstract base class for reachable set containers."""
    ts: jax.Array

    @abstractmethod
    def __call__(self, t):
        """Get the reachable set at time t."""
        pass


@register_pytree_node_class
@dataclass
class GenericReachSets(ReachableSets):
    """Generic container for sequence of sets (Pytree).

    This container stores reachable sets as stacked pytree leaves,
    allowing efficient JIT compilation with lax.scan.

    Attributes
    ----------
    ts : jax.Array
        Time points, shape (num_steps + 1,)
    set_stack : Any
        Pytree of sets with leaf arrays stacked along first dimension
    """
    ts: jax.Array
    set_stack: Any  # Pytree of sets (leaf arrays stacked)

    @property
    def sets(self):
        """Lazy reconstruction of set objects."""
        num_steps = len(self.ts)
        return [self[i] for i in range(num_steps)]

    def __call__(self, t):
        """Get the reachable set closest to time t."""
        i = jnp.searchsorted(self.ts, t) - 1
        idx = jnp.where(t - self.ts[i] < self.ts[i+1] - t, i, i+1)
        return self[idx]

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, i):
        return jax.tree.map(lambda x: x[i], self.set_stack)

    # --- PyTree Implementation ---
    def tree_flatten(self):
        return ((self.ts, self.set_stack), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# --- Abstract Generator ---

class ReachableSetGenerator(ABC):
    """Abstract base class for reachable set generators.

    Subclasses must implement:
    - step(t, set0, f_args): Single time step of the algorithm
    - _enforce_limit(Z): Reduce set complexity to maintain fixed shapes
    """
    sys: System
    dt: float

    @abstractmethod
    def step(self, t: float, set0, f_args):
        """Perform a single reachability step.

        Parameters
        ----------
        t : float
            Current time
        set0 : Any
            Current set (Zonotope, ConstrainedZonotope, etc.)
        f_args : tuple
            Extra arguments for system dynamics

        Returns
        -------
        Any
            Next reachable set
        """
        pass

    @abstractmethod
    def _enforce_limit(self, Z) -> Any:
        """Ensure set complexity is limited (via reduction).

        This is required for lax.scan compatibility - all sets must have
        the same shape throughout the computation.
        """
        pass

    def compute_reach_sets(self, t0: float, num_steps: int, set0, f_args=()) -> GenericReachSets:
        """Compute reachable sets using jax.lax.scan.

        Parameters
        ----------
        t0 : float
            Start time
        num_steps : int
            Exact number of steps to take (static for JIT)
        set0 : Any
            Initial set (Zonotope, ConstrainedZonotope, etc.)
        f_args : tuple
            Extra arguments for system dynamics

        Returns
        -------
        GenericReachSets
            Container with all reachable sets
        """
        if not hasattr(self, 'dt'):
            raise ValueError("ReachableSetGenerator subclass must have 'dt' attribute set.")

        times = t0 + jnp.arange(num_steps + 1) * self.dt

        # 1. Enforce limit on initial set
        Z0_fixed = self._enforce_limit(set0)

        # 2. Define Scan Function
        def scan_fn(carrier_Z, t):
            next_Z = self.step(t, carrier_Z, f_args)
            next_Z_fixed = self._enforce_limit(next_Z)
            return next_Z_fixed, next_Z_fixed

        # 3. Run Scan
        _, stacked_sets = lax.scan(scan_fn, Z0_fixed, times[:-1])

        # 4. Prepend Initial Condition
        all_sets = jax.tree.map(
            lambda z0, zs: jnp.concatenate([z0[None, ...], zs], axis=0),
            Z0_fixed,
            stacked_sets
        )

        return GenericReachSets(times, all_sets)


# --- Base Implementation ---

class BaseSetGenerator(ReachableSetGenerator):
    """Base implementation with common functionality for all algorithms.

    Provides rough enclosure computation via Picard iteration.
    """

    def __init__(self, sys: System, dt: float):
        """Initialize the generator.

        Parameters
        ----------
        sys : System
            The dynamical system
        dt : float
            Time step size
        """
        self.sys = sys
        self.dt = dt

    def _compute_rough_enclosure(self, t: float, Z, f_args) -> Interval:
        """Compute a rough enclosure of the reachable set over [t, t+dt].

        Uses Picard iteration to find an interval that contains
        the flow of all points in Z over the time interval.

        Parameters
        ----------
        t : float
            Current time
        Z : Any
            Current set (must have interval_hull() method)
        f_args : tuple
            Extra arguments for system dynamics

        Returns
        -------
        Interval
            Enclosure of reachable states over [t, t+dt]
        """
        # Initial guess: current set inflated slightly
        R = Z.interval_hull().scale(1.1)
        dt_int = Interval(0.0, self.dt)

        # Fixed 3 iterations of Picard
        def body(i, R_curr):
            f_bound = natif(lambda x: self.sys.f(t, x, *f_args))(R_curr)
            R_next = Z.interval_hull() + dt_int * f_bound
            return Interval(
                jnp.minimum(R_curr.lower, R_next.lower),
                jnp.maximum(R_curr.upper, R_next.upper)
            )

        return lax.fori_loop(0, 3, body, R)
