"""Taylor Model Flowpipe Algorithm for Reachability Analysis.

This module implements flowpipe computation using Taylor models with
Taylor expansion in both state and time. The algorithm computes rigorous
overapproximations of reachable sets for nonlinear dynamical systems.

References
----------
.. [1] Chen, X., et al. "Taylor model flowpipe construction for non-linear
       hybrid systems." RTSS 2012.
.. [2] Makino, K., and Berz, M. "Taylor models and other validated functional
       inclusion methods." Int. J. Pure Appl. Math. 4.4 (2003): 379-456.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.jet import jet
from typing import Callable, Tuple

from ...system import System
from ...inclusion import Interval, interval, natif
from ...utils import fact, inv_fact
from ...taylor.taylor_model import (
    TaylorModel,
    taylor_model_identity,
    taylor_model_concatenate,
)
from ...taylor.pjetm import pjetm
from .base import BaseSetGenerator, GenericReachSets


def tm_prolongation(f: Callable, order: int) -> Callable:
    """Compute Taylor coefficients in time for the flow using Taylor models.

    Given ẋ = f(t, x), computes the coefficients x^(k) in the expansion:
        x(t+h) = Σₖ x^(k)(x₀) * h^k / k!

    Each coefficient x^(k) is computed as a function that can be applied
    to Taylor models to get TM representations.

    Parameters
    ----------
    f : Callable
        Vector field f(t, x) -> dx/dt
    order : int
        Order of time Taylor expansion

    Returns
    -------
    Callable
        Function (t, x) -> [x^(0), x^(1), ..., x^(order)] where each
        x^(k) is the k-th time derivative of x at (t, x)
    """
    def prolonged(t, x):
        # Use jet to compute Taylor coefficients of the flow
        # x^(0) = x
        # x^(1) = f(t, x)
        # x^(k) = d^(k-1)/dt^(k-1) f(t, x(t))

        def _f(t, x):
            return f(t, x)

        # Build Taylor series: t_series represents t + τ, x_series represents x(t+τ)
        t_series = [t, 1.0]  # t + 1*τ
        x_series = [x, _f(t, x)]  # x + f(t,x)*τ + ...

        for k in range(order - 1):
            # Extend the series using jet
            _, derivs = jet(_f, (t_series[0], x_series[0]), (t_series[1:], x_series[1:]))
            t_series.append(0.0)
            x_series.append(derivs[-1])

        return x_series

    return prolonged


def tm_flowpipe_step(
    f: Callable,
    t: float,
    tm0: TaylorModel,
    h: float,
    order_time: int,
    order_state: int | None = None,
) -> TaylorModel:
    """Compute one step of Taylor model flowpipe.

    Computes an overapproximation of the reachable set at time t+h starting
    from the Taylor model tm0 at time t.

    The algorithm:
    1. Compute time Taylor coefficients x^(k)(x₀) for k=0,...,order_time
       using pjetm to propagate TMs through the prolongation
    2. Evaluate the time Taylor polynomial: Σₖ x^(k) * h^k / k!
    3. Bound the Lagrange remainder using interval arithmetic

    Parameters
    ----------
    f : Callable
        Vector field f(t, x) -> dx/dt
    t : float
        Current time
    tm0 : TaylorModel
        Initial Taylor model representing the set at time t
    h : float
        Time step
    order_time : int
        Order of Taylor expansion in time
    order_state : int, optional
        Order of Taylor expansion in state. If None, uses tm0's order.

    Returns
    -------
    TaylorModel
        Taylor model overapproximating the reachable set at time t+h
    """
    if order_state is None:
        order_state = tm0._static_order

    # Step 1: Compute prolongation as TMs
    # Each x^(k) is a TM in the initial state
    prolonged = tm_prolongation(f, order_time)

    # Apply pjetm to get Taylor coefficients as Taylor models
    def get_tm_coeffs(x0):
        return prolonged(t, x0)

    # x_coeffs is a list of TMs: [x^(0), x^(1), ..., x^(order_time)]
    x_coeffs = pjetm(get_tm_coeffs, max_order=order_state)(tm0)

    # Step 2: Evaluate time Taylor polynomial using Horner's method
    # x(t+h) ≈ Σₖ x^(k) * h^k / k! = x^(0) + h*(x^(1)/1! + h*(x^(2)/2! + ...))

    # Start from highest order term
    result = x_coeffs[order_time] * inv_fact(order_time)

    for k in range(order_time - 1, -1, -1):
        # result = x^(k)/k! + h * result
        coeff_k = x_coeffs[k] * inv_fact(k)
        result = coeff_k + h * result

    # Step 3: Bound the Lagrange remainder
    # R = x^(order_time+1)(ξ) * h^(order_time+1) / (order_time+1)!
    # where ξ is in the interval hull of the flow over [0, h]

    # Compute rough enclosure for remainder bounding
    rough_enclosure = _compute_rough_enclosure_tm(f, t, tm0, h)

    # Compute (order_time+1)-th derivative bound over rough enclosure
    prolonged_extra = tm_prolongation(f, order_time + 1)

    def get_highest_deriv(x):
        coeffs = prolonged_extra(t, x)
        return coeffs[order_time + 1]

    deriv_bound = natif(get_highest_deriv)(rough_enclosure)

    # Remainder bound: |R| ≤ |x^(p+1)(ξ)| * h^(p+1) / (p+1)!
    h_power = h ** (order_time + 1)
    remainder_bound = deriv_bound * (h_power * inv_fact(order_time + 1))

    # Add remainder to result
    new_remainder = result.remainder + remainder_bound

    return TaylorModel(
        result.coeffs,
        result.multiindices,
        new_remainder,
        result.domain_center,
        result.domain_radius,
        _static_order=order_state,
    )


def _compute_rough_enclosure_tm(
    f: Callable,
    t: float,
    tm0: TaylorModel,
    h: float,
    max_iters: int = 10,
) -> Interval:
    """Compute a rough enclosure of the flow over [t, t+h].

    Uses Picard iteration to find an interval containing
    all states reachable from tm0 over the time interval [t, t+h].

    Parameters
    ----------
    f : Callable
        Vector field
    t : float
        Current time
    tm0 : TaylorModel
        Initial Taylor model
    h : float
        Time step
    max_iters : int
        Maximum Picard iterations

    Returns
    -------
    Interval
        Enclosure of reachable states over [t, t+h]
    """
    # Initial guess: inflate the interval hull
    x0_hull = tm0.interval_hull()
    R = interval(x0_hull.lower * 0.9 - 0.1, x0_hull.upper * 1.1 + 0.1)
    dt_int = interval(jnp.zeros(()), h * jnp.ones(()))

    for _ in range(max_iters):
        # Picard operator: K(X) = x0 + [0,h]*f(t, X)
        f_bound = natif(lambda x: f(t, x))(R)
        R_next = x0_hull + dt_int * f_bound

        # Check if R contains R_next (fixed point)
        is_subset = jnp.all(R_next.lower >= R.lower) & jnp.all(R_next.upper <= R.upper)

        if is_subset:
            return R

        # Inflate and continue
        R = interval(
            jnp.minimum(R.lower, R_next.lower) - 0.05 * jnp.abs(R.lower),
            jnp.maximum(R.upper, R_next.upper) + 0.05 * jnp.abs(R.upper),
        )

    return R


class TMFlowpipeGenerator(BaseSetGenerator):
    """Taylor Model Flowpipe Generator.

    Computes reachable set flowpipes using Taylor models with
    Taylor expansion in both state and time.

    Parameters
    ----------
    sys : System
        The dynamical system (must be continuous)
    dt : float
        Time step size
    order_time : int
        Order of Taylor expansion in time (default: 4)
    order_state : int
        Order of Taylor expansion in state (default: 3)
    max_generators : int
        Maximum number of generators when converting to zonotope (default: 20)

    Example
    -------
    >>> from immrax.system import System
    >>> from immrax.inclusion import icentpert
    >>> from immrax.taylor.taylor_model import taylor_model_identity
    >>>
    >>> class VanDerPol(System):
    ...     def __init__(self):
    ...         super().__init__('continuous', 2)
    ...     def f(self, t, x):
    ...         return jnp.array([x[1], (1 - x[0]**2) * x[1] - x[0]])
    >>>
    >>> sys = VanDerPol()
    >>> gen = TMFlowpipeGenerator(sys, dt=0.1, order_time=4, order_state=3)
    >>> x0 = taylor_model_identity(icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1])), order=3)
    >>> reach = gen.compute_reach_sets(0.0, 10, x0)
    """

    def __init__(
        self,
        sys: System,
        dt: float,
        order_time: int = 4,
        order_state: int = 3,
        max_generators: int = 20,
    ):
        super().__init__(sys, dt)

        if sys.evolution != "continuous":
            raise ValueError("TMFlowpipeGenerator only supports continuous systems")

        self.order_time = order_time
        self.order_state = order_state
        self.max_generators = max_generators

    def step(self, t: float, tm0: TaylorModel, f_args: Tuple = ()) -> TaylorModel:
        """Perform one flowpipe step.

        Parameters
        ----------
        t : float
            Current time
        tm0 : TaylorModel
            Current Taylor model
        f_args : tuple
            Extra arguments for the vector field

        Returns
        -------
        TaylorModel
            Taylor model at time t + dt
        """
        def f(t, x):
            return self.sys.f(t, x, *f_args)

        return tm_flowpipe_step(
            f, t, tm0, self.dt,
            order_time=self.order_time,
            order_state=self.order_state,
        )

    def _enforce_limit(self, tm: TaylorModel) -> TaylorModel:
        """Ensure TM has consistent structure for lax.scan.

        For Taylor models, we ensure they have canonical exponent structure
        and bounded order.
        """
        return tm.to_canonical(self.order_state)


def tm_reachtube(
    sys: System,
    x0: TaylorModel | Interval,
    t_span: Tuple[float, float],
    dt: float,
    order_time: int = 4,
    order_state: int = 3,
    f_args: Tuple = (),
) -> GenericReachSets:
    """Compute reachable set flowpipe using Taylor models.

    Convenience function for computing Taylor model flowpipes.

    Parameters
    ----------
    sys : System
        Continuous dynamical system
    x0 : TaylorModel or Interval
        Initial set (converted to TM if Interval)
    t_span : tuple
        (t_start, t_end) time interval
    dt : float
        Time step
    order_time : int
        Order of time Taylor expansion
    order_state : int
        Order of state Taylor expansion
    f_args : tuple
        Extra arguments for system dynamics

    Returns
    -------
    GenericReachSets
        Flowpipe of Taylor models

    Example
    -------
    >>> reach = tm_reachtube(sys, x0_interval, (0, 1), dt=0.05, order_time=4)
    >>> for i, tm in enumerate(reach.sets):
    ...     print(f"t={reach.ts[i]:.2f}: {tm.interval_hull()}")
    """
    # Convert interval to Taylor model if needed
    if isinstance(x0, Interval):
        x0 = taylor_model_identity(x0, order=order_state)

    t0, tf = t_span
    num_steps = int((tf - t0) / dt)

    gen = TMFlowpipeGenerator(
        sys, dt,
        order_time=order_time,
        order_state=order_state,
    )

    return gen.compute_reach_sets(t0, num_steps, x0, f_args)
