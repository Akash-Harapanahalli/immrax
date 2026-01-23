"""Taylor Model reachability algorithm.

Combines Taylor expansion in time with rigorous Taylor Model arithmetic
for computing reachable sets of nonlinear systems.
"""

from typing import Any
import jax.numpy as jnp
from jax.experimental.jet import jet

from ...system import System
from ...inclusion import natif, Interval
from ..sets import TaylorModel, Zonotope
from .base import BaseSetGenerator, prolongation, inv_fact

class TaylorModelReachability(BaseSetGenerator):
    """Taylor Model algorithm for reachability analysis.

    Uses Taylor Models to rigorously enclose the state variables.
    Performs Taylor expansion in time (for the flow map) and treats
    state variables as Taylor Models to perform expansion in state.

    Parameters
    ----------
    sys : System
        The dynamical system
    dt : float
        Time step size
    taylor_order : int
        Order of Taylor expansion in time (default: 3)
    """
    taylor_order: int

    def __init__(
        self,
        sys: System,
        dt: float,
        taylor_order: int = 3
    ):
        super().__init__(sys, dt)
        self.taylor_order = taylor_order
        # Prolongation returns [x, f(x), f'(x), ...] which correspond to
        # x, x', x'', ... (time derivatives)
        self._get_series = prolongation(self.sys.f, self.taylor_order)

    def _enforce_limit(self, Z) -> Any:
        """For TMs, this might involve reducing the polynomial remainder or degree.
        Currently assumes no reduction or delegates to TM's own methods if available.
        """
        # TODO: Implement TM order/degree reduction if necessary.
        # For now, pass through.
        return Z

    def step(self, t: float, tm: TaylorModel, f_args):
        """Perform one step of Taylor Model reachability.

        1. Compute Taylor series of the trajectory x(t) expanding in time.
           Coefficients are computed using TM arithmetic (state expansion).
           x(t+dt) = x(t) + dt*x'(t) + dt^2/2 * x''(t) + ...
        2. Evaluate the remainder term rigorously.
        """
        dt = self.dt
        
        # Compute Taylor series in time
        # series = [x, x', x'', ..., x^(k)] where each term is a TaylorModel
        series = self._get_series(t, tm, *f_args)
        
        # Sum the series
        res = tm # Initial term x(t)
        dt_pow = 1.0
        
        # We start summing from k=1 because k=0 is x(t) which we initialized with
        for k in range(1, len(series)-1): # go up to order-1
            term = series[k]
            dt_pow *= dt
            # term * (dt^k / k!)
            res = res + (term * (dt_pow * inv_fact(k)))
            
        # Remainder term: bounded by next order term
        # R_k = (dt^(k+1) / (k+1)!) * x^(k+1)(xi)
        # We use the (k+1)-th term from the series as an approximation/bound
        # Ideally we should evaluate over the interval hull [t, t+dt]
        
        # Get Picard rough enclosure for [t, t+dt] to bound the derivative
        R_hull = self._compute_rough_enclosure(t, tm, f_args)
        
        # Function to compute (k+1)-th derivative norm
        def get_highest_deriv(y):
             # We want the last element of prolongation
             s = self._get_series(t, y, *f_args)
             return s[-1] # This works for Interval/Zonotope inputs via 'natif' if supported
             
        # Evaluate highest derivative on the rough enclosure
        # Note: 'natif' works on Intervals. We convert R_hull (Interval) through the system.
        # But 'prolongation' returns a list of objects.
        
        # Explicitly compute bound for x^(p+1)
        p = self.taylor_order
        
        # Use natif to bound the (p+1)-th derivative over the rough enclosure R_hull
        # We define a helper that extracts just the (p+1)-th term
        def deriv_p_plus_1(y):
             s = self._get_series(t, y, *f_args)
             return s[-1]

        # Bound it
        deriv_bound = natif(deriv_p_plus_1)(R_hull) # Returns Interval
        
        # Remainder interval
        rem_coeff = (dt**(p + 1)) * inv_fact(p + 1)
        rem_val = deriv_bound * rem_coeff
        
        # Add remainder to the result TM
        # We add it as a constant interval (to the remainder part of TM)
        # scalar + TM adds to the constant coeff or remainder? 
        # TM + Interval adds to the remainder.
        
        res = res + rem_val
        
        return res
