"""Zonotope-based reachability algorithms for nonlinear systems.

This module implements two reachability algorithms:
1. Lohner's algorithm with QR-based generator rotation
2. Althoff-Girard algorithm with conservative linearization

References:
- Lohner, R. (1992). "Computation of guaranteed enclosures for the solutions
  of ordinary initial and boundary value problems."
- Althoff, M., Stursberg, O., & Buss, M. (2008). "Reachability analysis of
  nonlinear systems with uncertain parameters using conservative linearization."
- Girard, A. (2005). "Reachability of uncertain linear systems using zonotopes."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array

from ..system import System
from ..inclusion import Interval, natif
from .sets import Zonotope, zonotope, zonotope_from_interval
from jax.experimental.jet import jet

# --- Reachable Tube Data Structures ---


@dataclass
class ZonotopeReachTube:
    """A reachable tube represented as a sequence of zonotopes.

    Attributes
    ----------
    times : Array
        Time points, shape (N+1,)
    zonotopes : Zonotope
        Batched zonotope with centers shape (N+1, n) and generators shape (N+1, n, m)
    """
    times: Array
    zonotopes: Zonotope

    def __getitem__(self, i: int) -> Zonotope:
        """Get the zonotope at index i."""
        return Zonotope(self.zonotopes.ox[i], self.zonotopes.G[i])

    def __len__(self) -> int:
        return len(self.times)

    def at_time(self, t: float) -> Zonotope:
        """Get the zonotope at the closest time to t."""
        i = jnp.argmin(jnp.abs(self.times - t))
        return self[i]


# --- Base Class for Zonotope Reachability ---


class ZonotopeReachability(ABC):
    """Base class for zonotope-based reachability algorithms.

    Parameters
    ----------
    sys : System
        The dynamical system (must be continuous)
    dt : float
        Time step for reachability computation
    """

    sys: System
    dt: float

    def __init__(self, sys: System, dt: float) -> None:
        if sys.evolution != "continuous":
            raise ValueError("ZonotopeReachability only supports continuous systems")
        self.sys = sys
        self.dt = dt

    def _compute_rough_enclosure(self, t: float, Z: Zonotope) -> Interval:
        """Compute a priori bound (rough enclosure) for the time step.

        Uses Picard iteration to find a box R such that for all s in [0, dt],
        the trajectory starting in Z remains in R.
        """
        # Initial guess: current set inflated slightly
        R = Z.interval_hull().scale(1.1)
        
        # Picard Iteration
        # R_{k+1} = Z(0) + [0, dt] * f(R_k)
        dt_int = Interval(0.0, self.dt)
        
        # Perform 3 iterations (usually sufficient for convergence)
        for _ in range(3):
            # Evaluate f over current rough guess
            f_bound = natif(lambda x: self.sys.f(t, x))(R)
            
            # Picard step
            R_next = Z.interval_hull() + dt_int * f_bound
            
            # Intersect/Union to ensure validity (here we union to be safe)
            R = Interval(
                jnp.minimum(R.lower, R_next.lower),
                jnp.maximum(R.upper, R_next.upper)
            )
        return R

    @abstractmethod
    def step(self, t: float, Z: Zonotope) -> Zonotope:
        """Compute the reachable set after one time step.

        Parameters
        ----------
        t : float
            Current time
        Z : Zonotope
            Current reachable set

        Returns
        -------
        Zonotope
            Reachable set at time t + dt
        """
        pass

    def compute_reachtube(
        self,
        t0: float,
        tf: float,
        Z0: Zonotope,
    ) -> ZonotopeReachTube:
        """Compute the reachable tube from t0 to tf.

        Parameters
        ----------
        t0 : float
            Initial time
        tf : float
            Final time
        Z0 : Zonotope
            Initial set

        Returns
        -------
        ZonotopeReachTube
            The reachable tube
        """
        n_steps = int(jnp.ceil((tf - t0) / self.dt))
        times = jnp.linspace(t0, t0 + n_steps * self.dt, n_steps + 1)

        # Use Python loop since generator count may vary
        zonotope_list = [Z0]
        Z = Z0
        for i in range(n_steps):
            t = times[i]
            Z = self.step(float(t), Z)
            zonotope_list.append(Z)

        # Find max generator count for padding
        max_m = max(z.m for z in zonotope_list)

        # Pad generators to uniform shape
        def pad_generators(Z, target_m):
            if Z.m < target_m:
                padding = jnp.zeros((Z.n, target_m - Z.m), dtype=Z.G.dtype)
                return Zonotope(Z.ox, jnp.concatenate([Z.G, padding], axis=1))
            return Z

        padded_zonotopes = [pad_generators(z, max_m) for z in zonotope_list]

        all_centers = jnp.stack([z.ox for z in padded_zonotopes], axis=0)
        all_generators = jnp.stack([z.G for z in padded_zonotopes], axis=0)

        return ZonotopeReachTube(
            times=times,
            zonotopes=Zonotope(all_centers, all_generators)
        )


# --- Lohner's Algorithm ---

def prolongation(f: Callable, p: int) -> Callable:
    """Generates a function that computes the Taylor coefficient series of the flow.
    
    Returns a function `f_prolonged(t, x, *args)` that returns a list of 
    derivatives [x, x', x'', ..., x^(p+1)].
    """
    # Iteratively call jet with a growing series
    def f_prolonged(t, x, *args):
        # Any remaining args at this stage are treated as constants for the Taylor series
        def _f(t, x): return f(t, x, *args)
        
        # Initialize series for t (linear time) and x (state)
        # t_series represents t + 1*dt + 0*dt^2 ...
        t_series = [t, 1.]
        # x_series represents x, x'
        x_series = [x, _f(t, x)]
        
        for k in range(p):
            # jet computes the pushforward of derivatives
            # The input series are truncated to match the current order
            # We pass the primal (t, x) and the series of derivatives
            primals = (t_series[0], x_series[0])
            series_in = (t_series[1:], x_series[1:])
            
            out1, out2 = jet(_f, primals, series_in)
            
            t_series.append(0.)
            # out2 contains the derivative series of f(x) -> x'
            # So out2[-1] is the new highest derivative x^(k+2)
            x_series.append(out2[-1])
            
        return x_series
    return f_prolonged


# --- Lohner's Algorithm (Fixed) ---

class LohnerReachability(ZonotopeReachability):
    """Lohner's algorithm using high-order Taylor expansion via Jet."""

    max_generators: int
    taylor_order: int

    def __init__(
        self,
        sys: System,
        dt: float,
        max_generators: Optional[int] = None,
        taylor_order: int = 3,
    ) -> None:
        super().__init__(sys, dt)
        self.max_generators = max_generators
        self.taylor_order = taylor_order
        
        # Pre-compile the prolongation function for the requested order
        # Note: We need order 'p' such that we get terms up to dt^p
        # The prolongation function loop runs 'p' times.
        # Initial: [x, x']. Loop 1: +x''. Loop 2: +x'''.
        # So passing p=taylor_order-1 gives derivatives up to order taylor_order.
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _taylor_map(self, t: float, x: Array) -> Array:
        """Evaluates the Taylor polynomial at time t+dt."""
        series = self._get_series(t, x)
        
        # Sum the series: x(t+dt) = sum (1/k!) * x^(k) * dt^k
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        factorial = 1.0
        
        for k, term in enumerate(series):
            if k > 0:
                dt_pow *= self.dt
                factorial *= k
            res = res + (term * dt_pow / factorial)
            
        return res

    def step(self, t: float, Z: Zonotope) -> Zonotope:
        dt = self.dt
        c = Z.ox
        
        # 1. Rough Enclosure (A Priori Bound)
        R = self._compute_rough_enclosure(t, Z)

        # 2. Compute Center via Taylor Map
        c_new = self._taylor_map(t, c)

        # 3. Compute Linear Map A (Jacobian of the Taylor Map)
        # This accurately captures rotation and shearing (including J_dot terms)
        A = jax.jacfwd(lambda x: self._taylor_map(t, x))(c)
        G_new = A @ Z.G

        # 4. Compute Remainder (Lagrange Term)
        # Error = (1/(p+1)!) * x^(p+1)(xi) * dt^(p+1)
        # We need to bound the (p+1)-th derivative over the Rough Enclosure R.
        
        # Helper to extract the (p+1)-th derivative
        def get_highest_derivative_norm(x):
            # Get the series up to order p
            series = self._get_series(t, x)
            # Use the user's efficient jet trick to get the next term (p+1)
            # The t-series for the jet call must match the length of x-series
            t_series = [1.] + [0.] * (len(series) - 1)
            next_term = jet(self.sys.f, (t, x), (t_series, series))[-1]
            return jnp.abs(next_term)

        # Bound this function over the rough enclosure
        # natif returns an Interval of the (p+1)-th derivative
        deriv_bound_interval = natif(get_highest_derivative_norm)(R)
        max_deriv = deriv_bound_interval.upper

        # Compute error radius
        # term = (1/(p+1)!) * max_deriv * dt^(p+1)
        import math
        p = self.taylor_order
        coeff = (dt**(p + 1)) / math.factorial(p + 1)
        error_radius = coeff * max_deriv

        # Add error as diagonal generators
        G_rem = jnp.diag(error_radius)
        
        # Combine
        Z_new = Zonotope(c_new, jnp.concatenate([G_new, G_rem], axis=1))

        return self._reduce_generators(Z_new)


# --- Althoff-Girard Algorithm (Fixed) ---

class AlthoffGirardReachability(ZonotopeReachability):
    """Althoff-Girard algorithm using Conservative Linearization & Rough Enclosure."""

    max_generators: int

    def __init__(
        self,
        sys: System,
        dt: float,
        max_generators: Optional[int] = None,
    ) -> None:
        super().__init__(sys, dt)
        self.max_generators = max_generators if max_generators is not None else 2 * sys.xlen

    def step(self, t: float, Z: Zonotope) -> Zonotope:
        dt = self.dt
        c = Z.ox
        n = Z.n
        
        # 1. Rough Enclosure (A Priori Bound)
        # Critical for valid remainder evaluation
        R = self._compute_rough_enclosure(t, Z)

        # 2. Linearization Point
        # We linearize around the center c
        def f_x(x): return self.sys.f(t, x)
        J_c = jax.jacfwd(f_x)(c)
        f_c = self.sys.f(t, c)
        
        # 3. Lagrange Remainder Bound (evaluated over R)
        # L_i <= 0.5 * max_{x in R} ||H_i(x)|| * ||x-c||^2
        # Max deviation of enclosure from linearization point c
        delta = R - Interval(c, c)
        delta_max = jnp.maximum(jnp.abs(delta.lower), jnp.abs(delta.upper))
        dx_sq_sum = jnp.sum(delta_max**2)

        # Vector of remainder bounds
        L_bounds = []
        for i in range(n):
            def hessian_norm(x):
                return jnp.max(jnp.abs(jax.hessian(lambda s: self.sys.f(t, s)[i])(x)))
            
            H_max = natif(hessian_norm)(R).upper
            L_bounds.append(0.5 * H_max * dx_sq_sum)
        
        L_bound = jnp.array(L_bounds)

        # 4. Propagation (Linear + Input)
        # System: dx/dt = J_c * x + (f(c) - J_c*c) + U, where U = [-L, L]
        # x(t+dt) = Phi*x(t) + int_Phi * (affine_term) + int_Phi * U
        
        Phi = jax.scipy.linalg.expm(J_c * dt)
        
        # Affine input: v = f(c) - J_c*c
        v = f_c - J_c @ c
        
        # Integral of Phi approx (Taylor order 2 is usually sufficient for the input)
        # int_0^dt e^{As} ds = dt*I + dt^2/2 * A
        int_Phi = dt * jnp.eye(n) + 0.5 * (dt**2) * J_c
        
        c_new = Phi @ c + int_Phi @ v
        G_lin = Phi @ Z.G
        
        # Input error generator (scaled by int_Phi or approx dt)
        # Using dt is conservative for the input box
        G_rem = jnp.diag(dt * L_bound)
        
        Z_new = Zonotope(c_new, jnp.concatenate([G_lin, G_rem], axis=1))

        return self._reduce_girard(Z_new)

# --- Convenience Functions ---

def lohner_reachtube(
    sys: System,
    t0: float,
    tf: float,
    Z0: Zonotope,
    dt: float,
    max_generators: Optional[int] = None,
    taylor_order: int = 2,
) -> ZonotopeReachTube:
    """Compute reachable tube using Lohner's algorithm."""
    reach = LohnerReachability(sys, dt, max_generators, taylor_order)
    return reach.compute_reachtube(t0, tf, Z0)


def althoff_girard_reachtube(
    sys: System,
    t0: float,
    tf: float,
    Z0: Zonotope,
    dt: float,
    max_generators: Optional[int] = None,
) -> ZonotopeReachTube:
    """Compute reachable tube using Althoff-Girard algorithm."""
    reach = AlthoffGirardReachability(sys, dt, max_generators)
    return reach.compute_reachtube(t0, tf, Z0)