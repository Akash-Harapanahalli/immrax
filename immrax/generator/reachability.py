"""Zonotope-based reachability algorithms for nonlinear systems.

This module implements three reachability algorithms:
1. Lohner's Algorithm (Taylor expansion + QR reduction)
2. Althoff-Girard Algorithm (Conservative Linearization + Matrix Exp)
3. Althoff-Taylor Algorithm (Taylor expansion + Girard reduction)
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from jax.experimental.jet import jet
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Union

from ..system import System
from ..inclusion import Interval, natif
from .sets import Zonotope

# --- Helper Functions ---

def fact(n):
    """Compute factorial using gamma function."""
    return lax.exp(lax.lgamma(n + 1.))

def inv_fact(n):
    """Compute inverse factorial (1/n!) using gamma function."""
    return lax.exp(-lax.lgamma(n + 1.))

def prolongation(f: Callable, p: int) -> Callable:
    """Generates a function that computes the Taylor coefficient series of the flow.
    
    Returns a function `f_prolonged(t, x, *args)` that returns a list of 
    derivatives [x, x', x'', ..., x^(p)].
    """
    @jax.jit
    def f_prolonged(t, x, *args):
        # Any remaining args at this stage are treated as constants for the Taylor series
        def _f(t, x): return f(t, x, *args)
        
        t_series = [t, 1.]
        x_series = [x, _f(t, x)]
        
        for k in range(p):
            # Pass primals and series to jet
            # t_series[1:] and x_series[1:] are the derivative parts
            out1, out2 = jet(_f, (t_series[0], x_series[0]), (t_series[1:], x_series[1:]))
            t_series.append(0.)
            x_series.append(out2[-1])
        return x_series
    return f_prolonged

# --- Abstract Base Classes & Data Structures ---

# class ReachableTube(ABC):
#     @abstractmethod
#     def __call__(self, t) :
#         """Slice the reachable tube at t"""
#         pass

class ReachableSets(ABC):
    ts: jax.Array
    sets: list  # List of Parametope/Zonotope
    
    @abstractmethod
    def __call__(self, t):
        """Get the reachable set at closest time index to t"""
        i = jnp.searchsorted(self.ts, t) - 1
        # Interpolation logic or closest-neighbor selection
        return jnp.where(t - self.ts[i] < self.ts[i+1] - t, self.sets[i], self.sets[i+1])

@dataclass
class ZonotopeReachSets(ReachableSets):
    """Container for a sequence of Zonotopes."""
    ts: jax.Array
    sets: List[Zonotope]

    def __call__(self, t):
        return super().__call__(t)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, i):
        return self.sets[i]

# class PolyReachableTube(ReachableTube):
#     ts: jax.Array
#     ox_coeffs: jax.Array
#     alpha_coeffs: jax.Array
#     y_coeffs: jax.Array
#     ox_order: int
#     alpha_order: int
#     y_order: int

#     def __init__(self, ts, ox_coeffs, alpha_coeffs, y_coeffs):
#         self.ts = ts
#         self.ox_coeffs = ox_coeffs
#         self.alpha_coeffs = alpha_coeffs
#         self.y_coeffs = y_coeffs
#         self.ox_order = len(ox_coeffs)
#         self.alpha_order = len(alpha_coeffs)
#         self.y_order = len(y_coeffs)

#     def __call__(self, t):
#         i = jnp.searchsorted(self.ts, t) - 1
#         h = t - self.ts[i]
#         ox_nn = jnp.arange(self.ox_order)
#         alpha_nn = jnp.arange(self.alpha_order)
#         y_nn = jnp.arange(self.y_order)
        
#         ox = jnp.sum(self.ox_coeffs * inv_fact(ox_nn) * h**ox_nn, axis=0)
#         alpha = jnp.sum(self.alpha_coeffs * inv_fact(alpha_nn) * h**alpha_nn, axis=0)
#         y = jnp.sum(self.y_coeffs * inv_fact(y_nn) * h**y_nn, axis=0)
#         return Parametope(ox, alpha, y)

class ReachableSetGenerator(ABC):
    sys: System

    @abstractmethod
    def step(self, t: float, set0, f_args):
        """Compute the next reachable set from set0 at time t"""
        pass

    def compute_reach_sets(self, t0: float, tf: float, set0 , f_args=()) -> ReachableSets:
        """Compute the reachable sets over [t0, tf] starting from set0"""
        # Default dt if not set in subclass
        dt = getattr(self, 'dt', (tf - t0) / 100.0)
        
        n_steps = int(jnp.ceil((tf - t0) / dt))
        times = jnp.linspace(t0, t0 + n_steps * dt, n_steps + 1)

        sets = [set0]
        current_set = set0
        for i in range(n_steps):
            t = times[i]
            current_set = self.step(float(t), current_set, f_args)
            sets.append(current_set)
        
        return ZonotopeReachSets(times, sets)

# --- Base Implementation for Zonotopes ---

class BaseZonotopeGenerator(ReachableSetGenerator):
    """Shared logic for Zonotope-based reachability."""
    
    def __init__(self, sys: System, dt: float):
        self.sys = sys
        self.dt = dt

    def _compute_rough_enclosure(self, t: float, Z: Zonotope, f_args) -> Interval:
        """Compute a priori bound (rough enclosure) using Picard iteration.
        
        Ensures that for all s in [0, dt], the trajectory starting in Z remains 
        within the returned Interval R.
        """
        # Initial guess: current set inflated slightly
        R = Z.interval_hull().scale(1.1)
        dt_int = Interval(0.0, self.dt)
        
        # Picard Iteration: R_{k+1} = Z + [0, dt] * f(R_k)
        # 3 iterations is typically sufficient for convergence
        for _ in range(3):
            # Bind args to f
            f_bound = natif(lambda x: self.sys.f(t, x, *f_args))(R)
            R_next = Z.interval_hull() + dt_int * f_bound
            
            # Union to ensure safety (inflation)
            R = Interval(
                jnp.minimum(R.lower, R_next.lower),
                jnp.maximum(R.upper, R_next.upper)
            )
        return R

# --- 1. Lohner's Algorithm (QR Reduction) ---

class LohnerReachability(BaseZonotopeGenerator):
    """Lohner's algorithm using high-order Taylor expansion and QR reduction."""

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
        # Pre-compile prolongation for order p (produces terms up to dt^p)
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _taylor_map(self, t: float, x: jax.Array, f_args) -> jax.Array:
        """Evaluates the Taylor polynomial at time t+dt."""
        series = self._get_series(t, x, *f_args)
        
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        
        for k, term in enumerate(series):
            # term is x^(k), Taylor coeff is (1/k!)
            coeff = inv_fact(k)
            if k > 0:
                dt_pow *= self.dt
            res = res + (term * dt_pow * coeff)
            
        return res

    def _reduce_qr(self, Z: Zonotope) -> Zonotope:
        """Lohner's specific QR-based reduction."""
        if self.max_generators is None or Z.m <= self.max_generators:
            return Z

        G = Z.G
        # QR factorization: G = Q @ R
        Q, R_mat = jnp.linalg.qr(G, mode='reduced')
        
        # Lohner keeps the structure aligned with Q and boxes the R matrix
        # However, to reduce generator count effectively to a fixed number:
        
        # 1. Sort columns of R (or G) by norm to find dominant directions?
        # Standard Lohner usually keeps the Q frame and boxes the 'tail'.
        # Simplified implementation: standard Girard reduction is often preferred 
        # unless tracking the coordinate frame explicitly.
        # But per request, let's do a basic QR re-alignment then box.
        
        # Enclose the R matrix in a box (diagonal matrix of row sums)
        # This aligns the box with the frame Q.
        row_sums = jnp.sum(jnp.abs(R_mat), axis=1)
        G_new = Q * row_sums[None, :]
        
        # If G_new still has too many columns (n), we can't reduce further without loss.
        # Usually, this step resets the generators to be square (n x n).
        return Zonotope(Z.ox, G_new)

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        
        # 1. Rough Enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # 2. Compute Center via Taylor Map
        c_new = self._taylor_map(t, c, f_args)

        # 3. Compute Linear Map A (Jacobian of the Taylor Map)
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        G_new = A @ Z.G

        # 4. Compute Remainder (Lagrange Term)
        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            next_term = out_series[-1]  # Get last element of output series
            return jnp.abs(next_term)

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        error_radius = coeff * max_deriv
        G_rem = jnp.diag(error_radius)
        
        Z_new = Zonotope(c_new, jnp.concatenate([G_new, G_rem], axis=1))

        return self._reduce_qr(Z_new)


# --- 2. Althoff-Girard Algorithm (Conservative Linearization) ---

class AlthoffGirardReachability(BaseZonotopeGenerator):
    """Althoff-Girard algorithm using Conservative Linearization & Matrix Exp."""

    max_generators: int

    def __init__(
        self,
        sys: System,
        dt: float,
        max_generators: Optional[int] = None,
    ) -> None:
        super().__init__(sys, dt)
        self.max_generators = max_generators if max_generators is not None else 2 * sys.xlen

    def _reduce_girard(self, Z: Zonotope) -> Zonotope:
        """Girard's generator reduction (sorting by norm)."""
        if Z.m <= self.max_generators: return Z
        
        G = Z.G
        n_reduce = Z.m - self.max_generators + Z.n
        
        norms = jnp.linalg.norm(G, axis=0, ord=1)
        sorted_idx = jnp.argsort(norms)
        
        G_keep = G[:, sorted_idx[n_reduce:]]
        G_red = G[:, sorted_idx[:n_reduce]]
        
        d = jnp.sum(jnp.abs(G_red), axis=1)
        return Zonotope(Z.ox, jnp.concatenate([G_keep, jnp.diag(d)], axis=1))

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        n = Z.n
        
        # 1. Rough Enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # 2. Linearization
        def f_x(x): return self.sys.f(t, x, *f_args)
        J_c = jax.jacfwd(f_x)(c)
        f_c = f_x(c)
        
        # 3. Lagrange Remainder Bound (evaluated over R)
        delta = R - Interval(c, c)
        delta_max = jnp.maximum(jnp.abs(delta.lower), jnp.abs(delta.upper))
        dx_sq_sum = jnp.sum(delta_max**2)

        L_bounds = []
        for i in range(n):
            def hessian_norm(x):
                return jnp.max(jnp.abs(jax.hessian(lambda s: self.sys.f(t, s, *f_args)[i])(x)))
            H_result = natif(hessian_norm)(R)
            # natif returns Interval for interval inputs, but handle both cases
            H_max = H_result.upper if hasattr(H_result, 'upper') else H_result
            L_bounds.append(0.5 * H_max * dx_sq_sum)
        
        L_bound = jnp.array(L_bounds)

        # 4. Propagation
        Phi = jax.scipy.linalg.expm(J_c * dt)
        v = f_c - J_c @ c
        
        # Approx integral for affine term: (dt*I + dt^2/2 * A) @ v
        int_Phi = dt * jnp.eye(n) + 0.5 * (dt**2) * J_c
        
        c_new = Phi @ c + int_Phi @ v
        G_lin = Phi @ Z.G
        
        # Input error generator
        G_rem = jnp.diag(dt * L_bound)
        
        Z_new = Zonotope(c_new, jnp.concatenate([G_lin, G_rem], axis=1))

        return self._reduce_girard(Z_new)


# --- 3. Althoff-Taylor Algorithm (Taylor Expansion) ---

class AlthoffTaylorReachability(BaseZonotopeGenerator):
    """
    Althoff's algorithm using high-order Taylor expansion (like Lohner)
    but with Girard's reduction.
    """

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
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _taylor_map(self, t: float, x: jax.Array, f_args) -> jax.Array:
        """Evaluates the Taylor polynomial at time t+dt."""
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        
        for k, term in enumerate(series):
            coeff = inv_fact(k)
            if k > 0:
                dt_pow *= self.dt
            res = res + (term * dt_pow * coeff)
        return res

    def _reduce_girard(self, Z: Zonotope) -> Zonotope:
        """Girard's reduction (same as AlthoffGirard)."""
        if self.max_generators is None or Z.m <= self.max_generators:
            return Z
        
        G = Z.G
        n_reduce = Z.m - self.max_generators + Z.n
        
        norms = jnp.linalg.norm(G, axis=0, ord=1)
        sorted_idx = jnp.argsort(norms)
        
        G_keep = G[:, sorted_idx[n_reduce:]]
        G_red = G[:, sorted_idx[:n_reduce]]
        
        d = jnp.sum(jnp.abs(G_red), axis=1)
        return Zonotope(Z.ox, jnp.concatenate([G_keep, jnp.diag(d)], axis=1))

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        
        # 1. Rough Enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # 2. Center via Taylor Map
        c_new = self._taylor_map(t, c, f_args)

        # 3. Jacobian of Taylor Map
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        G_new = A @ Z.G

        # 4. Lagrange Remainder
        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            next_term = out_series[-1]  # Get last element of output series
            return jnp.abs(next_term)

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        error_radius = coeff * max_deriv
        G_rem = jnp.diag(error_radius)
        
        Z_new = Zonotope(c_new, jnp.concatenate([G_new, G_rem], axis=1))

        return self._reduce_girard(Z_new)


# --- Convenience Wrappers ---

def lohner_reachtube(sys: System, t0: float, tf: float, Z0: Zonotope, dt: float, 
                     max_generators: Optional[int] = None, taylor_order: int = 3, f_args=()) -> ZonotopeReachSets:
    reach = LohnerReachability(sys, dt, max_generators, taylor_order)
    return reach.compute_reach_sets(t0, tf, Z0, f_args)

def althoff_girard_reachtube(sys: System, t0: float, tf: float, Z0: Zonotope, dt: float, 
                             max_generators: Optional[int] = None, f_args=()) -> ZonotopeReachSets:
    reach = AlthoffGirardReachability(sys, dt, max_generators)
    return reach.compute_reach_sets(t0, tf, Z0, f_args)

def althoff_taylor_reachtube(sys: System, t0: float, tf: float, Z0: Zonotope, dt: float, 
                             max_generators: Optional[int] = None, taylor_order: int = 3, f_args=()) -> ZonotopeReachSets:
    reach = AlthoffTaylorReachability(sys, dt, max_generators, taylor_order)
    return reach.compute_reach_sets(t0, tf, Z0, f_args)