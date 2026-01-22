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
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
from functools import partial

from ..system import System
from ..inclusion import Interval, natif
from .sets import Zonotope

# --- Helper Functions ---

def fact(n):
    return lax.exp(lax.lgamma(n + 1.))

def inv_fact(n):
    return lax.exp(-lax.lgamma(n + 1.))

def prolongation(f: Callable, p: int) -> Callable:
    """Generates a function that computes the Taylor coefficient series of the flow."""
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
    ts: jax.Array
    
    @abstractmethod
    def __call__(self, t):
        pass

@register_pytree_node_class
@dataclass
class ZonotopeReachSets(ReachableSets):
    """Container for a sequence of Zonotopes (fixed shape)."""
    ts: jax.Array
    # Stored as stacked arrays for JAX compatibility
    center_stack: jax.Array  # (T, n)
    gen_stack: jax.Array     # (T, n, m)

    @property
    def sets(self):
        """Lazy reconstruction of Zonotope objects if needed for python iteration."""
        return [Zonotope(c, G) for c, G in zip(self.center_stack, self.gen_stack)]

    def __call__(self, t):
        i = jnp.searchsorted(self.ts, t) - 1
        idx = jnp.where(t - self.ts[i] < self.ts[i+1] - t, i, i+1)
        return Zonotope(self.center_stack[idx], self.gen_stack[idx])

    def __len__(self):
        return self.center_stack.shape[0]

    def __getitem__(self, i):
        return Zonotope(self.center_stack[i], self.gen_stack[i])

    # --- PyTree Implementation ---
    def tree_flatten(self):
        # All three attributes are JAX arrays, so they are children
        children = (self.ts, self.center_stack, self.gen_stack)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class ReachableSetGenerator(ABC):
    sys: System

    @abstractmethod
    def step(self, t: float, set0, f_args):
        pass
    
    @abstractmethod
    def _enforce_limit(self, Z: Zonotope) -> Zonotope:
        """Ensure Zonotope has exactly max_generators (via reduction or padding)."""
        pass

    def compute_reach_sets(self, t0: float, num_steps: int, set0: Zonotope, f_args=()) -> ZonotopeReachSets:
        """Compute reachable sets using jax.lax.scan.
        
        Args:
            t0: Start time.
            num_steps: Exact number of steps to take (static int for JIT).
            set0: Initial zonotope.
            f_args: Extra arguments for system dynamics.
        """
        if not hasattr(self, 'dt'):
            raise ValueError("ReachableSetGenerator subclass must have 'dt' attribute set.")
            
        times = t0 + jnp.arange(num_steps + 1) * self.dt

        # 1. Enforce shape on initial set (required for scan carry)
        Z0_fixed = self._enforce_limit(set0)

        # 2. Define Scan Function
        def scan_fn(carrier_Z, t):
            next_Z = self.step(t, carrier_Z, f_args)
            next_Z_fixed = self._enforce_limit(next_Z)
            return next_Z_fixed, (next_Z_fixed.ox, next_Z_fixed.G)

        # 3. Run Scan
        # The loop runs for times[0]...times[n-1] to produce Z1...Zn
        _, (centers, gens) = lax.scan(scan_fn, Z0_fixed, times[:-1])

        # 4. Prepend Initial Condition
        all_centers = jnp.concatenate([Z0_fixed.ox[None, :], centers], axis=0)
        all_gens = jnp.concatenate([Z0_fixed.G[None, :, :], gens], axis=0)

        return ZonotopeReachSets(times, all_centers, all_gens)

# --- Base Implementation ---

class BaseZonotopeGenerator(ReachableSetGenerator):
    
    def __init__(self, sys: System, dt: float):
        self.sys = sys
        self.dt = dt

    def _compute_rough_enclosure(self, t: float, Z: Zonotope, f_args) -> Interval:
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

    def _pad_or_reduce(self, Z: Zonotope, max_gen: int) -> Zonotope:
        """Helper to strictly enforce generator count."""
        n, m = Z.G.shape
        if m > max_gen:
            norms = jnp.linalg.norm(Z.G, axis=0, ord=1)
            sorted_idx = jnp.argsort(norms)
            
            n_reduce = m - max_gen + n
            
            idx_keep = sorted_idx[n_reduce:]
            idx_reduce = sorted_idx[:n_reduce]
            
            G_keep = Z.G[:, idx_keep]
            G_reduce = Z.G[:, idx_reduce]
            
            d = jnp.sum(jnp.abs(G_reduce), axis=1)
            G_new = jnp.concatenate([G_keep, jnp.diag(d)], axis=1)
            return Zonotope(Z.ox, G_new)
            
        elif m < max_gen:
            padding = jnp.zeros((n, max_gen - m))
            G_new = jnp.concatenate([Z.G, padding], axis=1)
            return Zonotope(Z.ox, G_new)
            
        return Z


class LohnerReachability(BaseZonotopeGenerator):
    max_generators: int
    taylor_order: int

    def __init__(self, sys: System, dt: float, max_generators: int = 10, taylor_order: int = 3):
        super().__init__(sys, dt)
        self.max_generators = max_generators
        self.taylor_order = taylor_order
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _enforce_limit(self, Z: Zonotope) -> Zonotope:
        return self._pad_or_reduce(Z, self.max_generators)

    def _taylor_map(self, t, x, f_args):
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        for k, term in enumerate(series):
            if k > 0: dt_pow *= self.dt
            res = res + (term * dt_pow * inv_fact(k))
        return res

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        
        R = self._compute_rough_enclosure(t, Z, f_args)
        c_new = self._taylor_map(t, c, f_args)
        
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        G_new = A @ Z.G

        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        G_rem = jnp.diag(coeff * max_deriv)
        
        Z_full = Zonotope(c_new, jnp.concatenate([G_new, G_rem], axis=1))
        
        # QR reduction is explicit in Lohner
        G_full = Z_full.G
        Q, R_mat = jnp.linalg.qr(G_full, mode='reduced')
        row_sums = jnp.sum(jnp.abs(R_mat), axis=1)
        G_qr = Q * row_sums[None, :]
        
        return Zonotope(c_new, G_qr)


class AlthoffGirardReachability(BaseZonotopeGenerator):
    max_generators: int

    def __init__(self, sys: System, dt: float, max_generators: int = 10):
        super().__init__(sys, dt)
        self.max_generators = max_generators

    def _enforce_limit(self, Z: Zonotope) -> Zonotope:
        return self._pad_or_reduce(Z, self.max_generators)

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        n = Z.n
        
        R = self._compute_rough_enclosure(t, Z, f_args)

        def f_x(x): return self.sys.f(t, x, *f_args)
        J_c = jax.jacfwd(f_x)(c)
        f_c = f_x(c)
        
        # Linearization error bound (Lagrange remainder)
        # Error <= 0.5 * max(||H||) * ||x - c||^2
        # We use component-wise bound: |L_i| <= 0.5 * h_max_i * (sum(|x-c|))^2
        # Note: sum(x^2) is NOT safe, (sum(|x|))^2 is required for strict over-approximation
        
        delta = R - Interval(c, c)
        delta_max = jnp.maximum(jnp.abs(delta.lower), jnp.abs(delta.upper))
        
        # The deviation vector norm squared (L1 norm squared is conservative but safe)
        dx_sq_sum = (jnp.sum(delta_max))**2

        def get_hess_bound(i):
            h_max = natif(lambda x: jnp.max(jnp.abs(jax.hessian(lambda s: self.sys.f(t, s, *f_args)[i])(x))))(R).upper
            return 0.5 * h_max * dx_sq_sum
            
        L_bound = jax.vmap(get_hess_bound)(jnp.arange(n))

        # Exact computation of integral of exponential
        # exp([ [J*dt, I*dt], [0, 0] ]) = [ [Phi, int_Phi], [0, I] ]
        M = jnp.zeros((2*n, 2*n))
        M = M.at[:n, :n].set(J_c * dt)
        M = M.at[:n, n:].set(jnp.eye(n) * dt)
        
        expM = jax.scipy.linalg.expm(M)
        Phi = expM[:n, :n]
        int_Phi = expM[:n, n:]
        
        v = f_c - J_c @ c
        
        c_new = Phi @ c + int_Phi @ v
        G_lin = Phi @ Z.G
        G_rem = jnp.diag(L_bound) # L_bound is already the error magnitude over the step
        
        # Note: L_bound is the error rate? No, standard derivation:
        # x(t) = Phi x(0) + ... + integral(remainder)
        # If we bound remainder by L, integral is L * dt.
        # Check if L_bound includes dt?
        # In code above: L_bound = 0.5 * h * dx^2. This is the spatial error bound.
        # We need to integrate it over time.
        # Simple bound: L_int <= dt * L_bound
        
        G_rem = jnp.diag(dt * L_bound)
        
        return Zonotope(c_new, jnp.concatenate([G_lin, G_rem], axis=1))


class TaylorGirardReachability(BaseZonotopeGenerator):
    max_generators: int
    taylor_order: int

    def __init__(self, sys: System, dt: float, max_generators: int = 10, taylor_order: int = 3):
        super().__init__(sys, dt)
        self.max_generators = max_generators
        self.taylor_order = taylor_order
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _enforce_limit(self, Z: Zonotope) -> Zonotope:
        return self._pad_or_reduce(Z, self.max_generators)

    def _taylor_map(self, t, x, f_args):
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        for k, term in enumerate(series):
            if k > 0: dt_pow *= self.dt
            res = res + (term * dt_pow * inv_fact(k))
        return res

    def step(self, t: float, Z: Zonotope, f_args) -> Zonotope:
        dt = self.dt
        c = Z.ox
        
        R = self._compute_rough_enclosure(t, Z, f_args)
        c_new = self._taylor_map(t, c, f_args)
        
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        G_new = A @ Z.G

        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        G_rem = jnp.diag(coeff * max_deriv)
        
        return Zonotope(c_new, jnp.concatenate([G_new, G_rem], axis=1))
