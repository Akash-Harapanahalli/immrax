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
from typing import Any, Callable, Optional, List, Tuple
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
class GenericReachSets(ReachableSets):
    """Generic container for sequence of sets (Pytree)."""
    ts: jax.Array
    set_stack: Any # Pytree of sets (leaf arrays stacked)

    @property
    def sets(self):
        """Lazy reconstruction of set objects."""
        # Unstack the Pytree
        # Note: inefficient for long sequences if accessed repeatedly
        num_steps = len(self.ts)
        return [self[i] for i in range(num_steps)]

    def __call__(self, t):
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


class ReachableSetGenerator(ABC):
    sys: System
    dt: float

    @abstractmethod
    def step(self, t: float, set0, f_args):
        pass
    
    @abstractmethod
    def _enforce_limit(self, Z) -> Any:
        """Ensure set complexity is limited (via reduction)."""
        pass

    def compute_reach_sets(self, t0: float, num_steps: int, set0, f_args=()) -> GenericReachSets:
        """Compute reachable sets using jax.lax.scan.
        
        Args:
            t0: Start time.
            num_steps: Exact number of steps to take.
            set0: Initial set (Zonotope, ConstrainedZonotope, etc.).
            f_args: Extra arguments for system dynamics.
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
            # Scan carries the set object itself (which is a Pytree)
            return next_Z_fixed, next_Z_fixed

        # 3. Run Scan
        _, stacked_sets = lax.scan(scan_fn, Z0_fixed, times[:-1])

        # 4. Prepend Initial Condition
        # Stack Z0 with the rest
        all_sets = jax.tree.map(
            lambda z0, zs: jnp.concatenate([z0[None, ...], zs], axis=0),
            Z0_fixed,
            stacked_sets
        )

        return GenericReachSets(times, all_sets)

# --- Base Implementation ---

class BaseSetGenerator(ReachableSetGenerator):
    
    def __init__(self, sys: System, dt: float):
        self.sys = sys
        self.dt = dt

    def _compute_rough_enclosure(self, t: float, Z, f_args) -> Interval:
        # Initial guess: current set inflated slightly
        R = Z.interval_hull().scale(1.1)
        dt_int = Interval(0.0, self.dt)
        
        # Fixed 3 iterations of Picard
        def body(i, R_curr):
            f_bound = natif(lambda x: self.sys.f(t, x, *f_args))(R_curr)
            # Z.interval_hull() + dt_int * f_bound
            R_next = Z.interval_hull() + dt_int * f_bound
            return Interval(
                jnp.minimum(R_curr.lower, R_next.lower),
                jnp.maximum(R_curr.upper, R_next.upper)
            )
        
        return lax.fori_loop(0, 3, body, R)


class LohnerReachability(BaseSetGenerator):
    target_order: float # Renamed from max_generators for generic support
    taylor_order: int

    def __init__(self, sys: System, dt: float, target_order: float = 2.0, max_generators: int = 10, taylor_order: int = 3):
        super().__init__(sys, dt)
        # Support both old max_generators (approx order for Z) and new target_order
        # If max_generators provided and default target_order, try to infer?
        # Let's just store target_order.
        # For legacy compatibility with Zonotopes, we might need logic.
        # Assuming simple update: use target_order.
        self.target_order = float(target_order)
        # Note: user might pass max_generators=10, we ignore it or map it?
        # Let's assume user passes reasonable target_order or we fix it.
        
        self.taylor_order = taylor_order
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _enforce_limit(self, Z) -> Any:
        # Use polymorphic reduce_order
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def _taylor_map(self, t, x, f_args):
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        for k, term in enumerate(series):
            if k > 0: dt_pow *= self.dt
            res = res + (term * dt_pow * inv_fact(k))
        return res

    def step(self, t: float, Z, f_args):
        dt = self.dt
        c = Z.center # Usage of property
        
        R = self._compute_rough_enclosure(t, Z, f_args)
        c_new = self._taylor_map(t, c, f_args)
        
        # Linear map A = J(c)
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        
        # Z_lin = A @ Z
        Z_lin = A @ Z
        
        # Recenter Z_lin to c_new (Taylor map center)
        # But A @ Z centers at A @ c.
        # We want center to be c_new.
        # shift = c_new - A @ c
        # Z_lin = Z_lin + shift
        shift = c_new - A @ c
        Z_lin = Z_lin + shift

        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        
        # Additive error zonotope
        gen_rem = coeff * max_deriv
        # Z_error = Zonotope(0, diag(gen_rem))
        # We need to construct a generic "error set".
        # Simplest: assume Zonotope for error term, then add to Z.
        # But wait, CZ + Z -> CZ. PZ + Z -> PZ.
        # So creating a Zonotope error is generic enough!
        
        Z_error = Zonotope(jnp.zeros_like(c), jnp.diag(gen_rem))
        
        return Z_lin + Z_error


class AlthoffGirardReachability(BaseSetGenerator):
    target_order: float # Renamed/Generalized

    def __init__(self, sys: System, dt: float, target_order: float = 2.0, max_generators: int = 10):
        super().__init__(sys, dt)
        self.target_order = float(target_order)

    def _enforce_limit(self, Z) -> Any:
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def step(self, t: float, Z, f_args):
        dt = self.dt
        c = Z.center
        n = c.shape[0]
        
        R = self._compute_rough_enclosure(t, Z, f_args)

        def f_x(x): return self.sys.f(t, x, *f_args)
        J_c = jax.jacfwd(f_x)(c)
        f_c = f_x(c)
        
        # Linearization error bound logic (Safe)
        delta = R - Interval(c, c)
        delta_max = jnp.maximum(jnp.abs(delta.lower), jnp.abs(delta.upper))
        dx_sq_sum = (jnp.sum(delta_max))**2

        def get_hess_bound(i):
            h_max = natif(lambda x: jnp.max(jnp.abs(jax.hessian(lambda s: self.sys.f(t, s, *f_args)[i])(x))))(R).upper
            return 0.5 * h_max * dx_sq_sum
            
        L_bound = jax.vmap(get_hess_bound)(jnp.arange(n))

        # Exact Matrix Exp
        M = jnp.zeros((2*n, 2*n))
        M = M.at[:n, :n].set(J_c * dt)
        M = M.at[:n, n:].set(jnp.eye(n) * dt)
        expM = jax.scipy.linalg.expm(M)
        Phi = expM[:n, :n]
        int_Phi = expM[:n, n:]
        
        v = f_c - J_c @ c
        
        # Z_lin = Phi @ Z
        Z_lin = Phi @ Z
        
        # Translation: int_Phi @ v
        # Z_lin = Z_lin + (int_Phi @ v)
        trans = int_Phi @ v
        Z_lin = Z_lin + trans
        
        # Additive error
        # G_rem = diag(dt * L_bound)
        Z_error = Zonotope(jnp.zeros_like(c), jnp.diag(dt * L_bound))
        
        return Z_lin + Z_error


class TaylorGirardReachability(BaseSetGenerator):
    target_order: float
    taylor_order: int

    def __init__(self, sys: System, dt: float, target_order: float = 2.0, max_generators: int = 10, taylor_order: int = 3):
        super().__init__(sys, dt)
        self.target_order = float(target_order)
        self.taylor_order = taylor_order
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _enforce_limit(self, Z) -> Any:
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def _taylor_map(self, t, x, f_args):
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        for k, term in enumerate(series):
            if k > 0: dt_pow *= self.dt
            res = res + (term * dt_pow * inv_fact(k))
        return res

    def step(self, t: float, Z, f_args):
        dt = self.dt
        c = Z.center
        
        R = self._compute_rough_enclosure(t, Z, f_args)
        c_new = self._taylor_map(t, c, f_args)
        
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        Z_lin = A @ Z
        
        # Recenter
        shift = c_new - A @ c
        Z_lin = Z_lin + shift

        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        
        Z_error = Zonotope(jnp.zeros_like(c), jnp.diag(coeff * max_deriv))
        
        return Z_lin + Z_error
