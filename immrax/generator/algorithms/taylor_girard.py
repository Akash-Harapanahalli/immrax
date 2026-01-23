"""Taylor-Girard reachability algorithm.

Combines Taylor expansion with Girard-style generator reduction for
computing reachable sets of nonlinear systems.
"""

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from typing import Any

from ...system import System
from ...inclusion import natif
from ..sets import Zonotope
from .base import BaseSetGenerator, prolongation, inv_fact


class TaylorGirardReachability(BaseSetGenerator):
    """Taylor-Girard algorithm for reachability analysis.

    Combines Taylor expansion of the flow map with Girard-style
    generator reduction. This provides a balance between accuracy
    (from higher-order Taylor terms) and efficiency (from zonotope
    order reduction).

    Parameters
    ----------
    sys : System
        The dynamical system
    dt : float
        Time step size
    target_order : float
        Target order for set reduction (generators / dimension)
    taylor_order : int
        Order of Taylor expansion (default: 3)

    References
    ----------
    .. [1] Althoff, M. "Reachability analysis of nonlinear systems using
           conservative polynomialization and non-convex sets." HSCC 2013.
    .. [2] Girard, A. "Reachability of uncertain linear systems using
           zonotopes." HSCC 2005.
    """
    target_order: float
    taylor_order: int

    def __init__(
        self,
        sys: System,
        dt: float,
        target_order: float = 2.0,
        taylor_order: int = 3
    ):
        super().__init__(sys, dt)
        self.target_order = float(target_order)
        self.taylor_order = taylor_order
        self._get_series = prolongation(self.sys.f, self.taylor_order - 1)

    def _enforce_limit(self, Z) -> Any:
        """Reduce set order using polymorphic reduce_order method."""
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def _taylor_map(self, t, x, f_args):
        """Compute Taylor expansion of the flow map at point x."""
        series_list = self._get_series(t, x, *f_args)
        series = jnp.stack(series_list)
        kk = jnp.arange(len(series))
        coeffs = (self.dt**kk * inv_fact(kk))[:, None]
        return jnp.sum(series * coeffs, axis=0)

    def step(self, t: float, Z, f_args):
        """Perform one step of Taylor-Girard algorithm.

        1. Compute Taylor expansion of flow at center
        2. Linearize using Jacobian of Taylor map
        3. Bound remainder using highest derivative
        4. Combine linear image with remainder bound
        """
        dt = self.dt
        c = Z.center

        # Compute rough enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # Taylor map at center
        c_new = self._taylor_map(t, c, f_args)

        # Jacobian of Taylor map
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)

        # Linear image
        Z_lin = A @ Z

        # Recenter
        shift = c_new - A @ c
        Z_lin = Z_lin + shift

        # Bound highest derivative for remainder
        # Bound highest derivative for time remainder (Lagrange in time)
        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        
        # Remainder coefficient for time
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)
        
        # Time error zonotope
        Z_error_time = Zonotope(jnp.zeros_like(c), jnp.diag(coeff * max_deriv))

        # --- Spatial Remainder ---
        # Bound linearization error: R_s = 1/2 * (x-c)^T * H(xi) * (x-c)
        # Compute Hessian of the Taylor map w.r.t state
        def taylor_map_fn(x): 
            return self._taylor_map(t, x, f_args)
            
        hessian_fn = jax.jacfwd(jax.jacrev(taylor_map_fn))
        
        # Bound Hessian over rough enclosure R
        # H_int has shape (n, n, n) of Intervals
        H_int = natif(hessian_fn)(R)
        
        dx_int = R - c
        dx_abs = jnp.maximum(jnp.abs(dx_int.lower), jnp.abs(dx_int.upper)) # (n,)

        # Check if set supports quadratic map (e.g. PolynomialZonotope)
        if hasattr(Z, 'quadratic_map'):
            # Use center of Hessian for quadratic map
            # Q = 0.5 * H_center
            H_center = (H_int.lower + H_int.upper) / 2
            Q = 0.5 * H_center
            
            # Map the centered set Z-c through quadratic form Q
            Z_quad = (Z - c).quadratic_map(Q)
            
            # Use radius of Hessian for unstructured error bound
            # H_radius = (H_upper - H_lower) / 2
            H_radius = (H_int.upper - H_int.lower) / 2
            H_for_bound = H_radius
        else:
            # No quadratic map support, bound entire Hessian
            Z_quad = Zonotope(jnp.zeros_like(c), jnp.zeros((len(c), 0)))
            H_for_bound = jnp.maximum(jnp.abs(H_int.lower), jnp.abs(H_int.upper))

        # Contraction: (n,n,n) * (n) * (n) -> (n)
        # quad_bound[i] = 0.5 * sum_jk H_ijk * dx_j * dx_k
        
        # First contract last dim
        temp = jnp.dot(H_for_bound, dx_abs) # (n, n)
        # Contract next dim
        quad_bound = 0.5 * jnp.dot(temp, dx_abs) # (n,)
        
        Z_error_spatial = Zonotope(jnp.zeros_like(c), jnp.diag(quad_bound))

        return Z_lin + Z_quad + Z_error_time + Z_error_spatial
