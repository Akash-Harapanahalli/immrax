"""Lohner's reachability algorithm.

Implements Taylor expansion with QR-based generator reduction for
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


class LohnerReachability(BaseSetGenerator):
    """Lohner's algorithm for reachability analysis.

    Uses Taylor expansion of the flow map combined with interval
    remainder bounds. Suitable for smooth nonlinear systems.

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
    .. [1] Lohner, R. J. "Enclosing the solutions of ordinary initial and
           boundary value problems." Computer Arithmetic, 1987.
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
        """Ensure constant shape (2n generators) for lax.scan compatibility."""
        n = Z.center.shape[0]
        target_gens = 2 * n
        current_gens = Z.generators.shape[1]
        
        if current_gens > target_gens:
             return Z.reduce_order(float(target_gens) / n)
        elif current_gens < target_gens:
             padding = jnp.zeros((n, target_gens - current_gens), dtype=Z.generators.dtype)
             new_G = jnp.concatenate([Z.generators, padding], axis=1)
             return Zonotope(Z.center, new_G)
        return Z

    def _taylor_map(self, t, x, f_args):
        """Compute Taylor expansion of the flow map at point x."""
        series_list = self._get_series(t, x, *f_args)
        series = jnp.stack(series_list)
        kk = jnp.arange(len(series))
        coeffs = (self.dt**kk * inv_fact(kk))[:, None]
        return jnp.sum(series * coeffs, axis=0)

    def step(self, t: float, Z, f_args):
        """Perform one step of Lohner's algorithm with QR reduction.

        1. Compute Taylor expansion and Jacobian
        2. Compute time and spatial remainders
        3. Propagate generators
        4. Apply QR-based reduction to control wrapping effect
        """
        dt = self.dt
        c = Z.center
        n = c.shape[0]

        # Compute rough enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # Taylor map and Jacobian
        c_new = self._taylor_map(t, c, f_args)
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)
        
        # Propagated generators
        G = Z.generators
        G_lin = A @ G
        shift = c_new - A @ c
        
        # --- Time Remainder ---
        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper
        p = self.taylor_order
        time_rem_coeff = (dt**(p + 1)) * inv_fact(p + 1)
        G_time_err = jnp.diag(time_rem_coeff * max_deriv)

        # --- Spatial Remainder ---
        def taylor_map_fn(x): 
            return self._taylor_map(t, x, f_args)
            
        hessian_fn = jax.jacfwd(jax.jacrev(taylor_map_fn))
        H_int = natif(hessian_fn)(R)
        
        # User requested using rough enclosure for second order terms
        dx_int = R - c
        H_abs = jnp.maximum(jnp.abs(H_int.lower), jnp.abs(H_int.upper))
        dx_abs = jnp.maximum(jnp.abs(dx_int.lower), jnp.abs(dx_int.upper))
        
        temp = jnp.dot(H_abs, dx_abs)
        spatial_err_bound = 0.5 * jnp.dot(temp, dx_abs)
        G_spatial_err = jnp.diag(spatial_err_bound)
        
        # --- Combine and Reduce (Lohner QR) ---
        # Combine all generators
        G_all = jnp.concatenate([G_lin, G_time_err, G_spatial_err], axis=1)
        
        # QR decomposition of the Jacobian A to find local frame
        # Q captures the rotation of the flow
        Q, _ = jnp.linalg.qr(A)
        
        # Project generators onto Q basis
        G_proj = Q.T @ G_all
        
        # Keep the first n generators (corresponds to A @ G_old's main axes if G_old was aligned)
        # Assuming G starts with n basis vectors.
        # If G grows, this assumption might drift, but Lohner resets G structure.
        # We take first n columns as the "main" parallelepiped.
        G_main = G_proj[:, :n]
        
        # The rest are treated as error/noise
        G_rest = G_proj[:, n:]
        
        # Bound the rest by an axis-aligned box (in Q frame)
        # Sum of absolute values of rows
        box_radius = jnp.sum(jnp.abs(G_rest), axis=1)
        G_box = jnp.diag(box_radius)
        
        # Reconstruct generators in original frame
        # G_new = Q @ [G_main, G_box]
        G_new_proj = jnp.concatenate([G_main, G_box], axis=1)
        G_new = Q @ G_new_proj
        
        # Result zonotope (c + shift is c_new)
        return Zonotope(c_new, G_new)
