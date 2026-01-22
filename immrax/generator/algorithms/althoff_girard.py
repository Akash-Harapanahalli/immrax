"""Althoff-Girard reachability algorithm.

Implements conservative linearization with matrix exponential for
computing reachable sets of nonlinear systems.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from typing import Any

from ...system import System
from ...inclusion import Interval, natif
from ..sets import Zonotope
from .base import BaseSetGenerator


class AlthoffGirardReachability(BaseSetGenerator):
    """Althoff-Girard algorithm for reachability analysis.

    Uses conservative linearization around the set center combined with
    exact matrix exponential computation. The linearization error is
    bounded using Hessian-based Lagrange remainder estimates.

    Parameters
    ----------
    sys : System
        The dynamical system
    dt : float
        Time step size
    target_order : float
        Target order for set reduction (generators / dimension)

    References
    ----------
    .. [1] Althoff, M., et al. "Reachability analysis of nonlinear systems
           with uncertain parameters using conservative linearization."
           CDC 2008.
    .. [2] Girard, A. "Reachability of uncertain linear systems using
           zonotopes." HSCC 2005.
    """
    target_order: float

    def __init__(
        self,
        sys: System,
        dt: float,
        target_order: float = 2.0
    ):
        super().__init__(sys, dt)
        self.target_order = float(target_order)

    def _enforce_limit(self, Z) -> Any:
        """Reduce set order using polymorphic reduce_order method."""
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def step(self, t: float, Z, f_args):
        """Perform one step of Althoff-Girard algorithm.

        1. Linearize dynamics around center
        2. Compute matrix exponential for exact linear propagation
        3. Bound linearization error using Hessian
        4. Combine linear image with error bound
        """
        dt = self.dt
        c = Z.center
        n = c.shape[0]

        # Compute rough enclosure
        R = self._compute_rough_enclosure(t, Z, f_args)

        # Evaluate dynamics and Jacobian at center
        def f_x(x):
            return self.sys.f(t, x, *f_args)
        J_c = jax.jacfwd(f_x)(c)
        f_c = f_x(c)

        # Linearization error bound using Lagrange remainder
        # Error <= 0.5 * max(||H||) * ||x - c||^2
        delta = R - Interval(c, c)
        delta_max = jnp.maximum(jnp.abs(delta.lower), jnp.abs(delta.upper))
        dx_sq_sum = (jnp.sum(delta_max))**2

        def get_hess_bound(i):
            """Bound the i-th component's Hessian over the enclosure."""
            h_max = natif(
                lambda x: jnp.max(jnp.abs(
                    jax.hessian(lambda s: self.sys.f(t, s, *f_args)[i])(x)
                ))
            )(R).upper
            return 0.5 * h_max * dx_sq_sum

        L_bound = jax.vmap(get_hess_bound)(jnp.arange(n))

        # Exact matrix exponential computation
        # exp([J*dt, I*dt; 0, 0]) = [Phi, int_Phi; 0, I]
        M = jnp.zeros((2*n, 2*n))
        M = M.at[:n, :n].set(J_c * dt)
        M = M.at[:n, n:].set(jnp.eye(n) * dt)
        expM = jax.scipy.linalg.expm(M)
        Phi = expM[:n, :n]
        int_Phi = expM[:n, n:]

        # Affine term
        v = f_c - J_c @ c

        # Linear image of set
        Z_lin = Phi @ Z

        # Translation by integral term
        trans = int_Phi @ v
        Z_lin = Z_lin + trans

        # Error zonotope from linearization error
        Z_error = Zonotope(jnp.zeros_like(c), jnp.diag(dt * L_bound))

        return Z_lin + Z_error
