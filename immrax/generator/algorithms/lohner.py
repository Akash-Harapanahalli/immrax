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
        """Reduce set order using polymorphic reduce_order method."""
        if hasattr(Z, 'reduce_order'):
            return Z.reduce_order(self.target_order)
        return Z

    def _taylor_map(self, t, x, f_args):
        """Compute Taylor expansion of the flow map at point x.

        Evaluates φ(x) ≈ x + dt*f(x) + dt²/2*f'(x)*f(x) + ...
        """
        series = self._get_series(t, x, *f_args)
        res = jnp.zeros_like(x)
        dt_pow = 1.0
        for k, term in enumerate(series):
            if k > 0:
                dt_pow *= self.dt
            res = res + (term * dt_pow * inv_fact(k))
        return res

    def step(self, t: float, Z, f_args):
        """Perform one step of Lohner's algorithm.

        1. Compute Taylor expansion of flow at center
        2. Linearize around center using Jacobian
        3. Bound remainder using interval arithmetic
        4. Combine linear image with remainder bound
        """
        dt = self.dt
        c = Z.center

        # Compute rough enclosure for remainder bounds
        R = self._compute_rough_enclosure(t, Z, f_args)

        # Taylor map at center
        c_new = self._taylor_map(t, c, f_args)

        # Jacobian of Taylor map at center
        A = jax.jacfwd(lambda x: self._taylor_map(t, x, f_args))(c)

        # Linear image of set
        Z_lin = A @ Z

        # Recenter to Taylor map center
        shift = c_new - A @ c
        Z_lin = Z_lin + shift

        # Bound highest derivative for remainder
        def get_highest_derivative_norm(x):
            series = self._get_series(t, x, *f_args)
            t_series = [1.] + [0.] * (len(series) - 1)
            _, out_series = jet(self.sys.f, (t, x), (t_series, series))
            return jnp.abs(out_series[-1])

        max_deriv = natif(get_highest_derivative_norm)(R).upper

        # Remainder bound coefficient
        p = self.taylor_order
        coeff = (dt**(p + 1)) * inv_fact(p + 1)

        # Error zonotope (works with any set type via __add__)
        gen_rem = coeff * max_deriv
        Z_error = Zonotope(jnp.zeros_like(c), jnp.diag(gen_rem))

        return Z_lin + Z_error
