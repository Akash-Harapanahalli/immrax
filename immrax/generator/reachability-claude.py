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
from jaxtyping import Array

from ..system import System
from ..inclusion import Interval, natif
from .sets import Zonotope, zonotope, zonotope_from_interval


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
        # (JAX scan requires fixed shapes)
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
                padding = jnp.zeros((Z.n, target_m - Z.m), dtype=Z.dtype)
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


class LohnerReachability(ZonotopeReachability):
    """Lohner's algorithm for reachability with QR-based generator management.

    This algorithm maintains a fixed number of generators by using QR
    decomposition to rotate and compress the generator matrix. The zonotope
    is represented as Z = c + Q @ R @ B where Q is orthogonal and B is the
    unit box.

    Parameters
    ----------
    sys : System
        The dynamical system
    dt : float
        Time step
    max_generators : int
        Maximum number of generators to maintain
    taylor_order : int
        Order of Taylor expansion for the flow (default: 2)

    References
    ----------
    Lohner, R. (1992). "Computation of guaranteed enclosures for the solutions
    of ordinary initial and boundary value problems."
    """

    max_generators: int
    taylor_order: int

    def __init__(
        self,
        sys: System,
        dt: float,
        max_generators: Optional[int] = None,
        taylor_order: int = 2,
    ) -> None:
        super().__init__(sys, dt)
        self.max_generators = max_generators
        self.taylor_order = taylor_order

    def _reduce_generators(self, Z: Zonotope) -> Zonotope:
        """Reduce generators using QR decomposition.

        Maintains a fixed number of generators by:
        1. Computing QR decomposition of G
        2. Keeping the most significant generators
        3. Bounding the rest with an interval and adding as axis-aligned generators
        """
        if self.max_generators is None or Z.m <= self.max_generators:
            return Z

        G = Z.G
        n, m = G.shape

        # Sort generators by their norm (importance)
        norms = jnp.linalg.norm(G, axis=0)
        sorted_indices = jnp.argsort(-norms)  # Descending order
        G_sorted = G[:, sorted_indices]

        # Keep the most important generators
        n_keep = self.max_generators - n  # Reserve n for interval remainder
        G_keep = G_sorted[:, :n_keep]
        G_reduce = G_sorted[:, n_keep:]

        # Bound the reduced generators with an interval hull
        # The reduced generators contribute |G_reduce| @ 1 to each dimension
        remainder = jnp.sum(jnp.abs(G_reduce), axis=1)
        G_remainder = jnp.diag(remainder)

        # Combine kept generators with remainder
        G_new = jnp.concatenate([G_keep, G_remainder], axis=1)

        return Zonotope(Z.ox, G_new)

    def _qr_reorthogonalize(self, Z: Zonotope) -> Zonotope:
        """Reorthogonalize generators using QR decomposition.

        This prevents the "wrapping effect" where generators become
        increasingly aligned, causing overapproximation.
        """
        G = Z.G
        n, m = G.shape

        if m == 0:
            return Z

        # QR decomposition: G = Q @ R
        Q, R = jnp.linalg.qr(G, mode='reduced')

        # The new zonotope is c + Q @ R @ B = c + Q @ (R @ B)
        # Since B is the unit box, R @ B is bounded by sum of |R| rows
        # We represent this as Q @ diag(row_sums)
        row_sums = jnp.sum(jnp.abs(R), axis=1)
        G_new = Q * row_sums[None, :]

        return Zonotope(Z.ox, G_new)

    def _compute_jacobian_bounds(self, t: float, Z: Zonotope) -> Tuple[Array, Array]:
        """Compute bounds on the Jacobian over the zonotope.

        Returns interval bounds [J_lo, J_hi] such that
        J_lo <= df/dx(t,x) <= J_hi for all x in Z.
        """
        # Get interval hull of zonotope
        hull = Z.interval_hull()

        # Compute Jacobian at center
        def f_x(x):
            return self.sys.f(t, x)

        J_center = jax.jacfwd(f_x)(Z.ox)

        # Use natural interval extension to bound Jacobian entries
        # This is a simplification - in practice you'd want tighter bounds
        def jacobian_entry(i, j):
            def f_ij(x):
                return jax.jacfwd(f_x)(x)[i, j]
            return natif(f_ij)(hull)

        n = Z.n
        J_lo = jnp.zeros((n, n))
        J_hi = jnp.zeros((n, n))

        for i in range(n):
            for j in range(n):
                ij_bounds = jacobian_entry(i, j)
                J_lo = J_lo.at[i, j].set(ij_bounds.lower)
                J_hi = J_hi.at[i, j].set(ij_bounds.upper)

        return J_lo, J_hi

    def step(self, t: float, Z: Zonotope) -> Zonotope:
        """Perform one step of Lohner's algorithm.

        Uses a Taylor expansion of the flow map:
        φ(t+dt, x) ≈ x + dt*f(t,x) + (dt²/2)*f'(t,x)*f(t,x) + ...

        The zonotope is propagated through this map with interval
        remainder bounds.
        """
        dt = self.dt
        c = Z.ox
        G = Z.G
        n = Z.n

        # Evaluate dynamics at center
        f_c = self.sys.f(t, c)

        # Compute Jacobian at center
        def f_x(x):
            return self.sys.f(t, x)
        A = jax.jacfwd(f_x)(c)

        # First-order Taylor: φ(x) ≈ x + dt*f(t,x)
        # For zonotope: c_new = c + dt*f(c), G_new = G + dt*A@G

        # Second-order correction for center
        if self.taylor_order >= 2:
            # df/dt + df/dx * f
            def f_t(t_):
                return self.sys.f(t_, c)
            dfdt = jax.jacfwd(f_t)(t)
            second_order = dfdt + A @ f_c
            c_new = c + dt * f_c + 0.5 * dt**2 * second_order
        else:
            c_new = c + dt * f_c

        # Propagate generators: G_new = (I + dt*A) @ G
        # This is the linear part of the flow
        G_linear = G + dt * (A @ G)

        # Compute remainder bound using interval arithmetic
        # The remainder accounts for nonlinearity over the zonotope
        hull = Z.interval_hull()

        # Bound the nonlinear remainder: ||f(x) - f(c) - A(x-c)||
        # Use natural interval extension
        def remainder_func(x):
            return self.sys.f(t, x) - f_c - A @ (x - c)

        remainder_interval = natif(remainder_func)(hull)
        remainder_radius = jnp.maximum(
            jnp.abs(remainder_interval.lower),
            jnp.abs(remainder_interval.upper)
        )

        # Add remainder as axis-aligned generators
        G_remainder = dt * jnp.diag(remainder_radius)

        # Combine generators
        G_new = jnp.concatenate([G_linear, G_remainder], axis=1)

        Z_new = Zonotope(c_new, G_new)

        # Reduce generators if needed
        Z_new = self._reduce_generators(Z_new)

        # Reorthogonalize to prevent wrapping
        Z_new = self._qr_reorthogonalize(Z_new)

        return Z_new


# --- Althoff-Girard Algorithm ---


class AlthoffGirardReachability(ZonotopeReachability):
    """Althoff-Girard algorithm with conservative linearization.

    This algorithm uses:
    1. Conservative linearization: f(x) ≈ f(c) + A(x-c) + L
    2. Lagrange remainder bounding using interval arithmetic
    3. A two-phase approach: rough enclosure then refinement

    Parameters
    ----------
    sys : System
        The dynamical system
    dt : float
        Time step
    max_generators : int, optional
        Maximum generators before reduction (default: 2*n)
    reduction_method : str
        Method for generator reduction: 'girard' or 'combastel'

    References
    ----------
    Althoff, M., Stursberg, O., & Buss, M. (2008). "Reachability analysis of
    nonlinear systems with uncertain parameters using conservative linearization."
    Girard, A. (2005). "Reachability of uncertain linear systems using zonotopes."
    """

    max_generators: int
    reduction_method: str

    def __init__(
        self,
        sys: System,
        dt: float,
        max_generators: Optional[int] = None,
        reduction_method: str = 'girard',
    ) -> None:
        super().__init__(sys, dt)
        self.reduction_method = reduction_method
        # Default: allow 2*n generators
        self.max_generators = max_generators if max_generators is not None else 2 * sys.xlen

    def _reduce_girard(self, Z: Zonotope) -> Zonotope:
        """Girard's generator reduction method.

        Reduces generators by bounding the smallest ones with an interval
        and representing them as axis-aligned generators.
        """
        if Z.m <= self.max_generators:
            return Z

        G = Z.G
        n, m = G.shape

        # Number of generators to reduce
        n_reduce = m - self.max_generators + n  # Keep max_generators - n, add n for box

        # Sort by norm (smallest first for reduction)
        norms = jnp.linalg.norm(G, axis=0)
        sorted_indices = jnp.argsort(norms)

        # Generators to reduce (smallest norms)
        reduce_indices = sorted_indices[:n_reduce]
        keep_indices = sorted_indices[n_reduce:]

        G_keep = G[:, keep_indices]
        G_reduce = G[:, reduce_indices]

        # Bound reduced generators with interval hull
        remainder = jnp.sum(jnp.abs(G_reduce), axis=1)
        G_box = jnp.diag(remainder)

        G_new = jnp.concatenate([G_keep, G_box], axis=1)

        return Zonotope(Z.ox, G_new)

    def _compute_lagrange_remainder(
        self,
        t: float,
        Z: Zonotope,
        A: Array,
    ) -> Array:
        """Compute the Lagrange remainder bound for conservative linearization.

        For f(x) = f(c) + A(x-c) + L, bounds ||L|| where L is the
        Lagrange remainder of the first-order Taylor expansion.

        The remainder is bounded using the Hessian:
        L_i <= (1/2) * max_{x in Z} ||H_i(x)|| * ||x - c||^2

        where H_i is the Hessian of f_i.
        """
        c = Z.ox
        n = Z.n
        hull = Z.interval_hull()

        # Compute maximum deviation from center
        radius = (hull.upper - hull.lower) / 2
        max_deviation_sq = jnp.sum(radius**2)

        # Compute Hessian at center and bound over the zonotope
        # For efficiency, we compute the Hessian at center and add a margin
        remainder_bound = jnp.zeros(n)

        for i in range(n):
            def f_i(x):
                return self.sys.f(t, x)[i]

            # Hessian at center
            H_center = jax.hessian(f_i)(c)

            # For the Lagrange remainder, we need to bound the Hessian over Z
            # Use a simple approach: evaluate at corners of the interval hull
            # and take the maximum
            H_max = jnp.max(jnp.abs(H_center))

            # Also evaluate at interval corners for better bounds
            corners = jnp.array([
                [hull.lower[0], hull.lower[1]] if n >= 2 else [hull.lower[0]],
                [hull.upper[0], hull.upper[1]] if n >= 2 else [hull.upper[0]],
            ])

            for corner in corners[:min(2, 2**n)]:
                if n == 1:
                    corner = jnp.array([corner])
                H_corner = jax.hessian(f_i)(corner[:n])
                H_max = jnp.maximum(H_max, jnp.max(jnp.abs(H_corner)))

            # Lagrange remainder bound: (1/2) * ||H|| * ||x-c||^2
            remainder_bound = remainder_bound.at[i].set(0.5 * H_max * n * max_deviation_sq)

        return remainder_bound

    def _linear_reach_step(
        self,
        A: Array,
        b: Array,
        Z: Zonotope,
        dt: float,
    ) -> Zonotope:
        """Compute one step for affine system dx/dt = Ax + b.

        Uses the matrix exponential and its integral:
        x(t+dt) = exp(A*dt) @ x(t) + ∫₀^dt exp(A*s) ds @ b
        """
        n = A.shape[0]

        # For small dt, use Taylor approximation of matrix exponential
        # exp(A*dt) ≈ I + A*dt + (A*dt)²/2 + ...
        I = jnp.eye(n)
        eA = I + dt * A + 0.5 * (dt**2) * (A @ A)

        # Integral of exp(A*s) from 0 to dt ≈ dt*I + (dt²/2)*A + ...
        int_eA = dt * I + 0.5 * (dt**2) * A

        # Propagate zonotope
        c_new = eA @ Z.ox + int_eA @ b
        G_new = eA @ Z.G

        return Zonotope(c_new, G_new)

    def _bloating_factor(self, t: float, Z: Zonotope, dt: float) -> Zonotope:
        """Compute the bloating zonotope that accounts for all trajectories over [t, t+dt].

        This ensures the reachable tube contains all states, not just the endpoints.
        """
        c = Z.ox
        n = Z.n

        # Compute velocity bounds over the zonotope
        hull = Z.interval_hull()

        def f_interval(x):
            return self.sys.f(t, x)

        f_bounds = natif(f_interval)(hull)

        # Maximum velocity in each direction
        v_max = jnp.maximum(jnp.abs(f_bounds.lower), jnp.abs(f_bounds.upper))

        # Bloating: over time dt, state can deviate by at most v_max * dt
        bloat_radius = v_max * dt
        G_bloat = jnp.diag(bloat_radius)

        return Zonotope(jnp.zeros(n), G_bloat)

    def step(self, t: float, Z: Zonotope) -> Zonotope:
        """Perform one step of the Althoff-Girard algorithm.

        Steps:
        1. Compute Jacobian A at zonotope center
        2. Compute Lagrange remainder bound L
        3. Propagate zonotope through linearized system
        4. Add remainder as generators
        5. Reduce generators if needed
        """
        dt = self.dt
        c = Z.ox
        n = Z.n

        # Step 1: Linearize at center
        # f(x) ≈ f(c) + A @ (x - c) = (f(c) - A @ c) + A @ x = b + A @ x
        def f_x(x):
            return self.sys.f(t, x)

        f_c = self.sys.f(t, c)
        A = jax.jacfwd(f_x)(c)

        # Affine offset: b = f(c) - A @ c
        b = f_c - A @ c

        # Step 2: Compute Lagrange remainder bound
        L_bound = self._compute_lagrange_remainder(t, Z, A)

        # Step 3: Propagate through linearized affine system dx/dt = Ax + b
        Z_linear = self._linear_reach_step(A, b, Z, dt)

        # Step 4: Add remainder as generators
        # The Lagrange remainder contributes ∫₀^dt L ds ≈ dt * L
        G_remainder = dt * jnp.diag(L_bound)

        G_combined = jnp.concatenate([Z_linear.G, G_remainder], axis=1)
        Z_new = Zonotope(Z_linear.ox, G_combined)

        # Step 5: Add bloating for continuous-time overapproximation
        Z_bloat = self._bloating_factor(t, Z, dt)
        Z_new = Z_new + Z_bloat

        # Step 6: Reduce generators
        Z_new = self._reduce_girard(Z_new)

        return Z_new


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
    """Compute reachable tube using Lohner's algorithm.

    Parameters
    ----------
    sys : System
        Continuous dynamical system
    t0, tf : float
        Time interval
    Z0 : Zonotope
        Initial set
    dt : float
        Time step
    max_generators : int, optional
        Maximum number of generators
    taylor_order : int
        Order of Taylor expansion

    Returns
    -------
    ZonotopeReachTube
        The computed reachable tube
    """
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
    """Compute reachable tube using Althoff-Girard algorithm.

    Parameters
    ----------
    sys : System
        Continuous dynamical system
    t0, tf : float
        Time interval
    Z0 : Zonotope
        Initial set
    dt : float
        Time step
    max_generators : int, optional
        Maximum number of generators

    Returns
    -------
    ZonotopeReachTube
        The computed reachable tube
    """
    reach = AlthoffGirardReachability(sys, dt, max_generators)
    return reach.compute_reachtube(t0, tf, Z0)
