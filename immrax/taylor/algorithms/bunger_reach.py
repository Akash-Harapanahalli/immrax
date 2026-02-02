"""Validated ODE integration using epsilon-inflation (Bünger 2019).

Implements the second validated integration algorithm from TaylorModels.jl,
adapted for JAX and immrax.  Each integration step produces a TaylorModel
whose domain variables are ``(dt, x_0)`` — time elapsed within the step and
initial-condition parameters.  The polynomial part is obtained by Taylor-
expanding the flow map jointly in ``(dt, x_0)`` via
``taylor_model_from_function``, and the remainder is validated using
epsilon-inflation of the Picard operator.

Algorithm overview
------------------
At each time step:

1. Extract the centre of the current spatial TaylorModel.

2. Select the step size from derivative norms at the centre.

3. Build the flow-map function φ(dt, x_0) using prolongation +
   Horner evaluation, and Taylor-expand it jointly in ``(dt, x_0)``
   with ``taylor_model_from_function``.

4. **Validate** the remainder using epsilon-inflation of the Picard
   operator (operating on the interval hull of the TM).

5. Replace the TM remainder with the Picard-validated remainder.

6. Evaluate the time-space TM at ``dt = h`` to obtain the next
   spatial TM (initial condition for the following step).

7. Store the time-space TM in the solution buffer.

Reference
---------
Florian Bünger, *A Taylor model toolbox for solving ODEs implemented in
MATLAB/INTLAB*, J. Comput. Appl. Math. 368 (2020), 112511.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from immrax.inclusion import Interval, interval, natif
from immrax.utils import inv_fact, prolongation

from immrax.taylor.taylor_model import (
    TaylorModel,
    taylor_model_from_interval,
    evaluate_at_variable,
    _get_canonical_exponents,
)
from immrax.taylor.algorithms.base import (
    TMFlowpipe,
    _step_size,
    _eval_taylor,
    _eval_taylor_interval,
    _is_contractive,
)


def _contractive_mask(delta: Interval, delta_x: Interval):
    """Return a boolean mask: ``True`` where component is contractive."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Picard operator (operates on interval enclosures)
# ---------------------------------------------------------------------------

def _picard_remainder(f: Callable, t0, derivs: list,
                      dt, order: int,
                      remainder: Interval,
                      include_initial_rem,
                      initial_rem: Interval) -> Interval:
    """Evaluate the Picard operator for the truncation remainder."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Epsilon-inflation validation (JAX-traceable)
# ---------------------------------------------------------------------------

def _validate_step(f: Callable, t0, derivs: list,
                   x_hull: Interval, dt, sign,
                   order: int, initial_rem: Interval,
                   red_abstol, adaptive: bool,
                   minabstol,
                   epsilon: float = 1e-10,
                   delta: float = 1e-6,
                   validatesteps: int = 30):
    """Validate one integration step using epsilon-inflation.

    Returns ``(success, remainder, dt, reduced_abstol)``.
    """
    n = x_hull.lower.shape[0]
    zero_n = jnp.zeros(n, dtype=x_hull.lower.dtype)

    eps_lo = jnp.full(n, 1.0 - epsilon, dtype=x_hull.lower.dtype)
    eps_hi = jnp.full(n, 1.0 + epsilon, dtype=x_hull.lower.dtype)
    delta_lo = jnp.full(n, -delta, dtype=x_hull.lower.dtype)
    delta_hi = jnp.full(n, delta, dtype=x_hull.lower.dtype)

    def outer_cond(state):
        success, _, _, _, _, continue_flag = state
        return (~success) & continue_flag

    def outer_body(state):
        success, _, _, dt_, ra, _ = state

        # TODO: compute initial E and E_prime

        def inner_body(carry, _):
            s, e_lo, e_hi = carry

            def do_step(args):
                el, eh = args
                # TODO: Picard iteration + contractive check + inflation
                raise NotImplementedError

            def no_op(args):
                el, eh = args
                return jnp.bool_(True), el, eh

            new_s, new_el, new_eh = lax.cond(s, no_op, do_step, (e_lo, e_hi))
            return (new_s, new_el, new_eh), None

        (inner_success, final_el, final_eh), _ = lax.scan(
            inner_body,
            (jnp.bool_(False), zero_n, zero_n),
            None, length=validatesteps)

        # TODO: update dt_, ra based on inner_success
        raise NotImplementedError

    final_state = lax.while_loop(
        outer_cond, outer_body,
        (jnp.bool_(False), zero_n, zero_n, dt, red_abstol, jnp.bool_(True)))

    success, e_lo, e_hi, dt_out, ra_out, _ = final_state
    return success, interval(e_lo, e_hi), dt_out, ra_out


# ---------------------------------------------------------------------------
# Flow-map function builder
# ---------------------------------------------------------------------------

def _make_flow_map(f: Callable, order_time: int):
    """Build the flow map φ(t0, dt, x0) = eval_taylor(prolongation(f)(t0, x0), dt).

    Returns a function ``(t0, z) -> x`` where ``z = [dt, x0]``.
    ``t0`` is a separate argument so the function identity is stable
    across integration steps (avoiding repeated re-tracing).
    """
    raise NotImplementedError


def _build_tm_from_flow(f: Callable, order_time: int, n_state: int,
                         max_order: int):
    """Pre-build the jacfwd chain for the flow map and return a callable
    that efficiently computes the TaylorModel at any (t0, center_z, radius_z).

    The expensive jacfwd chain is built once; subsequent calls just evaluate
    the JIT-compiled derivative functions.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validated_integ2(
    f: Callable,
    x0,
    t0: float,
    tmax: float,
    order_time: int,
    order_space: int = 2,
    abstol: float = 1e-10,
    maxsteps: int = 2000,
    adaptive: bool = True,
    minabstol: float = 1e-50,
    epsilon: float = 1e-10,
    delta: float = 1e-6,
    validatesteps: int = 30,
) -> TMFlowpipe:
    r"""Validated ODE integration using epsilon-inflation.

    Computes a rigorous flowpipe for ``ẋ = f(t, x)``, ``x(t₀) ∈ X₀``,
    propagating TaylorModels between steps to preserve polynomial
    dependence on initial conditions.

    Parameters
    ----------
    f : (float, Array) -> Array
        Right-hand side of the ODE.
    x0 : Interval or TaylorModel
        Initial condition set.  If an ``Interval``, it is converted to a
        ``TaylorModel`` of order ``order_space``.
    t0 : float
        Start time.
    tmax : float
        End time.
    order_time : int
        Order of the Taylor expansion in time.
    order_space : int
        Order of the Taylor expansion in initial-condition variables
        (default 2).
    abstol : float
        Absolute tolerance for step-size selection (default ``1e-10``).
    maxsteps : int
        Maximum number of steps (default 2000).
    adaptive : bool
        Reduce step size on validation failure (default ``True``).
    minabstol : float
        Minimum tolerance before giving up (default ``1e-50``).
    epsilon : float
        Multiplicative inflation factor (default ``1e-10``).
    delta : float
        Additive perturbation (default ``1e-6``).
    validatesteps : int
        Maximum Picard iterations per step (default 30).

    Returns
    -------
    TMFlowpipe
        Contains ``times``, TaylorModel buffers, ``nsteps``, and ``success``.
    """
    # -- Convert x0 to TaylorModel if needed --
    if isinstance(x0, Interval):
        x_tm = taylor_model_from_interval(x0, order=order_space)
    elif isinstance(x0, TaylorModel):
        x_tm = x0
    else:
        raise TypeError(f"x0 must be Interval or TaylorModel, got {type(x0)}")

    n = x_tm.shape[0] if len(x_tm.shape) > 0 else 1
    d_space = x_tm.d
    d_full = 1 + d_space
    sign = jnp.where(tmax > t0, 1.0, -1.0)
    dtype = x_tm.coeffs.dtype
    t0_arr = jnp.asarray(t0, dtype=dtype)
    tmax_arr = jnp.asarray(tmax, dtype=dtype)
    abstol_arr = jnp.asarray(abstol, dtype=dtype)
    minabstol_arr = jnp.asarray(minabstol, dtype=dtype)
    max_order = max(order_time, order_space)

    # -- Pre-allocate output buffers --
    # TODO: allocate times, coeffs, remainders, etc. for maxsteps+1

    # -- Initial carry state --
    # TODO: pack (t_cur, x_tm_cur flattened, red_abstol, success, nsteps)

    def step_body(carry, _):
        # TODO: unpack carry

        done = jnp.bool_(False)  # TODO: t_cur * sign >= tmax * sign

        def do_step(carry_inner):
            # TODO: steps 1-7 of the algorithm
            raise NotImplementedError

        def no_op(carry_inner):
            # Return carry unchanged, emit a dummy output slice
            return carry_inner

        # TODO: new_carry = lax.cond(done, no_op, do_step, carry)
        # TODO: return new_carry, output_slice
        raise NotImplementedError

    # -- Scan over maxsteps --
    # final_carry, output_slices = lax.scan(step_body, init_carry, None, length=maxsteps)

    # TODO: assemble TMFlowpipe from output_slices and final_carry
    raise NotImplementedError


def _spatial_tm_to_full(
    x_tm: TaylorModel, t_center, t_radius, max_order: int
) -> TaylorModel:
    """Embed a spatial TM into a time-space TM with zero-width time domain.

    Adds a time variable (index 0) to the domain.  All exponents for the
    time variable are zero (the polynomial doesn't depend on time).
    """
    raise NotImplementedError
