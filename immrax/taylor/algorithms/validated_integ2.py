"""Validated ODE integration using epsilon-inflation (Bünger 2019).

Implements the second validated integration algorithm from TaylorModels.jl,
adapted for JAX and immrax.

Algorithm overview
------------------
At each time step:

1. Compute time Taylor coefficients ``x^(k)`` at the centre of the current
   set using jet-based prolongation.

2. Select the step size from Taylor coefficient norms.

3. **Validate** the remainder using epsilon-inflation:

   a. Start with ``E = 0`` (zero remainder).
   b. Compute ``E' = Picard(E)`` including the initial-condition remainder.
   c. For each subsequent iteration, compute ``E' = Picard(E)`` (without
      initial remainder) and check componentwise contractivity.
   d. For every component where ``E'_i ⊄ int(E_i)``, inflate::

          E_i ← E'_i · (1 ± ε) + [-δ, δ]

   e. Repeat until contractive or ``validatesteps`` exhausted.

4. Build the flowpipe enclosure from the Taylor polynomial + validated
   remainder.

5. If validation fails and ``adaptive=True``, shrink the step size and
   retry.

Key difference from ``validated_integ``
---------------------------------------
Instead of iterating the Picard operator with a widening strategy (method 1),
this algorithm *systematically inflates* only the non-contractive components.
This is generally simpler and converges faster for non-stiff problems.

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

from immrax.taylor.algorithms.base import (
    ValidatedSolution,
    _step_size,
    _eval_taylor,
    _eval_taylor_interval,
    _is_contractive,
)


def _contractive_mask(delta: Interval, delta_x: Interval):
    """Return a boolean mask: ``True`` where component is contractive."""
    strict = (delta.lower > delta_x.lower) & (delta.upper < delta_x.upper)
    both_zero = ((delta.lower == 0) & (delta.upper == 0) &
                 (delta_x.lower == 0) & (delta_x.upper == 0))
    return strict | both_zero


# ---------------------------------------------------------------------------
# Picard operator
# ---------------------------------------------------------------------------

def _picard_remainder(f: Callable, t0, derivs: list,
                      dt, order: int,
                      remainder: Interval,
                      include_initial_rem,
                      initial_rem: Interval) -> Interval:
    """Evaluate the Picard operator for the truncation remainder."""
    abs_h = jnp.abs(jnp.asarray(dt))
    h_iv = interval(jnp.zeros(()), abs_h)

    x_poly = _eval_taylor_interval(derivs, h_iv, order)
    extended = x_poly + remainder
    f_extended = natif(lambda x: f(t0, x))(extended)

    xdot_poly = _eval_taylor_interval(derivs[1:], h_iv, order - 1)
    residual = f_extended - xdot_poly
    picard = h_iv * residual

    # Conditionally add initial remainder (JAX-traceable)
    picard_lo = jnp.where(include_initial_rem,
                          picard.lower + initial_rem.lower, picard.lower)
    picard_hi = jnp.where(include_initial_rem,
                          picard.upper + initial_rem.upper, picard.upper)
    return interval(picard_lo, picard_hi)


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

    # Outer adaptive retry via lax.while_loop
    # State: (success, E_lo, E_hi, dt, red_abstol, continue_flag)
    def outer_cond(state):
        success, _, _, _, _, continue_flag = state
        return (~success) & continue_flag

    def outer_body(state):
        success, _, _, dt_, ra, _ = state

        # 0-th iteration: include initial remainder
        E = interval(zero_n, zero_n)
        E_prime = _picard_remainder(f, t0, derivs, dt_, order,
                                     E, jnp.bool_(True), initial_rem)
        # E = E_prime after 0th step

        # Inner inflation loop via lax.scan
        def inner_body(carry, _):
            s, e_lo, e_hi = carry

            def do_step(args):
                el, eh = args
                E_cur = interval(el, eh)
                E_p = _picard_remainder(f, t0, derivs, dt_, order,
                                         E_cur, jnp.bool_(False), initial_rem)
                contractive = _is_contractive(E_p, E_cur)

                # Inflate non-contractive components
                mask = _contractive_mask(E_p, E_cur)
                inflated_lo = E_p.lower * eps_lo + delta_lo
                inflated_hi = E_p.upper * eps_hi + delta_hi

                new_lo = jnp.where(mask, el, inflated_lo)
                new_hi = jnp.where(mask, eh, inflated_hi)

                return contractive, new_lo, new_hi

            def no_op(args):
                el, eh = args
                return jnp.bool_(True), el, eh

            new_s, new_el, new_eh = lax.cond(s, no_op, do_step, (e_lo, e_hi))
            return (new_s, new_el, new_eh), None

        (inner_success, final_el, final_eh), _ = lax.scan(
            inner_body,
            (jnp.bool_(False), E_prime.lower, E_prime.upper),
            None, length=validatesteps)

        # Adaptive reduction if not successful
        new_ra = jnp.where(inner_success, ra, ra / 10.0)
        new_dt = jnp.where(inner_success, dt_, dt_ * 0.1 ** (1.0 / order))

        can_continue = jnp.where(
            inner_success, jnp.bool_(False),
            jnp.bool_(adaptive) & (ra > minabstol))

        out_el = jnp.where(inner_success, final_el, zero_n)
        out_eh = jnp.where(inner_success, final_eh, zero_n)
        out_dt = jnp.where(inner_success, dt_, new_dt)
        out_ra = jnp.where(inner_success, ra, new_ra)

        return (inner_success, out_el, out_eh, out_dt, out_ra, can_continue)

    final_state = lax.while_loop(
        outer_cond, outer_body,
        (jnp.bool_(False), zero_n, zero_n, dt, red_abstol, jnp.bool_(True)))

    success, e_lo, e_hi, dt_out, ra_out, _ = final_state
    return success, interval(e_lo, e_hi), dt_out, ra_out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validated_integ2(
    f: Callable,
    x0: Interval,
    t0: float,
    tmax: float,
    order_time: int,
    abstol: float = 1e-10,
    maxsteps: int = 2000,
    adaptive: bool = True,
    minabstol: float = 1e-50,
    epsilon: float = 1e-10,
    delta: float = 1e-6,
    validatesteps: int = 30,
) -> ValidatedSolution:
    """Validated ODE integration using epsilon-inflation.

    Computes a rigorous flowpipe for ``ẋ = f(t, x)``, ``x(t₀) ∈ X₀``.

    Parameters
    ----------
    f : (float, Array) -> Array
        Right-hand side of the ODE.
    x0 : Interval
        Initial condition set (interval box).
    t0 : float
        Start time.
    tmax : float
        End time.
    order_time : int
        Order of the Taylor expansion in time.
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
    ValidatedSolution
        Contains ``times``, ``flowpipe_lo``, ``flowpipe_hi``, ``nsteps``,
        and ``success``.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from immrax.inclusion import icentpert
    >>> from immrax.taylor.algorithms import validated_integ2
    >>>
    >>> f = lambda t, x: -x
    >>> x0 = icentpert(jnp.array([1.0]), jnp.array([0.1]))
    >>> sol = validated_integ2(f, x0, 0.0, 1.0, order_time=5)
    """
    n = x0.lower.shape[0]
    sign = jnp.where(tmax > t0, 1.0, -1.0)
    t0_arr = jnp.asarray(t0, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    tmax_arr = jnp.asarray(tmax, dtype=t0_arr.dtype)
    abstol_arr = jnp.asarray(abstol, dtype=t0_arr.dtype)
    minabstol_arr = jnp.asarray(minabstol, dtype=t0_arr.dtype)

    # Pre-allocate output arrays
    times_buf = jnp.full((maxsteps + 1,), tmax, dtype=t0_arr.dtype)
    times_buf = times_buf.at[0].set(t0_arr)
    fp_lo_buf = jnp.zeros((maxsteps + 1, n), dtype=x0.lower.dtype)
    fp_hi_buf = jnp.zeros((maxsteps + 1, n), dtype=x0.lower.dtype)
    fp_lo_buf = fp_lo_buf.at[0].set(x0.lower)
    fp_hi_buf = fp_hi_buf.at[0].set(x0.upper)

    center0 = x0.center
    init_rem_lo = x0.lower - center0
    init_rem_hi = x0.upper - center0

    init_carry = (
        t0_arr,
        x0.lower,
        x0.upper,
        init_rem_lo,  # rem_lo
        init_rem_hi,  # rem_hi
        abstol_arr,
        jnp.bool_(True),  # success
        jnp.bool_(False),  # done
        jnp.int32(0),  # nsteps
        times_buf,
        fp_lo_buf,
        fp_hi_buf,
    )

    def scan_body(carry, _):
        (t, xh_lo, xh_hi, rem_lo, rem_hi,
         red_abstol, success, done, nsteps,
         t_buf, flo_buf, fhi_buf) = carry

        def real_step(carry_in):
            (t_, xh_lo_, xh_hi_, rem_lo_, rem_hi_,
             ra_, success_, nsteps_,
             t_buf_, flo_buf_, fhi_buf_) = carry_in

            x_hull = interval(xh_lo_, xh_hi_)
            center = x_hull.center
            rem = interval(rem_lo_, rem_hi_)

            # 1. Taylor coefficients
            derivs = prolongation(f, order_time)(t_, center)

            # 2. Step size
            dt = _step_size(derivs, ra_, order_time)
            dt = jnp.minimum(dt, sign * (tmax_arr - t_))
            dt = sign * dt

            # 3. Validate via epsilon-inflation
            step_ok, remainder, dt, ra_ = _validate_step(
                f, t_, derivs, x_hull, dt, sign, order_time, rem,
                ra_, adaptive, minabstol_arr,
                epsilon=epsilon, delta=delta, validatesteps=validatesteps)

            # 4. Build enclosure
            x_next_center = _eval_taylor(derivs, dt, order_time)
            enc_lo = x_next_center + remainder.lower
            enc_hi = x_next_center + remainder.upper

            new_t = t_ + dt
            new_nsteps = nsteps_ + 1

            t_buf_ = t_buf_.at[new_nsteps].set(new_t)
            flo_buf_ = flo_buf_.at[new_nsteps].set(enc_lo)
            fhi_buf_ = fhi_buf_.at[new_nsteps].set(enc_hi)

            new_ra = jnp.where(jnp.bool_(adaptive),
                               jnp.minimum(abstol_arr, 10.0 * ra_), ra_)
            new_success = success_ & step_ok

            return (new_t, enc_lo, enc_hi,
                    remainder.lower, remainder.upper,
                    new_ra, new_success, new_nsteps,
                    t_buf_, flo_buf_, fhi_buf_)

        def null_step(carry_in):
            (t_, xh_lo_, xh_hi_, rem_lo_, rem_hi_,
             ra_, success_, nsteps_,
             t_buf_, flo_buf_, fhi_buf_) = carry_in
            return (t_, xh_lo_, xh_hi_, rem_lo_, rem_hi_,
                    ra_, success_, nsteps_,
                    t_buf_, flo_buf_, fhi_buf_)

        inner_carry = (t, xh_lo, xh_hi, rem_lo, rem_hi,
                       red_abstol, success, nsteps,
                       t_buf, flo_buf, fhi_buf)

        (new_t, new_xh_lo, new_xh_hi, new_rem_lo, new_rem_hi,
         new_ra, new_success, new_nsteps,
         new_t_buf, new_flo_buf, new_fhi_buf) = lax.cond(
            done, null_step, real_step, inner_carry)

        new_done = done | (~new_success) | (sign * new_t >= sign * tmax_arr)

        new_carry = (new_t, new_xh_lo, new_xh_hi, new_rem_lo, new_rem_hi,
                     new_ra, new_success, new_done, new_nsteps,
                     new_t_buf, new_flo_buf, new_fhi_buf)
        return new_carry, None

    final_carry, _ = lax.scan(scan_body, init_carry, None, length=maxsteps)

    (_, _, _, _, _, _, final_success, _, final_nsteps,
     final_times, final_flo, final_fhi) = final_carry

    return ValidatedSolution(
        times=final_times,
        flowpipe_lo=final_flo,
        flowpipe_hi=final_fhi,
        nsteps=final_nsteps,
        success=final_success,
    )
