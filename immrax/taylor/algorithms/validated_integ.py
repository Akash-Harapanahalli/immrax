"""Validated ODE integration using Picard-Lindelöf iteration.

Implements the first validated integration algorithm from TaylorModels.jl,
adapted for JAX and immrax.

Algorithm overview
------------------
At each time step:

1. Compute time Taylor coefficients ``x^(k)`` at the center of the current
   set using jet-based prolongation (the recurrence
   ``(k+1) x_{k+1} = [f(x)]_k``).

2. Select the step size from the Taylor coefficient norms:
   ``h = min_{k} (abstol / ||x_k / k!||)^{1/k}``.

3. Compute a rough enclosure of the flow over ``[t, t+h]`` via Picard
   iteration on the interval hull.

4. **Validate** the remainder using Picard-Lindelöf iteration: find an
   interval ``Δx`` such that the Picard operator ``P[Δx] ⊂ int(Δx)``
   (contraction mapping theorem ⟹ existence + uniqueness).

5. Build the flowpipe enclosure: evaluate the Taylor polynomial at ``h``
   and add the validated remainder.

6. If validation fails and ``adaptive=True``, shrink the step size and
   retry.

Reference
---------
TaylorModels.jl ``validated_integ`` (``src/valid_integ/validated_integ.jl``).
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


# ---------------------------------------------------------------------------
# Picard remainder operator
# ---------------------------------------------------------------------------

def _picard_remainder(f: Callable, t0, derivs: list,
                      x_hull: Interval, dt, order: int,
                      delta_x: Interval,
                      initial_rem: Interval) -> Interval:
    """Evaluate the Picard operator for the truncation remainder."""
    abs_h = jnp.abs(jnp.asarray(dt))
    h_iv = interval(jnp.zeros(()), abs_h)

    x_poly = _eval_taylor_interval(derivs, h_iv, order)
    extended = x_poly + delta_x
    f_extended = natif(lambda x: f(t0, x))(extended)

    xdot_derivs = derivs[1:]
    xdot_poly = _eval_taylor_interval(xdot_derivs, h_iv, order - 1)

    residual = f_extended - xdot_poly
    return initial_rem + h_iv * residual


# ---------------------------------------------------------------------------
# Picard-Lindelöf validation (JAX-traceable via lax.while_loop + lax.scan)
# ---------------------------------------------------------------------------

def _validate_step(f: Callable, t0, derivs: list,
                   x_hull: Interval, dt, sign,
                   order: int, initial_rem: Interval,
                   red_abstol, adaptive: bool,
                   minabstol):
    """Validate one integration step using Picard-Lindelöf iteration.

    Returns ``(success, remainder, dt, reduced_abstol)``.
    """
    n = x_hull.lower.shape[0]
    abs_h = jnp.abs(jnp.asarray(dt))

    # Initial candidate Δx from Lagrange remainder bound
    derivs_extra = prolongation(f, order + 1)(t0, x_hull.center)
    d_next = derivs_extra[order + 1]
    lagrange_init = jnp.abs(d_next) * abs_h ** (order + 1) * inv_fact(order + 1)

    # Outer adaptive retry loop via lax.while_loop
    # State: (success, delta_x_lo, delta_x_hi, dt, red_abstol, continue_flag)
    def outer_cond(state):
        success, _, _, _, _, continue_flag = state
        return (~success) & continue_flag

    def outer_body(state):
        success, dx_lo, dx_hi, dt_, ra, _ = state
        delta_x = interval(dx_lo, dx_hi)

        # Inner 50-iteration Picard widening via lax.scan
        # scan state: (success, delta_x_lo, delta_x_hi)
        def inner_body(carry, _):
            s, dxl, dxh = carry

            def do_step(args):
                dxl_, dxh_ = args
                dx = interval(dxl_, dxh_)
                delta = _picard_remainder(f, t0, derivs, x_hull, dt_, order,
                                          dx, initial_rem)
                contractive = _is_contractive(delta, dx)

                # Check subset (but not strict)
                is_subset = (jnp.all(delta.lower >= dx.lower) &
                             jnp.all(delta.upper <= dx.upper))

                # Widen slightly if subset but not contractive
                eps = 1e-14 * (jnp.abs(dxh_ - dxl_) + 1e-30)
                widened_lo = jnp.where(delta.lower == dxl_, dxl_ - eps, dxl_)
                widened_hi = jnp.where(delta.upper == dxh_, dxh_ + eps, dxh_)

                # Expand to contain delta with margin if not subset
                lo = jnp.minimum(dxl_, delta.lower)
                hi = jnp.maximum(dxh_, delta.upper)
                margin = 0.1 * (hi - lo + 1e-20)
                expanded_lo = lo - margin
                expanded_hi = hi + margin

                # Pick: if contractive → keep dx; if subset → widen; else → expand
                new_lo = jnp.where(contractive, dxl_,
                         jnp.where(is_subset, widened_lo, expanded_lo))
                new_hi = jnp.where(contractive, dxh_,
                         jnp.where(is_subset, widened_hi, expanded_hi))

                return contractive, new_lo, new_hi

            def no_op(args):
                dxl_, dxh_ = args
                return jnp.bool_(True), dxl_, dxh_

            # Skip if already successful
            new_s, new_dxl, new_dxh = lax.cond(
                s, no_op, do_step, (dxl, dxh))

            return (new_s, new_dxl, new_dxh), None

        (inner_success, final_dxl, final_dxh), _ = lax.scan(
            inner_body, (jnp.bool_(False), dx_lo, dx_hi), None, length=50)

        # If inner loop didn't succeed, try adaptive reduction
        new_ra = jnp.where(inner_success, ra, ra / 10.0)
        new_dt = jnp.where(inner_success, dt_, dt_ * 0.1 ** (1.0 / order))
        new_abs_h = jnp.abs(new_dt)
        new_lagrange = jnp.abs(d_next) * new_abs_h ** (order + 1) * inv_fact(order + 1)

        # If not successful: reset delta_x for the new dt
        new_dxl_retry = -2.0 * new_lagrange
        new_dxh_retry = 2.0 * new_lagrange

        can_continue = jnp.where(
            inner_success, jnp.bool_(False),
            jnp.bool_(adaptive) & (ra > minabstol))

        out_dxl = jnp.where(inner_success, final_dxl, new_dxl_retry)
        out_dxh = jnp.where(inner_success, final_dxh, new_dxh_retry)
        out_dt = jnp.where(inner_success, dt_, new_dt)
        out_ra = jnp.where(inner_success, ra, new_ra)

        return (inner_success, out_dxl, out_dxh, out_dt, out_ra, can_continue)

    init_dx_lo = -2.0 * lagrange_init
    init_dx_hi = 2.0 * lagrange_init

    final_state = lax.while_loop(
        outer_cond, outer_body,
        (jnp.bool_(False), init_dx_lo, init_dx_hi, dt, red_abstol, jnp.bool_(True)))

    success, dx_lo, dx_hi, dt_out, ra_out, _ = final_state
    return success, interval(dx_lo, dx_hi), dt_out, ra_out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validated_integ(
    f: Callable,
    x0: Interval,
    t0: float,
    tmax: float,
    order_time: int,
    abstol: float = 1e-10,
    maxsteps: int = 2000,
    adaptive: bool = True,
    minabstol: float = 1e-50,
) -> ValidatedSolution:
    """Validated ODE integration using Picard-Lindelöf iteration.

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
        Maximum number of integration steps (default 2000).
    adaptive : bool
        Reduce step size on validation failure (default ``True``).
    minabstol : float
        Minimum tolerance before giving up (default ``1e-50``).

    Returns
    -------
    ValidatedSolution
        Contains ``times``, ``flowpipe_lo``, ``flowpipe_hi``, ``nsteps``,
        and ``success``.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from immrax.inclusion import icentpert
    >>> from immrax.taylor.algorithms import validated_integ
    >>>
    >>> # Simple ODE: x' = -x, x(0) ∈ [0.9, 1.1]
    >>> f = lambda t, x: -x
    >>> x0 = icentpert(jnp.array([1.0]), jnp.array([0.1]))
    >>> sol = validated_integ(f, x0, 0.0, 1.0, order_time=5)
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

    # Scan carry: (t, x_hull_lo, x_hull_hi, rem_lo, rem_hi,
    #              red_abstol, success, done, nsteps,
    #              times_buf, fp_lo_buf, fp_hi_buf)
    # Initial remainder captures the width of x0 around its center.
    # The enclosure is always: taylor_poly(center) + remainder, so rem
    # must carry the deviation of the initial set from the center.
    center0 = x0.center
    init_rem_lo = x0.lower - center0
    init_rem_hi = x0.upper - center0

    init_carry = (
        t0_arr,
        x0.lower,
        x0.upper,
        init_rem_lo,  # rem_lo
        init_rem_hi,  # rem_hi
        abstol_arr,  # red_abstol
        jnp.bool_(True),  # success (overall)
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

            # 3. Validate
            step_ok, remainder, dt, ra_ = _validate_step(
                f, t_, derivs, x_hull, dt, sign, order_time, rem,
                ra_, adaptive, minabstol_arr)

            # 4. Build enclosure
            x_next_center = _eval_taylor(derivs, dt, order_time)
            enc_lo = x_next_center + remainder.lower
            enc_hi = x_next_center + remainder.upper

            new_t = t_ + dt
            new_nsteps = nsteps_ + 1

            # Store in buffers
            t_buf_ = t_buf_.at[new_nsteps].set(new_t)
            flo_buf_ = flo_buf_.at[new_nsteps].set(enc_lo)
            fhi_buf_ = fhi_buf_.at[new_nsteps].set(enc_hi)

            # Adaptive tolerance recovery
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

        # Update done flag
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
