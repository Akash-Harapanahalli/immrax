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

3. Build the flow-map function ``φ(dt, x_0)`` using prolongation +
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
    taylor_model_identity,
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
    strict = (delta.lower > delta_x.lower) & (delta.upper < delta_x.upper)
    both_zero = ((delta.lower == 0) & (delta.upper == 0) &
                 (delta_x.lower == 0) & (delta_x.upper == 0))
    return strict | both_zero


# ---------------------------------------------------------------------------
# Picard operator (operates on interval enclosures)
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

    def outer_cond(state):
        success, _, _, _, _, continue_flag = state
        return (~success) & continue_flag

    def outer_body(state):
        success, _, _, dt_, ra, _ = state

        E = interval(zero_n, zero_n)
        E_prime = _picard_remainder(f, t0, derivs, dt_, order,
                                     E, jnp.bool_(True), initial_rem)

        def inner_body(carry, _):
            s, e_lo, e_hi = carry

            def do_step(args):
                el, eh = args
                E_cur = interval(el, eh)
                E_p = _picard_remainder(f, t0, derivs, dt_, order,
                                         E_cur, jnp.bool_(False), initial_rem)
                contractive = _is_contractive(E_p, E_cur)

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
# Flow-map function builder
# ---------------------------------------------------------------------------

def _make_flow_map(f: Callable, order_time: int):
    """Build the flow map φ(t0, dt, x0) = eval_taylor(prolongation(f)(t0, x0), dt).

    Returns a function ``(t0, z) -> x`` where ``z = [dt, x0]``.
    ``t0`` is a separate argument so the function identity is stable
    across integration steps (avoiding repeated re-tracing).
    """
    prolong = prolongation(f, order_time)

    def flow_map(t0, z):
        dt = z[0]
        x0 = z[1:]
        derivs = prolong(t0, x0)
        return _eval_taylor(derivs, dt, order_time)

    return flow_map


def _build_tm_from_flow(f: Callable, order_time: int, n_state: int,
                         max_order: int):
    """Pre-build the jacfwd chain for the flow map and return a callable
    that efficiently computes the TaylorModel at any (t0, center_z, radius_z).

    The expensive jacfwd chain is built once; subsequent calls just evaluate
    the JIT-compiled derivative functions.
    """
    from functools import partial

    prolong = prolongation(f, order_time)

    def flow_map(t0, z):
        dt = z[0]
        x0 = z[1:]
        derivs = prolong(t0, x0)
        return _eval_taylor(derivs, dt, order_time)

    # Ensure vector output for consistent derivative tensor shapes
    def flow_vec(t0, z):
        v = flow_map(t0, z)
        return jnp.atleast_1d(v)

    # Build jacfwd chain once
    deriv_fns = [flow_vec]
    curr_f = flow_vec
    for k in range(1, max_order + 2):
        curr_f = jax.jacfwd(curr_f, argnums=1)
        deriv_fns.append(curr_f)

    # JIT the derivative functions (t0 and z are dynamic)
    deriv_fns_jit = [jax.jit(fn) for fn in deriv_fns]

    def compute_tm(t0, center_z, radius_z):
        d = center_z.shape[0]
        n = n_state
        dtype = center_z.dtype

        exponents = _get_canonical_exponents(d, max_order)
        num_monomials = exponents.shape[1]

        # Evaluate derivative tensors at center
        deriv_tensors = []
        for k, fn in enumerate(deriv_fns_jit):
            tensor = fn(t0, center_z)
            deriv_tensors.append(tensor)

        # Assemble coefficients
        coeffs = jnp.zeros((n, num_monomials), dtype=dtype)

        for i in range(num_monomials):
            exp = exponents[:, i]
            order = int(jnp.sum(exp))

            if order > max_order:
                continue

            if order == 0:
                c = deriv_tensors[0]
            else:
                idx_list = []
                for var_idx in range(d):
                    count = int(exp[var_idx])
                    idx_list.extend([var_idx] * count)
                tensor = deriv_tensors[order]
                full_idx = (slice(None), *idx_list)
                c = tensor[full_idx]

            fact_prod = jnp.prod(jax.scipy.special.gamma(exp + 1))
            scale = jnp.prod(radius_z ** exp) / fact_prod
            coeffs = coeffs.at[:, i].set(c * scale)

        # Lagrange remainder bound
        next_order = max_order + 1
        D_next = deriv_tensors[next_order]
        current_bound = jnp.abs(D_next)
        for _ in range(next_order):
            current_bound = jnp.dot(current_bound, radius_z)
        fact = jax.scipy.special.gamma(next_order + 1)
        remainder_bound = current_bound / fact
        remainder = interval(
            jnp.zeros(n, dtype=dtype) - remainder_bound,
            jnp.zeros(n, dtype=dtype) + remainder_bound,
        )

        return TaylorModel(coeffs, exponents, remainder,
                           center_z, radius_z, _static_order=max_order)

    return compute_tm


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
    ValidatedSolution
        Contains ``times``, TaylorModel buffers, ``nsteps``, and ``success``.
        Use ``sol.flowpipe`` to get the list of TaylorModels, or
        ``sol.flowpipe_intervals`` for interval hulls.
    """
    # Convert Interval to TaylorModel if needed
    if isinstance(x0, Interval):
        x_tm = taylor_model_identity(x0, order=order_space)
    elif isinstance(x0, TaylorModel):
        x_tm = x0
    else:
        raise TypeError(f"x0 must be Interval or TaylorModel, got {type(x0)}")

    n = x_tm.shape[0] if len(x_tm.shape) > 0 else 1
    d_space = x_tm.d  # spatial domain dimension
    d_full = 1 + d_space  # time + space

    sign = jnp.where(tmax > t0, 1.0, -1.0)
    dtype = x_tm.coeffs.dtype
    t0_arr = jnp.asarray(t0, dtype=dtype)
    tmax_arr = jnp.asarray(tmax, dtype=dtype)
    abstol_arr = jnp.asarray(abstol, dtype=dtype)
    minabstol_arr = jnp.asarray(minabstol, dtype=dtype)

    max_order = max(order_time, order_space)

    # Store initial condition as a time-space TM (zero-width time domain)
    init_tm_full = _spatial_tm_to_full(x_tm, t0_arr, jnp.zeros((), dtype=dtype),
                                        max_order)

    # Integration state
    t_cur = float(t0)
    x_tm_cur = x_tm.to_canonical(max_order)
    red_abstol = float(abstol)
    success = True
    nsteps = 0

    times_list = [t0]
    tms_list = [init_tm_full]

    sign_f = float(sign)
    prolong = prolongation(f, order_time)

    # Pre-build the jacfwd chain once (expensive tracing happens here)
    compute_step_tm = _build_tm_from_flow(f, order_time, n, max_order)

    while t_cur * sign_f < float(tmax) * sign_f:
        # 1. Centre and interval hull of current spatial TM
        center = x_tm_cur.constant_term
        x_hull = x_tm_cur.interval_hull()
        init_rem = interval(x_hull.lower - center, x_hull.upper - center)

        # 2. Step size from derivative norms at centre
        t_cur_arr = jnp.asarray(t_cur, dtype=dtype)
        derivs = prolong(t_cur_arr, center)
        ra_arr = jnp.asarray(red_abstol, dtype=dtype)
        dt = _step_size(derivs, ra_arr, order_time)
        dt = jnp.minimum(dt, sign * (tmax_arr - t_cur_arr))
        dt = sign * dt

        # 3. Validate via epsilon-inflation (on interval hull)
        step_ok, remainder, dt, ra_arr = _validate_step(
            f, t_cur_arr, derivs, x_hull, dt, sign, order_time, init_rem,
            ra_arr, adaptive, minabstol_arr,
            epsilon=epsilon, delta=delta, validatesteps=validatesteps)

        # 4. Build the time-space TM via pre-built derivative chain
        # Time domain is [0, dt]: center at dt/2, radius dt/2
        abs_dt = jnp.abs(dt)
        t_half = abs_dt / 2.0
        center_z = jnp.concatenate([t_half[None], center])
        radius_z = jnp.concatenate([t_half[None], x_tm_cur.domain_radius])
        step_tm = compute_step_tm(t_cur_arr, center_z, radius_z)

        # 5. Replace TM remainder with Picard-validated remainder
        step_tm = TaylorModel(
            step_tm.coeffs, step_tm.exponents,
            remainder,
            step_tm.domain_center, step_tm.domain_radius,
            _static_order=max_order)

        # 6. Evaluate at dt to get next spatial TM
        next_tm = evaluate_at_variable(step_tm, 0, float(dt))
        x_tm_cur = next_tm.to_canonical(max_order)

        t_cur = t_cur + float(dt)
        nsteps += 1
        times_list.append(t_cur)
        tms_list.append(step_tm)

        # Update adaptive tolerance
        if adaptive:
            red_abstol = min(abstol, 10.0 * float(ra_arr))
        else:
            red_abstol = float(ra_arr)

        success = success and bool(step_ok)
        if not success:
            import warnings
            warnings.warn(f"Validation failed at t={t_cur}")
            break

        if nsteps >= maxsteps:
            import warnings
            warnings.warn("Maximum number of integration steps reached")
            break

    return TMFlowpipe(
        times=times_list,
        tms=tms_list,
        nsteps=nsteps,
        success=success,
    )


def _spatial_tm_to_full(
    x_tm: TaylorModel, t_center, t_radius, max_order: int
) -> TaylorModel:
    """Embed a spatial TM into a time-space TM with zero-width time domain.

    Adds a time variable (index 0) to the domain.  All exponents for the
    time variable are zero (the polynomial doesn't depend on time).
    """
    d_space = x_tm.d
    d_full = 1 + d_space

    # Target canonical exponents for full domain
    full_exponents = _get_canonical_exponents(d_full, max_order)
    num_full = full_exponents.shape[1]

    # The spatial exponents correspond to rows 1: of full_exponents
    # with row 0 (time) == 0.  Map spatial monomials into full monomials.
    x_tm_canon = x_tm.to_canonical(max_order)
    space_exponents = _get_canonical_exponents(d_space, max_order)

    # Hash-based matching
    base = max_order + 2
    powers_space = base ** jnp.arange(d_space)
    powers_full = base ** jnp.arange(d_full)

    space_hash = jnp.sum(space_exponents * powers_space[:, None], axis=0)

    # For full exponents, only consider those with time exponent == 0
    time_exp = full_exponents[0, :]  # (num_full,)
    spatial_part = full_exponents[1:, :]  # (d_space, num_full)
    spatial_hash_full = jnp.sum(spatial_part * powers_space[:, None], axis=0)

    # Match: for each spatial monomial, find its position in full exponents
    # (where time exp == 0)
    match = ((space_hash[:, None] == spatial_hash_full[None, :]) &
             (time_exp[None, :] == 0))  # (m_space, m_full)

    n_out = x_tm_canon.coeffs.shape[0] if x_tm_canon.coeffs.ndim > 1 else 1
    coeffs_space = x_tm_canon.coeffs
    if coeffs_space.ndim == 1:
        coeffs_space = coeffs_space[None, :]

    # Scatter spatial coefficients into full coefficient array
    new_coeffs = jnp.einsum('...j,ji->...i', coeffs_space,
                            match.astype(dtype=coeffs_space.dtype))

    domain_center = jnp.concatenate([t_center[None], x_tm_canon.domain_center])
    domain_radius = jnp.concatenate([t_radius[None], x_tm_canon.domain_radius])

    return TaylorModel(
        new_coeffs, full_exponents, x_tm_canon.remainder,
        domain_center, domain_radius, _static_order=max_order)
