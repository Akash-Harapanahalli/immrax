"""
This file implements the Natural Taylor Model Function as an interpreter of Jaxprs.

Similar to natif for intervals, nattm transforms functions to operate on Taylor models,
propagating polynomial approximations with rigorous remainder bounds.
"""

from functools import wraps, partial
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax._src import ad_util, config, source_info_util
from jax._src.core import (
    Atom,
    Jaxpr,
    Literal,
    Var,
    clean_up_dead_vars,
    last_used,
    typecheck,
)
from jax._src.util import safe_map
from jax.extend.core import Primitive
from jax._src.debugging import debug_callback_p

from immrax.inclusion.interval import Interval, interval, icentpert
from immrax.taylor.taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_concatenate,
    _generate_exponents,
    _get_canonical_exponents,
)

# Registry mapping JAX primitives to Taylor model operations
tm_inclusion_registry = {}


def istaylormodel(x) -> bool:
    """Check if x is a TaylorModel."""
    return isinstance(x, TaylorModel)


def nattm(
    f: Callable[..., jax.Array],
    *,
    fixed_argnums: int | Sequence[int] = None,
    max_order: int | None = None,
) -> Callable[..., TaylorModel]:
    """Creates a Natural Taylor Model Function of f.

    All (non-fixed) positional arguments are assumed to be replaced with
    TaylorModel arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Natural Taylor Model Function from
    fixed_argnums : int|Sequence[int]
        Positional arguments to be treated as jax.Array instead of TaylorModel
    max_order : int, optional
        Maximum polynomial order to maintain. If None, uses the order of
        the input TaylorModels.

    Returns
    -------
    Callable[..., TaylorModel]
        Natural Taylor Model Function of f
    """

    # Note: Cannot use @jit here because TaylorModel operations have
    # Python control flow (int(), for loops) that aren't JIT-compatible.
    # JIT can be applied to the underlying function f if needed.
    @wraps(f)
    def wrapped(*args, **kwargs):
        """Natural Taylor Model function."""
        # Get representative values for tracing (use domain centers)
        geteval = lambda x: x.evaluate_polynomial(x.domain_center) if istaylormodel(x) else jnp.asarray(x)
        buildargs = jax.tree_util.tree_map(geteval, args, is_leaf=istaylormodel)
        buildkwargs = jax.tree_util.tree_map(geteval, kwargs, is_leaf=istaylormodel)

        # Build jaxpr via evaluation on representative values
        closed_jaxpr = eqx.filter_make_jaxpr(f)(*buildargs, **buildkwargs)[0]

        # Determine max_order from inputs if not specified
        nonlocal max_order
        effective_order = max_order
        if effective_order is None:
            for arg in jax.tree_util.tree_leaves(args, is_leaf=istaylormodel):
                if istaylormodel(arg):
                    effective_order = arg._static_order if effective_order is None else max(effective_order, arg._static_order)
            if effective_order is None:
                effective_order = 2  # Default order

        # Evaluate the jaxpr on the Taylor model arguments
        out = nattm_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args, max_order=effective_order)
        if len(out) == 1:
            return out[0]
        return out

    return wrapped


def nattm_jaxpr(jaxpr: Jaxpr, consts, *args, max_order: int = 2, propagate_source_info=True) -> list[Any]:
    """Interpreter for Jaxpr with TaylorModel arguments."""

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)

    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None

        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any([istaylormodel(read(iv)) for iv in eqn.invars]):
                try:
                    # Pass max_order to registry functions that need it
                    if eqn.primitive in _needs_max_order:
                        ans = tm_inclusion_registry[eqn.primitive](
                            *subfuns, *invars, max_order=max_order, **bind_params
                        )
                    else:
                        ans = tm_inclusion_registry[eqn.primitive](
                            *subfuns, *invars, **bind_params
                        )
                except KeyError:
                    # Fallback: look up by primitive name
                    # This handles cases where primitive objects differ but share identity logic or name
                    found_handler = None
                    for prim, handler in tm_inclusion_registry.items():
                        if prim.name == eqn.primitive.name:
                            found_handler = handler
                            break
                    
                    if found_handler:
                        # Cache it for next time to avoid loop
                        tm_inclusion_registry[eqn.primitive] = found_handler
                        if eqn.primitive in _needs_max_order:
                            pass # handler is already correct
                        # Re-try call
                        if eqn.primitive in _needs_max_order or (found_handler == tm_inclusion_registry.get(lax.mul_p)): # heuristic
                             # We need to know if handler needs max_order. 
                             # We can check if the found key was in _needs_max_order
                             if prim in _needs_max_order:
                                 ans = found_handler(*subfuns, *invars, max_order=max_order, **bind_params)
                             else:
                                 ans = found_handler(*subfuns, *invars, **bind_params)
                        else:
                             # Default assumption: checking signature is hard, but most our new ops need it.
                             # If we found it by name, it likely matches the registered one's signature.
                             # Let's check _needs_max_order set using the FOUND key 'prim'
                             if prim in _needs_max_order:
                                 ans = found_handler(*subfuns, *invars, max_order=max_order, **bind_params)
                             else:
                                 ans = found_handler(*subfuns, *invars, **bind_params)
                    else:
                        raise NotImplementedError(
                            f"{eqn.primitive} (name: {eqn.primitive.name}) not in tm_inclusion_registry"
                        )
            else:
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)

        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)

    return safe_map(read, jaxpr.outvars)


# Set of primitives that need max_order parameter
_needs_max_order = set()


def _make_tm_passthrough_p(primitive: Primitive) -> Callable[..., TaylorModel]:
    """Creates a TM function that applies to the coefficients individually.

    For structural operations like reshape, slice, etc., we apply them
    to each row of coefficients (treating each output dimension separately).
    """
    def _tm_p(*args, **kwargs) -> TaylorModel:
        gettm = lambda x: x if istaylormodel(x) else None
        getval = lambda x: x.coeffs if istaylormodel(x) else x

        # Find a reference TaylorModel
        ref_tm = None
        for arg in jax.tree_util.tree_leaves(args, is_leaf=istaylormodel):
            if istaylormodel(arg):
                ref_tm = arg
                break

        if ref_tm is None:
            # No TaylorModel, just apply primitive
            return primitive.bind(*args, **kwargs)

        # For passthrough ops on TaylorModel, we need to handle them carefully
        # These ops typically change the output dimension structure
        # For now, we apply them to the constant term only and wrap result
        args_const = []
        for arg in args:
            if istaylormodel(arg):
                args_const.append(arg.constant_term)
            else:
                args_const.append(arg)

        result_const = primitive.bind(*args_const, **kwargs)
        n_out = result_const.shape[0] if result_const.ndim > 0 else 1

        # Create output TM with appropriate structure
        # This is a conservative overapproximation - we lose polynomial info
        if ref_tm is not None:
            hull = ref_tm.interval_hull()
            args_hull = []
            for arg in args:
                if istaylormodel(arg):
                    args_hull.append(arg.interval_hull())
                else:
                    args_hull.append(interval(arg))

            # Use interval arithmetic for the remainder
            from immrax.inclusion.nif import inclusion_registry
            if primitive in inclusion_registry:
                result_interval = inclusion_registry[primitive](*args_hull, **kwargs)
            else:
                # Fallback: use constant with wide remainder
                result_interval = interval(result_const)

            # Create TM from interval result
            return _tm_from_interval(result_interval, ref_tm.d, ref_tm._static_order,
                                     ref_tm.domain_center, ref_tm.domain_radius)

        # Fallback
        return taylor_model(
            result_const.reshape(-1, 1),
            jnp.zeros((1, 1), dtype=jnp.int32),
            interval(jnp.zeros(n_out)),
            jnp.zeros(1),
            jnp.ones(1),
        )

    return _tm_p


def _tm_from_interval(iv: Interval, d: int, order: int,
                       domain_center: jax.Array, domain_radius: jax.Array) -> TaylorModel:
    """Create a Taylor model from an interval (constant polynomial with remainder)."""
    n = iv.lower.shape[0] if iv.lower.ndim > 0 else 1
    center = iv.center.reshape(-1)
    pert = iv.pert.reshape(-1)

    exponents = _get_canonical_exponents(d, order)
    num_monomials = exponents.shape[1]

    coeffs = jnp.zeros((n, num_monomials), dtype=center.dtype)
    # Set constant term
    const_idx = jnp.argmin(jnp.sum(exponents, axis=0))  # Index of [0,0,...,0]
    coeffs = coeffs.at[:, const_idx].set(center)

    remainder = icentpert(jnp.zeros(n, dtype=center.dtype), pert)

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius,
                       _static_order=order)


def _add_tm_passthrough_to_registry(primitive: Primitive) -> None:
    """Helper to add a passthrough primitive to the TM registry."""
    tm_inclusion_registry[primitive] = _make_tm_passthrough_p(primitive)


# Register passthrough operations
_add_tm_passthrough_to_registry(lax.copy_p)
_add_tm_passthrough_to_registry(lax.reshape_p)
_add_tm_passthrough_to_registry(lax.squeeze_p)
_add_tm_passthrough_to_registry(lax.broadcast_in_dim_p)
_add_tm_passthrough_to_registry(lax.iota_p)
_add_tm_passthrough_to_registry(lax.convert_element_type_p)
_add_tm_passthrough_to_registry(debug_callback_p)


# --- Slice operation (special handling) ---

def _tm_slice_p(x, *, start_indices, limit_indices, strides=None) -> TaylorModel:
    """Handle slicing of Taylor models."""
    if not istaylormodel(x):
        return lax.slice_p.bind(x, start_indices=start_indices,
                                 limit_indices=limit_indices, strides=strides)

    # Slice the coefficients along the output dimension (axis 0)
    new_coeffs = lax.slice_p.bind(x.coeffs,
                                   start_indices=(start_indices[0], 0),
                                   limit_indices=(limit_indices[0], x.coeffs.shape[1]),
                                   strides=(strides[0] if strides else 1, 1))

    # Slice the remainder
    new_remainder = interval(
        lax.slice_p.bind(x.remainder.lower, start_indices=start_indices,
                         limit_indices=limit_indices, strides=strides),
        lax.slice_p.bind(x.remainder.upper, start_indices=start_indices,
                         limit_indices=limit_indices, strides=strides)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.slice_p] = _tm_slice_p


def _tm_dynamic_slice_p(x, *start_indices, slice_sizes) -> TaylorModel:
    """Handle dynamic slicing of Taylor models."""
    if not istaylormodel(x):
        return lax.dynamic_slice_p.bind(x, *start_indices, slice_sizes=slice_sizes)

    # Dynamic slice the coefficients
    new_coeffs = lax.dynamic_slice_p.bind(
        x.coeffs, start_indices[0], 0,
        slice_sizes=(slice_sizes[0], x.coeffs.shape[1])
    )

    # Slice the remainder
    new_remainder = interval(
        lax.dynamic_slice_p.bind(x.remainder.lower, *start_indices, slice_sizes=slice_sizes),
        lax.dynamic_slice_p.bind(x.remainder.upper, *start_indices, slice_sizes=slice_sizes)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.dynamic_slice_p] = _tm_dynamic_slice_p


# --- Concatenation ---

def _tm_concatenate_p(*args, dimension) -> TaylorModel:
    """Handle concatenation of Taylor models."""
    tms = [arg for arg in args if istaylormodel(arg)]

    if len(tms) == 0:
        return lax.concatenate_p.bind(*args, dimension=dimension)

    if dimension != 0:
        raise NotImplementedError("TaylorModel concatenation only supported along dimension 0")

    # Convert non-TM args to TMs
    ref_tm = tms[0]
    tm_args = []
    for arg in args:
        if istaylormodel(arg):
            tm_args.append(arg.to_canonical(ref_tm._static_order))
        else:
            # Wrap scalar/array as constant TM
            arr = jnp.asarray(arg)
            if arr.ndim == 0:
                arr = arr[None]
            tm_args.append(_tm_from_interval(
                interval(arr), ref_tm.d, ref_tm._static_order,
                ref_tm.domain_center, ref_tm.domain_radius
            ))

    return taylor_model_concatenate(tm_args)

tm_inclusion_registry[lax.concatenate_p] = _tm_concatenate_p


# --- Higher-order primitives ---

def _tm_pjit_p(*args, **bind_params) -> TaylorModel:
    """Handle pjit by evaluating the inner jaxpr."""
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr

    # Get max_order from args
    max_order = 2
    for arg in args:
        if istaylormodel(arg):
            max_order = max(max_order, arg._static_order)

    return nattm_jaxpr(bind_jaxpr, [], *args, max_order=max_order)

tm_inclusion_registry[jax._src.pjit.pjit_p] = _tm_pjit_p


# --- Arithmetic operations ---

def _tm_add_p(x: TaylorModel, y: TaylorModel) -> TaylorModel:
    """Taylor model addition."""
    if istaylormodel(x) and istaylormodel(y):
        return x + y
    elif istaylormodel(x):
        return x + jnp.asarray(y)
    elif istaylormodel(y):
        return jnp.asarray(x) + y
    else:
        return x + y

tm_inclusion_registry[lax.add_p] = _tm_add_p
tm_inclusion_registry[ad_util.add_any_p] = _tm_add_p


def _tm_sub_p(x: TaylorModel, y: TaylorModel) -> TaylorModel:
    """Taylor model subtraction."""
    if istaylormodel(x) and istaylormodel(y):
        return x - y
    elif istaylormodel(x):
        return x - jnp.asarray(y)
    elif istaylormodel(y):
        return jnp.asarray(x) - y
    else:
        return x - y

tm_inclusion_registry[lax.sub_p] = _tm_sub_p


def _tm_neg_p(x: TaylorModel) -> TaylorModel:
    """Taylor model negation."""
    if not istaylormodel(x):
        return -x
    return -x

tm_inclusion_registry[lax.neg_p] = _tm_neg_p


def _tm_mul_p(x: TaylorModel, y: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model multiplication."""
    if istaylormodel(x) and istaylormodel(y):
        order = max_order if max_order is not None else max(x._static_order, y._static_order)
        return x.multiply(y, max_order=order)
    elif istaylormodel(x):
        return x * jnp.asarray(y)
    elif istaylormodel(y):
        return jnp.asarray(x) * y
    else:
        return x * y

tm_inclusion_registry[lax.mul_p] = _tm_mul_p
_needs_max_order.add(lax.mul_p)


from jax.experimental import jet
from immrax.inclusion.nif import natif

def _tm_univariate(
    primitive_p: Primitive, 
    x: TaylorModel, 
    *, 
    max_order: int | None = None
) -> TaylorModel:
    """Generic implementation for univariate Taylor Model operations.
    
    Uses jax.experimental.jet for coefficients and jax.grad + natif for 
    Lagrange remainder bounds.
    """
    if not istaylormodel(x):
        return primitive_p.bind(x)

    order = max_order if max_order is not None else x._static_order
    
    # x = c + p(u) + r
    # f(x) = f(c + p + r) = T_f(c; p+r) + R
    # T_f(c; p+r) = sum f^(k)(c)/k! * (p+r)^k
    # This acts on the full interval of x?
    # No, standard TM arithmetic transforms the reference point.
    # We expand f around the constant term c of x.
    # f(c + p_rem) where p_rem = x - c
    
    c = x.constant_term
    
    # 1. Compute coefficients f^(k)(c)/k! using repeated grad
    # jet was failing for some primitives (like asin), so we use grad loop which is robust.
    
    # We need to map over the batch dimension of x (n output dims)
    # c is (n,)
    
    # Map primitive to corresponding jnp function for evaluation
    # This avoids issues with primitive.bind requiring extra params like accuracy
    _prim_to_func = {
        lax.sin_p: jnp.sin,
        lax.cos_p: jnp.cos,
        lax.tan_p: jnp.tan,
        lax.exp_p: jnp.exp,
        lax.log_p: jnp.log,
        lax.log1p_p: jnp.log1p,
        lax.tanh_p: jnp.tanh,
        lax.sqrt_p: jnp.sqrt,
        lax.asin_p: jnp.arcsin,
        lax.atan_p: jnp.arctan,
    }

    # Get the function for this primitive
    if hasattr(primitive_p, 'bind'):
        # Check if it's a known primitive
        prim_func = _prim_to_func.get(primitive_p, None)
        if prim_func is None:
            # Fall back to using the primitive with try/except
            def prim_func(v):
                try:
                    return primitive_p.bind(v)
                except TypeError:
                    # Try with accuracy=None for newer JAX
                    return primitive_p.bind(v, accuracy=None)
    else:
        # It's a custom object with a bind method (like ReciprocalPrimitive)
        prim_func = lambda v: primitive_p.bind(v)

    def get_coeffs(val):
        # val is scalar
        def scalar_f(v):
             return jnp.sum(prim_func(v))

        # Compute derivatives up to order
        # derivs = [f(c), f'(c), f''(c), ...]
        derivs = [scalar_f(val)]
        g = scalar_f
        for _ in range(order):
            g = jax.grad(g)
            derivs.append(g(val))

        return jnp.array(derivs)

    # Vectorize over n output dimensions
    # coeffs_raw shape: (n, order + 1)
    coeffs_raw = jax.vmap(get_coeffs)(c)
    
    # Divide by k! to get Taylor coefficients
    import math
    factorials = jnp.array([math.factorial(k) for k in range(order + 1)], dtype=coeffs_raw.dtype)
    coeffs_raw = coeffs_raw / factorials[None, :]
    
    # 2. Construct resulting polynomial
    # Result = sum_{k=0}^order a_k * (x - c)^k
    # We can do this efficiently by evaluating the polynomial P(z) = sum a_k z^k
    # where z = x - c. 
    # z has no constant term.
    
    z = x - c
    
    # Accumulate result using Horner-like scheme or simple summation
    # Res = a_0 + z * (a_1 + z * (a_2 + ...))
    # But z is a TM.
    
    # Initialize with highest order term
    result = _tm_constant(coeffs_raw[:, order], x.d, order, x.domain_center, x.domain_radius)
    
    for k in range(order - 1, -1, -1):
        # Multiply by z and add next coeff
        term_k = _tm_constant(coeffs_raw[:, k], x.d, order, x.domain_center, x.domain_radius)
        if k == 0:
             # Last step: just add a_0. No multiplication by z.
             # Wait, loop logic:
             # prev result was (a_n z + a_{n-1}) ...
             # We want a_0 + z * (...)
             result = result.multiply(z, max_order=order) + term_k
        else:
             # Ops are result * z + a_k?
             # Let's trace:
             # Init: a_n
             # Loop k=n-1: a_n * z + a_{n-1}
             # ...
             # Loop k=0: (...) * z + a_0
             # Yes.
             result = result.multiply(z, max_order=order) + term_k
             
    # 3. Compute Remainder
    # Lagrange remainder: f^(order+1)(xi) / (order+1)! * z^(order+1)
    # The term z^(order+1) is (x-c)^(order+1)
    # We bounds for f^(order+1)(xi) over the interval hull of x.
    

    # ... inside _tm_univariate ...
    
    # Compute (order+1)-th derivative function
    # We need to handle JAX transformations carefully.
    
    # Hull of x (vector n)
    x_hull = x.interval_hull()
    
    def get_deriv_bound(i):
        # We need to differentiate the primitive with respect to its input.
        # Use prim_func which maps to jnp functions to avoid bind issues.

        def scalar_f(v):
            # v is scalar
            # We assume primitive maps scalar to scalar for univariate
            return jnp.sum(prim_func(v))

        # We need the (order+1)-th derivative
        # We use a loop of grads
        g = scalar_f
        for _ in range(order + 1):
            g = jax.grad(g)

        # Evaluate g on the interval hull of component i
        # interval hull component i is interval(lower[i], upper[i])
        iv_i = interval(x_hull.lower[i], x_hull.upper[i])

        return natif(g)(iv_i)

    # Compute bounds for each dimension
    deriv_bounds_lower = []
    deriv_bounds_upper = []
    
    for i in range(x.n):
        # Note: If x.n is large, this loop is slow during tracing.
        # But TMs are usually low-dim output.
        wb = get_deriv_bound(i)
        deriv_bounds_lower.append(wb.lower)
        deriv_bounds_upper.append(wb.upper)
        
    deriv_bound = interval(jnp.stack(deriv_bounds_lower), jnp.stack(deriv_bounds_upper))
    
    # Factorial (order+1)!
    # Use float for division
    import math
    fact = float(math.factorial(order + 1))
    
    # z bound
    z_poly_bound = z._bound_polynomial()
    # z has no constant term, so z_poly_bound + z.remainder is bound of z
    z_bound = z_poly_bound + z.remainder
    z_mag = jnp.maximum(jnp.abs(z_bound.lower), jnp.abs(z_bound.upper))
    z_pow_bound = z_mag ** (order + 1)
    
    # Remainder term: deriv_bound * z_pow_bound / fact
    # Note: Interval multiplication handles signs correctly
    rem_term = deriv_bound * (z_pow_bound / fact)
    
    # Add to result remainder
    final_remainder = result.remainder + rem_term
    
    return TaylorModel(result.coeffs, result.exponents, final_remainder,
                       result.domain_center, result.domain_radius, _static_order=order)


# Re-implement TM primitives using _tm_univariate

def _tm_div_p(x: TaylorModel, y: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model division: x / y = x * (1/y)."""
    
    # We need a primitive for reciprocal to pass to _tm_univariate
    # We can't pass a class method or lambda directly if it's not a Primitive or doesn't have bind?
    # _tm_univariate calls primitive_p.bind(v).
    # We create a dummy object with a bind method.
    
    class ReciprocalPrimitive:
        def bind(self, val):
            return 1.0 / val
            
    recip_y = _tm_univariate(ReciprocalPrimitive(), y, max_order=max_order)
    
    if istaylormodel(x):
        return x.multiply(recip_y, max_order=max_order)
    else:
        return jnp.asarray(x) * recip_y

tm_inclusion_registry[lax.div_p] = _tm_div_p
_needs_max_order.add(lax.div_p)




def _tm_constant(value: jax.Array, d: int, order: int,
                  domain_center: jax.Array, domain_radius: jax.Array) -> TaylorModel:
    """Create a constant TaylorModel."""
    value = jnp.atleast_1d(value)
    n = value.shape[0]

    exponents = _get_canonical_exponents(d, order)
    num_monomials = exponents.shape[1]

    coeffs = jnp.zeros((n, num_monomials), dtype=value.dtype)
    const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
    coeffs = coeffs.at[:, const_idx].set(value)

    remainder = interval(jnp.zeros(n, dtype=value.dtype))

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius,
                       _static_order=order)


def _tm_integer_pow_p(x: TaylorModel, y: int, *, max_order: int = None) -> TaylorModel:
    """Taylor model integer power."""
    if not istaylormodel(x):
        return lax.integer_pow(x, y)

    order = max_order if max_order is not None else x._static_order

    if y == 0:
        return _tm_constant(jnp.ones(x.n, dtype=x.dtype), x.d, order,
                            x.domain_center, x.domain_radius)
    elif y < 0:
        # x^(-n) = 1/x^n
        pos_pow = _tm_integer_pow_p(x, -y, max_order=order)
        return _tm_div_p(_tm_constant(jnp.ones(x.n, dtype=x.dtype), x.d, order,
                                       x.domain_center, x.domain_radius),
                          pos_pow, max_order=order)
    else:
        result = x
        for _ in range(y - 1):
            result = result.multiply(x, max_order=order)
        return result

tm_inclusion_registry[lax.integer_pow_p] = _tm_integer_pow_p
_needs_max_order.add(lax.integer_pow_p)


def _tm_square_p(x: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model square."""
    return _tm_integer_pow_p(x, 2, max_order=max_order)

if hasattr(lax, 'square_p'):
    tm_inclusion_registry[lax.square_p] = _tm_square_p
    _needs_max_order.add(lax.square_p)


def _tm_pow_p(x: TaylorModel, y: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model general power: x^y = exp(y * log(x))."""
    order = max_order
    if order is None:
        order = 2
        if istaylormodel(x):
            order = max(order, x._static_order)
        if istaylormodel(y):
            order = max(order, y._static_order)

    log_x = _tm_log_p(x, max_order=order)
    y_log_x = _tm_mul_p(y, log_x, max_order=order)
    return _tm_exp_p(y_log_x, max_order=order)

tm_inclusion_registry[lax.pow_p] = _tm_pow_p
_needs_max_order.add(lax.pow_p)


# --- Transcendental functions ---


# --- Transcendental functions ---

def _tm_exp_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    return _tm_univariate(lax.exp_p, x, max_order=max_order)

tm_inclusion_registry[lax.exp_p] = _tm_exp_p
_needs_max_order.add(lax.exp_p)


def _tm_log_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    return _tm_univariate(lax.log_p, x, max_order=max_order)

tm_inclusion_registry[lax.log_p] = _tm_log_p
_needs_max_order.add(lax.log_p)


def _tm_log1p_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    return _tm_univariate(lax.log1p_p, x, max_order=max_order)

tm_inclusion_registry[lax.log1p_p] = _tm_log1p_p
_needs_max_order.add(lax.log1p_p)


def _tm_sin_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    return _tm_univariate(lax.sin_p, x, max_order=max_order)

tm_inclusion_registry[lax.sin_p] = _tm_sin_p
_needs_max_order.add(lax.sin_p)


def _tm_cos_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    return _tm_univariate(lax.cos_p, x, max_order=max_order)

tm_inclusion_registry[lax.cos_p] = _tm_cos_p
_needs_max_order.add(lax.cos_p)



def _taylor_sin_series(p: TaylorModel, order: int) -> TaylorModel:
    """Compute Taylor series for sin(p) where p has no constant term."""
    result = _tm_constant(jnp.zeros(p.n, dtype=p.dtype), p.d, order,
                          p.domain_center, p.domain_radius)
    power = p  # p^1
    sign = 1.0
    factorial = 1.0

    for k in range(order + 1):
        exp = 2 * k + 1
        if exp > order:
            break
        # Compute factorial(2k+1)
        for i in range(1 if k == 0 else 2*k, exp + 1):
            factorial *= i

        result = result + power * (sign / factorial)

        # p^{2k+1} -> p^{2k+3}
        power = power.multiply(p, max_order=order)
        power = power.multiply(p, max_order=order)
        sign = -sign

    return result


def _taylor_cos_series(p: TaylorModel, order: int) -> TaylorModel:
    """Compute Taylor series for cos(p) where p has no constant term."""
    result = _tm_constant(jnp.ones(p.n, dtype=p.dtype), p.d, order,
                          p.domain_center, p.domain_radius)
    power = p.multiply(p, max_order=order)  # p^2
    sign = -1.0
    factorial = 2.0

    for k in range(1, order + 1):
        exp = 2 * k
        if exp > order:
            break

        result = result + power * (sign / factorial)

        # p^{2k} -> p^{2k+2}
        power = power.multiply(p, max_order=order)
        power = power.multiply(p, max_order=order)
        sign = -sign
        factorial *= (exp + 1) * (exp + 2)

    return result


def _tm_tan_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    """Taylor model tangent: tan(x) = sin(x) / cos(x)."""
    return _tm_div_p(_tm_sin_p(x, max_order=max_order),
                      _tm_cos_p(x, max_order=max_order),
                      max_order=max_order)

tm_inclusion_registry[lax.tan_p] = _tm_tan_p
_needs_max_order.add(lax.tan_p)


def _tm_tanh_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    """Taylor model hyperbolic tangent."""
    if not istaylormodel(x):
        return jnp.tanh(x)

    order = max_order if max_order is not None else x._static_order

    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    two_x = x * 2.0
    exp_2x = _tm_exp_p(two_x, max_order=order)

    one = _tm_constant(jnp.ones(x.n, dtype=x.dtype), x.d, order,
                       x.domain_center, x.domain_radius)

    num = exp_2x - one
    den = exp_2x + one

    return _tm_div_p(num, den, max_order=order)

tm_inclusion_registry[lax.tanh_p] = _tm_tanh_p
_needs_max_order.add(lax.tanh_p)


def _tm_sqrt_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
    """Taylor model square root: sqrt(x) = x^0.5."""
    if not istaylormodel(x):
        return jnp.sqrt(x)

    order = max_order if max_order is not None else x._static_order

    # sqrt(c + p) = sqrt(c) * sqrt(1 + p/c)
    # Use binomial series: (1+u)^{1/2} = sum_{k=0}^n C(1/2,k) u^k
    c = x.constant_term
    sqrt_c = jnp.sqrt(c)

    u = (x - c) * (1.0 / c)

    # Compute binomial series
    result = _tm_constant(jnp.ones(x.n, dtype=x.dtype), x.d, order,
                          x.domain_center, x.domain_radius)
    power = u  # u^1
    coeff = 0.5  # C(1/2, 1) = 1/2

    for k in range(1, order + 1):
        result = result + power * coeff
        power = power.multiply(u, max_order=order)
        # C(1/2, k+1) = C(1/2, k) * (1/2 - k) / (k + 1)
        coeff = coeff * (0.5 - k) / (k + 1)

    result = result * sqrt_c

    # Add remainder bound
    u_bound = u._bound_polynomial() + u.remainder
    u_max = jnp.maximum(jnp.abs(u_bound.lower), jnp.abs(u_bound.upper))
    remainder_bound = jnp.abs(sqrt_c) * jnp.abs(coeff) * (u_max ** (order + 1))

    result_remainder = result.remainder + icentpert(jnp.zeros(x.n), remainder_bound)
    return TaylorModel(result.coeffs, result.exponents, result_remainder,
                       result.domain_center, result.domain_radius, _static_order=order)

tm_inclusion_registry[lax.sqrt_p] = _tm_sqrt_p
_needs_max_order.add(lax.sqrt_p)


def _tm_abs_p(x: TaylorModel) -> TaylorModel:
    """Taylor model absolute value.

    Note: abs is not smooth at 0, so we use interval bounds as fallback.
    """
    if not istaylormodel(x):
        return jnp.abs(x)

    # Use interval arithmetic for abs
    hull = x.interval_hull()

    # If interval doesn't contain 0, we can use sign * x
    contains_zero = jnp.logical_and(hull.lower <= 0, hull.upper >= 0)

    # Conservative: return interval hull as TM
    abs_lower = jnp.where(contains_zero, 0.0, jnp.minimum(jnp.abs(hull.lower), jnp.abs(hull.upper)))
    abs_upper = jnp.maximum(jnp.abs(hull.lower), jnp.abs(hull.upper))

    center = (abs_lower + abs_upper) / 2
    pert = (abs_upper - abs_lower) / 2

    return _tm_from_interval(icentpert(center, pert), x.d, x._static_order,
                              x.domain_center, x.domain_radius)

tm_inclusion_registry[lax.abs_p] = _tm_abs_p


# --- Linear algebra ---

def _tm_dot_general_p(A: TaylorModel, B: TaylorModel, *, max_order: int = None, **kwargs) -> TaylorModel:
    """Taylor model general dot product."""
    # Get dimension info
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = kwargs["dimension_numbers"]

    # For matrix-vector: A @ x where A is array and x is TM
    if not istaylormodel(A) and istaylormodel(B):
        A = jnp.asarray(A)
        # Use __rmatmul__ from TaylorModel
        return A @ B

    # For TM @ TM or TM @ array, fall back to interval
    if istaylormodel(A):
        order = max_order if max_order is not None else A._static_order

        A_hull = A.interval_hull()
        if istaylormodel(B):
            B_hull = B.interval_hull()
        else:
            B_hull = interval(jnp.asarray(B))

        # Use interval dot product
        from immrax.inclusion.nif import inclusion_registry
        result_interval = inclusion_registry[lax.dot_general_p](A_hull, B_hull, **kwargs)

        return _tm_from_interval(result_interval, A.d, order,
                                  A.domain_center, A.domain_radius)

    return lax.dot_general_p.bind(A, B, **kwargs)

tm_inclusion_registry[lax.dot_general_p] = _tm_dot_general_p
_needs_max_order.add(lax.dot_general_p)


# --- Comparison operations (return intervals/arrays, not TMs) ---

def _tm_max_p(x: TaylorModel, y: TaylorModel) -> TaylorModel:
    """Taylor model maximum."""
    if istaylormodel(x):
        x_hull = x.interval_hull()
    else:
        x_hull = interval(jnp.asarray(x))

    if istaylormodel(y):
        y_hull = y.interval_hull()
    else:
        y_hull = interval(jnp.asarray(y))

    # Result interval
    result_lower = jnp.maximum(x_hull.lower, y_hull.lower)
    result_upper = jnp.maximum(x_hull.upper, y_hull.upper)
    result_interval = interval(result_lower, result_upper)

    # Get reference TM for domain info
    ref = x if istaylormodel(x) else y

    return _tm_from_interval(result_interval, ref.d, ref._static_order,
                              ref.domain_center, ref.domain_radius)

tm_inclusion_registry[lax.max_p] = _tm_max_p


def _tm_min_p(x: TaylorModel, y: TaylorModel) -> TaylorModel:
    """Taylor model minimum."""
    if istaylormodel(x):
        x_hull = x.interval_hull()
    else:
        x_hull = interval(jnp.asarray(x))

    if istaylormodel(y):
        y_hull = y.interval_hull()
    else:
        y_hull = interval(jnp.asarray(y))

    result_lower = jnp.minimum(x_hull.lower, y_hull.lower)
    result_upper = jnp.minimum(x_hull.upper, y_hull.upper)
    result_interval = interval(result_lower, result_upper)

    ref = x if istaylormodel(x) else y

    return _tm_from_interval(result_interval, ref.d, ref._static_order,
                              ref.domain_center, ref.domain_radius)

tm_inclusion_registry[lax.min_p] = _tm_min_p


# --- Reduction operations ---

def _tm_reduce_sum_p(x: TaylorModel, *, axes) -> TaylorModel:
    """Taylor model sum reduction."""
    if not istaylormodel(x):
        return lax.reduce_sum_p.bind(x, axes=axes)

    # Sum over output dimension
    if 0 in axes:
        # Summing all output dimensions
        new_coeffs = jnp.sum(x.coeffs, axis=0, keepdims=True)
        new_remainder = interval(
            jnp.sum(x.remainder.lower, keepdims=True),
            jnp.sum(x.remainder.upper, keepdims=True)
        )
        return TaylorModel(new_coeffs, x.exponents, new_remainder,
                           x.domain_center, x.domain_radius, _static_order=x._static_order)

    # Other axes - return as-is for now
    return x

tm_inclusion_registry[lax.reduce_sum_p] = _tm_reduce_sum_p


def _tm_reduce_max_p(x: TaylorModel, *, axes) -> TaylorModel:
    """Taylor model max reduction."""
    if not istaylormodel(x):
        return lax.reduce_max_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(
        jnp.max(hull.lower, axis=axes[0] if axes else None, keepdims=True),
        jnp.max(hull.upper, axis=axes[0] if axes else None, keepdims=True)
    )

    return _tm_from_interval(result, x.d, x._static_order,
                              x.domain_center, x.domain_radius)

tm_inclusion_registry[lax.reduce_max_p] = _tm_reduce_max_p


def _tm_reduce_min_p(x: TaylorModel, *, axes) -> TaylorModel:
    """Taylor model min reduction."""
    if not istaylormodel(x):
        return lax.reduce_min_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(
        jnp.min(hull.lower, axis=axes[0] if axes else None, keepdims=True),
        jnp.min(hull.upper, axis=axes[0] if axes else None, keepdims=True)
    )

    return _tm_from_interval(result, x.d, x._static_order,
                              x.domain_center, x.domain_radius)

tm_inclusion_registry[lax.reduce_min_p] = _tm_reduce_min_p


# --- Inverse trig functions ---


def _tm_asin_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
     return _tm_univariate(lax.asin_p, x, max_order=max_order)

tm_inclusion_registry[lax.asin_p] = _tm_asin_p
_needs_max_order.add(lax.asin_p)


def _tm_atan_p(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
     return _tm_univariate(lax.atan_p, x, max_order=max_order)

tm_inclusion_registry[lax.atan_p] = _tm_atan_p
_needs_max_order.add(lax.atan_p)



# --- Reciprocal ---

def _tm_reciprocal_p(x: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model reciprocal: 1/x."""
    one = _tm_constant(jnp.ones(x.n if istaylormodel(x) else 1, dtype=x.dtype if istaylormodel(x) else jnp.float32),
                       x.d if istaylormodel(x) else 1,
                       max_order if max_order else (x._static_order if istaylormodel(x) else 2),
                       x.domain_center if istaylormodel(x) else jnp.zeros(1),
                       x.domain_radius if istaylormodel(x) else jnp.ones(1))
    return _tm_div_p(one, x, max_order=max_order)
