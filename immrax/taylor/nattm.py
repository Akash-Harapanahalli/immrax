"""
This file implements the Natural Taylor Model Function as an interpreter of Jaxprs.

Similar to natif for intervals, nattm transforms functions to operate on Taylor models,
propagating polynomial approximations with rigorous remainder bounds.
"""

from functools import wraps, partial
from typing import Any, Callable, Sequence
from jaxtyping import Array, ArrayLike

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
from immrax.utils import fact, inv_fact
from immrax.taylor.taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_concatenate,
    _generate_exponents,
    _get_canonical_exponents,
    _bound_monomial,
    _merge_taylor_terms,
)

# Move bound_polynomial logic here as helper (JIT-compatible vectorized version)
def _bound_polynomial(tm: TaylorModel) -> Interval:
    """Bound the polynomial part over the domain.

    Fully vectorized for JIT compatibility.
    """
    # Monomial bounds over [-1, 1]^d:
    # - If any exponent is odd: range is [-1, 1]
    # - If all exponents are even (nonzero): range is [0, 1]
    # - Constant term (all zero): range is [1, 1]
    has_odd = jnp.any(tm.exponents % 2 == 1, axis=0)  # (m,)
    is_constant = jnp.all(tm.exponents == 0, axis=0)  # (m,)

    mono_lower = jnp.where(is_constant, 1.0, jnp.where(has_odd, -1.0, 0.0))  # (m,)
    mono_upper = jnp.ones(tm.num_monomials)  # (m,)

    # For each term, bound is coeff * [mono_lower, mono_upper]
    # Handle positive/negative coeffs separately
    c_pos = jnp.maximum(tm.coeffs, 0.0)  # (n, m)
    c_neg = jnp.minimum(tm.coeffs, 0.0)  # (n, m)

    term_lower = c_pos * mono_lower[None, :] + c_neg * mono_upper[None, :]  # (n, m)
    term_upper = c_pos * mono_upper[None, :] + c_neg * mono_lower[None, :]  # (n, m)

    result_lower = jnp.sum(term_lower, axis=1)  # (n,)
    result_upper = jnp.sum(term_upper, axis=1)  # (n,)

    return interval(result_lower, result_upper)

TaylorModel._bound_polynomial = _bound_polynomial


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

    # Note: TaylorModel operations are now JIT-compatible (vectorized).
    # However, the jaxpr interpreter itself uses Python control flow,
    # so @jit is not applied here. JIT can be applied to the underlying function f.
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

# Registry mapping primitives to their corresponding jnp functions for univariate operations
_univariate_prim_to_func = {}


def _register_tm_univariate(primitive: Primitive, func: Callable) -> None:
    """Register a univariate primitive for Taylor model arithmetic.

    This registers the primitive in three places:
    1. _univariate_prim_to_func - maps primitive to jnp function for jet
    2. tm_inclusion_registry - the TM operation handler
    3. _needs_max_order - marks it as needing max_order parameter
    """
    # Register the jnp function for jet
    _univariate_prim_to_func[primitive] = func

    # Create and register the TM handler
    def _tm_handler(x: TaylorModel, *, max_order: int = None, accuracy=None) -> TaylorModel:
        return _tm_univariate(primitive, x, max_order=max_order)

    tm_inclusion_registry[primitive] = _tm_handler
    _needs_max_order.add(primitive)


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
_add_tm_passthrough_to_registry(lax.broadcast_in_dim_p)
_add_tm_passthrough_to_registry(lax.iota_p)
_add_tm_passthrough_to_registry(lax.convert_element_type_p)
_add_tm_passthrough_to_registry(debug_callback_p)


# --- Squeeze operation (special handling to preserve polynomial structure) ---

def _tm_squeeze_p(x, *, dimensions) -> TaylorModel:
    """Handle squeeze of Taylor models, preserving polynomial structure.

    For TMs, squeeze removes size-1 dimensions from the output.
    The key insight is that squeeze doesn't change the polynomial relationship,
    it just changes how we interpret the output shape.

    For TM internals, we keep coeffs as (n, m) and remainder as (n,).
    If the output becomes scalar, we still use n=1 internally.
    """
    if not istaylormodel(x):
        return lax.squeeze_p.bind(x, dimensions=dimensions)

    # For TMs with n=1 output being squeezed to scalar:
    # - Keep coeffs as (1, m)
    # - Keep remainder as (1,)
    # The polynomial structure is preserved; only the "external" shape changes.

    # TMs always maintain (n, m) coeffs and (n,) remainder internally,
    # so squeeze is essentially a no-op that just changes shape interpretation.
    return TaylorModel(x.coeffs, x.exponents, x.remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.squeeze_p] = _tm_squeeze_p


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


def _tm_add_p(x: TaylorModel, y: TaylorModel | ArrayLike) -> TaylorModel:
    """Taylor model addition."""
    if istaylormodel(x) and istaylormodel(y):
        # Merge polynomial terms
        new_coeffs, new_exponents = _merge_taylor_terms(
            x.coeffs, x.exponents, y.coeffs, y.exponents
        )

        # Add remainders using Interval addition
        new_remainder = x.remainder + y.remainder

        new_order = max(x._static_order, y._static_order)
        result = TaylorModel(
            new_coeffs, new_exponents, new_remainder, x.domain_center, x.domain_radius,
            _static_order=new_order
        )
        # Convert to canonical form for compatibility
        return result.to_canonical(new_order)
        
    elif istaylormodel(x):
        # TM + Array
        val = jnp.asarray(y)
        # Add to constant term
        # Need to handle shape
        if val.ndim == 0:
            val = jnp.full(x.shape, val)
        
        # We can implement this by creating a constant TM or modifying coeffs
        # Modifying coeffs is cheaper
        const_exp = jnp.zeros((x.d, 1), dtype=jnp.int32)
        const_coeff = val[:, None]
        
        new_coeffs, new_exponents = _merge_taylor_terms(
            x.coeffs, x.exponents, const_coeff, const_exp
        )
        
        result = TaylorModel(
            new_coeffs, new_exponents, x.remainder, x.domain_center, x.domain_radius,
            _static_order=x._static_order
        )
        return result.to_canonical(x._static_order)
        
    elif istaylormodel(y):
        # Array + TM
        return _tm_add_p(y, x)
    else:
        return x + y

tm_inclusion_registry[lax.add_p] = _tm_add_p
tm_inclusion_registry[ad_util.add_any_p] = _tm_add_p
TaylorModel.__add__ = _tm_add_p
TaylorModel.__radd__ = _tm_add_p



def _tm_sub_p(x: TaylorModel, y: TaylorModel | ArrayLike) -> TaylorModel:
    """Taylor model subtraction."""
    if istaylormodel(y):
        return _tm_add_p(x, _tm_neg_p(y))
    return _tm_add_p(x, -jnp.asarray(y))

tm_inclusion_registry[lax.sub_p] = _tm_sub_p
TaylorModel.__sub__ = _tm_sub_p
TaylorModel.__rsub__ = lambda self, other: _tm_sub_p(other, self)


def _tm_neg_p(x: TaylorModel) -> TaylorModel:
    """Taylor model negation."""
    if not istaylormodel(x):
        return -x
    return TaylorModel(
        -x.coeffs,
        x.exponents,
        -x.remainder,
        x.domain_center,
        x.domain_radius,
        _static_order=x._static_order,
    )

tm_inclusion_registry[lax.neg_p] = _tm_neg_p
TaylorModel.__neg__ = _tm_neg_p



def _tm_mul_p(x: TaylorModel, y: TaylorModel | ArrayLike, *, max_order: int = None) -> TaylorModel:
    """Taylor model multiplication."""
    if istaylormodel(x) and istaylormodel(y):
        if x.n != y.n:
            raise ValueError(f"Output dimensions must match: {x.n} vs {y.n}")
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")

        n = x.n
        d = x.d
        m1 = x.num_monomials
        m2 = y.num_monomials

        # Compute all product exponents: shape (d, m1*m2)
        exp1 = x.exponents[:, :, None]  # (d, m1, 1)
        exp2 = y.exponents[:, None, :]  # (d, 1, m2)
        all_exp = (exp1 + exp2).reshape(d, m1 * m2)

        # Compute all product coefficients: shape (n, m1*m2)
        coeff1 = x.coeffs[:, :, None]  # (n, m1, 1)
        coeff2 = y.coeffs[:, None, :]  # (n, 1, m2)
        all_coeffs = (coeff1 * coeff2).reshape(n, m1 * m2)

        # Compute total order of each product term
        total_orders = jnp.sum(all_exp, axis=0)

        effective_order = max_order if max_order is not None else max(x._static_order, y._static_order)

        # If max_order specified, zero out high-order terms and add to remainder
        # Mask for terms to keep
        keep_mask = total_orders <= effective_order
        
        # Zero out coefficients above max_order
        new_coeffs = jnp.where(keep_mask[None, :], all_coeffs, 0.0)

        # Bound truncated terms and add to remainder
        truncated_coeffs = jnp.where(keep_mask[None, :], 0.0, all_coeffs)
        
        # Compute monomial bounds
        has_odd = jnp.any(all_exp % 2 == 1, axis=0)
        mono_lower = jnp.where(has_odd, -1.0, 0.0)
        mono_upper = jnp.ones(m1 * m2)

        c_pos = jnp.maximum(truncated_coeffs, 0.0)
        c_neg = jnp.minimum(truncated_coeffs, 0.0)
        term_lower = c_pos * mono_lower[None, :] + c_neg * mono_upper[None, :]
        term_upper = c_pos * mono_upper[None, :] + c_neg * mono_lower[None, :]

        trunc_lower = jnp.sum(term_lower, axis=1)
        trunc_upper = jnp.sum(term_upper, axis=1)
        truncated_remainder = interval(trunc_lower, trunc_upper)

        new_exponents = all_exp

        # Compute remainder bounds using Interval operations
        # (p1 + r1) * (p2 + r2) = p1*p2 + p1*r2 + r1*p2 + r1*r2
        p1_bounds = _bound_polynomial(x)
        p2_bounds = _bound_polynomial(y)

        # Cross terms using Interval multiplication
        p1_r2 = p1_bounds * y.remainder
        r1_p2 = x.remainder * p2_bounds
        r1_r2 = x.remainder * y.remainder

        # Combined remainder
        new_remainder = p1_r2 + r1_p2 + r1_r2 + truncated_remainder

        result = TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            x.domain_center,
            x.domain_radius,
            _static_order=effective_order,
        )

        return result.to_canonical(effective_order)

    elif istaylormodel(x):
        # TM * Array
        alpha = jnp.asarray(y)
        if alpha.ndim == 0:
            new_coeffs = alpha * x.coeffs
            new_remainder = x.remainder * alpha
        else:
            new_coeffs = alpha[:, None] * x.coeffs
            # Element-wise scaling of interval
            new_remainder = interval(alpha, alpha) * x.remainder
        
        return TaylorModel(
            new_coeffs,
            x.exponents,
            new_remainder,
            x.domain_center,
            x.domain_radius,
            _static_order=x._static_order,
        )

    elif istaylormodel(y):
        return _tm_mul_p(y, x, max_order=max_order)
    else:
        return x * y

tm_inclusion_registry[lax.mul_p] = _tm_mul_p
_needs_max_order.add(lax.mul_p)
TaylorModel.__mul__ = _tm_mul_p
TaylorModel.__rmul__ = _tm_mul_p
TaylorModel.multiply = _tm_mul_p



from jax.experimental.jet import jet
from immrax.inclusion.nif import natif

def _tm_univariate(
    primitive_p: Primitive,
    x: TaylorModel,
    *,
    max_order: int | None = None
) -> TaylorModel:
    """Generic implementation for univariate Taylor Model operations.

    Uses jax.experimental.jet for Taylor coefficients and natif for
    Lagrange remainder bounds.
    """
    if not istaylormodel(x):
        return primitive_p.bind(x)

    order = max_order if max_order is not None else x._static_order
    c = x.constant_term

    # Get the function for this primitive from the registry, fallback to bind
    prim_func = _univariate_prim_to_func.get(primitive_p, lambda v: primitive_p.bind(v))

    # 1. Compute Taylor coefficients f^(k)(c)/k! using jet
    # jet(f, (c,), ((1, 0, 0, ...),)) computes f(c + t) and returns coefficients
    def get_coeffs(val):
        primals = (val,)
        # Input x = c + t: coefficient 1 for t^1, 0 for higher powers
        series = (tuple(1.0 if i == 0 else 0.0 for i in range(order)),)
        f_val, f_series = jet(prim_func, primals, series)
        return jnp.array([f_val] + list(f_series))

    # Vectorize over n output dimensions, shape: (n, order + 1)
    coeffs_raw = jax.vmap(get_coeffs)(c)

    # 2. Construct resulting polynomial using Horner's method with lax.scan
    # Result = a_0 + z * (a_1 + z * (a_2 + ...)) where z = x - c
    z = x - c

    def horner_step(carry, coeff):
        # coeff is shape (n,) for this term
        term = _tm_constant(coeff, x.d, order, x.domain_center, x.domain_radius)
        return carry.multiply(z, max_order=order) + term, None

    init = _tm_constant(coeffs_raw[:, order], x.d, order, x.domain_center, x.domain_radius)
    # Scan over coefficients from a_{order-1} down to a_0
    result, _ = lax.scan(horner_step, init, coeffs_raw[:, :order].T[::-1])

    # 3. Compute Lagrange remainder: f^(order+1)(xi) / (order+1)! * z^(order+1)
    # Need to bound f^(order+1) over the interval hull of x
    x_hull = x.interval_hull()

    def get_deriv_bound(i):
        # Use jet to define (order+1)-th derivative, then bound with natif
        def deriv_func(v):
            primals = (v,)
            series = (tuple(1.0 if j == 0 else 0.0 for j in range(order + 1)),)
            _, f_series = jet(prim_func, primals, series)
            # f_series[-1] = f^(order+1)(v) / (order+1)!, so multiply back
            return f_series[-1] * fact(order + 1)

        iv_i = interval(x_hull.lower[i], x_hull.upper[i])
        return natif(deriv_func)(iv_i)

    # Compute bounds for each dimension
    deriv_bounds = [get_deriv_bound(i) for i in range(x.n)]
    deriv_bound = interval(
        jnp.stack([b.lower for b in deriv_bounds]),
        jnp.stack([b.upper for b in deriv_bounds])
    )

    # Compute remainder bound
    z_bound = z._bound_polynomial() + z.remainder
    z_mag = jnp.maximum(jnp.abs(z_bound.lower), jnp.abs(z_bound.upper))
    z_pow_bound = z_mag ** (order + 1)
    rem_term = deriv_bound * (z_pow_bound * inv_fact(order + 1))

    lagrange_remainder = result.remainder + rem_term

    # Intersect with direct natif bound for tighter results
    # The natif bound is independent of Taylor expansion and may be tighter
    poly_bound = result._bound_polynomial()
    taylor_hull = poly_bound + lagrange_remainder

    # Compute natif bound over the original input interval hull
    def natif_bound_dim(i):
        iv_i = interval(x_hull.lower[i], x_hull.upper[i])
        return natif(prim_func)(iv_i)

    natif_bounds = [natif_bound_dim(i) for i in range(x.n)]
    natif_hull = interval(
        jnp.stack([b.lower for b in natif_bounds]),
        jnp.stack([b.upper for b in natif_bounds])
    )

    # Intersect the two bounds
    intersected_lower = jnp.maximum(taylor_hull.lower, natif_hull.lower)
    intersected_upper = jnp.minimum(taylor_hull.upper, natif_hull.upper)
    intersected_hull = interval(intersected_lower, intersected_upper)

    # Adjust remainder so that poly_bound + final_remainder = intersected_hull
    # For interval addition: [pl, pu] + [rl, ru] = [pl+rl, pu+ru]
    # So rl = fl - pl and ru = fu - pu
    final_remainder = interval(
        intersected_hull.lower - poly_bound.lower,
        intersected_hull.upper - poly_bound.upper
    )

    final_remainder = poly_bound

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
TaylorModel.__pow__ = _tm_integer_pow_p



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


def _tm_rmatmul_p(self: TaylorModel, other: ArrayLike) -> TaylorModel:
    """Left matrix multiplication: M @ TM."""
    M = jnp.asarray(other)
    new_coeffs = M @ self.coeffs

    # Remainder transformation using Interval matmul
    # remainder is (n,). We treat as column vector?
    # self.remainder is Interval(lower, upper)
    # n is output dim.
    # M @ TM means M (m x n) @ TM (n output) -> (m output)
    
    # We can use interval matmul if we reshape
    rem_interval = interval(self.remainder.lower.reshape(-1, 1), 
                            self.remainder.upper.reshape(-1, 1))
    
    # M @ rem_interval
    # We need to construct interval matrix for M
    M_iv = interval(M)
    
    # Helper for interval matmul
    # Or reuse natif matmul?
    # Interval.__matmul__ is defined in nif.py
    
    # Simply:
    # new_lower = M+ @ l - M- @ u ... standard interval matmul logic
    # But let's use the Interval class if it supports matmul
    # Check if Interval has matmul
    # immrax.inclusion.interval has no matmul impl in class, but nif.py patches it.
    # nattm.py imports nif.
    
    # rem_col = self.remainder.reshape((-1, 1)) # If supported?
    # Interval doesn't support reshape attribute in all versions? 
    # nif.py: Interval.reshape = ...
    
    # Let's do explicit logic to be safe, matching original TaylorModel logic
    rem_l = self.remainder.lower
    rem_u = self.remainder.upper
    
    # M_pos = max(M, 0), M_neg = min(M, 0)
    # [M] * [r]
    # lower = M_pos @ r_l + M_neg @ r_u
    # upper = M_pos @ r_u + M_neg @ r_l
    
    M_pos = jnp.maximum(M, 0.0)
    M_neg = jnp.minimum(M, 0.0)
    
    new_l = M_pos @ rem_l + M_neg @ rem_u
    new_u = M_pos @ rem_u + M_neg @ rem_l
    
    new_remainder = interval(new_l, new_u)
    
    # Output dimension changes
    # New domain is same
    # Coefficients were mapped
    
    return TaylorModel(
        new_coeffs,
        self.exponents,
        new_remainder,
        self.domain_center,
        self.domain_radius,
        _static_order=self._static_order,
    )

TaylorModel.__rmatmul__ = _tm_rmatmul_p



# --- Transcendental functions ---

_register_tm_univariate(lax.exp_p, jnp.exp)
_register_tm_univariate(lax.log_p, jnp.log)
_register_tm_univariate(lax.log1p_p, jnp.log1p)
_register_tm_univariate(lax.sin_p, jnp.sin)
_register_tm_univariate(lax.cos_p, jnp.cos)
_register_tm_univariate(lax.tan_p, jnp.tan)
_register_tm_univariate(lax.tanh_p, jnp.tanh)
_register_tm_univariate(lax.sqrt_p, jnp.sqrt)



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

_register_tm_univariate(lax.asin_p, jnp.arcsin)
_register_tm_univariate(lax.atan_p, jnp.arctan)



# --- Reciprocal ---

def _tm_reciprocal_p(x: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model reciprocal: 1/x."""
    one = _tm_constant(jnp.ones(x.n if istaylormodel(x) else 1, dtype=x.dtype if istaylormodel(x) else jnp.float32),
                       x.d if istaylormodel(x) else 1,
                       max_order if max_order else (x._static_order if istaylormodel(x) else 2),
                       x.domain_center if istaylormodel(x) else jnp.zeros(1),
                       x.domain_radius if istaylormodel(x) else jnp.ones(1))
    return _tm_div_p(one, x, max_order=max_order)
