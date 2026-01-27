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

    Uses exact optimization (critical points) for 1D domain case with vector output.
    Uses termwise bounds for multivariate domain case and general tensor outputs.

    For TMs with output shape (*output_shape,), returns Interval with same shape.
    """
    # For 1D domain and 1D output, use exact optimization
    if tm.d == 1 and len(tm._output_shape) == 1:
        return _bound_polynomial_1d_exact(tm)
    else:
        # Vectorized termwise bound for general case
        # coeffs has shape (*output_shape, m), exponents has shape (d, m)
        has_odd = jnp.any(tm.exponents % 2 == 1, axis=0)  # (m,)
        is_constant = jnp.all(tm.exponents == 0, axis=0)  # (m,)

        mono_lower = jnp.where(is_constant, 1.0, jnp.where(has_odd, -1.0, 0.0))  # (m,)
        mono_upper = jnp.ones(tm.num_monomials)  # (m,)

        c_pos = jnp.maximum(tm.coeffs, 0.0)  # (*output_shape, m)
        c_neg = jnp.minimum(tm.coeffs, 0.0)  # (*output_shape, m)

        # Broadcasting: mono_lower/upper (m,) with coeffs (*output_shape, m)
        term_lower = c_pos * mono_lower + c_neg * mono_upper  # (*output_shape, m)
        term_upper = c_pos * mono_upper + c_neg * mono_lower  # (*output_shape, m)

        # Sum along monomial axis (last axis)
        return interval(jnp.sum(term_lower, axis=-1), jnp.sum(term_upper, axis=-1))


def _bound_polynomial_1d_exact(tm: TaylorModel) -> Interval:
    """Bound 1D polynomial exactly by finding critical points.

    Only valid for TMs with 1D domain (d=1) and 1D output shape (n,).
    """
    # coeffs shape: (n, m) where n is output dim
    # exponents shape: (1, m)

    n = tm._output_shape[0]

    # 1. Reconstruct polynomial coefficients in standard basis [c_0, c_1, ..., c_k]
    # Use _static_order to ensure max_deg is static for JIT
    max_deg = tm._static_order

    if max_deg == 0:
        # Constant Taylor Model - constant term is first coefficient
        const_term = tm.constant_term  # (n,)
        return interval(const_term, const_term)

    # Standard basis coeffs: (n, max_deg + 1)
    # indexed by power: [const, x, x^2, ...]
    poly_coeffs = jnp.zeros((n, max_deg + 1), dtype=tm.dtype)

    # This loop is unrolled during JIT if dimensions are static, which they are for a given TM
    # But tm.num_monomials can be large.
    # Using scatter/index_add is better for JIT

    degrees = tm.exponents[0].astype(jnp.int32)  # (m,)
    poly_coeffs = poly_coeffs.at[:, degrees].add(tm.coeffs)

    # poly_coeffs are [c_0, c_1, ..., c_k] for p(x) = sum c_i x^i

    # 2. Compute bounds for each output dimension
    def get_dim_bounds(coeffs):
        # coeffs: (max_deg + 1,)
        # Derivative coefficients: [c_1, 2c_2, ..., k*c_k]
        # P'(x) = sum_{i=1}^k i*c_i x^{i-1}
        # New coeffs for P'(x): [1*c_1, 2*c_2, ..., k*c_k]

        k = jnp.arange(1, max_deg + 1, dtype=coeffs.dtype)
        deriv_coeffs = coeffs[1:] * k

        # Check if derivative is zero (constant polynomial, already handled but safe to check)
        # or if max_deg was 0 (handled above)

        # Find roots of derivative (roots takes high->low)
        # jnp.roots expects p[0] * x^n + ... + p[n]
        # So we reverse deriv_coeffs
        # deriv_coeffs has size max_deg.
        # If max_deg=1, deriv_coeffs has size 1 (the constant slope).

        if max_deg == 1:
             valid_roots = jnp.array([], dtype=coeffs.dtype)
             is_real = jnp.array([], dtype=bool)
             in_range = jnp.array([], dtype=bool)
        else:
             roots = jnp.roots(deriv_coeffs[::-1], strip_zeros=False)

             # Filter real roots in [-1, 1]
             is_real = jnp.abs(jnp.imag(roots)) < 1e-6
             in_range = jnp.abs(jnp.real(roots)) <= 1.0
             valid_roots = jnp.real(roots)

        # Evaluate polynomial at endpoints -1, 1 and valid roots
        # Points to check (max_deg is static now, so condition is static)
        check_points = jnp.concatenate([
            jnp.array([-1.0, 1.0]),
            jnp.where(is_real & in_range, valid_roots, -1.0) if max_deg > 1 else jnp.array([])
        ])

        # Evaluate P(x) using Horner or polyval (polyval expects high->low)
        vals = jnp.polyval(coeffs[::-1], check_points)

        return jnp.min(vals), jnp.max(vals)

    # Vectorize over output dimensions
    lowers, uppers = jax.vmap(get_dim_bounds)(poly_coeffs)

    return interval(lowers, uppers)

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
    def wrapped(*args, **kwargs) -> TaylorModel :
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
    """Create a Taylor model from an interval (constant polynomial with remainder).

    Supports intervals with arbitrary shape (*output_shape,).
    """
    output_shape = iv.lower.shape
    center = iv.center
    pert = iv.pert

    exponents = _get_canonical_exponents(d, order)
    num_monomials = exponents.shape[1]

    # Create coeffs with shape (*output_shape, num_monomials)
    if len(output_shape) == 0:
        # Scalar output
        coeffs = jnp.zeros((num_monomials,), dtype=center.dtype)
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
        coeffs = coeffs.at[const_idx].set(center)
        remainder = icentpert(jnp.zeros((), dtype=center.dtype), pert)
    else:
        coeffs = jnp.zeros((*output_shape, num_monomials), dtype=center.dtype)
        # Set constant term
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))  # Index of [0,0,...,0]
        coeffs = coeffs.at[..., const_idx].set(center)
        remainder = icentpert(jnp.zeros(output_shape, dtype=center.dtype), pert)

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius,
                       _static_order=order)


def _add_tm_passthrough_to_registry(primitive: Primitive) -> None:
    """Helper to add a passthrough primitive to the TM registry."""
    tm_inclusion_registry[primitive] = _make_tm_passthrough_p(primitive)


# Register passthrough operations
_add_tm_passthrough_to_registry(lax.copy_p)
_add_tm_passthrough_to_registry(lax.iota_p)
_add_tm_passthrough_to_registry(lax.convert_element_type_p)
_add_tm_passthrough_to_registry(debug_callback_p)


# --- Reshape operation (preserves polynomial structure) ---

def _tm_reshape_p(x, *, new_sizes, dimensions=None) -> TaylorModel:
    """Handle reshape of Taylor models, preserving polynomial structure.

    Reshapes the output shape while keeping the monomial axis last.
    """
    if not istaylormodel(x):
        return lax.reshape_p.bind(x, new_sizes=new_sizes, dimensions=dimensions)

    m = x.num_monomials

    # new_sizes is the target output shape
    # coeffs currently has shape (*old_output_shape, m)
    # We need to reshape to (*new_sizes, m)

    coeff_new_sizes = (*new_sizes, m)

    # Handle dimensions parameter for transposition before reshape
    if dimensions is not None:
        # dimensions specifies how to permute before reshape
        # We need to append the monomial dimension to the permutation
        coeff_dimensions = (*dimensions, len(x._output_shape))
        new_coeffs = lax.reshape(x.coeffs, new_sizes=coeff_new_sizes, dimensions=coeff_dimensions)
    else:
        new_coeffs = x.coeffs.reshape(*coeff_new_sizes)

    # Reshape remainder
    new_remainder = interval(
        x.remainder.lower.reshape(*new_sizes),
        x.remainder.upper.reshape(*new_sizes)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.reshape_p] = _tm_reshape_p


# --- Transpose operation (preserves polynomial structure) ---

def _tm_transpose_p(x, *, permutation) -> TaylorModel:
    """Handle transpose of Taylor models, preserving polynomial structure.

    Permutes the output shape dimensions while keeping the monomial axis last.
    """
    if not istaylormodel(x):
        return lax.transpose_p.bind(x, permutation=permutation)

    # permutation applies to output shape dimensions
    # coeffs has shape (*output_shape, m), so we need to adjust permutation
    # to keep monomial axis last

    ndim_out = len(x._output_shape)

    # Extend permutation to include monomial axis staying last
    coeff_permutation = (*permutation, ndim_out)

    new_coeffs = lax.transpose(x.coeffs, permutation=coeff_permutation)

    # Transpose remainder
    new_remainder = interval(
        lax.transpose(x.remainder.lower, permutation=permutation),
        lax.transpose(x.remainder.upper, permutation=permutation)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.transpose_p] = _tm_transpose_p


# --- Squeeze operation (special handling to preserve polynomial structure) ---

def _tm_squeeze_p(x, *, dimensions) -> TaylorModel:
    """Handle squeeze of Taylor models, preserving polynomial structure.

    For TMs, squeeze removes size-1 dimensions from the output shape.
    The monomial axis (last axis of coeffs) is never squeezed.
    """
    if not istaylormodel(x):
        return lax.squeeze_p.bind(x, dimensions=dimensions)

    # Squeeze only affects output shape, not monomial axis
    # coeffs has shape (*output_shape, m)
    # We need to squeeze the output shape dimensions

    # Apply squeeze to coeffs (but dimensions refers to output shape, not including monomial axis)
    new_coeffs = lax.squeeze(x.coeffs, dimensions=dimensions)

    # Apply squeeze to remainder
    new_remainder = interval(
        lax.squeeze(x.remainder.lower, dimensions=dimensions),
        lax.squeeze(x.remainder.upper, dimensions=dimensions)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.squeeze_p] = _tm_squeeze_p


# --- Broadcast_in_dim operation (special handling to preserve polynomial structure) ---

def _tm_broadcast_in_dim_p(x, *, shape, broadcast_dimensions, sharding=None) -> TaylorModel:
    """Handle broadcast_in_dim of Taylor models, preserving polynomial structure.

    This broadcasts the output shape while preserving the monomial axis.
    The shape parameter refers to the output shape, and we append the monomial count.
    """
    if not istaylormodel(x):
        return lax.broadcast_in_dim_p.bind(x, shape=shape,
                                            broadcast_dimensions=broadcast_dimensions,
                                            sharding=sharding)

    m = x.num_monomials

    # Broadcast coeffs: shape is output shape, we need (*shape, m)
    # broadcast_dimensions maps input dims to output dims
    # For coeffs (*input_output_shape, m), we broadcast the output shape part
    # and keep the monomial axis

    # Adjust broadcast_dimensions to account for monomial axis staying last
    # If input is (*in_shape, m) and output is (*out_shape, m),
    # broadcast_dimensions for output shape become same for coeffs
    coeff_shape = (*shape, m)
    # The monomial axis maps to itself (last axis)
    coeff_broadcast_dims = (*broadcast_dimensions, len(shape))

    new_coeffs = lax.broadcast_in_dim(x.coeffs, shape=coeff_shape,
                                       broadcast_dimensions=coeff_broadcast_dims)

    # Broadcast remainder
    new_remainder = interval(
        lax.broadcast_in_dim(x.remainder.lower, shape=shape,
                             broadcast_dimensions=broadcast_dimensions),
        lax.broadcast_in_dim(x.remainder.upper, shape=shape,
                             broadcast_dimensions=broadcast_dimensions)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.broadcast_in_dim_p] = _tm_broadcast_in_dim_p


# --- Slice operation (special handling) ---

def _tm_slice_p(x, *, start_indices, limit_indices, strides=None) -> TaylorModel:
    """Handle slicing of Taylor models.

    Slicing applies to the output shape dimensions, not the monomial axis.
    """
    if not istaylormodel(x):
        return lax.slice_p.bind(x, start_indices=start_indices,
                                 limit_indices=limit_indices, strides=strides)

    m = x.num_monomials

    # Extend indices to include the full monomial axis (last axis of coeffs)
    coeff_start = (*start_indices, 0)
    coeff_limit = (*limit_indices, m)
    coeff_strides = (*strides, 1) if strides else None

    new_coeffs = lax.slice_p.bind(x.coeffs,
                                   start_indices=coeff_start,
                                   limit_indices=coeff_limit,
                                   strides=coeff_strides)

    # Slice the remainder (same indices as original, no monomial axis)
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
    """Handle concatenation of Taylor models along any output dimension.

    The dimension parameter refers to output shape axes, not the monomial axis.
    """
    tms = [arg for arg in args if istaylormodel(arg)]

    if len(tms) == 0:
        return lax.concatenate_p.bind(*args, dimension=dimension)

    # Convert non-TM args to TMs
    ref_tm = tms[0]
    tm_args = []
    for arg in args:
        if istaylormodel(arg):
            tm_args.append(arg.to_canonical(ref_tm._static_order))
        else:
            # Wrap scalar/array as constant TM
            arr = jnp.asarray(arg)
            tm_args.append(_tm_from_interval(
                interval(arr), ref_tm.d, ref_tm._static_order,
                ref_tm.domain_center, ref_tm.domain_radius
            ))

    return taylor_model_concatenate(tm_args, axis=dimension)

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
    """Taylor model addition with broadcasting support."""
    if istaylormodel(x) and istaylormodel(y):
        # Check compatible domains
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")

        # Broadcast output shapes
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, y._output_shape)

        # Broadcast coefficients to compatible shapes before merging
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        y_coeffs = jnp.broadcast_to(y.coeffs, (*broadcast_shape, y.num_monomials))

        # Merge polynomial terms
        new_coeffs, new_exponents = _merge_taylor_terms(
            x_coeffs, x.exponents, y_coeffs, y.exponents
        )

        # Add remainders using Interval addition (with broadcasting)
        x_rem_lower = jnp.broadcast_to(x.remainder.lower, broadcast_shape)
        x_rem_upper = jnp.broadcast_to(x.remainder.upper, broadcast_shape)
        y_rem_lower = jnp.broadcast_to(y.remainder.lower, broadcast_shape)
        y_rem_upper = jnp.broadcast_to(y.remainder.upper, broadcast_shape)
        new_remainder = interval(x_rem_lower + y_rem_lower, x_rem_upper + y_rem_upper)

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

        # Broadcast val to match x's output shape if needed
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, val.shape)

        # Create constant term coefficient
        const_exp = jnp.zeros((x.d, 1), dtype=jnp.int32)

        # Shape the constant coefficient to match broadcast shape
        val_broadcast = jnp.broadcast_to(val, broadcast_shape)
        const_coeff = val_broadcast[..., None]  # (*broadcast_shape, 1)

        # Broadcast x's coefficients
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))

        new_coeffs, new_exponents = _merge_taylor_terms(
            x_coeffs, x.exponents, const_coeff, const_exp
        )

        # Broadcast remainder
        x_rem_lower = jnp.broadcast_to(x.remainder.lower, broadcast_shape)
        x_rem_upper = jnp.broadcast_to(x.remainder.upper, broadcast_shape)
        new_remainder = interval(x_rem_lower, x_rem_upper)

        result = TaylorModel(
            new_coeffs, new_exponents, new_remainder, x.domain_center, x.domain_radius,
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
    """Taylor model element-wise multiplication with broadcasting support."""
    if istaylormodel(x) and istaylormodel(y):
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")

        # Broadcast output shapes
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, y._output_shape)

        d = x.d
        m1 = x.num_monomials
        m2 = y.num_monomials

        # Compute all product exponents: shape (d, m1*m2)
        exp1 = x.exponents[:, :, None]  # (d, m1, 1)
        exp2 = y.exponents[:, None, :]  # (d, 1, m2)
        all_exp = (exp1 + exp2).reshape(d, m1 * m2)

        # Broadcast coefficients to compatible shapes
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, m1))
        y_coeffs = jnp.broadcast_to(y.coeffs, (*broadcast_shape, m2))

        # Compute all product coefficients: shape (*broadcast_shape, m1*m2)
        # Outer product on monomial axis
        coeff1 = x_coeffs[..., :, None]  # (*broadcast_shape, m1, 1)
        coeff2 = y_coeffs[..., None, :]  # (*broadcast_shape, 1, m2)
        all_coeffs = (coeff1 * coeff2).reshape(*broadcast_shape, m1 * m2)

        # Compute total order of each product term
        total_orders = jnp.sum(all_exp, axis=0)  # (m1*m2,)

        effective_order = max_order if max_order is not None else max(x._static_order, y._static_order)

        # Mask for terms to keep
        keep_mask = total_orders <= effective_order  # (m1*m2,)

        # Zero out coefficients above max_order
        new_coeffs = jnp.where(keep_mask, all_coeffs, 0.0)

        # Bound truncated terms and add to remainder
        truncated_coeffs = jnp.where(keep_mask, 0.0, all_coeffs)

        # Compute monomial bounds
        has_odd = jnp.any(all_exp % 2 == 1, axis=0)  # (m1*m2,)
        mono_lower = jnp.where(has_odd, -1.0, 0.0)  # (m1*m2,)
        mono_upper = jnp.ones(m1 * m2)  # (m1*m2,)

        c_pos = jnp.maximum(truncated_coeffs, 0.0)  # (*broadcast_shape, m1*m2)
        c_neg = jnp.minimum(truncated_coeffs, 0.0)  # (*broadcast_shape, m1*m2)
        term_lower = c_pos * mono_lower + c_neg * mono_upper  # (*broadcast_shape, m1*m2)
        term_upper = c_pos * mono_upper + c_neg * mono_lower  # (*broadcast_shape, m1*m2)

        trunc_lower = jnp.sum(term_lower, axis=-1)  # (*broadcast_shape,)
        trunc_upper = jnp.sum(term_upper, axis=-1)  # (*broadcast_shape,)
        truncated_remainder = interval(trunc_lower, trunc_upper)

        new_exponents = all_exp

        # Compute remainder bounds using Interval operations
        # (p1 + r1) * (p2 + r2) = p1*p2 + p1*r2 + r1*p2 + r1*r2
        p1_bounds = _bound_polynomial(x)
        p2_bounds = _bound_polynomial(y)

        # Broadcast polynomial bounds and remainders
        p1_lower = jnp.broadcast_to(p1_bounds.lower, broadcast_shape)
        p1_upper = jnp.broadcast_to(p1_bounds.upper, broadcast_shape)
        p2_lower = jnp.broadcast_to(p2_bounds.lower, broadcast_shape)
        p2_upper = jnp.broadcast_to(p2_bounds.upper, broadcast_shape)
        p1_bounds_bc = interval(p1_lower, p1_upper)
        p2_bounds_bc = interval(p2_lower, p2_upper)

        x_rem_lower = jnp.broadcast_to(x.remainder.lower, broadcast_shape)
        x_rem_upper = jnp.broadcast_to(x.remainder.upper, broadcast_shape)
        y_rem_lower = jnp.broadcast_to(y.remainder.lower, broadcast_shape)
        y_rem_upper = jnp.broadcast_to(y.remainder.upper, broadcast_shape)
        x_rem_bc = interval(x_rem_lower, x_rem_upper)
        y_rem_bc = interval(y_rem_lower, y_rem_upper)

        # Cross terms using Interval multiplication
        p1_r2 = p1_bounds_bc * y_rem_bc
        r1_p2 = x_rem_bc * p2_bounds_bc
        r1_r2 = x_rem_bc * y_rem_bc

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
        # TM * Array (element-wise with broadcasting)
        alpha = jnp.asarray(y)

        # Broadcast shapes
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, alpha.shape)

        # Broadcast alpha and coeffs
        alpha_bc = jnp.broadcast_to(alpha, broadcast_shape)
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))

        # Scale coefficients
        new_coeffs = alpha_bc[..., None] * x_coeffs

        # Scale remainder
        x_rem_lower = jnp.broadcast_to(x.remainder.lower, broadcast_shape)
        x_rem_upper = jnp.broadcast_to(x.remainder.upper, broadcast_shape)
        alpha_iv = interval(alpha_bc, alpha_bc)
        new_remainder = alpha_iv * interval(x_rem_lower, x_rem_upper)

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

    Supports arbitrary output shapes by flattening, processing, and reshaping.
    """
    if not istaylormodel(x):
        return primitive_p.bind(x)

    order = max_order if max_order is not None else x._static_order
    output_shape = x._output_shape
    c = x.constant_term  # (*output_shape,)

    # Get the function for this primitive from the registry, fallback to bind
    prim_func = _univariate_prim_to_func.get(primitive_p, lambda v: primitive_p.bind(v))

    # Flatten output shape for processing
    import math
    n_flat = math.prod(output_shape) if output_shape else 1
    c_flat = c.reshape(-1) if output_shape else c[None]  # (n_flat,)

    # 1. Compute Taylor coefficients f^(k)(c)/k! using jet
    def get_coeffs(val):
        primals = (val,)
        series = (tuple(1.0 if i == 0 else 0.0 for i in range(order)),)
        f_val, f_series = jet(prim_func, primals, series)
        return jnp.array([f_val] + list(f_series))

    # Vectorize over flattened output dimensions, shape: (n_flat, order + 1)
    coeffs_raw = jax.vmap(get_coeffs)(c_flat)

    # Reshape coeffs_raw back to (*output_shape, order + 1) for TM construction
    if output_shape:
        coeffs_raw_shaped = coeffs_raw.reshape(*output_shape, order + 1)
    else:
        coeffs_raw_shaped = coeffs_raw[0]  # (order + 1,) for scalar

    # 2. Construct resulting polynomial using Horner's method with lax.scan
    # Result = a_0 + z * (a_1 + z * (a_2 + ...)) where z = x - c
    z = x - c

    def horner_step(carry, coeff):
        # coeff has shape (*output_shape,) for this term
        term = _tm_constant(coeff, x.d, order, x.domain_center, x.domain_radius)
        return carry.multiply(z, max_order=order) + term, None

    # Get highest order coefficient
    if output_shape:
        init_coeff = coeffs_raw_shaped[..., order]  # (*output_shape,)
        # Transpose to (order, *output_shape) then reverse order dimension
        scan_coeffs = jnp.moveaxis(coeffs_raw_shaped[..., :order], -1, 0)[::-1]
    else:
        init_coeff = coeffs_raw_shaped[order]  # scalar
        scan_coeffs = coeffs_raw_shaped[:order][::-1]  # (order,)

    init = _tm_constant(init_coeff, x.d, order, x.domain_center, x.domain_radius)
    result, _ = lax.scan(horner_step, init, scan_coeffs)

    # 3. Compute Lagrange remainder: f^(order+1)(xi) / (order+1)! * z^(order+1)
    x_hull = x.interval_hull()
    x_hull_flat_lower = x_hull.lower.reshape(-1) if output_shape else x_hull.lower[None]
    x_hull_flat_upper = x_hull.upper.reshape(-1) if output_shape else x_hull.upper[None]

    def get_deriv_bound(i):
        def deriv_func(v):
            primals = (v,)
            series = (tuple(1.0 if j == 0 else 0.0 for j in range(order + 1)),)
            _, f_series = jet(prim_func, primals, series)
            return f_series[-1] * fact(order + 1)

        iv_i = interval(x_hull_flat_lower[i], x_hull_flat_upper[i])
        return natif(deriv_func)(iv_i)

    # Compute bounds for each flattened dimension
    deriv_bounds = [get_deriv_bound(i) for i in range(n_flat)]
    deriv_bound_flat = interval(
        jnp.stack([b.lower for b in deriv_bounds]),
        jnp.stack([b.upper for b in deriv_bounds])
    )

    # Reshape deriv_bound back to output_shape
    if output_shape:
        deriv_bound = interval(
            deriv_bound_flat.lower.reshape(*output_shape),
            deriv_bound_flat.upper.reshape(*output_shape)
        )
    else:
        deriv_bound = interval(deriv_bound_flat.lower[0], deriv_bound_flat.upper[0])

    # Compute remainder bound
    z_bound = z._bound_polynomial() + z.remainder
    z_mag = jnp.maximum(jnp.abs(z_bound.lower), jnp.abs(z_bound.upper))
    z_pow_bound = z_mag ** (order + 1)
    rem_term = deriv_bound * (z_pow_bound * inv_fact(order + 1))

    lagrange_remainder = result.remainder + rem_term

    # Intersect with direct natif bound for tighter results
    poly_bound = result._bound_polynomial()
    taylor_hull = poly_bound + lagrange_remainder

    # Compute natif bound over the original input interval hull
    natif_hull = natif(prim_func)(x_hull)

    # Intersect the two bounds
    intersected_hull = taylor_hull & natif_hull

    # Back-propagate intersection to remainder constraint
    r_compat = intersected_hull - poly_bound

    # Final remainder is intersection of lagrange_remainder and r_compat
    final_remainder = lagrange_remainder & r_compat

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
    """Create a constant TaylorModel with arbitrary output shape."""
    output_shape = value.shape

    exponents = _get_canonical_exponents(d, order)
    num_monomials = exponents.shape[1]

    if len(output_shape) == 0:
        # Scalar output
        coeffs = jnp.zeros((num_monomials,), dtype=value.dtype)
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
        coeffs = coeffs.at[const_idx].set(value)
        remainder = interval(jnp.zeros((), dtype=value.dtype))
    else:
        coeffs = jnp.zeros((*output_shape, num_monomials), dtype=value.dtype)
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
        coeffs = coeffs.at[..., const_idx].set(value)
        remainder = interval(jnp.zeros(output_shape, dtype=value.dtype))

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
    """Left matrix multiplication: M @ TM.

    For TM with output shape (n,), M @ TM computes M @ TM for M of shape (m, n).
    Result has output shape (m,).

    Note: Currently only supports TMs with 1D output shape.
    """
    if len(self._output_shape) != 1:
        raise NotImplementedError(
            f"Matrix multiplication only supports 1D output TMs, got shape {self._output_shape}"
        )

    M = jnp.asarray(other)
    n = self._output_shape[0]

    # M has shape (m, n), coeffs has shape (n, num_monomials)
    # Result coeffs has shape (m, num_monomials)
    new_coeffs = M @ self.coeffs

    # Remainder transformation using interval matrix multiplication
    rem_l = self.remainder.lower  # (n,)
    rem_u = self.remainder.upper  # (n,)

    # Standard interval matmul: [M] * [r]
    # lower = M_pos @ r_l + M_neg @ r_u
    # upper = M_pos @ r_u + M_neg @ r_l

    M_pos = jnp.maximum(M, 0.0)
    M_neg = jnp.minimum(M, 0.0)

    new_l = M_pos @ rem_l + M_neg @ rem_u
    new_u = M_pos @ rem_u + M_neg @ rem_l

    new_remainder = interval(new_l, new_u)

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
    """Taylor model general dot product with polynomial convolution.

    Implements proper polynomial multiplication for contracted dimensions:
    (Σ aᵢ xᵅ) · (Σ bⱼ xᵝ) = Σ aᵢbⱼ x^(α+β)

    For TMs with output shapes, dot_general contracts specified dimensions
    while performing polynomial multiplication.
    """
    dimension_numbers = kwargs["dimension_numbers"]
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

    # For matrix-vector: A @ x where A is array and x is TM (1D output)
    if not istaylormodel(A) and istaylormodel(B):
        A = jnp.asarray(A)
        # Use specialized __rmatmul__ for array @ TM with 1D output
        if len(B._output_shape) == 1:
            return A @ B
        # For general case, convert A to TM and use TM @ TM
        A_tm = _tm_from_interval(interval(A), B.d, B._static_order,
                                  B.domain_center, B.domain_radius)
        return _tm_dot_general_p(A_tm, B, max_order=max_order, **kwargs)

    # For TM @ array
    if istaylormodel(A) and not istaylormodel(B):
        B = jnp.asarray(B)
        B_tm = _tm_from_interval(interval(B), A.d, A._static_order,
                                  A.domain_center, A.domain_radius)
        return _tm_dot_general_p(A, B_tm, max_order=max_order, **kwargs)

    # TM @ TM case with polynomial convolution
    if istaylormodel(A) and istaylormodel(B):
        if A.d != B.d:
            raise ValueError(f"Domain dimensions must match: {A.d} vs {B.d}")

        effective_order = max_order if max_order is not None else max(A._static_order, B._static_order)
        d = A.d
        m1 = A.num_monomials
        m2 = B.num_monomials

        # Compute product exponents (same for all output positions)
        # exp1 (d, m1), exp2 (d, m2) -> product_exp (d, m1*m2)
        product_exp = (A.exponents[:, :, None] + B.exponents[:, None, :]).reshape(d, m1 * m2)

        # Compute product coefficients using double vmap over monomial axes
        # For each pair of monomials (i, j), compute standard dot_general on coefficients
        # A.coeffs has shape (*A_shape, m1), B.coeffs has shape (*B_shape, m2)
        # Result coeffs should have shape (*result_shape, m1*m2)

        # Use vmap to iterate over monomial pairs
        def contract_mono_pair(a_mono_coeffs, b_mono_coeffs):
            # a_mono_coeffs has shape (*A_shape,), b_mono_coeffs has shape (*B_shape,)
            # Perform standard dot_general contraction
            return lax.dot_general(a_mono_coeffs, b_mono_coeffs, dimension_numbers=dimension_numbers)

        # vmap over m2 (B's monomials), then over m1 (A's monomials)
        # Move monomial axis to front for vmapping
        A_coeffs_t = jnp.moveaxis(A.coeffs, -1, 0)  # (m1, *A_shape)
        B_coeffs_t = jnp.moveaxis(B.coeffs, -1, 0)  # (m2, *B_shape)

        # Double vmap: result has shape (m1, m2, *result_shape)
        contract_over_m2 = jax.vmap(contract_mono_pair, (None, 0), 0)  # over m2
        contract_over_m1m2 = jax.vmap(contract_over_m2, (0, None), 0)  # over m1

        result_coeffs_mm = contract_over_m1m2(A_coeffs_t, B_coeffs_t)  # (m1, m2, *result_shape)

        # Reshape to (*result_shape, m1*m2)
        result_shape = result_coeffs_mm.shape[2:]
        result_coeffs = jnp.moveaxis(result_coeffs_mm.reshape(m1 * m2, *result_shape), 0, -1)

        # Truncate high-order terms
        total_orders = jnp.sum(product_exp, axis=0)  # (m1*m2,)
        keep_mask = total_orders <= effective_order

        # Zero out high-order coefficients
        new_coeffs = jnp.where(keep_mask, result_coeffs, 0.0)

        # Bound truncated terms and add to remainder
        truncated_coeffs = jnp.where(keep_mask, 0.0, result_coeffs)

        has_odd = jnp.any(product_exp % 2 == 1, axis=0)
        mono_lower = jnp.where(has_odd, -1.0, 0.0)
        mono_upper = jnp.ones(m1 * m2)

        c_pos = jnp.maximum(truncated_coeffs, 0.0)
        c_neg = jnp.minimum(truncated_coeffs, 0.0)
        term_lower = c_pos * mono_lower + c_neg * mono_upper
        term_upper = c_pos * mono_upper + c_neg * mono_lower

        trunc_lower = jnp.sum(term_lower, axis=-1)
        trunc_upper = jnp.sum(term_upper, axis=-1)
        truncated_remainder = interval(trunc_lower, trunc_upper)

        # Compute remainder from polynomial bounds and input remainders
        # (p_A + r_A) · (p_B + r_B) involves cross terms
        p_A_bounds = _bound_polynomial(A)
        p_B_bounds = _bound_polynomial(B)

        # Use interval dot_general for remainder computation
        from immrax.inclusion.nif import inclusion_registry

        # Cross term: p_A · r_B
        p_A_r_B = inclusion_registry[lax.dot_general_p](
            p_A_bounds, B.remainder, dimension_numbers=dimension_numbers
        )

        # Cross term: r_A · p_B
        r_A_p_B = inclusion_registry[lax.dot_general_p](
            A.remainder, p_B_bounds, dimension_numbers=dimension_numbers
        )

        # Cross term: r_A · r_B
        r_A_r_B = inclusion_registry[lax.dot_general_p](
            A.remainder, B.remainder, dimension_numbers=dimension_numbers
        )

        # Combined remainder
        new_remainder = p_A_r_B + r_A_p_B + r_A_r_B + truncated_remainder

        result = TaylorModel(
            new_coeffs,
            product_exp,
            new_remainder,
            A.domain_center,
            A.domain_radius,
            _static_order=effective_order,
        )

        return result.to_canonical(effective_order)

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
    """Taylor model sum reduction over output shape dimensions.

    Sum preserves polynomial structure (sum of polynomials is a polynomial).
    """
    if not istaylormodel(x):
        return lax.reduce_sum_p.bind(x, axes=axes)

    # axes refer to output shape dimensions, not monomial axis
    # coeffs has shape (*output_shape, m)
    # Sum over specified axes while preserving monomial axis

    new_coeffs = jnp.sum(x.coeffs, axis=axes)
    new_remainder = interval(
        jnp.sum(x.remainder.lower, axis=axes),
        jnp.sum(x.remainder.upper, axis=axes)
    )

    return TaylorModel(new_coeffs, x.exponents, new_remainder,
                       x.domain_center, x.domain_radius, _static_order=x._static_order)

tm_inclusion_registry[lax.reduce_sum_p] = _tm_reduce_sum_p


def _tm_reduce_max_p(x: TaylorModel, *, axes) -> TaylorModel:
    """Taylor model max reduction.

    Max is not polynomial-preserving, so we fall back to interval bounds.
    """
    if not istaylormodel(x):
        return lax.reduce_max_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(
        jnp.max(hull.lower, axis=axes),
        jnp.max(hull.upper, axis=axes)
    )

    return _tm_from_interval(result, x.d, x._static_order,
                              x.domain_center, x.domain_radius)

tm_inclusion_registry[lax.reduce_max_p] = _tm_reduce_max_p


def _tm_reduce_min_p(x: TaylorModel, *, axes) -> TaylorModel:
    """Taylor model min reduction.

    Min is not polynomial-preserving, so we fall back to interval bounds.
    """
    if not istaylormodel(x):
        return lax.reduce_min_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(
        jnp.min(hull.lower, axis=axes),
        jnp.min(hull.upper, axis=axes)
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
