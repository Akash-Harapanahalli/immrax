"""
This file implements the Natural Taylor Model Function as an interpreter of Jaxprs.

Similar to natif for intervals, nattm transforms functions to operate on Taylor models,
propagating polynomial approximations with rigorous remainder bounds.
"""

from functools import wraps, partial
from typing import Any, Callable
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
    PyTreeShape,
    taylor_model,
    _taylor_model_constant_impl,
    taylor_model_concatenate,
    _generate_exponents,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _leaf_slice,
    _merge_taylor_terms,
    _max_order,
    _bound_monomials_over_domain,
)


def _bound_polynomial(tm: TaylorModel) -> Interval:
    """Bound the polynomial part over tm.domain.

    Computes the natural interval extension of the polynomial evaluation:
    bounds each monomial (x - center)^alpha over D using interval arithmetic,
    then combines with coefficients.

    For TMs with output shape (*output_shape,), returns Interval with same shape.
    """
    # Bound each monomial over the shifted domain (D - center): shape (m,)
    mono_bounds = _bound_monomials_over_domain(
        tm.exponents, tm.shifted_domain, tm.max_order
    )

    # Combine with coefficients: coeffs (*output_shape, m), mono bounds (m,)
    zeros = jnp.zeros_like(tm.coeffs)
    c_pos = jnp.maximum(tm.coeffs, zeros)
    c_neg = jnp.minimum(tm.coeffs, zeros)

    term_lower = c_pos * mono_bounds.lower + c_neg * mono_bounds.upper
    term_upper = c_pos * mono_bounds.upper + c_neg * mono_bounds.lower

    return interval(jnp.sum(term_lower, axis=-1), jnp.sum(term_upper, axis=-1))


TaylorModel._bound_polynomial = _bound_polynomial


# Registry mapping JAX primitives to Taylor model operations
tm_inclusion_registry = {}


def istaylormodel(x) -> bool:
    """Check if x is a TaylorModel."""
    return isinstance(x, TaylorModel)


def nattm(
    f: Callable[..., jax.Array],
    *,
    max_order: "int | tuple[int, ...] | None" = None,
) -> Callable[..., TaylorModel]:
    """Creates a Natural Taylor Model Function of f.

    Non-TaylorModel positional arguments are automatically closed over
    (treated as constants during tracing).

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Natural Taylor Model Function from
    max_order : int, tuple[int, ...], or None
        Maximum polynomial order to maintain.  If an ``int``, it is
        broadcast to every domain leaf.  If a ``tuple``, it must match
        the per-leaf structure of the input TaylorModels' domain.
        If ``None``, inferred from the input TaylorModels.

    Returns
    -------
    Callable[..., TaylorModel]
        Natural Taylor Model Function of f
    """

    @wraps(f)
    def wrapped(*args, **kwargs) -> TaylorModel:
        """Natural Taylor Model function."""

        # Separate TM args from non-TM args
        tm_args = [a for a in jax.tree_util.tree_leaves(args, is_leaf=istaylormodel) if istaylormodel(a)]

        if not tm_args:
            # No TM args at all — just evaluate f directly
            return f(*args, **kwargs)

        # Concatenate all TM args into one
        if len(tm_args) == 1:
            tm_concat = tm_args[0]
        else:
            tm_concat = taylor_model_concatenate(tm_args)
        output_pytree = tm_concat._output_pytree

        # Determine effective_order as tuple[int, ...], matching domain leaves
        if max_order is not None:
            if isinstance(max_order, int):
                effective_order = tuple(
                    max_order for _ in tm_concat._per_leaf_order
                )
            else:
                effective_order = tuple(max_order)
        else:
            effective_order = tm_concat._per_leaf_order

        # Build f_tm that closes over non-TM args and kwargs,
        # receives the TM output leaves as positional args
        def f_tm(*tm_leaves):
            full_args = []
            leaf_idx = 0
            for i in range(len(args)):
                if istaylormodel(args[i]):
                    n = args[i]._output_pytree.num_leaves
                    if n == 1:
                        full_args.append(tm_leaves[leaf_idx])
                        leaf_idx += 1
                    else:
                        leaves = tm_leaves[leaf_idx:leaf_idx + n]
                        reconstructed = args[i]._output_pytree.treedef.unflatten(leaves)
                        # If this was the only positional arg and f expects
                        # multiple args, spread the reconstructed pytree
                        if len(args) == 1 and isinstance(reconstructed, (list, tuple)):
                            full_args.extend(reconstructed)
                        else:
                            full_args.append(reconstructed)
                        leaf_idx += n
                else:
                    full_args.append(args[i])
            return f(*full_args, **kwargs)

        # Trace with zero-valued representatives
        rep_args = tuple(jnp.zeros(shape) for shape in output_pytree.leaf_shapes)
        closed_jaxpr = eqx.filter_make_jaxpr(f_tm)(*rep_args)[0]

        out = nattm_jaxpr(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            tm_concat,
            max_order=effective_order,
            output_pytree=output_pytree,
        )

        if len(out) == 1:
            return out[0]
        return out

    return wrapped


def nattm_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    max_order: "int | tuple[int, ...] " = 2,
    propagate_source_info=True,
    output_pytree: "PyTreeShape | None" = None,
) -> list[Any]:
    """Interpreter for Jaxpr with TaylorModel arguments.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to interpret.
    consts : list
        Constants for the jaxpr.
    *args : TaylorModel or Array
        Input arguments.
    max_order : int or tuple[int, ...]
        Maximum polynomial order (per-leaf tuple or broadcast int).
    propagate_source_info : bool
        Whether to propagate source info for debugging.
    output_pytree : PyTreeShape, optional
        If provided with num_leaves > 1, args is a single TaylorModel and
        jaxpr invars are mapped to slices of its output based on output_pytree.leaf_shapes.
    """
    import math

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)

    if output_pytree is not None:
        # Single TaylorModel with structured output - slice for each invar
        tm = args[0]
        sliced_args = []
        for i in range(output_pytree.num_leaves):
            slc = output_pytree.leaf_slice(i)
            sliced_args.append(tm[slc])
        safe_map(write, jaxpr.invars, sliced_args)
    else:
        safe_map(write, jaxpr.invars, args)

    lu = last_used(jaxpr)

    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None

        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any([istaylormodel(read(iv)) for iv in eqn.invars]):
                if eqn.primitive not in tm_inclusion_registry:
                    raise NotImplementedError(
                        f"{eqn.primitive} not in tm_inclusion_registry"
                    )
                ans = tm_inclusion_registry[eqn.primitive](
                    *subfuns, *invars, max_order=max_order, **bind_params
                )
            else:
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)

        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)

    return safe_map(read, jaxpr.outvars)


def _register_tm_univariate(primitive: Primitive, func: Callable = None) -> None:
    """Register a univariate primitive for Taylor model arithmetic.

    Parameters
    ----------
    primitive : Primitive
        The JAX primitive to register.
    func : Callable, optional
        The jnp-level function to use for jet tracing. If None, falls back
        to ``primitive.bind``. Some primitives (e.g. tan, asin, atan) need
        the jnp function because jet lacks direct rules for them.
    """

    # Create and register the TM handler
    def _tm_handler(
        x: TaylorModel, *, max_order: int = None, accuracy=None
    ) -> TaylorModel:
        return _tm_univariate(primitive, x, max_order=max_order, func=func)

    tm_inclusion_registry[primitive] = _tm_handler


def _make_tm_passthrough_p(primitive: Primitive) -> Callable[..., TaylorModel]:
    """Creates a TM handler that applies the primitive to each member separately.

    For operations like copy, convert_element_type, etc., we just apply
    the primitive independently to coeffs and remainder using tree_map.
    """

    def _tm_p(*args, max_order=None, **kwargs) -> TaylorModel:
        getcoeffs = lambda x: x.coeffs if istaylormodel(x) else x
        getlower = lambda x: x.remainder.lower if istaylormodel(x) else x
        getupper = lambda x: x.remainder.upper if istaylormodel(x) else x

        args_coeffs = jax.tree_util.tree_map(getcoeffs, args, is_leaf=istaylormodel)
        args_lower = jax.tree_util.tree_map(getlower, args, is_leaf=istaylormodel)
        args_upper = jax.tree_util.tree_map(getupper, args, is_leaf=istaylormodel)

        new_coeffs = primitive.bind(*args_coeffs, **kwargs)
        new_lower = primitive.bind(*args_lower, **kwargs)
        new_upper = primitive.bind(*args_upper, **kwargs)

        ref_tm = None
        for arg in jax.tree_util.tree_leaves(args, is_leaf=istaylormodel):
            if istaylormodel(arg):
                ref_tm = arg
                break

        return TaylorModel(
            new_coeffs,
            ref_tm.exponents,
            interval(new_lower, new_upper),
            ref_tm.flat_domain,
            flat_center=ref_tm.flat_center,
            _input_pytree=ref_tm._input_pytree,
            _output_pytree=ref_tm._output_pytree,
            _per_leaf_order=ref_tm._per_leaf_order,
        )

    return _tm_p


def _add_tm_passthrough_to_registry(primitive: Primitive) -> None:
    """Helper to add a passthrough primitive to the TM registry."""
    tm_inclusion_registry[primitive] = _make_tm_passthrough_p(primitive)


# Register passthrough operations
_add_tm_passthrough_to_registry(lax.copy_p)
_add_tm_passthrough_to_registry(lax.iota_p)
_add_tm_passthrough_to_registry(lax.convert_element_type_p)
_add_tm_passthrough_to_registry(debug_callback_p)


# --- Reshape operation (preserves polynomial structure) ---


def _make_tm_structural_p(primitive, adapt_coeff_kwargs):
    """Factory for structural TM operations that preserve polynomial structure.

    These operations act on the output shape of a TaylorModel. The monomial
    axis (last axis of coeffs) is left unchanged. ``adapt_coeff_kwargs``
    takes ``(kwargs, tm)`` and returns modified kwargs for the coeffs array
    (which has the extra trailing monomial axis).
    """

    def _tm_p(*args, max_order=None, **kwargs):
        # Find first TM arg
        ref_tm = None
        for arg in args:
            if istaylormodel(arg):
                ref_tm = arg
                break
        if ref_tm is None:
            return primitive.bind(*args, **kwargs)

        # Lift non-TM args to coefficient shape: place scalar value in
        # the constant-monomial column (index 0), zeros elsewhere.
        def _lift(arg):
            if istaylormodel(arg):
                return arg.coeffs
            if jnp.ndim(arg) < ref_tm.coeffs.ndim:
                m = ref_tm.num_monomials
                return jnp.concatenate(
                    [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
                )
            return arg

        # Apply primitive to coeffs with adapted kwargs
        coeff_kwargs = adapt_coeff_kwargs(kwargs, ref_tm)
        args_coeffs = [_lift(arg) for arg in args]
        new_coeffs = primitive.bind(*args_coeffs, **coeff_kwargs)

        # Apply primitive to remainder lower/upper with original kwargs
        args_lower = [
            arg.remainder.lower if istaylormodel(arg) else arg for arg in args
        ]
        args_upper = [
            arg.remainder.upper if istaylormodel(arg) else arg for arg in args
        ]
        new_lower = primitive.bind(*args_lower, **kwargs)
        new_upper = primitive.bind(*args_upper, **kwargs)

        return TaylorModel(
            new_coeffs,
            ref_tm.exponents,
            interval(new_lower, new_upper),
            ref_tm.flat_domain,
            flat_center=ref_tm.flat_center,
            _input_pytree=ref_tm._input_pytree,
            _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
            _per_leaf_order=ref_tm._per_leaf_order,
        )

    tm_inclusion_registry[primitive] = _tm_p


def _adapt_reshape(kwargs, tm):
    m = tm.num_monomials
    kw = {**kwargs, "new_sizes": (*kwargs["new_sizes"], m)}
    if kwargs.get("dimensions") is not None:
        kw["dimensions"] = (*kwargs["dimensions"], len(tm._output_shape))
    return kw


def _adapt_transpose(kwargs, tm):
    return {**kwargs, "permutation": (*kwargs["permutation"], len(tm._output_shape))}


def _adapt_squeeze(kwargs, tm):
    return kwargs  # monomial axis is never size-1


def _adapt_broadcast_in_dim(kwargs, tm):
    shape = kwargs["shape"]
    return {
        **kwargs,
        "shape": (*shape, tm.num_monomials),
        "broadcast_dimensions": (*kwargs["broadcast_dimensions"], len(shape)),
    }


def _adapt_slice(kwargs, tm):
    m = tm.num_monomials
    kw = {
        **kwargs,
        "start_indices": (*kwargs["start_indices"], 0),
        "limit_indices": (*kwargs["limit_indices"], m),
    }
    if kwargs.get("strides") is not None:
        kw["strides"] = (*kwargs["strides"], 1)
    return kw


def _adapt_concatenate(kwargs, tm):
    return kwargs  # dimension refers to output axes; monomial axis is last and shared


_make_tm_structural_p(lax.reshape_p, _adapt_reshape)
_make_tm_structural_p(lax.transpose_p, _adapt_transpose)
_make_tm_structural_p(lax.squeeze_p, _adapt_squeeze)
_make_tm_structural_p(lax.broadcast_in_dim_p, _adapt_broadcast_in_dim)
_make_tm_structural_p(lax.slice_p, _adapt_slice)
_make_tm_structural_p(lax.concatenate_p, _adapt_concatenate)


def _tm_dynamic_slice_p(x, *start_indices, slice_sizes, max_order=None) -> TaylorModel:
    """Handle dynamic slicing of Taylor models.

    start_indices are positional args, so this can't use the structural factory.
    """
    if not istaylormodel(x):
        return lax.dynamic_slice_p.bind(x, *start_indices, slice_sizes=slice_sizes)

    m = x.num_monomials
    new_coeffs = lax.dynamic_slice_p.bind(
        x.coeffs, *start_indices, 0, slice_sizes=(*slice_sizes, m)
    )
    new_lower = lax.dynamic_slice_p.bind(
        x.remainder.lower, *start_indices, slice_sizes=slice_sizes
    )
    new_upper = lax.dynamic_slice_p.bind(
        x.remainder.upper, *start_indices, slice_sizes=slice_sizes
    )

    return TaylorModel(
        new_coeffs,
        x.exponents,
        interval(new_lower, new_upper),
        x.flat_domain,
        flat_center=x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.dynamic_slice_p] = _tm_dynamic_slice_p


def _make_tm_scatter_p(primitive):
    """Factory for scatter TM operations (scatter, scatter_add, scatter_max, scatter_min).

    Scatter has positional args (operand, scatter_indices, updates) where
    scatter_indices are integer indices that must never be lifted to TM coefficients.
    The monomial axis is added to update_window_dims so the full [M] monomial
    slice is scattered as a unit.
    """

    def _tm_p(operand, scatter_indices, updates, *, dimension_numbers,
              max_order=None, **kwargs):
        op_is_tm = istaylormodel(operand)
        upd_is_tm = istaylormodel(updates)
        if not op_is_tm and not upd_is_tm:
            return primitive.bind(operand, scatter_indices, updates,
                                 dimension_numbers=dimension_numbers, **kwargs)

        ref_tm = operand if op_is_tm else updates
        m = ref_tm.num_monomials

        def _lift(arg):
            if istaylormodel(arg):
                return arg.coeffs
            return jnp.concatenate(
                [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
            )

        op_coeffs = _lift(operand)
        upd_coeffs = _lift(updates)

        # Add monomial axis (last axis of updates coeffs) to update_window_dims
        dn = dimension_numbers
        mono_ax = upd_coeffs.ndim - 1
        new_dn = lax.ScatterDimensionNumbers(
            update_window_dims=(*dn.update_window_dims, mono_ax),
            inserted_window_dims=dn.inserted_window_dims,
            scatter_dims_to_operand_dims=dn.scatter_dims_to_operand_dims,
            operand_batching_dims=dn.operand_batching_dims,
            scatter_indices_batching_dims=dn.scatter_indices_batching_dims,
        )

        new_coeffs = primitive.bind(op_coeffs, scatter_indices, upd_coeffs,
                                    dimension_numbers=new_dn, **kwargs)

        # Remainder bounds: non-TM args have zero remainder (their full value
        # is captured in the polynomial coefficients via _lift).
        op_lo = operand.remainder.lower if op_is_tm else jnp.zeros_like(operand)
        op_hi = operand.remainder.upper if op_is_tm else jnp.zeros_like(operand)
        upd_lo = updates.remainder.lower if upd_is_tm else jnp.zeros_like(updates)
        upd_hi = updates.remainder.upper if upd_is_tm else jnp.zeros_like(updates)

        new_lower = primitive.bind(op_lo, scatter_indices, upd_lo,
                                   dimension_numbers=dimension_numbers, **kwargs)
        new_upper = primitive.bind(op_hi, scatter_indices, upd_hi,
                                   dimension_numbers=dimension_numbers, **kwargs)

        return TaylorModel(
            new_coeffs,
            ref_tm.exponents,
            interval(new_lower, new_upper),
            ref_tm.flat_domain,
            flat_center=ref_tm.flat_center,
            _input_pytree=ref_tm._input_pytree,
            _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
            _per_leaf_order=ref_tm._per_leaf_order,
        )

    tm_inclusion_registry[primitive] = _tm_p


_make_tm_scatter_p(lax.scatter_p)
_make_tm_scatter_p(lax.scatter_add_p)


def _make_tm_scatter_nonlinear_p(primitive):
    """Factory for non-linear scatter ops (scatter_max, scatter_min).

    Unlike scatter (set) and scatter_add which are structural/linear,
    scatter_max/scatter_min apply a non-linear combiner. Applying max/min
    to polynomial coefficients is not meaningful, so we fall back to
    interval arithmetic: compute interval hulls, apply the primitive to
    bounds, and return a constant TM.
    """

    def _tm_p(operand, scatter_indices, updates, *, dimension_numbers,
              max_order=None, **kwargs):
        op_is_tm = istaylormodel(operand)
        upd_is_tm = istaylormodel(updates)
        if not op_is_tm and not upd_is_tm:
            return primitive.bind(operand, scatter_indices, updates,
                                 dimension_numbers=dimension_numbers, **kwargs)

        ref_tm = operand if op_is_tm else updates

        # Get interval bounds for operand and updates
        if op_is_tm:
            op_hull = operand.interval_hull()
            op_lo, op_hi = op_hull.lower, op_hull.upper
        else:
            op_lo = op_hi = jnp.asarray(operand)

        if upd_is_tm:
            upd_hull = updates.interval_hull()
            upd_lo, upd_hi = upd_hull.lower, upd_hull.upper
        else:
            upd_lo = upd_hi = jnp.asarray(updates)

        # scatter_max/scatter_min are monotone in both args, so
        # applying to (lo, lo) and (hi, hi) gives valid enclosure.
        new_lower = primitive.bind(op_lo, scatter_indices, upd_lo,
                                   dimension_numbers=dimension_numbers, **kwargs)
        new_upper = primitive.bind(op_hi, scatter_indices, upd_hi,
                                   dimension_numbers=dimension_numbers, **kwargs)

        return _taylor_model_constant_impl(
            interval(new_lower, new_upper),
            ref_tm.flat_domain,
            ref_tm._per_leaf_order,
            center=ref_tm.flat_center,
            _input_pytree=ref_tm._input_pytree,
            _per_leaf_order=ref_tm._per_leaf_order,
        )

    tm_inclusion_registry[primitive] = _tm_p


_make_tm_scatter_nonlinear_p(lax.scatter_max_p)
_make_tm_scatter_nonlinear_p(lax.scatter_min_p)


def _tm_select_n_p(pred, *cases, max_order=None):
    """Taylor model select_n: pred chooses among cases element-wise.

    pred is integer/boolean (never a TM). Each case may be a TM.
    The predicate is broadcast along the monomial axis for coefficients.
    """
    if not any(istaylormodel(c) for c in cases):
        return lax.select_n_p.bind(pred, *cases)

    ref_tm = next(c for c in cases if istaylormodel(c))
    m = ref_tm.num_monomials

    def _lift(arg):
        if istaylormodel(arg):
            return arg.coeffs
        return jnp.concatenate(
            [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
        )

    # Broadcast pred along monomial axis so it selects entire polynomial slices
    pred_coeffs = jnp.broadcast_to(pred[..., None], (*pred.shape, m))
    cases_coeffs = [_lift(c) for c in cases]
    new_coeffs = lax.select_n_p.bind(pred_coeffs, *cases_coeffs)

    # Remainder: non-TM cases have zero remainder
    cases_lo = [c.remainder.lower if istaylormodel(c) else jnp.zeros_like(c) for c in cases]
    cases_hi = [c.remainder.upper if istaylormodel(c) else jnp.zeros_like(c) for c in cases]
    new_lower = lax.select_n_p.bind(pred, *cases_lo)
    new_upper = lax.select_n_p.bind(pred, *cases_hi)

    return TaylorModel(
        new_coeffs,
        ref_tm.exponents,
        interval(new_lower, new_upper),
        ref_tm.flat_domain,
        flat_center=ref_tm.flat_center,
        _input_pytree=ref_tm._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=ref_tm._per_leaf_order,
    )


if hasattr(lax, "select_n_p"):
    tm_inclusion_registry[lax.select_n_p] = _tm_select_n_p


def _tm_split_p(x, *, sizes, axis, max_order=None):
    """Taylor model split: split along an output axis, preserving polynomials.

    split_p is a multiple_results primitive — returns a list of TaylorModels.
    The axis and sizes refer to output dimensions; the monomial axis (last)
    is carried through unchanged.
    """
    if not istaylormodel(x):
        return lax.split_p.bind(x, sizes=sizes, axis=axis)

    coeff_chunks = lax.split_p.bind(x.coeffs, sizes=sizes, axis=axis)
    lo_chunks = lax.split_p.bind(x.remainder.lower, sizes=sizes, axis=axis)
    hi_chunks = lax.split_p.bind(x.remainder.upper, sizes=sizes, axis=axis)

    return [
        TaylorModel(
            c, x.exponents, interval(lo, hi),
            x.flat_domain, flat_center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=PyTreeShape.flat(c.shape[:-1]),
            _per_leaf_order=x._per_leaf_order,
        )
        for c, lo, hi in zip(coeff_chunks, lo_chunks, hi_chunks)
    ]


if hasattr(lax, "split_p"):
    tm_inclusion_registry[lax.split_p] = _tm_split_p


# --- Higher-order primitives ---


def _tm_pjit_p(*args, max_order=None, **bind_params) -> TaylorModel:
    """Handle pjit by evaluating the inner jaxpr."""
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr

    if max_order is None:
        max_order = None
        for arg in args:
            if istaylormodel(arg):
                max_order = (
                    arg._per_leaf_order
                    if max_order is None
                    else _max_order(max_order, arg._per_leaf_order)
                )
        if max_order is None:
            max_order = (2,)

    return nattm_jaxpr(bind_jaxpr, [], *args, max_order=max_order)


tm_inclusion_registry[jax._src.pjit.pjit_p] = _tm_pjit_p


# --- Arithmetic operations ---


def _tm_add_p(
    x: TaylorModel, y: TaylorModel | ArrayLike, *, max_order=None
) -> TaylorModel:
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

        # Add remainders (Interval addition auto-broadcasts)
        new_remainder = x.remainder + y.remainder

        per_leaf_order = _max_order(x._per_leaf_order, y._per_leaf_order)

        result = TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            x.flat_domain,
            flat_center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=per_leaf_order,
        )
        # Convert to canonical form for compatibility
        return result.to_canonical(per_leaf_order)

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
        new_remainder = x.remainder.broadcast_to(broadcast_shape)

        result = TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            x.flat_domain,
            flat_center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )
        return result.to_canonical(x._per_leaf_order)

    elif istaylormodel(y):
        # Array + TM
        return _tm_add_p(y, x)
    else:
        return x + y


tm_inclusion_registry[lax.add_p] = _tm_add_p
tm_inclusion_registry[ad_util.add_any_p] = _tm_add_p
TaylorModel.__add__ = _tm_add_p
TaylorModel.__radd__ = _tm_add_p


def _tm_sub_p(
    x: TaylorModel, y: TaylorModel | ArrayLike, *, max_order=None
) -> TaylorModel:
    """Taylor model subtraction."""
    if istaylormodel(y):
        return _tm_add_p(x, _tm_neg_p(y))
    return _tm_add_p(x, -jnp.asarray(y))


tm_inclusion_registry[lax.sub_p] = _tm_sub_p
TaylorModel.__sub__ = _tm_sub_p
TaylorModel.__rsub__ = lambda self, other: _tm_sub_p(other, self)


def _tm_neg_p(x: TaylorModel, *, max_order=None) -> TaylorModel:
    """Taylor model negation."""
    if not istaylormodel(x):
        return -x
    return TaylorModel(
        -x.coeffs,
        x.exponents,
        -x.remainder,
        x.flat_domain,
        flat_center=x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=x._output_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.neg_p] = _tm_neg_p
TaylorModel.__neg__ = _tm_neg_p


def _truncate_product(
    coeffs, exponents, max_order, shifted_domain, leaf_shapes=None, per_leaf_order=None
):
    """Truncate polynomial product terms above max_order into an interval remainder.

    Parameters
    ----------
    coeffs : array, shape (*output_shape, num_terms)
        Product coefficients (e.g. from outer product of two monomial sets).
    exponents : array, shape (d, num_terms)
        Product exponents (sum of parent exponents for each term pair).
    max_order : int or tuple[int, ...]
        Per-variable maximum polynomial order to retain (used when leaf_shapes is None).
    shifted_domain : Interval, shape (d,)
        The shifted domain (D - center) for bounding monomials.
    leaf_shapes : tuple[tuple[int, ...], ...], optional
        If provided, use per-leaf total degree bounds instead of per-variable.
    per_leaf_order : tuple[int, ...], optional
        Per-leaf total degree bounds. Required if leaf_shapes is provided.

    Returns
    -------
    kept_coeffs : array, shape (*output_shape, num_terms)
        Coefficients with high-order terms zeroed out.
    truncated_remainder : Interval, shape (*output_shape,)
        Rigorous bound on the contribution of the removed terms.
    """
    # Determine keep mask based on mode
    if leaf_shapes is not None and per_leaf_order is not None:
        # Per-leaf total degree mode
        keep_mask = _check_per_leaf_bounds(exponents, leaf_shapes, per_leaf_order)
        static_max = max(per_leaf_order) * 2
    else:
        # Per-variable mode (original behavior)
        if isinstance(max_order, int):
            max_order = tuple([max_order] * exponents.shape[0])
        target_arr = jnp.array(max_order, dtype=jnp.int32)[:, None]  # (d, 1)
        keep_mask = jnp.all(exponents <= target_arr, axis=0)  # (num_terms,)
        static_max = (
            max(max_order) * 2 if isinstance(max_order, tuple) else max_order * 2
        )

    kept_coeffs = jnp.where(keep_mask, coeffs, 0.0)
    truncated_coeffs = jnp.where(keep_mask, 0.0, coeffs)

    # Bound truncated monomials over shifted domain
    mono_bounds = _bound_monomials_over_domain(exponents, shifted_domain, static_max)
    mono_lower = mono_bounds.lower
    mono_upper = mono_bounds.upper

    c_pos = jnp.maximum(truncated_coeffs, 0.0)
    c_neg = jnp.minimum(truncated_coeffs, 0.0)
    term_lower = c_pos * mono_lower + c_neg * mono_upper
    term_upper = c_pos * mono_upper + c_neg * mono_lower

    truncated_remainder = interval(
        jnp.sum(term_lower, axis=-1),
        jnp.sum(term_upper, axis=-1),
    )

    return kept_coeffs, truncated_remainder


def _tm_mul_p(
    x: TaylorModel, y: TaylorModel | ArrayLike, *, max_order: int = None
) -> TaylorModel:
    """Taylor model element-wise multiplication with broadcasting support."""
    if istaylormodel(x) and istaylormodel(y):
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")

        # Broadcast output shapes, throws an error if outputs are not compatible
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
        # Outer product (discrete convolution) along the last monomial axis
        coeff1 = x_coeffs[..., :, None]  # (*broadcast_shape, m1, 1)
        coeff2 = y_coeffs[..., None, :]  # (*broadcast_shape, 1, m2)
        all_coeffs = (coeff1 * coeff2).reshape(*broadcast_shape, m1 * m2)

        effective_order = (
            max_order
            if max_order is not None
            else _max_order(x._per_leaf_order, y._per_leaf_order)
        )

        # Truncate high-order product terms into remainder
        new_coeffs, truncated_remainder = _truncate_product(
            all_coeffs,
            all_exp,
            effective_order,
            x.shifted_domain,
            leaf_shapes=x._leaf_shapes,
            per_leaf_order=effective_order,
        )

        new_exponents = all_exp

        # Compute remainder bounds using Interval operations
        # (p1 + r1) * (p2 + r2) = p1*p2 + p1*r2 + r1*p2 + r1*r2
        p1_bounds = _bound_polynomial(x)
        p2_bounds = _bound_polynomial(y)

        # Cross terms using Interval multiplication (auto-broadcasts)
        p1_r2 = p1_bounds * y.remainder
        r1_p2 = x.remainder * p2_bounds
        r1_r2 = x.remainder * y.remainder

        # Combined remainder
        new_remainder = p1_r2 + r1_p2 + r1_r2 + truncated_remainder

        result = TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            x.flat_domain,
            flat_center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=effective_order,
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

        # Scale remainder (Interval * array auto-broadcasts)
        new_remainder = x.remainder * alpha

        return TaylorModel(
            new_coeffs,
            x.exponents,
            new_remainder,
            x.flat_domain,
            flat_center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )

    elif istaylormodel(y):
        return _tm_mul_p(y, x, max_order=max_order)
    else:
        return x * y


tm_inclusion_registry[lax.mul_p] = _tm_mul_p

TaylorModel.__mul__ = _tm_mul_p
TaylorModel.__rmul__ = _tm_mul_p
TaylorModel.multiply = _tm_mul_p


from jax.experimental.jet import jet
from immrax.inclusion.nif import natif


def _tm_univariate(
    primitive_p: Primitive,
    x: TaylorModel,
    *,
    max_order: int | None = None,
    func: Callable | None = None,
) -> TaylorModel:
    """Generic implementation for univariate Taylor Model operations.

    Uses jax.experimental.jet for Taylor coefficients and natif for
    Lagrange remainder bounds.

    Supports arbitrary output shapes by flattening, processing, and reshaping.
    """
    if not istaylormodel(x):
        return primitive_p.bind(x)

    per_leaf_order = max_order if max_order is not None else x._per_leaf_order
    # For univariate composition, use max of per-leaf orders as expansion depth
    if isinstance(per_leaf_order, tuple):
        order = max(per_leaf_order)
    else:
        order = per_leaf_order
    output_shape = x._output_shape
    c = x.constant_term  # (*output_shape,)

    prim_func = func if func is not None else (lambda v: primitive_p.bind(v))

    # Flatten output shape for processing
    import math

    n_flat = math.prod(output_shape) if output_shape else 1
    c_flat = c.reshape(-1) if output_shape else c[None]  # (n_flat,)

    # 1. Compute Taylor coefficients f^(k)(c)/k! using jet (univariate)
    def get_coeffs(val):
        primals = (val,)
        series = ((1.0,) + (0.0,) * (order - 1),)
        f_val, f_series = jet(prim_func, primals, series)
        # jet returns raw derivatives [f', f'', f''', ...], need to divide by k! for Taylor coefficients
        # taylor_coeffs = [f_val] + [f_series[k] * inv_fact(k + 1) for k in range(len(f_series))]
        return jnp.concatenate(
            (
                jnp.asarray([f_val]),
                jnp.asarray(f_series) * inv_fact(jnp.arange(1, order + 1)),
            )
        )

    # Vectorize over flattened output dimensions, shape: (n_flat, order + 1)
    coeffs_raw = jax.vmap(get_coeffs)(c_flat)

    # Reshape coeffs_raw back to (*output_shape, order + 1) for TM construction
    if output_shape:
        coeffs_raw_shaped = coeffs_raw.reshape(*output_shape, order + 1)
    else:
        coeffs_raw_shaped = coeffs_raw[0]  # (order + 1,) for scalar

    # 2. Construct resulting polynomial using Horner's method (Python loop)
    # Result = a_0 + z * (a_1 + z * (a_2 + ...)) where z = x - c
    z = x - c

    # Get highest order coefficient
    if output_shape:
        init_coeff = coeffs_raw_shaped[..., order]  # (*output_shape,)
        # Coefficients in descending order (order-1 down to 0)
        horner_coeffs = [coeffs_raw_shaped[..., i] for i in range(order - 1, -1, -1)]
    else:
        init_coeff = coeffs_raw_shaped[order]  # scalar
        horner_coeffs = [coeffs_raw_shaped[i] for i in range(order - 1, -1, -1)]

    result = _taylor_model_constant_impl(
        interval(init_coeff),
        x.flat_domain,
        per_leaf_order,
        center=x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=x._output_pytree,
        _per_leaf_order=x._per_leaf_order,
    )
    for coeff in horner_coeffs:
        term = _taylor_model_constant_impl(
            interval(coeff),
            x.flat_domain,
            per_leaf_order,
            center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )
        result = result.multiply(z, max_order=per_leaf_order) + term

    # 3. Compute Lagrange remainder: f^(order+1)(xi) / (order+1)! * z^(order+1)
    x_hull = x.interval_hull()
    x_hull_flat_lower = x_hull.lower.reshape(-1) if output_shape else x_hull.lower[None]
    x_hull_flat_upper = x_hull.upper.reshape(-1) if output_shape else x_hull.upper[None]

    def get_deriv_bound(i):
        def deriv_func(v):
            primals = (v,)
            series = ((1.0,) + (0.0,) * order,)
            _, f_series = jet(prim_func, primals, series)
            # jet returns raw derivative f^(order+1)(v), not normalized by factorial
            return f_series[-1]

        iv_i = interval(x_hull_flat_lower[i], x_hull_flat_upper[i])
        return natif(deriv_func)(iv_i)

    # Compute bounds for each flattened dimension
    deriv_bounds = [get_deriv_bound(i) for i in range(n_flat)]
    deriv_bound_flat = interval(
        jnp.stack([b.lower for b in deriv_bounds]),
        jnp.stack([b.upper for b in deriv_bounds]),
    )

    # Reshape deriv_bound back to output_shape
    if output_shape:
        deriv_bound = interval(
            deriv_bound_flat.lower.reshape(*output_shape),
            deriv_bound_flat.upper.reshape(*output_shape),
        )
    else:
        deriv_bound = interval(deriv_bound_flat.lower[0], deriv_bound_flat.upper[0])

    # Compute remainder bound
    # z^(n+1) ranges from 0 (at center) to ±z_mag^(n+1) (at boundaries)
    # For even n+1: z^(n+1) ∈ [0, z_mag^(n+1)]
    # For odd n+1: z^(n+1) ∈ [-z_mag^(n+1), z_mag^(n+1)]
    z_bound = z._bound_polynomial() + z.remainder
    z_mag = jnp.maximum(jnp.abs(z_bound.lower), jnp.abs(z_bound.upper))
    z_pow_mag = z_mag ** (order + 1)
    if (order + 1) % 2 == 0:
        # Even power: always non-negative, ranges from 0 to z_pow_mag
        z_pow_interval = interval(jnp.zeros_like(z_pow_mag), z_pow_mag)
    else:
        # Odd power: can be negative or positive
        z_pow_interval = interval(-z_pow_mag, z_pow_mag)
    rem_term = deriv_bound * z_pow_interval * inv_fact(order + 1)

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

    return TaylorModel(
        result.coeffs,
        result.exponents,
        final_remainder,
        result.flat_domain,
        flat_center=result.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=x._output_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


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


def _tm_integer_pow_p(x: TaylorModel, y: int, *, max_order=None) -> TaylorModel:
    """Taylor model integer power."""
    if not istaylormodel(x):
        return lax.integer_pow(x, y)

    order = max_order if max_order is not None else x._per_leaf_order

    if y == 0:
        return _taylor_model_constant_impl(
            interval(jnp.ones(x.n, dtype=x.dtype)),
            x.flat_domain,
            order,
            center=x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )
    elif y < 0:
        # x^(-n) = 1/x^n
        pos_pow = _tm_integer_pow_p(x, -y, max_order=order)
        return _tm_div_p(
            _taylor_model_constant_impl(
                interval(jnp.ones(x.n, dtype=x.dtype)),
                x.flat_domain,
                order,
                center=x.flat_center,
                _input_pytree=x._input_pytree,
                _output_pytree=x._output_pytree,
                _per_leaf_order=x._per_leaf_order,
            ),
            pos_pow,
            max_order=order,
        )
    else:
        result = x
        for _ in range(y - 1):
            result = result.multiply(x, max_order=order)
        return result


tm_inclusion_registry[lax.integer_pow_p] = _tm_integer_pow_p

TaylorModel.__pow__ = _tm_integer_pow_p


def _tm_square_p(x: TaylorModel, *, max_order: int = None) -> TaylorModel:
    """Taylor model square."""
    return _tm_integer_pow_p(x, 2, max_order=max_order)


if hasattr(lax, "square_p"):
    tm_inclusion_registry[lax.square_p] = _tm_square_p
    pass


def _tm_pow_p(x: TaylorModel, y: TaylorModel, *, max_order=None) -> TaylorModel:
    """Taylor model general power: x^y = exp(y * log(x))."""
    order = max_order
    if order is None:
        order = None
        if istaylormodel(x):
            order = x._per_leaf_order
        if istaylormodel(y):
            order = (
                y._per_leaf_order
                if order is None
                else _max_order(order, y._per_leaf_order)
            )
        if order is None:
            order = (2,)

    log_x = _tm_log_p(x, max_order=order)
    y_log_x = _tm_mul_p(y, log_x, max_order=order)
    return _tm_exp_p(y_log_x, max_order=order)


tm_inclusion_registry[lax.pow_p] = _tm_pow_p


# --- Transcendental functions ---

_register_tm_univariate(lax.exp_p)
_register_tm_univariate(lax.log_p)
_register_tm_univariate(lax.log1p_p)
_register_tm_univariate(lax.sin_p)
_register_tm_univariate(lax.cos_p)
_register_tm_univariate(lax.tan_p, jnp.tan)  # no jet rule for tan
_register_tm_univariate(lax.tanh_p)
_register_tm_univariate(lax.sqrt_p)


def _tm_abs_p(x: TaylorModel, *, max_order=None) -> TaylorModel:
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
    abs_lower = jnp.where(
        contains_zero, 0.0, jnp.minimum(jnp.abs(hull.lower), jnp.abs(hull.upper))
    )
    abs_upper = jnp.maximum(jnp.abs(hull.lower), jnp.abs(hull.upper))

    center = (abs_lower + abs_upper) / 2
    pert = (abs_upper - abs_lower) / 2

    return _taylor_model_constant_impl(
        icentpert(center, pert),
        x.flat_domain,
        x._per_leaf_order,
        center=x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=x._output_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.abs_p] = _tm_abs_p


# --- Linear algebra ---


def _tm_dot_general_array(arr, tm, dim_nums):
    """Handle dot_general(Array, TM) — the primary Array-TM implementation.

    The monomial axis (last axis of tm.coeffs) is on the rhs, so it
    naturally ends up at the end of the dot_general output.
    """
    from immrax.inclusion.nif import inclusion_registry

    # arr @ (P(x) + R) = arr @ P(x) + arr @ R

    # Linear operation on polynomial is just a linear operation on the coefficients
    new_coeffs = lax.dot_general(arr, tm.coeffs, dimension_numbers=dim_nums)
    # Interval arithmetic for remainder (use registry directly — natif would
    # re-trace lax.dot_general, which fails on dimension_numbers tuples)
    new_remainder = inclusion_registry[lax.dot_general_p](
        interval(arr, arr), tm.remainder, dimension_numbers=dim_nums
    )

    return TaylorModel(
        new_coeffs,
        tm.exponents,
        new_remainder,
        tm.flat_domain,
        flat_center=tm.flat_center,
        _input_pytree=tm._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=tm._per_leaf_order,
    )


def _tm_dot_general_p(
    A: TaylorModel, B: TaylorModel, *, max_order: int = None, **kwargs
) -> TaylorModel:
    """Taylor model general dot product.

    Three cases:
    - Array @ TM: apply dot_general to coefficients, interval dot_general for remainder
    - TM @ Array: swap to Array @ TM with swapped dimension_numbers, transpose result
    - TM @ TM: polynomial convolution with proper monomial multiplication
    """
    dimension_numbers = kwargs["dimension_numbers"]
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

    # --- Array @ TM ---
    if not istaylormodel(A) and istaylormodel(B):
        return _tm_dot_general_array(jnp.asarray(A), B, dimension_numbers)

    # --- TM @ Array: swap to Array @ TM, then transpose output ---
    if istaylormodel(A) and not istaylormodel(B):
        B_arr = jnp.asarray(B)
        swapped_dims = ((rhs_contracting, lhs_contracting), (rhs_batch, lhs_batch))
        result = _tm_dot_general_array(B_arr, A, swapped_dims)

        # Fix output dimension order.
        # dot_general(A, B) output: (*batch, *rem_A, *rem_B)
        # dot_general(B, A, swapped) output: (*batch, *rem_B, *rem_A)
        # For coeffs, monomial axis is at the end in both cases.
        nb = len(lhs_batch)
        n_rem_A = len(A._output_shape) - len(lhs_contracting) - len(lhs_batch)
        n_rem_B = B_arr.ndim - len(rhs_contracting) - len(rhs_batch)

        if n_rem_A > 0 and n_rem_B > 0:
            # Transpose: (*batch, *rem_B, *rem_A, mono) → (*batch, *rem_A, *rem_B, mono)
            perm = (
                *range(nb),
                *range(nb + n_rem_B, nb + n_rem_B + n_rem_A),
                *range(nb, nb + n_rem_B),
                nb + n_rem_A + n_rem_B,  # mono axis
            )
            new_coeffs = jnp.transpose(result.coeffs, perm)
            rem_perm = perm[:-1]  # same without mono axis
            new_remainder = result.remainder.transpose(*rem_perm)
            return TaylorModel(
                new_coeffs,
                result.exponents,
                new_remainder,
                result.flat_domain,
                flat_center=result.flat_center,
                _input_pytree=result._input_pytree,
                _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
                _per_leaf_order=result._per_leaf_order,
            )
        return result

    # --- TM @ TM: polynomial convolution ---
    if istaylormodel(A) and istaylormodel(B):
        if A.d != B.d:
            raise ValueError(f"Domain dimensions must match: {A.d} vs {B.d}")

        effective_order = (
            max_order
            if max_order is not None
            else _max_order(A._per_leaf_order, B._per_leaf_order)
        )
        d = A.d
        m1 = A.num_monomials
        m2 = B.num_monomials

        # Compute product exponents: (d, m1*m2)
        product_exp = (A.exponents[:, :, None] + B.exponents[:, None, :]).reshape(
            d, m1 * m2
        )

        # Compute product coefficients via double vmap over monomial axes (last axis)
        def contract_mono_pair(a_mono_coeffs, b_mono_coeffs):
            return lax.dot_general(
                a_mono_coeffs, b_mono_coeffs, dimension_numbers=dimension_numbers
            )

        # vmap over m2 (last axis of B.coeffs), then m1 (last axis of A.coeffs)
        contract_over_m2 = jax.vmap(contract_mono_pair, (None, -1), -1)
        contract_over_m1m2 = jax.vmap(contract_over_m2, (-1, None), -1)

        result_coeffs_mm = contract_over_m1m2(
            A.coeffs, B.coeffs
        )  # (*result_shape, m1, m2)

        # Merge monomial pair axes into single axis: (*result_shape, m1*m2)
        result_coeffs = result_coeffs_mm.reshape(*result_coeffs_mm.shape[:-2], m1 * m2)

        # Truncate high-order product terms into remainder
        new_coeffs, truncated_remainder = _truncate_product(
            result_coeffs,
            product_exp,
            effective_order,
            A.shifted_domain,
            leaf_shapes=A._leaf_shapes,
            per_leaf_order=A._per_leaf_order,
        )

        # Cross terms: p_A · r_B + r_A · p_B + r_A · r_B
        p_A_bounds = _bound_polynomial(A)
        p_B_bounds = _bound_polynomial(B)

        from immrax.inclusion.nif import inclusion_registry

        iv_dot = lambda a, b: inclusion_registry[lax.dot_general_p](
            a, b, dimension_numbers=dimension_numbers
        )

        new_remainder = (
            iv_dot(p_A_bounds, B.remainder)
            + iv_dot(A.remainder, p_B_bounds)
            + iv_dot(A.remainder, B.remainder)
            + truncated_remainder
        )

        result = TaylorModel(
            new_coeffs,
            product_exp,
            new_remainder,
            A.flat_domain,
            flat_center=A.flat_center,
            _input_pytree=A._input_pytree,
            _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
            _per_leaf_order=effective_order,
        )
        return result.to_canonical(effective_order)

    return lax.dot_general_p.bind(A, B, **kwargs)


tm_inclusion_registry[lax.dot_general_p] = _tm_dot_general_p

TaylorModel.__matmul__ = nattm(jnp.matmul)
TaylorModel.__rmatmul__ = lambda self, other: nattm(jnp.matmul)(other, self)


# --- Comparison operations (return intervals/arrays, not TMs) ---


def _tm_max_p(x: TaylorModel, y: TaylorModel, *, max_order=None) -> TaylorModel:
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

    return _taylor_model_constant_impl(
        result_interval,
        ref.flat_domain,
        ref._per_leaf_order,
        center=ref.flat_center,
        _input_pytree=ref._input_pytree,
        _output_pytree=ref._output_pytree,
        _per_leaf_order=ref._per_leaf_order,
    )


tm_inclusion_registry[lax.max_p] = _tm_max_p


def _tm_min_p(x: TaylorModel, y: TaylorModel, *, max_order=None) -> TaylorModel:
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

    return _taylor_model_constant_impl(
        result_interval,
        ref.flat_domain,
        ref._per_leaf_order,
        center=ref.flat_center,
        _input_pytree=ref._input_pytree,
        _output_pytree=ref._output_pytree,
        _per_leaf_order=ref._per_leaf_order,
    )


tm_inclusion_registry[lax.min_p] = _tm_min_p


# --- Reduction operations ---


def _tm_reduce_sum_p(x: TaylorModel, *, axes, max_order=None) -> TaylorModel:
    """Taylor model sum reduction over output shape dimensions.

    Sum preserves polynomial structure (sum of polynomials is a polynomial).
    """
    if not istaylormodel(x):
        return lax.reduce_sum_p.bind(x, axes=axes)

    # axes refer to output shape dimensions, not monomial axis
    # coeffs has shape (*output_shape, m)
    # Sum over specified axes while preserving monomial axis

    new_coeffs = jnp.sum(x.coeffs, axis=axes)
    new_remainder = x.remainder.sum(axis=axes)

    return TaylorModel(
        new_coeffs,
        x.exponents,
        new_remainder,
        x.flat_domain,
        flat_center=x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.reduce_sum_p] = _tm_reduce_sum_p


def _tm_reduce_max_p(x: TaylorModel, *, axes, max_order=None) -> TaylorModel:
    """Taylor model max reduction.

    Max is not polynomial-preserving, so we fall back to interval bounds.
    """
    if not istaylormodel(x):
        return lax.reduce_max_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(jnp.max(hull.lower, axis=axes), jnp.max(hull.upper, axis=axes))

    return _taylor_model_constant_impl(
        result,
        x.flat_domain,
        x._per_leaf_order,
        center=x.flat_center,
        _input_pytree=x._input_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.reduce_max_p] = _tm_reduce_max_p


def _tm_reduce_min_p(x: TaylorModel, *, axes, max_order=None) -> TaylorModel:
    """Taylor model min reduction.

    Min is not polynomial-preserving, so we fall back to interval bounds.
    """
    if not istaylormodel(x):
        return lax.reduce_min_p.bind(x, axes=axes)

    hull = x.interval_hull()
    result = interval(jnp.min(hull.lower, axis=axes), jnp.min(hull.upper, axis=axes))

    return _taylor_model_constant_impl(
        result,
        x.flat_domain,
        x._per_leaf_order,
        center=x.flat_center,
        _input_pytree=x._input_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


tm_inclusion_registry[lax.reduce_min_p] = _tm_reduce_min_p


# --- Inverse trig functions ---

_register_tm_univariate(lax.asin_p, jnp.arcsin)  # no jet rule for asin
_register_tm_univariate(lax.atan_p, jnp.arctan)  # no jet rule for atan


# --- Reciprocal ---


def _tm_reciprocal_p(x: TaylorModel, *, max_order=None) -> TaylorModel:
    """Taylor model reciprocal: 1/x."""
    eff_order = (
        max_order if max_order else (x._per_leaf_order if istaylormodel(x) else (2,))
    )
    one = _taylor_model_constant_impl(
        interval(
            jnp.ones(
                x.n if istaylormodel(x) else 1,
                dtype=x.dtype if istaylormodel(x) else jnp.float32,
            )
        ),
        x.flat_domain if istaylormodel(x) else icentpert(jnp.zeros(1), jnp.ones(1)),
        eff_order,
        center=x.flat_center if istaylormodel(x) else None,
        _input_pytree=x._input_pytree if istaylormodel(x) else None,
        _output_pytree=x._output_pytree if istaylormodel(x) else None,
        _per_leaf_order=x._per_leaf_order if istaylormodel(x) else None,
    )
    return _tm_div_p(one, x, max_order=max_order)
