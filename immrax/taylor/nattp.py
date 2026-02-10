"""Natural Taylor Polynomial Function — Jaxpr interpreter for TaylorPolynomial.

Analogous to nattm for TaylorModel, but propagates TaylorPolynomials through
functions (composition) by silently discarding high-order terms instead of
tracking them in an interval remainder.
"""

from functools import wraps
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
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
from jax.experimental.jet import jet
from jaxtyping import Array, ArrayLike

from immrax.taylor.taylor_polynomial import TaylorPolynomial, taylor_polynomial_concatenate
from immrax.taylor.taylor_model import (
    PyTreeShape,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _leaf_slice,
    _merge_taylor_terms,
    _max_order,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

tp_inclusion_registry: dict[Primitive, Callable] = {}
_tp_needs_max_order: set[Primitive] = set()
_tp_univariate_prim_to_func: dict[Primitive, Callable] = {}


def istaylorpolynomial(x) -> bool:
    return isinstance(x, TaylorPolynomial)


# ---------------------------------------------------------------------------
# Helper: constant TP
# ---------------------------------------------------------------------------


def _tp_constant(
    value: jax.Array,
    d: int,
    order: "int | tuple[int, ...]",
    flat_center: jax.Array,
    _domain_treedef: "jax.tree_util.PyTreeDef | None" = None,
    _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
    _per_leaf_order: "tuple[int, ...] | None" = None,
    _input_pytree: "PyTreeShape | None" = None,
    _output_pytree: "PyTreeShape | None" = None,
) -> TaylorPolynomial:

    # Resolve _input_pytree
    if _input_pytree is None:
        _input_pytree = PyTreeShape(_domain_treedef, _leaf_shapes)
    _leaf_shapes = _input_pytree.leaf_shapes
    _per_leaf_order_resolved = _per_leaf_order if _per_leaf_order is not None else (
        tuple([order] * len(_leaf_shapes)) if isinstance(order, int) else tuple(order)
    )

    output_shape = value.shape
    exponents = _get_leaf_total_degree_exponents(_leaf_shapes, _per_leaf_order_resolved)

    num_monomials = exponents.shape[1]

    if len(output_shape) == 0:
        coeffs = jnp.zeros((num_monomials,), dtype=value.dtype)
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
        coeffs = coeffs.at[const_idx].set(value)
    else:
        coeffs = jnp.zeros((*output_shape, num_monomials), dtype=value.dtype)
        const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
        coeffs = coeffs.at[..., const_idx].set(value)

    if _output_pytree is None:
        _output_pytree = PyTreeShape.flat(output_shape)

    return TaylorPolynomial(
        coeffs,
        exponents,
        flat_center,
        _input_pytree=_input_pytree,
        _output_pytree=_output_pytree,
        _per_leaf_order=_per_leaf_order_resolved,
    )


def _tp_from_array(
    arr: jax.Array,
    d: int,
    order: "int | tuple[int, ...]",
    flat_center: jax.Array,
    _domain_treedef: jax.tree_util.PyTreeDef,
    _leaf_shapes: tuple[tuple[int, ...], ...],
    _per_leaf_order: tuple[int, ...],
) -> TaylorPolynomial:
    """Wrap a plain array as a constant TaylorPolynomial."""
    return _tp_constant(
        arr, d, order, flat_center, _domain_treedef, _leaf_shapes, _per_leaf_order
    )


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def nattp(
    f: Callable[..., jax.Array],
    *,
    max_order: int | None = None,
) -> Callable[..., TaylorPolynomial]:
    """Creates a Natural Taylor Polynomial Function of *f*.

    Propagates ``TaylorPolynomial`` arguments through *f* by interpreting
    the traced Jaxpr.  High-order terms beyond ``max_order`` are silently
    discarded (no remainder is tracked).

    Parameters
    ----------
    f : Callable
        Function to transform.
    max_order : int, optional
        Maximum polynomial order.  If ``None``, inferred from inputs.
    """

    @wraps(f)
    def wrapped(*args, **kwargs) -> TaylorPolynomial:
        effective_order = max_order
        if effective_order is None:
            for arg in jax.tree_util.tree_leaves(args, is_leaf=istaylorpolynomial):
                if istaylorpolynomial(arg):
                    effective_order = (
                        arg.max_order
                        if effective_order is None
                        else max(effective_order, arg.max_order)
                    )

            if effective_order is None:
                effective_order = 2

        # Separate TP args from non-TP args
        tp_args = [a for a in jax.tree_util.tree_leaves(args, is_leaf=istaylorpolynomial) if istaylorpolynomial(a)]

        if not tp_args:
            # No TP args at all — just evaluate f directly
            return f(*args, **kwargs)

        # Concatenate all TP args into one
        if len(tp_args) == 1:
            tp_concat = tp_args[0]
        else:
            tp_concat = taylor_polynomial_concatenate(tp_args)
        output_pytree = tp_concat._output_pytree

        # Build f_tp that closes over non-TP args and kwargs,
        # receives the TP output leaves as positional args
        def f_tp(*tp_leaves):
            full_args = []
            leaf_idx = 0
            for i in range(len(args)):
                if istaylorpolynomial(args[i]):
                    n = args[i]._output_pytree.num_leaves
                    if n == 1:
                        full_args.append(tp_leaves[leaf_idx])
                        leaf_idx += 1
                    else:
                        leaves = tp_leaves[leaf_idx:leaf_idx + n]
                        reconstructed = args[i]._output_pytree.treedef.unflatten(leaves)
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
        closed_jaxpr = eqx.filter_make_jaxpr(f_tp)(*rep_args)[0]

        out = nattp_jaxpr(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            tp_concat,
            max_order=effective_order,
            output_pytree=output_pytree,
        )

        if len(out) == 1:
            return out[0]
        return out

    return wrapped


def nattp_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    max_order: int = 2,
    propagate_source_info: bool = True,
    output_pytree: "PyTreeShape | None" = None,
) -> list[Any]:
    """Interpreter for Jaxpr with TaylorPolynomial arguments."""

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)

    if output_pytree is not None:
        # Single TaylorPolynomial with structured output - slice for each invar
        tp = args[0]
        sliced_args = []
        for i in range(output_pytree.num_leaves):
            slc = output_pytree.leaf_slice(i)
            sliced_args.append(tp[slc])
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
            if any(istaylorpolynomial(read(iv)) for iv in eqn.invars):
                handler = _resolve_handler(eqn.primitive)
                if handler is None:
                    raise NotImplementedError(
                        f"{eqn.primitive} (name: {eqn.primitive.name}) "
                        f"not in tp_inclusion_registry"
                    )
                if eqn.primitive in _tp_needs_max_order:
                    ans = handler(*subfuns, *invars, max_order=max_order, **bind_params)
                else:
                    ans = handler(*subfuns, *invars, **bind_params)
            else:
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)

        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)

    return safe_map(read, jaxpr.outvars)


def _resolve_handler(primitive: Primitive):
    """Look up handler by object identity, then by name as fallback."""
    handler = tp_inclusion_registry.get(primitive)
    if handler is not None:
        return handler
    for prim, h in tp_inclusion_registry.items():
        if prim.name == primitive.name:
            tp_inclusion_registry[primitive] = h
            if prim in _tp_needs_max_order:
                _tp_needs_max_order.add(primitive)
            return h
    return None


# ---------------------------------------------------------------------------
# Passthrough / structural operations
# ---------------------------------------------------------------------------


def _make_tp_passthrough(primitive: Primitive):
    def _handler(*args, **kwargs):
        ref = None
        for a in jax.tree_util.tree_leaves(args, is_leaf=istaylorpolynomial):
            if istaylorpolynomial(a):
                ref = a
                break
        if ref is None:
            return primitive.bind(*args, **kwargs)

        args_const = [a.constant_term if istaylorpolynomial(a) else a for a in args]
        result_const = primitive.bind(*args_const, **kwargs)
        return _tp_from_array(
            result_const,
            ref.d,
            ref.max_order,
            ref.flat_center,
            ref._domain_treedef,
            ref._leaf_shapes,
            ref._per_leaf_order,
        )

    tp_inclusion_registry[primitive] = _handler


_make_tp_passthrough(lax.copy_p)
_make_tp_passthrough(lax.iota_p)
_make_tp_passthrough(lax.convert_element_type_p)
_make_tp_passthrough(debug_callback_p)


# --- Reshape ---


def _tp_reshape_p(x, *, new_sizes, dimensions=None):
    if not istaylorpolynomial(x):
        return lax.reshape_p.bind(x, new_sizes=new_sizes, dimensions=dimensions)
    m = x.num_monomials
    coeff_new_sizes = (*new_sizes, m)
    if dimensions is not None:
        coeff_dimensions = (*dimensions, len(x._output_shape))
        new_coeffs = lax.reshape(
            x.coeffs, new_sizes=coeff_new_sizes, dimensions=coeff_dimensions
        )
    else:
        new_coeffs = x.coeffs.reshape(*coeff_new_sizes)
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.reshape_p] = _tp_reshape_p


# --- Transpose ---


def _tp_transpose_p(x, *, permutation):
    if not istaylorpolynomial(x):
        return lax.transpose_p.bind(x, permutation=permutation)
    ndim_out = len(x._output_shape)
    coeff_permutation = (*permutation, ndim_out)
    new_coeffs = lax.transpose(x.coeffs, permutation=coeff_permutation)
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.transpose_p] = _tp_transpose_p


# --- Squeeze ---


def _tp_squeeze_p(x, *, dimensions):
    if not istaylorpolynomial(x):
        return lax.squeeze_p.bind(x, dimensions=dimensions)
    new_coeffs = lax.squeeze(x.coeffs, dimensions=dimensions)
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.squeeze_p] = _tp_squeeze_p


# --- Broadcast_in_dim ---


def _tp_broadcast_in_dim_p(x, *, shape, broadcast_dimensions, sharding=None):
    if not istaylorpolynomial(x):
        return lax.broadcast_in_dim_p.bind(
            x, shape=shape, broadcast_dimensions=broadcast_dimensions, sharding=sharding
        )
    m = x.num_monomials
    coeff_shape = (*shape, m)
    coeff_broadcast_dims = (*broadcast_dimensions, len(shape))
    new_coeffs = lax.broadcast_in_dim(
        x.coeffs, shape=coeff_shape, broadcast_dimensions=coeff_broadcast_dims
    )
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.broadcast_in_dim_p] = _tp_broadcast_in_dim_p


# --- Slice ---


def _tp_slice_p(x, *, start_indices, limit_indices, strides=None):
    if not istaylorpolynomial(x):
        return lax.slice_p.bind(
            x, start_indices=start_indices, limit_indices=limit_indices, strides=strides
        )
    m = x.num_monomials
    coeff_start = (*start_indices, 0)
    coeff_limit = (*limit_indices, m)
    coeff_strides = (*strides, 1) if strides else None
    new_coeffs = lax.slice_p.bind(
        x.coeffs,
        start_indices=coeff_start,
        limit_indices=coeff_limit,
        strides=coeff_strides,
    )
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.slice_p] = _tp_slice_p


# --- Dynamic slice ---


def _tp_dynamic_slice_p(x, *start_indices, slice_sizes):
    if not istaylorpolynomial(x):
        return lax.dynamic_slice_p.bind(x, *start_indices, slice_sizes=slice_sizes)
    # The original code had a bug here, assuming start_indices[0] and slice_sizes[0]
    # for all dimensions. This is incorrect for multi-dimensional slices.
    # The correct way is to extend start_indices and slice_sizes with the monomial dimension.
    m = x.num_monomials
    extended_start_indices = (*start_indices, 0)
    extended_slice_sizes = (*slice_sizes, m)

    new_coeffs = lax.dynamic_slice_p.bind(
        x.coeffs,
        *extended_start_indices,
        slice_sizes=extended_slice_sizes,
    )
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.dynamic_slice_p] = _tp_dynamic_slice_p


# --- Concatenation ---


def _tp_concatenate_p(*args, dimension):
    tps = [a for a in args if istaylorpolynomial(a)]
    if len(tps) == 0:
        return lax.concatenate_p.bind(*args, dimension=dimension)

    ref = tps[0]
    tp_args = []
    for a in args:
        if istaylorpolynomial(a):
            tp_args.append(a.to_canonical(ref._per_leaf_order))
        else:
            tp_args.append(
                _tp_from_array(
                    jnp.asarray(a),
                    ref.d,
                    ref.max_order,
                    ref.flat_center,
                    ref._domain_treedef,
                    ref._leaf_shapes,
                    ref._per_leaf_order,
                )
            )

    coeffs = jnp.concatenate([t.coeffs for t in tp_args], axis=dimension)
    return TaylorPolynomial(
        coeffs,
        tp_args[0].exponents,
        ref.flat_center,
        _input_pytree=ref._input_pytree,
        _output_pytree=PyTreeShape.flat(coeffs.shape[:-1]),
        _per_leaf_order=ref._per_leaf_order,
    )


tp_inclusion_registry[lax.concatenate_p] = _tp_concatenate_p


# --- pjit ---


def _tp_pjit_p(*args, **bind_params):
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    eff_order = None
    for a in args:
        if istaylorpolynomial(a):
            eff_order = (
                a.max_order if eff_order is None else max(eff_order, a.max_order)
            )

    if eff_order is None:
        eff_order = 2
    return nattp_jaxpr(bind_jaxpr, [], *args, max_order=eff_order)


tp_inclusion_registry[jax._src.pjit.pjit_p] = _tp_pjit_p


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def _tp_add_p(x, y):
    if istaylorpolynomial(x) and istaylorpolynomial(y):
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, y._output_shape)
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        y_coeffs = jnp.broadcast_to(y.coeffs, (*broadcast_shape, y.num_monomials))
        new_coeffs, new_exp = _merge_taylor_terms(
            x_coeffs, x.exponents, y_coeffs, y.exponents
        )

        # For addition, we can just use the per-leaf order from one of the operands
        # (assuming they are compatible/same structure)

        per_leaf_order = _max_order(x._per_leaf_order, y._per_leaf_order)

        result = TaylorPolynomial(
            new_coeffs,
            new_exp,
            x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=per_leaf_order,
        )
        return result.to_canonical(per_leaf_order)

    elif istaylorpolynomial(x):
        val = jnp.asarray(y)
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, val.shape)
        const_exp = jnp.zeros((x.d, 1), dtype=jnp.int32)
        val_broadcast = jnp.broadcast_to(val, broadcast_shape)
        const_coeff = val_broadcast[..., None]
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        new_coeffs, new_exp = _merge_taylor_terms(
            x_coeffs, x.exponents, const_coeff, const_exp
        )
        result = TaylorPolynomial(
            new_coeffs,
            new_exp,
            x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )
        return result.to_canonical(x._per_leaf_order)

    elif istaylorpolynomial(y):
        return _tp_add_p(y, x)
    else:
        return x + y


tp_inclusion_registry[lax.add_p] = _tp_add_p
tp_inclusion_registry[ad_util.add_any_p] = _tp_add_p
TaylorPolynomial.__add__ = _tp_add_p
TaylorPolynomial.__radd__ = _tp_add_p


def _tp_neg_p(x):
    if not istaylorpolynomial(x):
        return -x
    return TaylorPolynomial(
        -x.coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=x._output_pytree,
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.neg_p] = _tp_neg_p
TaylorPolynomial.__neg__ = _tp_neg_p


def _tp_sub_p(x, y):
    if istaylorpolynomial(y):
        return _tp_add_p(x, _tp_neg_p(y))
    return _tp_add_p(x, -jnp.asarray(y))


tp_inclusion_registry[lax.sub_p] = _tp_sub_p
TaylorPolynomial.__sub__ = _tp_sub_p
TaylorPolynomial.__rsub__ = lambda self, other: _tp_sub_p(other, self)


def _tp_mul_p(x, y, *, max_order: int = None):
    if istaylorpolynomial(x) and istaylorpolynomial(y):
        if x.d != y.d:
            raise ValueError(f"Domain dimensions must match: {x.d} vs {y.d}")

        broadcast_shape = jnp.broadcast_shapes(x._output_shape, y._output_shape)
        d = x.d
        m1 = x.num_monomials
        m2 = y.num_monomials

        exp1 = x.exponents[:, :, None]
        exp2 = y.exponents[:, None, :]
        all_exp = (exp1 + exp2).reshape(d, m1 * m2)

        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, m1))
        y_coeffs = jnp.broadcast_to(y.coeffs, (*broadcast_shape, m2))

        coeff1 = x_coeffs[..., :, None]
        coeff2 = y_coeffs[..., None, :]
        all_coeffs = (coeff1 * coeff2).reshape(*broadcast_shape, m1 * m2)

        per_leaf_order = _max_order(x._per_leaf_order, y._per_leaf_order)
        keep_mask = _check_per_leaf_bounds(all_exp, x._leaf_shapes, per_leaf_order)

        kept_coeffs = jnp.where(keep_mask, all_coeffs, 0.0)

        result = TaylorPolynomial(
            kept_coeffs,
            all_exp,
            x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=per_leaf_order,
        )
        return result.to_canonical(per_leaf_order)

    elif istaylorpolynomial(x):
        alpha = jnp.asarray(y)
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, alpha.shape)
        alpha_bc = jnp.broadcast_to(alpha, broadcast_shape)
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        new_coeffs = alpha_bc[..., None] * x_coeffs
        return TaylorPolynomial(
            new_coeffs,
            x.exponents,
            x.flat_center,
            _input_pytree=x._input_pytree,
            _output_pytree=x._output_pytree,
            _per_leaf_order=x._per_leaf_order,
        )

    elif istaylorpolynomial(y):
        return _tp_mul_p(y, x, max_order=max_order)
    else:
        return x * y


tp_inclusion_registry[lax.mul_p] = _tp_mul_p
_tp_needs_max_order.add(lax.mul_p)
TaylorPolynomial.__mul__ = _tp_mul_p
TaylorPolynomial.__rmul__ = _tp_mul_p


def _tp_integer_pow_p(x, y: int, *, max_order: int = None):
    if not istaylorpolynomial(x):
        return lax.integer_pow(x, y)

    order = max_order if max_order is not None else x.max_order

    # NOTE: If y=0, we need to return a constant TP.
    # We must use x's structure.

    if y == 0:
        return _tp_constant(
            jnp.ones(x._output_shape, dtype=x.dtype),
            x.d,
            x._per_leaf_order,  # Pass tuple order
            x.flat_center,
            x._domain_treedef,
            x._leaf_shapes,
            x._per_leaf_order,
        )
    if y < 0:
        raise ValueError(
            "Negative integer powers not supported for TaylorPolynomial (no remainder for reciprocal)"
        )
    result = x
    for _ in range(y - 1):
        result = _tp_mul_p(result, x, max_order=order)
    return result


tp_inclusion_registry[lax.integer_pow_p] = _tp_integer_pow_p
_tp_needs_max_order.add(lax.integer_pow_p)
TaylorPolynomial.__pow__ = _tp_integer_pow_p

if hasattr(lax, "square_p"):

    def _tp_square_p(x, *, max_order: int = None):
        return _tp_integer_pow_p(x, 2, max_order=max_order)

    tp_inclusion_registry[lax.square_p] = _tp_square_p
    _tp_needs_max_order.add(lax.square_p)


# ---------------------------------------------------------------------------
# Dot general
# ---------------------------------------------------------------------------


def _tp_dot_general_array(arr, tp, dim_nums):
    """Array @ TP: linear operation on coefficients."""
    new_coeffs = lax.dot_general(arr, tp.coeffs, dimension_numbers=dim_nums)
    return TaylorPolynomial(
        new_coeffs,
        tp.exponents,
        tp.flat_center,
        _input_pytree=tp._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=tp._per_leaf_order,
    )


def _tp_dot_general_p(A, B, *, max_order: int = None, **kwargs):
    dimension_numbers = kwargs["dimension_numbers"]
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

    if not istaylorpolynomial(A) and istaylorpolynomial(B):
        return _tp_dot_general_array(jnp.asarray(A), B, dimension_numbers)

    if istaylorpolynomial(A) and not istaylorpolynomial(B):
        B_arr = jnp.asarray(B)
        swapped_dims = ((rhs_contracting, lhs_contracting), (rhs_batch, lhs_batch))
        result = _tp_dot_general_array(B_arr, A, swapped_dims)

        nb = len(lhs_batch)
        n_rem_A = len(A._output_shape) - len(lhs_contracting) - len(lhs_batch)
        n_rem_B = B_arr.ndim - len(rhs_contracting) - len(rhs_batch)

        if n_rem_A > 0 and n_rem_B > 0:
            perm = (
                *range(nb),
                *range(nb + n_rem_B, nb + n_rem_B + n_rem_A),
                *range(nb, nb + n_rem_B),
                nb + n_rem_A + n_rem_B,
            )
            new_coeffs = jnp.transpose(result.coeffs, perm)
            return TaylorPolynomial(
                new_coeffs,
                result.exponents,
                result.flat_center,
                _input_pytree=result._input_pytree,
                _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
                _per_leaf_order=result._per_leaf_order,
            )
        return result

    if istaylorpolynomial(A) and istaylorpolynomial(B):
        if A.d != B.d:
            raise ValueError(f"Domain dimensions must match: {A.d} vs {B.d}")

        per_leaf_order = _max_order(A._per_leaf_order, B._per_leaf_order)

        d = A.d
        m1 = A.num_monomials
        m2 = B.num_monomials

        product_exp = (A.exponents[:, :, None] + B.exponents[:, None, :]).reshape(
            d, m1 * m2
        )

        contract_over_m2 = jax.vmap(
            lambda a, b: lax.dot_general(a, b, dimension_numbers=dimension_numbers),
            (None, -1),
            -1,
        )
        contract_over_m1m2 = jax.vmap(contract_over_m2, (-1, None), -1)
        result_coeffs_mm = contract_over_m1m2(A.coeffs, B.coeffs)
        result_coeffs = result_coeffs_mm.reshape(*result_coeffs_mm.shape[:-2], m1 * m2)

        keep_mask = _check_per_leaf_bounds(product_exp, A._leaf_shapes, per_leaf_order)

        kept_coeffs = jnp.where(keep_mask, result_coeffs, 0.0)

        result = TaylorPolynomial(
            kept_coeffs,
            product_exp,
            A.flat_center,
            _input_pytree=A._input_pytree,
            _output_pytree=PyTreeShape.flat(kept_coeffs.shape[:-1]),
            _per_leaf_order=per_leaf_order,
        )
        return result.to_canonical(per_leaf_order)

    return lax.dot_general_p.bind(A, B, **kwargs)


tp_inclusion_registry[lax.dot_general_p] = _tp_dot_general_p
_tp_needs_max_order.add(lax.dot_general_p)

TaylorPolynomial.__matmul__ = lambda self, other: nattp(jnp.matmul)(self, other)
TaylorPolynomial.__rmatmul__ = lambda self, other: nattp(jnp.matmul)(other, self)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _tp_reduce_sum_p(x, *, axes):
    if not istaylorpolynomial(x):
        return lax.reduce_sum_p.bind(x, axes=axes)
    new_coeffs = jnp.sum(x.coeffs, axis=axes)
    return TaylorPolynomial(
        new_coeffs,
        x.exponents,
        x.flat_center,
        _input_pytree=x._input_pytree,
        _output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        _per_leaf_order=x._per_leaf_order,
    )


tp_inclusion_registry[lax.reduce_sum_p] = _tp_reduce_sum_p


def _tp_reduce_max_p(x, *, axes):
    if not istaylorpolynomial(x):
        return lax.reduce_max_p.bind(x, axes=axes)
    # Max is not polynomial-preserving; fall back to constant from evaluation at center
    val = x.evaluate(x.flat_center)
    result = jnp.max(val, axis=axes)
    return _tp_constant(
        result,
        x.d,
        x._per_leaf_order,
        x.flat_center,
        x._domain_treedef,
        x._leaf_shapes,
        x._per_leaf_order,
    )


tp_inclusion_registry[lax.reduce_max_p] = _tp_reduce_max_p


def _tp_reduce_min_p(x, *, axes):
    if not istaylorpolynomial(x):
        return lax.reduce_min_p.bind(x, axes=axes)
    val = x.evaluate(x.flat_center)
    result = jnp.min(val, axis=axes)
    return _tp_constant(
        result,
        x.d,
        x._per_leaf_order,
        x.flat_center,
        x._domain_treedef,
        x._leaf_shapes,
        x._per_leaf_order,
    )


tp_inclusion_registry[lax.reduce_min_p] = _tp_reduce_min_p


# ---------------------------------------------------------------------------
# Univariate functions via jet (no remainder)
# ---------------------------------------------------------------------------


def _register_tp_univariate(primitive: Primitive, func: Callable) -> None:
    _tp_univariate_prim_to_func[primitive] = func

    def _handler(x, *, max_order: int = None, accuracy=None):
        return _tp_univariate(primitive, x, max_order=max_order)

    tp_inclusion_registry[primitive] = _handler
    _tp_needs_max_order.add(primitive)


def _tp_univariate(
    primitive_p: Primitive,
    x: TaylorPolynomial,
    *,
    max_order: int | None = None,
) -> TaylorPolynomial:
    """Generic univariate Taylor Polynomial composition via jet.

    Computes Taylor coefficients of ``g(p(x))`` up to ``max_order`` by
    using Horner's method:  result = a_k + z*(a_{k-1} + z*(...))
    where z = x - c and a_i = g^{(i)}(c) / i!.

    High-order terms are silently discarded.
    """
    if not istaylorpolynomial(x):
        return primitive_p.bind(x)

    per_var_order = max_order if max_order is not None else x.max_order
    # For univariate composition, use max of per-variable orders as expansion depth
    if isinstance(per_var_order, tuple):
        order = max(per_var_order)
    else:
        order = per_var_order
    output_shape = x._output_shape
    c = x.constant_term

    prim_func = _tp_univariate_prim_to_func.get(
        primitive_p, lambda v: primitive_p.bind(v)
    )

    import math

    n_flat = math.prod(output_shape) if output_shape else 1
    c_flat = c.reshape(-1) if output_shape else c[None]

    def get_coeffs(val):
        primals = (val,)
        series = ((1.0,) + (0.0,) * (order - 1),)
        f_val, f_series = jet(prim_func, primals, series)
        return jnp.array([f_val] + list(f_series))

    coeffs_raw = jax.vmap(get_coeffs)(c_flat)  # (n_flat, order+1)

    if output_shape:
        coeffs_raw_shaped = coeffs_raw.reshape(*output_shape, order + 1)
    else:
        coeffs_raw_shaped = coeffs_raw[0]

    z = x - c

    def horner_step(carry, coeff):
        term = _tp_constant(
            coeff,
            x.d,
            per_var_order,
            x.flat_center,
            x._domain_treedef,
            x._leaf_shapes,
            x._per_leaf_order,
        )
        return _tp_mul_p(carry, z, max_order=per_var_order) + term, None

    if output_shape:
        init_coeff = coeffs_raw_shaped[..., order]
        scan_coeffs = jnp.moveaxis(coeffs_raw_shaped[..., :order], -1, 0)[::-1]
    else:
        init_coeff = coeffs_raw_shaped[order]
        scan_coeffs = coeffs_raw_shaped[:order][::-1]

    init = _tp_constant(
        init_coeff,
        x.d,
        per_var_order,
        x.flat_center,
        x._domain_treedef,
        x._leaf_shapes,
        x._per_leaf_order,
    )
    result, _ = lax.scan(horner_step, init, scan_coeffs)

    return result


# Register standard univariate functions
_register_tp_univariate(lax.exp_p, jnp.exp)
_register_tp_univariate(lax.log_p, jnp.log)
_register_tp_univariate(lax.log1p_p, jnp.log1p)
_register_tp_univariate(lax.sin_p, jnp.sin)
_register_tp_univariate(lax.cos_p, jnp.cos)
_register_tp_univariate(lax.tan_p, jnp.tan)
_register_tp_univariate(lax.tanh_p, jnp.tanh)
_register_tp_univariate(lax.sqrt_p, jnp.sqrt)
_register_tp_univariate(lax.asin_p, jnp.arcsin)
_register_tp_univariate(lax.atan_p, jnp.arctan)


# --- abs (non-smooth, fall back to center evaluation) ---


def _tp_abs_p(x):
    if not istaylorpolynomial(x):
        return jnp.abs(x)
    # abs is not smooth at 0; just evaluate at center as a constant
    val = jnp.abs(x.evaluate(x.flat_center))
    return _tp_constant(
        val,
        x.d,
        x._per_leaf_order,
        x.flat_center,
        x._domain_treedef,
        x._leaf_shapes,
        x._per_leaf_order,
    )


tp_inclusion_registry[lax.abs_p] = _tp_abs_p


# --- div: x / y via reciprocal univariate + mul ---


def _tp_div_p(x, y, *, max_order: int = None):
    class _ReciprocalPrim:
        def bind(self, val):
            return 1.0 / val

    recip_y = _tp_univariate(_ReciprocalPrim(), y, max_order=max_order)
    if istaylorpolynomial(x):
        return _tp_mul_p(x, recip_y, max_order=max_order)
    else:
        return _tp_mul_p(jnp.asarray(x), recip_y, max_order=max_order)


tp_inclusion_registry[lax.div_p] = _tp_div_p
_tp_needs_max_order.add(lax.div_p)


# --- pow: x^y = exp(y * log(x)) ---


def _tp_pow_p(x, y, *, max_order=None):
    order = max_order
    if order is None:
        order = None
        if istaylorpolynomial(x):
            order = x.max_order
        if istaylorpolynomial(y):
            order = y.max_order if order is None else max(order, y.max_order)
        if order is None:
            order = 2

    log_x = _tp_univariate(lax.log_p, x, max_order=order)
    y_log_x = _tp_mul_p(y, log_x, max_order=order)
    return _tp_univariate(lax.exp_p, y_log_x, max_order=order)


tp_inclusion_registry[lax.pow_p] = _tp_pow_p
_tp_needs_max_order.add(lax.pow_p)


# --- max / min (non-polynomial, center evaluation) ---


def _tp_max_p(x, y):
    ref = x if istaylorpolynomial(x) else y
    x_val = x.evaluate(x.flat_center) if istaylorpolynomial(x) else jnp.asarray(x)
    y_val = y.evaluate(y.flat_center) if istaylorpolynomial(y) else jnp.asarray(y)
    return _tp_constant(
        jnp.maximum(x_val, y_val),
        ref.d,
        ref._per_leaf_order,
        ref.flat_center,
        ref._domain_treedef,
        ref._leaf_shapes,
        ref._per_leaf_order,
    )


tp_inclusion_registry[lax.max_p] = _tp_max_p


def _tp_min_p(x, y):
    ref = x if istaylorpolynomial(x) else y
    x_val = x.evaluate(x.flat_center) if istaylorpolynomial(x) else jnp.asarray(x)
    y_val = y.evaluate(y.flat_center) if istaylorpolynomial(y) else jnp.asarray(y)
    return _tp_constant(
        jnp.minimum(x_val, y_val),
        ref.d,
        ref._per_leaf_order,
        ref.flat_center,
        ref._domain_treedef,
        ref._leaf_shapes,
        ref._per_leaf_order,
    )


tp_inclusion_registry[lax.min_p] = _tp_min_p


# --- Reciprocal ---


def _tp_reciprocal_p(x, *, max_order: int = None):
    if not istaylorpolynomial(x):
        return 1.0 / x
    one = _tp_constant(
        jnp.ones(x._output_shape, dtype=x.dtype),
        x.d,
        max_order if max_order is not None else x._per_leaf_order,
        x.flat_center,
        x._domain_treedef,
        x._leaf_shapes,
        x._per_leaf_order,
    )
    return _tp_div_p(one, x, max_order=max_order)
