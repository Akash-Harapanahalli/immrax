"""Natural Taylor Polynomial Function — Jaxpr interpreter for TaylorPolynomial.

Analogous to pjetm for TaylorModel, but propagates TaylorPolynomials through
functions (composition) by silently discarding high-order terms instead of
tracking them in an interval remainder.
"""

from functools import wraps
from typing import Any, Callable

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

from immrax.taylor.taylor_polynomial import (
    TaylorPolynomial,
    taylor_polynomial_concatenate,
    _taylor_polynomial_constant_impl,
)
from immrax.taylor.base import (
    MultiIndexArray,
    PyTreeShape,
    check_leaf_bounds,
    _merge_taylor_terms,
    max_leaf_order,
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
# Interpreter
# ---------------------------------------------------------------------------


def pjet(
    f: Callable[..., jax.Array],
    *,
    max_order: "int | tuple[int, ...] | None" = None,
) -> Callable[..., TaylorPolynomial]:
    """Creates a Natural Taylor Polynomial Function of *f*.

    Propagates ``TaylorPolynomial`` arguments through *f* by interpreting
    the traced Jaxpr.  High-order terms beyond ``max_order`` are silently
    discarded (no remainder is tracked).

    Parameters
    ----------
    f : Callable
        Function to transform.
    max_order : int, tuple[int, ...], or None
        Maximum polynomial order.  If an ``int``, it is broadcast to
        every domain leaf.  If a ``tuple``, it must match the per-leaf
        structure of the input TaylorPolynomials' domain.
        If ``None``, inferred from inputs.
    """

    @wraps(f)
    def wrapped(*args, **kwargs) -> TaylorPolynomial:
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
        output_pytree = tp_concat.output_pytree

        # Determine effective_order as tuple[int, ...], matching domain leaves
        if max_order is not None:
            if isinstance(max_order, int):
                effective_order = tuple(
                    max_order for _ in tp_concat.leaf_order
                )
            else:
                effective_order = tuple(max_order)
        else:
            effective_order = tp_concat.leaf_order

        # Build f_tp that closes over non-TP args and kwargs,
        # receives the TP output leaves as positional args
        def f_tp(*tp_leaves):
            full_args = []
            leaf_idx = 0
            for i in range(len(args)):
                if istaylorpolynomial(args[i]):
                    n = args[i].output_pytree.num_leaves
                    if n == 1:
                        full_args.append(tp_leaves[leaf_idx])
                        leaf_idx += 1
                    else:
                        leaves = tp_leaves[leaf_idx:leaf_idx + n]
                        reconstructed = args[i].output_pytree.treedef.unflatten(leaves)
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

        out = pjet_jaxpr(
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


def pjet_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    max_order: "int | tuple[int, ...]" = 2,
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
    """Creates a TP handler that applies the primitive to coefficients directly.

    For operations like copy, convert_element_type, etc., we just apply
    the primitive independently to coeffs using tree_map.
    """

    def _handler(*args, **kwargs):
        ref = None
        for a in jax.tree_util.tree_leaves(args, is_leaf=istaylorpolynomial):
            if istaylorpolynomial(a):
                ref = a
                break
        if ref is None:
            return primitive.bind(*args, **kwargs)

        getcoeffs = lambda x: x.coeffs if istaylorpolynomial(x) else x
        args_coeffs = jax.tree_util.tree_map(getcoeffs, args, is_leaf=istaylorpolynomial)
        new_coeffs = primitive.bind(*args_coeffs, **kwargs)

        return TaylorPolynomial(
            new_coeffs,
            ref.multiindices,
            ref.flat_center,
            input_pytree=ref.input_pytree,
            output_pytree=ref.output_pytree,
            leaf_order=ref.leaf_order,
        )

    tp_inclusion_registry[primitive] = _handler


_make_tp_passthrough(lax.copy_p)
_make_tp_passthrough(lax.iota_p)
_make_tp_passthrough(lax.convert_element_type_p)
_make_tp_passthrough(debug_callback_p)


# --- Structural operations (factory pattern, matching pjetm) ---


def _make_tp_structural_p(primitive, adapt_coeff_kwargs):
    """Factory for structural TP operations that preserve polynomial structure.

    These operations act on the output shape of a TaylorPolynomial. The monomial
    axis (last axis of coeffs) is left unchanged. ``adapt_coeff_kwargs``
    takes ``(kwargs, tp)`` and returns modified kwargs for the coeffs array
    (which has the extra trailing monomial axis).
    """

    def _tp_p(*args, **kwargs):
        # Find first TP arg
        ref_tp = None
        for arg in args:
            if istaylorpolynomial(arg):
                ref_tp = arg
                break
        if ref_tp is None:
            return primitive.bind(*args, **kwargs)

        # Lift non-TP args to coefficient shape: place scalar value in
        # the constant-monomial column (index 0), zeros elsewhere.
        def _lift(arg):
            if istaylorpolynomial(arg):
                return arg.coeffs
            if jnp.ndim(arg) < ref_tp.coeffs.ndim:
                m = ref_tp.num_monomials
                return jnp.concatenate(
                    [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
                )
            return arg

        # Apply primitive to coeffs with adapted kwargs
        coeff_kwargs = adapt_coeff_kwargs(kwargs, ref_tp)
        args_coeffs = [_lift(arg) for arg in args]
        new_coeffs = primitive.bind(*args_coeffs, **coeff_kwargs)

        return TaylorPolynomial(
            new_coeffs,
            ref_tp.multiindices,
            ref_tp.flat_center,
            input_pytree=ref_tp.input_pytree,
            output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
            leaf_order=ref_tp.leaf_order,
        )

    tp_inclusion_registry[primitive] = _tp_p


def _adapt_reshape(kwargs, tp):
    m = tp.num_monomials
    kw = {**kwargs, "new_sizes": (*kwargs["new_sizes"], m)}
    if kwargs.get("dimensions") is not None:
        kw["dimensions"] = (*kwargs["dimensions"], len(tp._output_shape))
    return kw


def _adapt_transpose(kwargs, tp):
    return {**kwargs, "permutation": (*kwargs["permutation"], len(tp._output_shape))}


def _adapt_squeeze(kwargs, tp):
    return kwargs  # monomial axis is never size-1


def _adapt_broadcast_in_dim(kwargs, tp):
    shape = kwargs["shape"]
    return {
        **kwargs,
        "shape": (*shape, tp.num_monomials),
        "broadcast_dimensions": (*kwargs["broadcast_dimensions"], len(shape)),
    }


def _adapt_slice(kwargs, tp):
    m = tp.num_monomials
    kw = {
        **kwargs,
        "start_indices": (*kwargs["start_indices"], 0),
        "limit_indices": (*kwargs["limit_indices"], m),
    }
    if kwargs.get("strides") is not None:
        kw["strides"] = (*kwargs["strides"], 1)
    return kw


def _adapt_concatenate(kwargs, tp):
    return kwargs  # dimension refers to output axes; monomial axis is last and shared


_make_tp_structural_p(lax.reshape_p, _adapt_reshape)
_make_tp_structural_p(lax.transpose_p, _adapt_transpose)
_make_tp_structural_p(lax.squeeze_p, _adapt_squeeze)
_make_tp_structural_p(lax.broadcast_in_dim_p, _adapt_broadcast_in_dim)
_make_tp_structural_p(lax.slice_p, _adapt_slice)
_make_tp_structural_p(lax.concatenate_p, _adapt_concatenate)


# --- Dynamic slice (positional args, can't use structural factory) ---


def _tp_dynamic_slice_p(x, *start_indices, slice_sizes):
    if not istaylorpolynomial(x):
        return lax.dynamic_slice_p.bind(x, *start_indices, slice_sizes=slice_sizes)
    m = x.num_monomials
    new_coeffs = lax.dynamic_slice_p.bind(
        x.coeffs, *start_indices, 0, slice_sizes=(*slice_sizes, m)
    )
    return TaylorPolynomial(
        new_coeffs,
        x.multiindices,
        x.flat_center,
        input_pytree=x.input_pytree,
        output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        leaf_order=x.leaf_order,
    )


tp_inclusion_registry[lax.dynamic_slice_p] = _tp_dynamic_slice_p


# --- Scatter operations ---


def _make_tp_scatter_p(primitive):
    """Factory for linear scatter TP operations (scatter, scatter_add).

    Scatter has positional args (operand, scatter_indices, updates) where
    scatter_indices are integer indices. The monomial axis is added to
    update_window_dims so the full monomial slice is scattered as a unit.
    """

    def _tp_p(operand, scatter_indices, updates, *, dimension_numbers, **kwargs):
        op_is_tp = istaylorpolynomial(operand)
        upd_is_tp = istaylorpolynomial(updates)
        if not op_is_tp and not upd_is_tp:
            return primitive.bind(operand, scatter_indices, updates,
                                 dimension_numbers=dimension_numbers, **kwargs)

        ref_tp = operand if op_is_tp else updates
        m = ref_tp.num_monomials

        def _lift(arg):
            if istaylorpolynomial(arg):
                return arg.coeffs
            return jnp.concatenate(
                [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
            )

        op_coeffs = _lift(operand)
        upd_coeffs = _lift(updates)

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

        return TaylorPolynomial(
            new_coeffs,
            ref_tp.multiindices,
            ref_tp.flat_center,
            input_pytree=ref_tp.input_pytree,
            output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
            leaf_order=ref_tp.leaf_order,
        )

    tp_inclusion_registry[primitive] = _tp_p


_make_tp_scatter_p(lax.scatter_p)
_make_tp_scatter_p(lax.scatter_add_p)


def _make_tp_scatter_nonlinear_p(primitive):
    """Factory for non-linear scatter ops (scatter_max, scatter_min).

    Falls back to center evaluation since max/min on coefficients is not meaningful.
    """

    def _tp_p(operand, scatter_indices, updates, *, dimension_numbers, **kwargs):
        op_is_tp = istaylorpolynomial(operand)
        upd_is_tp = istaylorpolynomial(updates)
        if not op_is_tp and not upd_is_tp:
            return primitive.bind(operand, scatter_indices, updates,
                                 dimension_numbers=dimension_numbers, **kwargs)

        ref_tp = operand if op_is_tp else updates
        op_val = operand.evaluate(operand.flat_center) if op_is_tp else jnp.asarray(operand)
        upd_val = updates.evaluate(updates.flat_center) if upd_is_tp else jnp.asarray(updates)

        result = primitive.bind(op_val, scatter_indices, upd_val,
                                dimension_numbers=dimension_numbers, **kwargs)

        return _taylor_polynomial_constant_impl(
            result,
            ref_tp.flat_center,
            ref_tp.input_pytree,
            ref_tp.leaf_order,
        )

    tp_inclusion_registry[primitive] = _tp_p


_make_tp_scatter_nonlinear_p(lax.scatter_max_p)
_make_tp_scatter_nonlinear_p(lax.scatter_min_p)


# --- select_n ---


def _tp_select_n_p(pred, *cases):
    """Taylor polynomial select_n: pred chooses among cases element-wise.

    pred is integer/boolean (never a TP). Each case may be a TP.
    The predicate is broadcast along the monomial axis for coefficients.
    """
    if not any(istaylorpolynomial(c) for c in cases):
        return lax.select_n_p.bind(pred, *cases)

    ref_tp = next(c for c in cases if istaylorpolynomial(c))
    m = ref_tp.num_monomials

    def _lift(arg):
        if istaylorpolynomial(arg):
            return arg.coeffs
        return jnp.concatenate(
            [arg[..., None], jnp.zeros((*jnp.shape(arg), m - 1))], axis=-1
        )

    pred_coeffs = jnp.broadcast_to(pred[..., None], (*pred.shape, m))
    cases_coeffs = [_lift(c) for c in cases]
    new_coeffs = lax.select_n_p.bind(pred_coeffs, *cases_coeffs)

    return TaylorPolynomial(
        new_coeffs,
        ref_tp.multiindices,
        ref_tp.flat_center,
        input_pytree=ref_tp.input_pytree,
        output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        leaf_order=ref_tp.leaf_order,
    )


if hasattr(lax, "select_n_p"):
    tp_inclusion_registry[lax.select_n_p] = _tp_select_n_p


# --- split ---


def _tp_split_p(x, *, sizes, axis):
    """Taylor polynomial split: split along an output axis, preserving polynomials.

    split_p is a multiple_results primitive — returns a list of TaylorPolynomials.
    """
    if not istaylorpolynomial(x):
        return lax.split_p.bind(x, sizes=sizes, axis=axis)

    coeff_chunks = lax.split_p.bind(x.coeffs, sizes=sizes, axis=axis)

    return [
        TaylorPolynomial(
            c, x.multiindices, x.flat_center,
            input_pytree=x.input_pytree,
            output_pytree=PyTreeShape.flat(c.shape[:-1]),
            leaf_order=x.leaf_order,
        )
        for c in coeff_chunks
    ]


if hasattr(lax, "split_p"):
    tp_inclusion_registry[lax.split_p] = _tp_split_p


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
    return pjet_jaxpr(bind_jaxpr, [], *args, max_order=eff_order)


# jax >= 0.9 renamed pjit_p to jit_p
_jit_primitive = getattr(jax._src.pjit, "jit_p", getattr(jax._src.pjit, "pjit_p", None))
if _jit_primitive is not None:
    tp_inclusion_registry[_jit_primitive] = _tp_pjit_p


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
        new_coeffs, new_mia = _merge_taylor_terms(
            x_coeffs, x.multiindices, y_coeffs, y.multiindices
        )

        # For addition, we can just use the per-leaf order from one of the operands
        # (assuming they are compatible/same structure)

        leaf_order = max_leaf_order(x.leaf_order, y.leaf_order)

        result = TaylorPolynomial(
            new_coeffs,
            new_mia,
            x.flat_center,
            input_pytree=x.input_pytree,
            output_pytree=PyTreeShape.flat(broadcast_shape),
            leaf_order=leaf_order,
        )
        return result.to_canonical(leaf_order)

    elif istaylorpolynomial(x):
        val = jnp.asarray(y)
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, val.shape)
        const_mia = MultiIndexArray(jnp.zeros((x.d, 1), dtype=jnp.int32))
        val_broadcast = jnp.broadcast_to(val, broadcast_shape)
        const_coeff = val_broadcast[..., None]
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        new_coeffs, new_mia = _merge_taylor_terms(
            x_coeffs, x.multiindices, const_coeff, const_mia
        )
        result = TaylorPolynomial(
            new_coeffs,
            new_mia,
            x.flat_center,
            input_pytree=x.input_pytree,
            output_pytree=PyTreeShape.flat(broadcast_shape),
            leaf_order=x.leaf_order,
        )
        return result.to_canonical(x.leaf_order)

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
        x.multiindices,
        x.flat_center,
        input_pytree=x.input_pytree,
        output_pytree=x.output_pytree,
        leaf_order=x.leaf_order,
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

        exp1 = x.multiindices.to_jnp()[:, :, None]
        exp2 = y.multiindices.to_jnp()[:, None, :]
        all_exp = (exp1 + exp2).reshape(d, m1 * m2)
        all_mia = MultiIndexArray(all_exp)

        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, m1))
        y_coeffs = jnp.broadcast_to(y.coeffs, (*broadcast_shape, m2))

        coeff1 = x_coeffs[..., :, None]
        coeff2 = y_coeffs[..., None, :]
        all_coeffs = (coeff1 * coeff2).reshape(*broadcast_shape, m1 * m2)

        leaf_order = max_leaf_order(x.leaf_order, y.leaf_order)
        keep_mask = check_leaf_bounds(all_mia, x._leaf_shapes, leaf_order)

        kept_coeffs = jnp.where(keep_mask, all_coeffs, 0.0)

        result = TaylorPolynomial(
            kept_coeffs,
            all_mia,
            x.flat_center,
            input_pytree=x.input_pytree,
            output_pytree=PyTreeShape.flat(broadcast_shape),
            leaf_order=leaf_order,
        )
        return result.to_canonical(leaf_order)

    elif istaylorpolynomial(x):
        alpha = jnp.asarray(y)
        broadcast_shape = jnp.broadcast_shapes(x._output_shape, alpha.shape)
        alpha_bc = jnp.broadcast_to(alpha, broadcast_shape)
        x_coeffs = jnp.broadcast_to(x.coeffs, (*broadcast_shape, x.num_monomials))
        new_coeffs = alpha_bc[..., None] * x_coeffs
        return TaylorPolynomial(
            new_coeffs,
            x.multiindices,
            x.flat_center,
            input_pytree=x.input_pytree,
            output_pytree=PyTreeShape.flat(broadcast_shape),
            leaf_order=x.leaf_order,
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
        return _taylor_polynomial_constant_impl(
            jnp.ones(x._output_shape, dtype=x.dtype),
            x.flat_center,
            x.input_pytree,
            x.leaf_order,
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
        tp.multiindices,
        tp.flat_center,
        input_pytree=tp.input_pytree,
        output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        leaf_order=tp.leaf_order,
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
                result.multiindices,
                result.flat_center,
                input_pytree=result.input_pytree,
                output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
                leaf_order=result.leaf_order,
            )
        return result

    if istaylorpolynomial(A) and istaylorpolynomial(B):
        if A.d != B.d:
            raise ValueError(f"Domain dimensions must match: {A.d} vs {B.d}")

        leaf_order = max_leaf_order(A.leaf_order, B.leaf_order)

        d = A.d
        m1 = A.num_monomials
        m2 = B.num_monomials

        product_exp = (
            A.multiindices.to_jnp()[:, :, None] + B.multiindices.to_jnp()[:, None, :]
        ).reshape(d, m1 * m2)
        product_mia = MultiIndexArray(product_exp)

        contract_over_m2 = jax.vmap(
            lambda a, b: lax.dot_general(a, b, dimension_numbers=dimension_numbers),
            (None, -1),
            -1,
        )
        contract_over_m1m2 = jax.vmap(contract_over_m2, (-1, None), -1)
        result_coeffs_mm = contract_over_m1m2(A.coeffs, B.coeffs)
        result_coeffs = result_coeffs_mm.reshape(*result_coeffs_mm.shape[:-2], m1 * m2)

        keep_mask = check_leaf_bounds(product_mia, A._leaf_shapes, leaf_order)

        kept_coeffs = jnp.where(keep_mask, result_coeffs, 0.0)

        result = TaylorPolynomial(
            kept_coeffs,
            product_mia,
            A.flat_center,
            input_pytree=A.input_pytree,
            output_pytree=PyTreeShape.flat(kept_coeffs.shape[:-1]),
            leaf_order=leaf_order,
        )
        return result.to_canonical(leaf_order)

    return lax.dot_general_p.bind(A, B, **kwargs)


tp_inclusion_registry[lax.dot_general_p] = _tp_dot_general_p
_tp_needs_max_order.add(lax.dot_general_p)

TaylorPolynomial.__matmul__ = lambda self, other: pjet(jnp.matmul)(self, other)
TaylorPolynomial.__rmatmul__ = lambda self, other: pjet(jnp.matmul)(other, self)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _tp_reduce_sum_p(x, *, axes, **kwargs):
    if not istaylorpolynomial(x):
        return lax.reduce_sum_p.bind(x, axes=axes, **kwargs)
    new_coeffs = jnp.sum(x.coeffs, axis=axes)
    return TaylorPolynomial(
        new_coeffs,
        x.multiindices,
        x.flat_center,
        input_pytree=x.input_pytree,
        output_pytree=PyTreeShape.flat(new_coeffs.shape[:-1]),
        leaf_order=x.leaf_order,
    )


tp_inclusion_registry[lax.reduce_sum_p] = _tp_reduce_sum_p


def _tp_reduce_max_p(x, *, axes, **kwargs):
    if not istaylorpolynomial(x):
        return lax.reduce_max_p.bind(x, axes=axes, **kwargs)
    # Max is not polynomial-preserving; fall back to constant from evaluation at center
    val = x.evaluate(x.flat_center)
    result = jnp.max(val, axis=axes)
    return _taylor_polynomial_constant_impl(
        result,
        x.flat_center,
        x.input_pytree,
        x.leaf_order,
    )


tp_inclusion_registry[lax.reduce_max_p] = _tp_reduce_max_p


def _tp_reduce_min_p(x, *, axes, **kwargs):
    if not istaylorpolynomial(x):
        return lax.reduce_min_p.bind(x, axes=axes, **kwargs)
    val = x.evaluate(x.flat_center)
    result = jnp.min(val, axis=axes)
    return _taylor_polynomial_constant_impl(
        result,
        x.flat_center,
        x.input_pytree,
        x.leaf_order,
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
        # jet returns raw derivatives [f'(c), f''(c), ...]; divide by k! to get
        # Taylor coefficients a_k = f^(k)(c) / k! stored in coeffs convention.
        from immrax.utils import inv_fact
        raw = jnp.array([f_val] + list(f_series))
        scales = jnp.concatenate([jnp.ones(1), inv_fact(jnp.arange(1, order + 1))])
        return raw * scales

    coeffs_raw = jax.vmap(get_coeffs)(c_flat)  # (n_flat, order+1)

    if output_shape:
        coeffs_raw_shaped = coeffs_raw.reshape(*output_shape, order + 1)
    else:
        coeffs_raw_shaped = coeffs_raw[0]

    z = x - c

    def horner_step(carry, coeff):
        term = _taylor_polynomial_constant_impl(
            coeff,
            x.flat_center,
            x.input_pytree,
            x.leaf_order,
        )
        return _tp_mul_p(carry, z, max_order=per_var_order) + term, None

    if output_shape:
        init_coeff = coeffs_raw_shaped[..., order]
        scan_coeffs = jnp.moveaxis(coeffs_raw_shaped[..., :order], -1, 0)[::-1]
    else:
        init_coeff = coeffs_raw_shaped[order]
        scan_coeffs = coeffs_raw_shaped[:order][::-1]

    init = _taylor_polynomial_constant_impl(
        init_coeff,
        x.flat_center,
        x.input_pytree,
        x.leaf_order,
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
    return _taylor_polynomial_constant_impl(
        val,
        x.flat_center,
        x.input_pytree,
        x.leaf_order,
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
    return _taylor_polynomial_constant_impl(
        jnp.maximum(x_val, y_val),
        ref.flat_center,
        ref.input_pytree,
        ref.leaf_order,
    )


tp_inclusion_registry[lax.max_p] = _tp_max_p


def _tp_min_p(x, y):
    ref = x if istaylorpolynomial(x) else y
    x_val = x.evaluate(x.flat_center) if istaylorpolynomial(x) else jnp.asarray(x)
    y_val = y.evaluate(y.flat_center) if istaylorpolynomial(y) else jnp.asarray(y)
    return _taylor_polynomial_constant_impl(
        jnp.minimum(x_val, y_val),
        ref.flat_center,
        ref.input_pytree,
        ref.leaf_order,
    )


tp_inclusion_registry[lax.min_p] = _tp_min_p


# --- Reciprocal ---


def _tp_reciprocal_p(x, *, max_order: int = None):
    if not istaylorpolynomial(x):
        return 1.0 / x
    one = _taylor_polynomial_constant_impl(
        jnp.ones(x._output_shape, dtype=x.dtype),
        x.flat_center,
        x.input_pytree,
        x.leaf_order,
    )
    return _tp_div_p(one, x, max_order=max_order)
