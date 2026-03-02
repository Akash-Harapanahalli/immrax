from functools import wraps
from typing import Any, Callable

import equinox as eqx
import jax
from jax._src.debugging import debug_callback_p
import jax.numpy as jnp
from jax import jit, lax, vmap
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
from jax._src.lax import linalg as LA

# TODO: import only necessary things
from immrax.inclusion.interval import (
    Interval,
    interval,
    isinterval,
    widen,
    _get_rigorous,
    _set_rigorous,
)
from functools import partial

"""
This file implements the Natural Inclusion Function as an interpreter of Jaxprs.
"""

inclusion_registry = {}


def natif(
    f: Callable[..., jax.Array],
    rigorous: bool = True,
) -> Callable[..., Interval]:
    """Creates a Natural Inclusion Function of f.

    Non-Interval positional arguments are automatically closed over
    (treated as constants during tracing).  All keyword arguments are
    also closed over.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Natural Inclusion Function from.
    rigorous : bool
        When ``True`` (default), every inclusion-registry handler whose
        ``.n_ulps`` attribute is > 0 will have its output widened by
        that many ULPs, ensuring that computed interval bounds
        rigorously enclose the true mathematical result despite
        floating-point rounding.

    Returns
    -------
    Callable[..., Interval]
        Natural Inclusion Function of f

    Examples
    --------
    All arguments as intervals::

        natif(f)(iv_x, iv_y)

    First argument fixed (e.g. a matrix), second is an interval::

        natif(f)(M, iv_x)

    Mixed arguments::

        natif(f)(M, iv_x, dims)
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        # Separate interval args from non-interval (fixed) args
        interval_args = [arg for arg in args if isinterval(arg)]

        if not interval_args:
            return f(*args, **kwargs)

        # Build a closure that receives only interval args and
        # reconstructs the full argument list
        def f_interval(*iv_args):
            full_args = []
            iv_idx = 0
            for arg in args:
                if isinterval(arg):
                    full_args.append(iv_args[iv_idx])
                    iv_idx += 1
                else:
                    full_args.append(arg)
            return f(*full_args, **kwargs)

        # Representative values for tracing (lower bounds of intervals)
        getlower = lambda x: x.lower if isinterval(x) else jnp.asarray(x)
        build_iv_args = jax.tree_util.tree_map(
            getlower, interval_args, is_leaf=isinterval
        )

        # Build jaxpr from the closure — fixed args and kwargs become constants
        closed_jaxpr = eqx.filter_make_jaxpr(f_interval)(*build_iv_args)[0]

        # Evaluate the jaxpr with interval arguments
        out = natif_jaxpr(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            *interval_args,
            rigorous=rigorous,
        )
        if len(out) == 1:
            return out[0]
        return out

    return wrapped


def natif_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    rigorous: bool = True,
    propagate_source_info: bool = True,
) -> list[Any]:
    old_rigorous = _set_rigorous(rigorous)

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    try:
        env: dict[Var, Any] = {}
        safe_map(write, jaxpr.constvars, consts)
        safe_map(write, jaxpr.invars, args)
        lu = last_used(jaxpr)
        for eqn in jaxpr.eqns:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            name_stack = (
                source_info_util.current_name_stack() + eqn.source_info.name_stack
            )
            traceback = eqn.source_info.traceback if propagate_source_info else None
            with source_info_util.user_context(traceback, name_stack=name_stack):
                invars = safe_map(read, eqn.invars)
                if any([isinstance(read(iv), Interval) for iv in eqn.invars]):
                    try:
                        handler = inclusion_registry[eqn.primitive]
                        ans = handler(*subfuns, *invars, **bind_params)
                    except KeyError:
                        raise NotImplementedError(
                            f"{eqn.primitive} not in inclusion_registry"
                        )
                    # Rigorous widening
                    if _get_rigorous() and isinstance(ans, Interval):
                        n = getattr(handler, "n_ulps", 0)
                        if n > 0:
                            ans = widen(ans, n)
                else:
                    ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
            clean_up_dead_vars(eqn, env, lu)
        return safe_map(read, jaxpr.outvars)
    finally:
        _set_rigorous(old_rigorous)


def _make_inclusion_passthrough_p(
    primitive: Primitive, n_ulps: int = 0
) -> Callable[..., Interval]:
    """Creates an inclusion function that applies to the lower and upper bounds individually."""

    def _inclusion_p(*args, **kwargs) -> Interval:
        # Traverse args (possibly pytree) to get lower and upper bounds
        isinterval = lambda x: isinstance(x, Interval)
        getlower = lambda x: x.lower if isinstance(x, Interval) else x
        getupper = lambda x: x.upper if isinstance(x, Interval) else x
        args_l = jax.tree_util.tree_map(getlower, args, is_leaf=isinterval)
        args_u = jax.tree_util.tree_map(getupper, args, is_leaf=isinterval)
        return Interval(
            primitive.bind(*args_l, **kwargs), primitive.bind(*args_u, **kwargs)
        )

    _inclusion_p.n_ulps = n_ulps
    return _inclusion_p


def _add_passthrough_to_registry(primitive: Primitive, n_ulps: int = 0) -> None:
    """Helper to add a passthrough primitive to the inclusion registry."""
    inclusion_registry[primitive] = _make_inclusion_passthrough_p(primitive, n_ulps)


# We would like to passthrough array operations like reshaping, slicing, etc.
_add_passthrough_to_registry(lax.copy_p)
_add_passthrough_to_registry(lax.reshape_p)
_add_passthrough_to_registry(lax.slice_p)
if hasattr(lax, "split_p"):
    _add_passthrough_to_registry(lax.split_p)
_add_passthrough_to_registry(lax.dynamic_slice_p)
_add_passthrough_to_registry(lax.squeeze_p)
_add_passthrough_to_registry(lax.transpose_p)
_add_passthrough_to_registry(lax.broadcast_in_dim_p)
_add_passthrough_to_registry(lax.concatenate_p)
_add_passthrough_to_registry(lax.gather_p)
_add_passthrough_to_registry(lax.scatter_p)
_add_passthrough_to_registry(lax.scatter_add_p, n_ulps=1)
_add_passthrough_to_registry(lax.scatter_max_p)
_add_passthrough_to_registry(lax.scatter_min_p)
if hasattr(lax, "select_p"):
    _add_passthrough_to_registry(lax.select_p)
if hasattr(lax, "select_n_p"):
    _add_passthrough_to_registry(lax.select_n_p)
_add_passthrough_to_registry(lax.iota_p)
_add_passthrough_to_registry(lax.eq_p)
_add_passthrough_to_registry(lax.convert_element_type_p)
_add_passthrough_to_registry(lax.reduce_max_p)
_add_passthrough_to_registry(lax.reduce_min_p)
_add_passthrough_to_registry(lax.max_p)
_add_passthrough_to_registry(lax.min_p)
_add_passthrough_to_registry(lax.exp_p, n_ulps=1)
_add_passthrough_to_registry(lax.rev_p)


def _inclusion_reduce_sum_p(x: Interval, **kwargs) -> Interval:
    """Interval reduce_sum with rigorous widening scaled by reduction size."""
    axes = kwargs["axes"]
    lo = lax.reduce_sum_p.bind(x.lower, **kwargs)
    hi = lax.reduce_sum_p.bind(x.upper, **kwargs)
    result = Interval(lo, hi)
    if _get_rigorous():
        n = 1
        for ax in axes:
            n *= x.lower.shape[ax]
        if n > 1:
            result = widen(result, n - 1)
    return result


_inclusion_reduce_sum_p.n_ulps = 0
inclusion_registry[lax.reduce_sum_p] = _inclusion_reduce_sum_p


def _inclusion_reduce_prod_p(x: Interval, *, axes) -> Interval:
    """Interval reduce_prod: multiply intervals along specified axes.

    Uses sequential interval multiplication (handles signs correctly).
    """
    for axis in sorted(axes, reverse=True):
        n = x.lower.shape[axis]
        # Start with the first slice
        result = interval(
            jnp.take(x.lower, 0, axis=axis),
            jnp.take(x.upper, 0, axis=axis),
        )
        # Multiply remaining slices using interval arithmetic
        for i in range(1, n):
            next_iv = interval(
                jnp.take(x.lower, i, axis=axis),
                jnp.take(x.upper, i, axis=axis),
            )
            result = result * next_iv
        x = result
    return x


_inclusion_reduce_prod_p.n_ulps = 2
inclusion_registry[lax.reduce_prod_p] = _inclusion_reduce_prod_p
_add_passthrough_to_registry(lax.pad_p)
_add_passthrough_to_registry(lax.ne_p)
_add_passthrough_to_registry(lax.lt_p)
if hasattr(lax, "lt_to_p"):
    _add_passthrough_to_registry(lax.lt_to_p)
_add_passthrough_to_registry(debug_callback_p)
if hasattr(lax, "nextafter_p"):
    _add_passthrough_to_registry(lax.nextafter_p)

"""
TODO: Handle higher order primitives

natif_jaxpr should be thought of as an interpreter.
    - evaluates a jaxpr with interval arguments
    - uses the inclusion functions from the inclusion_registry
it cannot currently handle higher order primitives like scan, pjit
    - These HO primitives trace a jaxpr as their evaluation.
    - We can use natif_jaxpr to handle the jaxpr subexpression.
    - The inputs and outputs to the HO primitive itself are not correctly being handled.
Option 1:
    - make 'inclusion functions' which lowers intervals to pytrees and call the sub jaxpr with the proper conversions.
    - downside: this would be needed for each HO primitive.
Option 2:
    - handle them in a more principled manner, maybe during the tracing step
    - when we trace, we can also extract from the pytree which nodes will be intervals
    - somehow, if we see a HO primitive, perhaps changing the inputs will suffice
    - downside: requires more work to understand how to do this.

In principle, the only problem is the inputs to the HO primitive are not jax types
Natively, we cannot pass in pytrees like we are trying here.
"""

# Some higher order primitives


def _inclusion_pjit_p(*args, **bind_params) -> Interval:
    """Handles jit_p (jax >= 0.9) / pjit_p (jax < 0.9) by recursing into the inner jaxpr.

    Constants are always inlined by JAX as literal values in the inner jaxpr, so
    consts is always [] here.
    """
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    return natif_jaxpr(bind_jaxpr, [], *args, rigorous=_get_rigorous())


# jax >= 0.9 renamed pjit_p to jit_p
_jit_primitive = getattr(jax._src.pjit, "jit_p", getattr(jax._src.pjit, "pjit_p", None))
if _jit_primitive is not None:
    inclusion_registry[_jit_primitive] = _inclusion_pjit_p


def _inclusion_custom_jvp_call_p(primal_fn, jvp_fn, *args, **bind_params) -> Any:
    """Handle custom_jvp_call by natif-ing the primal, ignoring the JVP rule.

    The JVP rule exists for AD transforms and is irrelevant for interval
    arithmetic. primal_fn wraps jaxpr_as_fun(call_jaxpr); we extract the
    ClosedJaxpr directly and recurse with natif_jaxpr, mirroring pjit_p.
    """
    call_jaxpr = primal_fn.f.args[0]
    if isinstance(call_jaxpr, jax.extend.core.ClosedJaxpr):
        return natif_jaxpr(
            call_jaxpr.jaxpr, call_jaxpr.consts, *args, rigorous=_get_rigorous()
        )
    return natif_jaxpr(call_jaxpr, [], *args, rigorous=_get_rigorous())


_custom_jvp_call_primitive = getattr(
    jax._src.custom_derivatives, "custom_jvp_call_p", None
)
if _custom_jvp_call_primitive is not None:
    inclusion_registry[_custom_jvp_call_primitive] = _inclusion_custom_jvp_call_p


def _inclusion_scan_p(*args, **bind_params):
    bind_jaxpr = bind_params["jaxpr"]
    num_consts = bind_params["num_consts"]
    num_carry = bind_params["num_carry"]

    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        body_jaxpr = bind_jaxpr.jaxpr
        body_consts = bind_jaxpr.consts
    else:
        body_jaxpr = bind_jaxpr
        body_consts = []

    consts_vals = args[:num_consts]
    carry_init = args[num_consts : num_consts + num_carry]
    xs_vals = args[num_consts + num_carry :]

    # Flatten carry and xs pytrees (Intervals → (lower, upper) leaf pairs) so
    # lax.scan traces body_fn with plain abstract arrays.  Inside body_fn we
    # unflatten back to the original structure, so natif_jaxpr sees real
    # Interval objects (isinstance returns True even for abstract-valued ones).
    carry_flat, carry_treedef = jax.tree_util.tree_flatten(carry_init)
    xs_flat, xs_treedef = jax.tree_util.tree_flatten(xs_vals)

    y_treedef_ref = [None]

    def body_fn(carry_f, xs_f):
        carry_in = jax.tree_util.tree_unflatten(carry_treedef, carry_f)
        xs_in = jax.tree_util.tree_unflatten(xs_treedef, xs_f)
        out = natif_jaxpr(
            body_jaxpr,
            body_consts,
            *consts_vals,
            *carry_in,
            *xs_in,
            rigorous=_get_rigorous(),
        )
        carry_out = out[:num_carry]
        y_out = out[num_carry:]
        carry_out_flat, _ = jax.tree_util.tree_flatten(carry_out)
        y_out_flat, y_treedef = jax.tree_util.tree_flatten(y_out)
        y_treedef_ref[0] = y_treedef
        return carry_out_flat, y_out_flat

    final_carry_flat, ys_flat = lax.scan(
        body_fn,
        carry_flat,
        xs_flat,
        length=bind_params.get("length"),
        reverse=bind_params.get("reverse", False),
        unroll=bind_params.get("unroll", 1),
    )

    final_carry = jax.tree_util.tree_unflatten(carry_treedef, final_carry_flat)
    y_treedef = y_treedef_ref[0]
    ys = jax.tree_util.tree_unflatten(y_treedef, ys_flat) if y_treedef else []
    return [*final_carry, *ys]


inclusion_registry[lax.scan_p] = _inclusion_scan_p


def _inclusion_add_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(x.lower + y.lower, x.upper + y.upper)
    elif isinstance(x, Interval):
        return Interval(x.lower + y, x.upper + y)
    elif isinstance(y, Interval):
        return Interval(x + y.lower, x + y.upper)
    else:
        return x + y


_inclusion_add_p.n_ulps = 1
inclusion_registry[lax.add_p] = _inclusion_add_p
inclusion_registry[ad_util.add_any_p] = _inclusion_add_p
Interval.__add__ = _inclusion_add_p
Interval.__radd__ = _inclusion_add_p


def _inclusion_sub_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(x.lower - y.upper, x.upper - y.lower)
    elif isinstance(x, Interval):
        return Interval(x.lower - y, x.upper - y)
    elif isinstance(y, Interval):
        return Interval(x - y.upper, x - y.lower)
    else:
        return x - y


_inclusion_sub_p.n_ulps = 1
inclusion_registry[lax.sub_p] = _inclusion_sub_p
Interval.__sub__ = _inclusion_sub_p
Interval.__rsub__ = _inclusion_sub_p


def _inclusion_neg_p(x: Interval) -> Interval:
    return Interval(-x.upper, -x.lower)


_inclusion_neg_p.n_ulps = 0
inclusion_registry[lax.neg_p] = _inclusion_neg_p
Interval.__neg__ = _inclusion_neg_p


def _inclusion_mul_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        _1 = x.lower * y.lower
        _2 = x.lower * y.upper
        _3 = x.upper * y.lower
        _4 = x.upper * y.upper
        return Interval(
            jnp.minimum(jnp.minimum(_1, _2), jnp.minimum(_3, _4)),
            jnp.maximum(jnp.maximum(_1, _2), jnp.maximum(_3, _4)),
        )
    elif isinstance(x, Interval):
        _1 = x.lower * y
        _2 = x.upper * y
        return Interval(jnp.minimum(_1, _2), jnp.maximum(_1, _2))
    elif isinstance(y, Interval):
        _1 = x * y.lower
        _2 = x * y.upper
        return Interval(jnp.minimum(_1, _2), jnp.maximum(_1, _2))
    else:
        return x * y


_inclusion_mul_p.n_ulps = 1
inclusion_registry[lax.mul_p] = _inclusion_mul_p
Interval.__mul__ = _inclusion_mul_p
Interval.__rmul__ = _inclusion_mul_p


def _inclusion_div_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    elif isinstance(x, Interval):
        return _inclusion_mul_p(x, 1 / y)
    elif isinstance(y, Interval):
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    else:
        return x / y


_inclusion_div_p.n_ulps = 2
inclusion_registry[lax.div_p] = _inclusion_div_p
Interval.__truediv__ = _inclusion_div_p
Interval.__rtruediv__ = lambda x, y: _inclusion_div_p(y, x)


def _inclusion_reciprocal_p(x: Interval) -> Interval:
    if not isinstance(x, Interval):
        return 1 / x
    c = jnp.logical_or(
        jnp.logical_and(x.lower > 0, x.upper > 0),
        jnp.logical_and(x.lower < 0, x.upper < 0),
    )
    return Interval(
        jnp.where(c, (1.0 / x.upper), -jnp.inf), jnp.where(c, (1.0 / x.lower), jnp.inf)
    )


def _inclusion_integer_pow_p(x: Interval, y: int) -> Interval:
    if not isinstance(x, Interval):
        return x**y

    # x^0 = 1 for all x
    if isinstance(y, int) and y == 0:
        return Interval(jnp.ones_like(x.lower), jnp.ones_like(x.upper))

    def _inclusion_integer_pow_impl(x: Interval, y: int) -> Interval:
        l_pow = lax.integer_pow(x.lower, y)
        u_pow = lax.integer_pow(x.upper, y)

        def even():
            contains_zero = jnp.logical_and(
                jnp.less_equal(x.lower, 0), jnp.greater_equal(x.upper, 0)
            )
            lower = jnp.where(
                contains_zero, jnp.zeros_like(x.lower), jnp.minimum(l_pow, u_pow)
            )
            upper = jnp.maximum(l_pow, u_pow)
            return (lower, upper)

        odd = lambda: (l_pow, u_pow)

        return lax.cond(jnp.all(y % 2), odd, even)

    def _pos_pow():
        return _inclusion_integer_pow_impl(x, y)

    def _neg_pow():
        return _inclusion_integer_pow_impl(_inclusion_reciprocal_p(x), -y)

    ol, ou = lax.cond(jnp.all(y < 0), _neg_pow, _pos_pow)
    return Interval(ol, ou)


_inclusion_integer_pow_p.n_ulps = 2
inclusion_registry[lax.integer_pow_p] = _inclusion_integer_pow_p
Interval.__pow__ = _inclusion_integer_pow_p


def _inclusion_square_p(x: Interval) -> Interval:
    """Square an interval."""
    return _inclusion_integer_pow_p(x, 2)


_inclusion_square_p.n_ulps = 1
if hasattr(lax, "square_p"):
    inclusion_registry[lax.square_p] = _inclusion_square_p


def _inclusion_dot_general_p(A: Interval, B: Interval, **kwargs) -> Interval:
    # All checks of batch/contracting dims are done in first pass on lower bounds

    A = interval(A)
    B = interval(B)

    # Extract the contracting and batch dimensions
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = kwargs[
        "dimension_numbers"
    ]

    # Permute the batch then contracting dimensions to the front
    imoveaxis = lambda x, *args: Interval(
        jnp.moveaxis(x.lower, *args), jnp.moveaxis(x.upper, *args)
    )
    A = imoveaxis(
        A, lhs_batch + lhs_contracting, range(len(lhs_batch) + len(lhs_contracting))
    )
    B = imoveaxis(
        B, rhs_batch + rhs_contracting, range(len(rhs_batch) + len(rhs_contracting))
    )

    def _contract(A, B):
        # Multiplying two scalar intervals
        def _mul(a, b):
            _1 = a.lower * b.lower
            _2 = a.lower * b.upper
            _3 = a.upper * b.lower
            _4 = a.upper * b.upper
            result = Interval(
                jnp.minimum(jnp.minimum(_1, _2), jnp.minimum(_3, _4)),
                jnp.maximum(jnp.maximum(_1, _2), jnp.maximum(_3, _4)),
            )
            if _get_rigorous():
                result = widen(result, 1)
            return result

        def isum(x):
            result = Interval(jnp.sum(x.lower), jnp.sum(x.upper))
            if _get_rigorous():
                k = x.lower.size
                if k > 1:
                    result = widen(result, k - 1)
            return result

        # Two vectors -> scalar
        def f(a, b):
            _r = jax.vmap(_mul)
            return isum(_r(a, b))

        # Repeat over each contracting dimension
        for i in range(1, len(lhs_contracting)):
            _r = jax.vmap(f)
            f = lambda a, b: isum(_r(a, b))

        # vmap over non-contracting dimensions
        for i in range(len(lhs_contracting), len(A.shape)):
            f = jax.vmap(f, in_axes=(i, None), out_axes=-1)
        for j in range(len(rhs_contracting), len(B.shape)):
            f = jax.vmap(f, in_axes=(None, j), out_axes=-1)

        return f(A, B)

    # vmap over batch dimensions
    f = _contract
    for i in range(len(lhs_batch)):
        f = vmap(f, in_axes=(0, 0), out_axes=0)

    return f(A, B)


_inclusion_dot_general_p.n_ulps = 0
inclusion_registry[lax.dot_general_p] = _inclusion_dot_general_p


def _inclusion_sin_p(x: Interval, accuracy=None) -> Interval:
    if not isinstance(x, Interval):
        return lax.sin(x, accuracy=accuracy)

    def _sin_if(l: jnp.float32, u: jnp.float32):
        def case_lpi(l, u):
            cl = jnp.cos(l)
            cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2 * jnp.array(cu >= 0, "int32")
            case3 = lambda: (jnp.sin(l), jnp.sin(u))  # cl >= 0, cu >= 0
            case0 = lambda: (jnp.sin(u), jnp.sin(l))  # cl <= 0, cu <= 0
            case1 = lambda: (
                jnp.minimum(jnp.sin(l), jnp.sin(u)),
                1.0,
            )  # cl >= 0, cu <= 0
            case2 = lambda: (
                -1.0,
                jnp.maximum(jnp.sin(l), jnp.sin(u)),
            )  # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])

        def case_pi2pi(l, u):
            cl = jnp.cos(l)
            cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2 * jnp.array(cu >= 0, "int32")
            case3 = lambda: (-1.0, 1.0)  # cl >= 0, cu >= 0
            case0 = lambda: (-1.0, 1.0)  # cl <= 0, cu <= 0
            case1 = lambda: (
                jnp.minimum(jnp.sin(l), jnp.sin(u)),
                1.0,
            )  # cl >= 0, cu <= 0
            case2 = lambda: (
                -1.0,
                jnp.maximum(jnp.sin(l), jnp.sin(u)),
            )  # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])

        def case_else(l, u):
            return -1.0, 1.0

        diff = u - l
        c = jnp.array(diff <= jnp.pi, "int32") + jnp.array(diff <= 2 * jnp.pi, "int32")
        ol, ou = lax.switch(c, [case_else, case_pi2pi, case_lpi], l, u)
        return ol, ou

    _sin_if_vmap = jax.vmap(_sin_if, (0, 0))
    _x, x_ = _sin_if_vmap(x.lower.reshape(-1), x.upper.reshape(-1))
    return Interval(_x.reshape(x.shape), x_.reshape(x.shape))


_inclusion_sin_p.n_ulps = 2
inclusion_registry[lax.sin_p] = _inclusion_sin_p


def _inclusion_cos_p(x: Interval, accuracy=None) -> Interval:
    return _inclusion_sin_p(
        Interval(x.lower + jnp.pi / 2, x.upper + jnp.pi / 2), accuracy=accuracy
    )


_inclusion_cos_p.n_ulps = 2
inclusion_registry[lax.cos_p] = _inclusion_cos_p


def _inclusion_tan_p(x: Interval, accuracy=None) -> Interval:
    l = x.lower
    u = x.upper
    div = jnp.floor((u + jnp.pi / 2) / (jnp.pi)).astype(int)
    l -= div * jnp.pi
    u -= div * jnp.pi
    ol = jnp.where((l < -jnp.pi / 2), -jnp.inf, jnp.tan(l))
    ou = jnp.where((l < -jnp.pi / 2), jnp.inf, jnp.tan(u))
    return Interval(ol, ou)


_inclusion_tan_p.n_ulps = 2
inclusion_registry[lax.tan_p] = _inclusion_tan_p

# def _inclusion_atan_p (x:Interval, accuracy=None) -> Interval :
#     return Interval(lax.atan(x.lower), lax.atan(x.upper))
# inclusion_registry[lax.atan_p] = _inclusion_atan_p
_add_passthrough_to_registry(lax.atan_p, n_ulps=1)


def _inclusion_asin_p(x: Interval, accuracy=None) -> Interval:
    return Interval(lax.asin(x.lower), lax.asin(x.upper))


_inclusion_asin_p.n_ulps = 1
inclusion_registry[lax.asin_p] = _inclusion_asin_p


def _inclusion_sqrt_p(x: Interval, accuracy=None) -> Interval:
    ol = jnp.where((x.lower < 0), -jnp.inf, jnp.sqrt(x.lower))
    ou = jnp.where((x.lower < 0), -jnp.inf, jnp.sqrt(x.upper))
    return Interval(ol, ou)


_inclusion_sqrt_p.n_ulps = 1
inclusion_registry[lax.sqrt_p] = _inclusion_sqrt_p


def _inclusion_rsqrt_p(x: Interval, accuracy=None) -> Interval:
    # rsqrt = 1/sqrt(x) is monotonically decreasing
    # Map [a, b] -> [rsqrt(b), rsqrt(a)]
    # Handle domain x > 0.
    ol = jnp.where(
        (x.upper <= 0), -jnp.inf, lax.rsqrt(x.upper)
    )  # if upper <= 0, invalid. if lower <= 0, rsqrt(lower) inv.
    # Actually, interval semantics: if input contains invalid points, result is usually entire real line or restricted.
    # Existing sqrt uses -inf for invalid.
    # rsqrt(0) -> inf.
    # if x.lower <= 0, rsqrt(x.lower) is usually nan/inf.
    # Let's match JAX behavior but swap bounds.

    # Simple swap:
    lower_r = lax.rsqrt(x.upper)
    upper_r = lax.rsqrt(x.lower)

    # Handle negative inputs:
    # If upper < 0, result is invalid.
    ol = jnp.where(x.upper < 0, -jnp.inf, lower_r)
    ou = jnp.where(x.lower < 0, jnp.inf, upper_r)

    return Interval(ol, ou)


_inclusion_rsqrt_p.n_ulps = 1
if hasattr(lax, "rsqrt_p"):
    inclusion_registry[lax.rsqrt_p] = _inclusion_rsqrt_p
else:
    # Fallback if rsqrt_p is not directly exposed (it usually is)
    pass


def _inclusion_pow_p(x: Interval, y: Interval) -> Interval:
    # if isinstance (y, Interval) :
    #     # if y.lower == y.upper :
    #     if True :
    #         y = y.upper
    #     else :
    #         raise Exception('y must be a constant')

    x = interval(x)
    y = interval(y)

    def _inclusion_pow_impl(xl, xu, yl, yu) -> Interval:
        # caluclate corners
        corners = jnp.array(
            [lax.pow(xl, yl), lax.pow(xl, yu), lax.pow(xu, yl), lax.pow(xu, yu)]
        )
        # calculate the minimum and maximum of the corners
        cond = jnp.logical_and(x.lower >= 0, x.upper >= 0)
        ol = jnp.where(cond, jnp.min(corners), -jnp.inf)
        ou = jnp.where(cond, jnp.max(corners), jnp.inf)
        return ol, ou

    xl, yl = jnp.broadcast_arrays(x.lower, y.lower)
    xu, yu = jnp.broadcast_arrays(x.upper, y.upper)
    xsh = jnp.shape(xl)

    resl, resu = jax.vmap(_inclusion_pow_impl, (0, 0, 0, 0))(
        xl.reshape(-1), xu.reshape(-1), yl.reshape(-1), yu.reshape(-1)
    )
    return Interval(resl.reshape(xsh), resu.reshape(xsh))


_inclusion_pow_p.n_ulps = 2
inclusion_registry[lax.pow_p] = _inclusion_pow_p


def _inclusion_abs_p(x: Interval) -> Interval:
    ol = jnp.where(
        jnp.logical_and(x.lower <= 0, x.upper >= 0),
        0.0,
        jnp.minimum(lax.abs(x.lower), lax.abs(x.upper)),
    )
    ou = jnp.maximum(lax.abs(x.lower), lax.abs(x.upper))
    return Interval(ol, ou)


_inclusion_abs_p.n_ulps = 0
inclusion_registry[lax.abs_p] = _inclusion_abs_p

# def _inclusion_tanh_p(x: Interval, accuracy=None) -> Interval:
#     return Interval(lax.tanh(x.lower, accuracy=accuracy), lax.tanh(x.upper, accuracy=accuracy))

# inclusion_registry[lax.tanh_p] = _inclusion_tanh_p
_add_passthrough_to_registry(lax.tanh_p, n_ulps=1)
_add_passthrough_to_registry(lax.logistic_p, n_ulps=1)


def _inclusion_log_p(x: Interval, accuracy=None) -> Interval:
    ol = jnp.where((x.lower < 0), -jnp.inf, jnp.log(x.lower))
    ou = jnp.where((x.lower < 0), -jnp.inf, jnp.log(x.upper))
    return Interval(ol, ou)


_inclusion_log_p.n_ulps = 1
inclusion_registry[lax.log_p] = _inclusion_log_p


def _inclusion_log1p_p(x: Interval, accuracy=None) -> Interval:
    ol = jnp.where((x.lower < -1), -jnp.inf, jnp.log1p(x.lower))
    ou = jnp.where((x.lower < -1), -jnp.inf, jnp.log1p(x.upper))
    return Interval(ol, ou)


_inclusion_log1p_p.n_ulps = 1
inclusion_registry[lax.log1p_p] = _inclusion_log1p_p

Interval.__matmul__ = natif(jnp.matmul)
Interval.__rmatmul__ = lambda self, other: natif(jnp.matmul)(other, self)


def _inclusion_cumprod_p(x: Interval, *, axis=0, reverse=False) -> Interval:
    # Basic O(n) implementation of cumprod: scan over the axis, multiplying as we go

    def _cumprod_scan(carry, x):
        ic = interval(*carry)
        ix = interval(*x)

        new_ic = ic * ix
        new_carry = jnp.stack([new_ic.lower, new_ic.upper])
        return new_carry, new_carry

    # init = (jnp.ones_like(x.lower), jnp.ones_like(x.upper))
    # Move axis to the front for scanning, then move back at the end

    # print(x, axis)

    xl = jnp.moveaxis(x.lower, axis, 0)
    xu = jnp.moveaxis(x.upper, axis, 0)
    xs = jnp.stack([xl, xu], axis=1)  # shape (length, 2, ...)

    init = jnp.ones_like(xs[0])  # shape (2, ...)

    # print(init.shape)
    # print(xs)
    # print(xs.shape)

    _, result = lax.scan(_cumprod_scan, init, xs, reverse=reverse)

    # print(result)

    ret = interval(
        jnp.moveaxis(result[:, 0], 0, axis), jnp.moveaxis(result[:, 1], 0, axis)
    )

    return ret


def _fake_inclusion_cumprod_p(x: Interval, *, axis=0, reverse=False) -> Interval:
    # TODO: fix this
    # return interval(
    #     lax.cumprod(x.lower, axis=axis, reverse=reverse),
    #     lax.cumprod(x.upper, axis=axis, reverse=reverse),
    # )
    # ret = interval(jnp.ones_like(x.lower))
    retl = [jnp.ones_like(x.lower[0])]
    retu = [jnp.ones_like(x.upper[0])]
    for i in range(x.lower.shape[axis]):
        reti = interval(retl[-1], retu[-1]) * x[i]
        retl.append(reti.lower)
        retu.append(reti.upper)

    return interval(jnp.asarray(retl[1:]), jnp.array(retu[1:]))


# TODO:: correct this
_inclusion_cumprod_p.n_ulps = 0
inclusion_registry[lax.cumprod_p] = _inclusion_cumprod_p

# Some linear algebra routines


# Cholesky decomposition
def _manual_cholesky(A):
    """
    Computes the Cholesky decomposition of a symmetric positive definite matrix A using Python for loops.
    Returns lower-triangular matrix L such that A = L @ L.T
    """
    n = A.shape[0]
    L = jnp.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = jnp.sum(L[i, :j] * L[j, :j])
            val = jnp.where(i == j, jnp.sqrt(A[i, i] - s), (A[i, j] - s) / L[j, j])
            L = L.at[i, j].set(val)
    return L


inclusion_registry[LA.cholesky_p] = natif(_manual_cholesky)

# Triangular solve


def _manual_triangular_solve(
    A,
    b,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    # Apply transpose if needed
    A = jnp.where(transpose_a, A.T, A)
    # Apply conjugate if needed
    A = jnp.where(conjugate_a, jnp.conj(A), A)
    # # If unit_diagonal, set diagonal to 1
    # if unit_diagonal:
    #     A = A.at[jnp.diag_indices(A.shape[0])].set(1)

    def lower_triangular_solve(A, b):
        n = A.shape[0]
        x = jnp.zeros_like(b)
        for i in range(n):
            s = jnp.sum(A[i, :i] * x[:i])
            xi = (b[i] - s) / A[i, i]
            x = x.at[i].set(xi)
        return x

    def upper_triangular_solve(A, b):
        n = A.shape[0]
        x = jnp.zeros_like(b)
        for i in range(n - 1, -1, -1):
            s = jnp.sum(A[i, i + 1 :] * x[i + 1 :])
            x = x.at[i].set((b[i] - s) / A[i, i])
        return x

    # # Choose lower or upper triangular solve
    # x = lax.cond(lower,
    #              lambda _: lower_triangular_solve(A, b),
    #              lambda _: upper_triangular_solve(A, b),
    #              operand=None)

    # # If not left_side, solve xA = b instead of Ax = b
    # x = lax.cond(left_side,
    #              lambda x: x,
    #              lambda _: jnp.linalg.solve(A.T, b.T).T,
    #              x)

    if lower:
        if left_side:
            x = lower_triangular_solve(A, b)
        else:
            x = lower_triangular_solve(A.T, b.T).T
    else:
        if left_side:
            x = upper_triangular_solve(A, b)
        else:
            x = upper_triangular_solve(A.T, b.T).T

    # return lower_triangular_solve(A, b)
    return x


@partial(
    jit,
    static_argnames=(
        "left_side",
        "lower",
        "transpose_a",
        "conjugate_a",
        "unit_diagonal",
    ),
)
def _inclusion_triangular_solve(
    A,
    b,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    # return natif(partial(jax.vmap(_manual_triangular_solve, in_axes=()),
    #                      left_side=left_side, lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a, unit_diagonal=unit_diagonal))(A, b)
    return natif(
        partial(
            jax.vmap(_manual_triangular_solve, in_axes=()),
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=conjugate_a,
            unit_diagonal=unit_diagonal,
        )
    )(A, b)


inclusion_registry[LA.triangular_solve_p] = _inclusion_triangular_solve

# natif(lambda A, b, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False: _manual_triangular_solve(A, b, left_side=left_side, lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a, unit_diagonal=unit_diagonal))
