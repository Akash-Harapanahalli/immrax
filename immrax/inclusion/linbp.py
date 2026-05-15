from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import ad_util, source_info_util
from jax._src.core import (
    Atom,
    Jaxpr,
    Literal,
    Var,
    clean_up_dead_vars,
    last_used,
    typecheck,
)
from jax._src import config
from jax._src.util import safe_map
from jax.tree_util import register_pytree_node_class

from immrax.inclusion.interval import Interval, interval
from immrax.inclusion.nif import _inclusion_dot_general_p

"""
Forward linear bound propagation through a JAX function via Jaxpr interpretation.

For each intermediate variable y of shape S, we maintain:
    lA @ x_in + lb  <=  y  <=  uA @ x_in + ub
where lA, uA have shape (*S, n_in) and lb, ub have shape S.

We also maintain concrete bounds l, u (valid IBP bounds) for ReLU relaxation.
"""


@register_pytree_node_class
class LinearBound:
    """LinearBound: affine bounds as a function of network input x_in.

    For a variable y of shape S, represents:
        lA @ x_in + lb  <=  y  <=  uA @ x_in + ub

    where lA, uA have shape (*S, n_in) and lb, ub have shape S.
    Concrete bounds l, u (valid for any x_in in [x_lb, x_ub]) are
    also tracked for use in activation relaxations.
    """

    lA: jax.Array  # (*S, n_in)
    lb: jax.Array  # S
    uA: jax.Array  # (*S, n_in)
    ub: jax.Array  # S
    l: jax.Array  # S  concrete lower
    u: jax.Array  # S  concrete upper

    def __init__(
        self,
        lA: jax.Array,
        lb: jax.Array,
        uA: jax.Array,
        ub: jax.Array,
        l: jax.Array,
        u: jax.Array,
    ) -> None:
        self.lA = lA
        self.lb = lb
        self.uA = uA
        self.ub = ub
        self.l = l
        self.u = u

    def tree_flatten(self):
        return ((self.lA, self.lb, self.uA, self.ub, self.l, self.u), "LinearBound")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def shape(self):
        return self.l.shape

    @property
    def n_in(self):
        return self.lA.shape[-1]


# ---------------------------------------------------------------------------
# Registry and interpreter
# ---------------------------------------------------------------------------

linbp_registry = {}

# Primitives that recurse into inner jaxprs — these need tighten_bounds/x_lb/x_ub
# forwarded so they can pass them to nested _linbp_jaxpr calls.
_linbp_recursive_prims: set = set()

# Primitives where calling _concretize after the handler gives strictly tighter
# concrete bounds than the IBP l/u already computed by the handler.
#
# Only dot_general_p qualifies: IBP computes W_pos @ l + W_neg @ u treating each
# output neuron independently, but the accumulated affine bound (lA @ x + lb)
# respects that all output neurons share the same input x — so evaluating it at
# the input box recovers cross-neuron correlations that IBP discards.
#
# Activation primitives (max_p, min_p, logistic_p) do NOT benefit: their
# post-activation affine upper bound always evaluates to exactly max/min(u, c)
# at the input box, which matches the IBP concrete bound already stored in u.
# Structural primitives (reshape, broadcast, transpose, etc.) are pure shape
# operations whose concrete bounds are invariant to tightening.
_linbp_tighten_prims: set = {lax.dot_general_p}


def _linbp_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    relu_mode: str = "adaptive",
    propagate_source_info=True,
    tighten_bounds: bool = True,
    x_lb=None,
    x_ub=None,
    return_env: bool = False,
    iterated_bw: bool = False,
) -> list[Any]:
    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    def _tighten(a):
        """Intersect IBP concrete bounds with affine-evaluated bounds."""
        if not isinstance(a, LinearBound) or x_lb is None:
            return a
        concrete = _concretize(a, x_lb, x_ub)
        return LinearBound(
            a.lA, a.lb, a.uA, a.ub,
            jnp.maximum(a.l, concrete.lower),
            jnp.minimum(a.u, concrete.upper),
        )

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)

    # For iterated backward CROWN: map each var to the index of the eqn that produced it.
    producing_idx: dict = {}
    if iterated_bw:
        for j, e in enumerate(jaxpr.eqns):
            for ov in e.outvars:
                producing_idx[ov] = j

    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any(isinstance(v, LinearBound) for v in invars):
                if eqn.primitive not in linbp_registry:
                    raise NotImplementedError(f"{eqn.primitive} not in linbp_registry")
                # Iterated backward CROWN: at each activation / recursive-wrap eqn,
                # tighten its LinearBound invars' l, u via a backward CROWN pass
                # over the already-processed prefix of the jaxpr. Subsequent
                # activations benefit from these tighter pre-activation bounds.
                if iterated_bw and x_lb is not None and (
                    eqn.primitive in _activation_prims_bw
                    or eqn.primitive in _linbp_recursive_prims
                ):
                    for k, iv in enumerate(eqn.invars):
                        if isinstance(iv, Literal):
                            continue
                        val = env.get(iv)
                        if not isinstance(val, LinearBound):
                            continue
                        prod_idx = producing_idx.get(iv)
                        if prod_idx is None:
                            continue
                        l_new, u_new = _backward_to_concrete(
                            jaxpr, env, iv, prod_idx, x_lb, x_ub, relu_mode
                        )
                        env[iv] = LinearBound(
                            val.lA, val.lb, val.uA, val.ub,
                            jnp.maximum(val.l, l_new),
                            jnp.minimum(val.u, u_new),
                        )
                    invars = safe_map(read, eqn.invars)
                # Pass eqn.params directly (avoids get_bind_params moving things like
                # call_jaxpr/num_consts out of kwargs and into subfuns).
                # Only recursive primitives (jit_p, custom_jvp_call_p) need
                # tighten_bounds/x_lb/x_ub — they forward them to nested jaxpr calls.
                if eqn.primitive in _linbp_recursive_prims:
                    extra = dict(tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub,
                                 iterated_bw=iterated_bw)
                else:
                    extra = {}
                ans = linbp_registry[eqn.primitive](
                    *invars, relu_mode=relu_mode, **extra, **eqn.params,
                )
                if tighten_bounds and eqn.primitive in _linbp_tighten_prims:
                    if eqn.primitive.multiple_results:
                        ans = [_tighten(a) for a in ans]
                    else:
                        ans = _tighten(ans)
            else:
                bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
                subfuns = bind_params.pop('subfuns', ())
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        if not return_env:
            clean_up_dead_vars(eqn, env, lu)

    outs = safe_map(read, jaxpr.outvars)
    if return_env:
        return outs, env
    return outs


# ---------------------------------------------------------------------------
# Helper: concretize LinearBound to Interval using x_lb, x_ub
# ---------------------------------------------------------------------------


def _concretize(lb: LinearBound, x_lb: jax.Array, x_ub: jax.Array) -> Interval:
    """Evaluate the affine bounds at (x_lb, x_ub) to get concrete interval."""
    lAp = jnp.clip(lb.lA, 0, None)
    lAn = jnp.clip(lb.lA, None, 0)
    uAp = jnp.clip(lb.uA, 0, None)
    uAn = jnp.clip(lb.uA, None, 0)
    # lower = lA @ x gives the minimum over x in [x_lb, x_ub]
    lower = (lAp * x_lb + lAn * x_ub).sum(-1) + lb.lb
    upper = (uAp * x_ub + uAn * x_lb).sum(-1) + lb.ub
    return interval(lower, upper)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _linbp_add_p(x, y, *, relu_mode, **kwargs):
    if isinstance(x, LinearBound) and isinstance(y, LinearBound):
        return LinearBound(
            lA=x.lA + y.lA,
            lb=x.lb + y.lb,
            uA=x.uA + y.uA,
            ub=x.ub + y.ub,
            l=x.l + y.l,
            u=x.u + y.u,
        )
    elif isinstance(x, LinearBound):
        # y is a plain array (bias, etc.)
        return LinearBound(
            lA=x.lA,
            lb=x.lb + y,
            uA=x.uA,
            ub=x.ub + y,
            l=x.l + y,
            u=x.u + y,
        )
    else:
        # x is a plain array, y is LinearBound
        return LinearBound(
            lA=y.lA,
            lb=x + y.lb,
            uA=y.uA,
            ub=x + y.ub,
            l=x + y.l,
            u=x + y.u,
        )


linbp_registry[lax.add_p] = _linbp_add_p
linbp_registry[ad_util.add_any_p] = _linbp_add_p


def _linbp_neg_p(x, *, relu_mode, **kwargs):
    return LinearBound(
        lA=-x.uA,
        lb=-x.ub,
        uA=-x.lA,
        ub=-x.lb,
        l=-x.u,
        u=-x.l,
    )


linbp_registry[lax.neg_p] = _linbp_neg_p


def _linbp_sub_p(x, y, *, relu_mode, **kwargs):
    if isinstance(y, LinearBound):
        neg_y = _linbp_neg_p(y, relu_mode=relu_mode)
        return _linbp_add_p(x, neg_y, relu_mode=relu_mode)
    else:
        # x is LinearBound, y is array (may be a TypedNdArray from equinox)
        return _linbp_add_p(x, jnp.negative(jnp.asarray(y)), relu_mode=relu_mode)


linbp_registry[lax.sub_p] = _linbp_sub_p


def _linbp_mul_p(x, y, *, relu_mode, **kwargs):
    # One must be a non-LinearBound (constant scaling)
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        c = y
        cp = jnp.clip(c, 0, None)
        cn = jnp.clip(c, None, 0)
        # c has shape S; lA has shape (*S, n_in) — need [..., None] for broadcast
        uA = cp[..., None] * x.uA + cn[..., None] * x.lA
        lA = cp[..., None] * x.lA + cn[..., None] * x.uA
        ub = cp * x.ub + cn * x.lb
        lb = cp * x.lb + cn * x.ub
        l = jnp.minimum(c * x.l, c * x.u)
        u = jnp.maximum(c * x.l, c * x.u)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)
    elif not isinstance(x, LinearBound) and isinstance(y, LinearBound):
        return _linbp_mul_p(y, x, relu_mode=relu_mode)
    else:
        # Both LinearBound: fall back to IBP using concrete bounds
        l = jnp.minimum(
            jnp.minimum(x.l * y.l, x.l * y.u),
            jnp.minimum(x.u * y.l, x.u * y.u),
        )
        u = jnp.maximum(
            jnp.maximum(x.l * y.l, x.l * y.u),
            jnp.maximum(x.u * y.l, x.u * y.u),
        )
        n_in = x.n_in
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )


linbp_registry[lax.mul_p] = _linbp_mul_p


def _linbp_dot_general_p(x, y, *, relu_mode, dimension_numbers, **kwargs):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    dim_nums = dimension_numbers

    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        # x is LinearBound, y (W) is constant: e.g. x @ W
        # x.uA has shape (*S_x, n_in); n_in is an LHS free dim in dot_general,
        # so JAX would place it before RHS free dims, giving (*S_x_free, n_in, *S_out)
        # instead of (*S_out, n_in). Fix: vmap over the n_in axis.
        W = y
        Wp = jnp.clip(W, 0, None)
        Wn = jnp.clip(W, None, 0)
        dot = lambda a, b: lax.dot_general(a, b, dim_nums)
        dot_vmap = jax.vmap(dot, in_axes=(-1, None), out_axes=-1)
        uA = dot_vmap(x.uA, Wp) + dot_vmap(x.lA, Wn)
        lA = dot_vmap(x.lA, Wp) + dot_vmap(x.uA, Wn)
        ub = dot(x.ub, Wp) + dot(x.lb, Wn)
        lb = dot(x.lb, Wp) + dot(x.ub, Wn)
        ul = dot(x.u, Wp) + dot(x.l, Wn)
        ll = dot(x.l, Wp) + dot(x.u, Wn)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=ll, u=ul)

    elif not isinstance(x, LinearBound) and isinstance(y, LinearBound):
        # x (W) is constant, y is LinearBound: e.g. W @ y
        W = x
        Wp = jnp.clip(W, 0, None)
        Wn = jnp.clip(W, None, 0)
        uA = lax.dot_general(Wp, y.uA, dim_nums) + lax.dot_general(Wn, y.lA, dim_nums)
        lA = lax.dot_general(Wp, y.lA, dim_nums) + lax.dot_general(Wn, y.uA, dim_nums)
        ub = lax.dot_general(Wp, y.ub, dim_nums) + lax.dot_general(Wn, y.lb, dim_nums)
        lb = lax.dot_general(Wp, y.lb, dim_nums) + lax.dot_general(Wn, y.ub, dim_nums)
        ul = lax.dot_general(Wp, y.u, dim_nums) + lax.dot_general(Wn, y.l, dim_nums)
        ll = lax.dot_general(Wp, y.l, dim_nums) + lax.dot_general(Wn, y.u, dim_nums)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=ll, u=ul)

    else:
        # Both LinearBound: fall back to natif interval arithmetic, which does
        # element-wise 4-corner min/max then reduces over contracting dims.
        n_in = x.n_in
        result = _inclusion_dot_general_p(
            interval(x.l, x.u),
            interval(y.l, y.u),
            dimension_numbers=dim_nums,
            **kwargs,
        )
        ll, ul = result.lower, result.upper
        S = ll.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=ll,
            uA=jnp.zeros((*S, n_in)),
            ub=ul,
            l=ll,
            u=ul,
        )


linbp_registry[lax.dot_general_p] = _linbp_dot_general_p


def _make_linbp_structural_p(primitive):
    """Factory for pure structural (shape/index) linbp handlers.

    Mirrors :func:`_make_inclusion_passthrough_p` in nif.py.  Data arrays
    (l, u, lb, ub) are processed by applying the primitive directly.
    A-matrix arrays (lA, uA, shape ``(*S, n_in)``) are processed by vmapping
    over the trailing ``n_in`` axis so the primitive always sees shape ``S`` —
    no per-primitive kwarg adjustment is needed.

    For plain-array arguments the linear contribution is implicitly zero
    (their A-matrix slot is filled with zeros before vmapping).

    Works for both single-result and multiple-result primitives (e.g.
    ``split_p``); in the latter case a list of :class:`LinearBound` is
    returned.
    """
    is_multi = primitive.multiple_results

    def handler(*args, relu_mode, **kwargs):
        n_in = next(a.n_in for a in args if isinstance(a, LinearBound))

        def _data(a, f):
            return getattr(a, f) if isinstance(a, LinearBound) else a

        def _A(a, f):
            if isinstance(a, LinearBound):
                return getattr(a, f)
            arr = jnp.asarray(a)
            return jnp.zeros((*arr.shape, n_in), dtype=arr.dtype)

        l  = primitive.bind(*[_data(a, "l")  for a in args], **kwargs)
        u  = primitive.bind(*[_data(a, "u")  for a in args], **kwargs)
        lb = primitive.bind(*[_data(a, "lb") for a in args], **kwargs)
        ub = primitive.bind(*[_data(a, "ub") for a in args], **kwargs)

        _fn = lambda *xs: primitive.bind(*xs, **kwargs)

        def _apply_A(f):
            As = [_A(a, f) for a in args]
            return jax.vmap(_fn, in_axes=[-1] * len(As), out_axes=-1)(*As)

        if is_multi:
            lA_parts = _apply_A("lA")
            uA_parts = _apply_A("uA")
            return [
                LinearBound(lA=lAi, lb=lbi, uA=uAi, ub=ubi, l=li, u=ui)
                for lAi, lbi, uAi, ubi, li, ui
                in zip(lA_parts, lb, uA_parts, ub, l, u)
            ]
        return LinearBound(lA=_apply_A("lA"), lb=lb, uA=_apply_A("uA"), ub=ub, l=l, u=u)

    return handler


def _add_structural_to_linbp_registry(primitive):
    """Register a structural primitive using the unified vmap-based factory."""
    linbp_registry[primitive] = _make_linbp_structural_p(primitive)


for _p in [
    lax.reshape_p,
    lax.slice_p,
    lax.broadcast_in_dim_p,
    lax.transpose_p,
    lax.squeeze_p,
    lax.concatenate_p,
    lax.convert_element_type_p,
]:
    _add_structural_to_linbp_registry(_p)

if hasattr(lax, "split_p"):
    _add_structural_to_linbp_registry(lax.split_p)
if hasattr(lax, "dynamic_slice_p"):
    _add_structural_to_linbp_registry(lax.dynamic_slice_p)
if hasattr(lax, "gather_p"):
    _add_structural_to_linbp_registry(lax.gather_p)


def _make_linbp_scatter_p(primitive):
    """Factory for scatter-family linbp handlers.

    scatter_indices is always a plain integer array (never a LinearBound), so
    it is captured as a constant and not vmapped.  Only the A-matrices of
    operand and updates are vmapped over the trailing n_in axis.
    """
    def handler(operand, scatter_indices, updates, *, relu_mode, **kwargs):
        n_in = next(a.n_in for a in [operand, updates] if isinstance(a, LinearBound))

        def _d(a, f):
            return getattr(a, f) if isinstance(a, LinearBound) else a

        def _A(a, f):
            if isinstance(a, LinearBound):
                return getattr(a, f)
            arr = jnp.asarray(a)
            return jnp.zeros((*arr.shape, n_in), dtype=arr.dtype)

        _scat = lambda op, upd: primitive.bind(op, scatter_indices, upd, **kwargs)

        l  = _scat(_d(operand, "l"),  _d(updates, "l"))
        u  = _scat(_d(operand, "u"),  _d(updates, "u"))
        lb = _scat(_d(operand, "lb"), _d(updates, "lb"))
        ub = _scat(_d(operand, "ub"), _d(updates, "ub"))

        def _apply_A(f):
            return jax.vmap(
                lambda op_k, upd_k: primitive.bind(op_k, scatter_indices, upd_k, **kwargs),
                in_axes=(-1, -1), out_axes=-1,
            )(_A(operand, f), _A(updates, f))

        return LinearBound(lA=_apply_A("lA"), lb=lb, uA=_apply_A("uA"), ub=ub, l=l, u=u)

    return handler


for _p_name in ["scatter_p", "scatter_add_p", "scatter_mul_p"]:
    if hasattr(lax, _p_name):
        linbp_registry[getattr(lax, _p_name)] = _make_linbp_scatter_p(getattr(lax, _p_name))


def _linbp_reduce_sum_p(x, *, relu_mode, axes, out_sharding=None, **kwargs):
    """reduce_sum is linear: A-matrices reduce exactly over the same axes."""
    if not isinstance(x, LinearBound):
        return lax.reduce_sum_p.bind(x, axes=axes, out_sharding=out_sharding)
    # out_sharding is sized for the primal output rank; lA/uA have one extra
    # trailing n_in dimension, so omit out_sharding for them to avoid a rank
    # mismatch in JAX's sharding system.
    _rs = lambda t: lax.reduce_sum_p.bind(t, axes=axes, out_sharding=out_sharding)
    _rs_A = lambda t: lax.reduce_sum_p.bind(t, axes=axes)
    return LinearBound(
        lA=_rs_A(x.lA), lb=_rs(x.lb), uA=_rs_A(x.uA), ub=_rs(x.ub),
        l=_rs(x.l), u=_rs(x.u),
    )


linbp_registry[lax.reduce_sum_p] = _linbp_reduce_sum_p


def _linbp_select_n_p(which, *cases, relu_mode, **kwargs):
    """select_n_p (lax.select_n) handler.

    which is always a plain integer array (never a LinearBound), so it is
    captured as a constant and not vmapped.  A-matrices for each case are
    vmapped over the trailing n_in axis with which held fixed.
    """
    n_in = next(a.n_in for a in cases if isinstance(a, LinearBound))

    def _d(a, f):
        return getattr(a, f) if isinstance(a, LinearBound) else a

    def _A(a, f):
        if isinstance(a, LinearBound):
            return getattr(a, f)
        arr = jnp.asarray(a)
        return jnp.zeros((*arr.shape, n_in), dtype=arr.dtype)

    _sel = lambda *cs: lax.select_n_p.bind(which, *cs, **kwargs)

    l  = _sel(*[_d(c, "l")  for c in cases])
    u  = _sel(*[_d(c, "u")  for c in cases])
    lb = _sel(*[_d(c, "lb") for c in cases])
    ub = _sel(*[_d(c, "ub") for c in cases])

    def _apply_A(f):
        cases_A = [_A(c, f) for c in cases]
        return jax.vmap(
            lambda *cs_k: lax.select_n_p.bind(which, *cs_k, **kwargs),
            in_axes=(-1,) * len(cases), out_axes=-1,
        )(*cases_A)

    return LinearBound(lA=_apply_A("lA"), lb=lb, uA=_apply_A("uA"), ub=ub, l=l, u=u)


if hasattr(lax, "select_n_p"):
    linbp_registry[lax.select_n_p] = _linbp_select_n_p


def _linbp_max_p(x, y, *, relu_mode, **kwargs):
    """General max(x, c) handler for any constant c.

    Upper bound: chord from (l, c) to (u, u) for active neurons.
    Lower bound for active neurons is selected by relu_mode:
      'same-slope' — parallel to upper, tight at kink x = c
      'adaptive'   — slope 0 or 1 based on area heuristic (l+u >= 2c → slope 1)
      'zero'       — slope 0 (constant c)
      'one'        — slope 1 (identity)
    Both alpha values are always >= 0, so upper/lower A-matrices are used
    in the tightest direction.
    """
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        lb_in = x
        c = jnp.asarray(y)
    elif isinstance(y, LinearBound) and not isinstance(x, LinearBound):
        lb_in = y
        c = jnp.asarray(x)
    else:
        # Both LinearBound: IBP fallback
        n_in = x.n_in
        l = jnp.maximum(x.l, y.l)
        u = jnp.maximum(x.u, y.u)
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )

    l, u = lb_in.l, lb_in.u

    on = l >= c        # always above threshold: max(x, c) = x
    off = u <= c       # always below threshold: max(x, c) = c
    active = ~on & ~off

    safe_denom = jnp.where(active, u - l, 1.0)

    # Upper bound: chord from (l, c) to (u, u)
    alpha_u_act = (u - c) / safe_denom
    alpha_u = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_u_act))
    beta_u = jnp.where(on, 0.0, jnp.where(off, c, c - alpha_u * l))

    # Lower bound for active neurons (on/off are always identity/constant)
    if relu_mode == "same-slope":
        alpha_l_act = alpha_u_act
        beta_l_act = c * (1.0 - alpha_u_act)
    elif relu_mode == "adaptive":
        use_id = l + u >= 2.0 * c
        alpha_l_act = jnp.where(use_id, 1.0, 0.0)
        beta_l_act = jnp.where(use_id, 0.0, c)
    elif relu_mode == "zero":
        alpha_l_act = 0.0
        beta_l_act = c
    elif relu_mode == "one":
        alpha_l_act = 1.0
        beta_l_act = 0.0
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")

    alpha_l = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_l_act))
    beta_l = jnp.where(on, 0.0, jnp.where(off, c, beta_l_act))

    uA = alpha_u[..., None] * lb_in.uA
    ub = alpha_u * lb_in.ub + beta_u
    lA = alpha_l[..., None] * lb_in.lA
    lb = alpha_l * lb_in.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=jnp.maximum(l, c), u=jnp.maximum(u, c))


linbp_registry[lax.max_p] = _linbp_max_p


def _linbp_min_p(x, y, *, relu_mode, **kwargs):
    """General min(x, c) handler for any constant c.

    Lower bound: chord from (l, l) to (u, c) for active neurons.
    Upper bound for active neurons is selected by relu_mode:
      'same-slope' — parallel to lower, tight at kink x = c
      'adaptive'   — slope 0 or 1 based on area heuristic (l+u <= 2c → slope 1)
      'zero'       — slope 0 (constant c)
      'one'        — slope 1 (identity)
    Both alpha values are always >= 0.
    """
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        lb_in = x
        c = jnp.asarray(y)
    elif isinstance(y, LinearBound) and not isinstance(x, LinearBound):
        lb_in = y
        c = jnp.asarray(x)
    else:
        # Both LinearBound: IBP fallback
        n_in = x.n_in
        l = jnp.minimum(x.l, y.l)
        u = jnp.minimum(x.u, y.u)
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )

    l, u = lb_in.l, lb_in.u

    on = u <= c        # always below threshold: min(x, c) = x
    off = l >= c       # always above threshold: min(x, c) = c
    active = ~on & ~off

    safe_denom = jnp.where(active, u - l, 1.0)

    # Lower bound: chord from (l, l) to (u, c)
    alpha_l_act = (c - l) / safe_denom
    alpha_l = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_l_act))
    beta_l = jnp.where(on, 0.0, jnp.where(off, c, l * (1.0 - alpha_l_act)))

    # Upper bound for active neurons (on/off are always identity/constant)
    if relu_mode == "same-slope":
        alpha_u_act = alpha_l_act
        beta_u_act = c * (1.0 - alpha_l_act)
    elif relu_mode == "adaptive":
        use_id = l + u <= 2.0 * c
        alpha_u_act = jnp.where(use_id, 1.0, 0.0)
        beta_u_act = jnp.where(use_id, 0.0, c)
    elif relu_mode == "zero":
        alpha_u_act = 0.0
        beta_u_act = c
    elif relu_mode == "one":
        alpha_u_act = 1.0
        beta_u_act = 0.0
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")

    alpha_u = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_u_act))
    beta_u = jnp.where(on, 0.0, jnp.where(off, c, beta_u_act))

    uA = alpha_u[..., None] * lb_in.uA
    ub = alpha_u * lb_in.ub + beta_u
    lA = alpha_l[..., None] * lb_in.lA
    lb = alpha_l * lb_in.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=jnp.minimum(l, c), u=jnp.minimum(u, c))


linbp_registry[lax.min_p] = _linbp_min_p


def _linbp_logistic_p(x, *, relu_mode, **kwargs):
    """Sigmoid handler using a chord-based linear relaxation.

    For sigmoid σ on [l, u] we use the chord slope α = (σ(u)−σ(l))/(u−l)
    as the common linear coefficient (preserving linear information from
    earlier layers).  The tightest valid upper/lower intercepts for slope α
    are achieved at the critical points where σ'(x*) = α, i.e.
        σ(x*) = (1 ± √(1−4α)) / 2
    giving x* = log(σ(x*)/(1−σ(x*))).

    Upper intercept β_u = max_{x∈[l,u]} (σ(x) − α·x)
      → achieved at x*_upper (≥ 0, concave region) if it lies in [l, u],
        otherwise at the chord intercept (= σ(l)−α·l).

    Lower intercept β_l = min_{x∈[l,u]} (σ(x) − α·x)
      → achieved at x*_lower (≤ 0, convex region) if it lies in [l, u],
        otherwise at the chord intercept.

    Since α ≥ 0, the lA/uA slots keep their linear meaning (lA is used for
    the lower bound, uA for the upper bound), scaled by α.
    """
    l, u = x.l, x.u
    sig_l, sig_u = jax.nn.sigmoid(l), jax.nn.sigmoid(u)

    # Chord slope α ∈ (0, 0.25]; falls back to σ'(l) for degenerate intervals.
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, sig_l * (1.0 - sig_l), (sig_u - sig_l) / safe_denom)

    # Chord intercept (= σ(l)−α·l = σ(u)−α·u by construction)
    beta_chord = sig_l - alpha * l

    # Critical sigmoid values where σ'(x*) = α → σ(x*)(1−σ(x*)) = α
    # disc = √(1−4α) ≥ 0 because α ≤ max(σ') = 0.25
    # Use jnp.where to avoid sqrt(0) gradient NaN when alpha=0.25 (degenerate interval).
    one_minus_4alpha = jnp.maximum(1.0 - 4.0 * alpha, 0.0)
    safe_disc = jnp.where(one_minus_4alpha <= 0.0, jnp.ones_like(one_minus_4alpha), one_minus_4alpha)
    disc = jnp.where(one_minus_4alpha <= 0.0, jnp.zeros_like(one_minus_4alpha), jnp.sqrt(safe_disc))

    # Upper critical point in concave region (x*_upper ≥ 0)
    sig_xu = (1.0 + disc) / 2.0
    x_upper = jnp.log(sig_xu) - jnp.log(1.0 - sig_xu)   # logit
    beta_u_crit = sig_xu - alpha * x_upper

    # Lower critical point in convex region (x*_lower ≤ 0)
    sig_xl = (1.0 - disc) / 2.0
    x_lower = jnp.log(sig_xl) - jnp.log(1.0 - sig_xl)   # logit
    beta_l_crit = sig_xl - alpha * x_lower

    # Use the critical-point intercept only when the critical point is in [l, u].
    # x*_upper ≥ 0 ≥ l always, so the only check needed is x*_upper ≤ u.
    # x*_lower ≤ 0 ≤ u always, so the only check needed is x*_lower ≥ l.
    beta_u = jnp.where(x_upper <= u, beta_u_crit, beta_chord)
    beta_l = jnp.where(x_lower >= l, beta_l_crit, beta_chord)

    # α ≥ 0, so upper bound uses uA_x (upper side) and lower bound uses lA_x.
    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=sig_l, u=sig_u)


linbp_registry[lax.logistic_p] = _linbp_logistic_p


def _linbp_tanh_p(x, *, relu_mode, **kwargs):
    """Tanh handler using chord slope and critical-point tangent intercepts.

    tanh is S-shaped (convex x < 0, concave x > 0) with tanh'(x) = 1 − tanh²(x).
    Uses the same chord-slope strategy as _linbp_logistic_p:
      α = (tanh(u) − tanh(l)) / (u − l)   (chord slope, ∈ (0, 1])

    Critical points where tanh'(x*) = α  →  tanh(x*) = ±√(1−α):
      x*_upper = atanh(√(1−α))  (concave region, x ≥ 0, gives max of tanh − α·x)
      x*_lower = −x*_upper       (convex region,  x ≤ 0, gives min of tanh − α·x)

    β_u = tanh(x*_upper) − α·x*_upper  (upper intercept, used if x*_upper ≤ u)
    β_l = tanh(x*_lower) − α·x*_lower  (lower intercept, used if x*_lower ≥ l)

    Since α ≥ 0, upper bound uses uA and lower uses lA.
    """
    l, u = x.l, x.u
    tanh_l, tanh_u = jnp.tanh(l), jnp.tanh(u)

    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, 1.0 - tanh_l ** 2, (tanh_u - tanh_l) / safe_denom)

    chord_beta = tanh_l - alpha * l

    # Critical sigmoid value: tanh(x*) = sqrt(1 - alpha), clamped for numerics.
    # Use jnp.where to avoid sqrt(0) gradient NaN when alpha=1 (degenerate interval).
    one_minus_alpha = jnp.maximum(1.0 - alpha, 0.0)
    safe_oma = jnp.where(one_minus_alpha <= 0.0, jnp.ones_like(one_minus_alpha), one_minus_alpha)
    tanh_xu = jnp.where(one_minus_alpha <= 0.0, jnp.zeros_like(one_minus_alpha), jnp.sqrt(safe_oma))
    x_upper = jnp.arctanh(jnp.clip(tanh_xu, 0.0, 1.0 - 1e-7))
    beta_u_crit = tanh_xu - alpha * x_upper

    # x*_lower = -x*_upper, tanh(x*_lower) = -tanh_xu
    x_lower = -x_upper
    beta_l_crit = -tanh_xu - alpha * x_lower  # = -tanh_xu + alpha * x_upper

    beta_u = jnp.where(x_upper <= u, beta_u_crit, chord_beta)
    beta_l = jnp.where(x_lower >= l, beta_l_crit, chord_beta)

    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=tanh_l, u=tanh_u)


linbp_registry[lax.tanh_p] = _linbp_tanh_p



def _sin_concrete(l, u):
    """Tight concrete bounds for sin on [l, u]."""
    sin_l, sin_u = jnp.sin(l), jnp.sin(u)
    # Check if a maximum (π/2 + 2kπ) or minimum (-π/2 + 2kπ) lies in [l, u]
    has_max = jnp.floor((u - jnp.pi / 2) / (2 * jnp.pi)) > jnp.floor((l - jnp.pi / 2) / (2 * jnp.pi))
    has_min = jnp.floor((u + jnp.pi / 2) / (2 * jnp.pi)) > jnp.floor((l + jnp.pi / 2) / (2 * jnp.pi))
    lo = jnp.where(has_min, -1.0, jnp.minimum(sin_l, sin_u))
    hi = jnp.where(has_max,  1.0, jnp.maximum(sin_l, sin_u))
    return lo, hi



def _linbp_sin_p(x, *, relu_mode, **kwargs):
    """sin handler: chord slope with parallel-tangent upper/lower corrections.

    α = (sin(u) − sin(l)) / (u − l)  (chord slope; may be negative)

    Upper bound correction at x_+ = arccos(α) + 2kπ  (where sin−α·x is max):
      f(x_+) = √(1−α²),  β_u_crit = √(1−α²) − α·x_+

    Lower bound correction at x_- = −arccos(α) + 2kπ (where sin−α·x is min):
      f(x_-) = −√(1−α²), β_l_crit = −√(1−α²) − α·x_-

    k is chosen so x_± is nearest the interval midpoint.
    Wide intervals (≥ 2π) fall back to α = 0, β = ±1.
    Since α can be negative, A-matrices split into positive/negative parts.
    """
    l, u = x.l, x.u

    wide = (u - l) >= 2 * jnp.pi
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate | wide, 1.0, u - l)
    alpha = jnp.where(
        wide, 0.0,
        jnp.where(degenerate, jnp.cos(l), (jnp.sin(u) - jnp.sin(l)) / safe_denom)
    )

    beta_chord = jnp.sin(l) - alpha * l
    mid = (l + u) / 2
    alpha_c = jnp.clip(alpha, -1.0 + 1e-7, 1.0 - 1e-7)
    one_minus_a2 = jnp.maximum(1.0 - alpha ** 2, 0.0)
    safe_val = jnp.where(one_minus_a2 <= 0.0, jnp.ones_like(one_minus_a2), one_minus_a2)
    sin_sq = jnp.where(one_minus_a2 <= 0.0, jnp.zeros_like(one_minus_a2), jnp.sqrt(safe_val))
    x_base = jnp.arccos(alpha_c)                            # ∈ (0, π)

    # Upper: x_+ = arccos(α) + 2kπ
    x_plus = x_base + jnp.round((mid - x_base) / (2 * jnp.pi)) * 2 * jnp.pi
    in_plus = (x_plus >= l) & (x_plus <= u)
    beta_u = jnp.where(
        wide, 1.0,
        jnp.where(in_plus,
                  jnp.maximum(beta_chord, sin_sq - alpha * x_plus),
                  beta_chord)
    )

    # Lower: x_- = −arccos(α) + 2kπ
    x_minus = -x_base + jnp.round((mid + x_base) / (2 * jnp.pi)) * 2 * jnp.pi
    in_minus = (x_minus >= l) & (x_minus <= u)
    beta_l = jnp.where(
        wide, -1.0,
        jnp.where(in_minus,
                  jnp.minimum(beta_chord, -sin_sq - alpha * x_minus),
                  beta_chord)
    )

    # Concrete bounds
    lo, hi = _sin_concrete(l, u)

    # A-matrices: split α into positive/negative (α can be negative for sin)
    ap = jnp.clip(alpha, 0.0, None)
    an = jnp.clip(alpha, None, 0.0)
    uA = ap[..., None] * x.uA + an[..., None] * x.lA
    ub = ap * x.ub + an * x.lb + beta_u
    lA = ap[..., None] * x.lA + an[..., None] * x.uA
    lb = ap * x.lb + an * x.ub + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=lo, u=hi)


linbp_registry[lax.sin_p] = _linbp_sin_p


def _linbp_cos_p(x, *, relu_mode, **kwargs):
    """cos handler: chord slope with parallel-tangent corrections.

    Mirrors _linbp_sin_p with cos'(x) = −sin(x):
      α = (cos(u) − cos(l)) / (u − l)

    Upper bound correction at x_+ = arcsin(−α) + 2kπ  (f(x_+) = √(1−α²))
    Lower bound correction at x_- = π − arcsin(−α) + 2kπ (f(x_-) = −√(1−α²))
    """
    l, u = x.l, x.u

    wide = (u - l) >= 2 * jnp.pi
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate | wide, 1.0, u - l)
    alpha = jnp.where(
        wide, 0.0,
        jnp.where(degenerate, -jnp.sin(l), (jnp.cos(u) - jnp.cos(l)) / safe_denom)
    )

    beta_chord = jnp.cos(l) - alpha * l
    mid = (l + u) / 2
    alpha_c = jnp.clip(alpha, -1.0 + 1e-7, 1.0 - 1e-7)
    one_minus_a2 = jnp.maximum(1.0 - alpha ** 2, 0.0)
    safe_val = jnp.where(one_minus_a2 <= 0.0, jnp.ones_like(one_minus_a2), one_minus_a2)
    cos_sq = jnp.where(one_minus_a2 <= 0.0, jnp.zeros_like(one_minus_a2), jnp.sqrt(safe_val))
    x_base_p = jnp.arcsin(-alpha_c)                         # ∈ (−π/2, π/2)
    x_base_n = jnp.pi - x_base_p                            # ∈ (π/2, 3π/2)

    # Upper: x_+ = arcsin(−α) + 2kπ  (cos(x_+) = √(1−α²) > 0)
    x_plus = x_base_p + jnp.round((mid - x_base_p) / (2 * jnp.pi)) * 2 * jnp.pi
    in_plus = (x_plus >= l) & (x_plus <= u)
    beta_u = jnp.where(
        wide, 1.0,
        jnp.where(in_plus,
                  jnp.maximum(beta_chord, cos_sq - alpha * x_plus),
                  beta_chord)
    )

    # Lower: x_- = π − arcsin(−α) + 2kπ  (cos(x_-) = −√(1−α²) < 0)
    x_minus = x_base_n + jnp.round((mid - x_base_n) / (2 * jnp.pi)) * 2 * jnp.pi
    in_minus = (x_minus >= l) & (x_minus <= u)
    beta_l = jnp.where(
        wide, -1.0,
        jnp.where(in_minus,
                  jnp.minimum(beta_chord, -cos_sq - alpha * x_minus),
                  beta_chord)
    )

    # Concrete bounds: cos(x) = sin(x + π/2)
    lo, hi = _sin_concrete(l + jnp.pi / 2, u + jnp.pi / 2)

    ap = jnp.clip(alpha, 0.0, None)
    an = jnp.clip(alpha, None, 0.0)
    uA = ap[..., None] * x.uA + an[..., None] * x.lA
    ub = ap * x.ub + an * x.lb + beta_u
    lA = ap[..., None] * x.lA + an[..., None] * x.uA
    lb = ap * x.lb + an * x.ub + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=lo, u=hi)


linbp_registry[lax.cos_p] = _linbp_cos_p


def _linbp_tan_p(x, *, relu_mode, **kwargs):
    """tan handler: chord slope with tangent-line corrections.

    tan'(x) = sec²(x) ≥ 1, convex for x > 0, concave for x < 0 (within each
    period), inflection at 0 — same shape as tanh but unbounded.

    α = (tan(u) − tan(l)) / (u − l)   (chord slope, ≥ 1 within a period)

    Critical points where tan'(x*) = α → sec²(x*) = α → tan(x*) = ±√(α−1):
      x*_upper = arctan(√(α−1))   (convex region x > 0, max of tan − α·x)
      x*_lower = −x*_upper         (concave region x < 0, min of tan − α·x)

    β_u = tan(x*_upper) − α·x*_upper = √(α−1) − α·arctan(√(α−1))
    β_l = tan(x*_lower) − α·x*_lower = −√(α−1) + α·arctan(√(α−1))
    """
    l, u = x.l, x.u
    tan_l, tan_u = jnp.tan(l), jnp.tan(u)

    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, 1.0 + tan_l ** 2, (tan_u - tan_l) / safe_denom)

    chord_beta = tan_l - alpha * l

    # Critical tan value: tan(x*) = sqrt(alpha - 1)
    tan_crit = jnp.sqrt(jnp.clip(alpha - 1.0, 0.0))
    x_upper = jnp.arctan(tan_crit)
    beta_u_crit = tan_crit - alpha * x_upper

    x_lower = -x_upper
    beta_l_crit = -tan_crit - alpha * x_lower  # = -tan_crit + alpha * x_upper

    # h(x) = tan(x) - α·x has a LOCAL MAX at x*_lower (concave region, x<0)
    # and a LOCAL MIN at x*_upper (convex region, x>0).
    # Upper bound needs β_u = max h = h(x*_lower) when x*_lower ∈ [l, u].
    # Lower bound needs β_l = min h = h(x*_upper) when x*_upper ∈ [l, u].
    beta_u = jnp.where(x_lower >= l, beta_l_crit, chord_beta)
    beta_l = jnp.where(x_upper <= u, beta_u_crit, chord_beta)

    uA = alpha * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=tan_l, u=tan_u)


linbp_registry[lax.tan_p] = _linbp_tan_p


def _linbp_integer_pow_p(x, *, y, relu_mode, **kwargs):
    """x^n handler for static integer exponent n.

    n = 0: constant 1
    n = 1: identity (pass-through)
    n >= 2, even: globally convex — chord upper bound, tangent-at-critical lower bound
    n >= 3, odd: convex x>0 / concave x<0 (like tan) — chord slope with
                 critical-point intercept corrections
    n < 0: IBP fallback (zero-slope constant bounds from endpoints)
    """
    n = y
    l, u = x.l, x.u

    if n == 0:
        ones = jnp.ones_like(l)
        zero_A = jnp.zeros_like(x.lA)
        return LinearBound(lA=zero_A, lb=ones, uA=zero_A, ub=ones, l=ones, u=ones)

    if n == 1:
        return LinearBound(lA=x.lA, lb=x.lb, uA=x.uA, ub=x.ub, l=l, u=u)

    pow_l = l ** n
    pow_u = u ** n

    if n < 0:
        # IBP fallback: just use concrete endpoint bounds with zero slopes
        f_l = jnp.minimum(pow_l, pow_u)
        f_u = jnp.maximum(pow_l, pow_u)
        zero_A = jnp.zeros_like(x.lA)
        return LinearBound(lA=zero_A, lb=f_l, uA=zero_A, ub=f_u, l=f_l, u=f_u)

    # n >= 2: compute concrete bounds
    if n % 2 == 0:
        # Even n: globally convex, minimum at x=0 if 0 in [l, u]
        f_concrete_l = jnp.where(
            (l <= 0) & (0 <= u), jnp.zeros_like(l), jnp.minimum(pow_l, pow_u)
        )
        f_concrete_u = jnp.maximum(pow_l, pow_u)
    else:
        # Odd n >= 3: monotone increasing
        f_concrete_l = pow_l
        f_concrete_u = pow_u

    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(
        degenerate,
        jnp.array(n, dtype=l.dtype) * l ** (n - 1),
        (pow_u - pow_l) / safe_denom,
    )
    chord_beta = pow_l - alpha * l

    if n % 2 == 0:
        # Globally convex: upper = chord, lower = tangent parallel to chord.
        # f'(x0) = alpha  =>  n*x0^(n-1) = alpha  =>  x0 = sign(alpha/n)*|alpha/n|^(1/(n-1))
        # n-1 is odd for even n, so real root is always defined.
        safe_alpha_n = alpha / n
        x0 = jnp.sign(safe_alpha_n) * jnp.abs(safe_alpha_n) ** (1.0 / (n - 1))
        beta_l_crit = x0 ** n - alpha * x0
        in_range = (l <= x0) & (x0 <= u)
        beta_u = chord_beta
        beta_l = jnp.where(in_range, beta_l_crit, chord_beta)
    else:
        # Odd n >= 3: same critical-point structure as tan.
        # h(x) = x^n - alpha*x has local MAX at x_lower = -x_upper, local MIN at x_upper.
        # n-1 is even, so x^(n-1) = |x|^(n-1), and alpha/n >= 0 by MVT.
        safe_alpha_n = jnp.maximum(alpha / n, 0.0)
        x_upper = safe_alpha_n ** (1.0 / (n - 1))
        x_lower = -x_upper
        f_x_upper = x_upper ** n
        # h at x_lower (local MAX) -> beta_u; h at x_upper (local MIN) -> beta_l
        beta_at_x_lower = -f_x_upper + alpha * x_upper
        beta_at_x_upper = f_x_upper - alpha * x_upper
        beta_u = jnp.where(x_lower >= l, beta_at_x_lower, chord_beta)
        beta_l = jnp.where(x_upper <= u, beta_at_x_upper, chord_beta)

    uA = alpha * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=f_concrete_l, u=f_concrete_u)


linbp_registry[lax.integer_pow_p] = _linbp_integer_pow_p
linbp_registry[lax.square_p] = lambda x, **kw: _linbp_integer_pow_p(x, y=2, **kw)


def _linbp_exp_p(x, *, relu_mode, **kwargs):
    """Exponential handler using chord upper bound and parallel tangent lower bound.

    exp is globally convex, so:
      - Upper bound: chord from (l, eˡ) to (u, eᵘ) — lies above exp on [l, u].
      - Lower bound: tangent at x₀ = log(α_u) where α_u is the chord slope.
        This is the unique tangent parallel to the chord and gives the tightest
        lower affine bound of that slope (CROWN same-slope approach).

    For degenerate intervals (|u − l| < 1e-8) both bounds collapse to the
    tangent at l: α = eˡ, β = eˡ·(1 − l).

    Since α_u = α_l ≥ 0, the upper bound uses uA and the lower uses lA.
    """
    l, u = x.l, x.u
    exp_l, exp_u = jnp.exp(l), jnp.exp(u)

    # Chord slope α_u = (eᵘ − eˡ) / (u − l); falls back to eˡ = exp'(l).
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, exp_l, (exp_u - exp_l) / safe_denom)

    # Upper bound intercept: β_u = eˡ − α·l  (chord through (l, eˡ) and (u, eᵘ))
    beta_u = exp_l - alpha * l

    # Lower bound: tangent at x₀ = log(α) (same slope as chord).
    # β_l = exp(x₀) − α·x₀ = α − α·log(α) = α·(1 − log(α))
    log_alpha = jnp.where(degenerate, l, jnp.log(jnp.clip(alpha, 1e-30)))
    beta_l = alpha * (1.0 - log_alpha)

    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=exp_l, u=exp_u)


linbp_registry[lax.exp_p] = _linbp_exp_p


def _linbp_log1p_p(x, *, relu_mode, **kwargs):
    """log1p(x) = log(1+x) handler using chord lower bound and parallel tangent upper bound.

    log1p is globally concave (for x > −1), so:
      - Lower bound: chord from (l, log1p(l)) to (u, log1p(u)) — lies below.
      - Upper bound: tangent at x₀ = 1/α − 1 where α is the chord slope,
        i.e. where log1p'(x₀) = α.  Since log1p is concave, every tangent
        is a global upper bound, so this is always valid regardless of whether
        x₀ ∈ [l, u].

    For degenerate intervals (|u − l| < 1e-8) both bounds collapse to the
    tangent at l: α = 1/(1+l), β = log1p(l) − α·l.

    Since α ≥ 0 (log1p is increasing), the lower bound uses lA and the upper
    bound uses uA, matching the convention in the rest of the file.
    """
    l, u = x.l, x.u
    log1p_l = jnp.log1p(l)
    log1p_u = jnp.log1p(u)

    # Chord slope α = (log1p(u) − log1p(l)) / (u − l); falls back to 1/(1+l).
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, 1.0 / (1.0 + l), (log1p_u - log1p_l) / safe_denom)

    # Lower bound: chord  β_l = log1p(l) − α·l
    beta_l = log1p_l - alpha * l

    # Upper bound: tangent at x₀ = 1/α − 1  (log1p'(x₀) = α)
    #   log1p(x₀) = log(1/α) = −log(α)
    #   β_u = log1p(x₀) − α·x₀ = −log(α) − α·(1/α − 1) = α − 1 − log(α)
    log_alpha = jnp.log(jnp.clip(alpha, 1e-30))
    beta_u = jnp.where(degenerate, log1p_l - alpha * l, alpha - 1.0 - log_alpha)

    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=log1p_l, u=log1p_u)


linbp_registry[lax.log1p_p] = _linbp_log1p_p


def _linbp_sqrt_p(x, *, relu_mode, **kwargs):
    """sqrt handler using chord lower bound and parallel tangent upper bound.

    sqrt is globally concave on [0, ∞), so:
      - Lower bound: chord from (l, √l) to (u, √u) — lies below sqrt on [l, u].
      - Upper bound: tangent at x₀ = 1/(4α²) where sqrt'(x₀) = α (chord slope).
        Since sqrt is concave, every tangent is a global upper bound.
        β_u = sqrt(x₀) − α·x₀ = 1/(2α) − 1/(4α) = 1/(4α).

    For degenerate intervals (|u − l| < 1e-8) both bounds collapse to the
    tangent at l: α = 1/(2√l), β = √l/2.

    Inputs are clipped to [0, ∞) since sqrt is only defined there.
    Since α ≥ 0 (sqrt is increasing), the lower bound uses lA and the upper
    bound uses uA.
    """
    l = jnp.maximum(x.l, 0.0)
    u = jnp.maximum(x.u, 0.0)
    sqrt_l = jnp.sqrt(l)
    sqrt_u = jnp.sqrt(u)

    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    safe_sqrt_l = jnp.maximum(sqrt_l, 1e-15)
    # Chord slope α = (√u − √l) / (u − l) = 1 / (√u + √l)
    alpha = jnp.where(
        degenerate,
        1.0 / (2.0 * safe_sqrt_l),
        (sqrt_u - sqrt_l) / safe_denom,
    )

    # Lower bound: chord  β_l = √l − α·l
    beta_l = sqrt_l - alpha * l

    # Upper bound: tangent at x₀ = 1/(4α²),  β_u = 1/(4α)
    safe_alpha = jnp.maximum(alpha, 1e-30)
    beta_u = jnp.where(degenerate, sqrt_l - alpha * l, 1.0 / (4.0 * safe_alpha))

    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=sqrt_l, u=sqrt_u)


linbp_registry[lax.sqrt_p] = _linbp_sqrt_p


def _linbp_div_p(x, y, *, relu_mode, **kwargs):
    """Division handler.

    x / c (LinearBound / constant): delegates to mul with 1/c.
    c / y (constant / LinearBound): IBP fallback (nonlinear in y).
    Both LinearBound: IBP fallback.
    """
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        return _linbp_mul_p(x, jnp.reciprocal(y), relu_mode=relu_mode)
    elif not isinstance(x, LinearBound) and isinstance(y, LinearBound):
        n_in = y.n_in
        corners = [x / y.l, x / y.u]
        l = jnp.minimum(corners[0], corners[1])
        u = jnp.maximum(corners[0], corners[1])
        S = l.shape
        return LinearBound(lA=jnp.zeros((*S, n_in)), lb=l, uA=jnp.zeros((*S, n_in)), ub=u, l=l, u=u)
    else:
        n_in = x.n_in
        corners = [x.l / y.l, x.l / y.u, x.u / y.l, x.u / y.u]
        l = jnp.minimum(jnp.minimum(corners[0], corners[1]), jnp.minimum(corners[2], corners[3]))
        u = jnp.maximum(jnp.maximum(corners[0], corners[1]), jnp.maximum(corners[2], corners[3]))
        S = l.shape
        return LinearBound(lA=jnp.zeros((*S, n_in)), lb=l, uA=jnp.zeros((*S, n_in)), ub=u, l=l, u=u)


linbp_registry[lax.div_p] = _linbp_div_p


def _linbp_jit_p(*args, relu_mode, tighten_bounds=True, x_lb=None, x_ub=None,
                 iterated_bw=False, **bind_params):
    """Handle jit_p (jax >= 0.9) / pjit_p (jax < 0.9) by recursing."""
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    return _linbp_jaxpr(bind_jaxpr, [], *args, relu_mode=relu_mode,
                        tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub,
                        iterated_bw=iterated_bw)


_jit_primitive = getattr(jax._src.pjit, "jit_p", getattr(jax._src.pjit, "pjit_p", None))
if _jit_primitive is not None:
    linbp_registry[_jit_primitive] = _linbp_jit_p
    _linbp_recursive_prims.add(_jit_primitive)


def _linbp_custom_jvp_call_p(*args, relu_mode, call_jaxpr, num_consts,
                             tighten_bounds=True, x_lb=None, x_ub=None,
                             iterated_bw=False, **bind_params):
    """Handle custom_jvp_call by evaluating the primal call_jaxpr only."""
    if isinstance(call_jaxpr, jax.extend.core.ClosedJaxpr):
        consts = call_jaxpr.consts
        inner_jaxpr = call_jaxpr.jaxpr
    else:
        consts = []
        inner_jaxpr = call_jaxpr
    # Leading num_consts args come from the enclosing closure; remainder are invars
    extra_consts = list(args[:num_consts])
    actual_args = args[num_consts:]
    return _linbp_jaxpr(
        inner_jaxpr, consts + extra_consts, *actual_args, relu_mode=relu_mode,
        tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub,
        iterated_bw=iterated_bw,
    )


_custom_jvp_call_p = getattr(jax._src.custom_derivatives, "custom_jvp_call_p", None)
if _custom_jvp_call_p is not None:
    linbp_registry[_custom_jvp_call_p] = _linbp_custom_jvp_call_p
    _linbp_recursive_prims.add(_custom_jvp_call_p)


# ===========================================================================
# Backward linear bound propagation (true CROWN with sign-conditioned slopes)
# ===========================================================================
#
# Forward linbp (above) commits to a single ReLU lower-bound slope per neuron
# based on the geometric heuristic l+u >= 2c.  In deep networks that's loose:
# the optimal slope at each ReLU depends on the *sign* of the cumulative
# downstream coefficient, which is only available when sweeping backward from
# the output.
#
# Strategy: run forward linbp once to capture per-var LinearBounds (and thus
# pre-activation l, u at every ReLU).  Then walk the jaxpr equations in
# reverse, carrying a BackwardBound(A_lo, b_lo, A_hi, b_hi) per var.  At each
# activation, decompose A by sign and pick a slope per (output, neuron).
# At each linear primitive, transpose the operation.
# ---------------------------------------------------------------------------


from collections import namedtuple


class BackwardBound(namedtuple("BackwardBound", ["A_lo", "b_lo", "A_hi", "b_hi"])):
    """For a var v of shape S and an *original output* of size n_out,

        A_lo, A_hi: shape (n_out, *S)
        b_lo, b_hi: shape (n_out,)

    Interpretation: lb(v) := sum over S of A_lo * v + b_lo gives a lower
    bound on the original output (entry by entry along n_out); similarly
    upper bound from A_hi, b_hi.
    """
    pass


backward_registry: dict = {}
_bw_recursive_prims: set = set()


def _bw_sum_over_var(A, x):
    """Contract A (a covector) with x (broadcastable to A's shape), returning scalar.

    Called inside ``vmap`` over the leading n_out axis; A's shape is the var
    shape S, x is shape broadcastable to S (possibly scalar).
    """
    return jnp.sum(jnp.asarray(A) * jnp.asarray(x))


def _bw_dot_general(eqn, out_bd, env, **_):
    """y = dot_general(a, b) with one of a, b a constant (no LinearBound).

    Common cases for NNs: W @ x or x @ W with W constant.
    """
    dim_nums = eqn.params["dimension_numbers"]
    (lhs_c, rhs_c), (lhs_b, rhs_b) = dim_nums
    if lhs_b or rhs_b:
        raise NotImplementedError("backward CROWN: batched dot_general not supported yet")
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 and is_lb1:
        raise NotImplementedError("backward CROWN: dot_general of two LinearBounds")

    # We need A_input such that bound on output via A_output translates back.
    # output = sum over contracting axes of (lhs * rhs).
    # For each row j of A_output (a covector on the output shape),
    # bound_j = sum over output_dims of A_output[j] * output.
    # Substituting output = einsum(lhs, rhs) and re-grouping gives
    # A_input as another einsum.  We use jax.linear_transpose for safety.
    if is_lb1:
        # v0 is constant W; v1 is the variable input
        W = val0
        var = v1
        var_shape = var.aval.shape
        primal = lambda x: lax.dot_general(W, x, dim_nums)
    else:
        W = val1
        var = v0
        var_shape = var.aval.shape
        primal = lambda x: lax.dot_general(x, W, dim_nums)
    dummy = jnp.zeros(var_shape, dtype=jnp.result_type(jnp.float32))

    def transpose_one(cov):
        (tr,) = jax.linear_transpose(primal, dummy)(cov)
        return tr

    new_A_lo = jax.vmap(transpose_one)(out_bd.A_lo)
    new_A_hi = jax.vmap(transpose_one)(out_bd.A_hi)
    in_bd = BackwardBound(new_A_lo, out_bd.b_lo, new_A_hi, out_bd.b_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_add(eqn, out_bd, env, **_):
    """y = a + b. Linear; if one operand is constant, fold into bias terms."""
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 and is_lb1:
        # both variable: A propagates to both; bias only to first to avoid duplication
        z_lo = jnp.zeros_like(out_bd.b_lo)
        z_hi = jnp.zeros_like(out_bd.b_hi)
        in0 = BackwardBound(out_bd.A_lo, out_bd.b_lo, out_bd.A_hi, out_bd.b_hi)
        in1 = BackwardBound(out_bd.A_lo, z_lo, out_bd.A_hi, z_hi)
        return [in0, in1]
    # one is const
    if not is_lb0:
        const = val0
        var = v1
    else:
        const = val1
        var = v0
    db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_lo)
    db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_hi)
    in_bd = BackwardBound(out_bd.A_lo, out_bd.b_lo + db_lo, out_bd.A_hi, out_bd.b_hi + db_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_sub(eqn, out_bd, env, **_):
    """y = a - b."""
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 and is_lb1:
        z_lo = jnp.zeros_like(out_bd.b_lo)
        z_hi = jnp.zeros_like(out_bd.b_hi)
        in0 = BackwardBound(out_bd.A_lo, out_bd.b_lo, out_bd.A_hi, out_bd.b_hi)
        in1 = BackwardBound(-out_bd.A_lo, z_lo, -out_bd.A_hi, z_hi)
        return [in0, in1]
    if not is_lb0:
        # y = const - var
        const = val0
        var = v1
        db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_lo)
        db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_hi)
        in_bd = BackwardBound(
            -out_bd.A_lo, out_bd.b_lo + db_lo,
            -out_bd.A_hi, out_bd.b_hi + db_hi,
        )
    else:
        # y = var - const
        const = val1
        var = v0
        db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_lo)
        db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_hi)
        in_bd = BackwardBound(
            out_bd.A_lo, out_bd.b_lo - db_lo,
            out_bd.A_hi, out_bd.b_hi - db_hi,
        )
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_mul(eqn, out_bd, env, **_):
    """y = a * b with one of a, b a constant. Substituting y = c*x:
        bound = A @ y + b = A @ (c*x) + b = (A * c_broadcasted) @ x + b
    """
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 and is_lb1:
        raise NotImplementedError("backward CROWN: mul of two LinearBounds")
    if is_lb0:
        var = v0
        c = jnp.asarray(val1)
    else:
        var = v1
        c = jnp.asarray(val0)
    new_A_lo = out_bd.A_lo * c
    new_A_hi = out_bd.A_hi * c
    in_bd = BackwardBound(new_A_lo, out_bd.b_lo, new_A_hi, out_bd.b_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_neg(eqn, out_bd, env, **_):
    """y = -x. Swap lower/upper roles and negate."""
    in_bd = BackwardBound(-out_bd.A_hi, -out_bd.b_hi, -out_bd.A_lo, -out_bd.b_lo)
    return [in_bd]


def _bw_max_with_const(eqn, out_bd, env, *, relu_mode="adaptive"):
    """y = max(x, c). Pre-activation l, u read from env[x].l, env[x].u.

    Slope choice for active neurons follows ``relu_mode``:
    ``'adaptive'`` — α_l ∈ {0, 1} by l+u ≥ 2c heuristic
    ``'same-slope'`` — α_l = α_u (the chord); β_l = c(1-α_u)
    ``'zero'``      — α_l = 0, β_l = c
    ``'one'``       — α_l = 1, β_l = 0
    Sign-conditioning happens in the A decomposition below regardless.
    """
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 == is_lb1:
        raise NotImplementedError("backward CROWN: max of two LinearBounds")
    if is_lb0:
        lb_in = val0; c_val = jnp.asarray(val1); var = v0
    else:
        lb_in = val1; c_val = jnp.asarray(val0); var = v1

    l = lb_in.l
    u = lb_in.u
    on = l >= c_val
    off = u <= c_val
    active = ~on & ~off
    safe_denom = jnp.where(active, u - l, 1.0)
    alpha_u_act = (u - c_val) / safe_denom
    alpha_u = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_u_act))
    beta_u = jnp.where(on, 0.0, jnp.where(off, c_val, c_val - alpha_u * l))
    if relu_mode == "same-slope":
        alpha_l_act = alpha_u_act
        beta_l_act = c_val * (1.0 - alpha_u_act)
    elif relu_mode == "adaptive":
        use_id = l + u >= 2.0 * c_val
        alpha_l_act = jnp.where(use_id, 1.0, 0.0)
        beta_l_act = jnp.where(use_id, 0.0, c_val)
    elif relu_mode == "zero":
        alpha_l_act = jnp.zeros_like(alpha_u_act)
        beta_l_act = jnp.broadcast_to(c_val, alpha_u_act.shape).astype(alpha_u_act.dtype)
    elif relu_mode == "one":
        alpha_l_act = jnp.ones_like(alpha_u_act)
        beta_l_act = jnp.zeros_like(alpha_u_act)
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")
    alpha_l = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_l_act))
    beta_l = jnp.where(on, 0.0, jnp.where(off, c_val, beta_l_act))

    # Decompose A by sign over the var dims (trailing dims of A).
    # For upper bound: positive A contributes via upper bound of y; negative
    # A contributes via lower bound. β_l = 0 for ReLU (c=0); β_u handled below.
    Ahi_p = jnp.clip(out_bd.A_hi, 0, None)
    Ahi_n = jnp.clip(out_bd.A_hi, None, 0)
    Alo_p = jnp.clip(out_bd.A_lo, 0, None)
    Alo_n = jnp.clip(out_bd.A_lo, None, 0)

    # Broadcast alpha/beta (shape S) over leading n_out axis of A.
    new_A_hi = Ahi_p * alpha_u + Ahi_n * alpha_l
    new_A_lo = Alo_p * alpha_l + Alo_n * alpha_u
    db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, beta_u))(Ahi_p) + \
            jax.vmap(lambda a: _bw_sum_over_var(a, beta_l))(Ahi_n)
    db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, beta_l))(Alo_p) + \
            jax.vmap(lambda a: _bw_sum_over_var(a, beta_u))(Alo_n)
    in_bd = BackwardBound(new_A_lo, out_bd.b_lo + db_lo, new_A_hi, out_bd.b_hi + db_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_structural(eqn, out_bd, env, **_):
    """Generic structural op (slice, squeeze, reshape, broadcast_in_dim,
    concatenate, etc.): use jax.linear_transpose on the primitive."""
    var_idx = next(i for i, v in enumerate(eqn.invars) if isinstance(env.get(v), LinearBound))
    var = eqn.invars[var_idx]
    var_shape = var.aval.shape

    # Build primal of the primitive with the variable arg replaced by dummy
    bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
    subfuns = bind_params.pop("subfuns", ())
    fixed = [env.get(v) if env.get(v) is not None else (v.val if isinstance(v, Literal) else None)
             for v in eqn.invars]
    def primal(x):
        full = list(fixed)
        full[var_idx] = x
        return eqn.primitive.bind(*subfuns, *full, **bind_params)
    dummy = jnp.zeros(var_shape, dtype=jnp.float32)

    def transpose_one(cov):
        (tr,) = jax.linear_transpose(primal, dummy)(cov)
        return tr

    new_A_lo = jax.vmap(transpose_one)(out_bd.A_lo)
    new_A_hi = jax.vmap(transpose_one)(out_bd.A_hi)
    in_bd = BackwardBound(new_A_lo, out_bd.b_lo, new_A_hi, out_bd.b_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_concatenate(eqn, out_bd, env, **_):
    """y = concatenate(xs, axis). Split A along (axis+1) (n_out is leading).

    For const invars (not LinearBound): fold their fixed contribution
    A_output[:, const_slice] · const_value into the bias.
    """
    axis = eqn.params["dimension"]
    sizes = [v.aval.shape[axis] for v in eqn.invars]
    offsets = []
    s = 0
    for n in sizes:
        offsets.append((s, s + n))
        s += n

    # First pass: accumulate bias contributions from const invars
    b_lo_acc = out_bd.b_lo
    b_hi_acc = out_bd.b_hi
    n_var_dims = len(out_bd.A_lo.shape) - 1
    for (lo, hi), var in zip(offsets, eqn.invars):
        val = var.val if isinstance(var, Literal) else env.get(var)
        if isinstance(val, LinearBound):
            continue
        sl = (slice(None),) + tuple(
            slice(None) if d != axis else slice(lo, hi) for d in range(n_var_dims)
        )
        const_val = jnp.asarray(val)
        Alo_p = jnp.clip(out_bd.A_lo[sl], 0, None)
        Alo_n = jnp.clip(out_bd.A_lo[sl], None, 0)
        Ahi_p = jnp.clip(out_bd.A_hi[sl], 0, None)
        Ahi_n = jnp.clip(out_bd.A_hi[sl], None, 0)
        # For a const, lower-bound contribution = A @ const (same value top and bottom)
        # so just do A @ const for both directions.
        db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const_val))(out_bd.A_lo[sl])
        db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const_val))(out_bd.A_hi[sl])
        b_lo_acc = b_lo_acc + db_lo
        b_hi_acc = b_hi_acc + db_hi

    # Second pass: distribute A_output[sl] to each LB invar; bias only to first
    in_bds = []
    bias_given = False
    z_lo = jnp.zeros_like(out_bd.b_lo)
    z_hi = jnp.zeros_like(out_bd.b_hi)
    for (lo, hi), var in zip(offsets, eqn.invars):
        val = var.val if isinstance(var, Literal) else env.get(var)
        if not isinstance(val, LinearBound):
            in_bds.append(None)
            continue
        sl = (slice(None),) + tuple(
            slice(None) if d != axis else slice(lo, hi) for d in range(n_var_dims)
        )
        if not bias_given:
            in_bds.append(BackwardBound(
                out_bd.A_lo[sl], b_lo_acc,
                out_bd.A_hi[sl], b_hi_acc,
            ))
            bias_given = True
        else:
            in_bds.append(BackwardBound(
                out_bd.A_lo[sl], z_lo,
                out_bd.A_hi[sl], z_hi,
            ))
    if not bias_given:
        raise NotImplementedError("backward CROWN: concatenate with no LB invars")
    return in_bds


# Activation primitives that benefit from iterated backward CROWN at the
# point of slope selection. Recursive primitives (jit_p, custom_jvp_call_p)
# are also tightened because they wrap activations in practice.
_activation_prims_bw: set = {lax.max_p, lax.min_p, lax.logistic_p, lax.tanh_p}


backward_registry[lax.dot_general_p] = _bw_dot_general
backward_registry[lax.add_p] = _bw_add
backward_registry[lax.sub_p] = _bw_sub
backward_registry[lax.mul_p] = _bw_mul
backward_registry[lax.neg_p] = _bw_neg
backward_registry[lax.max_p] = _bw_max_with_const
backward_registry[lax.concatenate_p] = _bw_concatenate
for _p in (lax.slice_p, lax.squeeze_p, lax.reshape_p, lax.broadcast_in_dim_p,
           lax.transpose_p, lax.convert_element_type_p):
    backward_registry[_p] = _bw_structural


def _bw_monotone_positive_slope(out_bd, alpha, beta_l, beta_u):
    """Backward through a single-input activation with linear bounds
        α * x + β_l  ≤  σ(x)  ≤  α * x + β_u,   α ≥ 0
    Sign-condition the bias terms (slope is the same for both bounds).
    """
    new_A_hi = out_bd.A_hi * alpha
    new_A_lo = out_bd.A_lo * alpha
    Ahi_p = jnp.clip(out_bd.A_hi, 0, None)
    Ahi_n = jnp.clip(out_bd.A_hi, None, 0)
    Alo_p = jnp.clip(out_bd.A_lo, 0, None)
    Alo_n = jnp.clip(out_bd.A_lo, None, 0)
    db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, beta_u))(Ahi_p) + \
            jax.vmap(lambda a: _bw_sum_over_var(a, beta_l))(Ahi_n)
    db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, beta_l))(Alo_p) + \
            jax.vmap(lambda a: _bw_sum_over_var(a, beta_u))(Alo_n)
    return BackwardBound(new_A_lo, out_bd.b_lo + db_lo, new_A_hi, out_bd.b_hi + db_hi)


def _bw_logistic(eqn, out_bd, env, **_):
    """Backward through sigmoid. α, β_l, β_u match _linbp_logistic_p."""
    (var,) = eqn.invars
    lb_in = env[var]
    l, u = lb_in.l, lb_in.u
    sig_l, sig_u = jax.nn.sigmoid(l), jax.nn.sigmoid(u)
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, sig_l * (1.0 - sig_l), (sig_u - sig_l) / safe_denom)
    beta_chord = sig_l - alpha * l

    one_minus_4alpha = jnp.maximum(1.0 - 4.0 * alpha, 0.0)
    safe_disc = jnp.where(one_minus_4alpha <= 0.0, jnp.ones_like(one_minus_4alpha), one_minus_4alpha)
    disc = jnp.where(one_minus_4alpha <= 0.0, jnp.zeros_like(one_minus_4alpha), jnp.sqrt(safe_disc))

    sig_xu = (1.0 + disc) / 2.0
    x_upper = jnp.log(sig_xu) - jnp.log(1.0 - sig_xu)
    beta_u_crit = sig_xu - alpha * x_upper

    sig_xl = (1.0 - disc) / 2.0
    x_lower = jnp.log(sig_xl) - jnp.log(1.0 - sig_xl)
    beta_l_crit = sig_xl - alpha * x_lower

    beta_u = jnp.where(x_upper <= u, beta_u_crit, beta_chord)
    beta_l = jnp.where(x_lower >= l, beta_l_crit, beta_chord)
    return [_bw_monotone_positive_slope(out_bd, alpha, beta_l, beta_u)]


def _bw_tanh(eqn, out_bd, env, **_):
    """Backward through tanh. α, β_l, β_u match _linbp_tanh_p."""
    (var,) = eqn.invars
    lb_in = env[var]
    l, u = lb_in.l, lb_in.u
    tanh_l, tanh_u = jnp.tanh(l), jnp.tanh(u)
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, 1.0 - tanh_l ** 2, (tanh_u - tanh_l) / safe_denom)
    chord_beta = tanh_l - alpha * l

    one_minus_alpha = jnp.maximum(1.0 - alpha, 0.0)
    safe_oma = jnp.where(one_minus_alpha <= 0.0, jnp.ones_like(one_minus_alpha), one_minus_alpha)
    tanh_xu = jnp.where(one_minus_alpha <= 0.0, jnp.zeros_like(one_minus_alpha), jnp.sqrt(safe_oma))
    x_upper = jnp.arctanh(jnp.clip(tanh_xu, 0.0, 1.0 - 1e-7))
    beta_u_crit = tanh_xu - alpha * x_upper
    x_lower = -x_upper
    beta_l_crit = -tanh_xu - alpha * x_lower

    beta_u = jnp.where(x_upper <= u, beta_u_crit, chord_beta)
    beta_l = jnp.where(x_lower >= l, beta_l_crit, chord_beta)
    return [_bw_monotone_positive_slope(out_bd, alpha, beta_l, beta_u)]


backward_registry[lax.logistic_p] = _bw_logistic
backward_registry[lax.tanh_p] = _bw_tanh


def _bw_jit(eqn, out_bds_out, env, *, relu_mode):
    """Recurse into jit_p call_jaxpr."""
    bind_params = dict(eqn.params)
    inner = bind_params["jaxpr"]
    if isinstance(inner, jax.extend.core.ClosedJaxpr):
        inner_consts = inner.consts
        inner_jaxpr = inner.jaxpr
    else:
        inner_consts = []
        inner_jaxpr = inner
    # Map outer outvars' BackwardBounds to inner outvars (parallel).
    # Run a forward linbp on the inner jaxpr to populate an inner env, then
    # backward-walk it.
    args = [env[v] if isinstance(env.get(v), LinearBound) else
            (v.val if isinstance(v, Literal) else env.get(v))
            for v in eqn.invars]
    _, inner_env = _linbp_jaxpr(
        inner_jaxpr, inner_consts, *args, relu_mode=relu_mode,
        tighten_bounds=True, return_env=True,
    )
    inner_bw = {}
    for ov_outer, ov_inner in zip(eqn.outvars, inner_jaxpr.outvars):
        inner_bw[ov_inner] = out_bds_out[ov_outer]
    _bw_walk(inner_jaxpr, inner_env, inner_bw, relu_mode=relu_mode)
    # Map back: invars
    return [inner_bw.get(iv) for iv in inner_jaxpr.invars]


def _bw_custom_jvp_call(eqn, out_bds_out, env, *, relu_mode):
    """Recurse into custom_jvp_call primal jaxpr (ignore jvp branch)."""
    bind_params = dict(eqn.params)
    call_jaxpr = bind_params["call_jaxpr"]
    num_consts = bind_params.get("num_consts", 0)
    if isinstance(call_jaxpr, jax.extend.core.ClosedJaxpr):
        inner_consts = list(call_jaxpr.consts)
        inner_jaxpr = call_jaxpr.jaxpr
    else:
        inner_consts = []
        inner_jaxpr = call_jaxpr
    extra_consts = []
    actual_invars = eqn.invars
    if num_consts:
        extra_consts = [env.get(v) if env.get(v) is not None else (v.val if isinstance(v, Literal) else None) for v in eqn.invars[:num_consts]]
        actual_invars = eqn.invars[num_consts:]
    args = [env[v] if isinstance(env.get(v), LinearBound) else
            (v.val if isinstance(v, Literal) else env.get(v))
            for v in actual_invars]
    _, inner_env = _linbp_jaxpr(
        inner_jaxpr, inner_consts + extra_consts, *args, relu_mode=relu_mode,
        tighten_bounds=True, return_env=True,
    )
    inner_bw = {}
    for ov_outer, ov_inner in zip(eqn.outvars, inner_jaxpr.outvars):
        inner_bw[ov_inner] = out_bds_out[ov_outer]
    _bw_walk(inner_jaxpr, inner_env, inner_bw, relu_mode=relu_mode)
    out = [None] * len(eqn.invars)
    for j, iv_inner in enumerate(inner_jaxpr.invars):
        out[num_consts + j] = inner_bw.get(iv_inner)
    return out


if _jit_primitive is not None:
    backward_registry[_jit_primitive] = _bw_jit
    _bw_recursive_prims.add(_jit_primitive)
if _custom_jvp_call_p is not None:
    backward_registry[_custom_jvp_call_p] = _bw_custom_jvp_call
    _bw_recursive_prims.add(_custom_jvp_call_p)


def _bw_walk(jaxpr: Jaxpr, env: dict, bw_env: dict, *, relu_mode: str):
    """Walk jaxpr.eqns in reverse, propagating BackwardBounds through bw_env."""
    for eqn in reversed(jaxpr.eqns):
        # Collect output bounds for this eqn's outvars
        if not any(ov in bw_env for ov in eqn.outvars):
            continue
        # For single-output primitives (typical), grab the one bound
        handler = backward_registry.get(eqn.primitive)
        if handler is None:
            raise NotImplementedError(
                f"backward CROWN: no rule for primitive {eqn.primitive}"
            )
        if eqn.primitive in _bw_recursive_prims:
            out_bds_map = {ov: bw_env[ov] for ov in eqn.outvars if ov in bw_env}
            in_bds = handler(eqn, out_bds_map, env, relu_mode=relu_mode)
        else:
            out_bd = bw_env[eqn.outvars[0]]
            in_bds = handler(eqn, out_bd, env, relu_mode=relu_mode)
        for v, bd in zip(eqn.invars, in_bds):
            if bd is None:
                continue
            if v in bw_env:
                # Accumulate (multi-use var)
                prev = bw_env[v]
                bw_env[v] = BackwardBound(
                    prev.A_lo + bd.A_lo, prev.b_lo + bd.b_lo,
                    prev.A_hi + bd.A_hi, prev.b_hi + bd.b_hi,
                )
            else:
                bw_env[v] = bd


def _backward_to_concrete(jaxpr, env, target_var, target_idx, x_lb, x_ub, relu_mode):
    """Run a backward CROWN sweep treating ``target_var`` as the output.

    Walks ``jaxpr.eqns`` from ``target_idx`` down to 0, using the slopes
    currently stored in ``env`` (so iterated tightening cascades through
    earlier activations).  Concretizes at ``[x_lb, x_ub]`` and returns
    ``(l, u)`` matching ``target_var.aval.shape``.
    """
    import math
    S = tuple(target_var.aval.shape)
    n_out = math.prod(S) if S else 1
    I = jnp.eye(n_out).reshape((n_out,) + S)
    bw_env = {target_var: BackwardBound(I, jnp.zeros(n_out), I, jnp.zeros(n_out))}

    for i in range(target_idx, -1, -1):
        eqn = jaxpr.eqns[i]
        if not any(ov in bw_env for ov in eqn.outvars):
            continue
        handler = backward_registry.get(eqn.primitive)
        if handler is None:
            raise NotImplementedError(
                f"backward CROWN: no rule for primitive {eqn.primitive}"
            )
        if eqn.primitive in _bw_recursive_prims:
            out_bds_map = {ov: bw_env[ov] for ov in eqn.outvars if ov in bw_env}
            in_bds = handler(eqn, out_bds_map, env, relu_mode=relu_mode)
        else:
            out_bd = bw_env[eqn.outvars[0]]
            in_bds = handler(eqn, out_bd, env, relu_mode=relu_mode)
        for v, bd in zip(eqn.invars, in_bds):
            if bd is None:
                continue
            if v in bw_env:
                prev = bw_env[v]
                bw_env[v] = BackwardBound(
                    prev.A_lo + bd.A_lo, prev.b_lo + bd.b_lo,
                    prev.A_hi + bd.A_hi, prev.b_hi + bd.b_hi,
                )
            else:
                bw_env[v] = bd

    if len(jaxpr.invars) != 1:
        # Multi-input not supported here; fall back to forward-computed bounds.
        return env[target_var].l, env[target_var].u
    inp_var = jaxpr.invars[0]
    bd = bw_env.get(inp_var)
    if bd is None:
        return env[target_var].l, env[target_var].u
    Alo_p = jnp.clip(bd.A_lo, 0, None); Alo_n = jnp.clip(bd.A_lo, None, 0)
    Ahi_p = jnp.clip(bd.A_hi, 0, None); Ahi_n = jnp.clip(bd.A_hi, None, 0)
    l_flat = _bw_concrete(Alo_p, Alo_n, x_lb, x_ub) + bd.b_lo
    u_flat = _bw_concrete(Ahi_p, Ahi_n, x_ub, x_lb) + bd.b_hi
    return l_flat.reshape(S), u_flat.reshape(S)


def _linbp_backward_jaxpr(
    jaxpr: Jaxpr, consts, *args,
    relu_mode: str = "adaptive",
    tighten_bounds: bool = True,
    x_lb=None, x_ub=None,
    iterated_bw: bool = False,
):
    """Forward linbp to capture env, then backward sweep with sign-conditioned slopes.

    With ``iterated_bw=True``, the forward sweep additionally runs a backward
    CROWN pass at each activation / wrapper to tighten the pre-activation
    ``l, u`` before the slope is picked (pure backward CROWN).

    Returns a list of BackwardBound, one per jaxpr.outvar, expressed as a linear
    function of the input variable (jaxpr.invars[0]).
    """
    outs, env = _linbp_jaxpr(
        jaxpr, consts, *args, relu_mode=relu_mode,
        tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub,
        return_env=True, iterated_bw=iterated_bw,
    )
    bw_env = {}
    import math
    for ov in jaxpr.outvars:
        S = tuple(ov.aval.shape)
        n_out = math.prod(S) if S else 1
        # Initialise with identity coefficient: A has shape (n_out, *S)
        I = jnp.eye(n_out).reshape((n_out,) + S)
        bw_env[ov] = BackwardBound(I, jnp.zeros(n_out), I, jnp.zeros(n_out))
    _bw_walk(jaxpr, env, bw_env, relu_mode=relu_mode)
    return [bw_env[v] for v in jaxpr.invars]


def linbp_backward(f, relu_mode: str = "adaptive", tighten_bounds: bool = True,
                   iterated_bw: bool = False):
    """Backward CROWN: like ``linbp`` but with sign-conditioned per-(output,
    neuron) slope picking at each ReLU.

    Returns a callable mapping an ``Interval`` to a ``LinearBound`` shaped to
    match ``crown()``'s expectation (lA/uA = (n_out, n_in), lb/ub = (n_out,)).
    """
    def wrapped(inp) -> LinearBound:
        lb_init, trace_point, x_lb, x_ub = _resolve_linbp_input(inp)
        closed_jaxpr = eqx.filter_make_jaxpr(f)(trace_point)[0]
        in_bds = _linbp_backward_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, lb_init,
            relu_mode=relu_mode, tighten_bounds=tighten_bounds,
            x_lb=x_lb, x_ub=x_ub, iterated_bw=iterated_bw,
        )
        # We assume a single input variable.
        if len(in_bds) != 1:
            raise NotImplementedError("backward CROWN: multi-input functions not supported yet")
        bd = in_bds[0]
        # The function input was the identity LinearBound (lb_init); so the
        # backward coefficient at the input variable directly gives the linear
        # bound on the output with respect to x_in.  Concretize l, u using IBP
        # outputs from the forward pass for the concrete bounds.
        # We need to evaluate A @ x + b at [x_lb, x_ub] for l, u.
        if x_lb is None:
            l = jnp.full(bd.b_lo.shape, -jnp.inf)
            u = jnp.full(bd.b_hi.shape, jnp.inf)
        else:
            Alo_p = jnp.clip(bd.A_lo, 0, None); Alo_n = jnp.clip(bd.A_lo, None, 0)
            Ahi_p = jnp.clip(bd.A_hi, 0, None); Ahi_n = jnp.clip(bd.A_hi, None, 0)
            l = _bw_concrete(Alo_p, Alo_n, x_lb, x_ub) + bd.b_lo
            u = _bw_concrete(Ahi_p, Ahi_n, x_ub, x_lb) + bd.b_hi
        return LinearBound(lA=bd.A_lo, lb=bd.b_lo, uA=bd.A_hi, ub=bd.b_hi, l=l, u=u)

    return wrapped


def _bw_concrete(Ap, An, x_for_pos, x_for_neg):
    """Concretize: sum over var dims of (Ap * x_for_pos + An * x_for_neg)."""
    n_out = Ap.shape[0]
    return jnp.tensordot(Ap, x_for_pos, axes=Ap.ndim - 1) + \
           jnp.tensordot(An, x_for_neg, axes=An.ndim - 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_linbp_input(inp):
    """Normalize a linbp input to (lb_init, trace_point, x_lb, x_ub).

    Parameters
    ----------
    inp : Interval or LinearBound
        - Interval: converted to an identity LinearBound that represents
          y = x_in exactly.  x_lb/x_ub are set so tighten_bounds works.
        - LinearBound: passed through as-is.  x_lb/x_ub are None, which
          disables tighten_bounds (the original input interval is unknown).

    Returns
    -------
    lb_init : LinearBound
    trace_point : jax.Array
        A concrete array with the shape of f's input, used to trace the Jaxpr.
    x_lb, x_ub : jax.Array or None
        Original input box corners for bound tightening.
    """
    if isinstance(inp, Interval):
        n_in = inp.lower.size
        lb_init = LinearBound(
            lA=jnp.eye(n_in),
            lb=jnp.zeros(n_in),
            uA=jnp.eye(n_in),
            ub=jnp.zeros(n_in),
            l=inp.lower,
            u=inp.upper,
        )
        return lb_init, inp.lower, inp.lower, inp.upper
    elif isinstance(inp, LinearBound):
        # inp.l has the shape of the function's input variable; use it to
        # trace the Jaxpr.  x_lb/x_ub are unknown so tightening is skipped.
        return inp, inp.l, None, None
    else:
        raise TypeError(f"linbp wrapped function expects Interval or LinearBound, got {type(inp)}")


def linbp(f, relu_mode: str = "adaptive", tighten_bounds: bool = True):
    """Function transformation: forward linear bound propagation through f.

    Returns a function that maps an Interval or LinearBound to a LinearBound.

    Parameters
    ----------
    f : callable
        Function to propagate through (e.g. a NeuralNetwork).
    relu_mode : str
        How to relax active (ambiguous) neurons. One of:
        'same-slope' — lower slope parallel to upper, tight at kink
        'adaptive'   — slope 0 or 1 chosen by area heuristic
        'zero'       — lower slope always 0 (constant bound)
        'one'        — lower slope always 1 (identity bound)
    tighten_bounds : bool
        If True (default), after each linear layer the concrete bounds l, u are
        intersected with the affine-evaluated bounds: l ← max(l, lA·x̲ + lb),
        u ← min(u, uA·x̄ + ub).  This gives tighter neuron-status classification
        at subsequent activations.  Only applied when the input is an Interval
        (when the input is a LinearBound the original box is unknown).

    Returns
    -------
    Callable[[Interval | LinearBound], LinearBound]
        Maps an input Interval or LinearBound to a LinearBound representing
        affine bounds: lA @ x + lb <= f(x) <= uA @ x + ub for all x in ix.
    """

    def wrapped(inp) -> LinearBound:
        lb_init, trace_point, x_lb, x_ub = _resolve_linbp_input(inp)
        closed_jaxpr = eqx.filter_make_jaxpr(f)(trace_point)[0]
        outs = _linbp_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, lb_init,
            relu_mode=relu_mode, tighten_bounds=tighten_bounds,
            x_lb=x_lb, x_ub=x_ub,
        )
        return outs[0] if len(outs) == 1 else outs

    return wrapped
