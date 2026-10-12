from typing import Any

import equinox as eqx
import functools
from collections import namedtuple

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
from immrax.inclusion._jax_compat import split_bind_params, jit_primitive
from immrax.inclusion.nif import (
    _inclusion_dot_general_p,
    inclusion_registry,
    _set_rigorous,
)

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


def _degenerate_threshold(arr):
    """Width below which (f(u) - f(l)) / (u - l) suffers catastrophic cancellation.

    For float32 this is ~3.4e-4; for float64 ~1.5e-8. Above this width the chord
    slope is numerically reliable; at or below it we fall back to f'(l) (the
    tangent slope at l, which equals the chord slope in the limit u → l).
    """
    return jnp.sqrt(jnp.finfo(arr.dtype).eps)


linbp_registry = {}

# Primitives that recurse into inner jaxprs — these need tighten_bounds/x_lb/x_ub
# forwarded so they can pass them to nested _linbp_jaxpr calls.
_linbp_recursive_prims: set = set()

# Primitives whose concrete bounds tighten by evaluating the affine bound on the
# input box: only dot_general (IBP treats output neurons independently; the
# affine bound keeps their shared input). Activations and shape ops gain nothing.
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
            a.lA,
            a.lb,
            a.uA,
            a.ub,
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
                if (
                    iterated_bw
                    and x_lb is not None
                    and (
                        eqn.primitive in _activation_prims_bw
                        or eqn.primitive in _linbp_recursive_prims
                    )
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
                            val.lA,
                            val.lb,
                            val.uA,
                            val.ub,
                            jnp.maximum(val.l, l_new),
                            jnp.minimum(val.u, u_new),
                        )
                    invars = safe_map(read, eqn.invars)
                # Pass eqn.params directly (avoids split_bind_params moving things like
                # call_jaxpr/num_consts out of kwargs and into subfuns).
                # Only recursive primitives (jit_p, custom_jvp_call_p) need
                # tighten_bounds/x_lb/x_ub — they forward them to nested jaxpr calls.
                if eqn.primitive in _linbp_recursive_prims:
                    extra = dict(
                        tighten_bounds=tighten_bounds,
                        x_lb=x_lb,
                        x_ub=x_ub,
                        iterated_bw=iterated_bw,
                    )
                elif eqn.primitive in _linbp_tighten_prims:
                    # Tightening-eligible primitives need to know whether _tighten
                    # will overwrite their l/u so they can skip IBP propagation.
                    extra = dict(
                        _tighten_will_run=(tighten_bounds and x_lb is not None)
                    )
                else:
                    extra = {}
                ans = linbp_registry[eqn.primitive](
                    *invars,
                    relu_mode=relu_mode,
                    **extra,
                    **eqn.params,
                )
                if tighten_bounds and eqn.primitive in _linbp_tighten_prims:
                    if eqn.primitive.multiple_results:
                        ans = [_tighten(a) for a in ans]
                    else:
                        ans = _tighten(ans)
            else:
                subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)
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


def _ibp_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    relu_mode: str = "adaptive",
    propagate_source_info: bool = True,
    x_lb=None,
    x_ub=None,
    return_env: bool = False,
    iterated_bw: bool = False,
) -> list[Any]:
    """Pure interval-arithmetic (IBP) forward sweep.

    Walks ``jaxpr.eqns`` exactly like :func:`natif_jaxpr`, reusing the same
    per-primitive handlers in ``inclusion_registry``.  The only differences:

    1. Env entries are stored as ``LinearBound(lA=None, lb=None, uA=None,
       ub=None, l=..., u=...)`` so the downstream backward CROWN sweep can
       read ``env[var].l`` / ``env[var].u`` uniformly.
    2. Dead-var cleanup is suppressed so the env is intact for the backward
       walk.
    3. With ``iterated_bw=True``, at each activation / recursive-wrap eqn we
       run :func:`_backward_to_concrete` over the already-processed prefix
       to tighten the pre-activation ``l, u`` — yielding standard backward
       CROWN behaviour with no linear-bound forward propagation.

    Rigorous widening is disabled (handlers are evaluated in non-rigorous
    mode); linbp_backward does not produce ULP-widened bounds.
    """

    def _wrap(iv):
        return LinearBound(lA=None, lb=None, uA=None, ub=None, l=iv.lower, u=iv.upper)

    def _as_handler_arg(v):
        # Env entries are LinearBound with .l/.u; convert to Interval for handlers.
        if isinstance(v, LinearBound):
            return Interval(v.l, v.u)
        return v

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)

    # Inputs may be LinearBound (from _linbp_backward_jaxpr's identity init) or
    # Interval; either way the env entry needs .l/.u, so re-wrap uniformly.
    def _ingest(a):
        if isinstance(a, LinearBound):
            return LinearBound(lA=None, lb=None, uA=None, ub=None, l=a.l, u=a.u)
        if isinstance(a, Interval):
            return _wrap(a)
        return a

    safe_map(write, jaxpr.invars, [_ingest(a) for a in args])

    # For iterated CROWN: map each var to the index of the eqn that produced it.
    producing_idx: dict = {}
    if iterated_bw:
        for j, e in enumerate(jaxpr.eqns):
            for ov in e.outvars:
                producing_idx[ov] = j

    old_rigorous = _set_rigorous(False)
    try:
        for eqn in jaxpr.eqns:
            name_stack = (
                source_info_util.current_name_stack() + eqn.source_info.name_stack
            )
            traceback = eqn.source_info.traceback if propagate_source_info else None
            with source_info_util.user_context(traceback, name_stack=name_stack):
                # iterated CROWN: at each activation / recursive primitive, replace
                # the forward-IBP pre-activation l, u with the tighter bounds from
                # a backward CROWN sweep over the prefix processed so far.
                if (
                    iterated_bw
                    and x_lb is not None
                    and (
                        eqn.primitive in _activation_prims_bw
                        or eqn.primitive in _linbp_recursive_prims
                    )
                ):
                    for iv in eqn.invars:
                        if isinstance(iv, Literal):
                            continue
                        val = env.get(iv)
                        if not isinstance(val, LinearBound):
                            continue
                        prod_idx = producing_idx.get(iv)
                        if prod_idx is None:
                            continue
                        l_new, u_new = _backward_to_concrete(
                            jaxpr,
                            env,
                            iv,
                            prod_idx,
                            x_lb,
                            x_ub,
                            relu_mode,
                            forward_mode="ibp",
                        )
                        env[iv] = LinearBound(
                            None,
                            None,
                            None,
                            None,
                            jnp.maximum(val.l, l_new),
                            jnp.minimum(val.u, u_new),
                        )

                raw_invars = safe_map(read, eqn.invars)
                handler_invars = [_as_handler_arg(v) for v in raw_invars]
                any_interval = any(isinstance(v, Interval) for v in handler_invars)

                if any_interval:
                    if eqn.primitive in _linbp_recursive_prims:
                        # Recurse into nested jaxprs (jit_p, custom_jvp_call_p)
                        # using _ibp_jaxpr so the inner sweep stays IBP.
                        ans = _ibp_recursive(
                            eqn,
                            handler_invars,
                            relu_mode=relu_mode,
                            x_lb=x_lb,
                            x_ub=x_ub,
                            iterated_bw=iterated_bw,
                        )
                    else:
                        if eqn.primitive not in inclusion_registry:
                            raise NotImplementedError(
                                f"{eqn.primitive} not in inclusion_registry"
                            )
                        subfuns, bind_params = split_bind_params(
                            eqn.primitive, eqn.params
                        )
                        handler = inclusion_registry[eqn.primitive]
                        ans = handler(*subfuns, *handler_invars, **bind_params)
                else:
                    subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)
                    ans = eqn.primitive.bind(*subfuns, *handler_invars, **bind_params)

            if eqn.primitive.multiple_results:
                wrapped = [_wrap(a) if isinstance(a, Interval) else a for a in ans]
                safe_map(write, eqn.outvars, wrapped)
            else:
                wrapped = _wrap(ans) if isinstance(ans, Interval) else ans
                write(eqn.outvars[0], wrapped)
            # NOTE: no clean_up_dead_vars — env is needed for the backward sweep.
    finally:
        _set_rigorous(old_rigorous)

    outs = safe_map(read, jaxpr.outvars)
    if return_env:
        return outs, env
    return outs


def _ibp_recursive(eqn, handler_invars, *, relu_mode, x_lb, x_ub, iterated_bw):
    """Recurse _ibp_jaxpr into a nested jaxpr (jit_p, custom_jvp_call_p)."""
    params = dict(eqn.params)
    # jit_p / pjit_p carry the inner jaxpr under "jaxpr"
    if "jaxpr" in params:
        inner = params["jaxpr"]
        if isinstance(inner, jax.extend.core.ClosedJaxpr):
            inner_jaxpr = inner.jaxpr
            inner_consts = inner.literals
        else:
            inner_jaxpr = inner
            inner_consts = []
        outs = _ibp_jaxpr(
            inner_jaxpr,
            inner_consts,
            *handler_invars,
            relu_mode=relu_mode,
            x_lb=x_lb,
            x_ub=x_ub,
            iterated_bw=iterated_bw,
        )
    elif "call_jaxpr" in params:
        # custom_jvp_call_p
        num_consts = params.get("num_consts", 0)
        inner = params["call_jaxpr"]
        if isinstance(inner, jax.extend.core.ClosedJaxpr):
            inner_jaxpr = inner.jaxpr
            inner_consts = list(inner.consts)
        else:
            inner_jaxpr = inner
            inner_consts = []
        extra_consts = list(handler_invars[:num_consts])
        actual_args = handler_invars[num_consts:]
        outs = _ibp_jaxpr(
            inner_jaxpr,
            inner_consts + extra_consts,
            *actual_args,
            relu_mode=relu_mode,
            x_lb=x_lb,
            x_ub=x_ub,
            iterated_bw=iterated_bw,
        )
    else:
        raise NotImplementedError(
            f"_ibp_recursive: don't know how to unpack {eqn.primitive}"
        )
    # _ibp_jaxpr returns LinearBound entries; convert outputs back to Interval
    # so the caller can wrap them uniformly.
    if eqn.primitive.multiple_results:
        return [Interval(o.l, o.u) if isinstance(o, LinearBound) else o for o in outs]
    o = outs[0] if isinstance(outs, list) else outs
    return Interval(o.l, o.u) if isinstance(o, LinearBound) else o


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


def _linbp_dot_general_p(
    x, y, *, relu_mode, dimension_numbers, _tighten_will_run=False, **kwargs
):
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
        if _tighten_will_run:
            ll = jnp.full(lb.shape, -jnp.inf, dtype=lb.dtype)
            ul = jnp.full(ub.shape, jnp.inf, dtype=ub.dtype)
        else:
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
        if _tighten_will_run:
            ll = jnp.full(lb.shape, -jnp.inf, dtype=lb.dtype)
            ul = jnp.full(ub.shape, jnp.inf, dtype=ub.dtype)
        else:
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

        l = primitive.bind(*[_data(a, "l") for a in args], **kwargs)
        u = primitive.bind(*[_data(a, "u") for a in args], **kwargs)
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
                for lAi, lbi, uAi, ubi, li, ui in zip(lA_parts, lb, uA_parts, ub, l, u)
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
if hasattr(lax, "stack_p"):  # jax >= 0.10
    _add_structural_to_linbp_registry(lax.stack_p)
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

        l = _scat(_d(operand, "l"), _d(updates, "l"))
        u = _scat(_d(operand, "u"), _d(updates, "u"))
        lb = _scat(_d(operand, "lb"), _d(updates, "lb"))
        ub = _scat(_d(operand, "ub"), _d(updates, "ub"))

        def _apply_A(f):
            return jax.vmap(
                lambda op_k, upd_k: primitive.bind(
                    op_k, scatter_indices, upd_k, **kwargs
                ),
                in_axes=(-1, -1),
                out_axes=-1,
            )(_A(operand, f), _A(updates, f))

        return LinearBound(lA=_apply_A("lA"), lb=lb, uA=_apply_A("uA"), ub=ub, l=l, u=u)

    return handler


for _p_name in ["scatter_p", "scatter_add_p", "scatter_mul_p"]:
    if hasattr(lax, _p_name):
        linbp_registry[getattr(lax, _p_name)] = _make_linbp_scatter_p(
            getattr(lax, _p_name)
        )


def _linbp_reduce_sum_p(x, *, relu_mode, axes, out_sharding=None, **kwargs):
    """reduce_sum is linear: A-matrices reduce exactly over the same axes."""
    if not isinstance(x, LinearBound):
        return lax.reduce_sum_p.bind(x, axes=axes, out_sharding=out_sharding)
    # out_sharding is sized for the primal output rank; lA/uA have one extra
    # trailing n_in dimension, so omit it (pass None) for them to avoid a rank
    # mismatch in JAX's sharding system.  In JAX >= 0.9.2 the kwarg is
    # required by reduce_sum's ur rule, so always pass it explicitly.
    _rs = lambda t: lax.reduce_sum_p.bind(t, axes=axes, out_sharding=out_sharding)
    _rs_A = lambda t: lax.reduce_sum_p.bind(t, axes=axes, out_sharding=None)
    return LinearBound(
        lA=_rs_A(x.lA),
        lb=_rs(x.lb),
        uA=_rs_A(x.uA),
        ub=_rs(x.ub),
        l=_rs(x.l),
        u=_rs(x.u),
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

    l = _sel(*[_d(c, "l") for c in cases])
    u = _sel(*[_d(c, "u") for c in cases])
    lb = _sel(*[_d(c, "lb") for c in cases])
    ub = _sel(*[_d(c, "ub") for c in cases])

    def _apply_A(f):
        cases_A = [_A(c, f) for c in cases]
        return jax.vmap(
            lambda *cs_k: lax.select_n_p.bind(which, *cs_k, **kwargs),
            in_axes=(-1,) * len(cases),
            out_axes=-1,
        )(*cases_A)

    return LinearBound(lA=_apply_A("lA"), lb=lb, uA=_apply_A("uA"), ub=ub, l=l, u=u)


if hasattr(lax, "select_n_p"):
    linbp_registry[lax.select_n_p] = _linbp_select_n_p


def _read(env, v):
    """Value of jaxpr variable ``v`` in ``env`` (literals carry their own value)."""
    return v.val if isinstance(v, Literal) else env.get(v)


def _max_relax(l, u, c, relu_mode):
    """``a_l x + b_l <= max(x, c) <= a_u x + b_u`` on ``[l, u]``: chord above; below,
    the slope picked by ``relu_mode`` (``adaptive``: 0 or 1, ``same-slope``: the chord)."""
    on, off = l >= c, u <= c
    active = ~on & ~off
    chord = (u - c) / jnp.where(active, u - l, 1.0)
    if relu_mode == "same-slope":
        a_act, b_act = chord, c * (1.0 - chord)
    elif relu_mode == "adaptive":
        use_id = l + u >= 2.0 * c
        a_act, b_act = jnp.where(use_id, 1.0, 0.0), jnp.where(use_id, 0.0, c)
    elif relu_mode == "zero":
        a_act, b_act = (
            jnp.zeros_like(chord),
            jnp.broadcast_to(c, chord.shape).astype(chord.dtype),
        )
    elif relu_mode == "one":
        a_act, b_act = jnp.ones_like(chord), jnp.zeros_like(chord)
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")
    pick = lambda v_on, v_off, v_act: jnp.where(on, v_on, jnp.where(off, v_off, v_act))
    return (
        pick(1.0, 0.0, a_act),
        pick(0.0, c, b_act),
        pick(1.0, 0.0, chord),
        pick(0.0, c, c - chord * l),
    )


def _max_min_relax(l, u, c, relu_mode, sign):
    """``_max_relax`` for ``max`` (``sign=1``) or ``min(x, c) = -max(-x, -c)``."""
    if sign > 0:
        return _max_relax(l, u, c, relu_mode)
    a_l, b_l, a_u, b_u = _max_relax(-u, -l, -c, relu_mode)
    return a_u, -b_u, a_l, -b_l


def _linbp_max_min(sign):
    """Forward rule for ``max(x, c)`` / ``min(x, c)``; both slopes are ``>= 0``."""
    op = jnp.maximum if sign > 0 else jnp.minimum

    def rule(x, y, *, relu_mode, **kwargs):
        if isinstance(x, LinearBound) == isinstance(
            y, LinearBound
        ):  # two bounds: interval fallback
            l, u = op(x.l, y.l), op(x.u, y.u)
            zero = jnp.zeros((*l.shape, x.n_in))
            return LinearBound(lA=zero, lb=l, uA=zero, ub=u, l=l, u=u)
        x, c = (
            (x, jnp.asarray(y)) if isinstance(x, LinearBound) else (y, jnp.asarray(x))
        )
        a_l, b_l, a_u, b_u = _max_min_relax(x.l, x.u, c, relu_mode, sign)
        return LinearBound(
            lA=a_l[..., None] * x.lA,
            lb=a_l * x.lb + b_l,
            uA=a_u[..., None] * x.uA,
            ub=a_u * x.ub + b_u,
            l=op(x.l, c),
            u=op(x.u, c),
        )

    return rule


linbp_registry[lax.max_p] = _linbp_max_min(1)
linbp_registry[lax.min_p] = _linbp_max_min(-1)


# ---------------------------------------------------------------------------
# Scalar activations: one affine relaxation per primitive, shared by the
# forward (linbp) and backward (CROWN) passes.
#   relax(l, u)    -> (alpha, beta_l, beta_u):  alpha x + beta_l <= f(x) <= alpha x + beta_u  on [l, u]
#   concrete(l, u) -> (lo, hi):                 lo <= f(x) <= hi                              on [l, u]
# ---------------------------------------------------------------------------


def _safe_sqrt(v):
    """sqrt(max(v, 0)) with a finite gradient at 0."""
    pos = v > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, v, 1.0)), 0.0)


def _chord(f, df, l, u):
    """Chord slope of ``f`` on ``[l, u]``; the tangent ``f'(l)`` when ``u - l`` is tiny."""
    degenerate = jnp.abs(u - l) < _degenerate_threshold(l)
    alpha = jnp.where(
        degenerate, df(l), (f(u) - f(l)) / jnp.where(degenerate, 1.0, u - l)
    )
    return alpha, degenerate


def _relax_s_curve(f, df, finv_of_slope, l, u):
    """S-shaped ``f`` (convex for x < 0, concave for x > 0): chord slope, intercepts at the
    tangent points ``x* = +-finv_of_slope(alpha)`` when they fall in ``[l, u]``."""
    alpha, _ = _chord(f, df, l, u)
    chord = f(l) - alpha * l
    xs = finv_of_slope(alpha)  # >= 0, f'(xs) = alpha
    beta_u = jnp.where(xs <= u, f(xs) - alpha * xs, chord)
    beta_l = jnp.where(-xs >= l, f(-xs) + alpha * xs, chord)
    return alpha, beta_l, beta_u


def _relax_logistic(l, u, **_):
    sig = jax.nn.sigmoid
    # sigma'(x) = a  <=>  sigma(x) = (1 + sqrt(1 - 4a)) / 2  (x >= 0 branch)
    inv = lambda a: jax.scipy.special.logit((1.0 + _safe_sqrt(1.0 - 4.0 * a)) / 2.0)
    return _relax_s_curve(sig, lambda x: sig(x) * (1.0 - sig(x)), inv, l, u)


def _relax_tanh(l, u, **_):
    inv = lambda a: jnp.arctanh(jnp.clip(_safe_sqrt(1.0 - a), 0.0, 1.0 - 1e-7))
    return _relax_s_curve(jnp.tanh, lambda x: 1.0 - jnp.tanh(x) ** 2, inv, l, u)


def _relax_tan(l, u, **_):
    # tan is concave for x < 0, convex for x > 0 (within a period): the mirror of
    # an S-curve, so the tangent points swap roles.
    alpha, _ = _chord(jnp.tan, lambda x: 1.0 + jnp.tan(x) ** 2, l, u)
    chord = jnp.tan(l) - alpha * l
    xs = jnp.arctan(_safe_sqrt(alpha - 1.0))  # sec^2(xs) = alpha
    beta_u = jnp.where(-xs >= l, -jnp.tan(xs) + alpha * xs, chord)
    beta_l = jnp.where(xs <= u, jnp.tan(xs) - alpha * xs, chord)
    return alpha, beta_l, beta_u


def _relax_convex(f, df, inv_df, l, u):
    """Convex ``f``: chord above, the parallel tangent at ``inv_df(alpha)`` below."""
    alpha, degenerate = _chord(f, df, l, u)
    x0 = jnp.where(degenerate, l, inv_df(alpha))
    return alpha, f(x0) - alpha * x0, f(l) - alpha * l


def _relax_concave(f, df, inv_df, l, u):
    """Concave ``f``: chord below, the parallel tangent at ``inv_df(alpha)`` above."""
    alpha, degenerate = _chord(f, df, l, u)
    x0 = jnp.where(degenerate, l, inv_df(alpha))
    return alpha, f(l) - alpha * l, f(x0) - alpha * x0


def _relax_exp(l, u, **_):
    return _relax_convex(
        jnp.exp, jnp.exp, lambda a: jnp.log(jnp.maximum(a, 1e-30)), l, u
    )


def _relax_log1p(l, u, **_):
    return _relax_concave(
        jnp.log1p,
        lambda x: 1.0 / (1.0 + x),
        lambda a: 1.0 / jnp.maximum(a, 1e-30) - 1.0,
        l,
        u,
    )


def _relax_sqrt(l, u, **_):
    l, u = jnp.maximum(l, 0.0), jnp.maximum(u, 0.0)
    df = lambda x: 0.5 / jnp.maximum(jnp.sqrt(x), 1e-15)
    return _relax_concave(
        jnp.sqrt, df, lambda a: 0.25 / jnp.maximum(a, 1e-30) ** 2, l, u
    )


def _relax_periodic(f, df, x_hi, x_lo, l, u):
    """sin / cos: chord slope, intercepts at the parallel tangents ``x_hi(alpha)``
    (where f - alpha x peaks) and ``x_lo(alpha)`` shifted to the period nearest the
    midpoint; intervals of width >= 2 pi fall back to ``alpha = 0, beta = -+1``."""
    wide = (u - l) >= 2 * jnp.pi
    alpha, _ = _chord(f, df, l, jnp.where(wide, l + 1.0, u))
    alpha = jnp.where(wide, 0.0, alpha)
    chord, mid, s = f(l) - alpha * l, (l + u) / 2, _safe_sqrt(1.0 - alpha**2)
    a = jnp.clip(alpha, -1.0 + 1e-7, 1.0 - 1e-7)
    near = lambda x: x + jnp.round((mid - x) / (2 * jnp.pi)) * 2 * jnp.pi
    xp, xm = near(x_hi(a)), near(x_lo(a))
    beta_u = jnp.where((xp >= l) & (xp <= u), jnp.maximum(chord, s - alpha * xp), chord)
    beta_l = jnp.where(
        (xm >= l) & (xm <= u), jnp.minimum(chord, -s - alpha * xm), chord
    )
    return alpha, jnp.where(wide, -1.0, beta_l), jnp.where(wide, 1.0, beta_u)


def _relax_sin(l, u, **_):
    return _relax_periodic(jnp.sin, jnp.cos, jnp.arccos, lambda a: -jnp.arccos(a), l, u)


def _relax_cos(l, u, **_):
    x_hi = lambda a: jnp.arcsin(-a)
    return _relax_periodic(
        jnp.cos, lambda x: -jnp.sin(x), x_hi, lambda a: jnp.pi - x_hi(a), l, u
    )


def _relax_integer_pow(l, u, *, y, **_):
    n = int(y)
    if n == 0:
        return jnp.zeros_like(l), jnp.ones_like(l), jnp.ones_like(l)
    if n == 1:
        return jnp.ones_like(l), jnp.zeros_like(l), jnp.zeros_like(l)
    if n < 0:  # zero slope, endpoint bounds
        lo, hi = _concrete_integer_pow(l, u, y=n)
        return jnp.zeros_like(l), lo, hi
    f, df = (lambda x: x**n), (lambda x: n * x ** (n - 1))
    if n % 2 == 0:  # convex; n - 1 odd, so the tangent point is a real root
        inv = lambda a: jnp.sign(a) * jnp.abs(a / n) ** (1.0 / (n - 1))
        return _relax_convex(f, df, inv, l, u)
    # odd: concave for x < 0, convex for x > 0, like tan
    alpha, _ = _chord(f, df, l, u)
    chord = f(l) - alpha * l
    xs = jnp.maximum(alpha / n, 0.0) ** (1.0 / (n - 1))
    beta_u = jnp.where(-xs >= l, -f(xs) + alpha * xs, chord)
    beta_l = jnp.where(xs <= u, f(xs) - alpha * xs, chord)
    return alpha, beta_l, beta_u


def _concrete_monotone(f):
    return lambda l, u, **_: (f(l), f(u))


def _concrete_sin(l, u, **_):
    peak = lambda c: (
        jnp.floor((u - c) / (2 * jnp.pi)) > jnp.floor((l - c) / (2 * jnp.pi))
    )
    fl, fu = jnp.sin(l), jnp.sin(u)
    lo = jnp.where(peak(-jnp.pi / 2), -1.0, jnp.minimum(fl, fu))
    hi = jnp.where(peak(jnp.pi / 2), 1.0, jnp.maximum(fl, fu))
    return lo, hi


def _concrete_cos(l, u, **_):
    return _concrete_sin(l + jnp.pi / 2, u + jnp.pi / 2)


def _concrete_integer_pow(l, u, *, y, **_):
    n = int(y)
    if n == 0:
        return jnp.ones_like(l), jnp.ones_like(l)
    pl, pu = l**n, u**n
    lo, hi = jnp.minimum(pl, pu), jnp.maximum(pl, pu)
    if n > 0 and n % 2 == 0:
        lo = jnp.where((l <= 0) & (0 <= u), 0.0, lo)
    return lo, hi


_ACTIVATIONS = {
    lax.logistic_p: (_relax_logistic, _concrete_monotone(jax.nn.sigmoid)),
    lax.tanh_p: (_relax_tanh, _concrete_monotone(jnp.tanh)),
    lax.tan_p: (_relax_tan, _concrete_monotone(jnp.tan)),
    lax.exp_p: (_relax_exp, _concrete_monotone(jnp.exp)),
    lax.log1p_p: (_relax_log1p, _concrete_monotone(jnp.log1p)),
    lax.sqrt_p: (
        _relax_sqrt,
        lambda l, u, **_: (
            jnp.sqrt(jnp.maximum(l, 0.0)),
            jnp.sqrt(jnp.maximum(u, 0.0)),
        ),
    ),
    lax.sin_p: (_relax_sin, _concrete_sin),
    lax.cos_p: (_relax_cos, _concrete_cos),
    lax.integer_pow_p: (_relax_integer_pow, _concrete_integer_pow),
    lax.square_p: (
        functools.partial(_relax_integer_pow, y=2),
        functools.partial(_concrete_integer_pow, y=2),
    ),
}


def _linbp_activation(relax, concrete):
    """Forward rule: compose the input's linear bounds with the affine relaxation."""

    def rule(x, *, relu_mode, **params):
        alpha, beta_l, beta_u = relax(x.l, x.u, **params)
        lo, hi = concrete(x.l, x.u, **params)
        ap, an = jnp.maximum(alpha, 0.0)[..., None], jnp.minimum(alpha, 0.0)[..., None]
        return LinearBound(
            lA=ap * x.lA + an * x.uA,
            lb=ap[..., 0] * x.lb + an[..., 0] * x.ub + beta_l,
            uA=ap * x.uA + an * x.lA,
            ub=ap[..., 0] * x.ub + an[..., 0] * x.lb + beta_u,
            l=lo,
            u=hi,
        )

    return rule


for _p, (_relax, _conc) in _ACTIVATIONS.items():
    linbp_registry[_p] = _linbp_activation(_relax, _conc)


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
        return LinearBound(
            lA=jnp.zeros((*S, n_in)), lb=l, uA=jnp.zeros((*S, n_in)), ub=u, l=l, u=u
        )
    else:
        n_in = x.n_in
        corners = [x.l / y.l, x.l / y.u, x.u / y.l, x.u / y.u]
        l = jnp.minimum(
            jnp.minimum(corners[0], corners[1]), jnp.minimum(corners[2], corners[3])
        )
        u = jnp.maximum(
            jnp.maximum(corners[0], corners[1]), jnp.maximum(corners[2], corners[3])
        )
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)), lb=l, uA=jnp.zeros((*S, n_in)), ub=u, l=l, u=u
        )


linbp_registry[lax.div_p] = _linbp_div_p


def _linbp_jit_p(
    *args,
    relu_mode,
    tighten_bounds=True,
    x_lb=None,
    x_ub=None,
    iterated_bw=False,
    **bind_params,
):
    """Handle jit_p (jax >= 0.9) / pjit_p (jax < 0.9) by recursing."""
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    return _linbp_jaxpr(
        bind_jaxpr,
        [],
        *args,
        relu_mode=relu_mode,
        tighten_bounds=tighten_bounds,
        x_lb=x_lb,
        x_ub=x_ub,
        iterated_bw=iterated_bw,
    )


_jit_primitive = jit_primitive
if _jit_primitive is not None:
    linbp_registry[_jit_primitive] = _linbp_jit_p
    _linbp_recursive_prims.add(_jit_primitive)


def _linbp_custom_jvp_call_p(
    *args,
    relu_mode,
    call_jaxpr,
    num_consts,
    tighten_bounds=True,
    x_lb=None,
    x_ub=None,
    iterated_bw=False,
    **bind_params,
):
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
        inner_jaxpr,
        consts + extra_consts,
        *actual_args,
        relu_mode=relu_mode,
        tighten_bounds=tighten_bounds,
        x_lb=x_lb,
        x_ub=x_ub,
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
# A forward sweep records every variable's l, u; the reverse sweep carries
# BackwardBound(A_lo, b_lo, A_hi, b_hi) per variable, transposing linear
# primitives and picking each relaxation side by the sign of the downstream A.
# ---------------------------------------------------------------------------


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
        raise NotImplementedError(
            "backward CROWN: batched dot_general not supported yet"
        )
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)
    if is_lb0 and is_lb1:
        # y = dot_general(a, b), both depend on x.  Bilinear in x, so no exact
        # linear bound exists; concretize via IBP and propagate as a pure bias.
        # The accumulated bias is attributed to the first invar only; the
        # second carries zero bias to avoid double counting.
        a_iv = interval(val0.l, val0.u)
        b_iv = interval(val1.l, val1.u)
        y_iv = _inclusion_dot_general_p(a_iv, b_iv, dimension_numbers=dim_nums)
        y_l, y_u = y_iv.lower, y_iv.upper

        A_lo_p = jnp.clip(out_bd.A_lo, 0, None)
        A_lo_n = jnp.clip(out_bd.A_lo, None, 0)
        A_hi_p = jnp.clip(out_bd.A_hi, 0, None)
        A_hi_n = jnp.clip(out_bd.A_hi, None, 0)
        axes = tuple(range(1, out_bd.A_lo.ndim))
        db_lo = jnp.sum(A_lo_p * y_l + A_lo_n * y_u, axis=axes)
        db_hi = jnp.sum(A_hi_p * y_u + A_hi_n * y_l, axis=axes)

        n_out = out_bd.A_lo.shape[0]
        z_a = jnp.zeros((n_out, *val0.l.shape), dtype=out_bd.A_lo.dtype)
        z_b = jnp.zeros((n_out, *val1.l.shape), dtype=out_bd.A_lo.dtype)
        z_lo = jnp.zeros_like(out_bd.b_lo)
        z_hi = jnp.zeros_like(out_bd.b_hi)
        in0 = BackwardBound(z_a, out_bd.b_lo + db_lo, z_a, out_bd.b_hi + db_hi)
        in1 = BackwardBound(z_b, z_lo, z_b, z_hi)
        return [in0, in1]

    # One operand constant: the map is linear in the other; transpose it.
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
    dummy = jnp.zeros(var_shape, dtype=out_bd.A_lo.dtype)

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
    in_bd = BackwardBound(
        out_bd.A_lo, out_bd.b_lo + db_lo, out_bd.A_hi, out_bd.b_hi + db_hi
    )
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
            -out_bd.A_lo,
            out_bd.b_lo + db_lo,
            -out_bd.A_hi,
            out_bd.b_hi + db_hi,
        )
    else:
        # y = var - const
        const = val1
        var = v0
        db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_lo)
        db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const))(out_bd.A_hi)
        in_bd = BackwardBound(
            out_bd.A_lo,
            out_bd.b_lo - db_lo,
            out_bd.A_hi,
            out_bd.b_hi - db_hi,
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
        # Bilinear in x; concretize via interval arithmetic, attribute the
        # accumulated bias to the first invar only (zero bias on the second).
        l0, u0 = val0.l, val0.u
        l1, u1 = val1.l, val1.u
        c00 = l0 * l1
        c01 = l0 * u1
        c10 = u0 * l1
        c11 = u0 * u1
        y_l = jnp.minimum(jnp.minimum(c00, c01), jnp.minimum(c10, c11))
        y_u = jnp.maximum(jnp.maximum(c00, c01), jnp.maximum(c10, c11))

        A_lo_p = jnp.clip(out_bd.A_lo, 0, None)
        A_lo_n = jnp.clip(out_bd.A_lo, None, 0)
        A_hi_p = jnp.clip(out_bd.A_hi, 0, None)
        A_hi_n = jnp.clip(out_bd.A_hi, None, 0)
        axes = tuple(range(1, out_bd.A_lo.ndim))
        db_lo = jnp.sum(A_lo_p * y_l + A_lo_n * y_u, axis=axes)
        db_hi = jnp.sum(A_hi_p * y_u + A_hi_n * y_l, axis=axes)

        n_out = out_bd.A_lo.shape[0]
        z_a = jnp.zeros((n_out, *val0.l.shape), dtype=out_bd.A_lo.dtype)
        z_b = jnp.zeros((n_out, *val1.l.shape), dtype=out_bd.A_lo.dtype)
        z_lo = jnp.zeros_like(out_bd.b_lo)
        z_hi = jnp.zeros_like(out_bd.b_hi)
        in0 = BackwardBound(z_a, out_bd.b_lo + db_lo, z_a, out_bd.b_hi + db_hi)
        in1 = BackwardBound(z_b, z_lo, z_b, z_hi)
        return [in0, in1]
    if is_lb0:
        var = v0
        c = jnp.asarray(val1)
    else:
        var = v1
        c = jnp.asarray(val0)
    # Multiply by constant: handle broadcasting (e.g. scalar var * vector const
    # where the eqn implicitly broadcasts the scalar to vector before mul).
    # Use jax.linear_transpose to correctly reduce over any broadcast dims.
    var_shape = var.aval.shape
    dummy = jnp.zeros(var_shape, dtype=out_bd.A_lo.dtype)
    primal = lambda x_in: x_in * c

    def transpose_one(cov):
        (tr,) = jax.linear_transpose(primal, dummy)(cov)
        return tr

    new_A_lo = jax.vmap(transpose_one)(out_bd.A_lo)
    new_A_hi = jax.vmap(transpose_one)(out_bd.A_hi)
    in_bd = BackwardBound(new_A_lo, out_bd.b_lo, new_A_hi, out_bd.b_hi)
    return [in_bd if v is var else None for v in eqn.invars]


def _bw_neg(eqn, out_bd, env, **_):
    """y = -x.  Exact substitution: lb_orig ≥ A_lo @ y + b_lo = (-A_lo) @ x + b_lo.
    The same negation applies to the upper bound; biases are unchanged.
    """
    in_bd = BackwardBound(-out_bd.A_lo, out_bd.b_lo, -out_bd.A_hi, out_bd.b_hi)
    return [in_bd]


def _bw_two_slope(out_bd, a_l, b_l, a_u, b_u):
    """Backward through ``a_l x + b_l <= f(x) <= a_u x + b_u``: the sign of ``A``
    picks which side each entry uses."""

    def side(A, a_pos, b_pos, a_neg, b_neg):
        Ap, An = jnp.maximum(A, 0), jnp.minimum(A, 0)
        db = jax.vmap(
            lambda p, n: _bw_sum_over_var(p, b_pos) + _bw_sum_over_var(n, b_neg)
        )(Ap, An)
        return Ap * a_pos + An * a_neg, db

    A_lo, db_lo = side(out_bd.A_lo, a_l, b_l, a_u, b_u)
    A_hi, db_hi = side(out_bd.A_hi, a_u, b_u, a_l, b_l)
    return BackwardBound(A_lo, out_bd.b_lo + db_lo, A_hi, out_bd.b_hi + db_hi)


def _bw_max_min(sign):
    """y = max(x, c) (``sign=1``) or min(x, c) = -max(-x, -c) (``sign=-1``)."""

    def rule(eqn, out_bd, env, *, relu_mode="adaptive"):
        vals = [_read(env, v) for v in eqn.invars]
        is_lb = [isinstance(v, LinearBound) for v in vals]
        if is_lb[0] == is_lb[1]:
            raise NotImplementedError(
                "backward CROWN: max/min of two LinearBounds (or of none)"
            )
        k = 0 if is_lb[0] else 1
        x, c = vals[k], jnp.asarray(vals[1 - k])
        bd = _bw_two_slope(out_bd, *_max_min_relax(x.l, x.u, c, relu_mode, sign))
        return [bd if i == k else None for i in range(2)]

    return rule


def _bw_structural(eqn, out_bd, env, **_):
    """Generic structural op (slice, squeeze, reshape, broadcast_in_dim,
    concatenate, etc.): use jax.linear_transpose on the primitive."""
    var_idx = next(
        i for i, v in enumerate(eqn.invars) if isinstance(_read(env, v), LinearBound)
    )
    var = eqn.invars[var_idx]
    var_shape = var.aval.shape

    # Build primal of the primitive with the variable arg replaced by dummy
    subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)
    fixed = [_read(env, v) for v in eqn.invars]

    def primal(x):
        full = list(fixed)
        full[var_idx] = x
        return eqn.primitive.bind(*subfuns, *full, **bind_params)

    dummy = jnp.zeros(var_shape, dtype=out_bd.A_lo.dtype)

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
            in_bds.append(
                BackwardBound(
                    out_bd.A_lo[sl],
                    b_lo_acc,
                    out_bd.A_hi[sl],
                    b_hi_acc,
                )
            )
            bias_given = True
        else:
            in_bds.append(
                BackwardBound(
                    out_bd.A_lo[sl],
                    z_lo,
                    out_bd.A_hi[sl],
                    z_hi,
                )
            )
    if not bias_given:
        raise NotImplementedError("backward CROWN: concatenate with no LB invars")
    return in_bds


# Activation primitives that benefit from iterated backward CROWN at the
# point of slope selection. Recursive primitives (jit_p, custom_jvp_call_p)
# are also tightened because they wrap activations in practice.
_activation_prims_bw: set = {lax.max_p, lax.min_p, lax.logistic_p, lax.tanh_p}


def _bw_split(eqn, out_bds_out, env, **_):
    """y_1, ..., y_k = split(x, sizes=..., axis=...).

    Inverse (transpose) of split is concatenate; use ``jax.linear_transpose``
    over each cotangent row to recover ``A`` on the input.  Missing outvars
    (chunks not used downstream) are zero-filled.  Bias contributions from
    each used chunk are summed onto the single input invar.
    """
    var = eqn.invars[0]
    var_shape = var.aval.shape
    subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)

    def primal(x):
        return tuple(eqn.primitive.bind(*subfuns, x, **bind_params))

    sample_bd = next(iter(out_bds_out.values()))
    n_out = sample_bd.A_lo.shape[0]
    dtype = sample_bd.A_lo.dtype

    def make_covs(field):
        return tuple(
            getattr(out_bds_out[ov], field)
            if ov in out_bds_out
            else jnp.zeros((n_out, *ov.aval.shape), dtype=dtype)
            for ov in eqn.outvars
        )

    A_lo_covs = make_covs("A_lo")
    A_hi_covs = make_covs("A_hi")

    dummy = jnp.zeros(var_shape, dtype=dtype)
    T = jax.linear_transpose(primal, dummy)

    def apply_T(*covs):
        (tr,) = T(covs)
        return tr

    new_A_lo = jax.vmap(apply_T)(*A_lo_covs)
    new_A_hi = jax.vmap(apply_T)(*A_hi_covs)

    b_lo = sum(
        (out_bds_out[ov].b_lo for ov in eqn.outvars if ov in out_bds_out),
        start=jnp.zeros(n_out, dtype=dtype),
    )
    b_hi = sum(
        (out_bds_out[ov].b_hi for ov in eqn.outvars if ov in out_bds_out),
        start=jnp.zeros(n_out, dtype=dtype),
    )

    return [BackwardBound(new_A_lo, b_lo, new_A_hi, b_hi)]


backward_registry[lax.dot_general_p] = _bw_dot_general
backward_registry[lax.add_p] = _bw_add
backward_registry[ad_util.add_any_p] = _bw_add
backward_registry[lax.sub_p] = _bw_sub
backward_registry[lax.mul_p] = _bw_mul
backward_registry[lax.neg_p] = _bw_neg
backward_registry[lax.max_p] = _bw_max_min(1)
backward_registry[lax.min_p] = _bw_max_min(-1)
backward_registry[lax.concatenate_p] = _bw_concatenate


def _bw_stack(eqn, out_bd, env, **_):
    """y = stack(xs, axis): x_i gets slice i of A along that axis; constants fold
    into the bias, which goes to the first bound operand only."""
    ax = eqn.params["axis"] + 1  # A has a leading n_out axis
    vals = [_read(env, v) for v in eqn.invars]
    take = lambda A, i: jnp.take(A, i, axis=ax)
    b_lo, b_hi = out_bd.b_lo, out_bd.b_hi
    for i, v in enumerate(vals):
        if not isinstance(v, LinearBound):
            c = jnp.asarray(v)
            b_lo = b_lo + jax.vmap(lambda a: _bw_sum_over_var(a, c))(
                take(out_bd.A_lo, i)
            )
            b_hi = b_hi + jax.vmap(lambda a: _bw_sum_over_var(a, c))(
                take(out_bd.A_hi, i)
            )
    out, first = [], True
    for i, v in enumerate(vals):
        if not isinstance(v, LinearBound):
            out.append(None)
            continue
        lo, hi = (b_lo, b_hi) if first else (jnp.zeros_like(b_lo), jnp.zeros_like(b_hi))
        out.append(BackwardBound(take(out_bd.A_lo, i), lo, take(out_bd.A_hi, i), hi))
        first = False
    return out


if hasattr(lax, "stack_p"):  # jax >= 0.10
    backward_registry[lax.stack_p] = _bw_stack
for _p in (
    lax.slice_p,
    lax.squeeze_p,
    lax.reshape_p,
    lax.broadcast_in_dim_p,
    lax.transpose_p,
    lax.convert_element_type_p,
):
    backward_registry[_p] = _bw_structural
for _p_name in ("dynamic_slice_p", "gather_p"):
    if hasattr(lax, _p_name):
        backward_registry[getattr(lax, _p_name)] = _bw_structural


def _bw_scatter(eqn, out_bd, env, **_):
    """y = scatter(operand, indices, updates).

    Linear in (operand, updates).  Scatter is linear and its transpose is
    typically only implemented when ``unique_indices=True``; we override that
    flag for the transpose call.  The caller is responsible for ensuring the
    indices ARE unique (true for canonical at[...].set / .add patterns).

    If ``updates`` is the LinearBound invar (the common NN case), the input
    backward A becomes ``gather(out_bd.A, indices)``.  If ``operand`` is also
    a LinearBound, the unchanged positions pass straight through.
    """
    invars = eqn.invars
    operand_var, indices_var, updates_var = invars
    operand_val = (
        operand_var.val if isinstance(operand_var, Literal) else env.get(operand_var)
    )
    indices_val = (
        indices_var.val if isinstance(indices_var, Literal) else env.get(indices_var)
    )
    updates_val = (
        updates_var.val if isinstance(updates_var, Literal) else env.get(updates_var)
    )

    is_lb_operand = isinstance(operand_val, LinearBound)
    is_lb_updates = isinstance(updates_val, LinearBound)
    if not (is_lb_operand or is_lb_updates):
        return [None, None, None]

    subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)
    # Force unique_indices so jax.linear_transpose succeeds.
    bind_params["unique_indices"] = True

    def primal(operand_x, updates_x):
        return eqn.primitive.bind(
            *subfuns, operand_x, indices_val, updates_x, **bind_params
        )

    op_shape = operand_var.aval.shape
    up_shape = updates_var.aval.shape
    op_dummy = jnp.zeros(op_shape, dtype=out_bd.A_lo.dtype)
    up_dummy = jnp.zeros(up_shape, dtype=out_bd.A_lo.dtype)

    T = jax.linear_transpose(primal, op_dummy, up_dummy)

    def transpose_one(cov):
        return T(cov)  # returns (op_cot, up_cot)

    new_A_op_lo, new_A_up_lo = jax.vmap(transpose_one)(out_bd.A_lo)
    new_A_op_hi, new_A_up_hi = jax.vmap(transpose_one)(out_bd.A_hi)

    out_bds = [None, None, None]
    bias_given = False
    z_lo = jnp.zeros_like(out_bd.b_lo)
    z_hi = jnp.zeros_like(out_bd.b_hi)
    if is_lb_operand:
        out_bds[0] = BackwardBound(new_A_op_lo, out_bd.b_lo, new_A_op_hi, out_bd.b_hi)
        bias_given = True
    else:
        # operand is a constant — fold its contribution into bias.
        op_const = jnp.asarray(operand_val)
        db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, op_const))(new_A_op_lo)
        db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, op_const))(new_A_op_hi)
        out_bd_lo = out_bd.b_lo + db_lo
        out_bd_hi = out_bd.b_hi + db_hi
        if is_lb_updates:
            out_bds[2] = BackwardBound(new_A_up_lo, out_bd_lo, new_A_up_hi, out_bd_hi)
            bias_given = True
        else:
            # Updates is also const — shouldn't occur if forward was LB.
            return [None, None, None]

    if is_lb_updates and is_lb_operand:
        out_bds[2] = BackwardBound(
            new_A_up_lo,
            z_lo if bias_given else out_bd.b_lo,
            new_A_up_hi,
            z_hi if bias_given else out_bd.b_hi,
        )
    return out_bds


for _p_name in ("scatter_p", "scatter_add_p", "scatter_mul_p"):
    if hasattr(lax, _p_name):
        backward_registry[getattr(lax, _p_name)] = _bw_scatter
if hasattr(lax, "split_p"):
    # split is multi-output; needs the out_bds dict form, so register it as
    # a "recursive" prim (the _bw_walk dispatch flag that passes all outvar
    # bounds to the handler).
    backward_registry[lax.split_p] = _bw_split
    _bw_recursive_prims.add(lax.split_p)


def _bw_select_n(eqn, out_bd, env, **_):
    """y = select_n(which, case0, case1, ...).  ``which`` is an integer array
    (not a LinearBound); each case can be a LinearBound or constant.  The
    transpose maps the output cotangent to each case (zeroed outside the
    selected positions); constant cases fold their value into the bias.
    """
    which_var = eqn.invars[0]
    case_vars = eqn.invars[1:]
    which_val = which_var.val if isinstance(which_var, Literal) else env.get(which_var)
    case_vals = [_read(env, v) for v in case_vars]

    subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)

    def primal(*case_xs):
        return eqn.primitive.bind(*subfuns, which_val, *case_xs, **bind_params)

    dummies = [jnp.zeros(v.aval.shape, dtype=out_bd.A_lo.dtype) for v in case_vars]
    T = jax.linear_transpose(primal, *dummies)

    def transpose_one(cov):
        return T(cov)  # returns tuple aligned with case_vars

    cot_los = jax.vmap(transpose_one)(out_bd.A_lo)
    cot_his = jax.vmap(transpose_one)(out_bd.A_hi)

    in_bds = [None]  # `which` slot
    z_lo = jnp.zeros_like(out_bd.b_lo)
    z_hi = jnp.zeros_like(out_bd.b_hi)
    # Find first LB index to absorb biases (bias attribution rule).
    first_lb_idx = next(
        (i for i, cv in enumerate(case_vals) if isinstance(cv, LinearBound)),
        None,
    )
    for i, (case_var, case_val) in enumerate(zip(case_vars, case_vals)):
        if not isinstance(case_val, LinearBound):
            in_bds.append(None)
            continue
        if i == first_lb_idx:
            b_lo_i = out_bd.b_lo
            b_hi_i = out_bd.b_hi
            # Fold all const cases' contributions into this bias.
            for j, cv in enumerate(case_vals):
                if isinstance(cv, LinearBound):
                    continue
                const_val = jnp.asarray(cv)
                db_lo = jax.vmap(lambda a: _bw_sum_over_var(a, const_val))(cot_los[j])
                db_hi = jax.vmap(lambda a: _bw_sum_over_var(a, const_val))(cot_his[j])
                b_lo_i = b_lo_i + db_lo
                b_hi_i = b_hi_i + db_hi
            in_bds.append(BackwardBound(cot_los[i], b_lo_i, cot_his[i], b_hi_i))
        else:
            in_bds.append(BackwardBound(cot_los[i], z_lo, cot_his[i], z_hi))
    return in_bds


if hasattr(lax, "select_n_p"):
    backward_registry[lax.select_n_p] = _bw_select_n


def _bw_div(eqn, out_bd, env, **_):
    """y = x / z.
    - ``x / const``:   linear in x, delegate to mul-by-(1/const).
    - ``const / z`` or ``x / z`` with z a LinearBound: nonlinear (rational);
      concretize via IBP and propagate as a pure bias.
    """
    v0, v1 = eqn.invars
    val0 = v0.val if isinstance(v0, Literal) else env.get(v0)
    val1 = v1.val if isinstance(v1, Literal) else env.get(v1)
    is_lb0 = isinstance(val0, LinearBound)
    is_lb1 = isinstance(val1, LinearBound)

    if is_lb0 and not is_lb1:
        # x / c  ==  x * (1/c) ; reuse the const-mul backward.
        var = v0
        c = jnp.reciprocal(jnp.asarray(val1))
        var_shape = var.aval.shape
        dummy = jnp.zeros(var_shape, dtype=out_bd.A_lo.dtype)
        primal = lambda x_in: x_in * c

        def transpose_one(cov):
            (tr,) = jax.linear_transpose(primal, dummy)(cov)
            return tr

        new_A_lo = jax.vmap(transpose_one)(out_bd.A_lo)
        new_A_hi = jax.vmap(transpose_one)(out_bd.A_hi)
        in_bd = BackwardBound(new_A_lo, out_bd.b_lo, new_A_hi, out_bd.b_hi)
        return [in_bd if v is var else None for v in eqn.invars]

    # Both LB or denominator-LB: concretize via 4-corner IBP and propagate
    # as a pure bias.  Attribute bias to first LB invar.
    def _concrete_iv(val):
        if isinstance(val, LinearBound):
            return val.l, val.u
        a = jnp.asarray(val)
        return a, a

    l0, u0 = _concrete_iv(val0)
    l1, u1 = _concrete_iv(val1)
    c00 = l0 / l1
    c01 = l0 / u1
    c10 = u0 / l1
    c11 = u0 / u1
    y_l = jnp.minimum(jnp.minimum(c00, c01), jnp.minimum(c10, c11))
    y_u = jnp.maximum(jnp.maximum(c00, c01), jnp.maximum(c10, c11))

    A_lo_p = jnp.clip(out_bd.A_lo, 0, None)
    A_lo_n = jnp.clip(out_bd.A_lo, None, 0)
    A_hi_p = jnp.clip(out_bd.A_hi, 0, None)
    A_hi_n = jnp.clip(out_bd.A_hi, None, 0)
    axes = tuple(range(1, out_bd.A_lo.ndim))
    db_lo = jnp.sum(A_lo_p * y_l + A_lo_n * y_u, axis=axes)
    db_hi = jnp.sum(A_hi_p * y_u + A_hi_n * y_l, axis=axes)

    n_out = out_bd.A_lo.shape[0]
    z_lo = jnp.zeros_like(out_bd.b_lo)
    z_hi = jnp.zeros_like(out_bd.b_hi)
    out_bds = [None, None]
    bias_given = False
    for i, (v, val, is_lb) in enumerate([(v0, val0, is_lb0), (v1, val1, is_lb1)]):
        if not is_lb:
            continue
        z = jnp.zeros((n_out, *val.l.shape), dtype=out_bd.A_lo.dtype)
        if not bias_given:
            out_bds[i] = BackwardBound(z, out_bd.b_lo + db_lo, z, out_bd.b_hi + db_hi)
            bias_given = True
        else:
            out_bds[i] = BackwardBound(z, z_lo, z, z_hi)
    return out_bds


backward_registry[lax.div_p] = _bw_div


def _bw_affine_relaxation(out_bd, alpha, beta_l, beta_u):
    return _bw_two_slope(out_bd, alpha, beta_l, alpha, beta_u)


def _bw_activation(relax):
    def rule(eqn, out_bd, env, **_):
        x = env[eqn.invars[0]]
        return [_bw_affine_relaxation(out_bd, *relax(x.l, x.u, **eqn.params))]

    return rule


for _p, (_relax, _) in _ACTIVATIONS.items():
    backward_registry[_p] = _bw_activation(_relax)
backward_registry[lax.reduce_sum_p] = _bw_structural  # linear: transpose = broadcast


def _inner_forward(forward_mode, inner_jaxpr, inner_consts, *args, relu_mode):
    """Dispatch the inner forward sweep for recursive backward handlers."""
    if forward_mode == "linbp":
        return _linbp_jaxpr(
            inner_jaxpr,
            inner_consts,
            *args,
            relu_mode=relu_mode,
            tighten_bounds=True,
            return_env=True,
        )
    elif forward_mode == "ibp":
        return _ibp_jaxpr(
            inner_jaxpr,
            inner_consts,
            *args,
            relu_mode=relu_mode,
            return_env=True,
        )
    else:
        raise ValueError(f"forward_mode must be 'linbp' or 'ibp', got {forward_mode!r}")


def _bw_jit(eqn, out_bds_out, env, *, relu_mode, forward_mode="linbp"):
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
    # Run the forward sweep on the inner jaxpr to populate an inner env, then
    # backward-walk it.
    args = [_read(env, v) for v in eqn.invars]
    _, inner_env = _inner_forward(
        forward_mode,
        inner_jaxpr,
        inner_consts,
        *args,
        relu_mode=relu_mode,
    )
    inner_bw = {}
    for ov_outer, ov_inner in zip(eqn.outvars, inner_jaxpr.outvars):
        inner_bw[ov_inner] = out_bds_out[ov_outer]
    _bw_walk(
        inner_jaxpr, inner_env, inner_bw, relu_mode=relu_mode, forward_mode=forward_mode
    )
    # Map back: invars
    return [inner_bw.get(iv) for iv in inner_jaxpr.invars]


def _bw_custom_jvp_call(eqn, out_bds_out, env, *, relu_mode, forward_mode="linbp"):
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
        extra_consts = [_read(env, v) for v in eqn.invars[:num_consts]]
        actual_invars = eqn.invars[num_consts:]
    args = [_read(env, v) for v in actual_invars]
    _, inner_env = _inner_forward(
        forward_mode,
        inner_jaxpr,
        inner_consts + extra_consts,
        *args,
        relu_mode=relu_mode,
    )
    inner_bw = {}
    for ov_outer, ov_inner in zip(eqn.outvars, inner_jaxpr.outvars):
        inner_bw[ov_inner] = out_bds_out[ov_outer]
    _bw_walk(
        inner_jaxpr, inner_env, inner_bw, relu_mode=relu_mode, forward_mode=forward_mode
    )
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


def _bw_walk(
    jaxpr: Jaxpr,
    env: dict,
    bw_env: dict,
    *,
    relu_mode: str,
    forward_mode: str = "linbp",
):
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
            in_bds = handler(
                eqn, out_bds_map, env, relu_mode=relu_mode, forward_mode=forward_mode
            )
        else:
            out_bd = bw_env[eqn.outvars[0]]
            in_bds = handler(eqn, out_bd, env, relu_mode=relu_mode)
        for v, bd in zip(eqn.invars, in_bds):
            if bd is None or isinstance(v, Literal):
                continue
            if v in bw_env:
                # Accumulate (multi-use var)
                prev = bw_env[v]
                bw_env[v] = BackwardBound(
                    prev.A_lo + bd.A_lo,
                    prev.b_lo + bd.b_lo,
                    prev.A_hi + bd.A_hi,
                    prev.b_hi + bd.b_hi,
                )
            else:
                bw_env[v] = bd


def _backward_to_concrete(
    jaxpr,
    env,
    target_var,
    target_idx,
    x_lb,
    x_ub,
    relu_mode,
    forward_mode: str = "linbp",
):
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
            in_bds = handler(
                eqn, out_bds_map, env, relu_mode=relu_mode, forward_mode=forward_mode
            )
        else:
            out_bd = bw_env[eqn.outvars[0]]
            in_bds = handler(eqn, out_bd, env, relu_mode=relu_mode)
        for v, bd in zip(eqn.invars, in_bds):
            if bd is None or isinstance(v, Literal):
                continue
            if v in bw_env:
                prev = bw_env[v]
                bw_env[v] = BackwardBound(
                    prev.A_lo + bd.A_lo,
                    prev.b_lo + bd.b_lo,
                    prev.A_hi + bd.A_hi,
                    prev.b_hi + bd.b_hi,
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
    l_flat, u_flat = _bw_bounds(bd, x_lb, x_ub)
    return l_flat.reshape(S), u_flat.reshape(S)


def _linbp_backward_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    relu_mode: str = "adaptive",
    tighten_bounds: bool = True,
    x_lb=None,
    x_ub=None,
    iterated_bw: bool = False,
    forward_mode: str = "linbp",
):
    """Forward sweep to capture env, then backward sweep with sign-conditioned slopes.

    ``forward_mode`` selects the forward sweep:
      - ``'linbp'`` (default): linear-bound propagation, optionally tightened
        with IBP via ``tighten_bounds``. With ``iterated_bw=True`` this gives
        "pure backward CROWN".
      - ``'ibp'``: pure interval-arithmetic propagation. With
        ``iterated_bw=False`` this is standard **IBP+CROWN**; with
        ``iterated_bw=True`` it is standard **CROWN** (each pre-activation
        re-derived by backward sweep from the input). ``tighten_bounds`` has
        no effect under ``'ibp'``.

    Returns a list of BackwardBound, one per jaxpr.outvar, expressed as a linear
    function of the input variable (jaxpr.invars[0]).
    """
    if forward_mode == "linbp":
        outs, env = _linbp_jaxpr(
            jaxpr,
            consts,
            *args,
            relu_mode=relu_mode,
            tighten_bounds=tighten_bounds,
            x_lb=x_lb,
            x_ub=x_ub,
            return_env=True,
            iterated_bw=iterated_bw,
        )
    elif forward_mode == "ibp":
        outs, env = _ibp_jaxpr(
            jaxpr,
            consts,
            *args,
            relu_mode=relu_mode,
            x_lb=x_lb,
            x_ub=x_ub,
            return_env=True,
            iterated_bw=iterated_bw,
        )
    else:
        raise ValueError(f"forward_mode must be 'linbp' or 'ibp', got {forward_mode!r}")
    bw_env = {}
    import math

    for ov in jaxpr.outvars:
        S = tuple(ov.aval.shape)
        n_out = math.prod(S) if S else 1
        # Initialise with identity coefficient: A has shape (n_out, *S)
        I = jnp.eye(n_out).reshape((n_out,) + S)
        bw_env[ov] = BackwardBound(I, jnp.zeros(n_out), I, jnp.zeros(n_out))
    _bw_walk(jaxpr, env, bw_env, relu_mode=relu_mode, forward_mode=forward_mode)
    return [bw_env[v] for v in jaxpr.invars]


def linbp_backward(
    f,
    relu_mode: str = "adaptive",
    tighten_bounds: bool = True,
    iterated_bw: bool = False,
    forward_mode: str = "linbp",
):
    """Backward CROWN: like ``linbp`` but with sign-conditioned per-(output,
    neuron) slope picking at each ReLU.

    Parameters
    ----------
    forward_mode : {"linbp", "ibp"}
        Forward sweep used to populate per-variable concrete ``l, u`` consumed
        by the backward CROWN sweep. ``'linbp'`` (default) uses linear-bound
        propagation; combined with ``iterated_bw=True`` this is "pure backward
        CROWN". ``'ibp'`` uses pure interval arithmetic: combined with
        ``iterated_bw=False`` this is **IBP+CROWN**, and combined with
        ``iterated_bw=True`` this is **standard CROWN**. ``tighten_bounds`` is
        ignored when ``forward_mode='ibp'``.

    Returns a callable mapping an ``Interval`` to a ``LinearBound``.  Shapes
    match the forward :func:`linbp` convention: ``lA, uA`` are ``(*S_out, *S_in)``
    and ``lb, ub, l, u`` are ``S_out`` — where ``S_out`` is the traced output
    shape of ``f`` (so e.g. a matrix-valued ``f`` yields matrix-shaped bounds).
    """

    def wrapped(inp) -> LinearBound:
        lb_init, trace_point, x_lb, x_ub = _resolve_linbp_input(inp)
        closed_jaxpr = eqx.filter_make_jaxpr(f)(trace_point)[0]
        in_bds = _linbp_backward_jaxpr(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            lb_init,
            relu_mode=relu_mode,
            tighten_bounds=tighten_bounds,
            x_lb=x_lb,
            x_ub=x_ub,
            iterated_bw=iterated_bw,
            forward_mode=forward_mode,
        )
        if len(in_bds) != 1:
            raise NotImplementedError(
                "backward CROWN: multi-input functions not supported yet"
            )
        bd = in_bds[0]
        if x_lb is None:
            l = jnp.full(bd.b_lo.shape, -jnp.inf)
            u = jnp.full(bd.b_hi.shape, jnp.inf)
        else:
            l, u = _bw_bounds(bd, x_lb, x_ub)

        # Reshape bounds from the backward CROWN's flat layout (n_out, *S_in)
        # to the forward linbp convention (*S_out, *S_in).
        S_out = tuple(closed_jaxpr.jaxpr.outvars[0].aval.shape)
        in_shape = bd.A_lo.shape[1:]  # *S_in
        lA = bd.A_lo.reshape(S_out + in_shape)
        uA = bd.A_hi.reshape(S_out + in_shape)
        lb = bd.b_lo.reshape(S_out)
        ub = bd.b_hi.reshape(S_out)
        l = l.reshape(S_out)
        u = u.reshape(S_out)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)

    return wrapped


def _bw_concrete(Ap, An, x_for_pos, x_for_neg):
    """Concretize: sum over var dims of (Ap * x_for_pos + An * x_for_neg)."""
    return jnp.tensordot(Ap, x_for_pos, axes=Ap.ndim - 1) + jnp.tensordot(
        An, x_for_neg, axes=An.ndim - 1
    )


def _bw_bounds(bd, x_lb, x_ub):
    """Concrete ``(l, u)`` of a :class:`BackwardBound` over the input box."""
    clip = lambda A: (jnp.clip(A, 0, None), jnp.clip(A, None, 0))
    return (
        _bw_concrete(*clip(bd.A_lo), x_lb, x_ub) + bd.b_lo,
        _bw_concrete(*clip(bd.A_hi), x_ub, x_lb) + bd.b_hi,
    )


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
        raise TypeError(
            f"linbp wrapped function expects Interval or LinearBound, got {type(inp)}"
        )


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
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            lb_init,
            relu_mode=relu_mode,
            tighten_bounds=tighten_bounds,
            x_lb=x_lb,
            x_ub=x_ub,
        )
        return outs[0] if len(outs) == 1 else outs

    return wrapped
