"""Soundness and tightness tests for backward linear bound propagation.

Verifies that:
- ``crown(net)`` (default = one-pass backward CROWN) produces sound
  overapproximations of the network output.
- ``crown(net, iterated=True)`` (pure backward CROWN with re-derived
  pre-activation bounds at each ReLU) also produces sound overapproximations.
- Same for ``fastlin(net)`` and ``fastlin(net, iterated=True)``.
- One-pass backward bounds are no looser than forward-only ``linbp`` bounds
  on the sample mean width across a battery of random nets — strict
  monotonicity isn't required for a single network (slope choice at
  individual ReLUs can flip), but on average backward should win.
"""

import jax
import jax.numpy as jnp
import equinox.nn as nn
import pytest

import immrax as irx
from immrax import interval, crown, fastlin
from immrax.inclusion.linbp import linbp, linbp_backward


N_SAMPLES = 200
TOL = 1e-5  # numerical slack for "sound containment"


# ---------------------------------------------------------------------------
# Test fixtures (mirror test_linbp.py)
# ---------------------------------------------------------------------------

def _sample_in_interval(ix, n, key):
    shape = (n, *ix.lower.shape)
    u = jax.random.uniform(key, shape)
    return ix.lower + u * (ix.upper - ix.lower)


def _build_net(arch, activation, key):
    act_fns = {
        'relu':    jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'tanh':    lambda x: 2 * jax.nn.sigmoid(2 * x) - 1,
    }
    act_fn = act_fns[activation]
    layers = []
    for i in range(len(arch) - 1):
        key, subkey = jax.random.split(key)
        layers.append(nn.Linear(arch[i], arch[i + 1], key=subkey))
        if i < len(arch) - 2:
            layers.append(nn.Lambda(act_fn))
    return nn.Sequential(layers)


_ARCH_ACTIVATION_PARAMS = [
    pytest.param(([2, 4, 1],    'relu'),    id="2-4-1/relu"),
    pytest.param(([3, 8, 4, 2], 'relu'),    id="3-8-4-2/relu"),
    pytest.param(([2, 4, 2],    'relu'),    id="2-4-2/relu"),
    pytest.param(([4, 8, 4, 2], 'relu'),    id="4-8-4-2/relu"),
    pytest.param(([2, 6, 1],    'sigmoid'), id="2-6-1/sigmoid"),
    pytest.param(([3, 8, 4, 1], 'sigmoid'), id="3-8-4-1/sigmoid"),
    pytest.param(([2, 6, 2],    'tanh'),    id="2-6-2/tanh"),
]

_NET_KEY_PARAMS = [pytest.param(k, id=f"key{k}") for k in (0, 1, 7)]

_INPUT_INTERVAL_PARAMS = [
    pytest.param('unit',  id="unit-box"),
    pytest.param('small', id="small-box"),
    pytest.param('asym',  id="asym-box"),
]


def _make_interval(kind, n_in):
    if kind == 'unit':
        return interval(jnp.full(n_in, -1.0), jnp.full(n_in, 1.0))
    if kind == 'small':
        center = jnp.arange(1, n_in + 1, dtype=jnp.float32) * 0.5
        return irx.icentpert(center, 0.2)
    lb = jnp.array([-0.5 * (i + 1) for i in range(n_in)], dtype=jnp.float32)
    ub = jnp.array([0.3 * (i + 1) for i in range(n_in)], dtype=jnp.float32)
    return interval(lb, ub)


@pytest.fixture(params=_ARCH_ACTIVATION_PARAMS)
def arch_act(request):
    return request.param


@pytest.fixture(params=_NET_KEY_PARAMS)
def net_key(request):
    return request.param


@pytest.fixture(params=_INPUT_INTERVAL_PARAMS)
def input_kind(request):
    return request.param


@pytest.fixture
def net_and_ix(arch_act, net_key, input_kind):
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    return net, _make_interval(input_kind, arch[0])


# ---------------------------------------------------------------------------
# Shape / type sanity for backward variants
# ---------------------------------------------------------------------------

def test_backward_crown_shapes(arch_act, net_key):
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    n_in, n_out = arch[0], arch[-1]
    ix = _make_interval('unit', n_in)

    cr = crown(net, backward=True)(ix)
    assert cr.lC.shape == (n_out, n_in)
    assert cr.uC.shape == (n_out, n_in)
    assert cr.ld.shape == (n_out,)
    assert cr.ud.shape == (n_out,)


def test_iterated_crown_shapes(arch_act, net_key):
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    n_in, n_out = arch[0], arch[-1]
    ix = _make_interval('unit', n_in)

    cr = crown(net, iterated=True)(ix)
    assert cr.lC.shape == (n_out, n_in)
    assert cr.uC.shape == (n_out, n_in)
    assert cr.ld.shape == (n_out,)
    assert cr.ud.shape == (n_out,)


def test_backward_fastlin_single_C(arch_act, net_key):
    """FastLin (same-slope) backward must yield a single coefficient matrix
    (lA == uA in the underlying LinearBound)."""
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('unit', arch[0])

    fl = fastlin(net, backward=True)(ix)
    n_in, n_out = arch[0], arch[-1]
    assert fl.C.shape  == (n_out, n_in)
    assert fl.ld.shape == (n_out,)
    assert fl.ud.shape == (n_out,)


# ---------------------------------------------------------------------------
# Soundness: backward CROWN must contain net(x) for all sampled x
# ---------------------------------------------------------------------------

def _assert_contains(out_interval, outputs, label):
    sample_lo = outputs.min(axis=0)
    sample_hi = outputs.max(axis=0)
    assert jnp.all(out_interval.lower - TOL <= sample_lo), (
        f"{label}: lower bound violated.\n"
        f"  bound.lower = {out_interval.lower}\n"
        f"  sample.min  = {sample_lo}"
    )
    assert jnp.all(sample_hi <= out_interval.upper + TOL), (
        f"{label}: upper bound violated.\n"
        f"  sample.max  = {sample_hi}\n"
        f"  bound.upper = {out_interval.upper}"
    )


def test_backward_crown_contains_output(net_and_ix):
    net, ix = net_and_ix
    out = crown(net, backward=True)(ix)(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(net)(samples)
    _assert_contains(out, outputs, "backward crown")


def test_iterated_crown_contains_output(net_and_ix):
    net, ix = net_and_ix
    out = crown(net, iterated=True)(ix)(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(43))
    outputs = jax.vmap(net)(samples)
    _assert_contains(out, outputs, "iterated crown")


def test_backward_fastlin_contains_output(net_and_ix):
    net, ix = net_and_ix
    out = fastlin(net, backward=True)(ix)(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(44))
    outputs = jax.vmap(net)(samples)
    _assert_contains(out, outputs, "backward fastlin")


def test_iterated_fastlin_contains_output(net_and_ix):
    net, ix = net_and_ix
    out = fastlin(net, iterated=True)(ix)(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(45))
    outputs = jax.vmap(net)(samples)
    _assert_contains(out, outputs, "iterated fastlin")


def test_backward_crown_pointwise(net_and_ix):
    """Per-sample containment (not just min/max across the sample cloud)."""
    net, ix = net_and_ix
    cr = crown(net, backward=True)(ix)
    out = cr(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(99))

    def ok(x):
        y = net(x)
        return jnp.all(out.lower - TOL <= y) & jnp.all(y <= out.upper + TOL)

    results = jax.vmap(ok)(samples)
    n_failed = int(jnp.sum(~results))
    assert n_failed == 0, f"{n_failed}/{N_SAMPLES} samples fell outside backward-CROWN bounds"


def test_iterated_crown_pointwise(net_and_ix):
    net, ix = net_and_ix
    cr = crown(net, iterated=True)(ix)
    out = cr(ix)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(100))

    def ok(x):
        y = net(x)
        return jnp.all(out.lower - TOL <= y) & jnp.all(y <= out.upper + TOL)

    results = jax.vmap(ok)(samples)
    n_failed = int(jnp.sum(~results))
    assert n_failed == 0, f"{n_failed}/{N_SAMPLES} samples fell outside iterated-CROWN bounds"


# ---------------------------------------------------------------------------
# Tightness: backward CROWN should beat forward linbp on average for
# multi-layer ReLU networks. (Iterated isn't strictly tighter per-bound, so
# only checked on the *aggregate* width.)
# ---------------------------------------------------------------------------

def _width(out):
    return float(jnp.sum(out.upper - out.lower))


def test_backward_at_least_as_tight_aggregate():
    """Across a battery of multi-layer ReLU networks, total width from
    backward CROWN should be no worse than forward CROWN.

    Strict per-network monotonicity is not guaranteed when the adaptive
    {0,1} slope choices differ, but the aggregate must hold."""
    fwd_total = 0.0
    bwd_total = 0.0
    for k in range(5):
        for arch in ([4, 8, 4, 1], [3, 8, 8, 2], [2, 5, 5, 5, 1]):
            net = _build_net(arch, 'relu', jax.random.PRNGKey(k))
            ix = _make_interval('unit', arch[0])
            fwd_total += _width(crown(net, backward=False)(ix)(ix))
            bwd_total += _width(crown(net, backward=True)(ix)(ix))
    assert bwd_total <= fwd_total + TOL, (
        f"aggregate width regressed: fwd={fwd_total:.4f} bwd={bwd_total:.4f}"
    )


def test_backward_strictly_tighter_on_deep_relu():
    """On a deep ReLU network forward CROWN is known to be loose; backward
    should give a meaningful improvement (at least 1.5x tighter total width)."""
    fwd_total = 0.0
    bwd_total = 0.0
    for k in range(3):
        net = _build_net([3, 16, 16, 16, 16, 1], 'relu', jax.random.PRNGKey(k))
        ix = _make_interval('unit', 3)
        fwd_total += _width(crown(net, backward=False)(ix)(ix))
        bwd_total += _width(crown(net, backward=True)(ix)(ix))
    assert bwd_total * 1.5 <= fwd_total, (
        f"expected ≥1.5x tightening from backward CROWN on deep ReLU: "
        f"fwd={fwd_total:.4f} bwd={bwd_total:.4f}"
    )


# ---------------------------------------------------------------------------
# Direct linbp_backward smoke tests
# ---------------------------------------------------------------------------

def test_linbp_backward_returns_linearbound(arch_act, net_key):
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('small', arch[0])

    lb = linbp_backward(net, relu_mode='adaptive')(ix)
    n_in, n_out = arch[0], arch[-1]
    assert lb.lA.shape == (n_out, n_in)
    assert lb.uA.shape == (n_out, n_in)
    assert lb.lb.shape == (n_out,)
    assert lb.ub.shape == (n_out,)
    assert jnp.all(lb.l <= lb.u)


def test_linbp_backward_same_slope_lA_equals_uA(arch_act, net_key):
    """With ``same-slope`` and a function whose output is built only via
    monotone-positive-slope activations (ReLU/sigmoid/tanh), backward should
    keep ``lA == uA`` throughout, so the resulting bound has a single C."""
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('small', arch[0])
    lb = linbp_backward(net, relu_mode='same-slope')(ix)
    assert jnp.allclose(lb.lA, lb.uA), "same-slope backward must have lA == uA"


# ---------------------------------------------------------------------------
# Backward CROWN dot_general(LinearBound, LinearBound): bilinear concretization
# ---------------------------------------------------------------------------

def test_backward_dot_general_lb_lb_zero_amatrices():
    """Bilinear ``y = x @ x`` has no exact linear bound in x; the backward
    handler must concretize, yielding ``lA = uA = 0`` and a constant interval."""
    n_in = 4
    ix = irx.icentpert(jnp.array([1.0, -0.5, 2.0, 0.0]), 0.5)
    lb = linbp_backward(lambda x: jnp.dot(x, x))(ix)
    # backward CROWN follows the forward (*S_out, *S_in) convention; scalar
    # output ⇒ S_out = (), so lA/uA shape == (n_in,).
    assert lb.lA.shape == (n_in,)
    assert lb.uA.shape == (n_in,)
    assert jnp.allclose(lb.lA, jnp.zeros_like(lb.lA))
    assert jnp.allclose(lb.uA, jnp.zeros_like(lb.uA))


def test_backward_dot_general_lb_lb_soundness():
    """Concretized bound on ``y = x @ x`` must contain all sampled outputs."""
    n_in = 4
    ix = irx.icentpert(jnp.array([1.0, -0.5, 2.0, 0.0]), 0.5)
    lb = linbp_backward(lambda x: jnp.dot(x, x))(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(11))
    ys = jax.vmap(lambda x: jnp.dot(x, x))(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)


def test_backward_dot_general_lb_lb_quadratic_form_soundness():
    """Bilinear ``y = (Wx) @ x`` after a linear layer must still be sound."""
    n_in = 3
    key = jax.random.PRNGKey(2)
    W = jax.random.normal(key, (n_in, n_in))
    ix = irx.icentpert(jnp.array([0.5, -0.3, 0.8]), 0.4)

    def f(x):
        return jnp.dot(W @ x, x)

    lb = linbp_backward(f)(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(3))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)
    # The bound must be a constant interval (no linear dependence preserved).
    assert jnp.allclose(lb.lA, jnp.zeros_like(lb.lA))
    assert jnp.allclose(lb.uA, jnp.zeros_like(lb.uA))


# ---------------------------------------------------------------------------
# Backward CROWN split: multi-output linear primitive
# ---------------------------------------------------------------------------

def test_backward_split_identity():
    """``concat(split(x)) == x``: backward CROWN of this should give an exact
    affine identity (lA = uA = I, biases zero, l = x_lb, u = x_ub)."""
    n_in = 5
    ix = irx.icentpert(jnp.arange(1.0, n_in + 1), 0.5)

    def f(x):
        a, b = jnp.split(x, [2])
        return jnp.concatenate([a, b])

    lb = linbp_backward(f)(ix)
    assert jnp.allclose(lb.lA, jnp.eye(n_in), atol=1e-6)
    assert jnp.allclose(lb.uA, jnp.eye(n_in), atol=1e-6)
    assert jnp.allclose(lb.lb, jnp.zeros(n_in), atol=1e-6)
    assert jnp.allclose(lb.ub, jnp.zeros(n_in), atol=1e-6)


def test_backward_split_then_linear_soundness():
    """``y = W_a @ a + W_b @ b`` where ``a, b = split(x, ...)``. Sound bounds."""
    n_in = 6
    split_idx = 2
    n_a, n_b = split_idx, n_in - split_idx
    n_out = 3
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    Wa = jax.random.normal(k1, (n_out, n_a))
    Wb = jax.random.normal(k2, (n_out, n_b))
    ix = irx.icentpert(jnp.arange(1.0, n_in + 1) * 0.1, 0.3)

    def f(x):
        a, b = jnp.split(x, [split_idx])
        return Wa @ a + Wb @ b

    lb = linbp_backward(f)(ix)
    assert lb.lA.shape == (n_out, n_in)
    assert lb.uA.shape == (n_out, n_in)

    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(1))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)


def test_backward_split_three_chunks_soundness():
    """``split(x, [2, 5])`` yields three chunks; each contributes to a vector
    output through its own linear / nonlinear pipeline."""
    n_in = 8
    ix = irx.icentpert(jnp.arange(1.0, n_in + 1) * 0.2, 0.2)
    Wa = jnp.array([[1.0, -1.0], [0.5, 0.5]])           # (2, 2)
    Wb = jnp.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]) # (2, 3)
    Wc = jnp.array([[1.0, -1.0, 0.5], [0.0, 1.0, 1.0]]) # (2, 3)

    def f(x):
        a, b, c = jnp.split(x, [2, 5])
        # Mix of activations through each chunk; avoid LB*LB (separate handler).
        return Wa @ jax.nn.relu(a) + Wb @ jax.nn.relu(-b) + Wc @ jax.nn.relu(c)

    lb = linbp_backward(f)(ix)
    assert lb.lA.shape == (2, n_in)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(2))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL), f"lower violated: ys.min={ys.min(0)} l={lb.l}"
    assert jnp.all(ys <= lb.u + TOL), f"upper violated: ys.max={ys.max(0)} u={lb.u}"


def test_backward_split_only_one_chunk_used():
    """Use only the first chunk; the second's BackwardBound is never created.
    The handler must zero-fill the missing outvar's cotangent."""
    n_in = 6
    split_idx = 3
    n_a = split_idx
    ix = irx.icentpert(jnp.arange(1.0, n_in + 1) * 0.1, 0.4)
    w = jnp.array([1.0, 2.0, -0.5])  # weights for the first chunk

    def f(x):
        a, _b = jnp.split(x, [split_idx])
        return w @ a  # linear; depends only on first chunk

    lb = linbp_backward(f)(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(3))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)
    # Linear function ⇒ tight: lA == uA, with weights on first chunk and zeros on second
    assert jnp.allclose(lb.lA, lb.uA, atol=1e-6)
    expected_A = jnp.concatenate([w, jnp.zeros(n_in - n_a)])
    # f outputs a scalar (w @ a), so lb.lA has shape (n_in,) under the
    # (*S_out, *S_in) convention.
    assert jnp.allclose(lb.lA, expected_A, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward CROWN add_any: produced by JAX autodiff (vjp/grad)
# ---------------------------------------------------------------------------

def test_backward_add_any_direct_injection():
    """``add_any_p`` is semantically identical to ``add_p`` and should pass
    through the same backward handler. Inject one directly via the primitive."""
    from jax._src import ad_util
    n = 4
    ix = irx.icentpert(jnp.arange(1.0, n + 1) * 0.1, 0.3)
    W = jnp.array([[1.0, -1.0, 0.5, 0.0], [0.0, 2.0, -0.5, 1.0]])

    def f(x):
        a = 2.0 * x
        b = -0.5 * x
        y = ad_util.add_any_p.bind(a, b)
        return W @ y

    # Sanity: add_any must actually appear in the traced jaxpr.
    import equinox as eqx
    closed = eqx.filter_make_jaxpr(f)(jnp.zeros(n))[0]
    prims = {eqn.primitive.name for eqn in closed.jaxpr.eqns}
    assert "add_any" in prims, f"expected add_any in jaxpr; got {prims}"

    lb = linbp_backward(f)(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(0))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)
    # Linear function ⇒ tight: lA == uA, and lA matches W * (2 + (-0.5)) = W * 1.5.
    assert jnp.allclose(lb.lA, lb.uA, atol=1e-6)
    assert jnp.allclose(lb.lA, 1.5 * W, atol=1e-6)


def test_backward_neg_soundness_after_nonlinear_path():
    """``y = neg(x)`` must propagate biases unchanged (not swap lo↔hi).
    Regression: prior implementation swapped A_lo/A_hi and negated biases,
    which was silently OK when lA == uA (linear-only paths) but produced
    sign-flipped, unsound bounds once a nonlinear primitive (here ``mul`` of
    two LinearBounds via jacfwd) introduced a ``b_lo != b_hi`` along the path.
    """
    ix = irx.icentpert(jnp.array([1.0, 0.5]), 0.3)

    def f(x):
        return jax.jacfwd(lambda y: -y[0] * jnp.sin(y[1]))(x)

    lb = linbp_backward(f)(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(0))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL), f"lower violated: ys.min={ys.min(0)} l={lb.l}"
    assert jnp.all(ys <= lb.u + TOL), f"upper violated: ys.max={ys.max(0)} u={lb.u}"


def test_backward_add_any_both_lb_operands():
    """``add_any(a, b)`` with both ``a, b`` LinearBounds. Both must receive
    a BackwardBound; bias attributed to first only to avoid double counting."""
    from jax._src import ad_util
    n = 3
    ix = irx.icentpert(jnp.array([0.5, -0.3, 0.8]), 0.4)
    W = jnp.array([[1.0, -1.0, 0.5], [0.0, 1.0, 1.0]])

    def f(x):
        a = jax.nn.relu(x)             # nonlinear LB
        b = jax.nn.relu(-x)            # nonlinear LB
        y = ad_util.add_any_p.bind(a, b)  # = |x|
        return W @ y

    import equinox as eqx
    closed = eqx.filter_make_jaxpr(f)(jnp.zeros(n))[0]
    prims = {eqn.primitive.name for eqn in closed.jaxpr.eqns}
    assert "add_any" in prims, f"expected add_any in jaxpr; got {prims}"

    lb = linbp_backward(f)(ix)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(1))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL)
    assert jnp.all(ys <= lb.u + TOL)


def test_backward_dot_general_lb_lb_vector_output_soundness():
    """Vector-output bilinear: y[i] = (W_i @ x) @ x for n_out output rows.
    Exercises backward CROWN through dot_general(LB, LB) with n_out > 1."""
    n_in, n_out = 3, 4
    key = jax.random.PRNGKey(5)
    Ws = jax.random.normal(key, (n_out, n_in, n_in))
    ix = irx.icentpert(jnp.array([0.5, -0.2, 0.7]), 0.3)

    def f(x):
        # For each row i, compute (W_i @ x) @ x — bilinear in x.
        return jax.vmap(lambda W: (W @ x) @ x)(Ws)

    lb = linbp_backward(f)(ix)
    assert lb.lA.shape == (n_out, n_in)
    assert lb.uA.shape == (n_out, n_in)
    xs = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(6))
    ys = jax.vmap(f)(xs)
    assert jnp.all(ys >= lb.l - TOL), f"lower violated: ys.min={ys.min(0)} l={lb.l}"
    assert jnp.all(ys <= lb.u + TOL), f"upper violated: ys.max={ys.max(0)} u={lb.u}"
