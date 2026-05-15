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
