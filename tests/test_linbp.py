"""Tests for forward linear bound propagation (linbp).

Verifies that crown() and fastlin() produce valid overapproximations
of neural network outputs for randomly initialized networks.
"""

import jax
import jax.numpy as jnp
import equinox.nn as nn
import pytest

import immrax as irx
from immrax import interval, crown, fastlin
from immrax.inclusion.linbp import LinearBound, linbp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 200  # random points sampled per test


def _sample_in_interval(ix: irx.Interval, n: int, key: jax.Array) -> jax.Array:
    """Draw n uniform samples from the hyperrectangle ix."""
    shape = (n, *ix.lower.shape)
    u = jax.random.uniform(key, shape)
    return ix.lower + u * (ix.upper - ix.lower)


def _build_net(arch: list[int], activation: str, key: jax.Array):
    """Build an equinox Sequential net from an architecture list.

    arch = [n_in, h1, h2, ..., n_out]
    activation = 'relu' | 'sigmoid' | 'tanh'
    """
    act_fns = {
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'tanh': lambda x: 2 * jax.nn.sigmoid(2 * x) - 1,
    }
    act_fn = act_fns[activation]
    layers = []
    for i in range(len(arch) - 1):
        key, subkey = jax.random.split(key)
        layers.append(nn.Linear(arch[i], arch[i + 1], key=subkey))
        if i < len(arch) - 2:  # no activation after last layer
            layers.append(nn.Lambda(act_fn))
    return nn.Sequential(layers)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ARCH_ACTIVATION_PARAMS = [
    pytest.param(([2, 4, 1], 'relu'),    id="2-4-1/relu"),
    pytest.param(([3, 8, 4, 2], 'relu'), id="3-8-4-2/relu"),
    pytest.param(([2, 4, 2], 'relu'),    id="2-4-2/relu"),
    pytest.param(([2, 6, 1], 'sigmoid'), id="2-6-1/sigmoid"),
    pytest.param(([2, 6, 2], 'tanh'),    id="2-6-2/tanh"),
    pytest.param(([4, 8, 4, 2], 'relu'), id="4-8-4-2/relu"),
]

_NET_KEY_PARAMS = [
    pytest.param(0, id="key0"),
    pytest.param(1, id="key1"),
    pytest.param(7, id="key7"),
]

_INPUT_INTERVAL_PARAMS = [
    pytest.param('unit',  id="unit-box"),   # [-1,1]^n
    pytest.param('small', id="small-box"),  # center±0.2
    pytest.param('asym',  id="asym-box"),   # asymmetric
]


def _make_interval(kind: str, n_in: int) -> irx.Interval:
    if kind == 'unit':
        return interval(jnp.full(n_in, -1.0), jnp.full(n_in, 1.0))
    elif kind == 'small':
        center = jnp.arange(1, n_in + 1, dtype=jnp.float32) * 0.5
        return irx.icentpert(center, 0.2)
    else:  # asym
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
    n_in = arch[0]
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval(input_kind, n_in)
    return net, ix


# ---------------------------------------------------------------------------
# Tests: LinearBound / linbp
# ---------------------------------------------------------------------------

def test_linbp_returns_linearbound(arch_act, net_key):
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    n_in, n_out = arch[0], arch[-1]
    ix = _make_interval('unit', n_in)

    lb = linbp(net, relu_mode='adaptive')(ix)

    assert isinstance(lb, LinearBound)
    assert lb.lA.shape == (n_out, n_in)
    assert lb.uA.shape == (n_out, n_in)
    assert lb.lb.shape == (n_out,)
    assert lb.ub.shape == (n_out,)
    assert lb.l.shape  == (n_out,)
    assert lb.u.shape  == (n_out,)


def test_linbp_pytree(arch_act, net_key):
    """LinearBound survives a JAX pytree round-trip."""
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('unit', arch[0])
    lb = linbp(net, relu_mode='adaptive')(ix)

    leaves, treedef = jax.tree_util.tree_flatten(lb)
    lb2 = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jnp.allclose(lb.lA, lb2.lA)
    assert jnp.allclose(lb.ub, lb2.ub)


def test_linbp_bounds_valid(arch_act, net_key):
    """Concrete bounds l <= u hold at the output.

    Note: lb and ub are the constant offsets of separate affine bounds
    (lA @ x + lb <= y <= uA @ x + ub) and are not required to satisfy
    lb <= ub pointwise — the linear terms can compensate.
    """
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('small', arch[0])
    lb = linbp(net, relu_mode='adaptive')(ix)

    assert jnp.all(lb.l <= lb.u), "Concrete lower must be <= upper"


# ---------------------------------------------------------------------------
# Tests: crown() overapproximation
# ---------------------------------------------------------------------------

def test_crown_output_type(net_and_ix):
    net, ix = net_and_ix
    crown_fn = crown(net)
    result = crown_fn(ix)

    assert isinstance(result, irx.CROWNResult)
    n_in = ix.lower.size
    n_out = result.lC.shape[0]
    assert result.lC.shape == (n_out, n_in)
    assert result.uC.shape == (n_out, n_in)
    assert result.ld.shape == (n_out,)
    assert result.ud.shape == (n_out,)


def test_crown_interval_valid(net_and_ix):
    """The evaluated CROWN interval has lower <= upper."""
    net, ix = net_and_ix
    crown_fn = crown(net)
    cr = crown_fn(ix)
    result = cr(ix)

    assert isinstance(result, irx.Interval)
    assert jnp.all(result.lower <= result.upper)


def test_crown_contains_output(net_and_ix):
    """CROWN interval contains net(x) for all x sampled from ix."""
    net, ix = net_and_ix
    crown_fn = crown(net)
    cr = crown_fn(ix)
    cr_interval = cr(ix)

    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(net)(samples)  # (N_SAMPLES, n_out)

    tol = 1e-5
    assert jnp.all(cr_interval.lower - tol <= outputs.min(axis=0)), (
        f"Crown lower bound violated.\n"
        f"  bound: {cr_interval.lower}\n"
        f"  min output: {outputs.min(axis=0)}"
    )
    assert jnp.all(outputs.max(axis=0) <= cr_interval.upper + tol), (
        f"Crown upper bound violated.\n"
        f"  max output: {outputs.max(axis=0)}\n"
        f"  bound: {cr_interval.upper}"
    )


def test_crown_pointwise_contains(net_and_ix):
    """For each individual sample, net(x) lies within the CROWN interval."""
    net, ix = net_and_ix
    crown_fn = crown(net)
    cr = crown_fn(ix)
    cr_interval = cr(ix)

    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(99))
    tol = 1e-5

    def check_sample(x):
        y = net(x)
        lower_ok = jnp.all(cr_interval.lower - tol <= y)
        upper_ok = jnp.all(y <= cr_interval.upper + tol)
        return lower_ok & upper_ok

    results = jax.vmap(check_sample)(samples)
    n_failed = jnp.sum(~results)
    assert n_failed == 0, f"{n_failed}/{N_SAMPLES} samples fell outside CROWN bounds"


# ---------------------------------------------------------------------------
# Tests: fastlin() overapproximation
# ---------------------------------------------------------------------------

def test_fastlin_output_type(net_and_ix):
    net, ix = net_and_ix
    fl_fn = fastlin(net)
    result = fl_fn(ix)

    assert isinstance(result, irx.FastlinResult)
    n_in = ix.lower.size
    n_out = result.C.shape[0]
    assert result.C.shape == (n_out, n_in)
    assert result.ld.shape == (n_out,)
    assert result.ud.shape == (n_out,)


def test_fastlin_interval_valid(net_and_ix):
    net, ix = net_and_ix
    fl_fn = fastlin(net)
    fl = fl_fn(ix)
    result = fl(ix)

    assert isinstance(result, irx.Interval)
    assert jnp.all(result.lower <= result.upper)


def test_fastlin_contains_output(net_and_ix):
    """FastLin interval contains net(x) for all x sampled from ix."""
    net, ix = net_and_ix
    fl_fn = fastlin(net)
    fl = fl_fn(ix)
    fl_interval = fl(ix)

    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(net)(samples)

    tol = 1e-5
    assert jnp.all(fl_interval.lower - tol <= outputs.min(axis=0)), (
        f"FastLin lower bound violated.\n"
        f"  bound: {fl_interval.lower}\n"
        f"  min output: {outputs.min(axis=0)}"
    )
    assert jnp.all(outputs.max(axis=0) <= fl_interval.upper + tol), (
        f"FastLin upper bound violated.\n"
        f"  max output: {outputs.max(axis=0)}\n"
        f"  bound: {fl_interval.upper}"
    )


@pytest.mark.parametrize("arch_act,net_key", [
    (([2, 4, 1], 'relu'), 0),
    (([3, 8, 4, 2], 'relu'), 1),
    (([2, 4, 2], 'relu'), 7),
    (([4, 8, 4, 2], 'relu'), 0),
])
def test_fastlin_symmetric_matrix(arch_act, net_key):
    """FastLin uses lA == uA (the shared ReLU slope is the same for upper/lower).

    Only tested for ReLU networks: sigmoid/tanh fall back to IBP (lA=uA=0
    trivially), which isn't an interesting check.
    """
    arch, activation = arch_act
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('unit', arch[0])

    lb = linbp(net, relu_mode='same-slope')(ix)
    assert jnp.allclose(lb.lA, lb.uA, atol=1e-6), (
        "FastLin should have lA == uA (parallel upper/lower slopes)"
    )


# ---------------------------------------------------------------------------
# Tests: general max/min (non-zero threshold)
# ---------------------------------------------------------------------------

_GENERAL_FUNCTIONS = [
    pytest.param(
        lambda x: jnp.maximum(x, 0.5),
        interval(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        id="max_x_0.5",
    ),
    pytest.param(
        lambda x: jnp.minimum(x, 0.3),
        interval(jnp.array([-0.5, -0.5]), jnp.array([1.0, 1.0])),
        id="min_x_0.3",
    ),
    pytest.param(
        lambda x: jnp.clip(x, -0.2, 0.8),
        interval(jnp.array([-1.0, -0.5]), jnp.array([0.5, 1.5])),
        id="clip_-0.2_0.8",
    ),
    pytest.param(
        lambda x: jnp.maximum(x, -1.0) + jnp.minimum(x, 2.0),
        interval(jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        id="max_plus_min",
    ),
]


@pytest.mark.parametrize("fn,ix", _GENERAL_FUNCTIONS)
@pytest.mark.parametrize("relu_mode", ["adaptive", "same-slope"])
def test_general_fn_bounds_valid(fn, ix, relu_mode):
    """Concrete bounds l <= u hold for general functions with max/min."""
    lb = linbp(fn, relu_mode=relu_mode)(ix)
    assert jnp.all(lb.l <= lb.u), "Concrete lower must be <= upper"


@pytest.mark.parametrize("fn,ix", _GENERAL_FUNCTIONS)
@pytest.mark.parametrize("relu_mode", ["adaptive", "same-slope"])
def test_general_fn_contains_output(fn, ix, relu_mode):
    """Linear bounds contain f(x) for all x sampled from ix."""
    from immrax.inclusion.linbp import _concretize

    lb = linbp(fn, relu_mode=relu_mode)(ix)
    result = _concretize(lb, ix.lower, ix.upper)

    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(fn)(samples)  # (N_SAMPLES, *out_shape)

    tol = 1e-5
    assert jnp.all(result.lower - tol <= outputs.min(axis=0)), (
        f"[{relu_mode}] lower bound violated.\n"
        f"  bound: {result.lower}\n  min output: {outputs.min(axis=0)}"
    )
    assert jnp.all(outputs.max(axis=0) <= result.upper + tol), (
        f"[{relu_mode}] upper bound violated.\n"
        f"  max output: {outputs.max(axis=0)}\n  bound: {result.upper}"
    )


# ---------------------------------------------------------------------------
# Tests: sigmoid/logistic linear relaxation correctness
# ---------------------------------------------------------------------------

_SIGMOID_INTERVALS = [
    pytest.param((-2.0, -0.5), id="neg-neg"),        # fully convex region
    pytest.param((0.5, 2.0),   id="pos-pos"),         # fully concave region
    pytest.param((-1.0, 1.0),  id="sym-mixed"),       # symmetric straddles inflection
    pytest.param((-1.0, 2.0),  id="mixed-wide-pos"),  # |u| > |l|
    pytest.param((-2.0, 0.5),  id="mixed-wide-neg"),  # |l| > |u|
    pytest.param((-3.0, 3.0),  id="large-sym"),       # large symmetric
    pytest.param((-0.01, 0.01), id="tiny"),           # near-degenerate interval
]

N_DENSE = 2000  # dense grid for pointwise bound checks


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_logistic_upper_bound_valid(lu):
    """Chord-based sigmoid upper affine bound is >= sigma(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jax.nn.sigmoid, relu_mode='same-slope')(ix)

    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    upper_line = alpha_u * xs + beta_u
    sig = jax.nn.sigmoid(xs)

    tol = 1e-5
    violation = jnp.max(sig - upper_line)
    assert violation <= tol, (
        f"Sigmoid upper bound violated on [{l_val}, {u_val}].\n"
        f"  alpha_u={alpha_u:.6f}, beta_u={beta_u:.6f}\n"
        f"  max violation: {float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_logistic_lower_bound_valid(lu):
    """Chord-based sigmoid lower affine bound is <= sigma(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jax.nn.sigmoid, relu_mode='same-slope')(ix)

    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    lower_line = alpha_l * xs + beta_l
    sig = jax.nn.sigmoid(xs)

    tol = 1e-5
    violation = jnp.max(lower_line - sig)
    assert violation <= tol, (
        f"Sigmoid lower bound violated on [{l_val}, {u_val}].\n"
        f"  alpha_l={alpha_l:.6f}, beta_l={beta_l:.6f}\n"
        f"  max violation: {float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_logistic_nontrivial_slope(lu):
    """The sigmoid relaxation preserves a non-zero slope (lA != 0)."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jax.nn.sigmoid, relu_mode='same-slope')(ix)

    assert float(jnp.abs(lb.lA[0, 0])) > 1e-8, (
        f"Sigmoid lA should be non-zero on [{l_val}, {u_val}], got {float(lb.lA[0, 0])}"
    )


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_tanh_upper_bound_valid(lu):
    """Chord-based tanh (2*sigmoid(2x)-1) upper bound is valid."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    tanh_fn = lambda x: 2 * jax.nn.sigmoid(2 * x) - 1
    lb = linbp(tanh_fn, relu_mode='same-slope')(ix)

    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    upper_line = alpha_u * xs + beta_u
    tanh_vals = jnp.tanh(xs)

    tol = 1e-5
    violation = jnp.max(tanh_vals - upper_line)
    assert violation <= tol, (
        f"Tanh upper bound violated on [{l_val}, {u_val}].\n"
        f"  max violation: {float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_tanh_lower_bound_valid(lu):
    """Chord-based tanh (2*sigmoid(2x)-1) lower bound is valid."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    tanh_fn = lambda x: 2 * jax.nn.sigmoid(2 * x) - 1
    lb = linbp(tanh_fn, relu_mode='same-slope')(ix)

    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    lower_line = alpha_l * xs + beta_l
    tanh_vals = jnp.tanh(xs)

    tol = 1e-5
    violation = jnp.max(lower_line - tanh_vals)
    assert violation <= tol, (
        f"Tanh lower bound violated on [{l_val}, {u_val}].\n"
        f"  max violation: {float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIGMOID_INTERVALS)
def test_logistic_concrete_bounds_valid(lu):
    """Concrete bounds l, u from logistic handler contain sigma([l_val, u_val])."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jax.nn.sigmoid, relu_mode='same-slope')(ix)

    sig_l, sig_u = float(jax.nn.sigmoid(l_val)), float(jax.nn.sigmoid(u_val))
    tol = 1e-6
    assert float(lb.l[0]) <= sig_l + tol, (
        f"Concrete lower {lb.l[0]:.6f} > sigma(l) = {sig_l:.6f}"
    )
    assert float(lb.u[0]) >= sig_u - tol, (
        f"Concrete upper {lb.u[0]:.6f} < sigma(u) = {sig_u:.6f}"
    )


# ---------------------------------------------------------------------------
# Tests: LinearBound input to linbp
# ---------------------------------------------------------------------------

def test_linbp_accepts_linearbound_input(arch_act, net_key):
    """linbp's returned function accepts a LinearBound and produces a valid result."""
    arch, activation = arch_act
    n_in, n_out = arch[0], arch[-1]
    net = _build_net(arch, activation, jax.random.PRNGKey(net_key))
    ix = _make_interval('small', n_in)

    # Get a LinearBound from the first call
    lb_in = linbp(net, relu_mode='adaptive')(ix)

    # Build a trivial identity function to re-propagate through
    identity = lambda x: x
    lb_out = linbp(identity, relu_mode='adaptive')(lb_in)

    # Identity should leave the LinearBound unchanged
    assert jnp.allclose(lb_out.lA, lb_in.lA, atol=1e-6)
    assert jnp.allclose(lb_out.uA, lb_in.uA, atol=1e-6)
    assert jnp.allclose(lb_out.lb, lb_in.lb, atol=1e-6)
    assert jnp.allclose(lb_out.ub, lb_in.ub, atol=1e-6)


def test_linbp_chained_contains_output(arch_act, net_key):
    """Chaining two linbp calls via LinearBound gives valid bounds."""
    arch, activation = arch_act
    n_in, n_out = arch[0], arch[-1]

    # Build two separate networks of the same width to chain
    net1 = _build_net([n_in, 6, 4], activation, jax.random.PRNGKey(net_key))
    net2 = _build_net([4, 4, n_out], activation, jax.random.PRNGKey(net_key + 100))
    chained = lambda x: net2(net1(x))

    ix = _make_interval('small', n_in)

    # Chain via LinearBound: linbp(net2)(linbp(net1)(ix))
    lb1 = linbp(net1, relu_mode='adaptive')(ix)
    lb2 = linbp(net2, relu_mode='adaptive')(lb1)

    # Concretize to an interval
    from immrax.inclusion.linbp import _concretize
    result = _concretize(lb2, ix.lower, ix.upper)

    # Sample and verify containment
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(chained)(samples)

    tol = 1e-5
    assert jnp.all(result.lower - tol <= outputs.min(axis=0)), (
        f"Chained lower bound violated.\n"
        f"  bound: {result.lower}\n  min output: {outputs.min(axis=0)}"
    )
    assert jnp.all(outputs.max(axis=0) <= result.upper + tol), (
        f"Chained upper bound violated.\n"
        f"  max output: {outputs.max(axis=0)}\n  bound: {result.upper}"
    )


def test_resolve_linbp_input_type_error():
    """_resolve_linbp_input raises TypeError for unsupported inputs."""
    from immrax.inclusion.linbp import _resolve_linbp_input
    import pytest
    with pytest.raises(TypeError, match="Interval or LinearBound"):
        _resolve_linbp_input(jnp.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tests: neg / sub / mul / div arithmetic primitives
# ---------------------------------------------------------------------------

_ARITH_FUNCTIONS = [
    pytest.param(
        lambda x: -x,
        interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5])),
        id="neg",
    ),
    pytest.param(
        lambda x: x - 1.0,
        interval(jnp.array([-2.0, -1.0]), jnp.array([1.0, 2.0])),
        id="sub-constant",
    ),
    pytest.param(
        lambda x: 1.0 - x,
        interval(jnp.array([-0.5, -0.5]), jnp.array([2.0, 2.0])),
        id="sub-constant-reverse",
    ),
    pytest.param(
        lambda x: 3.0 * x,
        interval(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        id="mul-pos-scalar",
    ),
    pytest.param(
        lambda x: -2.0 * x,
        interval(jnp.array([-1.0, 0.0]), jnp.array([0.5, 1.0])),
        id="mul-neg-scalar",
    ),
    pytest.param(
        lambda x: x / 4.0,
        interval(jnp.array([-2.0, -1.0]), jnp.array([1.0, 2.0])),
        id="div-pos-scalar",
    ),
    pytest.param(
        lambda x: x / -3.0,
        interval(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        id="div-neg-scalar",
    ),
]


@pytest.mark.parametrize("fn,ix", _ARITH_FUNCTIONS)
def test_arith_bounds_valid(fn, ix):
    """Concrete bounds l <= u hold for arithmetic primitives."""
    lb = linbp(fn)(ix)
    assert jnp.all(lb.l <= lb.u)


@pytest.mark.parametrize("fn,ix", _ARITH_FUNCTIONS)
def test_arith_contains_output(fn, ix):
    """Arithmetic op linear bounds contain f(x) for all x in ix."""
    from immrax.inclusion.linbp import _concretize
    lb = linbp(fn)(ix)
    result = _concretize(lb, ix.lower, ix.upper)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(fn)(samples)
    tol = 1e-5
    assert jnp.all(result.lower - tol <= outputs.min(axis=0))
    assert jnp.all(outputs.max(axis=0) <= result.upper + tol)


def test_neg_flips_concrete_bounds():
    """Negation exactly flips concrete bounds: l_out = -u_in, u_out = -l_in."""
    ix = interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5]))
    lb = linbp(lambda x: -x)(ix)
    assert jnp.allclose(lb.l, -ix.upper, atol=1e-6)
    assert jnp.allclose(lb.u, -ix.lower, atol=1e-6)


def test_mul_positive_scalar_exact():
    """Scaling by a positive constant exactly scales the concrete bounds."""
    c = 3.0
    ix = interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5]))
    lb = linbp(lambda x: c * x)(ix)
    assert jnp.allclose(lb.l, c * ix.lower, atol=1e-6)
    assert jnp.allclose(lb.u, c * ix.upper, atol=1e-6)


def test_mul_negative_scalar_flips():
    """Scaling by a negative constant flips the concrete bounds."""
    c = -2.0
    ix = interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5]))
    lb = linbp(lambda x: c * x)(ix)
    assert jnp.allclose(lb.l, c * ix.upper, atol=1e-6)
    assert jnp.allclose(lb.u, c * ix.lower, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: sin linear relaxation
# ---------------------------------------------------------------------------

_SIN_INTERVALS = [
    pytest.param((-0.5, 0.5),           id="small-sym"),
    pytest.param((0.0, 1.0),            id="monotone-inc"),
    pytest.param((1.0, 2.5),            id="straddles-max"),    # contains π/2 ≈ 1.57
    pytest.param((-2.5, -1.0),          id="straddles-min"),    # contains -π/2 ≈ -1.57
    pytest.param((-2.0, 2.0),           id="both-extrema"),     # contains both ±π/2
    pytest.param((0.0, 7.0),            id="wide-over-2pi"),    # > 2π: fallback α=0
    pytest.param((1.0, 1.0 + 1e-6),    id="degenerate"),
]


@pytest.mark.parametrize("lu", _SIN_INTERVALS)
def test_sin_upper_bound_valid(lu):
    """sin upper affine bound alpha_u*x + beta_u >= sin(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.sin)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(jnp.sin(xs) - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"sin upper bound violated on [{l_val}, {u_val}]: "
        f"alpha_u={alpha_u:.6f}, beta_u={beta_u:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIN_INTERVALS)
def test_sin_lower_bound_valid(lu):
    """sin lower affine bound alpha_l*x + beta_l <= sin(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.sin)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - jnp.sin(xs))
    assert float(violation) <= 1e-5, (
        f"sin lower bound violated on [{l_val}, {u_val}]: "
        f"alpha_l={alpha_l:.6f}, beta_l={beta_l:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _SIN_INTERVALS)
def test_sin_concrete_bounds_valid(lu):
    """Concrete l, u from sin handler contain sin([l, u])."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.sin)(ix)
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    vals = jnp.sin(xs)
    tol = 1e-6
    assert float(lb.l[0]) <= float(jnp.min(vals)) + tol, (
        f"sin concrete lower {lb.l[0]:.6f} > min sin={float(jnp.min(vals)):.6f} on [{l_val}, {u_val}]"
    )
    assert float(lb.u[0]) >= float(jnp.max(vals)) - tol, (
        f"sin concrete upper {lb.u[0]:.6f} < max sin={float(jnp.max(vals)):.6f} on [{l_val}, {u_val}]"
    )


# ---------------------------------------------------------------------------
# Tests: cos linear relaxation
# ---------------------------------------------------------------------------

_COS_INTERVALS = [
    pytest.param((-0.5, 0.5),           id="small-sym"),         # near max at 0
    pytest.param((0.5, 2.0),            id="crossing-zero"),     # cos goes pos→neg
    pytest.param((2.5, 4.0),            id="straddles-cos-min"), # contains π ≈ 3.14
    pytest.param((-2.0, 2.0),           id="both-sides"),
    pytest.param((0.0, 7.0),            id="wide-over-2pi"),
    pytest.param((0.5, 0.5 + 1e-6),    id="degenerate"),
]


@pytest.mark.parametrize("lu", _COS_INTERVALS)
def test_cos_upper_bound_valid(lu):
    """cos upper affine bound alpha_u*x + beta_u >= cos(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.cos)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(jnp.cos(xs) - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"cos upper bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _COS_INTERVALS)
def test_cos_lower_bound_valid(lu):
    """cos lower affine bound alpha_l*x + beta_l <= cos(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.cos)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - jnp.cos(xs))
    assert float(violation) <= 1e-5, (
        f"cos lower bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _COS_INTERVALS)
def test_cos_concrete_bounds_valid(lu):
    """Concrete l, u from cos handler contain cos([l, u])."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.cos)(ix)
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    vals = jnp.cos(xs)
    tol = 1e-6
    assert float(lb.l[0]) <= float(jnp.min(vals)) + tol
    assert float(lb.u[0]) >= float(jnp.max(vals)) - tol


# ---------------------------------------------------------------------------
# Tests: tan linear relaxation
# ---------------------------------------------------------------------------

_TAN_INTERVALS = [
    pytest.param((-0.3, 0.3),        id="small-sym"),          # straddles inflection at 0
    pytest.param((0.1, 1.0),         id="fully-convex"),        # x > 0, convex region
    pytest.param((-1.0, -0.1),       id="fully-concave"),       # x < 0, concave region
    pytest.param((-0.5, 0.0),        id="neg-to-zero"),         # ends at inflection
    pytest.param((0.0, 0.5),         id="zero-to-pos"),         # starts at inflection
    pytest.param((-1.2, 1.2),        id="wide-sym"),            # wider symmetric
    pytest.param((0.5, 0.5 + 1e-6), id="degenerate"),
]


@pytest.mark.parametrize("lu", _TAN_INTERVALS)
def test_tan_upper_bound_valid(lu):
    """tan upper affine bound alpha_u*x + beta_u >= tan(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.tan)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(jnp.tan(xs) - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"tan upper bound violated on [{l_val}, {u_val}]: "
        f"alpha_u={alpha_u:.6f}, beta_u={beta_u:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _TAN_INTERVALS)
def test_tan_lower_bound_valid(lu):
    """tan lower affine bound alpha_l*x + beta_l <= tan(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.tan)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - jnp.tan(xs))
    assert float(violation) <= 1e-5, (
        f"tan lower bound violated on [{l_val}, {u_val}]: "
        f"alpha_l={alpha_l:.6f}, beta_l={beta_l:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _TAN_INTERVALS)
def test_tan_concrete_bounds_valid(lu):
    """Concrete l, u from tan handler contain tan([l, u]) (tan is monotone within a period)."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.tan)(ix)
    tol = 1e-6
    # tan is monotone on (-π/2, π/2), so min=tan(l), max=tan(u)
    assert float(lb.l[0]) <= float(jnp.tan(l_val)) + tol, (
        f"tan concrete lower {lb.l[0]:.6f} > tan(l)={float(jnp.tan(l_val)):.6f}"
    )
    assert float(lb.u[0]) >= float(jnp.tan(u_val)) - tol, (
        f"tan concrete upper {lb.u[0]:.6f} < tan(u)={float(jnp.tan(u_val)):.6f}"
    )


# ---------------------------------------------------------------------------
# Tests: exp linear relaxation
# ---------------------------------------------------------------------------

_EXP_INTERVALS = [
    pytest.param((-1.0, 1.0),    id="sym"),
    pytest.param((-2.0, 0.0),    id="neg-to-zero"),
    pytest.param((0.0, 2.0),     id="zero-to-pos"),
    pytest.param((-3.0, -1.0),   id="fully-neg"),
    pytest.param((1.0, 3.0),     id="pos-large"),
    pytest.param((0.0, 1e-6),    id="degenerate"),
]


@pytest.mark.parametrize("lu", _EXP_INTERVALS)
def test_exp_upper_bound_valid(lu):
    """exp upper bound (chord): alpha*x + beta_u >= exp(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.exp)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(jnp.exp(xs) - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"exp upper bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _EXP_INTERVALS)
def test_exp_lower_bound_valid(lu):
    """exp lower bound (parallel tangent): alpha*x + beta_l <= exp(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.exp)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - jnp.exp(xs))
    assert float(violation) <= 1e-5, (
        f"exp lower bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _EXP_INTERVALS)
def test_exp_concrete_bounds_exact(lu):
    """Concrete bounds from exp handler are exp(l_val) and exp(u_val)."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.exp)(ix)
    tol = 1e-6
    assert abs(float(lb.l[0]) - float(jnp.exp(l_val))) <= tol
    assert abs(float(lb.u[0]) - float(jnp.exp(u_val))) <= tol


# ---------------------------------------------------------------------------
# Tests: log1p linear relaxation
# ---------------------------------------------------------------------------

_LOG1P_INTERVALS = [
    pytest.param((-0.5, 0.5),   id="sym"),
    pytest.param((0.0, 2.0),    id="pos"),
    pytest.param((-0.8, -0.1),  id="near-neg1"),  # valid domain l > -1
    pytest.param((1.0, 5.0),    id="pos-large"),
    pytest.param((0.0, 1e-6),   id="degenerate"),
]


@pytest.mark.parametrize("lu", _LOG1P_INTERVALS)
def test_log1p_upper_bound_valid(lu):
    """log1p upper bound (tangent): alpha*x + beta_u >= log1p(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.log1p)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(jnp.log1p(xs) - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"log1p upper bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _LOG1P_INTERVALS)
def test_log1p_lower_bound_valid(lu):
    """log1p lower bound (chord): alpha*x + beta_l <= log1p(x) for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.log1p)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - jnp.log1p(xs))
    assert float(violation) <= 1e-5, (
        f"log1p lower bound violated on [{l_val}, {u_val}]: max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu", _LOG1P_INTERVALS)
def test_log1p_concrete_bounds_exact(lu):
    """Concrete bounds from log1p handler are log1p(l_val) and log1p(u_val)."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(jnp.log1p)(ix)
    tol = 1e-6
    assert abs(float(lb.l[0]) - float(jnp.log1p(l_val))) <= tol
    assert abs(float(lb.u[0]) - float(jnp.log1p(u_val))) <= tol


# ---------------------------------------------------------------------------
# Tests: structural primitives (reshape, concatenate, slice)
# ---------------------------------------------------------------------------

_STRUCTURAL_FUNCTIONS = [
    pytest.param(
        lambda x: x.reshape(2, 2),
        interval(jnp.zeros(4), jnp.ones(4)),
        id="reshape-4-to-2x2",
    ),
    pytest.param(
        lambda x: jnp.concatenate([x, x]),
        interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5])),
        id="concatenate-self",
    ),
    pytest.param(
        lambda x: jnp.concatenate([x[:1], jnp.zeros(1), x[1:]]),
        interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5])),
        id="concatenate-with-zeros",
    ),
    pytest.param(
        lambda x: x[1:],
        interval(jnp.array([-1.0, 0.5, 0.3]), jnp.array([0.5, 1.5, 1.0])),
        id="slice",
    ),
]


@pytest.mark.parametrize("fn,ix", _STRUCTURAL_FUNCTIONS)
def test_structural_bounds_valid(fn, ix):
    """Concrete bounds l <= u hold for structural ops."""
    lb = linbp(fn)(ix)
    assert jnp.all(lb.l <= lb.u)


@pytest.mark.parametrize("fn,ix", _STRUCTURAL_FUNCTIONS)
def test_structural_contains_output(fn, ix):
    """Structural op bounds contain f(x) for all x in ix."""
    from immrax.inclusion.linbp import _concretize
    lb = linbp(fn)(ix)
    result = _concretize(lb, ix.lower, ix.upper)
    samples = _sample_in_interval(ix, N_SAMPLES, jax.random.PRNGKey(42))
    outputs = jax.vmap(fn)(samples)
    tol = 1e-5
    assert jnp.all(result.lower - tol <= outputs.min(axis=0))
    assert jnp.all(outputs.max(axis=0) <= result.upper + tol)


# ---------------------------------------------------------------------------
# Tests: integer_pow linear relaxation
# ---------------------------------------------------------------------------

_INTEGER_POW_CASES = [
    # (interval, exponent)
    pytest.param((-1.0, 1.0),  2, id="sq-sym"),            # straddles 0, even
    pytest.param((0.5, 1.5),   2, id="sq-pos"),             # fully positive, even
    pytest.param((-1.5, -0.5), 2, id="sq-neg"),             # fully negative, even
    pytest.param((-2.0, 1.0),  2, id="sq-asym"),            # asymmetric straddle
    pytest.param((-0.5, 0.5),  3, id="cube-sym"),           # straddles 0, odd
    pytest.param((0.2, 1.2),   3, id="cube-pos"),           # fully positive, odd
    pytest.param((-1.2, -0.2), 3, id="cube-neg"),           # fully negative, odd
    pytest.param((-1.0, 1.0),  4, id="pow4-sym"),           # even, higher order
    pytest.param((0.5, 1.5),   4, id="pow4-pos"),
    pytest.param((-0.5, 0.5),  5, id="pow5-sym"),           # odd, higher order
    pytest.param((0.0, 1e-6),  2, id="sq-degenerate"),
]


@pytest.mark.parametrize("lu,n", _INTEGER_POW_CASES)
def test_integer_pow_upper_bound_valid(lu, n):
    """x^n upper affine bound >= x^n for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(lambda x: x ** n)(ix)
    alpha_u, beta_u = float(lb.uA[0, 0]), float(lb.ub[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max(xs ** n - (alpha_u * xs + beta_u))
    assert float(violation) <= 1e-5, (
        f"x^{n} upper bound violated on [{l_val}, {u_val}]: "
        f"alpha_u={alpha_u:.6f}, beta_u={beta_u:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu,n", _INTEGER_POW_CASES)
def test_integer_pow_lower_bound_valid(lu, n):
    """x^n lower affine bound <= x^n for all x in [l, u]."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(lambda x: x ** n)(ix)
    alpha_l, beta_l = float(lb.lA[0, 0]), float(lb.lb[0])
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    violation = jnp.max((alpha_l * xs + beta_l) - xs ** n)
    assert float(violation) <= 1e-5, (
        f"x^{n} lower bound violated on [{l_val}, {u_val}]: "
        f"alpha_l={alpha_l:.6f}, beta_l={beta_l:.6f}, max violation={float(violation):.2e}"
    )


@pytest.mark.parametrize("lu,n", _INTEGER_POW_CASES)
def test_integer_pow_concrete_bounds_valid(lu, n):
    """Concrete l, u from integer_pow handler contain x^n([l, u])."""
    l_val, u_val = lu
    ix = irx.interval(jnp.array([l_val]), jnp.array([u_val]))
    lb = linbp(lambda x: x ** n)(ix)
    xs = jnp.linspace(l_val, u_val, N_DENSE)
    vals = xs ** n
    tol = 1e-5
    assert float(lb.l[0]) <= float(jnp.min(vals)) + tol, (
        f"x^{n} concrete lower {lb.l[0]:.6f} > min={float(jnp.min(vals)):.6f} on [{l_val}, {u_val}]"
    )
    assert float(lb.u[0]) >= float(jnp.max(vals)) - tol, (
        f"x^{n} concrete upper {lb.u[0]:.6f} < max={float(jnp.max(vals)):.6f} on [{l_val}, {u_val}]"
    )


def test_integer_pow_zero():
    """x^0 = 1 everywhere: zero A-matrices, constant 1 bounds."""
    ix = irx.interval(jnp.array([-2.0, -1.0]), jnp.array([1.0, 3.0]))
    lb = linbp(lambda x: x ** 0)(ix)
    assert jnp.allclose(lb.lA, jnp.zeros_like(lb.lA))
    assert jnp.allclose(lb.uA, jnp.zeros_like(lb.uA))
    assert jnp.allclose(lb.lb, jnp.ones(2))
    assert jnp.allclose(lb.ub, jnp.ones(2))


def test_integer_pow_one():
    """x^1 = x: identity pass-through."""
    ix = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([0.5, 1.5]))
    lb = linbp(lambda x: x ** 1)(ix)
    assert jnp.allclose(lb.l, ix.lower)
    assert jnp.allclose(lb.u, ix.upper)


# ---------------------------------------------------------------------------
# reduce_sum
# ---------------------------------------------------------------------------


def test_reduce_sum_linear_amatrices():
    """reduce_sum is linear: A-matrices should sum exactly (no relaxation)."""
    ix = irx.interval(jnp.array([-1.0, 0.0, 2.0]), jnp.array([1.0, 1.0, 3.0]))
    lb = linbp(lambda x: jnp.sum(x))(ix)
    # sum is linear: lA == uA == ones
    assert jnp.allclose(lb.lA, jnp.ones((1, 3)))
    assert jnp.allclose(lb.uA, jnp.ones((1, 3)))


def test_reduce_sum_concrete_bounds():
    """reduce_sum concrete bounds: l=sum(l_i), u=sum(u_i)."""
    ix = irx.interval(jnp.array([-1.0, 0.0, 2.0]), jnp.array([1.0, 1.0, 3.0]))
    lb = linbp(lambda x: jnp.sum(x))(ix)
    assert jnp.allclose(lb.l, jnp.array([1.0]))   # -1+0+2
    assert jnp.allclose(lb.u, jnp.array([5.0]))   # 1+1+3


def test_reduce_sum_partial_axis():
    """reduce_sum over one axis of a 2D input."""
    ix = irx.interval(
        jnp.array([[-1.0, 0.0], [1.0, 2.0]]),
        jnp.array([[0.0, 1.0], [2.0, 3.0]]),
    )
    lb = linbp(lambda x: jnp.sum(x, axis=0))(ix)
    assert jnp.allclose(lb.l, jnp.array([0.0, 2.0]))
    assert jnp.allclose(lb.u, jnp.array([2.0, 4.0]))


# ---------------------------------------------------------------------------
# dot_general Branch 1 (x=LinearBound, y=constant) and Branch 3 (both LB)
# ---------------------------------------------------------------------------

def test_dot_general_branch1_shape():
    """Branch 1: uA/lA must have shape (*S_out, n_in), not (n_in, *S_out)."""
    n_in, n_out = 3, 5
    W = jax.random.normal(jax.random.PRNGKey(0), (n_in, n_out))
    ix = irx.icentpert(jnp.ones(n_in), 0.3)

    lb = linbp(lambda x: x @ W)(ix)

    assert lb.uA.shape == (n_out, n_in), f"uA shape {lb.uA.shape} != ({n_out}, {n_in})"
    assert lb.lA.shape == (n_out, n_in)


def test_dot_general_branch1_soundness():
    """Branch 1: concrete bounds must contain all true output values."""
    key = jax.random.PRNGKey(1)
    n_in, n_out = 3, 5
    W = jax.random.normal(key, (n_in, n_out))
    ix = irx.icentpert(jnp.zeros(n_in), 0.5)

    lb = linbp(lambda x: x @ W)(ix)

    sample_key = jax.random.PRNGKey(42)
    xs = _sample_in_interval(ix, N_SAMPLES, sample_key)
    ys = jax.vmap(lambda x: x @ W)(xs)
    assert jnp.all(ys >= lb.l - 1e-5)
    assert jnp.all(ys <= lb.u + 1e-5)


def test_dot_general_branch3_zero_amatrices():
    """Branch 3 fallback returns zero A-matrices (non-affine output)."""
    n_in = 4
    ix = irx.icentpert(jnp.array([1.0, -0.5, 2.0, 0.0]), 0.5)

    lb = linbp(lambda x: jnp.dot(x, x))(ix)

    assert lb.uA.shape == (n_in,)
    assert jnp.allclose(lb.uA, jnp.zeros(n_in))
    assert jnp.allclose(lb.lA, jnp.zeros(n_in))


def test_dot_general_branch3_soundness():
    """Branch 3 fallback: concrete bounds must contain all true output values."""
    n_in = 4
    ix = irx.icentpert(jnp.array([1.0, -0.5, 2.0, 0.0]), 0.5)

    lb = linbp(lambda x: jnp.dot(x, x))(ix)

    sample_key = jax.random.PRNGKey(7)
    xs = _sample_in_interval(ix, N_SAMPLES, sample_key)
    ys = jax.vmap(lambda x: jnp.dot(x, x))(xs)
    assert jnp.all(ys >= lb.l - 1e-5)
    assert jnp.all(ys <= lb.u + 1e-5)
