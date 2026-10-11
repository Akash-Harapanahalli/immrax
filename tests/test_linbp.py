"""Soundness of linear bound propagation: forward ``linbp``, backward CROWN, and the
network verifiers built on them."""

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import pytest
from jax._src import ad_util

import immrax as irx
from immrax import crown, fastlin, interval
from immrax.inclusion.linbp import (
    LinearBound,
    _concretize,
    _resolve_linbp_input,
    linbp,
    linbp_backward,
)

TOL = 1e-5
W3 = jax.random.normal(jax.random.PRNGKey(3), (3, 3))


def _samples(ix, n=500, key=0):
    u = jax.random.uniform(jax.random.PRNGKey(key), (n, *ix.lower.shape))
    return jnp.concatenate(
        [ix.lower[None], ix.upper[None], ix.lower + u * (ix.upper - ix.lower)]
    )


def _box(lo, hi):
    return interval(jnp.asarray(lo, float), jnp.asarray(hi, float))


def _check(lb: LinearBound, f, ix):
    """Linear and concrete bounds hold at samples of ``ix`` (corners included)."""
    xs = _samples(ix)
    ys = jax.vmap(f)(xs).reshape(xs.shape[0], -1)
    nin = ix.lower.ndim
    lin = lambda A, b: jax.vmap(lambda x: (jnp.tensordot(A, x, nin) + b).reshape(-1))(
        xs
    )
    assert jnp.all(lin(lb.lA, lb.lb) <= ys + TOL), "lower linear bound violated"
    assert jnp.all(ys <= lin(lb.uA, lb.ub) + TOL), "upper linear bound violated"
    assert jnp.all(lb.l.reshape(-1) <= ys.min(0) + TOL), "concrete lower bound violated"
    assert jnp.all(ys.max(0) <= lb.u.reshape(-1) + TOL), "concrete upper bound violated"


def _add_any(x):
    return ad_util.add_any_p.bind(jax.nn.relu(x), jax.nn.relu(-x))


# (id, f, input box). Elementwise cases put each coordinate in a different regime;
# "mixed" cases go through W3 first so the linear-bound matrices are dense.
CASES = [
    ("relu", jax.nn.relu, _box([-1, 0.2, -2], [1, 1, -0.5])),
    (
        "max-min-clip",
        lambda x: jnp.clip(jnp.maximum(x, -1.0) + jnp.minimum(x, 0.3), -0.2, 0.8),
        _box([-2, -0.5, 0], [3, 1, 2]),
    ),
    ("sigmoid", jax.nn.sigmoid, _box([-3, 0.5, -4], [3, 4, -0.5])),
    ("tanh", jnp.tanh, _box([-3, 0.5, -4], [3, 4, -0.5])),
    ("tanh-mixed", lambda x: jnp.tanh(W3 @ x), _box([-1, -1, -1], [1, 1, 1])),
    ("sin", jnp.sin, _box([-0.5, 1.0, 0.0], [0.5, 2.5, 7.0])),
    ("sin-degenerate", jnp.sin, _box([1.0, -2.0, 0.3], [1.0 + 1e-6, 2.0, 0.3 + 1e-6])),
    ("cos", jnp.cos, _box([-0.5, 2.5, 0.0], [0.5, 4.0, 7.0])),
    ("cos-mixed", lambda x: jnp.cos(W3 @ x), _box([-1, -1, -1], [1, 1, 1])),
    ("tan", jnp.tan, _box([-1.0, 0.2, -1.2], [1.0, 1.2, -0.1])),
    ("tan-mixed", lambda x: jnp.tan(0.3 * W3 @ x), _box([-1, -1, -1], [1, 1, 1])),
    ("exp", jnp.exp, _box([-2, 0, 1], [1, 3, 1 + 1e-7])),
    ("log1p", jnp.log1p, _box([-0.5, 0, 3], [2, 10, 3 + 1e-7])),
    ("sqrt", jnp.sqrt, _box([0, 1, 0.5], [4, 9, 0.5 + 1e-7])),
    ("square", jnp.square, _box([-1, 0.5, -2], [2, 1.5, -1])),
    ("pow3", lambda x: x**3, _box([-1, 0.5, -2], [2, 1.5, -1])),
    ("pow4-mixed", lambda x: (W3 @ x) ** 4, _box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])),
    ("pow-1", lambda x: x**-1, _box([0.5, 1, 2], [1, 3, 4])),
    (
        "arith",
        lambda x: 3.0 * (1.0 - x) / 2.0 - (-2.0 * x),
        _box([-1, 0, 0.5], [0.5, 1, 1.5]),
    ),
    ("div", lambda x: x[0] / (x[1] + 2.0), _box([-1, 0.5, 0], [1, 1.5, 1])),
    (
        "mul",
        lambda x: jnp.stack([x[0] * x[1], x[1] * x[2], x[2] * x[2]]),
        _box([-1, 0.5, -0.3], [0.5, 1.5, 0.8]),
    ),
    ("dot", lambda x: jnp.dot(W3 @ x, x), _box([0.5, -0.5, 0.3], [1.0, 0.0, 0.8])),
    (
        "quadratic-forms",
        lambda x: jax.vmap(lambda W: (W @ x) @ x)(jnp.stack([W3, W3.T, -W3])),
        _box([0.2, -0.5, 0.4], [0.8, 0.1, 1.0]),
    ),
    (
        "structural",
        lambda x: jnp.concatenate([x[1:], jnp.zeros(1), x[:1]]).reshape(2, 2),
        _box([-1, 0.5, 0.3], [0.5, 1.5, 1.0]),
    ),
    (
        "split",
        lambda x: jnp.concatenate([jax.nn.relu(s) for s in jnp.split(x, [1])]) @ W3,
        _box([-1, -1, -1], [1, 1, 1]),
    ),
    (
        "reduce-sum",
        lambda x: jnp.sum(jnp.tanh(x) * x),
        _box([-1, 0.5, -2], [1, 1.5, 0]),
    ),
    ("add-any", lambda x: W3 @ _add_any(x), _box([-0.5, -0.3, 0.4], [0.5, 0.5, 1.2])),
    (
        "jacfwd",
        lambda x: jax.jacfwd(lambda y: -y[0] * jnp.sin(y[1]))(x),
        _box([0.7, 0.2, 0], [1.3, 0.8, 1]),
    ),
]

METHODS = {
    "forward": lambda f: linbp(f),
    "backward": lambda f: linbp_backward(f),
    "backward-ibp": lambda f: linbp_backward(f, forward_mode="ibp"),
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,f,ix", CASES, ids=[c[0] for c in CASES])
def test_sound(method, name, f, ix):
    _check(METHODS[method](f)(ix), f, ix)


def test_add_any_reaches_handler():
    prims = {
        e.primitive.name
        for e in eqx.filter_make_jaxpr(_add_any)(jnp.zeros(3))[0].jaxpr.eqns
    }
    assert "add_any" in prims


def test_exact_on_linear():
    W = jnp.array([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]])
    f = lambda x: W @ jnp.concatenate(jnp.split(x, [2])) + 1.0
    for method in METHODS.values():
        lb = method(f)(_box([-1, 0, 1], [1, 2, 3]))
        assert jnp.allclose(lb.lA, W, atol=1e-6) and jnp.allclose(lb.uA, W, atol=1e-6)
        assert jnp.allclose(lb.lb, 1.0, atol=1e-6) and jnp.allclose(
            lb.ub, 1.0, atol=1e-6
        )


# ------------------------------------------------------------------ networks


def _net(arch, act, key):
    layers, key = [], jax.random.PRNGKey(key)
    for i in range(len(arch) - 1):
        key, k = jax.random.split(key)
        layers.append(nn.Linear(arch[i], arch[i + 1], key=k))
        if i < len(arch) - 2:
            layers.append(
                nn.Lambda(
                    {"relu": jax.nn.relu, "sigmoid": jax.nn.sigmoid, "tanh": jnp.tanh}[
                        act
                    ]
                )
            )
    return nn.Sequential(layers)


NETS = [
    ([2, 4, 1], "relu"),
    ([3, 8, 4, 2], "relu"),
    ([2, 6, 1], "sigmoid"),
    ([2, 6, 2], "tanh"),
]
VERIFIERS = {
    "crown-forward": lambda net: crown(net, backward=False),
    "crown": crown,
    "crown-iterated": lambda net: crown(net, iterated=True),
    "fastlin": fastlin,
}


@pytest.mark.parametrize("verifier", VERIFIERS)
@pytest.mark.parametrize(
    "arch,act", NETS, ids=[f"{'-'.join(map(str, a))}/{b}" for a, b in NETS]
)
def test_network_sound(verifier, arch, act):
    net = _net(arch, act, 0)
    ix = irx.icentpert(jnp.linspace(-0.5, 0.5, arch[0]), 0.4)
    out = VERIFIERS[verifier](net)(ix)(ix)
    ys = jax.vmap(net)(_samples(ix))
    assert jnp.all(out.lower <= ys.min(0) + TOL) and jnp.all(
        ys.max(0) <= out.upper + TOL
    )


def test_same_slope_shares_matrices():
    net = _net([3, 8, 4, 2], "relu", 1)
    ix = irx.icentpert(jnp.zeros(3), 0.5)
    for lb in (
        linbp(net, relu_mode="same-slope")(ix),
        linbp_backward(net, relu_mode="same-slope")(ix),
    ):
        assert jnp.allclose(lb.lA, lb.uA, atol=1e-6)


def _width(i):
    return float(jnp.sum(i.upper - i.lower))


def test_backward_tighter_than_forward():
    fwd = bwd = 0.0
    for k in range(3):
        net = _net([3, 16, 16, 16, 16, 1], "relu", k)
        ix = irx.icentpert(jnp.zeros(3), 1.0)
        fwd += _width(crown(net, backward=False)(ix)(ix))
        bwd += _width(crown(net)(ix)(ix))
    assert 1.5 * bwd <= fwd


def test_chained_linearbound_input():
    net1, net2 = _net([2, 6, 4], "tanh", 0), _net([4, 4, 2], "tanh", 1)
    ix = irx.icentpert(jnp.array([0.5, 1.0]), 0.2)
    lb = linbp(net2)(linbp(net1)(ix))
    out = _concretize(lb, ix.lower, ix.upper)
    ys = jax.vmap(lambda x: net2(net1(x)))(_samples(ix))
    assert jnp.all(out.lower <= ys.min(0) + TOL) and jnp.all(
        ys.max(0) <= out.upper + TOL
    )


def test_rejects_unsupported_input():
    with pytest.raises(TypeError, match="Interval or LinearBound"):
        _resolve_linbp_input(jnp.array([1.0, 2.0]))
