import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.inclusion import (
    KaucherInterval,
    kaucher_dual,
    kaucher_interval,
    kaucher_pro,
)


def _proper(key, shape):
    a, b = jax.random.normal(jax.random.PRNGKey(key), (2, *shape))
    return jnp.minimum(a, b), jnp.maximum(a, b)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "matmul"])
def test_proper_matches_interval_arithmetic(op):
    """On proper intervals Kaucher arithmetic is ordinary interval arithmetic."""
    (al, au), (bl, bu) = _proper(0, (3, 3)), _proper(1, (3, 3))
    f = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "matmul": lambda x, y: x @ y,
    }[op]
    k = f(KaucherInterval(al, au), KaucherInterval(bl, bu))
    i = irx.natif(f)(irx.interval(al, au), irx.interval(bl, bu))
    assert jnp.allclose(k.lower, i.lower, atol=1e-6) and jnp.allclose(
        k.upper, i.upper, atol=1e-6
    )


def test_dual_and_pro():
    lo, hi = _proper(2, (4,))
    x = kaucher_interval(lo, hi)
    assert jnp.allclose((x + (-x).dual).lower, 0.0) and jnp.allclose(
        (x + (-x).dual).upper, 0.0
    )
    d = kaucher_dual(x)
    assert jnp.all(d.lower == hi) and jnp.all(d.upper == lo)
    p = kaucher_pro(d)
    assert jnp.all(p.lower == lo) and jnp.all(p.upper == hi)


def test_mixed_operands_and_jit():
    lo, hi = _proper(3, (2, 2))
    x = KaucherInterval(lo, hi)
    A = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    y = jax.jit(lambda x: (A @ x + 1.0) * 2.0)(x)
    ref = irx.natif(lambda x: (A @ x + 1.0) * 2.0)(irx.interval(lo, hi))
    assert jnp.allclose(y.lower, ref.lower, atol=1e-6) and jnp.allclose(
        y.upper, ref.upper, atol=1e-6
    )
    assert x.sum(axis=0).shape == (2,) and x.T.shape == (2, 2)
