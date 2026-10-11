from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from .interval import Interval


@register_pytree_node_class
class KaucherInterval:
    """Kaucher (generalized) interval, allowing proper [a,b] (a <= b) and improper (b <= a) intervals."""

    lower: jax.Array
    upper: jax.Array

    def __init__(self, lower: jax.Array, upper: jax.Array) -> None:
        self.lower = lower
        self.upper = upper

    def tree_flatten(self):
        return ((self.lower, self.upper), "KaucherInterval")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def dtype(self) -> jnp.dtype:
        return self.lower.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lower.shape

    @property
    def ndim(self) -> int:
        return self.lower.ndim

    @property
    def dual(self) -> "KaucherInterval":
        return KaucherInterval(self.upper, self.lower)

    @property
    def pro(self) -> Interval:
        return Interval(
            jnp.minimum(self.lower, self.upper), jnp.maximum(self.lower, self.upper)
        )

    def __neg__(self) -> "KaucherInterval":
        return KaucherInterval(-self.upper, -self.lower)

    def __add__(self, other) -> "KaucherInterval":
        o = _promote(other)
        return KaucherInterval(self.lower + o.lower, self.upper + o.upper)

    __radd__ = __add__

    def __sub__(self, other) -> "KaucherInterval":
        o = _promote(other)
        return KaucherInterval(self.lower - o.upper, self.upper - o.lower)

    def __rsub__(self, other) -> "KaucherInterval":
        return _promote(other) - self

    def __mul__(self, other) -> "KaucherInterval":
        return _kaucher_mul(self, _promote(other))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "KaucherInterval":
        return _kaucher_mul(self, _kaucher_reciprocal(_promote(other)))

    def __rtruediv__(self, other) -> "KaucherInterval":
        return _kaucher_mul(_promote(other), _kaucher_reciprocal(self))

    def __matmul__(self, other) -> "KaucherInterval":
        return _kaucher_matmul(self, _promote(other))

    def __rmatmul__(self, other) -> "KaucherInterval":
        return _kaucher_matmul(_promote(other), self)

    def __getitem__(self, i) -> "KaucherInterval":
        return KaucherInterval(self.lower[i], self.upper[i])

    def __len__(self) -> int:
        return len(self.lower)

    def sum(self, axis=None, keepdims=False) -> "KaucherInterval":
        return KaucherInterval(
            jnp.sum(self.lower, axis=axis, keepdims=keepdims),
            jnp.sum(self.upper, axis=axis, keepdims=keepdims),
        )

    def reshape(self, *args, **kwargs) -> "KaucherInterval":
        return KaucherInterval(
            self.lower.reshape(*args, **kwargs), self.upper.reshape(*args, **kwargs)
        )

    def transpose(self, *args) -> "KaucherInterval":
        return KaucherInterval(self.lower.transpose(*args), self.upper.transpose(*args))

    @property
    def T(self) -> "KaucherInterval":
        return self.transpose()

    def __str__(self) -> str:
        s = Interval._format_bounds(self)
        if s is None:
            return f"KaucherInterval(lower={self.lower}, upper={self.upper})"
        return s

    def __repr__(self) -> str:
        s = Interval._format_bounds(self)
        if s is None:
            return f"KaucherInterval(lower={self.lower}, upper={self.upper})"
        return f"KaucherInterval({s})"


def _promote(x: Union["KaucherInterval", Interval, ArrayLike]) -> KaucherInterval:
    if isinstance(x, KaucherInterval):
        return x
    if isinstance(x, Interval):
        return KaucherInterval(x.lower, x.upper)
    x = jnp.asarray(x)
    return KaucherInterval(x, x)


def _kaucher_mul(x: KaucherInterval, y: KaucherInterval) -> KaucherInterval:
    # Closed form of Kaucher's 16-case multiplication table, t+ = max(t,0), t- = max(-t,0)
    a, b, c, d = x.lower, x.upper, y.lower, y.upper
    ap, an = jnp.maximum(a, 0), jnp.maximum(-a, 0)
    bp, bn = jnp.maximum(b, 0), jnp.maximum(-b, 0)
    cp, cn = jnp.maximum(c, 0), jnp.maximum(-c, 0)
    dp, dn = jnp.maximum(d, 0), jnp.maximum(-d, 0)
    lower = jnp.maximum(ap * cp, bn * dn) - jnp.maximum(bp * cn, an * dp)
    upper = jnp.maximum(bp * dp, an * cn) - jnp.maximum(ap * dn, bn * cp)
    return KaucherInterval(lower, upper)


def _kaucher_reciprocal(x: KaucherInterval) -> KaucherInterval:
    # Defined only when 0 is not in pro(x); NaN otherwise
    valid = x.lower * x.upper > 0
    return KaucherInterval(
        jnp.where(valid, 1 / x.upper, jnp.nan), jnp.where(valid, 1 / x.lower, jnp.nan)
    )


def _kaucher_matmul(x: KaucherInterval, y: KaucherInterval) -> KaucherInterval:
    if x.ndim == 1 and y.ndim == 1:
        return _kaucher_mul(x, y).sum()
    if y.ndim == 1:
        return _kaucher_mul(x, y).sum(-1)
    if x.ndim == 1:
        return _kaucher_mul(x[:, None], y).sum(-2)
    xa = KaucherInterval(x.lower[..., :, :, None], x.upper[..., :, :, None])
    ya = KaucherInterval(y.lower[..., None, :, :], y.upper[..., None, :, :])
    return _kaucher_mul(xa, ya).sum(-2)


def kaucher_interval(
    lower: Union[KaucherInterval, Interval, ArrayLike],
    upper: Optional[ArrayLike] = None,
) -> KaucherInterval:
    """Create a KaucherInterval from bounds, an Interval, or a KaucherInterval."""
    if upper is None:
        return _promote(lower)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    if lower.shape != upper.shape:
        raise Exception(
            f"lower and upper shape should match, {lower.shape} != {upper.shape}"
        )
    return KaucherInterval(lower, upper)


def kaucher_dual(x: Union[KaucherInterval, Interval, ArrayLike]) -> KaucherInterval:
    """dual([a,b]) = [b,a]."""
    return _promote(x).dual


def kaucher_pro(x: Union[KaucherInterval, Interval, ArrayLike]) -> Interval:
    """pro([a,b]) = [min(a,b), max(a,b)] as a standard Interval."""
    return _promote(x).pro


def iskaucher(x) -> bool:
    """Check if x is a KaucherInterval."""
    return isinstance(x, KaucherInterval)


# Defer Interval op KaucherInterval to Kaucher arithmetic (Interval's forward op runs first)
def _defer_binop(name):
    orig = getattr(Interval, name)

    def op(self, other):
        if isinstance(other, KaucherInterval):
            return getattr(_promote(self), name)(other)
        return orig(self, other)

    return op


for _name in ("__add__", "__sub__", "__mul__", "__truediv__", "__matmul__"):
    setattr(Interval, _name, _defer_binop(_name))
del _name
