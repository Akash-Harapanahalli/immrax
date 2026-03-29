from typing import List
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from typing import Tuple, Iterable, Union, Optional
from jaxtyping import ArrayLike
import numpy as onp


# ---------------------------------------------------------------------------
# Rigorous-mode global state and helpers
# ---------------------------------------------------------------------------

_rigorous: bool = False  # global default
_rigorous_forced: Optional[bool] = None  # set by context managers


def _resolve_rigorous(kw_rigorous=None):
    """Resolve rigorous mode: kwarg > context manager > global."""
    if kw_rigorous is not None:
        return kw_rigorous
    if _rigorous_forced is not None:
        return _rigorous_forced
    return _rigorous


def _get_rigorous():
    """Return the current global rigorous flag."""
    return _rigorous


def _set_rigorous(val):
    """Set the global rigorous flag, returning the old value."""
    global _rigorous
    old = _rigorous
    _rigorous = val
    return old


def set_rigorous(val: bool) -> None:
    """Set the global default for rigorous FP widening.

    Parameters
    ----------
    val : bool
        ``True`` to enable widening (the default), ``False`` to disable.
    """
    global _rigorous
    _rigorous = val


class rigorous:
    """Context manager to force rigorous mode on."""

    def __enter__(self):
        global _rigorous_forced
        self._old = _rigorous_forced
        _rigorous_forced = True
        return self

    def __exit__(self, *a):
        global _rigorous_forced
        _rigorous_forced = self._old


class non_rigorous:
    """Context manager to force rigorous mode off."""

    def __enter__(self):
        global _rigorous_forced
        self._old = _rigorous_forced
        _rigorous_forced = False
        return self

    def __exit__(self, *a):
        global _rigorous_forced
        _rigorous_forced = self._old


@register_pytree_node_class
class Interval:
    """Interval: A class to represent an interval in :math:`\\mathbb{R}^n`.

    Use the helper functions :func:`interval`, :func:`icentpert`, :func:`i2centpert`, :func:`i2lu`, :func:`i2ut`, and :func:`ut2i` to create and manipulate intervals.

    Use the transforms :func:`natif`, :func:`jacif`, :func:`mjacif`, :func:`mjacM`, to create inclusion functions.

    Composable with typical jax transforms, such as :func:`jax.jit`, :func:`jax.grad`, and :func:`jax.vmap`.
    """

    lower: jax.Array
    upper: jax.Array

    def __init__(self, lower: jax.Array, upper: jax.Array) -> None:
        self.lower = lower
        self.upper = upper

    def tree_flatten(self):
        return ((self.lower, self.upper), "Interval")

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
    def size(self) -> int:
        return self.lower.size

    @property
    def width(self) -> jax.Array:
        return self.upper - self.lower

    @property
    def center(self) -> jax.Array:
        return (self.lower + self.upper) / 2

    @property
    def pert(self) -> jax.Array:
        return (self.upper - self.lower) / 2

    def __add__(self, other: Union["Interval", ArrayLike]) -> "Interval": ...
    def __radd__(self, other: ArrayLike) -> "Interval": ...
    def __sub__(self, other: Union["Interval", ArrayLike]) -> "Interval": ...
    def __rsub__(self, other: ArrayLike) -> "Interval": ...
    def __neg__(self) -> "Interval": ...
    def __mul__(self, other: Union["Interval", ArrayLike]) -> "Interval": ...
    def __rmul__(self, other: ArrayLike) -> "Interval": ...
    def __truediv__(self, other: Union["Interval", ArrayLike]) -> "Interval": ...
    def __rtruediv__(self, other: ArrayLike) -> "Interval": ...
    def __pow__(self, other: Union[int, "Interval"]) -> "Interval": ...
    def __matmul__(self, other: Union["Interval", ArrayLike]) -> "Interval": ...
    def __rmatmul__(self, other: ArrayLike) -> "Interval": ...

    def __len__(self) -> int:
        return len(self.lower)

    def reshape(self, *args, **kwargs):
        return Interval(
            self.lower.reshape(*args, **kwargs), self.upper.reshape(*args, **kwargs)
        )

    def ravel(self) -> List["Interval"]:
        return [Interval(l, u) for l, u in zip(self.lower.ravel(), self.upper.ravel())]

    def atleast_1d(self) -> "Interval":
        return Interval(jnp.atleast_1d(self.lower), jnp.atleast_1d(self.upper))

    def atleast_2d(self) -> "Interval":
        return Interval(jnp.atleast_2d(self.lower), jnp.atleast_2d(self.upper))

    def atleast_3d(self) -> "Interval":
        return Interval(jnp.atleast_3d(self.lower), jnp.atleast_3d(self.upper))

    @property
    def ndim(self) -> int:
        return self.lower.ndim

    def transpose(self, *args) -> "Interval":
        return Interval(self.lower.transpose(*args), self.upper.transpose(*args))

    def broadcast_to(self, shape) -> "Interval":
        return Interval(
            jnp.broadcast_to(self.lower, shape), jnp.broadcast_to(self.upper, shape)
        )

    def squeeze(self, axis=None) -> "Interval":
        return Interval(
            jnp.squeeze(self.lower, axis=axis), jnp.squeeze(self.upper, axis=axis)
        )

    def sum(self, axis=None, keepdims=False) -> "Interval":
        return Interval(
            jnp.sum(self.lower, axis=axis, keepdims=keepdims),
            jnp.sum(self.upper, axis=axis, keepdims=keepdims),
        )

    def scale(self, factor: Union[float, ArrayLike]) -> "Interval":
        return icentpert(self.center, self.pert * factor)

    @property
    def T(self) -> "Interval":
        return self.transpose()

    def __and__(self, other: "Interval") -> "Interval":
        return Interval(
            jnp.maximum(self.lower, other.lower), jnp.minimum(self.upper, other.upper)
        )

    def __or__(self, other: "Interval") -> "Interval":
        return Interval(
            jnp.minimum(self.lower, other.lower), jnp.maximum(self.upper, other.upper)
        )

    def _format_bounds(self) -> str:
        """Format interval bounds numpy-style, replacing each scalar with ⟦lo, hi⟧."""
        try:
            lo = onp.asarray(self.lower)
            hi = onp.asarray(self.upper)
        except Exception:
            return None

        lc, rc = "⟦", "⟧"
        SEP = "\x00"  # null byte won't appear in numeric strings

        if lo.ndim == 0:
            combined = onp.array([lo.item(), hi.item()])
            s = onp.array2string(combined, max_line_width=10**9, separator=SEP)
            lo_s, hi_s = (p.strip() for p in s.strip("[]").split(SEP))
            return f"{lc}{lo_s}, {hi_s}{rc}"

        # Format all lo and hi values together so both bounds share the same
        # numeric precision/notation that numpy would choose for this data.
        all_flat = onp.concatenate([lo.ravel(), hi.ravel()])
        all_s = onp.array2string(all_flat, max_line_width=10**9, separator=SEP)
        all_strs = [p.strip() for p in all_s.strip("[]").split(SEP)]

        n = lo.size
        interval_strs = [
            f"{lc}{l}, {h}{rc}" for l, h in zip(all_strs[:n], all_strs[n:])
        ]

        # Delegate all layout (brackets, indentation, line-wrapping, alignment)
        # to numpy via an object-dtype array.  JAX does not support object dtype.
        obj = onp.array(interval_strs, dtype=object).reshape(lo.shape)
        return onp.array2string(obj, formatter={"object": lambda s: s})

    def __str__(self) -> str:
        s = self._format_bounds()
        if s is None:
            return f"Interval(lower={self.lower}, upper={self.upper})"
        return s

    def __repr__(self) -> str:
        s = self._format_bounds()
        if s is None:
            return f"Interval(lower={self.lower}, upper={self.upper})"
        return f"Interval({s})"

    def __getitem__(self, i: Union[slice, ArrayLike]) -> "Interval":
        return Interval(self.lower[i], self.upper[i])

    def __iter__(self):
        """Return an iterator over the interval elements."""
        # Use the actual length to create a proper iterator
        # This avoids the infinite loop issue by using explicit indexing
        length = int(len(self))
        return (self[i] for i in range(length))


# HELPER FUNCTIONS


def interval(
    lower: ArrayLike, upper: Optional[ArrayLike] = None, rigorous: Optional[bool] = None
) -> Interval:
    """interval: Helper to create a Interval from a lower and upper bound.

    Parameters
    ----------
    lower : ArrayLike
        Lower bound of the interval.
    upper : ArrayLike
        Upper bound of the interval. Set to lower bound if None. Defaults to None.
    rigorous : bool, optional
        When True, widen the interval by 1 ULP in each direction to account
        for floating-point representation error.  Resolution order:
        context manager > kwarg > global default (True).

    Returns
    -------
    Interval
        [lower, upper], or [lower, lower] if upper is None.

    """
    if isinstance(lower, Interval) and upper is None:
        return lower
    if upper is None:
        v = jnp.asarray(lower)
        iv = Interval(v, v)
        if _resolve_rigorous(rigorous):
            iv = widen(iv, 1)
        return iv
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    if lower.dtype != upper.dtype:
        raise Exception(
            f"lower and upper dtype should match, {lower.dtype} != {upper.dtype}"
        )
    if lower.shape != upper.shape:
        raise Exception(
            f"lower and upper shape should match, {lower.shape} != {upper.shape}"
        )
    iv = Interval(lower, upper)
    if _resolve_rigorous(rigorous):
        iv = widen(iv, 1)
    return iv


def icopy(i: Interval) -> Interval:
    """icopy: Helper to copy an interval.

    Parameters
    ----------
    i : Interval
        interval to copy

    Returns
    -------
    Interval
        copy of the interval

    """
    return Interval(jnp.copy(i.lower), jnp.copy(i.upper))


def icentpert(
    cent: ArrayLike, pert: ArrayLike, rigorous: Optional[bool] = None
) -> Interval:
    """icentpert: Helper to create a Interval from a center of an interval and a perturbation.

    Parameters
    ----------
    cent : ArrayLike
        Center of the interval, i.e., (l + u)/2
    pert : ArrayLike
        l-inf perturbation from the center, i.e., (u - l)/2
    rigorous : bool, optional
        When True, widen the interval by 1 ULP in each direction.
        Resolution order: context manager > kwarg > global default (True).

    Returns
    -------
    Interval
        Interval [cent - pert, cent + pert]

    """
    cent = jnp.asarray(cent)
    pert = jnp.asarray(pert)
    iv = Interval(cent - pert, cent + pert)
    if _resolve_rigorous(rigorous):
        iv = widen(iv, 1)
    return iv


centpert2i = icentpert


def i2centpert(i: Interval) -> Tuple[jax.Array, jax.Array]:
    """i2centpert: Helper to get the center and perturbation from the center of a Interval.

    Parameters
    ----------
    i : Interval
        _description_

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        ((l + u)/2, (u - l)/2)

    """
    return (i.lower + i.upper) / 2, (i.upper - i.lower) / 2


def interval_intersect(Is: Iterable[Interval]) -> Interval:
    """interval_intersect: Helper to get the intersection of a list of intervals.

    Parameters
    ----------
    Is : Iterable[Interval]
        list of intervals

    Returns
    -------
    Interval
        intersection of the intervals

    """
    l = jnp.max(jnp.array([i.lower for i in Is]), axis=0)
    u = jnp.min(jnp.array([i.upper for i in Is]), axis=0)
    return Interval(l, u)


def interval_union(Is: Iterable[Interval]) -> Interval:
    """interval_union: Helper to get the union of a list of intervals.

    Parameters
    ----------
    Is : Iterable[Interval]
        list of intervals

    Returns
    -------
    Interval
        union of the intervals

    """
    l = jnp.min(jnp.array([i.lower for i in Is]), axis=0)
    u = jnp.max(jnp.array([i.upper for i in Is]), axis=0)
    return Interval(l, u)


def i2lu(i: Interval) -> Tuple[jax.Array, jax.Array]:
    """i2lu: Helper to get the lower and upper bound of a Interval.

    Parameters
    ----------
    interval : Interval
        _description_

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        (l, u)

    """
    return (i.lower, i.upper)


def lu2i(l: jax.Array, u: jax.Array) -> Interval:
    """lu2i: Helper to create a Interval from a lower and upper bound.

    Parameters
    ----------
    l : jax.Array
        Lower bound of the interval.
    u : jax.Array
        Upper bound of the interval.

    Returns
    -------
    Interval
        [l, u]

    """
    return Interval(l, u)


def i2ut(i: Interval) -> jax.Array:
    """i2ut: Helper to convert an interval to an upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}`.

    Parameters
    ----------
    interval : Interval
        interval to convert

    Returns
    -------
    jax.Array
        upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}`

    """
    return jnp.concatenate((i.lower, i.upper))


def ut2i(coordinate: jax.Array, n: Optional[int] = None) -> Interval:
    """ut2i: Helper to convert an upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}` to an interval.

    Parameters
    ----------
    coordinate : jax.Array
        upper triangular coordinate to convert
    n : int
        length of interval, automatically determined if None. Defaults to None.

    Returns
    -------
    Interval
        interval representation of the coordinate

    """
    if n is None:
        n = len(coordinate) // 2
    return Interval(coordinate[:n], coordinate[n:])


def izeros(shape: Tuple[int], dtype: onp.dtype = jnp.float32) -> Interval:
    """izeros: Helper to create a Interval of zeros.

    Parameters
    ----------
    shape : Tuple[int]
        shape of the interval
    dtype : np.dtype
        dtype of the interval. Defaults to jnp.float32.

    Returns
    -------
    Interval
        interval of zeros

    """
    return Interval(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))


def iconcatenate(intervals: Iterable[Interval], axis: int = 0) -> Interval:
    """iconcatenate: Helper to concatenate intervals (cartesian product).

    Parameters
    ----------
    intervals : Iterable[Interval]
        intervals to concatenate
    axis : int
        axis to concatenate on. Defaults to 0.

    Returns
    -------
    Interval
        concatenated interval

    """
    return Interval(
        jnp.concatenate([i.lower for i in intervals], axis=axis),
        jnp.concatenate([i.upper for i in intervals], axis=axis),
    )


def scale(i: Interval, factor: Union[float, ArrayLike]) -> Interval:
    """Scale an interval by a given factor around its center.

    Parameters
    ----------
    i : Interval
        The interval to scale.
    factor : float | ArrayLike
        Scaling factor.

    Returns
    -------
    Interval
        The scaled interval.
    """
    return icentpert(i.center, i.pert * factor)


def isinterval(x) -> bool:
    """Check if x is an Interval."""
    return isinstance(x, Interval)


# ---------------------------------------------------------------------------
# Rigorous widening primitives
# ---------------------------------------------------------------------------


@jax.custom_jvp
def _widen_lower(x):
    """Push *x* toward :math:`-\\infty` by one ULP."""
    return jnp.nextafter(x, jnp.full_like(x, -jnp.inf))


@_widen_lower.defjvp
def _widen_lower_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    return _widen_lower(x), t


@jax.custom_jvp
def _widen_upper(x):
    """Push *x* toward :math:`+\\infty` by one ULP."""
    return jnp.nextafter(x, jnp.full_like(x, jnp.inf))


@_widen_upper.defjvp
def _widen_upper_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    return _widen_upper(x), t


def widen(iv: Interval, n: int = 1) -> Interval:
    """Widen an interval by *n* ULPs in each direction.

    Pushes ``iv.lower`` toward :math:`-\\infty` and ``iv.upper`` toward
    :math:`+\\infty` by *n* units in the last place.  The custom JVP
    rules treat the widening as the identity so that automatic
    differentiation passes through unchanged.

    Parameters
    ----------
    iv : Interval
        Interval to widen.
    n : int
        Number of ULPs to widen by (default 1).

    Returns
    -------
    Interval
        Widened interval.
    """
    if n == 0:
        return iv
    lo, hi = iv.lower, iv.upper
    if n <= 4:
        for _ in range(n):
            lo = _widen_lower(lo)
            hi = _widen_upper(hi)
    else:
        # O(1) widening: compute 1-ULP step, scale by n, add 1 ULP margin
        lo_step = _widen_lower(lo) - lo  # negative
        hi_step = _widen_upper(hi) - hi  # positive
        lo = _widen_lower(lo + n * lo_step)
        hi = _widen_upper(hi + n * hi_step)
    return Interval(lo, hi)
