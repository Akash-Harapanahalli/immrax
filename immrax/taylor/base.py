"""Shared base utilities for the Taylor model and Taylor polynomial frameworks.

This module provides the common infrastructure used by both TaylorModel and
TaylorPolynomial: pytree helpers, the PyTreeShape descriptor, and per-leaf
exponent-generation functions.
"""

from typing import Tuple
from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, PyTreeDef

import jax.tree_util

from immrax.inclusion import Interval, iconcatenate


# ---------------------------------------------------------------------------
# MultiIndex and MultiIndexArray
# ---------------------------------------------------------------------------


class MultiIndex(tuple):
    """Multi-index α = (α₁, ..., αₙ) with αᵢ ∈ ℕ₀.

    Stores the count vector so that standard multi-index notation applies:
      |α| = Σαᵢ,  α! = Π(αᵢ!),  x^α = Πxᵢ^αᵢ, etc.
    """

    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __repr__(self):
        return f"MultiIndex{super().__repr__()}"

    # --- Component-wise arithmetic ---

    def __add__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return MultiIndex(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return MultiIndex(*(a - b for a, b in zip(self, other)))

    # --- Component-wise ordering (overrides tuple's lexicographic order) ---

    def __le__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a <= b for a, b in zip(self, other))

    def __lt__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a < b for a, b in zip(self, other))

    def __ge__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a >= b for a, b in zip(self, other))

    def __gt__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a > b for a, b in zip(self, other))

    # --- Multi-index operations ---

    def __abs__(self):
        """Order of the multi-index: |α| = Σαᵢ."""
        return sum(self)

    def factorial(self):
        """α! = α₁! · α₂! · ... · αₙ!"""
        result = 1
        for a in self:
            result *= math.factorial(a)
        return result

    def inv_factorial(self):
        """1/α! = exp(-Σᵢ lgamma(αᵢ + 1)), numerically stable via log-gamma."""
        return math.exp(-sum(math.lgamma(a + 1) for a in self))

    def binomial(self, other):
        """Binomial coefficient C(α, β) = α! / (β! · (α−β)!), requires β ≤ α."""
        if not (other <= self):
            raise ValueError(f"β={other} must be ≤ α={self} component-wise")
        diff = self - other
        return self.factorial() // (other.factorial() * diff.factorial())

    def multinomial(self):
        """Multinomial coefficient: |α|! / α!"""
        return math.factorial(abs(self)) // self.factorial()

    def power(self, x):
        """x^α = x₁^α₁ · x₂^α₂ · ... · xₙ^αₙ, requires len(x) == len(α)."""
        if len(x) != len(self):
            raise ValueError(f"len(x)={len(x)} must equal len(α)={len(self)}")
        result = 1
        for xi, ai in zip(x, self):
            result = result * xi**ai
        return result


class MultiIndexArray(tuple):
    """Ordered collection of MultiIndex objects, one per monomial.

    Stores monomials as a tuple of MultiIndex objects, providing a hashable
    representation of the exponent structure.  Use :meth:`to_jnp` to obtain
    a ``(d, num_monomials)`` integer JAX array for numerical computation.

    Parameters
    ----------
    indices : iterable of MultiIndex, or array-like of shape (d, m)
        Either a sequence of MultiIndex objects (one per monomial) or a
        2-D integer array whose columns are the per-monomial exponent vectors.
    """

    def __new__(cls, indices):
        if hasattr(indices, "shape"):
            # numpy / jnp array of shape (d, m) — columns are exponent vectors
            d, m = indices.shape
            mis = tuple(
                MultiIndex(*[int(indices[i, j]) for i in range(d)]) for j in range(m)
            )
        else:
            mis = tuple(indices)
        return super().__new__(cls, mis)

    def __repr__(self):
        return f"MultiIndexArray({list(self)!r})"

    # --- Conversion ---

    def to_jnp(self) -> Array:
        """Return a jnp integer array of shape ``(d, num_monomials)``."""
        if len(self) == 0:
            return jnp.zeros((0, 0), dtype=jnp.int32)
        d = len(self[0])
        return jnp.array([[mi[i] for mi in self] for i in range(d)], dtype=jnp.int32)

    def to_numpy(self) -> "onp.ndarray":
        """Return a numpy integer array of shape ``(d, num_monomials)``."""
        if len(self) == 0:
            return onp.zeros((0, 0), dtype=onp.int32)
        d = len(self[0])
        return onp.array([[mi[i] for mi in self] for i in range(d)], dtype=onp.int32)

    # --- Convenience properties ---

    @property
    def d(self) -> int:
        """Number of variables (length of each MultiIndex)."""
        return len(self[0]) if self else 0

    @property
    def num_monomials(self) -> int:
        """Number of monomials stored."""
        return len(self)


# ---------------------------------------------------------------------------
# PyTree Helpers
# ---------------------------------------------------------------------------


def _is_interval_or_array_leaf(x):
    """Check if x is an Interval or Array (leaf node in domain pytree)."""
    return isinstance(x, Interval) or isinstance(x, Array)


def pack_pytree(pytree, is_leaf=_is_interval_or_array_leaf):
    """Extract pytree structure and leaf info from pytree.

    Essentially a double flattening: PyTree -> Leaves -> Flattened Array/Interval.

    Parameters
    ----------
    pytree : PyTree
        PyTree to flatten.

    Returns
    -------
    treedef : jax.tree_util.PyTreeDef
        PyTree structure definition for the domain.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Array/Interval in tree traversal order.
    flat_domain : Interval or Array
        Single flattened Interval/Array containing all domain variables concatenated.
    """
    treedef = jax.tree_util.tree_structure(pytree, is_leaf=is_leaf)
    leaves = jax.tree_util.tree_leaves(pytree, is_leaf=is_leaf)

    if len(leaves) == 0:
        raise ValueError("Pytree must contain at least one Interval or Array")

    leaf_shapes = tuple(iv.shape for iv in leaves)

    flat = []
    for iv in leaves:
        if iv.shape == ():
            flat.append(iv[None])
        else:
            flat.append(iv.reshape(-1))

    if isinstance(leaves[0], Interval):
        flat_domain = iconcatenate(flat)
    else:
        flat_domain = jnp.concatenate(flat)

    return treedef, leaf_shapes, flat_domain


def unpack_pytree(treedef, leaf_shapes, flat_array):
    """Unflatten a 1D Array/Interval into a pytree structure matching leaf_shapes and treedef.

    Parameters
    ----------
    treedef : jax.tree_util.PyTreeDef
        Target pytree structure.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf.
    flat_array : Array or Interval
        Flattened array/interval to unflatten.

    Returns
    -------
    PyTree[Array or Interval]
        Array data reshaped into the pytree structure.
    """
    if len(leaf_shapes) == 1:
        shape = leaf_shapes[0]
        if shape == ():
            return flat_array[0] if flat_array.ndim > 0 else flat_array
        return flat_array.reshape(shape)

    leaves = []
    start = 0
    for shape in leaf_shapes:
        size = math.prod(shape) if shape else 1
        if shape == ():
            leaves.append(flat_array[start])
        else:
            leaves.append(flat_array[start : start + size].reshape(shape))
        start += size

    return jax.tree_util.tree_unflatten(treedef, leaves)


class PyTreeShape:
    """Lightweight, hashable, immutable descriptor for pytree structure.

    Unifies the ``(treedef, leaf_shapes)`` pair used throughout the Taylor
    model codebase.  Used for both input (domain) and output structure.
    """

    __slots__ = ("treedef", "leaf_shapes")

    def __init__(self, treedef: PyTreeDef, leaf_shapes: "tuple[tuple[int, ...], ...]"):
        object.__setattr__(self, "treedef", treedef)
        object.__setattr__(self, "leaf_shapes", tuple(tuple(s) for s in leaf_shapes))

    def __setattr__(self, name, value):
        raise AttributeError("PyTreeShape is immutable")

    # --- Properties ---

    @property
    def flat_size(self) -> int:
        """Total number of scalar elements across all leaves."""
        return sum(math.prod(s) if s else 1 for s in self.leaf_shapes)

    @property
    def num_leaves(self) -> int:
        """Number of leaves in the pytree."""
        return len(self.leaf_shapes)

    # --- Methods ---

    def unflatten(self, flat_array):
        """Reshape a flat array into the pytree structure."""
        return unpack_pytree(self.treedef, self.leaf_shapes, flat_array)

    def leaf_slice(self, idx: int) -> slice:
        """Get contiguous slice for leaf ``idx`` in the flat representation."""
        return leaf_slice(self.leaf_shapes, idx)

    # --- Constructors ---

    @classmethod
    def flat(cls, shape: "tuple[int, ...]") -> "PyTreeShape":
        """Create a PyTreeShape for a single flat leaf with the given shape."""
        dummy = jnp.zeros(shape)
        treedef = jax.tree_util.tree_structure(dummy)
        return cls(treedef, (tuple(shape),))

    @classmethod
    def from_pytree(cls, pytree) -> "PyTreeShape":
        """Extract PyTreeShape from a pytree, discarding data."""
        treedef = jax.tree_util.tree_structure(
            pytree, is_leaf=_is_interval_or_array_leaf
        )
        leaves = jax.tree_util.tree_leaves(pytree, is_leaf=_is_interval_or_array_leaf)
        leaf_shapes = tuple(leaf.shape for leaf in leaves)
        return cls(treedef, leaf_shapes)

    # --- Hashability (required for JAX aux_data) ---

    def __eq__(self, other):
        if not isinstance(other, PyTreeShape):
            return NotImplemented
        return self.treedef == other.treedef and self.leaf_shapes == other.leaf_shapes

    def __hash__(self):
        return hash((self.treedef, self.leaf_shapes))

    def __repr__(self):
        return f"PyTreeShape(treedef={self.treedef}, leaf_shapes={self.leaf_shapes})"


def normalize_leaf_order(order, domain_treedef, leaf_shapes):
    """Normalize order specification to a per-leaf tuple.

    Parameters
    ----------
    order : int or PyTree[int]
        Order specification. If int, broadcast to all leaves.
        If pytree, must match domain_treedef structure.
    domain_treedef : jax.tree_util.PyTreeDef
        PyTree structure from the domain.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.

    Returns
    -------
    leaf_order : tuple[int, ...]
        Total degree bound for each leaf (one per leaf).
    """
    num_leaves = len(leaf_shapes)

    if isinstance(order, int):
        leaf_order = tuple([order] * num_leaves)
    else:
        order_pytree = jax.tree_util.tree_structure(order)
        order_leaves = jax.tree_util.tree_leaves(order)
        if order_pytree != domain_treedef or len(order_leaves) != num_leaves:
            raise ValueError(
                "Order pytree structure must match domain pytree structure, got "
                f"{order_pytree} and {domain_treedef}, and "
                f"{len(order_leaves)} and {num_leaves} leaves respectively"
            )
        leaf_order = tuple(int(o) for o in order_leaves)

    return leaf_order


def leaf_slice(leaf_shapes: "tuple[tuple[int, ...], ...]", leaf_idx: int) -> slice:
    """Get slice for domain variables of a given leaf.

    Parameters
    ----------
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    leaf_idx : int
        Index of the leaf to get slice for.

    Returns
    -------
    slice
        Slice object for indexing into flat domain variables.
    """
    sizes = [math.prod(s) if s else 1 for s in leaf_shapes]
    start = sum(sizes[:leaf_idx])
    return slice(start, start + sizes[leaf_idx])


def max_leaf_order(a: "tuple[int, ...]", b: "tuple[int, ...]") -> "tuple[int, ...]":
    """Element-wise maximum of two order tuples."""
    return tuple(max(ai, bi) for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# Exponent generation
# ---------------------------------------------------------------------------


def _enumerate_total_degree(dim: int, max_deg: int) -> "list[tuple[int, ...]]":
    """Generate all multi-indices in R^dim with total degree <= max_deg.

    Parameters
    ----------
    dim : int
        Number of variables.
    max_deg : int
        Maximum total degree (sum of exponents).

    Returns
    -------
    list[tuple[int, ...]]
        List of exponent tuples, each of length dim.
    """
    if dim == 0:
        return [()]
    if dim == 1:
        return [(k,) for k in range(max_deg + 1)]
    result = []
    for first in range(max_deg + 1):
        for rest in _enumerate_total_degree(dim - 1, max_deg - first):
            result.append((first,) + rest)
    return result


@lru_cache(maxsize=128)
def leaf_total_degree_exponents(
    leaf_shapes: "tuple[tuple[int, ...], ...]", leaf_order: "tuple[int, ...]"
) -> "MultiIndexArray":
    """Generate exponents with per-leaf total degree bounds.

    For multi-leaf domains (pytree of Intervals), this generates all monomials
    where the total degree of exponents for each leaf is bounded separately.

    Parameters
    ----------
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    leaf_order : tuple[int, ...]
        Maximum total degree for each leaf.

    Returns
    -------
    MultiIndexArray
        Tuple of MultiIndex objects, one per monomial.

    Example
    -------
    For domain = [interval_t, interval_x] where t: () and x: (2,),
    with leaf_order = (2, 3):
    - Generates monomials t^a * x1^b1 * x2^b2 where a <= 2 and b1+b2 <= 3
    - Count: (2+1) * C(2+3, 3) = 3 * 10 = 30 monomials
    """
    from itertools import product

    leaf_sizes = [math.prod(s) if s else 1 for s in leaf_shapes]

    leaf_indices = []
    for size, max_ord in zip(leaf_sizes, leaf_order):
        leaf_indices.append(_enumerate_total_degree(size, max_ord))

    all_exponents = []
    for combo in product(*leaf_indices):
        flat_exp = [e for leaf_exp in combo for e in leaf_exp]
        all_exponents.append(tuple(flat_exp))

    return MultiIndexArray(MultiIndex(*exp) for exp in all_exponents)


def check_leaf_bounds(
    multiindices: "MultiIndexArray",
    leaf_shapes: "tuple[tuple[int, ...], ...]",
    leaf_order: "tuple[int, ...]",
) -> Array:
    """Return boolean mask for monomials within per-leaf total degree bounds.

    Parameters
    ----------
    multiindices : MultiIndexArray
        Tuple of MultiIndex objects, one per monomial.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    leaf_order : tuple[int, ...]
        Maximum total degree for each leaf.

    Returns
    -------
    Array
        Boolean mask of shape (m,), True for monomials within bounds.
    """
    exponents = multiindices.to_jnp()  # (d, m)
    num_leaves = len(leaf_shapes)
    masks = []
    for leaf_idx in range(num_leaves):
        slc = leaf_slice(leaf_shapes, leaf_idx)
        leaf_total = jnp.sum(exponents[slc, :], axis=0)
        masks.append(leaf_total <= leaf_order[leaf_idx])
    return jnp.all(jnp.stack(masks), axis=0)


def compute_leaf_order(
    multiindices: "MultiIndexArray", leaf_shapes: "tuple[tuple[int, ...], ...]"
) -> "tuple[int, ...]":
    """Compute per-leaf total degree from a MultiIndexArray.

    Parameters
    ----------
    multiindices : MultiIndexArray
        Tuple of MultiIndex objects, one per monomial.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.

    Returns
    -------
    tuple[int, ...]
        Per-leaf total degree bounds inferred from multiindices.
    """
    exponents = multiindices.to_jnp()  # (d, m)
    result = []
    for leaf_idx in range(len(leaf_shapes)):
        slc = leaf_slice(leaf_shapes, leaf_idx)
        leaf_total = jnp.max(jnp.sum(exponents[slc, :], axis=0))
        result.append(int(leaf_total))
    return tuple(result)


# ---------------------------------------------------------------------------
# Taylor term utilities
# ---------------------------------------------------------------------------


def _merge_taylor_terms(
    coeffs1: Array,
    mia1: "MultiIndexArray",
    coeffs2: Array,
    mia2: "MultiIndexArray",
) -> "Tuple[Array, MultiIndexArray]":
    """Merge Taylor terms from two polynomials.

    Coeffs have shape (*output_shape, num_monomials).
    Concatenates along the last (monomial) axis.
    """
    coeffs = jnp.concatenate([coeffs1, coeffs2], axis=-1)
    mia = MultiIndexArray(mia1 + mia2)
    # return _compact_taylor_terms(coeffs, mia)
    return coeffs, mia


def _compact_taylor_terms(
    coeffs: Array, multiindices: "MultiIndexArray"
) -> "Tuple[Array, MultiIndexArray]":
    """Compact Taylor terms by zeroing negligible coefficients.

    Coeffs have shape (*output_shape, num_monomials).
    """
    original_shape = coeffs.shape
    output_shape = original_shape[:-1]
    num_monomials = original_shape[-1]

    if len(output_shape) == 0:
        norms = jnp.abs(coeffs)
    else:
        flat_coeffs = coeffs.reshape(-1, num_monomials)
        norms = jnp.linalg.norm(flat_coeffs, axis=0)

    nonzero_mask = norms > 1e-12
    coeffs_compact = jnp.where(nonzero_mask, coeffs, 0.0)
    return coeffs_compact, multiindices
