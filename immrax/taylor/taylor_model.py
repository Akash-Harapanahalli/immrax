"""Taylor Model implementation for reachability analysis.

Taylor models represent sets as polynomial approximations with rigorous
interval remainder bounds, providing a powerful tool for propagating
uncertainty through nonlinear functions.
"""

from typing import Callable, Tuple
from dataclasses import dataclass
from functools import lru_cache, cached_property
import math

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike

import jax.tree_util

from immrax.inclusion import Interval, interval, icentpert, iconcatenate
from immrax.utils import inv_fact


# ---------------------------------------------------------------------------
# PyTree Domain Helpers
# ---------------------------------------------------------------------------

def _is_interval_leaf(x):
    """Check if x is an Interval (leaf node in domain pytree)."""
    return isinstance(x, Interval)


def _get_domain_metadata(domain):
    """Extract pytree structure and leaf info from domain.

    Parameters
    ----------
    domain : Interval or PyTree[Interval]
        Domain intervals. Can be single Interval or any pytree structure
        (list, dict, nested) with Interval leaves.

    Returns
    -------
    treedef : jax.tree_util.PyTreeDef
        PyTree structure definition for the domain.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval in tree traversal order.
    flat_domain : Interval
        Single flattened Interval containing all domain variables concatenated.
    """
    treedef = jax.tree_util.tree_structure(domain, is_leaf=_is_interval_leaf)
    leaves = jax.tree_util.tree_leaves(domain, is_leaf=_is_interval_leaf)

    if len(leaves) == 0:
        raise ValueError("Domain must contain at least one Interval")

    leaf_shapes = tuple(iv.lower.shape for iv in leaves)

    # Flatten all leaves into single domain interval
    flat = []
    for iv in leaves:
        if iv.shape == ():
            flat.append(iv[None])
        else:
            flat.append(iv.reshape(-1))

    flat_domain = iconcatenate(flat)

    return treedef, leaf_shapes, flat_domain


def _unflatten_array_to_pytree(treedef, leaf_shapes, flat_array):
    """Unflatten a 1D array into a pytree structure matching leaf_shapes.

    Parameters
    ----------
    treedef : jax.tree_util.PyTreeDef
        Target pytree structure.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf.
    flat_array : Array
        Flattened array to unflatten.

    Returns
    -------
    PyTree[Array]
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
            leaves.append(flat_array[start:start + size].reshape(shape))
        start += size

    return jax.tree_util.tree_unflatten(treedef, leaves)


def _get_domain_from_metadata(treedef, leaf_shapes, flat_domain) -> "Interval | list | dict":
    """Reconstruct domain Interval pytree from metadata.

    Parameters
    ----------
    treedef : jax.tree_util.PyTreeDef
        Target pytree structure.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    flat_domain : Interval
        Flattened domain interval.

    Returns
    -------
    Interval or PyTree[Interval]
        Domain Intervals reshaped into the pytree structure.
    """
    if len(leaf_shapes) == 1:
        shape = leaf_shapes[0]
        if shape == ():
            return flat_domain[0] if flat_domain.shape != () else flat_domain
        return flat_domain.reshape(shape)

    leaves = []
    start = 0
    for shape in leaf_shapes:
        size = math.prod(shape) if shape else 1
        if shape == ():
            leaves.append(flat_domain[start])
        else:
            leaves.append(flat_domain[start:start + size].reshape(shape))
        start += size

    return jax.tree_util.tree_unflatten(treedef, leaves)


def _normalize_order_pytree(order, domain_treedef, leaf_shapes):
    """Normalize order specification to per-leaf and per-variable tuples.

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
    per_leaf_order : tuple[int, ...]
        Total degree bound for each leaf (one per leaf).
    per_var_order : tuple[int, ...]
        Per-variable order bounds (one per scalar domain variable).
    """
    num_leaves = len(leaf_shapes)

    if isinstance(order, int):
        per_leaf_order = tuple([order] * num_leaves)
    else:
        # order is a pytree - flatten it
        order_leaves = jax.tree_util.tree_leaves(order)
        if len(order_leaves) != num_leaves:
            raise ValueError(
                f"Order pytree has {len(order_leaves)} leaves but domain has {num_leaves} leaves"
            )
        per_leaf_order = tuple(int(o) for o in order_leaves)

    # Compute per-variable order from per-leaf order
    per_var_order = []
    for shape, leaf_ord in zip(leaf_shapes, per_leaf_order):
        size = math.prod(shape) if shape else 1
        per_var_order.extend([leaf_ord] * size)

    return per_leaf_order, tuple(per_var_order)


def _leaf_slice(leaf_shapes: "tuple[tuple[int, ...], ...]", leaf_idx: int) -> slice:
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


# ---------------------------------------------------------------------------
# Legacy ArgumentStructure (deprecated, for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgumentStructure:
    """Metadata mapping domain variables to function arguments.

    For multi-argument Taylor models where f(*args) has arguments with
    different shapes, this tracks which domain variables correspond to
    which argument.

    Example
    -------
    For f(t, x) where t: () (scalar) and x: (2,):
    - arg_shapes = ((), (2,))
    - arg_sizes = (1, 2)
    - arg_starts = (0, 1)
    - total_dim = 3
    """
    arg_shapes: tuple[tuple[int, ...], ...]  # Shape of each argument

    @cached_property
    def arg_sizes(self) -> tuple[int, ...]:
        """Number of scalar domain variables per argument."""
        return tuple(math.prod(s) if s else 1 for s in self.arg_shapes)

    @cached_property
    def arg_starts(self) -> tuple[int, ...]:
        """Starting index in flat domain for each argument."""
        starts = [0]
        for size in self.arg_sizes[:-1]:
            starts.append(starts[-1] + size)
        return tuple(starts)

    @property
    def num_args(self) -> int:
        """Number of arguments."""
        return len(self.arg_shapes)

    @property
    def total_dim(self) -> int:
        """Total number of domain variables across all arguments."""
        return sum(self.arg_sizes)

    def arg_slice(self, arg_idx: int) -> slice:
        """Get slice for domain variables of argument arg_idx."""
        start = self.arg_starts[arg_idx]
        return slice(start, start + self.arg_sizes[arg_idx])

    def __hash__(self):
        return hash(self.arg_shapes)

# Zonotope is imported lazily in to_zonotope() to avoid circular import


def _normalize_order(order: "int | tuple[int, ...] | None", d: int) -> "tuple[int, ...]":
    """Normalize an order specification to a tuple of length d.

    Parameters
    ----------
    order : int, tuple[int, ...], or None
        If int, broadcast to all variables.  If tuple, must have length d.
        If None, raises ValueError.
    d : int
        Number of domain variables.

    Returns
    -------
    tuple[int, ...]
        Per-variable order bounds, length d.
    """
    if order is None:
        raise ValueError("order must not be None")
    if isinstance(order, (int,)):
        return tuple([order] * d)
    order = tuple(order)
    if len(order) != d:
        raise ValueError(f"order tuple length {len(order)} != d={d}")
    return order


def _max_order(a: "tuple[int, ...]", b: "tuple[int, ...]") -> "tuple[int, ...]":
    """Element-wise maximum of two order tuples."""
    return tuple(max(ai, bi) for ai, bi in zip(a, b))


# Cache for canonical exponent structures
@lru_cache(maxsize=128)
def _get_canonical_exponents(d: int, max_orders: "int | tuple[int, ...]") -> Array:
    """Get cached canonical exponents for given dimension and per-variable orders.

    Parameters
    ----------
    d : int
        Number of variables.
    max_orders : int or tuple[int, ...]
        Per-variable maximum exponent.  If int, broadcast to all variables.
    """
    if isinstance(max_orders, int):
        max_orders = tuple([max_orders] * d)
    return _generate_exponents_impl(d, max_orders)


def _generate_exponents_impl(d: int, max_orders: "tuple[int, ...]") -> Array:
    """Generate all exponent multi-indices with per-variable bounds."""
    from itertools import product

    ranges = [range(max_orders[i] + 1) for i in range(d)]
    import numpy as np
    exponents = list(product(*ranges))
    return np.array(exponents, dtype=np.int32).T


def _enumerate_total_degree(dim: int, max_deg: int) -> list[tuple[int, ...]]:
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
def _get_leaf_total_degree_exponents(
    leaf_shapes: "tuple[tuple[int, ...], ...]",
    per_leaf_order: tuple[int, ...]
) -> Array:
    """Generate exponents with per-leaf total degree bounds.

    For multi-leaf domains (pytree of Intervals), this generates all monomials
    where the total degree of exponents for each leaf is bounded separately.

    Parameters
    ----------
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    per_leaf_order : tuple[int, ...]
        Maximum total degree for each leaf.

    Returns
    -------
    Array
        Exponent matrix, shape (total_dim, num_monomials).

    Example
    -------
    For domain = [interval_t, interval_x] where t: () and x: (2,),
    with per_leaf_order = (2, 3):
    - Generates monomials t^a * x1^b1 * x2^b2 where a <= 2 and b1+b2 <= 3
    - Count: (2+1) * C(2+3, 3) = 3 * 10 = 30 monomials
    """
    from itertools import product
    import numpy as np

    # Compute sizes from shapes
    leaf_sizes = [math.prod(s) if s else 1 for s in leaf_shapes]

    # Generate exponents for each leaf separately
    leaf_indices = []
    for size, max_ord in zip(leaf_sizes, per_leaf_order):
        leaf_indices.append(_enumerate_total_degree(size, max_ord))

    # Take Cartesian product across leaves
    all_exponents = []
    for combo in product(*leaf_indices):
        # Flatten the tuple of tuples into a single exponent vector
        flat_exp = [e for leaf_exp in combo for e in leaf_exp]
        all_exponents.append(flat_exp)

    return np.array(all_exponents, dtype=np.int32).T


def _check_per_leaf_bounds(
    exponents: Array,
    leaf_shapes: "tuple[tuple[int, ...], ...]",
    per_leaf_order: tuple[int, ...]
) -> Array:
    """Return boolean mask for monomials within per-leaf total degree bounds.

    Parameters
    ----------
    exponents : Array, shape (d, m)
        Exponent matrix.
    leaf_shapes : tuple[tuple[int, ...], ...]
        Shape of each leaf Interval.
    per_leaf_order : tuple[int, ...]
        Maximum total degree for each leaf.

    Returns
    -------
    Array
        Boolean mask of shape (m,), True for monomials within bounds.
    """
    num_leaves = len(leaf_shapes)
    masks = []
    for leaf_idx in range(num_leaves):
        slc = _leaf_slice(leaf_shapes, leaf_idx)
        # Sum exponents for this leaf's variables (total degree for this leaf)
        leaf_total = jnp.sum(exponents[slc, :], axis=0)
        masks.append(leaf_total <= per_leaf_order[leaf_idx])
    return jnp.all(jnp.stack(masks), axis=0)


# Backward compatibility aliases
def _get_arg_total_degree_exponents(arg_structure, per_arg_order):
    """Deprecated: Use _get_leaf_total_degree_exponents instead."""
    return _get_leaf_total_degree_exponents(arg_structure.arg_shapes, per_arg_order)


def _check_per_arg_bounds(exponents, arg_structure, per_arg_order):
    """Deprecated: Use _check_per_leaf_bounds instead."""
    return _check_per_leaf_bounds(exponents, arg_structure.arg_shapes, per_arg_order)


@register_pytree_node_class
class TaylorModel:
    r"""Defines a Taylor model set representation with arbitrary output shape.

    .. math::
        TM = \{ p(x) + r : x \in D, r \in I \}

    where:
    - :math:`p(x)` is a multivariate polynomial in the domain variables
    - :math:`D \subseteq \mathbb{R}^d` is the domain (typically a box)
    - :math:`I \subseteq \mathbb{R}^{*output\_shape}` is the interval remainder

    The polynomial is represented in the form:

    .. math::
        p(x) = c + \sum_{|\alpha| \leq k} a_\alpha (x - x_0)^\alpha

    where :math:`\alpha` is a multi-index, :math:`x_0` is the expansion point,
    and :math:`k` is the polynomial order.

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients. Shape is (*output_shape, num_monomials) where
        num_monomials = C(d+k, k). The monomial axis is always the **last** axis.
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials), integer entries.
        Each column is a multi-index α.
    remainder : Interval
        Interval remainder bounds with shape (*output_shape,).
    domain : Interval
        Domain box with shape (d,). Center and half-width are accessed
        via ``domain.center`` and ``domain.pert``.

    References
    ----------
    .. [1] Makino, K., and Berz, M. "Taylor models and other validated functional
           inclusion methods." Int. J. Pure Appl. Math. 4.4 (2003): 379-456.
    .. [2] Chen, X., et al. "Taylor model flowpipe construction for non-linear
           hybrid systems." RTSS 2012.
    """

    coeffs: Array  # Polynomial coefficients, shape (*output_shape, num_monomials)
    exponents: Array  # Exponent matrix, shape (d, num_monomials)
    remainder: Interval  # Interval remainder, shape (*output_shape,)
    domain: Interval  # Domain box, shape (d,)
    center: Array  # Expansion point, shape (d,)

    def __init__(
        self,
        coeffs: ArrayLike,
        exponents: ArrayLike,
        remainder: Interval,
        domain: Interval,
        center: ArrayLike = None,
        _static_order: "int | tuple[int, ...] | None" = None,
        _domain_treedef: "jax.tree_util.PyTreeDef | None" = None,
        _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
        _per_leaf_order: "tuple[int, ...] | None" = None,
    ) -> None:
        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.remainder = remainder
        self.domain = domain
        if center is None:
            self.center = domain.center
        else:
            self.center = jnp.asarray(center)

        d = self.exponents.shape[0]

        # Store the polynomial order as static data (for JIT compatibility)
        # _static_order is always a tuple[int, ...] of length d.
        if _static_order is not None:
            if isinstance(_static_order, int):
                self._static_order = tuple([_static_order] * d)
            else:
                self._static_order = tuple(_static_order)
        else:
            # Compute per-variable max from exponents (requires concrete values)
            self._static_order = tuple(
                int(jnp.max(self.exponents[i])) for i in range(d)
            )

        # Store output shape (all axes except the last monomial axis)
        # Monomials are always the LAST axis of coeffs
        self._output_shape = self.coeffs.shape[:-1]

        # Structured domain support: store pytree metadata
        self._domain_treedef = _domain_treedef
        self._leaf_shapes = _leaf_shapes
        self._per_leaf_order = _per_leaf_order

        # Validate dimensions
        if self.coeffs.ndim < 1:
            raise ValueError(f"coeffs must be at least 1D, got shape {self.coeffs.shape}")
        if self.exponents.ndim != 2:
            raise ValueError(f"exponents must be 2D, got shape {self.exponents.shape}")
        if self.coeffs.shape[-1] != self.exponents.shape[1]:
            raise ValueError(
                f"coeffs and exponents must have same number of monomials: "
                f"{self.coeffs.shape[-1]} vs {self.exponents.shape[1]}"
            )
        if self.remainder.shape != self._output_shape:
            raise ValueError(
                f"remainder and coeffs must have same output shape: "
                f"{self.remainder.shape} vs {self._output_shape}"
            )
        if self.domain.lower.shape[0] != self.exponents.shape[0]:
            raise ValueError(
                f"domain must match exponents dimension: "
                f"{self.domain.lower.shape[0]} vs {self.exponents.shape[0]}"
            )

    # --- Pytree methods ---

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, Array, Interval, Interval, Array], dict]:
        return (
            (
                self.coeffs,
                self.exponents,
                self.remainder,
                self.domain,
                self.center,
            ),
            {
                "_static_order": self._static_order,
                "_output_shape": self._output_shape,
                "_domain_treedef": self._domain_treedef,
                "_leaf_shapes": self._leaf_shapes,
                "_per_leaf_order": self._per_leaf_order,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorModel":
        static_order = aux_data.get("_static_order") if aux_data else None
        domain_treedef = aux_data.get("_domain_treedef") if aux_data else None
        leaf_shapes = aux_data.get("_leaf_shapes") if aux_data else None
        per_leaf_order = aux_data.get("_per_leaf_order") if aux_data else None
        # _output_shape is recomputed in __init__ from coeffs.shape[:-1]
        coeffs, exponents, remainder, domain, center = children
        return cls(
            coeffs, exponents, remainder, domain,
            center=center,
            _static_order=static_order,
            _domain_treedef=domain_treedef,
            _leaf_shapes=leaf_shapes,
            _per_leaf_order=per_leaf_order,
        )

    # --- Properties ---

    @property
    def n(self) -> int:
        """Total number of output elements (product of output_shape)."""
        import math
        return math.prod(self._output_shape) if self._output_shape else 1

    @property
    def domain_center(self) -> Array:
        """Center of the domain box, shape (d,). Deprecated: use center or domain.center."""
        return self.domain.center

    @property
    def domain_radius(self) -> Array:
        """Half-width of the domain box, shape (d,). Deprecated: use domain.pert."""
        return self.domain.pert

    @property
    def shifted_domain(self) -> Interval:
        """Domain shifted by center: D - center, for bounding monomials (x - center)^alpha."""
        return interval(self.domain.lower - self.center, self.domain.upper - self.center)

    @property
    def d(self) -> int:
        """Domain (input) dimension."""
        return self.exponents.shape[0]

    @property
    def num_monomials(self) -> int:
        """Number of monomial terms."""
        return self.coeffs.shape[-1]

    @property
    def order(self) -> "tuple[int, ...]":
        """Per-variable maximum polynomial order."""
        return self._static_order

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape."""
        return self._output_shape

    @property
    def domain_treedef(self) -> "jax.tree_util.PyTreeDef | None":
        """PyTree structure of the original domain, or None for single Interval."""
        return self._domain_treedef

    @property
    def leaf_shapes(self) -> "tuple[tuple[int, ...], ...] | None":
        """Shapes of each leaf Interval in the domain pytree, or None."""
        return self._leaf_shapes

    @property
    def per_leaf_order(self) -> "tuple[int, ...] | None":
        """Per-leaf total degree bounds, or None for per-variable mode."""
        return self._per_leaf_order

    @property
    def is_structured(self) -> bool:
        """True if this TM uses per-leaf total degree bounds (structured domain)."""
        return self._domain_treedef is not None
    
    @property
    def structured_center(self):
        """Get the center of the domain as a structured pytree.

        Returns the center reshaped to match the original domain pytree structure.
        For non-structured TaylorModels, returns self.center unchanged.
        """
        if not self.is_structured:
            return self.center
        else:
            return _unflatten_array_to_pytree(
                self._domain_treedef, self._leaf_shapes, self.center
            )

    def unpack_to_pytree(self) -> "TaylorModel | list | dict":
        """Unpack a structured TaylorModel into a pytree of sub-TaylorModels.

        For a TaylorModel created from a pytree domain (e.g., list or dict of Intervals),
        this returns a matching pytree of TaylorModels, where each sub-TM represents
        the identity over the corresponding leaf interval.

        For non-structured TaylorModels, returns self unchanged.

        Returns
        -------
        TaylorModel or PyTree[TaylorModel]
            If structured: pytree of sub-TaylorModels matching domain structure.
            If not structured: self.
        """
        if not self.is_structured:
            return self

        # Extract sub-TMs for each leaf
        sub_tms = []
        start = 0
        for leaf_idx, shape in enumerate(self._leaf_shapes):
            size = math.prod(shape) if shape else 1
            end = start + size

            # Slice coefficients and remainder for this leaf's outputs
            if size == 1:
                sub_coeffs = self.coeffs[start]  # scalar output
                sub_remainder = self.remainder[start]
            else:
                sub_coeffs = self.coeffs[start:end]
                sub_remainder = self.remainder[start:end]

            # Create sub-TM (shares the full domain but only represents this leaf's outputs)
            sub_tm = TaylorModel(
                sub_coeffs,
                self.exponents,
                sub_remainder,
                self.domain,
                center=self.center,
                _static_order=self._static_order,
                _domain_treedef=self._domain_treedef,
                _leaf_shapes=self._leaf_shapes,
                _per_leaf_order=self._per_leaf_order,
            )
            sub_tms.append(sub_tm)
            start = end

        # Reconstruct the pytree structure
        return jax.tree_util.tree_unflatten(self._domain_treedef, sub_tms)
            

    def __getitem__(self, idx) -> "TaylorModel":
        """Get component(s) of the Taylor Model.

        Indexing preserves the monomial axis (last axis of coeffs).
        The index applies to the output shape dimensions.
        """
        # Index into output shape dimensions (coeffs has monomials as last axis)
        c = self.coeffs[idx]

        # Ensure coeffs has at least the monomial dimension
        if c.ndim == 0:
            # Single monomial coefficient was selected - shouldn't happen with proper indexing
            raise ValueError("Cannot index into monomial dimension directly")

        # Index remainder the same way
        rem = self.remainder[idx]

        # Ensure remainder shape matches coeffs output shape
        rem_shape = rem.shape if hasattr(rem, 'shape') else ()
        coeff_out_shape = c.shape[:-1]

        if rem_shape != coeff_out_shape:
            # Need to reshape remainder to match
            if rem_shape == ():
                # Scalar interval - keep as scalar output shape
                pass
            else:
                # Reshape not needed, shapes should match from same indexing
                pass

        return TaylorModel(c, self.exponents, rem, self.domain,
                           center=self.center, _static_order=self._static_order,
                           _domain_treedef=self._domain_treedef,
                           _leaf_shapes=self._leaf_shapes,
                           _per_leaf_order=self._per_leaf_order)

    def __len__(self) -> int:
        """Length along first axis of output shape."""
        if len(self._output_shape) == 0:
            raise TypeError("Scalar TaylorModel has no len()")
        return self._output_shape[0]

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return self.coeffs.dtype

    @property
    def constant_term(self) -> Array:
        """Get the constant (order 0) term of the polynomial.

        Returns array with shape (*output_shape,).
        """
        # Find the column where all exponents are 0
        is_constant = jnp.all(self.exponents == 0, axis=0)  # (num_monomials,)
        # Broadcast mask to match coeffs shape and sum along monomial axis (last)
        # is_constant has shape (m,), coeffs has shape (*output_shape, m)
        # We need to broadcast is_constant to match coeffs for the where operation
        return jnp.sum(
            jnp.where(is_constant, self.coeffs, 0.0), axis=-1
        )

    # --- Polynomial evaluation ---

    @property
    def polynomial (self) -> 'TaylorPolynomial' :
        from immrax.taylor.taylor_polynomial import TaylorPolynomial

        return TaylorPolynomial(
            self.coeffs, self.exponents, self.center,
            _static_order=self._static_order,
        )

    def evaluate_polynomial(self, x: ArrayLike) -> Array:
        """Evaluate the polynomial part at a point.

        Parameters
        ----------
        x : ArrayLike
            Point in the domain, shape (d,)

        Returns
        -------
        Array
            Polynomial value, shape (*output_shape,)
        """
        x = jnp.asarray(x)
        # Shift to centered coordinates (x - center)
        x_centered = x - self.center

        # Evaluate each monomial: monomial_i = prod_j x_centered[j]^exponents[j, i]
        # x_centered (d,) -> (d, 1), exponents (d, m) -> x_centered^exponents (d, m)
        # Note: No factorial division here - coefficients already store D^α f(c) / α!
        monomials = jnp.prod(x_centered[:, None] ** self.exponents, axis=0)  # (m,)

        # Sum coeffs * monomials along the last (monomial) axis
        # coeffs has shape (*output_shape, m), monomials has shape (m,)

        return jnp.sum(self.coeffs * monomials, axis=-1)

    def evaluate(self, x: ArrayLike) -> Interval:
        """Evaluate the Taylor model at a point, returning an interval.

        Parameters
        ----------
        x : ArrayLike
            Point in the domain, shape (d,)

        Returns
        -------
        Interval
            Interval containing the true value
        """
        poly_val = self.evaluate_polynomial(x)
        return poly_val + self.remainder
    
    def __call__(self, x: ArrayLike) -> Interval:
        """Evaluate the Taylor model at a point, returning an interval.

        Parameters
        ----------
        x : ArrayLike
            Point in the domain, shape (d,)

        Returns
        -------
        Interval
            Interval containing the true value
        """
        return self.evaluate(x)

    # --- Set operations ---

    # --- Set operations ---
    # Arithmetic operations are implemented as standalone functions in nattm.py
    # and registered to the class similar to Interval/natif.
    
    def __add__(self, other: "TaylorModel" | ArrayLike) -> "TaylorModel": ...
    def __radd__(self, other: ArrayLike) -> "TaylorModel": ...
    def __sub__(self, other: "TaylorModel | ArrayLike") -> "TaylorModel": ...
    def __rsub__(self, other: ArrayLike) -> "TaylorModel": ...
    def __neg__(self) -> "TaylorModel": ...
    def __mul__(self, other: "TaylorModel | ArrayLike") -> "TaylorModel": ...
    def __rmul__(self, other: ArrayLike) -> "TaylorModel": ...
    def __pow__(self, power: int) -> "TaylorModel": ...
    def __rmatmul__(self, other: ArrayLike) -> "TaylorModel": ...
    
    def multiply(self, other: "TaylorModel", max_order: int | None = None) -> "TaylorModel": ...
    



    def _bound_polynomial(self) -> Interval: ...


    # --- Conversion methods ---

    def to_zonotope(self):
        """Convert to a zonotope (overapproximation).

        Each monomial term becomes a generator.
        Note: Currently only supports 1D output shapes (vectors).

        Returns
        -------
        Zonotope
            Zonotope overapproximation of the Taylor model
        """
        # Lazy import to avoid circular dependency
        from immrax.generator.sets.zonotope import Zonotope

        if len(self._output_shape) != 1:
            raise NotImplementedError(
                f"to_zonotope only supports 1D output shapes, got {self._output_shape}"
            )

        n = self._output_shape[0]

        # Center is the constant term plus remainder center
        remainder_center = self.remainder.center
        remainder_radius = self.remainder.pert

        # Compute monomial bounds over shifted domain (D - center)
        mono_bounds = _bound_monomials_over_domain(self.exponents, self.shifted_domain, max(self._static_order))
        mono_center = (mono_bounds.lower + mono_bounds.upper) / 2  # (m,)
        mono_radius = (mono_bounds.upper - mono_bounds.lower) / 2  # (m,)

        is_constant = jnp.all(self.exponents == 0, axis=0)  # (m,)

        # Center contribution: sum of coeff_i * mono_center_i for non-constant terms
        center_shift = jnp.sum(
            jnp.where(~is_constant, self.coeffs * mono_center, 0.0), axis=-1
        )  # (n,)

        ox = self.constant_term + remainder_center + center_shift

        # Non-constant generators from polynomial terms, scaled by mono_radius
        non_const_mask = ~is_constant  # (m,)
        poly_generators = jnp.where(
            non_const_mask, self.coeffs * mono_radius, 0.0
        )  # (n, m)

        # Remainder generators (axis-aligned)
        remainder_generators = jnp.diag(remainder_radius)  # (n, n)

        # Concatenate all generators
        G = jnp.concatenate([poly_generators, remainder_generators], axis=1)

        # Remove zero generators (columns with near-zero norm)
        col_norms = jnp.linalg.norm(G, axis=0)
        nonzero_mask = col_norms > 1e-12
        # For JIT compatibility, keep all columns but zero out small ones
        G = jnp.where(nonzero_mask[None, :], G, 0.0)

        return Zonotope(ox, G)

    def interval_hull(self) -> Interval:
        """Compute the interval hull (bounding box)."""
        poly_bounds = self._bound_polynomial()
        return poly_bounds + self.remainder

    def contains(self, x: ArrayLike) -> Array:
        """Check if a point is in the range (necessary condition)."""
        x = jnp.asarray(x)
        hull = self.interval_hull()
        return jnp.all((x >= hull.lower) & (x <= hull.upper))

    # --- Order reduction and canonicalization ---

    def to_canonical(
        self,
        target_order: "int | tuple[int, ...] | None" = None,
        method: str | None = None,
    ) -> "TaylorModel":
        """Convert to canonical exponent structure.

        The canonical structure includes all monomials up to target_order,
        sorted consistently. This ensures TMs are compatible for concatenation.

        For multi-argument TMs (with _arg_structure set), this preserves the
        per-argument total degree structure.

        Parameters
        ----------
        target_order : int, tuple[int, ...], or None
            Target per-variable order (for standard TMs) or per-argument order
            (for multi-arg TMs). If int, broadcast to all variables/arguments.
            If None, uses current order.
        method : str, optional
            Algorithm for index mapping: "broadcast" (O(m * num_canonical), fast
            for small num_canonical), "searchsorted" (O(m log num_canonical),
            better for large num_canonical), or None to auto-select based on
            num_canonical threshold.

        Returns
        -------
        TaylorModel
            TM with canonical exponent structure
        """
        # Handle structured domain mode (per-leaf total degree bounds)
        if self._leaf_shapes is not None and self._per_leaf_order is not None:
            return self._to_canonical_per_leaf(target_order, method)

        # Standard per-variable mode
        if target_order is None:
            target_order = self._static_order
        if isinstance(target_order, int):
            target_order = tuple([target_order] * self.d)
        else:
            target_order = tuple(target_order)

        # Generate canonical exponents (uses cached version for JIT compatibility)
        canonical_exp = _get_canonical_exponents(self.d, target_order)
        num_canonical = canonical_exp.shape[1]

        # Auto-select method based on num_canonical if not specified
        # Threshold ~50: below this broadcast is faster due to lower overhead
        if method is None:
            method = "searchsorted" if num_canonical > 50 else "broadcast"

        # Compute a unique hash for each exponent vector
        # Use weighted sum: sum_i exp[i] * (max_order+1)^i
        base = max(target_order) + 2  # Ensure no collisions
        powers = base ** jnp.arange(self.d)

        # Hash current exponents: (m,)
        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)

        # Hash canonical exponents: (num_canonical,)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        # Check if each current term fits within per-variable bounds
        target_arr = jnp.array(target_order, dtype=jnp.int32)[:, None]  # (d, 1)
        has_match = jnp.all(self.exponents <= target_arr, axis=0)  # (m,)

        if method == "broadcast":
            # O(m * num_canonical) - fast for small num_canonical
            match_matrix = current_hash[:, None] == canonical_hash[None, :]  # (m, num_canonical)
            scatter_matrix = match_matrix.astype(self.dtype)  # (m, num_canonical)
        elif method == "searchsorted":
            # O(m log num_canonical) - better for large num_canonical
            sort_perm = jnp.argsort(canonical_hash)
            sorted_canonical_hash = canonical_hash[sort_perm]
            sorted_indices = jnp.searchsorted(sorted_canonical_hash, current_hash)
            sorted_indices = jnp.clip(sorted_indices, 0, num_canonical - 1)
            canonical_indices = sort_perm[sorted_indices]
            scatter_matrix = jax.nn.one_hot(canonical_indices, num_canonical, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'broadcast' or 'searchsorted'.")

        # Mask coefficients: broadcasting has_match (m,) with coeffs (*output_shape, m)
        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)  # (*output_shape, m)

        # Contract along monomial axis: (*output_shape, m) @ (m, num_canonical) -> (*output_shape, num_canonical)
        new_coeffs = masked_coeffs @ scatter_matrix  # (*output_shape, num_canonical)

        # Bound terms above target_order and add to remainder
        absorb_mask = ~has_match  # (m,)

        # Compute monomial bounds over shifted domain (D - center)
        mono_bounds = _bound_monomials_over_domain(self.exponents, self.shifted_domain, max(self._static_order))

        absorb_coeffs = jnp.where(absorb_mask, self.coeffs, 0.0)  # (*output_shape, m)

        zeros = jnp.zeros_like(absorb_coeffs)
        c_pos = jnp.maximum(absorb_coeffs, zeros)
        c_neg = jnp.minimum(absorb_coeffs, zeros)

        term_lower = c_pos * mono_bounds.lower + c_neg * mono_bounds.upper
        term_upper = c_pos * mono_bounds.upper + c_neg * mono_bounds.lower

        absorbed_lower = jnp.sum(term_lower, axis=-1)
        absorbed_upper = jnp.sum(term_upper, axis=-1)

        new_remainder = self.remainder + interval(absorbed_lower, absorbed_upper)

        return TaylorModel(
            new_coeffs,
            canonical_exp,
            new_remainder,
            self.domain,
            center=self.center,
            _static_order=tuple(target_order),
        )

    def _to_canonical_per_leaf(
        self,
        target_order: "tuple[int, ...] | None" = None,
        method: str | None = None,
    ) -> "TaylorModel":
        """Convert to canonical exponent structure for structured domain mode.

        Uses per-leaf total degree bounds instead of per-variable bounds.
        """
        leaf_shapes = self._leaf_shapes
        num_leaves = len(leaf_shapes)

        # Determine per_leaf_order: use target_order only if it has the correct
        # length (number of leaves), otherwise use stored _per_leaf_order.
        # This handles the case where to_canonical is called with per-variable
        # order from arithmetic operations.
        if target_order is not None:
            if isinstance(target_order, int):
                per_leaf_order = tuple([target_order] * num_leaves)
            elif len(target_order) == num_leaves:
                per_leaf_order = tuple(target_order)
            else:
                # target_order has wrong length (probably per-variable order),
                # use stored per_leaf_order instead
                per_leaf_order = self._per_leaf_order
        else:
            per_leaf_order = self._per_leaf_order

        per_leaf_order = tuple(per_leaf_order)

        # Generate canonical exponents with per-leaf total degree bounds
        canonical_exp = _get_leaf_total_degree_exponents(leaf_shapes, per_leaf_order)
        num_canonical = canonical_exp.shape[1]

        # Auto-select method
        if method is None:
            method = "searchsorted" if num_canonical > 50 else "broadcast"

        # Compute hash for current and canonical exponents
        # Need a hash that's unique across all possible exponents
        max_exp_per_var = max(max(self._static_order), max(per_leaf_order)) + 1
        base = max_exp_per_var + 2
        powers = base ** jnp.arange(self.d)

        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        # Check if current terms are within per-leaf bounds
        has_match = _check_per_leaf_bounds(self.exponents, leaf_shapes, per_leaf_order)

        if method == "broadcast":
            match_matrix = current_hash[:, None] == canonical_hash[None, :]
            scatter_matrix = match_matrix.astype(self.dtype)
        elif method == "searchsorted":
            sort_perm = jnp.argsort(canonical_hash)
            sorted_canonical_hash = canonical_hash[sort_perm]
            sorted_indices = jnp.searchsorted(sorted_canonical_hash, current_hash)
            sorted_indices = jnp.clip(sorted_indices, 0, num_canonical - 1)
            canonical_indices = sort_perm[sorted_indices]
            scatter_matrix = jax.nn.one_hot(canonical_indices, num_canonical, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'broadcast' or 'searchsorted'.")

        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)
        new_coeffs = masked_coeffs @ scatter_matrix

        # Bound terms above target order
        absorb_mask = ~has_match
        mono_bounds = _bound_monomials_over_domain(
            self.exponents, self.shifted_domain, max(self._static_order)
        )

        absorb_coeffs = jnp.where(absorb_mask, self.coeffs, 0.0)
        zeros = jnp.zeros_like(absorb_coeffs)
        c_pos = jnp.maximum(absorb_coeffs, zeros)
        c_neg = jnp.minimum(absorb_coeffs, zeros)

        term_lower = c_pos * mono_bounds.lower + c_neg * mono_bounds.upper
        term_upper = c_pos * mono_bounds.upper + c_neg * mono_bounds.lower

        absorbed_lower = jnp.sum(term_lower, axis=-1)
        absorbed_upper = jnp.sum(term_upper, axis=-1)

        new_remainder = self.remainder + interval(absorbed_lower, absorbed_upper)

        # Compute per-variable order from per-leaf order
        per_var_order = []
        leaf_sizes = [math.prod(s) if s else 1 for s in leaf_shapes]
        for size, max_ord in zip(leaf_sizes, per_leaf_order):
            per_var_order.extend([max_ord] * size)

        return TaylorModel(
            new_coeffs,
            canonical_exp,
            new_remainder,
            self.domain,
            center=self.center,
            _static_order=tuple(per_var_order),
            _domain_treedef=self._domain_treedef,
            _leaf_shapes=leaf_shapes,
            _per_leaf_order=per_leaf_order,
        )

    def reduce_order(self, target_order: "int | tuple[int, ...]") -> "TaylorModel":
        """Reduce polynomial order, absorbing high-order terms into remainder.

        Parameters
        ----------
        target_order : int or tuple[int, ...]
            Target maximum polynomial order (per-variable or uniform).

        Returns
        -------
        TaylorModel
            Reduced order Taylor model (overapproximation)
        """
        if isinstance(target_order, int):
            target_order = tuple([target_order] * self.d)
        else:
            target_order = tuple(target_order)

        # Separate terms to keep and terms to absorb
        target_arr = jnp.array(target_order, dtype=jnp.int32)[:, None]  # (d, 1)
        keep_mask = jnp.all(self.exponents <= target_arr, axis=0)  # (m,)
        absorb_mask = ~keep_mask  # (m,)

        # Compute monomial bounds over shifted domain (D - center)
        mono_bounds_iv = _bound_monomials_over_domain(self.exponents, self.shifted_domain, max(self._static_order))
        mono_lower = mono_bounds_iv.lower  # (m,)
        mono_upper = mono_bounds_iv.upper  # (m,)

        # Only absorb terms above target_order
        # Broadcasting: absorb_mask (m,) with coeffs (*output_shape, m)
        absorb_coeffs = jnp.where(absorb_mask, self.coeffs, 0.0)  # (*output_shape, m)

        zeros = jnp.zeros_like(absorb_coeffs)
        c_pos = jnp.maximum(absorb_coeffs, zeros)  # (*output_shape, m)
        c_neg = jnp.minimum(absorb_coeffs, zeros)  # (*output_shape, m)

        # mono_lower/upper have shape (m,), broadcasting works naturally
        term_lower = c_pos * mono_lower + c_neg * mono_upper  # (*output_shape, m)
        term_upper = c_pos * mono_upper + c_neg * mono_lower  # (*output_shape, m)

        # Sum along monomial axis (last axis)
        absorbed_lower = jnp.sum(term_lower, axis=-1)  # (*output_shape,)
        absorbed_upper = jnp.sum(term_upper, axis=-1)  # (*output_shape,)

        absorbed_remainder = interval(absorbed_lower, absorbed_upper)

        # New remainder includes absorbed terms
        new_remainder = self.remainder + absorbed_remainder

        # Filter coefficients (zero out instead of filtering for JIT)
        # Broadcasting: keep_mask (m,) with coeffs (*output_shape, m)
        new_coeffs = jnp.where(keep_mask, self.coeffs, 0.0)

        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain,
            center=self.center,
            _static_order=tuple(target_order),
        )

    # --- String representation ---

    def __str__(self) -> str:
        return (
            f"TaylorModel(shape={self.shape}, d={self.d}, order={self.order}, "
            f"monomials={self.num_monomials})"
        )

    def __repr__(self) -> str:
        return (
            f"TaylorModel(coeffs={self.coeffs!r}, exponents={self.exponents!r}, "
            f"remainder={self.remainder!r}, domain={self.domain!r})"
        )


# --- Helper functions ---


def _bound_monomials_over_domain(exponents: Array, norm_domain: Interval, max_order: int | None = None) -> Interval:
    """Bound each monomial prod_j u_j^{e_j} over a domain interval.

    Uses natif(integer_pow) vmapped over variables, then interval reduce-product
    over the variable dimension.

    Parameters
    ----------
    exponents : Array, shape (d, m)
    norm_domain : Interval, shape (d,)
    max_order : int, optional
        Maximum exponent value (static). If None, computed from exponents
        (requires concrete values).

    Returns
    -------
    Interval
        Monomial bounds, shape (m,).
    """
    from immrax.inclusion import natif
    d = exponents.shape[0]
    max_e = max_order if max_order is not None else int(jnp.max(exponents))

    # Build a (d, max_e+1) table of interval powers: pow_table[j, e] = norm_domain[j] ** e
    # Using natif(integer_pow) for each static exponent value, vmapped over d variables.
    lo_table = jnp.ones((d, max_e + 1))
    hi_table = jnp.ones((d, max_e + 1))
    for e in range(max_e + 1):
        pow_fn = natif(lambda x, _e=e: jax.lax.integer_pow(x, _e))
        pow_vals = jax.vmap(pow_fn)(norm_domain)  # Interval shape (d,)
        lo_table = lo_table.at[:, e].set(pow_vals.lower)
        hi_table = hi_table.at[:, e].set(pow_vals.upper)

    # Gather per-(variable, monomial) powers: shape (d, m)
    gather_lo = jax.vmap(lambda row, idx: row[idx])(lo_table, exponents)
    gather_hi = jax.vmap(lambda row, idx: row[idx])(hi_table, exponents)
    grid = Interval(gather_lo, gather_hi)  # (d, m)

    # Reduce-product over variable dimension (axis 0)
    result = Interval(grid.lower[0], grid.upper[0])
    for j in range(1, d):
        result = result * Interval(grid.lower[j], grid.upper[j])

    return result



def _merge_taylor_terms(
    coeffs1: Array, exp1: Array, coeffs2: Array, exp2: Array
) -> Tuple[Array, Array]:
    """Merge Taylor terms from two Taylor models.

    Coeffs have shape (*output_shape, num_monomials).
    Concatenates along the last (monomial) axis.
    """
    # Simple concatenation along monomial axis (last axis)
    coeffs = jnp.concatenate([coeffs1, coeffs2], axis=-1)
    exp = jnp.concatenate([exp1, exp2], axis=1)
    return _compact_taylor_terms(coeffs, exp)


def _compact_taylor_terms(coeffs: Array, exponents: Array) -> Tuple[Array, Array]:
    """Compact Taylor terms by combining terms with identical exponents.

    Simplified version that removes zero coefficients.
    Coeffs have shape (*output_shape, num_monomials).
    """
    # Remove terms with zero coefficients
    # Flatten output shape for norm computation, then reshape back
    original_shape = coeffs.shape
    output_shape = original_shape[:-1]
    num_monomials = original_shape[-1]

    if len(output_shape) == 0:
        # Scalar output: coeffs has shape (m,)
        norms = jnp.abs(coeffs)
    else:
        # Flatten to (prod(output_shape), m) for norm computation
        flat_coeffs = coeffs.reshape(-1, num_monomials)
        norms = jnp.linalg.norm(flat_coeffs, axis=0)  # (m,)

    nonzero_mask = norms > 1e-12  # (m,)
    # Broadcasting: mask (m,) with coeffs (*output_shape, m)
    coeffs_compact = jnp.where(nonzero_mask, coeffs, 0.0)
    return coeffs_compact, exponents


def _generate_exponents(d: int, max_order: "int | tuple[int, ...]") -> Array:
    """Generate all exponent multi-indices up to given order.

    Parameters
    ----------
    d : int
        Number of variables
    max_order : int or tuple[int, ...]
        Maximum per-variable degree (int is broadcast to all variables).

    Returns
    -------
    Array
        Exponent matrix, shape (d, num_monomials)
    """
    # Use cached version for consistency and JIT compatibility
    return _get_canonical_exponents(d, max_order)


def taylor_model(
    coeffs: ArrayLike,
    exponents: ArrayLike,
    remainder: Interval | None = None,
    domain: Interval | None = None,
    center: ArrayLike | None = None,
) -> TaylorModel:
    """Create a Taylor model.

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients, shape (*output_shape, num_monomials).
        Monomials are always the last axis.
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials)
    remainder : Interval, optional
        Remainder interval with shape (*output_shape,). Default is zero interval.
    domain : Interval, optional
        Domain box with shape (d,). Default is [-1, 1]^d.
    center : ArrayLike, optional
        Expansion point, shape (d,). Default is domain center.

    Returns
    -------
    TaylorModel
        The constructed Taylor model
    """
    coeffs = jnp.asarray(coeffs)
    exponents = jnp.asarray(exponents, dtype=jnp.int32)

    output_shape = coeffs.shape[:-1]
    d = exponents.shape[0]

    if remainder is None:
        if len(output_shape) == 0:
            remainder = interval(jnp.zeros((), dtype=coeffs.dtype))
        else:
            remainder = interval(jnp.zeros(output_shape, dtype=coeffs.dtype))

    if domain is None:
        domain = icentpert(jnp.zeros(d, dtype=coeffs.dtype), jnp.ones(d, dtype=coeffs.dtype))

    # Compute per-variable order from exponents
    computed_order = tuple(int(jnp.max(exponents[i])) for i in range(d))
    return TaylorModel(coeffs, exponents, remainder, domain, center=center, _static_order=computed_order)


def taylor_model_constant(iv: Interval, domain: Interval, order: "int | tuple[int, ...]" = 1, center=None) -> TaylorModel:
    """Create a Taylor model from an interval (constant polynomial with remainder).

    The interval center becomes the constant polynomial term, and the
    interval perturbation becomes the remainder.

    Supports intervals with arbitrary shape (*output_shape,).

    Parameters
    ----------
    iv : Interval
        The interval to represent. Center becomes polynomial constant,
        perturbation becomes remainder.
    domain : Interval
        The domain box with shape (d,).
    order : int or tuple[int, ...]
        Polynomial degree of the Taylor Model (default: 1)
    center : ArrayLike, optional
        Expansion point. Default is domain center.

    Returns
    -------
    TaylorModel
        Constant Taylor model with the interval encoded as polynomial + remainder.
    """
    d = domain.lower.shape[0]
    output_shape = iv.lower.shape

    exponents = _get_canonical_exponents(d, order)
    num_monomials = exponents.shape[1]
    const_idx = jnp.argmin(jnp.sum(exponents, axis=0))

    coeffs = jnp.zeros((*output_shape, num_monomials)).at[..., const_idx].set(iv.center)
    return TaylorModel(coeffs, exponents, iv - iv.center, domain, center=center, _static_order=order)


def taylor_model_identity(domain, order=1) -> TaylorModel:
    """Create a Taylor model representing the identity function over a domain.

    Supports domain as a single Interval or any pytree structure of Intervals
    (list, dict, nested structures). When domain is a pytree, order can also
    be specified as a matching pytree to give per-leaf total degree bounds.

    Parameters
    ----------
    domain : Interval or PyTree[Interval]
        The domain. Can be:
        - Single Interval with shape (*output_shape,)
        - List of Intervals: [interval_t, interval_x]
        - Dict of Intervals: {'t': interval_t, 'x': interval_x}
        - Any nested pytree with Interval leaves
    order : int or PyTree[int]
        Polynomial degree bounds. Can be:
        - int: Same order for all leaves (default: 1)
        - PyTree matching domain structure: Per-leaf total degree bounds

    Returns
    -------
    TaylorModel
        Taylor model for identity function. Output shape is (total_dim,) where
        total_dim is the sum of flattened sizes of all leaf Intervals.

    Examples
    --------
    >>> # Single interval (backward compatible)
    >>> tm = taylor_model_identity(interval(x), order=2)

    >>> # List of intervals with per-leaf orders
    >>> tm = taylor_model_identity([interval_t, interval_x], order=[2, 3])

    >>> # Dict of intervals
    >>> tm = taylor_model_identity({'t': interval_t, 'x': interval_x}, order={'t': 2, 'x': 3})
    """
    # Check if domain is a single Interval (leaf case)
    if _is_interval_leaf(domain):
        # Simple case: single Interval domain (backward compatible path)
        iv = domain
        output_shape = iv.center.shape
        scalar_output = len(output_shape) == 0

        # Flatten to get domain dimension (scalars become 1D)
        center_flat = iv.center.reshape(-1) if not scalar_output else iv.center[None]
        radius_flat = iv.pert.reshape(-1) if not scalar_output else iv.pert[None]
        n = center_flat.shape[0]

        exponents = _generate_exponents(n, order)
        num_monomials = exponents.shape[1]

        # Identify constant (all-zero exponent) and linear (unit vector) columns
        is_constant = jnp.sum(exponents, axis=0) == 0  # (m,)
        eye_n = jnp.eye(n, dtype=jnp.int32)  # (n, n)
        is_linear = jnp.all(exponents[:, None, :] == eye_n[:, :, None], axis=0)  # (n, m)

        # coeffs[i, j] = center[i] if j is constant, 1.0 if j is linear for variable i
        coeffs_flat = (
            jnp.where(is_constant[None, :], center_flat[:, None], 0.0)
            + jnp.where(is_linear, 1.0, 0.0)
        )  # (n, m)

        if scalar_output:
            coeffs = coeffs_flat[0]  # (m,)
            remainder = interval(jnp.zeros((), dtype=center_flat.dtype))
        else:
            coeffs = coeffs_flat.reshape(*output_shape, num_monomials)
            remainder = interval(jnp.zeros(output_shape, dtype=center_flat.dtype))

        flat_domain = icentpert(center_flat, radius_flat)
        return TaylorModel(coeffs, exponents, remainder, flat_domain,
                           center=center_flat, _static_order=order)

    # PyTree case: domain is a structure of Intervals
    treedef, leaf_shapes, flat_domain = _get_domain_metadata(domain)
    per_leaf_order, per_var_order = _normalize_order_pytree(order, treedef, leaf_shapes)

    center_flat = flat_domain.center
    total_dim = center_flat.shape[0]

    # Generate exponents with per-leaf total degree bounds
    exponents = _get_leaf_total_degree_exponents(leaf_shapes, per_leaf_order)
    num_monomials = exponents.shape[1]

    # Build identity coefficients:
    # - Constant term: center_flat[i] for each output i
    # - Linear term: 1.0 for variable i in output i

    # Identify constant monomial (all zeros)
    is_constant = jnp.sum(exponents, axis=0) == 0  # (m,)

    # Identify linear monomials (unit vectors)
    eye_n = jnp.eye(total_dim, dtype=jnp.int32)  # (total_dim, total_dim)
    is_linear = jnp.all(exponents[:, None, :] == eye_n[:, :, None], axis=0)  # (total_dim, m)

    # Build coefficients: coeffs[i, j] = center[i] if constant, 1.0 if linear for var i
    coeffs = (
        jnp.where(is_constant[None, :], center_flat[:, None], 0.0)
        + jnp.where(is_linear, 1.0, 0.0)
    )  # (total_dim, m)

    # Remainder is zero since this is exact
    remainder = interval(jnp.zeros(total_dim, dtype=center_flat.dtype))

    return TaylorModel(
        coeffs,
        exponents,
        remainder,
        flat_domain,
        center=center_flat,
        _static_order=per_var_order,
        _domain_treedef=treedef,
        _leaf_shapes=leaf_shapes,
        _per_leaf_order=per_leaf_order,
    )


def taylor_model_multiarg_identity(
    *arg_domains: Interval,
    per_arg_order: tuple[int, ...],
) -> TaylorModel:
    """Create identity Taylor model for multiple arguments with per-argument total degree bounds.

    .. deprecated::
        Use ``taylor_model_identity(list(arg_domains), order=list(per_arg_order))`` instead.

    For a function f(*args), this creates a Taylor model where each argument
    can have arbitrary shape, and the polynomial order is specified as
    per-argument total degree bounds (not per-variable bounds).

    Parameters
    ----------
    *arg_domains : Interval
        Domain intervals for each argument. Each can have arbitrary shape.
    per_arg_order : tuple[int, ...]
        Maximum total degree for each argument. Length must match number of
        arg_domains. For example, (2, 3) means degree ≤ 2 in first argument
        and degree ≤ 3 in second argument.

    Returns
    -------
    TaylorModel
        Taylor model for identity with:
        - Flattened domain containing all argument variables
        - Exponents generated with per-argument total degree bounds
        - Output shape matching the concatenated flattened arguments

    Example
    -------
    For f(t, x) where t: () (scalar) and x: (2,):
    - per_arg_order = (2, 3) means |alpha_t| <= 2 and |alpha_x| <= 3
    - Generates monomials t^a * x1^b1 * x2^b2 where a <= 2 and b1 + b2 <= 3
    - Number of monomials: (2+1) * C(2+3, 3) = 3 * 10 = 30
    """
    if len(arg_domains) != len(per_arg_order):
        raise ValueError(
            f"Number of arg_domains ({len(arg_domains)}) must match "
            f"length of per_arg_order ({len(per_arg_order)})"
        )

    # Delegate to unified taylor_model_identity
    return taylor_model_identity(list(arg_domains), order=list(per_arg_order))


def taylor_model_from_function(
    f: Callable,
    domain: Interval,
    max_order: "int | tuple[int, ...]",
    n_out: int | None = None,
) -> TaylorModel:
    """Create a Taylor model by Taylor expansion of a function.

    Uses automatic differentiation to compute Taylor coefficients.

    Parameters
    ----------
    f : Callable
        Function to approximate, f: R^d -> R^n
    domain : Interval
        Domain box with shape (d,)
    max_order : int
        Maximum Taylor expansion order
    n_out : int, optional
        Output dimension. Inferred from f if not provided.

    Returns
    -------
    TaylorModel
        Taylor model approximation with remainder bounds
    """
    expansion_center = domain.center
    domain_radius = domain.pert
    d = expansion_center.shape[0]

    # Evaluate at center to get output dimension
    f_center = f(expansion_center)
    if f_center.ndim == 0:
        f_center = f_center[None]
        f = lambda x, _f=f: jnp.atleast_1d(_f(x))
    n = f_center.shape[0] if n_out is None else n_out

    # Generate exponents
    exponents = _generate_exponents(d, max_order)
    num_monomials = exponents.shape[1]

    # Compute Taylor coefficients using recursive jet
    # This gives us the full derivative tensor up to max_order
    # tensor[k1, k2, ..., kd] corresponds to coeff for x1^k1 * ... * xd^kd

    if max_order == 0:
        return TaylorModel(
            f(expansion_center).reshape(-1, 1),
            jnp.zeros((d, 1), dtype=jnp.int32),
            icentpert(jnp.zeros(n), jnp.zeros(n)),
            domain,
            center=expansion_center,
        )

    # Compute higher order derivatives using jacfwd loop
    # This computes the full dense derivative tensor at each order
    # T_0 = f(x)
    # T_1 = jacfwd(f)(x)
    # T_2 = jacfwd(jacfwd(f))(x)
    # ...

    # Compute higher order derivatives using jacfwd loop
    # We go up to max_order + 1 to compute the Lagrange remainder term

    deriv_tensors = []

    # Order 0
    val = f(expansion_center)
    if val.ndim == 0:
        val = val[None] # (1,) if scalar
    deriv_tensors.append(val)

    # Ensure we strictly differentiate a vector-valued function
    # to maintain consistent tensor shape (n, d, d...)
    if f(expansion_center).ndim == 0:
        def f_vec(x):
            v = f(x)
            return v[None]
        curr_f = f_vec
    else:
        curr_f = f

    # Compute up to max_order + 1
    for k in range(1, max_order + 2):
            curr_f = jax.jacfwd(curr_f)
            tensor = curr_f(expansion_center)
            deriv_tensors.append(tensor)

    coeffs = jnp.zeros((n, num_monomials), dtype=f_center.dtype)

    for i in range(num_monomials):
        exp = exponents[:, i] # (d,)
        order = jnp.sum(exp)

        # Only compute coeffs up to max_order
        if order > max_order:
             continue

        if order == 0:
            c = deriv_tensors[0]
        else:
            idx_list = []
            for var_idx in range(d):
                count = exp[var_idx]
                idx_list.extend([var_idx] * int(count))

            tensor = deriv_tensors[int(order)]
            full_idx = (slice(None), *idx_list)
            c = tensor[full_idx] # (n,)

        fact_prod = jnp.prod(jax.scipy.special.gamma(exp + 1))

        # Coefficients in raw (x - center) coordinates: D^alpha f(c) / alpha!
        scale = 1.0 / fact_prod

        coeffs = coeffs.at[:, i].set(c * scale)

    # Estimate remainder using Lagrange remainder bound:
    # |R_n(x)| <= (1/(n+1)!) * sup |D^{n+1}f(xi)| * |x-c|^{n+1}
    # We approximate sup |D^{n+1}f(xi)| with |D^{n+1}f(c)| (centered evaluation)
    # Ideally this should be evaluated over the interval domain.

    next_order = max_order + 1
    # Tensor of shape (n, d, d, ..., d) (k+1 'd's)
    D_next = deriv_tensors[next_order]

    # Take absolute value for bounding
    abs_D_next = jnp.abs(D_next) # (n, d, ..., d)

    # Contract with radius vector r (d,) repeatedly (next_order times)
    # We can use a loop or reshape tricks.
    # We want sum_{j1...j_{k+1}} |T_{...}| * r_{j1} * ... * r_{j_{k+1}}

    current_bound = abs_D_next
    for _ in range(next_order):
        # Contract last dimension with domain_radius
        # current is (n, ..., d)
        current_bound = jnp.dot(current_bound, domain_radius)
        # jnp.dot sums product over last axis of a and first of b (b is 1D)
        # result reduces rank by 1.

    # current_bound is now (n,)

    # Divide by (n+1)!
    factorial = jax.scipy.special.gamma(next_order + 1)
    remainder_bound = current_bound / factorial

    # Ensure non-zero for safety if needed, though 0 is valid for exact polynomials
    remainder = icentpert(jnp.zeros(n, dtype=f_center.dtype), remainder_bound)

    return TaylorModel(coeffs, exponents, remainder, domain, center=expansion_center, _static_order=max_order)


def evaluate_at_variable(tm: TaylorModel, var_idx: int, value: float) -> TaylorModel:
    """Partially evaluate a TaylorModel at a fixed value for one domain variable.

    Given a TM with domain (z_0, ..., z_{d-1}), substitutes z_{var_idx} = value
    and returns a TM with domain dimension d-1.

    Monomials that share the same reduced exponent (after removing var_idx) are
    collected and their contributions summed.

    Parameters
    ----------
    tm : TaylorModel
        Input Taylor model with domain dimension d >= 2.
    var_idx : int
        Index of the domain variable to evaluate (0-indexed).
    value : float
        Value at which to fix the variable.

    Returns
    -------
    TaylorModel
        Taylor model with domain dimension d-1.
    """
    d = tm.d
    if d < 2:
        raise ValueError("Cannot reduce domain dimension below 1")

    # Compute (value - center[var_idx]) for the target variable
    v_shifted = value - tm.center[var_idx]

    # Compute the scalar factor for each monomial: v_shifted^{exp[var_idx]}
    var_exps = tm.exponents[var_idx, :]  # (m,)
    var_factors = v_shifted ** var_exps  # (m,)

    # Multiply coefficients by the variable factors
    # coeffs: (*output_shape, m), var_factors: (m,)
    scaled_coeffs = tm.coeffs * var_factors  # (*output_shape, m)

    # Remove the variable from exponents → reduced exponents (d-1, m)
    keep = jnp.concatenate([
        jnp.arange(var_idx),
        jnp.arange(var_idx + 1, d)
    ])
    reduced_exps = tm.exponents[keep, :]  # (d-1, m)

    # Group monomials with identical reduced exponents by hashing
    new_d = d - 1
    # Remove the evaluated variable's order from the per-variable tuple
    max_order = tuple(o for i, o in enumerate(tm._static_order) if i != var_idx)
    canonical_exp = _get_canonical_exponents(new_d, max_order)
    num_canonical = canonical_exp.shape[1]

    # Hash for grouping
    base = max(max_order) + 2 if max_order else 2
    powers = base ** jnp.arange(new_d)
    reduced_hash = jnp.sum(reduced_exps * powers[:, None], axis=0)  # (m,)
    canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)  # (mc,)

    # For each canonical monomial, sum contributions from matching reduced monomials
    # match[i, j] = True if reduced monomial j maps to canonical monomial i
    match = (reduced_hash[None, :] == canonical_hash[:, None])  # (mc, m)

    # Sum scaled coefficients for matching monomials
    # scaled_coeffs: (*output_shape, m), match: (mc, m)
    new_coeffs = jnp.einsum('...j,ij->...i', scaled_coeffs, match.astype(scaled_coeffs.dtype))

    # New domain: remove var_idx
    new_lower = jnp.concatenate([
        tm.domain.lower[:var_idx],
        tm.domain.lower[var_idx + 1:]
    ])
    new_upper = jnp.concatenate([
        tm.domain.upper[:var_idx],
        tm.domain.upper[var_idx + 1:]
    ])
    new_domain = interval(new_lower, new_upper)

    # New center: remove var_idx
    new_center = jnp.concatenate([
        tm.center[:var_idx],
        tm.center[var_idx + 1:]
    ])

    return TaylorModel(
        new_coeffs, canonical_exp, tm.remainder,
        new_domain, center=new_center, _static_order=max_order
    )


def integrate_variable(
    tm: TaylorModel,
    var_idx: int,
    start: float | None = None,
    keep_order: bool = True,
) -> TaylorModel:
    r"""Integrate a TaylorModel with respect to one domain variable.

    Computes :math:`\int_{a}^{x_i} \text{tm}(x) \, dx_i` where *a* = ``start``
    and *i* = ``var_idx``.  The result is a TaylorModel in the same domain
    variables (with the order of variable *i* incremented by 1).

    The operation decomposes into five parts:

    1. **Coefficient shift** – each coefficient :math:`a_\alpha` becomes
       :math:`a_\alpha / (\alpha_i + 1)` with exponent :math:`\alpha_i`
       incremented by 1.
    2. **Starting-point subtraction** – the shifted polynomial evaluated at
       :math:`x_i = a` is subtracted (these are terms with :math:`\alpha_i = 0`
       in the result).
    3. **Order reduction** – terms at the new highest degree
       (:math:`\alpha_i = k_i + 1`) are bounded over the domain and absorbed
       into the remainder, keeping the polynomial at the original order.
    4. **Remainder integration** – the original remainder interval is multiplied
       by the interval :math:`[x_i - a]` over the domain and added.
    5. **Order reduction** - if keep_order is True, the highest-degree terms
       produced by integration are bounded over the domain and absorbed
       into the remainder, keeping the polynomial at the original order.

    Parameters
    ----------
    tm : TaylorModel
        Input Taylor model.
    var_idx : int
        Index of the domain variable to integrate over (0-indexed).
    start : float or None
        Lower limit of integration (the upper limit is the variable itself).
        If None, defaults to ``tm.center[var_idx]``.
    keep_order : bool, optional
        If True (default), the highest-degree terms produced by integration
        are bounded and absorbed into the remainder so the output has the
        same polynomial order as the input.  If False, the output order for
        variable ``var_idx`` is incremented by 1.

    Returns
    -------
    TaylorModel
        Integrated Taylor model.
    """
    d = tm.d
    if var_idx < 0 or var_idx >= d:
        raise ValueError(f"var_idx={var_idx} out of range for d={d}")

    if start is None:
        start = tm.center[var_idx]

    # --- 1. Shift coefficients ---
    # Each monomial (x_i - c_i)^{e_i} integrates to (x_i - c_i)^{e_i+1} / (e_i+1)
    var_exps = tm.exponents[var_idx, :]  # (m,)
    divisors = (var_exps + 1).astype(tm.coeffs.dtype)  # (m,)
    shifted_coeffs = tm.coeffs / divisors  # (*output_shape, m)

    # New exponents: increment var_idx row by 1
    shifted_exponents = tm.exponents.at[var_idx, :].add(1)

    # --- 2. Starting-point evaluation ---
    # The antiderivative evaluated at x_i = start gives a constant (in x_i)
    # contribution: a_α / (e_i+1) * (start - c_i)^{e_i+1} * prod_{j≠i} (x_j-c_j)^{e_j}
    # These terms have the original exponents in all variables except var_idx,
    # where the exponent is 0.
    a_shifted = start - tm.center[var_idx]
    start_powers = a_shifted ** (var_exps + 1).astype(tm.coeffs.dtype)  # (m,)
    start_coeffs = shifted_coeffs * start_powers  # (*output_shape, m)

    # The starting-point terms have exponent 0 for var_idx (constant in x_i)
    start_exponents = tm.exponents.copy()  # keep other exponents, set var_idx to 0
    # (these are the original exponents since var_idx exponent maps to the
    #  "other variables" part of each monomial)

    # --- 3. Combine: shifted part - starting point part ---
    # Merge into canonical exponent structure for the new order
    new_order = list(tm._static_order)
    new_order[var_idx] += 1
    new_order = tuple(new_order)

    canonical_exp = _get_canonical_exponents(d, new_order)
    num_canonical = canonical_exp.shape[1]

    # Hash-based scatter for both shifted and start terms
    base = max(new_order) + 2
    powers_hash = base ** jnp.arange(d)

    canonical_hash = jnp.sum(canonical_exp * powers_hash[:, None], axis=0)  # (mc,)
    shifted_hash = jnp.sum(shifted_exponents * powers_hash[:, None], axis=0)  # (m,)
    start_hash = jnp.sum(start_exponents * powers_hash[:, None], axis=0)  # (m,)

    # Scatter matrices: (m, mc)
    shifted_scatter = (shifted_hash[:, None] == canonical_hash[None, :]).astype(tm.coeffs.dtype)
    start_scatter = (start_hash[:, None] == canonical_hash[None, :]).astype(tm.coeffs.dtype)

    # New coeffs = shifted_coeffs @ shifted_scatter - start_coeffs @ start_scatter
    new_coeffs = shifted_coeffs @ shifted_scatter - start_coeffs @ start_scatter

    # --- 4. Remainder from integrating the original remainder ---
    # ∫_a^{x_i} r dx_i ∈ remainder * (x_i - a)
    xi_minus_a = tm.domain[var_idx] - start
    integ_remainder = tm.remainder * xi_minus_a

    # --- 5. Optionally reduce order back to original ---
    if keep_order:
        orig_order_i = tm._static_order[var_idx]
        high_mask = canonical_exp[var_idx, :] > orig_order_i  # (mc,)

        # Bound high-order monomials over shifted domain
        mono_bounds = _bound_monomials_over_domain(
            canonical_exp, tm.shifted_domain, max(new_order)
        )

        high_coeffs = jnp.where(high_mask, new_coeffs, 0.0)
        zeros = jnp.zeros_like(high_coeffs)
        c_pos = jnp.maximum(high_coeffs, zeros)
        c_neg = jnp.minimum(high_coeffs, zeros)

        trunc_lower = jnp.sum(c_pos * mono_bounds.lower + c_neg * mono_bounds.upper, axis=-1)
        trunc_upper = jnp.sum(c_pos * mono_bounds.upper + c_neg * mono_bounds.lower, axis=-1)
        trunc_remainder = interval(trunc_lower, trunc_upper)

        kept_coeffs = jnp.where(high_mask, 0.0, new_coeffs)
        total_remainder = trunc_remainder + integ_remainder

        return TaylorModel(
            kept_coeffs, canonical_exp, total_remainder, tm.domain,
            center=tm.center, _static_order=tm._static_order,
        )
    else:
        return TaylorModel(
            new_coeffs, canonical_exp, integ_remainder, tm.domain,
            center=tm.center, _static_order=new_order,
        )


def taylor_model_concatenate(tms: list["TaylorModel"], axis: int = 0) -> "TaylorModel":
    """Concatenate multiple TaylorModels along an output dimension.

    This is the equivalent of jnp.concatenate for TaylorModels.
    All input TaylorModels must have the same domain and exponent structure.

    For JIT compatibility, this function requires all TMs to have identical
    exponent arrays. Use taylor_model_unify_exponents first if needed.

    Parameters
    ----------
    tms : list of TaylorModel
        TaylorModels to concatenate.
    axis : int
        The axis in the output shape along which to concatenate (default: 0).
        Does not affect the monomial axis (which is always last in coeffs).

    Returns
    -------
    TaylorModel
        Concatenated TaylorModel.
    """
    if len(tms) == 0:
        raise ValueError("Cannot concatenate empty list of TaylorModels")

    if len(tms) == 1:
        return tms[0]

    # Use first TM as reference
    ref = tms[0]
    domain = ref.domain
    exponents = ref.exponents
    # Element-wise max of per-variable orders
    from functools import reduce
    static_order = reduce(_max_order, (tm._static_order for tm in tms))

    # Stack coefficients along specified axis of output shape
    # coeffs have shape (*output_shape, m), so axis applies to output_shape
    coeffs = jnp.concatenate([tm.coeffs for tm in tms], axis=axis)

    # Stack remainders along same axis
    remainder = interval(
        jnp.concatenate([tm.remainder.lower for tm in tms], axis=axis),
        jnp.concatenate([tm.remainder.upper for tm in tms], axis=axis)
    )

    return TaylorModel(coeffs, exponents, remainder, domain, center=ref.center, _static_order=static_order)
