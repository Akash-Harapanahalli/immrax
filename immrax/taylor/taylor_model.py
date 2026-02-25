"""Taylor Model implementation for reachability analysis.

Taylor models represent sets as polynomial approximations with rigorous
interval remainder bounds, providing a powerful tool for propagating
uncertainty through nonlinear functions.
"""

from typing import Callable, Tuple
import math

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, PyTree, PyTreeDef

import jax.tree_util

from immrax.inclusion import Interval, interval, icentpert

from immrax.taylor.base import (
    _is_interval_or_array_leaf,
    _pytree_to_flattened_array,
    _unflatten_array_to_pytree,
    PyTreeShape,
    _normalize_order_pytree,
    _leaf_slice,
    _max_order,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _compute_per_leaf_order_from_exponents,
    _merge_taylor_terms,
    _compact_taylor_terms,
    _generate_exponents,
)


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
    _input_pytree: PyTreeShape
    _output_pytree: PyTreeShape
    _per_leaf_order: tuple[int, ...]

    def __init__(
        self,
        coeffs,
        exponents,
        remainder: Interval,
        flat_domain: Interval,
        flat_center,
        *,
        _input_pytree: "PyTreeShape | None" = None,
        _output_pytree: "PyTreeShape | None" = None,
        _per_leaf_order: "tuple[int, ...] | None" = None,
    ) -> None:
        """Low-level constructor — converts arrays and assigns attributes.

        No shape validation is performed; callers are trusted to provide
        compatible shapes.  Use the high-level constructor
        :func:`taylor_model` for user-facing construction with full input
        validation.
        """
        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.remainder = remainder
        self.flat_domain = flat_domain
        self.flat_center = jnp.asarray(flat_center) if flat_center is not None else None
        self._input_pytree = _input_pytree
        self._output_pytree = _output_pytree
        self._per_leaf_order = _per_leaf_order

    @property
    def _domain_treedef(self):
        """PyTree structure of the domain (convenience accessor)."""
        return self._input_pytree.treedef

    @property
    def _leaf_shapes(self):
        """Shapes of each leaf Interval in the domain pytree (convenience accessor)."""
        return self._input_pytree.leaf_shapes

    @property
    def domain(self):
        """Structured domain (PyTree of Intervals)."""
        return self._input_pytree.unflatten(self.flat_domain)

    @property
    def center(self):
        """Structured center (PyTree of Arrays)."""
        return self._input_pytree.unflatten(self.flat_center)

    # --- Pytree methods ---

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, Array, Interval, Interval, Array], dict]:
        return (
            (
                self.coeffs,
                self.exponents,
                self.remainder,
                self.flat_domain,
                self.flat_center,
            ),
            {
                "_input_pytree": self._input_pytree,
                "_output_pytree": self._output_pytree,
                "_per_leaf_order": self._per_leaf_order,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorModel":
        # Bypass __init__ to avoid jnp.asarray — JAX/equinox may pass
        # non-array sentinels (e.g. booleans) through tree operations.
        coeffs, exponents, remainder, flat_domain, flat_center = children
        obj = object.__new__(cls)
        obj.coeffs = coeffs
        obj.exponents = exponents
        obj.remainder = remainder
        obj.flat_domain = flat_domain
        obj.flat_center = flat_center
        obj._input_pytree = aux_data.get("_input_pytree") if aux_data else None
        obj._output_pytree = aux_data.get("_output_pytree") if aux_data else None
        obj._per_leaf_order = aux_data.get("_per_leaf_order") if aux_data else None
        return obj

    # --- Properties ---

    @property
    def _output_shape(self) -> Tuple[int, ...]:
        """Output shape, derived from coeffs. Last axis is monomial axis."""
        return self.coeffs.shape[:-1]

    @property
    def n(self) -> int:
        """Total number of output elements (product of output_shape)."""
        import math

        return math.prod(self._output_shape) if self._output_shape else 1

    @property
    def shifted_domain(self) -> Interval:
        """Domain shifted by center: D - center, for bounding monomials (x - center)^alpha."""
        return self.flat_domain - self.flat_center

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
        """Per-leaf total degree bounds."""
        return self._per_leaf_order

    @property
    def max_order(self) -> int:
        """Maximum order across all leaves."""
        return max(self._per_leaf_order)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape."""
        return self._output_shape

    @property
    def domain_treedef(self) -> "jax.tree_util.PyTreeDef | None":
        """PyTree structure of the original domain, or None for single Interval."""
        return self._input_pytree.treedef

    @property
    def leaf_shapes(self) -> "tuple[tuple[int, ...], ...] | None":
        """Shapes of each leaf Interval in the domain pytree, or None."""
        return self._input_pytree.leaf_shapes

    @property
    def per_leaf_order(self) -> "tuple[int, ...]":
        """Per-leaf total degree bounds."""
        return self._per_leaf_order

    @property
    def structured_center(self):
        """Get the center as a structured pytree matching the output pytree structure.

        If output has multiple leaves, uses _output_pytree to unflatten.
        Otherwise falls back to _input_pytree (backward-compatible for identity TMs).
        """
        if self._output_pytree.num_leaves > 1:
            return self._output_pytree.unflatten(self.flat_center)
        return self._input_pytree.unflatten(self.flat_center)

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

        return TaylorModel(
            c,
            self.exponents,
            rem,
            self.flat_domain,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=PyTreeShape.flat(c.shape[:-1]),
            _per_leaf_order=self._per_leaf_order,
        )

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
        return jnp.sum(jnp.where(is_constant, self.coeffs, 0.0), axis=-1)

    # --- Polynomial evaluation ---

    @property
    def polynomial(self) -> "TaylorPolynomial":
        from immrax.taylor.taylor_polynomial import TaylorPolynomial

        return TaylorPolynomial(
            self.coeffs,
            self.exponents,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=self._output_pytree,
            _per_leaf_order=self._per_leaf_order,
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
        x_centered = x - self.flat_center

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
    # Arithmetic operations are implemented as standalone functions in pjetm.py
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

    def multiply(
        self, other: "TaylorModel", max_order: int | None = None
    ) -> "TaylorModel": ...

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
        mono_bounds = _bound_monomials_over_domain(
            self.exponents, self.shifted_domain, self.max_order
        )
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

        Uses per-leaf total degree bounds.

        Parameters
        ----------
        target_order : int, tuple[int, ...], or None
            Target per-leaf order. If int, broadcast to all leaves.
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
        leaf_shapes = self._leaf_shapes
        num_leaves = len(leaf_shapes)

        # Determine per_leaf_order: use target_order only if it has the correct
        # length (number of leaves), otherwise use stored _per_leaf_order.
        if target_order is not None:
            if isinstance(target_order, int):
                per_leaf_order = tuple([target_order] * num_leaves)
            elif len(target_order) == num_leaves:
                per_leaf_order = tuple(target_order)
            else:
                # target_order has wrong length, use stored per_leaf_order instead
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
        base = self.max_order + max(per_leaf_order) + 2
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
            scatter_matrix = jax.nn.one_hot(
                canonical_indices, num_canonical, dtype=self.dtype
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'broadcast' or 'searchsorted'."
            )

        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)
        new_coeffs = masked_coeffs @ scatter_matrix

        # Bound terms above target order
        absorb_mask = ~has_match
        mono_bounds = _bound_monomials_over_domain(
            self.exponents, self.shifted_domain, self.max_order
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

        return TaylorModel(
            new_coeffs,
            canonical_exp,
            new_remainder,
            self.flat_domain,
            self.flat_center,
            _input_pytree=PyTreeShape(self._domain_treedef, leaf_shapes),
            _output_pytree=self._output_pytree,
            _per_leaf_order=per_leaf_order,
        )

    def reduce_order(self, target_order: "int | tuple[int, ...]") -> "TaylorModel":
        """Reduce polynomial order, absorbing high-order terms into remainder.

        Parameters
        ----------
        target_order : int or tuple[int, ...]
            Target maximum polynomial order (per-leaf).

        Returns
        -------
        TaylorModel
            Reduced order Taylor model (overapproximation)
        """
        num_leaves = len(self._leaf_shapes)
        if isinstance(target_order, int):
            target_order = tuple([target_order] * num_leaves)
        else:
            target_order = tuple(target_order)

        # Separate terms to keep and terms to absorb using per-leaf bounds
        keep_mask = _check_per_leaf_bounds(
            self.exponents, self._leaf_shapes, target_order
        )
        absorb_mask = ~keep_mask  # (m,)

        # Compute monomial bounds over shifted domain (D - center)
        mono_bounds_iv = _bound_monomials_over_domain(
            self.exponents, self.shifted_domain, self.max_order
        )
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
            self.flat_domain,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=self._output_pytree,
            _per_leaf_order=target_order,
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
            f"remainder={self.remainder!r}, flat_domain={self.flat_domain!r}, "
            f"flat_center={self.flat_center!r})"
        )


# --- Helper functions ---


def _bound_monomials_over_domain(
    exponents: Array, norm_domain: Interval, max_order: int | None = None
) -> Interval:
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




def taylor_model(
    coeffs: ArrayLike,
    exponents: ArrayLike,
    remainder: Interval | None = None,
    domain: "Interval | PyTree[Interval] | None" = None,
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
    domain : Interval or PyTree[Interval], optional
        Domain box. Can be a single Interval or a PyTree of Intervals.
        Default is [-1, 1]^d.
        If a PyTree is provided, it is flattened to a single domain Interval,
        and the structure is stored in the TaylorModel.
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
        # Default domain is [-1, 1]^d (single interval)
        flat_domain = icentpert(
            jnp.zeros(d, dtype=coeffs.dtype), jnp.ones(d, dtype=coeffs.dtype)
        )
        # Assume single interval structure
        _domain_treedef, _leaf_shapes, _ = _pytree_to_flattened_array(flat_domain)
    else:
        # Domain provided - flatten it and get metadata
        _domain_treedef, _leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)

    _per_leaf_order = _compute_per_leaf_order_from_exponents(exponents, _leaf_shapes)

    if center is not None:
        # Flatten center if provided, assuming it matches domain structure
        # If domain was None (default single interval), center should be flat (Array)
        if domain is None:
            flat_center = center
        else:
            _, _, flat_center = _pytree_to_flattened_array(center)
    else:
        flat_center = None

    # Validate shapes
    if coeffs.ndim < 1:
        raise ValueError(f"coeffs must be at least 1D, got shape {coeffs.shape}")
    if exponents.ndim != 2:
        raise ValueError(f"exponents must be 2D, got shape {exponents.shape}")
    if coeffs.shape[-1] != exponents.shape[1]:
        raise ValueError(
            f"coeffs and exponents must have same number of monomials: "
            f"{coeffs.shape[-1]} vs {exponents.shape[1]}"
        )
    if remainder.shape != output_shape:
        raise ValueError(
            f"remainder and coeffs must have same output shape: "
            f"{remainder.shape} vs {output_shape}"
        )
    if flat_domain.lower.shape[0] != exponents.shape[0]:
        raise ValueError(
            f"domain must match exponents dimension: "
            f"{flat_domain.lower.shape[0]} vs {exponents.shape[0]}"
        )

    return TaylorModel(
        coeffs,
        exponents,
        remainder,
        flat_domain,
        flat_center,
        _input_pytree=PyTreeShape(_domain_treedef, _leaf_shapes),
        _output_pytree=PyTreeShape.flat(coeffs.shape[:-1]),
        _per_leaf_order=_per_leaf_order,
    )


def _taylor_model_constant_impl(
    iv: Interval,
    domain: Interval,
    order: "int | tuple[int, ...]" = 1,
    center=None,
    _domain_treedef: "PyTreeDef | None" = None,
    _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
    _per_leaf_order: "tuple[int, ...] | None" = None,
    _input_pytree: "PyTreeShape | None" = None,
    _output_pytree: "PyTreeShape | None" = None,
) -> TaylorModel:
    """Internal implementation for taylor_model_constant."""
    iv = interval(iv)

    output_shape = iv.lower.shape
    flat_domain = domain

    # Resolve _input_pytree
    if _input_pytree is None:
        if _leaf_shapes is None:
            _leaf_shapes = (flat_domain.shape,)
        if _domain_treedef is None:
            _domain_treedef = jax.tree_util.tree_structure(jnp.zeros(flat_domain.shape))
        _input_pytree = PyTreeShape(_domain_treedef, _leaf_shapes)
    else:
        _leaf_shapes = _input_pytree.leaf_shapes

    num_leaves = len(_leaf_shapes)

    # Set default per_leaf_order if not provided
    if _per_leaf_order is None:
        if isinstance(order, int):
            _per_leaf_order = tuple([order] * num_leaves)
        else:
            _per_leaf_order = tuple(order)

    # Generate exponents with per-leaf total degree bounds
    exponents = _get_leaf_total_degree_exponents(_leaf_shapes, _per_leaf_order)

    # Coeffs: constant term is iv.center
    coeffs = jnp.zeros((*output_shape, exponents.shape[1]), dtype=iv.lower.dtype)
    coeffs = coeffs.at[..., 0].set(iv.center)

    if _output_pytree is None:
        _output_pytree = PyTreeShape.flat(output_shape)

    return TaylorModel(
        coeffs,
        exponents,
        iv - iv.center,
        flat_domain,
        center,
        _input_pytree=_input_pytree,
        _output_pytree=_output_pytree,
        _per_leaf_order=_per_leaf_order,
    )


def taylor_model_constant(
    iv: Interval,
    domain: "Interval | PyTree[Interval]",
    order: "int | tuple[int, ...]" = 1,
    center=None,
) -> TaylorModel:
    """Create a Taylor model from an interval (constant polynomial with remainder).

    The interval center becomes the constant polynomial term, and the
    interval perturbation becomes the remainder.

    Supports intervals with arbitrary shape (*output_shape,).

    Parameters
    ----------
    iv : Interval
        The interval to represent. Center becomes polynomial constant,
        perturbation becomes remainder.
    domain : Interval or PyTree[Interval]
        The domain. Can be a single Interval or a PyTree of Intervals.
    order : int or tuple[int, ...]
        Polynomial degree of the Taylor Model (default: 1)
    center : ArrayLike, optional
        Expansion point. Default is domain center.

    Returns
    -------
    TaylorModel
        Constant Taylor model with the interval encoded as polynomial + remainder.
    """
    # Process domain and metadata
    treedef, leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)

    # Infer per_leaf_order
    num_leaves = len(leaf_shapes)
    if isinstance(order, int):
        per_leaf_order = tuple([order] * num_leaves)
    else:
        per_leaf_order = tuple(order)

    if center is not None:
        _, _, flat_center = _pytree_to_flattened_array(center)
    else:
        flat_center = None

    return _taylor_model_constant_impl(
        iv,
        flat_domain,
        order,
        center=flat_center,
        _input_pytree=PyTreeShape(treedef, leaf_shapes),
        _per_leaf_order=per_leaf_order,
    )


def taylor_model_identity(
    domain: "Interval | PyTree[Interval]",
    center: "ArrayLike | PyTree[ArrayLike]" = None,
    order: "int | PyTree[int]" = 1,
) -> TaylorModel:
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
    center : ArrayLike or PyTree[ArrayLike], optional
        Expansion point. Default is domain center.
    order : int or PyTree[int], optional
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
    >>> # Single interval
    >>> tm = taylor_model_identity(interval(x), order=2)

    >>> # List of intervals with per-leaf orders
    >>> tm = taylor_model_identity([interval_t, interval_x], order=[2, 3])

    >>> # Dict of intervals
    >>> tm = taylor_model_identity({'t': interval_t, 'x': interval_x}, order={'t': 2, 'x': 3})
    """
    # domain is a structure of Intervals
    treedef, leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)
    per_leaf_order = _normalize_order_pytree(order, treedef, leaf_shapes)

    if center is None:
        center_flat = flat_domain.center
    else:
        # Make sure center is a pytree matching domain structure
        center_treedef, center_leaf_shapes, center_flat = _pytree_to_flattened_array(
            center
        )
        if center_treedef != treedef or center_leaf_shapes != leaf_shapes:
            raise ValueError("Center must have the same structure as domain")

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
    is_linear = jnp.all(
        exponents[:, None, :] == eye_n[:, :, None], axis=0
    )  # (total_dim, m)

    # Build coefficients: coeffs[i, j] = center[i] if constant, 1.0 if linear for var i
    coeffs = jnp.where(is_constant[None, :], center_flat[:, None], 0.0) + jnp.where(
        is_linear, 1.0, 0.0
    )  # (total_dim, m)

    # Remainder is zero since this is exact
    remainder = interval(jnp.zeros(total_dim, dtype=center_flat.dtype))

    input_pytree = PyTreeShape(treedef, leaf_shapes)
    # For identity, output matches domain structure
    output_pytree = PyTreeShape(treedef, leaf_shapes)

    return TaylorModel(
        coeffs,
        exponents,
        remainder,
        flat_domain,
        center_flat,
        _input_pytree=input_pytree,
        _output_pytree=output_pytree,
        _per_leaf_order=per_leaf_order,
    )


def taylor_model_from_function(
    f: Callable,
    domain: "Interval | PyTree[Interval]",
    max_order: "int | tuple[int, ...]",
    n_out: int | None = None,
) -> TaylorModel:
    """Create a Taylor model by Taylor expansion of a function.

    Uses automatic differentiation to compute Taylor coefficients.

    Parameters
    ----------
    f : Callable
        Function to approximate, f: R^d -> R^n
    domain : Interval or PyTree[Interval]
        Domain box. Can be a single Interval or a PyTree of Intervals.
    max_order : int or tuple[int, ...]
        Maximum Taylor expansion order. Can be int (total degree) or tuple (per-leaf).
    n_out : int, optional
        Output dimension. Inferred from f if not provided.

    Returns
    -------
    TaylorModel
        Taylor model approximation with remainder bounds
    """
    # Process domain and metadata
    treedef, leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)

    expansion_center = flat_domain.center
    domain_radius = flat_domain.pert
    d = expansion_center.shape[0]

    # Evaluate at center (unflattened) to get output dimension
    # Reconstruct structured center to pass to f
    structured_center = _unflatten_array_to_pytree(
        treedef, leaf_shapes, expansion_center
    )

    # We need f to accept flattened array input for derivatives?
    # No, jax.jacfwd works on pytrees, but here we are doing Taylor expansion in R^d (flat).
    # If f implies structure, we might need to wrap it.
    # TaylorModel logic works on flattened domain coordinates (x - center).
    # So we need a wrapper around f that takes flat input, unflattens it, calls f.

    def f_flat(x_flat):
        x_struct = _unflatten_array_to_pytree(treedef, leaf_shapes, x_flat)
        return f(x_struct)

    # Check if f works with flat input or structured input
    # The user might have passed a function expecting structured input if domain is structured.
    # But usually one passes a function defined on the domain variables.
    # Let's try calling f with structured center first.
    try:
        f_center = f(structured_center)
        # Use f_flat for derivatives
        work_f = f_flat
    except Exception:
        # Maybe f expects flat input even if domain is structured?
        # Or maybe passing structured_center failed for another reason.
        # Fallback: assume f expects flat array if domain effectively flat?
        # But for correctness with PyTree domain, we generally expect f to accept PyTree.
        # If f expects flat input, f(structured_center) where structured_center is PyTree might fail.
        # Let's assume f matches domain structure.
        raise

    if f_center.ndim == 0:
        f_center = f_center[None]

        def f_vec(x, _f=work_f):
            res = _f(x)
            return jnp.atleast_1d(res)

        work_f = f_vec

    n = f_center.shape[0] if n_out is None else n_out

    # Parse max_order
    # If it's a tuple matching leaves, it's per_leaf_order
    # If it's an int, broadcast
    if isinstance(max_order, int):
        per_leaf_order = tuple([max_order] * len(leaf_shapes))
        # max_order for generation is just int
        gen_max_order = max_order
    else:
        # It's a tuple. Is it per-variable max order (old style) or per-leaf order?
        # The new standard is per-leaf order.
        # If length matches leaves, assume per-leaf order.
        # If length matches d (variables), it might be per-variable (deprecated but maybe lingering).
        # We'll assume per-leaf if it matches leaf count.
        if len(max_order) == len(leaf_shapes):
            per_leaf_order = tuple(max_order)
            gen_max_order = per_leaf_order  # Use per-leaf for generation
        else:
            # Fallback or error?
            # Let's assume it's per-leaf order passed as tuple even if 1 leaf?
            # For now, trust the tuple is per-leaf.
            per_leaf_order = tuple(max_order)
            gen_max_order = per_leaf_order

    # Generate exponents
    # We use _get_leaf_total_degree_exponents instead of _generate_exponents for per-leaf bounds
    exponents = _get_leaf_total_degree_exponents(leaf_shapes, per_leaf_order)
    num_monomials = exponents.shape[1]

    # Compute Taylor coefficients using recursive jet
    # work_f takes flat input R^d -> R^n

    # We need the max order integer for the loop
    max_k = max(per_leaf_order)

    input_pytree = PyTreeShape(treedef, leaf_shapes)

    if max_k == 0:
        return TaylorModel(
            f_center.reshape(-1, 1),
            jnp.zeros((d, 1), dtype=jnp.int32),
            icentpert(jnp.zeros(n), jnp.zeros(n)),
            flat_domain,
            flat_center=expansion_center,
            _input_pytree=input_pytree,
            _output_pytree=PyTreeShape.flat(f_center.shape),
            _per_leaf_order=per_leaf_order,
        )

    # Compute higher order derivatives using jacfwd loop
    deriv_tensors = []

    # Order 0
    val = f_center
    if val.ndim == 0:
        val = val[None]
    deriv_tensors.append(val)

    curr_f = work_f

    # Compute up to max_order + 1
    for k in range(1, max_k + 2):
        curr_f = jax.jacfwd(curr_f)
        tensor = curr_f(expansion_center)
        deriv_tensors.append(tensor)

    coeffs = jnp.zeros((n, num_monomials), dtype=f_center.dtype)

    for i in range(num_monomials):
        exp = exponents[:, i]  # (d,)
        order = jnp.sum(exp)

        # Only compute coeffs up to per-term order
        # (Though we filtered exponents by per_leaf_order, so this check validates against max_k)
        if order > max_k:
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
            c = tensor[full_idx]  # (n,)

        fact_prod = jnp.prod(jax.scipy.special.gamma(exp + 1))
        scale = 1.0 / fact_prod
        coeffs = coeffs.at[:, i].set(c * scale)

    # Estimate remainder using Lagrange remainder bound
    # Using max_k + 1 as the next order
    next_order = max_k + 1

    # Note: This remainder estimation assumes uniform max order.
    # With per-leaf orders, different leaves might have different bounds.
    # But differentiation gives full tensor. We can bound everything at max_k + 1.
    # This might be conservative but correct.

    D_next = deriv_tensors[next_order]
    abs_D_next = jnp.abs(D_next)

    current_bound = abs_D_next
    for _ in range(next_order):
        current_bound = jnp.dot(current_bound, domain_radius)

    factorial = jax.scipy.special.gamma(next_order + 1)
    remainder_bound = current_bound / factorial

    remainder = icentpert(jnp.zeros(n, dtype=f_center.dtype), remainder_bound)

    return TaylorModel(
        coeffs,
        exponents,
        remainder,
        flat_domain,
        flat_center=expansion_center,
        _input_pytree=input_pytree,
        _output_pytree=PyTreeShape.flat(f_center.shape),
        _per_leaf_order=per_leaf_order,
    )


def _resolve_var_idx(
    var_idx: int | None,
    leaf_idx: int | None,
    leaf_var_idx: tuple[int, ...] | int | None,
    leaf_shapes: list[tuple[int, ...]],
) -> int:
    """Resolve a flat variable index from leaf indices."""
    if var_idx is not None:
        return var_idx

    if leaf_idx is None or leaf_var_idx is None:
        raise ValueError(
            "Either var_idx or both leaf_idx and leaf_var_idx must be provided"
        )

    if leaf_idx >= len(leaf_shapes):
        raise ValueError(
            f"leaf_idx={leaf_idx} out of range for {len(leaf_shapes)} leaves"
        )

    var_idx = 0
    for i, shape in enumerate(leaf_shapes[: leaf_idx - 1]):
        var_idx += math.prod(shape)

    if isinstance(leaf_var_idx, int):
        return var_idx + leaf_var_idx
    else:
        # Convert tuple to flat index wrt the leaf shape - does validation
        return var_idx + jnp.ravel_multi_index(leaf_var_idx, leaf_shapes[leaf_idx])


def tm_evaluate_at_variable(
    tm: TaylorModel,
    var_idx: int | None = None,
    value: float | None = None,
    *,
    leaf_idx: int | None = None,
    leaf_var_idx: int | None = None,
) -> TaylorModel:
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
    v_shifted = value - tm.flat_center[var_idx]

    # Compute the scalar factor for each monomial: v_shifted^{exp[var_idx]}
    var_exps = tm.exponents[var_idx, :]  # (m,)
    var_factors = v_shifted**var_exps  # (m,)

    # Multiply coefficients by the variable factors
    # coeffs: (*output_shape, m), var_factors: (m,)
    scaled_coeffs = tm.coeffs * var_factors  # (*output_shape, m)

    # Remove the variable from exponents → reduced exponents (d-1, m)
    keep = jnp.concatenate([jnp.arange(var_idx), jnp.arange(var_idx + 1, d)])
    reduced_exps = tm.exponents[keep, :]  # (d-1, m)

    # Group monomials with identical reduced exponents by hashing
    new_d = d - 1

    # Find which leaf contains var_idx and update leaf_shapes/per_leaf_order
    leaf_shapes = tm._leaf_shapes
    per_leaf_order = tm._per_leaf_order
    new_leaf_shapes = []
    new_per_leaf_order = []

    for li in range(len(leaf_shapes)):
        slc = _leaf_slice(leaf_shapes, li)
        leaf_size = slc.stop - slc.start
        if slc.start <= var_idx < slc.stop:
            # This leaf contains the removed variable
            if leaf_size == 1:
                # Remove entire leaf
                continue
            else:
                # Update leaf shape (remove one element from this leaf)
                old_shape = leaf_shapes[li]
                if old_shape == ():
                    # Scalar leaf being removed
                    continue
                else:
                    # Multi-element leaf - compute new shape (flatten and reduce by 1)
                    new_size = leaf_size - 1
                    new_leaf_shapes.append((new_size,))
                    new_per_leaf_order.append(per_leaf_order[li])
        else:
            new_leaf_shapes.append(leaf_shapes[li])
            new_per_leaf_order.append(per_leaf_order[li])

    new_leaf_shapes = tuple(new_leaf_shapes)
    new_per_leaf_order = tuple(new_per_leaf_order)

    # Generate canonical exponents for reduced domain
    canonical_exp = _get_leaf_total_degree_exponents(
        new_leaf_shapes, new_per_leaf_order
    )
    num_canonical = canonical_exp.shape[1]

    # Hash for grouping
    base = max(new_per_leaf_order) + 2 if new_per_leaf_order else 2
    powers = base ** jnp.arange(new_d)
    reduced_hash = jnp.sum(reduced_exps * powers[:, None], axis=0)  # (m,)
    canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)  # (mc,)

    # For each canonical monomial, sum contributions from matching reduced monomials
    # match[i, j] = True if reduced monomial j maps to canonical monomial i
    match = reduced_hash[None, :] == canonical_hash[:, None]  # (mc, m)

    # Sum scaled coefficients for matching monomials
    # scaled_coeffs: (*output_shape, m), match: (mc, m)
    new_coeffs = jnp.einsum(
        "...j,ij->...i", scaled_coeffs, match.astype(scaled_coeffs.dtype)
    )

    # New domain: remove var_idx
    new_lower = jnp.concatenate(
        [tm.flat_domain.lower[:var_idx], tm.flat_domain.lower[var_idx + 1 :]]
    )
    new_upper = jnp.concatenate(
        [tm.flat_domain.upper[:var_idx], tm.flat_domain.upper[var_idx + 1 :]]
    )
    new_domain = interval(new_lower, new_upper)

    # New center: remove var_idx
    new_center = jnp.concatenate(
        [tm.flat_center[:var_idx], tm.flat_center[var_idx + 1 :]]
    )

    # Determine new domain treedef
    if len(new_leaf_shapes) == len(leaf_shapes):
        # Structure preserved (just reduced dimension within leaves)
        new_domain_treedef = tm._domain_treedef
    else:
        # Structure changed (entire leaf removed)
        # We can't easily reconstruct the original structure type, so default to tuple
        dummy_leaves = [jnp.zeros(s) for s in new_leaf_shapes]
        new_domain_treedef = jax.tree_util.tree_structure(tuple(dummy_leaves))

    return TaylorModel(
        new_coeffs,
        canonical_exp,
        tm.remainder,
        new_domain,
        flat_center=new_center,
        _input_pytree=PyTreeShape(new_domain_treedef, new_leaf_shapes),
        _output_pytree=tm._output_pytree,
        _per_leaf_order=new_per_leaf_order,
    )


def tm_integrate_variable(
    tm: TaylorModel,
    var_idx: int | None = None,
    start: float | None = None,
    keep_order: bool = True,
    *,
    leaf_idx: int | None = None,
    leaf_var_idx: int | None = None,
) -> TaylorModel:
    r"""Integrate a TaylorModel with respect to one domain variable.

    By one domain variable, this is one dimension of the flattened domain.

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
    var_idx : int or None
        Index of the domain variable to integrate over (0-indexed into the
        flattened domain). Either ``var_idx`` or both ``leaf_idx`` and
        ``leaf_var_idx`` must be provided.
    start : float or None
        Lower limit of integration (the upper limit is the variable itself).
        If None, defaults to ``tm.center[var_idx]``.
    keep_order : bool, optional
        If True (default), the highest-degree terms produced by integration
        are bounded and absorbed into the remainder so the output has the
        same polynomial order as the input.  If False, the output order for
        variable ``var_idx`` is incremented by 1.
    leaf_idx : int or None
        For structured domains: index of the leaf in the domain pytree.
        Must be used together with ``leaf_var_idx``.
    leaf_var_idx : int or None
        For structured domains: index of the variable within the specified
        leaf (0-indexed into the flattened leaf). Must be used together
        with ``leaf_idx``.

    Returns
    -------
    TaylorModel
        Integrated Taylor model. For structured domains, the result preserves
        the pytree metadata.

    Examples
    --------
    Flat domain (original API):

    >>> tm_result = integrate_variable(tm, var_idx=0)

    Structured domain with leaf/variable indices:

    >>> # For domain = [interval_t, interval_x] where t is leaf 0, x is leaf 1
    >>> tm_result = integrate_variable(tm, leaf_idx=0, leaf_var_idx=0)  # integrate over t
    >>> tm_result = integrate_variable(tm, leaf_idx=1, leaf_var_idx=1)  # integrate over x[1]
    """
    var_idx = _resolve_var_idx(var_idx, leaf_idx, leaf_var_idx, tm.leaf_shapes)
    d = tm.d
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
    start_exponents = tm.exponents.at[var_idx, :].set(
        0
    )  # keep other exponents, set var_idx to 0
    # (these are the original exponents since var_idx exponent maps to the
    #  "other variables" part of each monomial)

    # --- 3. Combine: shifted part - starting point part ---
    # Find which leaf contains var_idx and compute new per_leaf_order
    leaf_shapes = tm._leaf_shapes
    per_leaf_order = list(tm._per_leaf_order)
    containing_leaf_idx = None
    for li in range(len(leaf_shapes)):
        slc = _leaf_slice(leaf_shapes, li)
        if slc.start <= var_idx < slc.stop:
            containing_leaf_idx = li
            break

    # For the new exponent structure, increment the containing leaf's order by 1
    new_per_leaf_order = list(per_leaf_order)
    new_per_leaf_order[containing_leaf_idx] += 1
    new_per_leaf_order = tuple(new_per_leaf_order)

    # Generate canonical exponents with the new per-leaf order
    canonical_exp = _get_leaf_total_degree_exponents(leaf_shapes, new_per_leaf_order)
    num_canonical = canonical_exp.shape[1]

    # Hash-based scatter for both shifted and start terms
    base = max(new_per_leaf_order) + 2
    powers_hash = base ** jnp.arange(d)

    canonical_hash = jnp.sum(canonical_exp * powers_hash[:, None], axis=0)  # (mc,)
    shifted_hash = jnp.sum(shifted_exponents * powers_hash[:, None], axis=0)  # (m,)
    start_hash = jnp.sum(start_exponents * powers_hash[:, None], axis=0)  # (m,)

    # Scatter matrices: (m, mc)
    shifted_scatter = (shifted_hash[:, None] == canonical_hash[None, :]).astype(
        tm.coeffs.dtype
    )
    start_scatter = (start_hash[:, None] == canonical_hash[None, :]).astype(
        tm.coeffs.dtype
    )

    # New coeffs = shifted_coeffs @ shifted_scatter - start_coeffs @ start_scatter
    new_coeffs = shifted_coeffs @ shifted_scatter - start_coeffs @ start_scatter

    # --- 4. Remainder from integrating the original remainder ---
    # ∫_a^{x_i} r dx_i ∈ remainder * (x_i - a)
    xi_minus_a = tm.domain[var_idx] - start
    integ_remainder = tm.remainder * xi_minus_a

    # --- 5. Optionally reduce order back to original ---
    if keep_order:
        orig_per_leaf_order = tm._per_leaf_order

        # Original canonical exponents (static shape, known from per_leaf_order)
        orig_canonical_exp = _get_leaf_total_degree_exponents(
            leaf_shapes, orig_per_leaf_order
        )

        # Scatter from expanded to original canonical basis
        orig_hash = jnp.sum(orig_canonical_exp * powers_hash[:, None], axis=0)
        keep_scatter = (canonical_hash[:, None] == orig_hash[None, :]).astype(
            tm.coeffs.dtype
        )

        # Project coefficients to original basis (high-order monomials have no
        # hash match in orig_hash, so their contribution is zero)
        kept_coeffs = new_coeffs @ keep_scatter

        # Bound high-order terms and absorb into remainder
        high_mask = jnp.sum(keep_scatter, axis=1) == 0
        mono_bounds = _bound_monomials_over_domain(
            canonical_exp, tm.shifted_domain, max(new_per_leaf_order)
        )

        high_coeffs = jnp.where(high_mask, new_coeffs, 0.0)
        c_pos = jnp.maximum(high_coeffs, 0.0)
        c_neg = jnp.minimum(high_coeffs, 0.0)

        trunc_lower = jnp.sum(
            c_pos * mono_bounds.lower + c_neg * mono_bounds.upper, axis=-1
        )
        trunc_upper = jnp.sum(
            c_pos * mono_bounds.upper + c_neg * mono_bounds.lower, axis=-1
        )
        trunc_remainder = interval(trunc_lower, trunc_upper)

        total_remainder = trunc_remainder + integ_remainder

        return TaylorModel(
            kept_coeffs,
            orig_canonical_exp,
            total_remainder,
            tm.flat_domain,
            flat_center=tm.flat_center,
            _input_pytree=tm._input_pytree,
            _output_pytree=tm._output_pytree,
            _per_leaf_order=tm._per_leaf_order,
        )
    else:
        return TaylorModel(
            new_coeffs,
            canonical_exp,
            integ_remainder,
            tm.flat_domain,
            flat_center=tm.flat_center,
            _input_pytree=tm._input_pytree,
            _output_pytree=tm._output_pytree,
            _per_leaf_order=new_per_leaf_order,
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
    exponents = ref.exponents
    # Element-wise max of per-leaf orders
    from functools import reduce

    per_leaf_order = reduce(_max_order, (tm._per_leaf_order for tm in tms))

    # Stack coefficients along specified axis of output shape
    # coeffs have shape (*output_shape, m), so axis applies to output_shape
    coeffs = jnp.concatenate([tm.coeffs for tm in tms], axis=axis)

    # Stack remainders along same axis
    remainder = interval(
        jnp.concatenate([tm.remainder.lower for tm in tms], axis=axis),
        jnp.concatenate([tm.remainder.upper for tm in tms], axis=axis),
    )

    # Build merged _output_pytree from all input TMs' output pytrees
    merged_leaf_shapes = []
    for tm in tms:
        merged_leaf_shapes.extend(tm._output_pytree.leaf_shapes)
    merged_leaf_shapes = tuple(merged_leaf_shapes)
    # Use a list treedef for the merged output
    dummy_leaves = [jnp.zeros(s) for s in merged_leaf_shapes]
    merged_treedef = jax.tree_util.tree_structure(dummy_leaves, is_leaf=_is_interval_or_array_leaf)
    output_pytree = PyTreeShape(merged_treedef, merged_leaf_shapes)

    return TaylorModel(
        coeffs,
        exponents,
        remainder,
        ref.flat_domain,
        flat_center=ref.flat_center,
        _input_pytree=ref._input_pytree,
        _output_pytree=output_pytree,
        _per_leaf_order=per_leaf_order,
    )


# Alias for backward compatibility
integrate_variable = tm_integrate_variable
