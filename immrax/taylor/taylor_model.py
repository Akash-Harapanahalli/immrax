"""Taylor Model implementation for reachability analysis.

Taylor models represent sets as polynomial approximations with rigorous
interval remainder bounds, providing a powerful tool for propagating
uncertainty through nonlinear functions.
"""

from typing import Callable, Tuple
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental import jet
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike

from immrax.inclusion import Interval, interval, icentpert

# Zonotope is imported lazily in to_zonotope() to avoid circular import


# Cache for canonical exponent structures
@lru_cache(maxsize=128)
def _get_canonical_exponents(d: int, max_order: int) -> Array:
    """Get cached canonical exponents for given dimension and order."""
    return _generate_exponents_impl(d, max_order)


def _generate_exponents_impl(d: int, max_order: int) -> Array:
    """Generate all exponent multi-indices up to given order."""
    from itertools import product

    exponents = [
        exp for exp in product(range(max_order + 1), repeat=d)
        if sum(exp) <= max_order
    ]
    return jnp.array(exponents, dtype=jnp.int32).T


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
    domain_center : ArrayLike
        Center of the domain box, shape (d,)
    domain_radius : ArrayLike
        Half-width of the domain box, shape (d,)

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
    domain_center: Array  # Domain center, shape (d,)
    domain_radius: Array  # Domain half-width, shape (d,)

    def __init__(
        self,
        coeffs: ArrayLike,
        exponents: ArrayLike,
        remainder: Interval,
        domain_center: ArrayLike,
        domain_radius: ArrayLike,
        _static_order: int | None = None,
    ) -> None:
        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.remainder = remainder
        self.domain_center = jnp.asarray(domain_center)
        self.domain_radius = jnp.asarray(domain_radius)

        # Store the polynomial order as static data (for JIT compatibility)
        # If not provided, compute from exponents (only works outside JIT)
        if _static_order is not None:
            self._static_order = _static_order
        else:
            # Compute order from exponents (requires concrete values)
            self._static_order = int(jnp.max(jnp.sum(self.exponents, axis=0)))

        # Store output shape (all axes except the last monomial axis)
        # Monomials are always the LAST axis of coeffs
        self._output_shape = self.coeffs.shape[:-1]

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
        if self.domain_center.shape[0] != self.exponents.shape[0]:
            raise ValueError(
                f"domain_center must match exponents dimension: "
                f"{self.domain_center.shape[0]} vs {self.exponents.shape[0]}"
            )
        if self.domain_radius.shape != self.domain_center.shape:
            raise ValueError(
                f"domain_radius must match domain_center shape: "
                f"{self.domain_radius.shape} vs {self.domain_center.shape}"
            )

    # --- Pytree methods ---

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Array, Array, Interval, Array, Array], dict]:
        return (
            (
                self.coeffs,
                self.exponents,
                self.remainder,
                self.domain_center,
                self.domain_radius,
            ),
            {"_static_order": self._static_order, "_output_shape": self._output_shape},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorModel":
        static_order = aux_data.get("_static_order") if aux_data else None
        # _output_shape is recomputed in __init__ from coeffs.shape[:-1]
        return cls(*children, _static_order=static_order)

    # --- Properties ---

    @property
    def n(self) -> int:
        """Total number of output elements (product of output_shape)."""
        import math
        return math.prod(self._output_shape) if self._output_shape else 1

    @property
    def d(self) -> int:
        """Domain (input) dimension."""
        return self.exponents.shape[0]

    @property
    def num_monomials(self) -> int:
        """Number of monomial terms."""
        return self.coeffs.shape[-1]

    @property
    def order(self) -> int:
        """Maximum polynomial order (sum of exponents)."""
        return int(jnp.max(jnp.sum(self.exponents, axis=0)))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape."""
        return self._output_shape

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

        return TaylorModel(c, self.exponents, rem, self.domain_center, self.domain_radius,
                           _static_order=self._static_order)

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
        # Normalize to centered coordinates
        x_centered = (x - self.domain_center) / self.domain_radius

        # Evaluate each monomial
        # monomial_i = prod_j x_centered[j]^exponents[j, i]
        log_abs_x = jnp.log(jnp.abs(x_centered) + 1e-30)
        log_monomials = self.exponents.T @ log_abs_x

        # Handle signs for negative x values
        is_negative = x_centered < 0
        odd_exp = self.exponents % 2
        sign_flips = odd_exp.T @ is_negative.astype(jnp.float32)
        signs = jnp.where(sign_flips % 2 == 0, 1.0, -1.0)

        monomials = signs * jnp.exp(log_monomials)  # (num_monomials,)

        # Handle x_centered = 0 cases (0^0 = 1, 0^k = 0 for k > 0)
        zero_mask = jnp.abs(x_centered) < 1e-30
        has_zero_exp = jnp.any(
            (self.exponents > 0) & zero_mask[:, None], axis=0
        )
        monomials = jnp.where(has_zero_exp, 0.0, monomials)

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
        return interval(
            poly_val + self.remainder.lower, poly_val + self.remainder.upper
        )

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

        # Vectorized computation of monomial properties
        is_constant = jnp.all(self.exponents == 0, axis=0)  # (m,)
        has_odd = jnp.any(self.exponents % 2 == 1, axis=0)  # (m,)

        # For even monomials (non-constant), we shift center by 0.5*coeff
        # and scale generators by 0.5
        even_non_const = ~has_odd & ~is_constant  # (m,)
        center_shift = jnp.sum(
            jnp.where(even_non_const, 0.5 * self.coeffs, 0.0), axis=-1
        )  # (n,)

        ox = self.constant_term + remainder_center + center_shift

        # Generator scaling: 0.5 for even non-constant, 1.0 for odd
        gen_scale = jnp.where(even_non_const, 0.5, 1.0)  # (m,)

        # Non-constant generators from polynomial terms
        non_const_mask = ~is_constant  # (m,)
        poly_generators = jnp.where(
            non_const_mask, self.coeffs * gen_scale, 0.0
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
        target_order: int | None = None,
        method: str | None = None,
    ) -> "TaylorModel":
        """Convert to canonical exponent structure.

        The canonical structure includes all monomials up to target_order,
        sorted consistently. This ensures TMs are compatible for concatenation.

        Parameters
        ----------
        target_order : int, optional
            Target order. If None, uses current maximum order.
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
        if target_order is None:
            target_order = self._static_order

        # Generate canonical exponents (uses cached version for JIT compatibility)
        canonical_exp = _get_canonical_exponents(self.d, target_order)
        num_canonical = canonical_exp.shape[1]

        # Auto-select method based on num_canonical if not specified
        # Threshold ~50: below this broadcast is faster due to lower overhead
        if method is None:
            method = "searchsorted" if num_canonical > 50 else "broadcast"

        # Compute a unique hash for each exponent vector
        # Use weighted sum: sum_i exp[i] * (max_order+1)^i
        base = target_order + 2  # Ensure no collisions
        powers = base ** jnp.arange(self.d)

        # Hash current exponents: (m,)
        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)

        # Hash canonical exponents: (num_canonical,)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        # Check if each current term has a match (is within target_order)
        term_orders = jnp.sum(self.exponents, axis=0)  # (m,)
        has_match = term_orders <= target_order  # (m,)

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
        # Terms with order > target_order need to be absorbed
        absorb_mask = term_orders > target_order  # (m,)

        # For absorbed terms, compute their bounds
        # Monomial bounds: odd exponents -> [-1,1], even -> [0,1]
        has_odd = jnp.any(self.exponents % 2 == 1, axis=0)  # (m,)
        mono_lower = jnp.where(has_odd, -1.0, 0.0)  # (m,)
        mono_upper = jnp.ones(self.num_monomials)  # (m,)

        # Compute term bounds for absorbed terms
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

        new_remainder = self.remainder + interval(absorbed_lower, absorbed_upper)

        return TaylorModel(
            new_coeffs,
            canonical_exp,
            new_remainder,
            self.domain_center,
            self.domain_radius,
            _static_order=target_order,
        )

    def reduce_order(self, target_order: int) -> "TaylorModel":
        """Reduce polynomial order, absorbing high-order terms into remainder.

        Parameters
        ----------
        target_order : int
            Target maximum polynomial order

        Returns
        -------
        TaylorModel
            Reduced order Taylor model (overapproximation)
        """
        # Separate terms to keep and terms to absorb
        term_orders = jnp.sum(self.exponents, axis=0)  # (m,)
        keep_mask = term_orders <= target_order  # (m,)
        absorb_mask = ~keep_mask  # (m,)

        # Vectorized monomial bounds for absorbed terms
        has_odd = jnp.any(self.exponents % 2 == 1, axis=0)  # (m,)
        is_constant = jnp.all(self.exponents == 0, axis=0)  # (m,)
        mono_lower = jnp.where(is_constant, 1.0, jnp.where(has_odd, -1.0, 0.0))  # (m,)
        mono_upper = jnp.ones(self.num_monomials)  # (m,)

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
            self.domain_center,
            self.domain_radius,
            _static_order=target_order,
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
            f"remainder={self.remainder!r}, domain_center={self.domain_center!r}, "
            f"domain_radius={self.domain_radius!r})"
        )


# --- Helper functions ---


def _bound_monomial(exp: Array) -> Interval:
    """Bound a monomial over [-1, 1]^d.

    Parameters
    ----------
    exp : Array
        Exponent multi-index

    Returns
    -------
    Interval
        Scalar interval bounding the monomial
    """
    # Constant term (all exponents = 0): x^0 = 1 exactly
    is_constant = jnp.all(exp == 0)
    # If any exponent is odd: range is [-1, 1]
    # If all exponents are even (but not constant): range is [0, 1]
    has_odd = jnp.any(exp % 2 == 1)

    lower = jnp.where(is_constant, 1.0, jnp.where(has_odd, -1.0, 0.0))
    upper = 1.0
    return interval(lower, upper)


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


def _generate_exponents(d: int, max_order: int) -> Array:
    """Generate all exponent multi-indices up to given order.

    Parameters
    ----------
    d : int
        Number of variables
    max_order : int
        Maximum total degree

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
    domain_center: ArrayLike | None = None,
    domain_radius: ArrayLike | None = None,
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
    domain_center : ArrayLike, optional
        Domain center, shape (d,). Default is origin.
    domain_radius : ArrayLike, optional
        Domain half-width, shape (d,). Default is ones.

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
            # Scalar output
            remainder = interval(jnp.zeros((), dtype=coeffs.dtype))
        else:
            remainder = interval(jnp.zeros(output_shape, dtype=coeffs.dtype))

    if domain_center is None:
        domain_center = jnp.zeros(d, dtype=coeffs.dtype)
    else:
        domain_center = jnp.asarray(domain_center)

    if domain_radius is None:
        domain_radius = jnp.ones(d, dtype=coeffs.dtype)
    else:
        domain_radius = jnp.asarray(domain_radius)

    # Compute order from exponents
    computed_order = int(jnp.max(jnp.sum(exponents, axis=0)))
    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=computed_order)


def taylor_model_from_interval(iv: Interval, order: int = 1) -> TaylorModel:
    """Create a Taylor model representing a box interval.

    Represents the interval [c-r, c+r] as the polynomial p(u) = c + r*u
    where u in [-1, 1].

    Parameters
    ----------
    iv : Interval
        The domain interval with shape (*output_shape,).
        For scalar intervals, use interval(scalar_value).
    order : int
        Polynomial degree of the Taylor Model (default: 1)

    Returns
    -------
    TaylorModel
        Taylor model for identity function with output shape matching interval shape.
        Domain dimension d = prod(output_shape).
    """
    center = iv.center
    radius = iv.pert
    output_shape = center.shape

    # Handle scalar case
    if len(output_shape) == 0:
        center = center[None]
        radius = radius[None]
        output_shape = (1,)
        scalar_output = True
    else:
        scalar_output = False

    # Flatten to get domain dimension
    import math
    n = math.prod(output_shape)
    center_flat = center.reshape(-1)
    radius_flat = radius.reshape(-1)

    # For identity: p(x_norm) = center + radius * x_norm
    # Note: Higher order terms are simply zero.

    # Generate exponents for full order (d = n = prod(output_shape))
    exponents = _generate_exponents(n, order)
    num_monomials = exponents.shape[1]

    # Vectorized coefficient computation
    # Constant term: where sum(exponents) == 0
    is_constant = jnp.sum(exponents, axis=0) == 0  # (m,)

    # Linear terms: where exponents == e_i for each dimension i
    # A term is linear in dimension i if exponents[:, j] == e_i
    eye_n = jnp.eye(n, dtype=jnp.int32)  # (n, n)
    # is_linear[i, j] = True if monomial j is x_i (linear in dim i only)
    # Compare exponents (d, m) with each unit vector e_i, reduce over d dimension
    is_linear = jnp.all(exponents[:, None, :] == eye_n[:, :, None], axis=0)  # (n, m)

    # Build coefficients matrix (flat first, then reshape)
    # Constant contribution: center broadcasted to constant term column
    const_coeffs = jnp.where(is_constant[None, :], center_flat[:, None], 0.0)  # (n, m)

    # Linear contribution: radius[i] for the x_i term in row i
    linear_coeffs = jnp.where(is_linear, radius_flat[:, None], 0.0)  # (n, m)

    coeffs_flat = const_coeffs + linear_coeffs  # (n, m)

    # Reshape to output shape
    if scalar_output:
        # For scalar output, coeffs should have shape (m,) not (1, m)
        coeffs = coeffs_flat[0]  # (m,)
        remainder = interval(jnp.zeros((), dtype=center_flat.dtype))
        domain_center = center_flat
        domain_radius = radius_flat
    else:
        coeffs = coeffs_flat.reshape(*output_shape, num_monomials)  # (*output_shape, m)
        remainder = interval(jnp.zeros(output_shape, dtype=center_flat.dtype))
        domain_center = center_flat
        domain_radius = radius_flat

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=order)


def taylor_model_from_function(
    f: Callable,
    domain_center: ArrayLike,
    domain_radius: ArrayLike,
    max_order: int,
    n_out: int | None = None,
) -> TaylorModel:
    """Create a Taylor model by Taylor expansion of a function.

    Uses automatic differentiation to compute Taylor coefficients.

    Parameters
    ----------
    f : Callable
        Function to approximate, f: R^d -> R^n
    domain_center : ArrayLike
        Expansion point, shape (d,)
    domain_radius : ArrayLike
        Domain half-width, shape (d,)
    max_order : int
        Maximum Taylor expansion order
    n_out : int, optional
        Output dimension. Inferred from f if not provided.

    Returns
    -------
    TaylorModel
        Taylor model approximation with remainder bounds
    """
    domain_center = jnp.asarray(domain_center)
    domain_radius = jnp.asarray(domain_radius)
    d = domain_center.shape[0]

    # Evaluate at center to get output dimension
    f_center = f(domain_center)
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
            f(domain_center).reshape(-1, 1),
            jnp.zeros((d, 1), dtype=jnp.int32),
            icentpert(jnp.zeros(n), jnp.zeros(n)),
            domain_center,
            domain_radius
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
    val = f(domain_center)
    if val.ndim == 0:
        val = val[None] # (1,) if scalar
    deriv_tensors.append(val)

    # Ensure we strictly differentiate a vector-valued function
    # to maintain consistent tensor shape (n, d, d...)
    if f(domain_center).ndim == 0:
        def f_vec(x):
            v = f(x)
            return v[None]
        curr_f = f_vec
    else:
        curr_f = f

    # Compute up to max_order + 1
    for k in range(1, max_order + 2):
            curr_f = jax.jacfwd(curr_f)
            tensor = curr_f(domain_center)
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

        scale = jnp.prod(domain_radius ** exp) / fact_prod

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
    fact = jax.scipy.special.gamma(next_order + 1)
    remainder_bound = current_bound / fact

    # Ensure non-zero for safety if needed, though 0 is valid for exact polynomials
    remainder = icentpert(jnp.zeros(n, dtype=f_center.dtype), remainder_bound)

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=max_order)


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

    # Normalize value to centered coordinate for the target variable
    v_norm = (value - tm.domain_center[var_idx]) / tm.domain_radius[var_idx]

    # Compute the scalar factor for each monomial: v_norm^{exp[var_idx]}
    var_exps = tm.exponents[var_idx, :]  # (m,)
    # v_norm^k for each monomial
    var_factors = v_norm ** var_exps  # (m,)

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
    max_order = tm._static_order
    canonical_exp = _get_canonical_exponents(new_d, max_order)
    num_canonical = canonical_exp.shape[1]

    # Hash for grouping
    base = max_order + 2
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
    new_center = jnp.concatenate([
        tm.domain_center[:var_idx],
        tm.domain_center[var_idx + 1:]
    ])
    new_radius = jnp.concatenate([
        tm.domain_radius[:var_idx],
        tm.domain_radius[var_idx + 1:]
    ])

    return TaylorModel(
        new_coeffs, canonical_exp, tm.remainder,
        new_center, new_radius, _static_order=max_order
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
    domain_center = ref.domain_center
    domain_radius = ref.domain_radius
    exponents = ref.exponents
    static_order = max(tm._static_order for tm in tms)

    # Stack coefficients along specified axis of output shape
    # coeffs have shape (*output_shape, m), so axis applies to output_shape
    coeffs = jnp.concatenate([tm.coeffs for tm in tms], axis=axis)

    # Stack remainders along same axis
    remainder = interval(
        jnp.concatenate([tm.remainder.lower for tm in tms], axis=axis),
        jnp.concatenate([tm.remainder.upper for tm in tms], axis=axis)
    )

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=static_order)
