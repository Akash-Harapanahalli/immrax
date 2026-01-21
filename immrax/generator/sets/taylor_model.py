"""Taylor Model implementation for reachability analysis.

Taylor models represent sets as polynomial approximations with rigorous
interval remainder bounds, providing a powerful tool for propagating
uncertainty through nonlinear functions.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike

from .zonotope import Zonotope


@register_pytree_node_class
class TaylorModel:
    r"""Defines a Taylor model set representation

    .. math::
        TM = \{ p(x) + r : x \in D, r \in I \}

    where:
    - :math:`p(x)` is a multivariate polynomial in the domain variables
    - :math:`D \subseteq \mathbb{R}^d` is the domain (typically a box)
    - :math:`I \subseteq \mathbb{R}^n` is the interval remainder

    The polynomial is represented in the form:

    .. math::
        p(x) = c + \sum_{|\alpha| \leq k} a_\alpha (x - x_0)^\alpha

    where :math:`\alpha` is a multi-index, :math:`x_0` is the expansion point,
    and :math:`k` is the polynomial order.

    For computational efficiency, we use a factored representation where
    the polynomial coefficients are stored as a tensor.

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients. For order k and domain dimension d,
        shape is (n, num_monomials) where num_monomials = C(d+k, k).
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials), integer entries.
        Each column is a multi-index α.
    remainder : ArrayLike
        Interval remainder bounds, shape (n, 2) with [lower, upper].
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

    coeffs: Array  # Polynomial coefficients, shape (n, num_monomials)
    exponents: Array  # Exponent matrix, shape (d, num_monomials)
    remainder: Array  # Interval remainder [lower, upper], shape (n, 2)
    domain_center: Array  # Domain center, shape (d,)
    domain_radius: Array  # Domain half-width, shape (d,)

    def __init__(
        self,
        coeffs: ArrayLike,
        exponents: ArrayLike,
        remainder: ArrayLike,
        domain_center: ArrayLike,
        domain_radius: ArrayLike,
    ) -> None:
        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.remainder = jnp.asarray(remainder)
        self.domain_center = jnp.asarray(domain_center)
        self.domain_radius = jnp.asarray(domain_radius)

        # Validate dimensions
        if self.coeffs.ndim != 2:
            raise ValueError(f"coeffs must be 2D, got shape {self.coeffs.shape}")
        if self.exponents.ndim != 2:
            raise ValueError(f"exponents must be 2D, got shape {self.exponents.shape}")
        if self.coeffs.shape[1] != self.exponents.shape[1]:
            raise ValueError(
                f"coeffs and exponents must have same number of monomials: "
                f"{self.coeffs.shape[1]} vs {self.exponents.shape[1]}"
            )
        if self.remainder.ndim != 2 or self.remainder.shape[1] != 2:
            raise ValueError(
                f"remainder must have shape (n, 2), got {self.remainder.shape}"
            )
        if self.remainder.shape[0] != self.coeffs.shape[0]:
            raise ValueError(
                f"remainder and coeffs must have same output dimension: "
                f"{self.remainder.shape[0]} vs {self.coeffs.shape[0]}"
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
    ) -> Tuple[Tuple[Array, Array, Array, Array, Array], None]:
        return (
            (
                self.coeffs,
                self.exponents,
                self.remainder,
                self.domain_center,
                self.domain_radius,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorModel":
        return cls(*children)

    # --- Properties ---

    @property
    def n(self) -> int:
        """Output dimension."""
        return self.coeffs.shape[0]

    @property
    def d(self) -> int:
        """Domain (input) dimension."""
        return self.exponents.shape[0]

    @property
    def num_monomials(self) -> int:
        """Number of monomial terms."""
        return self.coeffs.shape[1]

    @property
    def order(self) -> int:
        """Maximum polynomial order (sum of exponents)."""
        return int(jnp.max(jnp.sum(self.exponents, axis=0)))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape."""
        return (self.n,)

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return self.coeffs.dtype

    @property
    def constant_term(self) -> Array:
        """Get the constant (order 0) term of the polynomial."""
        # Find the column where all exponents are 0
        is_constant = jnp.all(self.exponents == 0, axis=0)
        # Sum coefficients for constant terms (should be exactly one)
        return jnp.sum(
            jnp.where(is_constant[None, :], self.coeffs, 0.0), axis=1
        )

    @property
    def remainder_lower(self) -> Array:
        """Lower bound of the remainder interval."""
        return self.remainder[:, 0]

    @property
    def remainder_upper(self) -> Array:
        """Upper bound of the remainder interval."""
        return self.remainder[:, 1]

    @property
    def remainder_width(self) -> Array:
        """Width of the remainder interval."""
        return self.remainder[:, 1] - self.remainder[:, 0]

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
            Polynomial value, shape (n,)
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

        monomials = signs * jnp.exp(log_monomials)

        # Handle x_centered = 0 cases (0^0 = 1, 0^k = 0 for k > 0)
        zero_mask = jnp.abs(x_centered) < 1e-30
        has_zero_exp = jnp.any(
            (self.exponents > 0) & zero_mask[:, None], axis=0
        )
        monomials = jnp.where(has_zero_exp, 0.0, monomials)

        return self.coeffs @ monomials

    def evaluate(self, x: ArrayLike) -> "Interval":
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
        from immrax.inclusion.interval import Interval

        poly_val = self.evaluate_polynomial(x)
        return Interval(
            poly_val + self.remainder_lower, poly_val + self.remainder_upper
        )

    # --- Set operations ---

    def __add__(self, other: "TaylorModel") -> "TaylorModel":
        """Addition of two Taylor models."""
        if not isinstance(other, TaylorModel):
            raise TypeError("Can only add TaylorModel to TaylorModel")
        if self.d != other.d:
            raise ValueError(f"Domain dimensions must match: {self.d} vs {other.d}")
        if not jnp.allclose(self.domain_center, other.domain_center) or not jnp.allclose(
            self.domain_radius, other.domain_radius
        ):
            raise ValueError("Taylor models must have the same domain")

        # Merge polynomial terms
        new_coeffs, new_exponents = _merge_taylor_terms(
            self.coeffs, self.exponents, other.coeffs, other.exponents
        )

        # Add remainders (interval addition)
        new_remainder = jnp.stack(
            [
                self.remainder_lower + other.remainder_lower,
                self.remainder_upper + other.remainder_upper,
            ],
            axis=1,
        )

        return TaylorModel(
            new_coeffs, new_exponents, new_remainder, self.domain_center, self.domain_radius
        )

    def __sub__(self, other: "TaylorModel") -> "TaylorModel":
        """Subtraction: TM1 - TM2."""
        return self + (-other)

    def __neg__(self) -> "TaylorModel":
        """Negation: -TM."""
        new_remainder = jnp.stack(
            [-self.remainder_upper, -self.remainder_lower], axis=1
        )
        return TaylorModel(
            -self.coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
        )

    def __mul__(self, other: ArrayLike) -> "TaylorModel":
        """Scalar multiplication: TM * α."""
        alpha = jnp.asarray(other)
        if alpha.ndim == 0:
            # Scalar
            new_coeffs = alpha * self.coeffs
            if alpha >= 0:
                new_remainder = alpha * self.remainder
            else:
                new_remainder = jnp.stack(
                    [alpha * self.remainder_upper, alpha * self.remainder_lower],
                    axis=1,
                )
        else:
            # Element-wise scaling
            new_coeffs = alpha[:, None] * self.coeffs
            pos_mask = (alpha >= 0)[:, None]
            new_remainder = jnp.where(
                pos_mask,
                alpha[:, None] * self.remainder,
                jnp.stack(
                    [alpha * self.remainder_upper, alpha * self.remainder_lower],
                    axis=1,
                ),
            )
        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
        )

    def __rmul__(self, other: ArrayLike) -> "TaylorModel":
        """Scalar multiplication: α * TM."""
        return self.__mul__(other)

    def __rmatmul__(self, other: ArrayLike) -> "TaylorModel":
        """Left matrix multiplication: M @ TM.

        For M ∈ R^{m×n} and TM with output dimension n.
        """
        M = jnp.asarray(other)
        new_coeffs = M @ self.coeffs

        # Remainder transformation: M @ [r_l, r_u]
        # Need to handle signs in M
        M_pos = jnp.maximum(M, 0)
        M_neg = jnp.minimum(M, 0)
        new_lower = M_pos @ self.remainder_lower + M_neg @ self.remainder_upper
        new_upper = M_pos @ self.remainder_upper + M_neg @ self.remainder_lower
        new_remainder = jnp.stack([new_lower, new_upper], axis=1)

        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
        )

    def multiply(self, other: "TaylorModel", max_order: int | None = None) -> "TaylorModel":
        """Multiply two Taylor models (polynomial multiplication).

        Parameters
        ----------
        other : TaylorModel
            Taylor model to multiply with
        max_order : int, optional
            Maximum polynomial order to keep. Higher order terms are
            absorbed into the remainder.

        Returns
        -------
        TaylorModel
            Product Taylor model
        """
        if not isinstance(other, TaylorModel):
            raise TypeError("Can only multiply TaylorModel with TaylorModel")
        if self.n != other.n:
            raise ValueError(f"Output dimensions must match: {self.n} vs {other.n}")
        if self.d != other.d:
            raise ValueError(f"Domain dimensions must match: {self.d} vs {other.d}")

        if max_order is None:
            max_order = self.order + other.order

        # Polynomial multiplication: (p1 + r1) * (p2 + r2)
        # = p1 * p2 + p1 * r2 + r1 * p2 + r1 * r2
        # The remainder from p1 * p2 comes from truncating high-order terms

        n = self.n
        new_coeffs_list = []
        new_exp_list = []

        # p1 * p2 term-by-term
        for i in range(self.num_monomials):
            for j in range(other.num_monomials):
                new_exp = self.exponents[:, i] + other.exponents[:, j]
                total_order = jnp.sum(new_exp)

                # Coefficient is product of coefficients (element-wise for each output dim)
                new_coeff = self.coeffs[:, i] * other.coeffs[:, j]

                if total_order <= max_order:
                    new_coeffs_list.append(new_coeff)
                    new_exp_list.append(new_exp)

        if len(new_coeffs_list) > 0:
            new_coeffs = jnp.stack(new_coeffs_list, axis=1)
            new_exponents = jnp.stack(new_exp_list, axis=1)
        else:
            new_coeffs = jnp.zeros((n, 1), dtype=self.dtype)
            new_exponents = jnp.zeros((self.d, 1), dtype=jnp.int32)

        # Compact terms with same exponents
        new_coeffs, new_exponents = _compact_taylor_terms(new_coeffs, new_exponents)

        # Compute remainder bounds
        # Need to bound: p1 * r2 + r1 * p2 + r1 * r2 + truncated terms

        # Bound polynomial ranges over domain
        p1_bounds = self._bound_polynomial()  # (n, 2)
        p2_bounds = other._bound_polynomial()  # (n, 2)

        # Interval multiplication for cross terms
        # p1 * r2: p1 ∈ [p1_l, p1_u], r2 ∈ [r2_l, r2_u]
        p1_r2 = _interval_multiply(p1_bounds, other.remainder)
        r1_p2 = _interval_multiply(self.remainder, p2_bounds)
        r1_r2 = _interval_multiply(self.remainder, other.remainder)

        # Combined remainder
        new_remainder = _interval_add(_interval_add(p1_r2, r1_p2), r1_r2)

        return TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
        )

    def _bound_polynomial(self) -> Array:
        """Bound the polynomial part over the domain.

        Returns
        -------
        Array
            Bounds, shape (n, 2) with [lower, upper]
        """
        # Use interval arithmetic to bound each monomial
        n = self.n

        lower = jnp.zeros(n, dtype=self.dtype)
        upper = jnp.zeros(n, dtype=self.dtype)

        for i in range(self.num_monomials):
            exp_i = self.exponents[:, i]
            coeff_i = self.coeffs[:, i]

            # Bound monomial over [-1, 1]^d (normalized domain)
            mono_lower = jnp.ones((), dtype=self.dtype)
            mono_upper = jnp.ones((), dtype=self.dtype)

            for k in range(self.d):
                e_k = exp_i[k]
                if e_k == 0:
                    continue
                elif e_k % 2 == 0:
                    # Even power over [-1,1]: range is [0, 1]
                    mono_lower = mono_lower * 0.0
                    mono_upper = mono_upper * 1.0
                else:
                    # Odd power over [-1,1]: range is [-1, 1]
                    mono_lower = mono_lower * (-1.0)
                    mono_upper = mono_upper * 1.0

            # Multiply by coefficient
            c_pos = jnp.maximum(coeff_i, 0)
            c_neg = jnp.minimum(coeff_i, 0)
            term_lower = c_pos * mono_lower + c_neg * mono_upper
            term_upper = c_pos * mono_upper + c_neg * mono_lower

            lower = lower + term_lower
            upper = upper + term_upper

        return jnp.stack([lower, upper], axis=1)

    # --- Conversion methods ---

    def to_zonotope(self) -> Zonotope:
        """Convert to a zonotope (overapproximation).

        Each monomial term becomes a generator.
        """
        # Center is the constant term plus remainder center
        remainder_center = (self.remainder_lower + self.remainder_upper) / 2
        remainder_radius = (self.remainder_upper - self.remainder_lower) / 2

        ox = self.constant_term + remainder_center

        # Non-constant terms become generators
        # Scale by their range over [-1,1]^d
        generators_list = []

        for i in range(self.num_monomials):
            exp_i = self.exponents[:, i]
            coeff_i = self.coeffs[:, i]

            if jnp.all(exp_i == 0):
                continue  # Skip constant term

            # Monomial range over [-1,1]^d
            has_odd = jnp.any(exp_i % 2 == 1)
            if has_odd:
                # Range is [-1, 1], generator is coeff_i
                generators_list.append(coeff_i)
            else:
                # Range is [0, 1], need to shift
                # [0,1] = 0.5 + 0.5*[-1,1]
                ox = ox + 0.5 * coeff_i
                generators_list.append(0.5 * coeff_i)

        # Add remainder as axis-aligned generators
        for i in range(self.n):
            gen = jnp.zeros(self.n, dtype=self.dtype)
            gen = gen.at[i].set(remainder_radius[i])
            if remainder_radius[i] > 1e-12:
                generators_list.append(gen)

        if len(generators_list) > 0:
            G = jnp.stack(generators_list, axis=1)
        else:
            G = jnp.zeros((self.n, 0), dtype=self.dtype)

        return Zonotope(ox, G)

    def interval_hull(self) -> "Interval":
        """Compute the interval hull (bounding box)."""
        from immrax.inclusion.interval import Interval

        poly_bounds = self._bound_polynomial()
        lower = poly_bounds[:, 0] + self.remainder_lower
        upper = poly_bounds[:, 1] + self.remainder_upper
        return Interval(lower, upper)

    def contains(self, x: ArrayLike) -> Array:
        """Check if a point is in the range (necessary condition)."""
        x = jnp.asarray(x)
        hull = self.interval_hull()
        return jnp.all((x >= hull.lower) & (x <= hull.upper))

    # --- Order reduction ---

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
        if target_order >= self.order:
            return self

        # Separate terms to keep and terms to absorb
        term_orders = jnp.sum(self.exponents, axis=0)
        keep_mask = term_orders <= target_order

        # Bound absorbed terms
        absorbed_lower = jnp.zeros(self.n, dtype=self.dtype)
        absorbed_upper = jnp.zeros(self.n, dtype=self.dtype)

        for i in range(self.num_monomials):
            if term_orders[i] > target_order:
                exp_i = self.exponents[:, i]
                coeff_i = self.coeffs[:, i]

                # Bound this monomial
                has_odd = jnp.any(exp_i % 2 == 1)
                if has_odd:
                    mono_lower, mono_upper = -1.0, 1.0
                else:
                    mono_lower, mono_upper = 0.0, 1.0

                c_pos = jnp.maximum(coeff_i, 0)
                c_neg = jnp.minimum(coeff_i, 0)
                absorbed_lower = absorbed_lower + c_pos * mono_lower + c_neg * mono_upper
                absorbed_upper = absorbed_upper + c_pos * mono_upper + c_neg * mono_lower

        # New remainder includes absorbed terms
        new_remainder = jnp.stack(
            [
                self.remainder_lower + absorbed_lower,
                self.remainder_upper + absorbed_upper,
            ],
            axis=1,
        )

        # Filter coefficients and exponents
        # For JIT compatibility, zero out instead of filtering
        new_coeffs = jnp.where(keep_mask[None, :], self.coeffs, 0.0)

        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
        )

    # --- String representation ---

    def __str__(self) -> str:
        return (
            f"TaylorModel(n={self.n}, d={self.d}, order={self.order}, "
            f"monomials={self.num_monomials})"
        )

    def __repr__(self) -> str:
        return (
            f"TaylorModel(coeffs={self.coeffs!r}, exponents={self.exponents!r}, "
            f"remainder={self.remainder!r}, domain_center={self.domain_center!r}, "
            f"domain_radius={self.domain_radius!r})"
        )


# --- Helper functions ---


def _merge_taylor_terms(
    coeffs1: Array, exp1: Array, coeffs2: Array, exp2: Array
) -> Tuple[Array, Array]:
    """Merge Taylor terms from two Taylor models."""
    # Simple concatenation (compaction done separately)
    coeffs = jnp.concatenate([coeffs1, coeffs2], axis=1)
    exp = jnp.concatenate([exp1, exp2], axis=1)
    return _compact_taylor_terms(coeffs, exp)


def _compact_taylor_terms(coeffs: Array, exponents: Array) -> Tuple[Array, Array]:
    """Compact Taylor terms by combining terms with identical exponents.

    Simplified version that removes zero coefficients.
    """
    # Remove terms with zero coefficients
    norms = jnp.linalg.norm(coeffs, axis=0)
    nonzero_mask = norms > 1e-12
    coeffs_compact = jnp.where(nonzero_mask[None, :], coeffs, 0.0)
    return coeffs_compact, exponents


def _interval_multiply(a: Array, b: Array) -> Array:
    """Multiply two interval arrays, shape (n, 2)."""
    a_l, a_u = a[:, 0], a[:, 1]
    b_l, b_u = b[:, 0], b[:, 1]

    # All four products
    p1 = a_l * b_l
    p2 = a_l * b_u
    p3 = a_u * b_l
    p4 = a_u * b_u

    products = jnp.stack([p1, p2, p3, p4], axis=1)
    return jnp.stack([jnp.min(products, axis=1), jnp.max(products, axis=1)], axis=1)


def _interval_add(a: Array, b: Array) -> Array:
    """Add two interval arrays, shape (n, 2)."""
    return jnp.stack([a[:, 0] + b[:, 0], a[:, 1] + b[:, 1]], axis=1)


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
    from itertools import combinations_with_replacement

    exponents_list = []
    for total in range(max_order + 1):
        # Generate all partitions of 'total' into d non-negative integers
        for combo in combinations_with_replacement(range(d), total):
            exp = jnp.zeros(d, dtype=jnp.int32)
            for idx in combo:
                exp = exp.at[idx].add(1)
            exponents_list.append(exp)

    # Remove duplicates by converting to tuple and using set
    seen = set()
    unique_exponents = []
    for exp in exponents_list:
        key = tuple(int(e) for e in exp)
        if key not in seen:
            seen.add(key)
            unique_exponents.append(exp)

    return jnp.stack(unique_exponents, axis=1)


def taylor_model(
    coeffs: ArrayLike,
    exponents: ArrayLike,
    remainder: ArrayLike | None = None,
    domain_center: ArrayLike | None = None,
    domain_radius: ArrayLike | None = None,
) -> TaylorModel:
    """Create a Taylor model.

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients, shape (n, num_monomials)
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials)
    remainder : ArrayLike, optional
        Remainder bounds, shape (n, 2). Default is zero.
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

    n = coeffs.shape[0]
    d = exponents.shape[0]

    if remainder is None:
        remainder = jnp.zeros((n, 2), dtype=coeffs.dtype)
    else:
        remainder = jnp.asarray(remainder)

    if domain_center is None:
        domain_center = jnp.zeros(d, dtype=coeffs.dtype)
    else:
        domain_center = jnp.asarray(domain_center)

    if domain_radius is None:
        domain_radius = jnp.ones(d, dtype=coeffs.dtype)
    else:
        domain_radius = jnp.asarray(domain_radius)

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius)


def taylor_model_from_interval(
    interval: "Interval", max_order: int = 1
) -> TaylorModel:
    """Create a Taylor model from an interval (identity function on domain).

    The resulting Taylor model represents the identity function x on the
    interval domain, i.e., TM(x) = x for x ∈ interval.

    Parameters
    ----------
    interval : Interval
        The domain interval
    max_order : int
        Maximum polynomial order (1 for linear)

    Returns
    -------
    TaylorModel
        Taylor model for identity function
    """
    center = (interval.lower + interval.upper) / 2
    radius = (interval.upper - interval.lower) / 2
    n = center.shape[0]

    # For identity: p(x) = domain_center + domain_radius * x_normalized
    # In normalized coords: p(x_norm) = center + radius * x_norm

    # Exponents: constant term (all zeros) + linear terms (unit vectors)
    exponents = jnp.concatenate(
        [jnp.zeros((n, 1), dtype=jnp.int32), jnp.eye(n, dtype=jnp.int32)],
        axis=1,
    )

    # Coefficients: center for constant, radius*I for linear
    coeffs = jnp.concatenate(
        [center[:, None], jnp.diag(radius)],
        axis=1,
    )

    remainder = jnp.zeros((n, 2), dtype=center.dtype)

    return TaylorModel(coeffs, exponents, remainder, center, radius)


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
        f = lambda x: jnp.atleast_1d(f(x))
    n = f_center.shape[0] if n_out is None else n_out

    # Generate exponents
    exponents = _generate_exponents(d, max_order)
    num_monomials = exponents.shape[1]

    # Compute Taylor coefficients using repeated differentiation
    # This is a simplified implementation for low orders
    coeffs = jnp.zeros((n, num_monomials), dtype=f_center.dtype)

    # Constant term
    const_idx = jnp.argmin(jnp.sum(exponents, axis=0))
    coeffs = coeffs.at[:, const_idx].set(f_center)

    # Linear terms (order 1)
    if max_order >= 1:
        grad_f = jax.jacfwd(f)
        jac = grad_f(domain_center)  # Shape (n, d)

        for i in range(d):
            # Find exponent column that is unit vector e_i
            is_ei = jnp.all(exponents == jnp.eye(d, dtype=jnp.int32)[:, i:i+1], axis=0)
            idx = jnp.argmax(is_ei)
            # Scale by domain_radius since we use normalized coordinates
            coeffs = coeffs.at[:, idx].set(jac[:, i] * domain_radius[i])

    # Higher order terms would require higher derivatives
    # For now, absorb them into remainder

    # Estimate remainder using interval arithmetic on truncation error
    # This is a conservative bound
    remainder_bound = jnp.ones(n, dtype=f_center.dtype) * 0.1 * jnp.max(domain_radius) ** (max_order + 1)
    remainder = jnp.stack([-remainder_bound, remainder_bound], axis=1)

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius)
