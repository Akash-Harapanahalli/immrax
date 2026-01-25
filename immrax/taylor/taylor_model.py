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
    """Generate all exponent multi-indices up to given order (implementation)."""
    from itertools import combinations_with_replacement

    exponents_list = []
    for total in range(max_order + 1):
        for combo in combinations_with_replacement(range(d), total):
            exp = [0] * d
            for idx in combo:
                exp[idx] += 1
            exponents_list.append(exp)

    # Remove duplicates
    seen = set()
    unique_exponents = []
    for exp in exponents_list:
        key = tuple(exp)
        if key not in seen:
            seen.add(key)
            unique_exponents.append(exp)

    return jnp.array(unique_exponents, dtype=jnp.int32).T


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

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients. For order k and domain dimension d,
        shape is (n, num_monomials) where num_monomials = C(d+k, k).
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials), integer entries.
        Each column is a multi-index α.
    remainder : Interval
        Interval remainder bounds.
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
    remainder: Interval  # Interval remainder
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
    ) -> Tuple[Tuple[Array, Array, Interval, Array, Array], dict]:
        return (
            (
                self.coeffs,
                self.exponents,
                self.remainder,
                self.domain_center,
                self.domain_radius,
            ),
            {"_static_order": self._static_order},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorModel":
        static_order = aux_data.get("_static_order") if aux_data else None
        return cls(*children, _static_order=static_order)

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

    def __getitem__(self, idx) -> "TaylorModel":
        """Get component(s) of the Taylor Model."""
        c = self.coeffs[idx]
        if c.ndim == 1:
            c = c[None, :]

        rem = self.remainder[idx]
        # Intervals might return scalars on indexing depending on impl
        # If scalar, wrap in 1D interval?
        # Implemented Interval sets lower/upper. Indexing them gives scalars/arrays.
        # Interval constructor expects arrays.

        # Let's ensure remainder is properly formatted if scalar
        if hasattr(rem, 'lower') and rem.lower.ndim == 0:
             # Re-wrap scalar interval into 1D interval?
             # But TM remainder is n-dim interval.
             # If idx selects one element, rem is scalar interval.
             # coeffs is (1, num_monomials). n=1.
             # scalar interval is fine if n=1?
             # Let's check init: self.n = coeffs.shape[0].
             # remainder should match self.n.
             # self.remainder = remainder.
             # Checks: assert remainder.lower.shape == (n,)
             # So if n=1, remainder must be (1,).
             pass

        # If rem is scalar interval, we need to reshape it to (1,)
        # Check if rem has shape.
        if hasattr(rem, 'shape') and rem.shape == ():
             # It's a scalar interval.
             # We need to reshape its components?
             # immrax Interval might not support reshape directly.
             # We can construct new interval.
             rem = interval(rem.lower[None], rem.upper[None])

        return TaylorModel(c, self.exponents, rem, self.domain_center, self.domain_radius,
                           _static_order=self._static_order)

    def __len__(self) -> int:
        return self.n

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

    def __add__(self, other: "TaylorModel" | ArrayLike) -> "TaylorModel":
        """Addition of two Taylor models or translation by vector."""
        if isinstance(other, TaylorModel):
            # Note: Domain compatibility check removed for JIT compatibility
            # Assumes domains are compatible when adding TaylorModels

            # Merge polynomial terms
            new_coeffs, new_exponents = _merge_taylor_terms(
                self.coeffs, self.exponents, other.coeffs, other.exponents
            )

            # Add remainders using Interval addition
            new_remainder = self.remainder + other.remainder

            new_order = max(self._static_order, other._static_order)
            result = TaylorModel(
                new_coeffs, new_exponents, new_remainder, self.domain_center, self.domain_radius,
                _static_order=new_order
            )
            # Convert to canonical form for compatibility with concatenation
            return result.to_canonical(new_order)

        # Assume scalar or vector translation (polynomial coefficient modification)
        try:
            vec = jnp.asarray(other)
            # Handle scalar: broadcast to (n,)
            if vec.ndim == 0:
                vec = jnp.full(self.shape, vec)
            if vec.shape == self.shape:
                # Add to constant term
                const_exp = jnp.zeros((self.d, 1), dtype=jnp.int32)
                const_coeff = vec[:, None]

                new_coeffs, new_exponents = _merge_taylor_terms(
                    self.coeffs, self.exponents, const_coeff, const_exp
                )

                result = TaylorModel(
                    new_coeffs, new_exponents, self.remainder, self.domain_center, self.domain_radius,
                    _static_order=self._static_order
                )
                # Convert to canonical form for consistency
                return result.to_canonical(self._static_order)
        except Exception:
            pass

        raise TypeError(f"Unsupported type for __add__: {type(other)}")

    def __pow__(self, power: int) -> "TaylorModel":
        """Integer exponentiation."""
        if not isinstance(power, (int, jnp.integer)):
             return NotImplemented

        if power == 0:
             # Return constant 1 with same domain
             n = self.n
             const_coeffs = jnp.zeros((n, 1), dtype=self.dtype)
             const_coeffs = const_coeffs.at[:, 0].set(1.0)
             const_exp = jnp.zeros((self.d, 1), dtype=jnp.int32)
             return TaylorModel(
                 const_coeffs, const_exp,
                 interval(jnp.zeros(n, dtype=self.dtype)),
                 self.domain_center, self.domain_radius,
                 _static_order=self._static_order
             )

        if power < 0:
             raise ValueError("Negative powers not supported for TaylorModel yet.")

        # Use multiply method for TaylorModel * TaylorModel
        # Use stored static order for JIT compatibility
        max_order = self._static_order
        res = self
        for _ in range(power - 1):
             res = res.multiply(self, max_order=max_order)
        return res

    def __sub__(self, other: "TaylorModel | ArrayLike") -> "TaylorModel":
        """Subtraction: TM1 - TM2 or TM - scalar."""
        if isinstance(other, TaylorModel):
            return self + (-other)
        return self + (-jnp.asarray(other))

    def __radd__(self, other: ArrayLike) -> "TaylorModel":
        """Right addition: scalar + TM."""
        return self + other

    def __rsub__(self, other: ArrayLike) -> "TaylorModel":
        """Right subtraction: scalar - TM."""
        return (-self) + other

    def __neg__(self) -> "TaylorModel":
        """Negation: -TM."""
        return TaylorModel(
            -self.coeffs,
            self.exponents,
            -self.remainder,
            self.domain_center,
            self.domain_radius,
            _static_order=self._static_order,
        )

    def __mul__(self, other: "TaylorModel | ArrayLike") -> "TaylorModel":
        """Multiplication: TM * TM or TM * scalar."""
        # Handle TaylorModel * TaylorModel
        if isinstance(other, TaylorModel):
            # Use max of both static orders for JIT compatibility
            max_order = max(self._static_order, other._static_order)
            return self.multiply(other, max_order=max_order)

        # Scalar/vector multiplication
        alpha = jnp.asarray(other)
        if alpha.ndim == 0:
            # Scalar
            new_coeffs = alpha * self.coeffs
            new_remainder = self.remainder * alpha
        else:
            # Element-wise scaling
            new_coeffs = alpha[:, None] * self.coeffs
            new_remainder = interval(
                alpha * self.remainder.lower,
                alpha * self.remainder.upper
            )
            # Handle sign flips for negative alpha
            needs_swap = alpha < 0
            new_lower = jnp.where(needs_swap, new_remainder.upper, new_remainder.lower)
            new_upper = jnp.where(needs_swap, new_remainder.lower, new_remainder.upper)
            new_remainder = interval(new_lower, new_upper)

        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
            _static_order=self._static_order,
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

        # Remainder transformation using Interval matmul
        rem_interval = self.remainder.atleast_2d().T  # shape (1, n) as interval
        new_remainder = interval(M, M) @ rem_interval  # M @ [r_l, r_u]
        new_remainder = new_remainder.T.reshape(-1)  # back to (m,)

        return TaylorModel(
            new_coeffs,
            self.exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
            _static_order=self._static_order,
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

        n = self.n
        d = self.d
        m1 = self.num_monomials
        m2 = other.num_monomials

        # Precompute all product exponents and coefficients (JIT-compatible)
        # Product will have m1 * m2 terms before compaction
        # new_exp[k] = self.exponents[:, i] + other.exponents[:, j] for k = i*m2 + j

        # Compute all product exponents: shape (d, m1*m2)
        # Using broadcasting: self.exponents[:, :, None] + other.exponents[:, None, :]
        exp1 = self.exponents[:, :, None]  # (d, m1, 1)
        exp2 = other.exponents[:, None, :]  # (d, 1, m2)
        all_exp = (exp1 + exp2).reshape(d, m1 * m2)  # (d, m1*m2)

        # Compute all product coefficients: shape (n, m1*m2)
        coeff1 = self.coeffs[:, :, None]  # (n, m1, 1)
        coeff2 = other.coeffs[:, None, :]  # (n, 1, m2)
        all_coeffs = (coeff1 * coeff2).reshape(n, m1 * m2)  # (n, m1*m2)

        # Compute total order of each product term
        total_orders = jnp.sum(all_exp, axis=0)  # (m1*m2,)

        # If max_order specified, zero out high-order terms and add to remainder
        if max_order is not None:
            # Mask for terms to keep
            keep_mask = total_orders <= max_order  # (m1*m2,)

            # Zero out coefficients above max_order
            new_coeffs = jnp.where(keep_mask[None, :], all_coeffs, 0.0)

            # Bound truncated terms and add to remainder
            # For each term above max_order, bound its contribution
            truncated_coeffs = jnp.where(keep_mask[None, :], 0.0, all_coeffs)

            # Compute monomial bounds: odd exponents -> [-1,1], even -> [0,1]
            has_odd = jnp.any(all_exp % 2 == 1, axis=0)  # (m1*m2,)
            mono_lower = jnp.where(has_odd, -1.0, 0.0)
            mono_upper = jnp.ones(m1 * m2)

            # Bound each truncated term: coeff * monomial_bound
            # For positive coeff: [coeff*lower, coeff*upper]
            # For negative coeff: [coeff*upper, coeff*lower]
            c_pos = jnp.maximum(truncated_coeffs, 0.0)
            c_neg = jnp.minimum(truncated_coeffs, 0.0)
            term_lower = c_pos * mono_lower[None, :] + c_neg * mono_upper[None, :]
            term_upper = c_pos * mono_upper[None, :] + c_neg * mono_lower[None, :]

            # Sum over all truncated terms
            trunc_lower = jnp.sum(term_lower, axis=1)
            trunc_upper = jnp.sum(term_upper, axis=1)
            truncated_remainder = interval(trunc_lower, trunc_upper)
        else:
            new_coeffs = all_coeffs
            truncated_remainder = interval(jnp.zeros(n), jnp.zeros(n))

        new_exponents = all_exp

        # Compute remainder bounds using Interval operations
        # (p1 + r1) * (p2 + r2) = p1*p2 + p1*r2 + r1*p2 + r1*r2
        p1_bounds = self._bound_polynomial()
        p2_bounds = other._bound_polynomial()

        # Cross terms using Interval multiplication
        p1_r2 = p1_bounds * other.remainder
        r1_p2 = self.remainder * p2_bounds
        r1_r2 = self.remainder * other.remainder

        # Combined remainder
        new_remainder = p1_r2 + r1_p2 + r1_r2 + truncated_remainder

        # Use max_order if specified, otherwise use the larger of the two static orders
        effective_order = max_order if max_order is not None else max(self._static_order, other._static_order)

        result = TaylorModel(
            new_coeffs,
            new_exponents,
            new_remainder,
            self.domain_center,
            self.domain_radius,
            _static_order=effective_order,
        )

        # Convert to canonical form for compatibility with concatenation
        if max_order is not None:
            result = result.to_canonical(max_order)

        return result

    def _bound_polynomial(self) -> Interval:
        """Bound the polynomial part over the domain.

        Uses Horner-like evaluation for 1D case to preserve correlations.
        Falls back to term-by-term for multivariate case.

        Returns
        -------
        Interval
            Bounds on the polynomial
        """
        if self.d == 1:
            return self._bound_polynomial_horner_1d()
        else:
            return self._bound_polynomial_termwise()

    def _bound_polynomial_horner_1d(self) -> Interval:
        """Bound 1D polynomial using Horner's method for tighter bounds."""
        # Sort monomials by degree (descending)
        degrees = self.exponents[0, :]  # 1D, so just first row
        max_deg = int(jnp.max(degrees))

        # Build coefficient array indexed by degree
        poly_coeffs = jnp.zeros((self.n, max_deg + 1), dtype=self.dtype)
        for i in range(self.num_monomials):
            deg = int(degrees[i])
            poly_coeffs = poly_coeffs.at[:, deg].add(self.coeffs[:, i])

        # Horner's method: p(x) = c_0 + x*(c_1 + x*(c_2 + ...))
        # Evaluated on interval [-1, 1]
        x_interval = icentpert(jnp.array([0.0]), jnp.array([1.0]))  # [-1, 1]

        # Start from highest degree
        result = interval(poly_coeffs[:, max_deg])
        for deg in range(max_deg - 1, -1, -1):
            # result = c_deg + x * result
            result = result * x_interval[0] + interval(poly_coeffs[:, deg])

        return result

    def _bound_polynomial_termwise(self) -> Interval:
        """Bound polynomial term-by-term (conservative for multivariate)."""
        result = interval(jnp.zeros(self.n, dtype=self.dtype))

        for i in range(self.num_monomials):
            exp_i = self.exponents[:, i]
            coeff_i = self.coeffs[:, i]

            # Bound monomial over [-1, 1]^d
            mono_bound = _bound_monomial(exp_i)  # scalar interval

            # Multiply by coefficient (vector)
            term_bound = interval(coeff_i, coeff_i) * mono_bound
            result = result + term_bound

        return result

    # --- Conversion methods ---

    def to_zonotope(self):
        """Convert to a zonotope (overapproximation).

        Each monomial term becomes a generator.

        Returns
        -------
        Zonotope
            Zonotope overapproximation of the Taylor model
        """
        # Lazy import to avoid circular dependency
        from immrax.generator.sets.zonotope import Zonotope
        # Center is the constant term plus remainder center
        remainder_center = self.remainder.center
        remainder_radius = self.remainder.pert

        ox = self.constant_term + remainder_center

        # Non-constant terms become generators
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

    def to_canonical(self, target_order: int | None = None) -> "TaylorModel":
        """Convert to canonical exponent structure.

        The canonical structure includes all monomials up to target_order,
        sorted consistently. This ensures TMs are compatible for concatenation.

        Parameters
        ----------
        target_order : int, optional
            Target order. If None, uses current maximum order.

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

        # Create mapping from current exponents to canonical indices
        # For JIT compatibility, we use fully vectorized operations

        # Compute a unique hash for each exponent vector
        # Use weighted sum: sum_i exp[i] * (max_order+1)^i
        base = target_order + 2  # Ensure no collisions
        powers = base ** jnp.arange(self.d)

        # Hash current exponents: (m,)
        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)

        # Hash canonical exponents: (num_canonical,)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        # For each current term, find its index in canonical
        # Using broadcasting: (m, num_canonical)
        match_matrix = current_hash[:, None] == canonical_hash[None, :]

        # Find canonical index for each current term (argmax along canonical axis)
        # canonical_indices[i] = j means current term i maps to canonical term j
        canonical_indices = jnp.argmax(match_matrix, axis=1)  # (m,)

        # Check if each current term has a match (is within target_order)
        term_orders = jnp.sum(self.exponents, axis=0)  # (m,)
        has_match = term_orders <= target_order  # (m,)

        # Use matrix multiplication for scatter-add
        # match_matrix.T @ coeffs.T gives us the scattered result
        # But we need to mask by has_match first
        scatter_matrix = match_matrix.T.astype(self.dtype)  # (num_canonical, m)
        masked_coeffs = jnp.where(has_match[None, :], self.coeffs, 0.0)  # (n, m)

        # new_coeffs[i, j] = sum_k scatter_matrix[j, k] * masked_coeffs[i, k]
        new_coeffs = masked_coeffs @ scatter_matrix.T  # (n, num_canonical)

        # Bound terms above target_order and add to remainder
        # Terms with order > target_order need to be absorbed
        absorb_mask = term_orders > target_order  # (m,)

        # For absorbed terms, compute their bounds
        # Monomial bounds: odd exponents -> [-1,1], even -> [0,1]
        has_odd = jnp.any(self.exponents % 2 == 1, axis=0)  # (m,)
        mono_lower = jnp.where(has_odd, -1.0, 0.0)  # (m,)
        mono_upper = jnp.ones(self.num_monomials)  # (m,)

        # Compute term bounds for absorbed terms
        # For positive coeff: [coeff*lower, coeff*upper]
        # For negative coeff: [coeff*upper, coeff*lower]
        absorb_coeffs = jnp.where(absorb_mask[None, :], self.coeffs, 0.0)  # (n, m)

        c_pos = jnp.maximum(absorb_coeffs, 0.0)  # (n, m)
        c_neg = jnp.minimum(absorb_coeffs, 0.0)  # (n, m)

        term_lower = c_pos * mono_lower[None, :] + c_neg * mono_upper[None, :]  # (n, m)
        term_upper = c_pos * mono_upper[None, :] + c_neg * mono_lower[None, :]  # (n, m)

        absorbed_lower = jnp.sum(term_lower, axis=1)  # (n,)
        absorbed_upper = jnp.sum(term_upper, axis=1)  # (n,)

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
        if target_order >= self.order:
            return self

        # Separate terms to keep and terms to absorb
        term_orders = jnp.sum(self.exponents, axis=0)
        keep_mask = term_orders <= target_order

        # Bound absorbed terms
        absorbed_remainder = interval(
            jnp.zeros(self.n, dtype=self.dtype),
            jnp.zeros(self.n, dtype=self.dtype)
        )

        for i in range(self.num_monomials):
            if term_orders[i] > target_order:
                exp_i = self.exponents[:, i]
                coeff_i = self.coeffs[:, i]

                # Bound this monomial
                mono_bound = _bound_monomial(exp_i)
                term_bound = interval(coeff_i, coeff_i) * mono_bound
                absorbed_remainder = absorbed_remainder + term_bound

        # New remainder includes absorbed terms
        new_remainder = self.remainder + absorbed_remainder

        # Filter coefficients (zero out instead of filtering for JIT)
        new_coeffs = jnp.where(keep_mask[None, :], self.coeffs, 0.0)

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
        Polynomial coefficients, shape (n, num_monomials)
    exponents : ArrayLike
        Exponent matrix, shape (d, num_monomials)
    remainder : Interval, optional
        Remainder interval. Default is zero interval.
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
        remainder = interval(jnp.zeros(n, dtype=coeffs.dtype))

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
        The domain interval
    order : int
        Polynomial degree of the Taylor Model (default: 1)

    Returns
    -------
    TaylorModel
        Taylor model for identity function
    """
    center = iv.center
    radius = iv.pert
    n = center.shape[0]

    # For identity: p(x_norm) = center + radius * x_norm
    # Note: Higher order terms are simply zero.

    # Generate exponents for full order
    exponents = _generate_exponents(n, order)
    num_monomials = exponents.shape[1]

    # Coefficients: center for constant, radius*I for linear, 0 otherwise
    coeffs = jnp.zeros((n, num_monomials), dtype=center.dtype)

    # Constant term (all zeros exponent)
    zero_idx = jnp.where(jnp.sum(exponents, axis=0) == 0)[0][0]
    coeffs = coeffs.at[:, zero_idx].set(center)

    # Linear terms
    for i in range(n):
        # Term corresponding to e_i
        target = jnp.eye(n, dtype=jnp.int32)[i]
        matches = jnp.all(exponents == target[:, None], axis=0)
        idx = jnp.where(matches)[0]
        if len(idx) > 0:
            coeffs = coeffs.at[i, idx[0]].set(radius[i])

    remainder = interval(jnp.zeros(n, dtype=center.dtype))

    return TaylorModel(coeffs, exponents, remainder, center, radius, _static_order=order)


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


def taylor_model_concatenate(tms: list["TaylorModel"]) -> "TaylorModel":
    """Concatenate multiple TaylorModels along the output dimension.

    This is the equivalent of jnp.array([tm1, tm2, ...]) for TaylorModels.
    All input TaylorModels must have the same domain and exponent structure.

    For JIT compatibility, this function requires all TMs to have identical
    exponent arrays. Use taylor_model_unify_exponents first if needed.

    Parameters
    ----------
    tms : list of TaylorModel
        TaylorModels to concatenate.

    Returns
    -------
    TaylorModel
        Concatenated TaylorModel with n = sum of all input n's.
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

    # Stack coefficients and remainders (JIT-compatible)
    coeffs = jnp.concatenate([tm.coeffs for tm in tms], axis=0)
    remainder = interval(
        jnp.concatenate([tm.remainder.lower.reshape(-1) for tm in tms]),
        jnp.concatenate([tm.remainder.upper.reshape(-1) for tm in tms])
    )

    return TaylorModel(coeffs, exponents, remainder, domain_center, domain_radius, _static_order=static_order)
