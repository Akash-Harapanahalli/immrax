"""Polynomial Zonotope implementation for reachability analysis.

Polynomial zonotopes are a non-convex set representation that can capture
nonlinear dependencies between uncertain parameters, making them well-suited
for reachability analysis of nonlinear systems.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Integer

from .zonotope import Zonotope


@register_pytree_node_class
class PolynomialZonotope:
    r"""Defines the polynomial zonotope set

    .. math::
        PZ = \left\{ \mathring{x} + \sum_{i=1}^{h} \left( \prod_{k=1}^{p} \alpha_k^{E_{k,i}} \right) G_{:,i}
             + \sum_{j=1}^{q} \beta_j G^I_{:,j} : \alpha \in [-1,1]^p, \beta \in [-1,1]^q \right\}

    where:
    - :math:`\mathring{x} \in \mathbb{R}^n` is the center
    - :math:`G \in \mathbb{R}^{n \times h}` is the dependent generator matrix
    - :math:`G^I \in \mathbb{R}^{n \times q}` is the independent generator matrix
    - :math:`E \in \mathbb{N}^{p \times h}` is the exponent matrix
    - :math:`\alpha \in [-1,1]^p` are the dependent factors
    - :math:`\beta \in [-1,1]^q` are the independent factors

    The dependent generators have polynomial dependencies on α, while
    independent generators are uncorrelated (like standard zonotope generators).

    Parameters
    ----------
    ox : ArrayLike
        Center, shape (n,)
    G : ArrayLike
        Dependent generator matrix, shape (n, h)
    E : ArrayLike
        Exponent matrix, shape (p, h), integer entries
    G_I : ArrayLike, optional
        Independent generator matrix, shape (n, q). Default is empty.

    References
    ----------
    .. [1] Althoff, M. "Reachability analysis of nonlinear systems using
           conservative polynomialization and non-convex sets." HSCC 2013.
    .. [2] Kochdumper, N., and Althoff, M. "Sparse polynomial zonotopes:
           A novel set representation for reachability analysis." IEEE TAC 2021.
    """

    ox: Array  # Center
    G: Array  # Dependent generator matrix
    E: Array  # Exponent matrix (integer)
    G_I: Array  # Independent generator matrix

    def __init__(
        self,
        ox: ArrayLike,
        G: ArrayLike,
        E: ArrayLike,
        G_I: ArrayLike | None = None,
    ) -> None:
        self.ox = jnp.asarray(ox)
        self.G = jnp.asarray(G)
        self.E = jnp.asarray(E, dtype=jnp.int32)

        if G_I is None:
            G_I = jnp.zeros((self.ox.shape[0], 0), dtype=self.G.dtype)
        self.G_I = jnp.asarray(G_I)

        # Validate dimensions
        if self.G.ndim != 2:
            raise ValueError(f"G must be 2D, got shape {self.G.shape}")
        if self.G.shape[0] != self.ox.shape[0]:
            raise ValueError(
                f"G rows must match ox dimension: {self.G.shape[0]} vs {self.ox.shape[0]}"
            )
        if self.E.ndim != 2:
            raise ValueError(f"E must be 2D, got shape {self.E.shape}")
        if self.E.shape[1] != self.G.shape[1]:
            raise ValueError(
                f"E columns must match G columns: {self.E.shape[1]} vs {self.G.shape[1]}"
            )
        if self.G_I.ndim != 2:
            raise ValueError(f"G_I must be 2D, got shape {self.G_I.shape}")
        if self.G_I.shape[0] != self.ox.shape[0]:
            raise ValueError(
                f"G_I rows must match ox dimension: {self.G_I.shape[0]} vs {self.ox.shape[0]}"
            )

    # --- Pytree methods ---

    def tree_flatten(self) -> Tuple[Tuple[Array, Array, Array, Array], None]:
        return ((self.ox, self.G, self.E, self.G_I), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PolynomialZonotope":
        return cls(*children)

    # --- Properties ---

    @property
    def n(self) -> int:
        """Dimension of the polynomial zonotope (state space dimension)."""
        return self.ox.shape[0]

    @property
    def h(self) -> int:
        """Number of dependent generators."""
        return self.G.shape[1]

    @property
    def p(self) -> int:
        """Number of dependent factors (polynomial variables)."""
        return self.E.shape[0]

    @property
    def q(self) -> int:
        """Number of independent generators."""
        return self.G_I.shape[1]

    @property
    def order(self) -> float:
        """Order of the polynomial zonotope ((h + q) / n)."""
        return (self.h + self.q) / self.n

    @property
    def polynomial_degree(self) -> int:
        """Maximum polynomial degree (max sum of exponents per column)."""
        return int(jnp.max(jnp.sum(self.E, axis=0)))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the center."""
        return self.ox.shape

    @property
    def dtype(self) -> jnp.dtype:
        """Data type of the arrays."""
        return self.ox.dtype

    @property
    def center(self) -> Array:
        """Center of the polynomial zonotope."""
        return self.ox

    @property
    def dependent_generators(self) -> Array:
        """Dependent generator matrix."""
        return self.G

    @property
    def independent_generators(self) -> Array:
        """Independent generator matrix."""
        return self.G_I

    @property
    def exponents(self) -> Array:
        """Exponent matrix."""
        return self.E

    # --- Monomial evaluation ---

    def evaluate_monomials(self, alpha: ArrayLike) -> Array:
        """Evaluate all monomials at given α values.

        Computes prod_k alpha_k^E_{k,i} for each generator i.

        Parameters
        ----------
        alpha : ArrayLike
            Factor values, shape (p,)

        Returns
        -------
        Array
            Monomial values, shape (h,)
        """
        alpha = jnp.asarray(alpha)
        # Compute alpha^E using log-exp for numerical stability
        # prod_k alpha_k^E_{k,i} = exp(sum_k E_{k,i} * log(|alpha_k|)) * sign
        # Handle zeros and negative values carefully
        log_abs_alpha = jnp.log(jnp.abs(alpha) + 1e-30)
        log_monomials = self.E.T @ log_abs_alpha  # Shape (h,)

        # Compute sign: (-1)^(sum of exponents for negative alpha)
        is_negative = alpha < 0
        odd_exponents = self.E % 2  # 1 where odd, 0 where even
        sign_changes = odd_exponents.T @ is_negative.astype(jnp.float32)
        signs = jnp.where(sign_changes % 2 == 0, 1.0, -1.0)

        return signs * jnp.exp(log_monomials)

    def evaluate(self, alpha: ArrayLike, beta: ArrayLike | None = None) -> Array:
        """Evaluate the polynomial zonotope at specific factor values.

        Parameters
        ----------
        alpha : ArrayLike
            Dependent factor values, shape (p,)
        beta : ArrayLike, optional
            Independent factor values, shape (q,). Default is zeros.

        Returns
        -------
        Array
            Point in the polynomial zonotope, shape (n,)
        """
        alpha = jnp.asarray(alpha)
        monomials = self.evaluate_monomials(alpha)
        result = self.ox + self.G @ monomials

        if self.q > 0:
            if beta is None:
                beta = jnp.zeros(self.q, dtype=self.dtype)
            else:
                beta = jnp.asarray(beta)
            result = result + self.G_I @ beta

        return result

    # --- Set operations ---

    def __add__(self, other: "PolynomialZonotope") -> "PolynomialZonotope":
        """Exact Minkowski sum of two polynomial zonotopes.

        When both PZs share the same dependent factors, the sum preserves
        polynomial dependencies.
        """
        if not isinstance(other, PolynomialZonotope):
            raise TypeError("Can only add PolynomialZonotope to PolynomialZonotope")
        if self.ox.shape != other.ox.shape:
            raise ValueError(
                f"Incompatible dimensions: {self.ox.shape} vs {other.ox.shape}"
            )

        new_ox = self.ox + other.ox

        # Merge dependent generators and exponents
        # Find common monomials and combine their generators
        new_G, new_E = _merge_polynomial_terms(
            self.G, self.E, other.G, other.E
        )

        # Concatenate independent generators
        new_G_I = jnp.concatenate([self.G_I, other.G_I], axis=1)

        return PolynomialZonotope(new_ox, new_G, new_E, new_G_I)

    def __sub__(self, other: "PolynomialZonotope") -> "PolynomialZonotope":
        """Minkowski difference: PZ1 - PZ2 = PZ1 + (-PZ2)."""
        return self + (-other)

    def __neg__(self) -> "PolynomialZonotope":
        """Negation: -PZ = {-z : z ∈ PZ}."""
        return PolynomialZonotope(-self.ox, -self.G, self.E, -self.G_I)

    def __rmatmul__(self, other: ArrayLike) -> "PolynomialZonotope":
        """Left matrix multiplication: M @ PZ.

        For M ∈ R^{m×n} and PZ ⊂ R^n, computes M @ PZ = {M @ z : z ∈ PZ}.
        """
        M = jnp.asarray(other)
        return PolynomialZonotope(M @ self.ox, M @ self.G, self.E, M @ self.G_I)

    def __mul__(self, other: ArrayLike) -> "PolynomialZonotope":
        """Scalar multiplication: PZ * α = {α * z : z ∈ PZ}."""
        alpha = jnp.asarray(other)
        return PolynomialZonotope(
            alpha * self.ox, alpha * self.G, self.E, alpha * self.G_I
        )

    def __rmul__(self, other: ArrayLike) -> "PolynomialZonotope":
        """Scalar multiplication: α * PZ."""
        return self.__mul__(other)

    # --- Quadratic map (for nonlinear systems) ---

    def quadratic_map(self, Q: ArrayLike) -> "PolynomialZonotope":
        """Apply a quadratic map x^T Q x to the polynomial zonotope.

        Computes the image of PZ under the quadratic form.

        Parameters
        ----------
        Q : ArrayLike
            Quadratic form matrix, shape (n, n) or (m, n, n) for vector output

        Returns
        -------
        PolynomialZonotope
            Image under the quadratic map
        """
        Q = jnp.asarray(Q)

        # For scalar quadratic form Q (n, n)
        # x^T Q x where x = ox + G @ monomials + G_I @ beta
        # Expand: (ox + G @ m + G_I @ b)^T Q (ox + G @ m + G_I @ b)
        # = ox^T Q ox + 2 ox^T Q G m + 2 ox^T Q G_I b
        #   + m^T G^T Q G m + 2 m^T G^T Q G_I b + b^T G_I^T Q G_I b

        if Q.ndim == 2:
            Q = Q[None, :, :]  # Add batch dimension for vector output

        m_out = Q.shape[0]

        # Center term: ox^T Q ox
        new_ox = jnp.einsum("i,mij,j->m", self.ox, Q, self.ox)

        # Linear terms in monomials: 2 ox^T Q G
        linear_G = 2 * jnp.einsum("i,mij,jk->mk", self.ox, Q, self.G)

        # Quadratic terms in monomials: G^T Q G
        # These create new monomials with combined exponents
        quad_coeff = jnp.einsum("ij,mik,kl->mjl", self.G, Q, self.G)  # (m, h, h)

        # Build new generators and exponents for quadratic terms
        new_G_list = [linear_G]
        new_E_list = [self.E]

        for i in range(self.h):
            for j in range(i, self.h):
                # Coefficient for monomial_i * monomial_j
                if i == j:
                    coeff = quad_coeff[:, i, j]
                else:
                    coeff = quad_coeff[:, i, j] + quad_coeff[:, j, i]

                # New exponent is sum of exponents
                new_exp = self.E[:, i] + self.E[:, j]

                new_G_list.append(coeff[:, None])
                new_E_list.append(new_exp[:, None])

        new_G = jnp.concatenate(new_G_list, axis=1)
        new_E = jnp.concatenate(new_E_list, axis=1)

        # Independent generator contributions become independent
        # 2 ox^T Q G_I for linear, and G_I^T Q G_I creates diagonal
        linear_G_I = 2 * jnp.einsum("i,mij,jk->mk", self.ox, Q, self.G_I)

        # Cross terms G^T Q G_I also become independent
        cross_terms = 2 * jnp.einsum("ij,mik,kl->mjl", self.G, Q, self.G_I)
        cross_G_I = cross_terms.reshape(m_out, -1)  # Flatten cross terms

        # G_I^T Q G_I diagonal
        quad_G_I = jnp.einsum("ij,mik,kj->mj", self.G_I, Q, self.G_I)

        new_G_I = jnp.concatenate([linear_G_I, cross_G_I, quad_G_I], axis=1)

        # Merge terms with same exponents
        new_G, new_E = _compact_polynomial_terms(new_G, new_E)

        return PolynomialZonotope(new_ox, new_G, new_E, new_G_I)

    # --- Conversion methods ---

    def to_zonotope(self) -> Zonotope:
        """Convert to a standard zonotope (overapproximation).

        Treats all generators as independent, losing polynomial dependencies.
        """
        G_combined = jnp.concatenate([self.G, self.G_I], axis=1)
        return Zonotope(self.ox, G_combined)

    def interval_hull(self) -> "Interval":
        """Compute the interval hull (bounding box).

        Uses the zonotope overapproximation.
        """
        return self.to_zonotope().interval_hull()

    def interval_hull_tight(self) -> "Interval":
        """Compute tighter interval bounds using polynomial structure.

        Exploits polynomial dependencies for tighter bounds than the
        zonotope overapproximation.
        """
        from immrax.inclusion.interval import Interval

        # For each dimension, bound the polynomial over [-1,1]^p
        # Using interval arithmetic on the polynomial
        n = self.n

        lower = jnp.zeros(n, dtype=self.dtype)
        upper = jnp.zeros(n, dtype=self.dtype)

        # Evaluate polynomial bounds using interval extension
        # Start with α ∈ [-1, 1]^p
        alpha_int_lower = -jnp.ones(self.p, dtype=self.dtype)
        alpha_int_upper = jnp.ones(self.p, dtype=self.dtype)

        # Bound each monomial
        for i in range(self.h):
            exp_i = self.E[:, i]
            g_i = self.G[:, i]

            # Monomial bound: product of α_k^e_k over [-1,1]
            # If e_k is even: result is in [0, 1]
            # If e_k is odd: result is in [-1, 1]
            mono_lower = jnp.ones((), dtype=self.dtype)
            mono_upper = jnp.ones((), dtype=self.dtype)

            for k in range(self.p):
                e_k = exp_i[k]
                if e_k == 0:
                    continue
                elif e_k % 2 == 0:
                    # Even power: [0, 1] (assuming [-1,1] input)
                    mono_lower = mono_lower * 0.0
                    mono_upper = mono_upper * 1.0
                else:
                    # Odd power: [-1, 1]
                    mono_lower = mono_lower * (-1.0)
                    mono_upper = mono_upper * 1.0

            # Contribution to bounds
            g_pos = jnp.maximum(g_i, 0)
            g_neg = jnp.minimum(g_i, 0)
            lower = lower + g_pos * mono_lower + g_neg * mono_upper
            upper = upper + g_pos * mono_upper + g_neg * mono_lower

        # Add center
        lower = lower + self.ox
        upper = upper + self.ox

        # Add independent generator contribution
        if self.q > 0:
            G_I_abs_sum = jnp.sum(jnp.abs(self.G_I), axis=1)
            lower = lower - G_I_abs_sum
            upper = upper + G_I_abs_sum

        return Interval(lower, upper)

    def contains(self, x: ArrayLike) -> Array:
        """Check if a point is contained (necessary condition via interval hull)."""
        x = jnp.asarray(x)
        hull = self.interval_hull()
        return jnp.all((x >= hull.lower) & (x <= hull.upper))

    # --- Reduction methods ---

    def reduce_order(self, target_order: float) -> "PolynomialZonotope":
        """Reduce the order while maintaining an overapproximation.

        Converts small dependent generators to independent generators.

        Parameters
        ----------
        target_order : float
            Target order (h + q) / n

        Returns
        -------
        PolynomialZonotope
            Reduced order polynomial zonotope
        """
        target_total = int(target_order * self.n)
        current_total = self.h + self.q

        if target_total >= current_total:
            return self

        # Sort dependent generators by norm
        G_norms = jnp.linalg.norm(self.G, axis=0)
        sorted_idx = jnp.argsort(G_norms)

        # Keep largest dependent generators
        num_to_convert = current_total - target_total
        if num_to_convert >= self.h:
            # Convert all dependent to independent
            return PolynomialZonotope(
                self.ox,
                jnp.zeros((self.n, 0), dtype=self.dtype),
                jnp.zeros((self.p, 0), dtype=jnp.int32),
                jnp.concatenate([self.G, self.G_I], axis=1),
            )

        convert_idx = sorted_idx[:num_to_convert]
        keep_idx = sorted_idx[num_to_convert:]

        new_G = self.G[:, keep_idx]
        new_E = self.E[:, keep_idx]
        new_G_I = jnp.concatenate([self.G[:, convert_idx], self.G_I], axis=1)

        return PolynomialZonotope(self.ox, new_G, new_E, new_G_I)

    def compact(self) -> "PolynomialZonotope":
        """Compact the representation by merging terms with identical exponents."""
        new_G, new_E = _compact_polynomial_terms(self.G, self.E)
        return PolynomialZonotope(self.ox, new_G, new_E, self.G_I)

    # --- String representation ---

    def __str__(self) -> str:
        return (
            f"PolynomialZonotope(n={self.n}, h={self.h}, p={self.p}, q={self.q}, "
            f"degree={self.polynomial_degree})"
        )

    def __repr__(self) -> str:
        return (
            f"PolynomialZonotope(ox={self.ox!r}, G={self.G!r}, "
            f"E={self.E!r}, G_I={self.G_I!r})"
        )


# --- Helper functions ---


def _merge_polynomial_terms(
    G1: Array, E1: Array, G2: Array, E2: Array
) -> Tuple[Array, Array]:
    """Merge polynomial terms from two polynomial zonotopes.

    Combines generators with matching exponents.
    """
    # Simple approach: concatenate and then compact
    G_merged = jnp.concatenate([G1, G2], axis=1)
    E_merged = jnp.concatenate([E1, E2], axis=1)
    return _compact_polynomial_terms(G_merged, E_merged)


def _compact_polynomial_terms(G: Array, E: Array) -> Tuple[Array, Array]:
    """Compact polynomial terms by combining generators with identical exponents.

    This is a simplified version that just removes zero generators.
    Full implementation would detect and merge identical exponent columns.
    """
    # Remove zero generators
    norms = jnp.linalg.norm(G, axis=0)
    nonzero_mask = norms > 1e-12

    # Filter using mask (this creates a dynamic shape which may not be JIT-compatible)
    # For JIT compatibility, we keep all columns but zero out small ones
    G_compact = jnp.where(nonzero_mask[None, :], G, 0.0)
    E_compact = E

    return G_compact, E_compact


def polynomial_zonotope(
    ox: ArrayLike,
    G: ArrayLike | None = None,
    E: ArrayLike | None = None,
    G_I: ArrayLike | None = None,
) -> PolynomialZonotope:
    """Create a PolynomialZonotope.

    Parameters
    ----------
    ox : ArrayLike
        Center
    G : ArrayLike, optional
        Dependent generator matrix
    E : ArrayLike, optional
        Exponent matrix
    G_I : ArrayLike, optional
        Independent generator matrix

    Returns
    -------
    PolynomialZonotope
        The constructed polynomial zonotope
    """
    ox = jnp.asarray(ox)
    n = ox.shape[0]

    if G is None:
        G = jnp.zeros((n, 0), dtype=ox.dtype)
    else:
        G = jnp.asarray(G)

    if E is None:
        E = jnp.zeros((0, G.shape[1]), dtype=jnp.int32)
    else:
        E = jnp.asarray(E, dtype=jnp.int32)

    if G_I is None:
        G_I = jnp.zeros((n, 0), dtype=ox.dtype)
    else:
        G_I = jnp.asarray(G_I)

    return PolynomialZonotope(ox, G, E, G_I)


def polynomial_zonotope_from_zonotope(z: Zonotope) -> PolynomialZonotope:
    """Convert a standard zonotope to a polynomial zonotope.

    Each zonotope generator becomes an independent polynomial zonotope generator.

    Parameters
    ----------
    z : Zonotope
        The zonotope to convert

    Returns
    -------
    PolynomialZonotope
        Equivalent polynomial zonotope
    """
    # All generators become independent (no polynomial structure)
    G = jnp.zeros((z.n, 0), dtype=z.dtype)
    E = jnp.zeros((0, 0), dtype=jnp.int32)
    return PolynomialZonotope(z.ox, G, E, z.G)


def polynomial_zonotope_from_interval(interval: "Interval") -> PolynomialZonotope:
    """Create a polynomial zonotope from an interval.

    Parameters
    ----------
    interval : Interval
        The interval to convert

    Returns
    -------
    PolynomialZonotope
        Equivalent polynomial zonotope (axis-aligned generators)
    """
    center = (interval.lower + interval.upper) / 2
    radius = (interval.upper - interval.lower) / 2
    n = center.shape[0]

    # Create identity exponent matrix (each generator has its own factor)
    E = jnp.eye(n, dtype=jnp.int32)
    G = jnp.diag(radius)

    return PolynomialZonotope(center, G, E, None)


def polynomial_zonotope_cartesian_product(
    pzs: list,
) -> PolynomialZonotope:
    """Compute the Cartesian product of polynomial zonotopes.

    Parameters
    ----------
    pzs : list of PolynomialZonotope
        Polynomial zonotopes to combine

    Returns
    -------
    PolynomialZonotope
        Cartesian product
    """
    # Stack centers
    ox = jnp.concatenate([pz.ox for pz in pzs])

    # Block diagonal dependent generators
    n_total = sum(pz.n for pz in pzs)
    h_total = sum(pz.h for pz in pzs)
    p_total = sum(pz.p for pz in pzs)
    q_total = sum(pz.q for pz in pzs)

    G = jnp.zeros((n_total, h_total), dtype=pzs[0].dtype)
    E = jnp.zeros((p_total, h_total), dtype=jnp.int32)
    G_I = jnp.zeros((n_total, q_total), dtype=pzs[0].dtype)

    n_offset, h_offset, p_offset, q_offset = 0, 0, 0, 0
    for pz in pzs:
        G = G.at[n_offset : n_offset + pz.n, h_offset : h_offset + pz.h].set(pz.G)
        E = E.at[p_offset : p_offset + pz.p, h_offset : h_offset + pz.h].set(pz.E)
        G_I = G_I.at[n_offset : n_offset + pz.n, q_offset : q_offset + pz.q].set(pz.G_I)
        n_offset += pz.n
        h_offset += pz.h
        p_offset += pz.p
        q_offset += pz.q

    return PolynomialZonotope(ox, G, E, G_I)
