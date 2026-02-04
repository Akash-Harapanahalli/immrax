"""Taylor Polynomial implementation — polynomial arithmetic without interval remainder.

A TaylorPolynomial represents a multivariate polynomial centered at a point,
without the rigorous interval remainder of a TaylorModel. Useful for pure
polynomial manipulation where remainder tracking is not needed. High-order
terms beyond the specified order are silently discarded.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike

from immrax.inclusion import Interval, interval, icentpert
import jax.tree_util

from immrax.taylor.taylor_model import (
    TaylorModel,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _leaf_slice,
    _merge_taylor_terms,
)


@register_pytree_node_class
class TaylorPolynomial:
    r"""Multivariate polynomial centered at a point.

    .. math::
        p(x) = \sum_{|\alpha| \leq k} a_\alpha (x - x_0)^\alpha

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients, shape ``(*output_shape, num_monomials)``.
    exponents : ArrayLike
        Exponent matrix, shape ``(d, num_monomials)``.
    domain_center : ArrayLike
        Expansion point, shape ``(d,)``.
    """

    coeffs: Array
    exponents: Array
    domain_center: Array

    def __init__(
        self,
        coeffs: ArrayLike,
        exponents: ArrayLike,
        domain_center: ArrayLike,
        _static_order: "int | tuple[int, ...] | None" = None,
        _domain_treedef: "jax.tree_util.PyTreeDef | None" = None,
        _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
        _per_leaf_order: "tuple[int, ...] | None" = None,
    ) -> None:
        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.domain_center = jnp.asarray(domain_center)

        d = self.exponents.shape[0]

        if _static_order is not None:
            if isinstance(_static_order, int):
                self._static_order = tuple([_static_order] * d)
            else:
                self._static_order = tuple(_static_order)
        else:
            self._static_order = tuple(
                int(jnp.max(self.exponents[i])) for i in range(d)
            )

        self._output_shape = self.coeffs.shape[:-1]

        # Structured domain support
        self._domain_treedef = _domain_treedef
        self._leaf_shapes = _leaf_shapes
        self._per_leaf_order = _per_leaf_order

        if self.coeffs.ndim < 1:
            raise ValueError(f"coeffs must be at least 1D, got shape {self.coeffs.shape}")
        if self.exponents.ndim != 2:
            raise ValueError(f"exponents must be 2D, got shape {self.exponents.shape}")
        if self.coeffs.shape[-1] != self.exponents.shape[1]:
            raise ValueError(
                f"coeffs and exponents must have same number of monomials: "
                f"{self.coeffs.shape[-1]} vs {self.exponents.shape[1]}"
            )
        if self.domain_center.shape[0] != self.exponents.shape[0]:
            raise ValueError(
                f"domain_center must match exponents dimension: "
                f"{self.domain_center.shape[0]} vs {self.exponents.shape[0]}"
            )

    # --- Pytree methods ---

    def tree_flatten(self) -> Tuple[Tuple[Array, Array, Array], dict]:
        return (
            (self.coeffs, self.exponents, self.domain_center),
            {
                "_static_order": self._static_order,
                "_output_shape": self._output_shape,
                "_domain_treedef": self._domain_treedef,
                "_leaf_shapes": self._leaf_shapes,
                "_per_leaf_order": self._per_leaf_order,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorPolynomial":
        static_order = aux_data.get("_static_order") if aux_data else None
        domain_treedef = aux_data.get("_domain_treedef") if aux_data else None
        leaf_shapes = aux_data.get("_leaf_shapes") if aux_data else None
        per_leaf_order = aux_data.get("_per_leaf_order") if aux_data else None
        return cls(
            *children,
            _static_order=static_order,
            _domain_treedef=domain_treedef,
            _leaf_shapes=leaf_shapes,
            _per_leaf_order=per_leaf_order,
        )

    # --- Properties ---

    @property
    def n(self) -> int:
        import math
        return math.prod(self._output_shape) if self._output_shape else 1

    @property
    def d(self) -> int:
        return self.exponents.shape[0]

    @property
    def num_monomials(self) -> int:
        return self.coeffs.shape[-1]

    @property
    def order(self) -> "tuple[int, ...]":
        return self._static_order

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._output_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._output_shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.coeffs.dtype

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
        """True if this TP uses per-leaf total degree bounds (structured domain)."""
        return self._domain_treedef is not None

    @property
    def constant_term(self) -> Array:
        """Constant (order-0) term, shape ``(*output_shape,)``."""
        is_constant = jnp.all(self.exponents == 0, axis=0)
        return jnp.sum(jnp.where(is_constant, self.coeffs, 0.0), axis=-1)

    # --- Evaluation ---

    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate the polynomial at a point.

        Parameters
        ----------
        x : ArrayLike
            Point, shape ``(d,)``.

        Returns
        -------
        Array
            Polynomial value, shape ``(*output_shape,)``.
        """
        x = jnp.asarray(x)
        dx = x - self.domain_center

        # Evaluate each monomial: monomial_i = prod_j dx[j]^exponents[j, i]
        monomials = jnp.prod(dx[:, None] ** self.exponents, axis=0)  # (m,)

        return jnp.sum(self.coeffs * monomials, axis=-1)

    # --- Bounding ---

    def bound(self, domain_radius: ArrayLike) -> Interval:
        """Bound the polynomial over ``[center - r, center + r]``.

        Evaluates in normalized coordinates ``u = (x - center) / r`` so
        monomials are bounded over ``[-1, 1]^d``.  Coefficients are rescaled
        by ``r^alpha`` before applying termwise bounds.

        Parameters
        ----------
        domain_radius : ArrayLike
            Half-width of the bounding box, shape ``(d,)``.
        """
        domain_radius = jnp.asarray(domain_radius)

        # Scale coefficients: a_alpha * r^alpha
        # r^alpha for each monomial: prod_j r_j^{exp_j}
        log_r = jnp.log(jnp.abs(domain_radius) + 1e-30)
        log_scale = self.exponents.T @ log_r  # (num_monomials,)
        scale = jnp.exp(log_scale)

        scaled_coeffs = self.coeffs * scale  # (*output_shape, m)

        has_odd = jnp.any(self.exponents % 2 == 1, axis=0)
        is_constant = jnp.all(self.exponents == 0, axis=0)

        mono_lower = jnp.where(is_constant, 1.0, jnp.where(has_odd, -1.0, 0.0))
        mono_upper = jnp.ones(self.num_monomials)

        zeros = jnp.zeros_like(scaled_coeffs)
        c_pos = jnp.maximum(scaled_coeffs, zeros)
        c_neg = jnp.minimum(scaled_coeffs, zeros)

        term_lower = c_pos * mono_lower + c_neg * mono_upper
        term_upper = c_pos * mono_upper + c_neg * mono_lower

        return interval(jnp.sum(term_lower, axis=-1), jnp.sum(term_upper, axis=-1))

    # --- Canonicalization and order reduction ---

    def to_canonical(self, target_order: "int | tuple[int, ...] | None" = None) -> "TaylorPolynomial":
        """Convert to canonical exponent structure, discarding terms above target_order."""
        # Handle structured domain mode
        if self._leaf_shapes is not None and self._per_leaf_order is not None:
            return self._to_canonical_per_leaf(target_order)

        if target_order is None:
            target_order = self._static_order
        if isinstance(target_order, int):
            target_order = tuple([target_order] * self.d)
        else:
            target_order = tuple(target_order)

        canonical_exp = _get_canonical_exponents(self.d, target_order)
        num_canonical = canonical_exp.shape[1]

        base = max(target_order) + 2
        powers = base ** jnp.arange(self.d)

        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        target_arr = jnp.array(target_order, dtype=jnp.int32)[:, None]
        has_match = jnp.all(self.exponents <= target_arr, axis=0)

        if num_canonical > 50:
            sort_perm = jnp.argsort(canonical_hash)
            sorted_canonical_hash = canonical_hash[sort_perm]
            sorted_indices = jnp.searchsorted(sorted_canonical_hash, current_hash)
            sorted_indices = jnp.clip(sorted_indices, 0, num_canonical - 1)
            canonical_indices = sort_perm[sorted_indices]
            scatter_matrix = jax.nn.one_hot(canonical_indices, num_canonical, dtype=self.dtype)
        else:
            match_matrix = current_hash[:, None] == canonical_hash[None, :]
            scatter_matrix = match_matrix.astype(self.dtype)

        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)
        new_coeffs = masked_coeffs @ scatter_matrix

        return TaylorPolynomial(
            new_coeffs, canonical_exp, self.domain_center,
            _static_order=target_order,
        )

    def _to_canonical_per_leaf(
        self,
        target_order: "tuple[int, ...] | None" = None,
    ) -> "TaylorPolynomial":
        """Convert to canonical exponent structure for structured domain mode."""
        import math

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
                # target_order has wrong length (probably per-variable order),
                # use stored per_leaf_order instead
                per_leaf_order = self._per_leaf_order
        else:
            per_leaf_order = self._per_leaf_order

        per_leaf_order = tuple(per_leaf_order)

        canonical_exp = _get_leaf_total_degree_exponents(leaf_shapes, per_leaf_order)
        num_canonical = canonical_exp.shape[1]

        max_exp_per_var = max(max(self._static_order), max(per_leaf_order)) + 1
        base = max_exp_per_var + 2
        powers = base ** jnp.arange(self.d)

        current_hash = jnp.sum(self.exponents * powers[:, None], axis=0)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        has_match = _check_per_leaf_bounds(self.exponents, leaf_shapes, per_leaf_order)

        if num_canonical > 50:
            sort_perm = jnp.argsort(canonical_hash)
            sorted_canonical_hash = canonical_hash[sort_perm]
            sorted_indices = jnp.searchsorted(sorted_canonical_hash, current_hash)
            sorted_indices = jnp.clip(sorted_indices, 0, num_canonical - 1)
            canonical_indices = sort_perm[sorted_indices]
            scatter_matrix = jax.nn.one_hot(canonical_indices, num_canonical, dtype=self.dtype)
        else:
            match_matrix = current_hash[:, None] == canonical_hash[None, :]
            scatter_matrix = match_matrix.astype(self.dtype)

        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)
        new_coeffs = masked_coeffs @ scatter_matrix

        # Compute per-variable order from per-leaf order
        per_var_order = []
        leaf_sizes = [math.prod(s) if s else 1 for s in leaf_shapes]
        for size, max_ord in zip(leaf_sizes, per_leaf_order):
            per_var_order.extend([max_ord] * size)

        return TaylorPolynomial(
            new_coeffs, canonical_exp, self.domain_center,
            _static_order=tuple(per_var_order),
            _domain_treedef=self._domain_treedef,
            _leaf_shapes=leaf_shapes,
            _per_leaf_order=per_leaf_order,
        )

    def reduce_order(self, target_order: "int | tuple[int, ...]") -> "TaylorPolynomial":
        """Reduce polynomial order, silently discarding high-order terms."""
        if isinstance(target_order, int):
            target_order = tuple([target_order] * self.d)
        else:
            target_order = tuple(target_order)
        target_arr = jnp.array(target_order, dtype=jnp.int32)[:, None]
        keep_mask = jnp.all(self.exponents <= target_arr, axis=0)
        new_coeffs = jnp.where(keep_mask, self.coeffs, 0.0)

        return TaylorPolynomial(
            new_coeffs, self.exponents, self.domain_center,
            _static_order=target_order,
            _domain_treedef=self._domain_treedef,
            _leaf_shapes=self._leaf_shapes,
            _per_leaf_order=self._per_leaf_order,
        )

    # --- Conversion ---

    def to_taylor_model(
        self,
        domain_radius: ArrayLike,
        remainder: Interval | None = None,
    ) -> TaylorModel:
        """Convert to a TaylorModel by specifying a domain radius and remainder.

        Parameters
        ----------
        domain_radius : ArrayLike
            Half-width of the domain box, shape ``(d,)``.
        remainder : Interval, optional
            Remainder interval with shape ``(*output_shape,)``.
            Defaults to zero interval.
        """
        domain_radius = jnp.asarray(domain_radius)
        if remainder is None:
            if len(self._output_shape) == 0:
                remainder = interval(jnp.zeros((), dtype=self.dtype))
            else:
                remainder = interval(jnp.zeros(self._output_shape, dtype=self.dtype))

        # Coefficients are already in raw (x - center) coordinates, no rescaling needed.
        domain = icentpert(self.domain_center, domain_radius)

        return TaylorModel(
            self.coeffs, self.exponents, remainder,
            domain, center=self.domain_center,
            _static_order=self._static_order,
            _domain_treedef=self._domain_treedef,
            _leaf_shapes=self._leaf_shapes,
            _per_leaf_order=self._per_leaf_order,
        )

    # --- Indexing ---

    def __getitem__(self, idx) -> "TaylorPolynomial":
        c = self.coeffs[idx]
        if c.ndim == 0:
            raise ValueError("Cannot index into monomial dimension directly")
        return TaylorPolynomial(
            c, self.exponents, self.domain_center,
            _static_order=self._static_order,
            _domain_treedef=self._domain_treedef,
            _leaf_shapes=self._leaf_shapes,
            _per_leaf_order=self._per_leaf_order,
        )

    def __len__(self) -> int:
        if len(self._output_shape) == 0:
            raise TypeError("Scalar TaylorPolynomial has no len()")
        return self._output_shape[0]

    # --- Arithmetic stubs (implemented in nattp.py) ---

    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __neg__(self): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __pow__(self, power): ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...

    # --- String representation ---

    def __str__(self) -> str:
        return (
            f"TaylorPolynomial(shape={self.shape}, d={self.d}, order={self.order}, "
            f"monomials={self.num_monomials})"
        )

    def __repr__(self) -> str:
        return (
            f"TaylorPolynomial(coeffs={self.coeffs!r}, exponents={self.exponents!r}, "
            f"domain_center={self.domain_center!r})"
        )
