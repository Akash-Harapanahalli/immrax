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
    PyTreeShape,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _leaf_slice,
    _merge_taylor_terms,
    _pytree_to_flattened_array,
    _unflatten_array_to_pytree,
    _compute_per_leaf_order_from_exponents,
    _normalize_order_pytree,
    _is_interval_or_array_leaf,
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
    flat_center : ArrayLike
        Expansion point, shape ``(d,)``.
    """

    coeffs: Array
    exponents: Array
    flat_center: Array

    def __init__(
        self,
        coeffs: ArrayLike,
        exponents: ArrayLike,
        flat_center: ArrayLike,
        *,
        _input_pytree: "PyTreeShape | None" = None,
        _output_pytree: "PyTreeShape | None" = None,
        _per_leaf_order: "tuple[int, ...] | None" = None,
        # Legacy kwargs for backward compatibility
        _domain_treedef: "jax.tree_util.PyTreeDef | None" = None,
        _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
        _static_order: "int | tuple[int, ...] | None" = None,  # Deprecated, ignored
    ) -> None:
        import math

        self.coeffs = jnp.asarray(coeffs)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.flat_center = jnp.asarray(flat_center)

        self._output_shape = self.coeffs.shape[:-1]

        # Handle _input_pytree: either from new kwarg or legacy kwargs
        if _input_pytree is not None:
            self._input_pytree = _input_pytree
        elif _domain_treedef is not None and _leaf_shapes is not None:
            self._input_pytree = PyTreeShape(_domain_treedef, _leaf_shapes)
        else:
            raise ValueError("Either _input_pytree or both _domain_treedef and _leaf_shapes must be provided")

        # Handle _output_pytree
        if _output_pytree is not None:
            self._output_pytree = _output_pytree
        else:
            self._output_pytree = PyTreeShape.flat(self._output_shape)

        self._per_leaf_order = _per_leaf_order

        if self._per_leaf_order is None:
            raise ValueError("_per_leaf_order cannot be None")

        # Validate _output_pytree.flat_size matches output shape
        expected_flat = math.prod(self._output_shape) if self._output_shape else 1
        if self._output_pytree.flat_size != expected_flat:
            raise ValueError(
                f"_output_pytree.flat_size ({self._output_pytree.flat_size}) must match "
                f"prod(coeffs.shape[:-1]) ({expected_flat})"
            )

        if self.coeffs.ndim < 1:
            raise ValueError(
                f"coeffs must be at least 1D, got shape {self.coeffs.shape}"
            )
        if self.exponents.ndim != 2:
            raise ValueError(f"exponents must be 2D, got shape {self.exponents.shape}")
        if self.coeffs.shape[-1] != self.exponents.shape[1]:
            raise ValueError(
                f"coeffs and exponents must have same number of monomials: "
                f"{self.coeffs.shape[-1]} vs {self.exponents.shape[1]}"
            )
        if self.flat_center.shape[0] != self.exponents.shape[0]:
            raise ValueError(
                f"flat_center must match exponents dimension: "
                f"{self.flat_center.shape[0]} vs {self.exponents.shape[0]}"
            )

    @property
    def domain_center(self) -> Array:
        """Alias for flat_center for backward compatibility."""
        return self.flat_center

    @property
    def _domain_treedef(self):
        """PyTree structure of the domain (convenience accessor)."""
        return self._input_pytree.treedef

    @property
    def _leaf_shapes(self):
        """Shapes of each leaf in the domain pytree (convenience accessor)."""
        return self._input_pytree.leaf_shapes

    @property
    def center(self):
        """Structured center (PyTree of Arrays)."""
        return self._input_pytree.unflatten(self.flat_center)

    @property
    def structured_center(self):
        """Get the center as a structured pytree matching the output pytree structure.

        If output has multiple leaves, uses _output_pytree to unflatten.
        Otherwise falls back to _input_pytree.
        """
        if self._output_pytree.num_leaves > 1:
            return self._output_pytree.unflatten(self.flat_center)
        return self._input_pytree.unflatten(self.flat_center)

    # --- Pytree methods ---

    def tree_flatten(self) -> Tuple[Tuple[Array, Array, Array], dict]:
        return (
            (self.coeffs, self.exponents, self.flat_center),
            {
                "_input_pytree": self._input_pytree,
                "_output_pytree": self._output_pytree,
                "_per_leaf_order": self._per_leaf_order,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorPolynomial":
        input_pytree = aux_data.get("_input_pytree") if aux_data else None
        output_pytree = aux_data.get("_output_pytree") if aux_data else None
        per_leaf_order = aux_data.get("_per_leaf_order") if aux_data else None
        return cls(
            *children,
            _input_pytree=input_pytree,
            _output_pytree=output_pytree,
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
        return self._per_leaf_order

    @property
    def max_order(self) -> int:
        """Maximum order across all leaves."""
        return max(self._per_leaf_order)

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
        return self._input_pytree.treedef

    @property
    def leaf_shapes(self) -> "tuple[tuple[int, ...], ...] | None":
        """Shapes of each leaf Interval in the domain pytree, or None."""
        return self._input_pytree.leaf_shapes

    @property
    def per_leaf_order(self) -> "tuple[int, ...] | None":
        """Per-leaf total degree bounds, or None for per-variable mode."""
        return self._per_leaf_order

    @property
    def is_structured(self) -> bool:
        """Always True now."""
        return True

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
        dx = x - self.flat_center

        # Evaluate each monomial: monomial_i = prod_j dx[j]^exponents[j, i]
        monomials = jnp.prod(dx[:, None] ** self.exponents, axis=0)  # (m,)

        return jnp.sum(self.coeffs * monomials, axis=-1)

    def evaluate_structured(self, *args):
        treedef, leaf_shapes, flat_x = _pytree_to_flattened_array(args)
        if treedef != self._domain_treedef or leaf_shapes != self._leaf_shapes:
            raise ValueError(
                "Domain pytree structure must match the polynomial's domain structure,"
                f"got {treedef} and {leaf_shapes} instead of {self._domain_treedef} and {self._leaf_shapes}."
            )
        return self.evaluate(flat_x)

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

    def to_canonical(
        self, target_order: "int | tuple[int, ...] | None" = None
    ) -> "TaylorPolynomial":
        """Convert to canonical exponent structure, discarding terms above target_order."""
        import math

        leaf_shapes = self._leaf_shapes
        num_leaves = len(leaf_shapes)

        # Determine new per_leaf_order
        if target_order is not None:
            if isinstance(target_order, int):
                per_leaf_order = tuple([target_order] * num_leaves)
            elif len(target_order) == num_leaves:
                per_leaf_order = tuple(target_order)
            else:
                # target_order has wrong length (probably per-variable order),
                # use stored per_leaf_order to avoid guessing
                per_leaf_order = self._per_leaf_order
        else:
            per_leaf_order = self._per_leaf_order

        per_leaf_order = tuple(per_leaf_order)

        canonical_exp = _get_leaf_total_degree_exponents(leaf_shapes, per_leaf_order)
        num_canonical = canonical_exp.shape[1]

        # Max order approximation for hashing: take max over all leaves
        max_order_val = max(per_leaf_order)
        base = max_order_val + 2
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
            scatter_matrix = jax.nn.one_hot(
                canonical_indices, num_canonical, dtype=self.dtype
            )
        else:
            match_matrix = current_hash[:, None] == canonical_hash[None, :]
            scatter_matrix = match_matrix.astype(self.dtype)

        masked_coeffs = jnp.where(has_match, self.coeffs, 0.0)
        new_coeffs = masked_coeffs @ scatter_matrix

        return TaylorPolynomial(
            new_coeffs,
            canonical_exp,
            self.flat_center,
            _input_pytree=PyTreeShape(self._domain_treedef, leaf_shapes),
            _output_pytree=self._output_pytree,
            _per_leaf_order=per_leaf_order,
        )

    def reduce_order(self, target_order: "int | tuple[int, ...]") -> "TaylorPolynomial":
        """Reduce polynomial order, silently discarding high-order terms."""
        num_leaves = len(self._leaf_shapes)
        if isinstance(target_order, int):
            target_order = tuple([target_order] * num_leaves)
        else:
            target_order = tuple(target_order)

        # Check against per-leaf bounds
        keep_mask = _check_per_leaf_bounds(
            self.exponents, self._leaf_shapes, target_order
        )
        new_coeffs = jnp.where(keep_mask, self.coeffs, 0.0)

        return TaylorPolynomial(
            new_coeffs,
            self.exponents,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=self._output_pytree,
            _per_leaf_order=target_order,
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
        domain = icentpert(self.flat_center, domain_radius)

        return TaylorModel(
            self.coeffs,
            self.exponents,
            remainder,
            domain,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=self._output_pytree,
            _per_leaf_order=self._per_leaf_order,
        )

    # --- Indexing ---

    def __getitem__(self, idx) -> "TaylorPolynomial":
        c = self.coeffs[idx]
        if c.ndim == 0:
            raise ValueError("Cannot index into monomial dimension directly")
        return TaylorPolynomial(
            c,
            self.exponents,
            self.flat_center,
            _input_pytree=self._input_pytree,
            _output_pytree=PyTreeShape.flat(c.shape[:-1]),
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
            f"flat_center={self.flat_center!r})"
        )


def _taylor_polynomial_constant_impl(
    val: ArrayLike,
    d: int,
    order: "int | tuple[int, ...]" = 1,
    center=None,
    _domain_treedef: "jax.tree_util.PyTreeDef | None" = None,
    _leaf_shapes: "tuple[tuple[int, ...], ...] | None" = None,
    _per_leaf_order: "tuple[int, ...] | None" = None,
    flat_center=None,
    _input_pytree: "PyTreeShape | None" = None,
    _output_pytree: "PyTreeShape | None" = None,
) -> TaylorPolynomial:
    """Internal implementation for taylor_polynomial_constant."""
    val = jnp.asarray(val)
    output_shape = val.shape

    # Resolve _input_pytree
    if _input_pytree is None:
        if _leaf_shapes is None:
            _leaf_shapes = ((d,),)
        if _domain_treedef is None:
            _domain_treedef = jax.tree_util.tree_structure(jnp.zeros(d))
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

    # Coeffs: constant term is val
    coeffs = jnp.zeros((*output_shape, exponents.shape[1]), dtype=val.dtype)
    coeffs = coeffs.at[..., 0].set(val)

    # If flat_center not provided, assume zero
    if flat_center is None:
        flat_center = jnp.zeros(d, dtype=val.dtype)

    if _output_pytree is None:
        _output_pytree = PyTreeShape.flat(output_shape)

    return TaylorPolynomial(
        coeffs,
        exponents,
        flat_center,
        _input_pytree=_input_pytree,
        _output_pytree=_output_pytree,
        _per_leaf_order=_per_leaf_order,
    )


def taylor_polynomial_constant(
    val: ArrayLike,
    domain: "Interval | PyTree[Interval]",
    order: "int | tuple[int, ...]" = 1,
    center=None,
) -> TaylorPolynomial:
    """Create a constant Taylor polynomial.

    Parameters
    ----------
    val : ArrayLike
        The constant value.
    domain : Interval or PyTree[Interval]
        The domain. Can be a single Interval or a PyTree of Intervals.
    order : int or tuple[int, ...]
        Polynomial degree of the Taylor Polynomial (default: 1)
    center : ArrayLike, optional
        Expansion point. Default is domain center.

    Returns
    -------
    TaylorPolynomial
        Constant Taylor polynomial.
    """
    # Process domain and metadata
    treedef, leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)
    d = flat_domain.shape[0]

    # Infer per_leaf_order
    num_leaves = len(leaf_shapes)
    if isinstance(order, int):
        per_leaf_order = tuple([order] * num_leaves)
    else:
        per_leaf_order = tuple(order)

    if center is not None:
        _, _, flat_center = _pytree_to_flattened_array(center)
    else:
        flat_center = flat_domain.center

    return _taylor_polynomial_constant_impl(
        val,
        d,
        order,
        center=None,
        _input_pytree=PyTreeShape(treedef, leaf_shapes),
        _per_leaf_order=per_leaf_order,
        flat_center=flat_center,
    )


def taylor_polynomial_identity(
    domain: "Interval | PyTree[Interval]",
    center: "ArrayLike | PyTree[ArrayLike]" = None,
    order: "int | PyTree[int]" = 1,
) -> TaylorPolynomial:
    """Create a Taylor polynomial representing the identity function over a domain.

    Supports domain as a single Interval or any pytree structure of Intervals
    (list, dict, nested structures). When domain is a pytree, order can also
    be specified as a matching pytree to give per-leaf total degree bounds.

    Parameters
    ----------
    domain : Interval or PyTree[Interval]
        The domain.
    center : ArrayLike or PyTree[ArrayLike], optional
        Expansion point. Default is domain center.
    order : int or PyTree[int], optional
        Polynomial degree bounds.

    Returns
    -------
    TaylorPolynomial
        Identity polynomial.
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

    input_pytree = PyTreeShape(treedef, leaf_shapes)
    # For identity, output matches domain structure
    output_pytree = PyTreeShape(treedef, leaf_shapes)

    return TaylorPolynomial(
        coeffs,
        exponents,
        center_flat,
        _input_pytree=input_pytree,
        _output_pytree=output_pytree,
        _per_leaf_order=per_leaf_order,
    )


def taylor_polynomial_concatenate(
    tps: list["TaylorPolynomial"], axis: int = 0
) -> "TaylorPolynomial":
    """Concatenate multiple TaylorPolynomials along an output dimension.

    Mirrors :func:`taylor_model_concatenate` for TaylorPolynomials.
    All input TaylorPolynomials must share the same domain (exponents, center).

    Parameters
    ----------
    tps : list of TaylorPolynomial
        TaylorPolynomials to concatenate.
    axis : int
        Axis in the output shape along which to concatenate (default: 0).

    Returns
    -------
    TaylorPolynomial
        Concatenated TaylorPolynomial.
    """
    if len(tps) == 0:
        raise ValueError("Cannot concatenate empty list of TaylorPolynomials")

    if len(tps) == 1:
        return tps[0]

    ref = tps[0]
    from functools import reduce

    per_leaf_order = reduce(
        lambda a, b: tuple(max(ai, bi) for ai, bi in zip(a, b)),
        (tp._per_leaf_order for tp in tps),
    )

    coeffs = jnp.concatenate([tp.coeffs for tp in tps], axis=axis)

    # Build merged _output_pytree from all input TPs' output pytrees
    merged_leaf_shapes = []
    for tp in tps:
        merged_leaf_shapes.extend(tp._output_pytree.leaf_shapes)
    merged_leaf_shapes = tuple(merged_leaf_shapes)
    dummy_leaves = [jnp.zeros(s) for s in merged_leaf_shapes]
    merged_treedef = jax.tree_util.tree_structure(
        dummy_leaves, is_leaf=_is_interval_or_array_leaf
    )
    output_pytree = PyTreeShape(merged_treedef, merged_leaf_shapes)

    return TaylorPolynomial(
        coeffs,
        ref.exponents,
        ref.flat_center,
        _input_pytree=ref._input_pytree,
        _output_pytree=output_pytree,
        _per_leaf_order=per_leaf_order,
    )
