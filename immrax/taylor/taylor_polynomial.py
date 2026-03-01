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
from jaxtyping import Array, ArrayLike, PyTree
import numpy as onp

from immrax.inclusion import Interval, interval, icentpert
import jax.tree_util

from immrax.taylor.base import (
    MultiIndex,
    MultiIndexArray,
    PyTreeShape,
    leaf_total_degree_exponents,
    leaf_slice,
    check_leaf_bounds,
    _merge_taylor_terms,
    pack_pytree,
    normalize_leaf_order,
    _is_interval_or_array_leaf,
)

# TaylorModel imported here for to_taylor_model; the property TaylorModel.polynomial
# lazily imports TaylorPolynomial to avoid a circular eager import.
from immrax.taylor.taylor_model import TaylorModel


def _multiindex_in_leaf_bounds(
    mi: MultiIndex,
    leaf_shapes: "tuple[tuple[int, ...], ...]",
    leaf_order: "tuple[int, ...]",
) -> bool:
    """Pure-Python per-leaf bounds check for a single MultiIndex."""
    for leaf_idx in range(len(leaf_shapes)):
        slc = leaf_slice(leaf_shapes, leaf_idx)
        leaf_total = sum(mi[j] for j in range(slc.start, slc.stop))
        if leaf_total > leaf_order[leaf_idx]:
            return False
    return True


@register_pytree_node_class
class TaylorPolynomial:
    r"""Multivariate polynomial centered at a point.

    .. math::
        p(x) = \sum_{|\alpha| \leq k} a_\alpha (x - x_0)^\alpha

    Parameters
    ----------
    coeffs : ArrayLike
        Polynomial coefficients, shape ``(*output_shape, num_monomials)``.
    multiindices : MultiIndexArray
        Tuple of MultiIndex objects, one per monomial.
    flat_center : ArrayLike
        Expansion point, shape ``(d,)``.
    """

    coeffs: Array
    multiindices: MultiIndexArray
    flat_center: Array

    def __init__(
        self,
        coeffs,
        multiindices,
        flat_center,
        *,
        input_pytree: "PyTreeShape | None" = None,
        output_pytree: "PyTreeShape | None" = None,
        leaf_order: "tuple[int, ...] | None" = None,
    ) -> None:
        """Low-level constructor — converts arrays and assigns attributes.

        No shape validation is performed; callers are trusted to provide
        compatible shapes.  Use the high-level constructors
        :func:`taylor_polynomial_identity` or :func:`taylor_polynomial_constant`
        for user-facing construction with full input validation.
        """
        self.coeffs = jnp.asarray(coeffs)
        self.multiindices = (
            multiindices
            if isinstance(multiindices, MultiIndexArray)
            else MultiIndexArray(multiindices)
        )
        self.flat_center = jnp.asarray(flat_center)
        self.input_pytree = input_pytree
        self.output_pytree = output_pytree
        self.leaf_order = leaf_order

    @property
    def _output_shape(self) -> "tuple[int, ...]":
        """Output shape, derived from coeffs. Last axis is monomial axis."""
        return self.coeffs.shape[:-1]

    @property
    def _domain_treedef(self):
        """PyTree structure of the domain (convenience accessor)."""
        return self.input_pytree.treedef

    @property
    def _leaf_shapes(self):
        """Shapes of each leaf in the domain pytree (convenience accessor)."""
        return self.input_pytree.leaf_shapes

    @property
    def center(self):
        """Structured center (PyTree of Arrays)."""
        return self.input_pytree.unflatten(self.flat_center)

    @property
    def structured_center(self):
        """Get the center as a structured pytree matching the output pytree structure.

        If output has multiple leaves, uses output_pytree to unflatten.
        Otherwise falls back to input_pytree.
        """
        if self.output_pytree.num_leaves > 1:
            return self.output_pytree.unflatten(self.flat_center)
        return self.input_pytree.unflatten(self.flat_center)

    # --- Pytree methods ---

    def tree_flatten(self) -> Tuple[Tuple[Array, Array], tuple]:
        return (
            (self.coeffs, self.flat_center),
            (self.multiindices, self.input_pytree, self.output_pytree, self.leaf_order),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "TaylorPolynomial":
        # Bypass __init__ to avoid jnp.asarray — JAX/equinox may pass
        # non-array sentinels (e.g. booleans) through tree operations.
        coeffs, flat_center = children
        multiindices, input_pytree, output_pytree, leaf_order = aux_data
        obj = object.__new__(cls)
        obj.coeffs = coeffs
        obj.multiindices = multiindices  # already a MultiIndexArray from aux_data
        obj.flat_center = flat_center
        obj.input_pytree = input_pytree
        obj.output_pytree = output_pytree
        obj.leaf_order = leaf_order
        return obj

    # --- Properties ---

    @property
    def n(self) -> int:
        import math

        return math.prod(self._output_shape) if self._output_shape else 1

    @property
    def d(self) -> int:
        return self.multiindices.d

    @property
    def num_monomials(self) -> int:
        return self.coeffs.shape[-1]

    @property
    def order(self) -> "tuple[int, ...]":
        return self.leaf_order

    @property
    def max_order(self) -> int:
        """Maximum order across all leaves."""
        return max(self.leaf_order)

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
        return self.input_pytree.treedef

    @property
    def leaf_shapes(self) -> "tuple[tuple[int, ...], ...] | None":
        """Shapes of each leaf Interval in the domain pytree, or None."""
        return self.input_pytree.leaf_shapes

    @property
    def constant_term(self) -> Array:
        """Constant (order-0) term, shape ``(*output_shape,)``."""
        exponents = self.multiindices.to_jnp()  # (d, m)
        is_constant = jnp.all(exponents == 0, axis=0)
        return jnp.sum(jnp.where(is_constant, self.coeffs, 0.0), axis=-1)

    # --- Evaluation ---

    def evaluate(self, x: ArrayLike, method: str = 'horner') -> Array:
        """Evaluate the polynomial at a point.

        Parameters
        ----------
        x : ArrayLike
            Point, shape ``(d,)``.
        method : str, optional
            Evaluation method.  ``'horner'`` (default) uses a tensor Horner
            sweep; ``'standard'`` uses cumprod to compute monomials directly.
            This is a static Python argument — it is not traced by JAX.

        Returns
        -------
        Array
            Polynomial value, shape ``(*output_shape,)``.
        """
        x = jnp.asarray(x)
        dx = x - self.flat_center

        if method == 'standard':
            monomials = self.evaluate_monomials(dx)
            return jnp.sum(self.coeffs * monomials, axis=-1)
        elif method == 'horner':
            d = self.d
            K = self.max_order
            output_shape = self._output_shape
            base = K + 1  # degree slots per variable: 0 … K

            # Scatter coefficients into a dense (*output_shape, K+1, …, K+1) tensor.
            exps_np = self.multiindices.to_numpy()  # (d, m), always concrete
            if d > 0:
                strides = onp.array(
                    [base ** (d - 1 - i) for i in range(d)], dtype=onp.int64
                )
                flat_idx = strides @ exps_np  # (m,) flat index into (K+1)^d cube
            else:
                flat_idx = onp.zeros(self.num_monomials, dtype=onp.int64)

            T_flat = jnp.zeros((*output_shape, base**d), dtype=self.dtype)
            T_flat = T_flat.at[..., flat_idx].add(self.coeffs)
            T = T_flat.reshape(*output_shape, *([base] * d))

            # One 1-D Horner sweep per variable, last to first.
            # Each sweep collapses the trailing axis.
            current = T
            for i in range(d - 1, -1, -1):
                result = current[..., K]
                for k in range(K - 1, -1, -1):
                    result = current[..., k] + dx[i] * result
                current = result

            return current
        else:
            raise ValueError(f"Unknown evaluation method {method!r}; expected 'horner' or 'standard'")

    def evaluate_monomials(self, dx: ArrayLike) -> Array:
        """Evaluate each monomial (x-center)^alpha at dx = x - center.

        Uses ``jnp.cumprod`` to compute all powers of each variable up to
        its maximum exponent, then gathers the required power per monomial
        using static Python indices derived from the MultiIndexArray.
        """
        dx = jnp.asarray(dx)
        m = self.num_monomials
        monomials = jnp.ones(m, dtype=dx.dtype)
        for i in range(self.d):
            # Static list of exponents for variable i across all monomials
            exps_i = [int(mi[i]) for mi in self.multiindices]
            max_k = max(exps_i)
            if max_k == 0:
                continue
            # cumprod of [dx[i], dx[i], ..., dx[i]] (length max_k) gives
            # [dx[i]^1, dx[i]^2, ..., dx[i]^max_k]; prepend 1 for exponent 0.
            pows_pos = jnp.cumprod(jnp.full(max_k, dx[i]))
            pows = jnp.concatenate([jnp.ones(1, dtype=dx.dtype), pows_pos])
            # Gather per-monomial power using static Python integer indices
            # print(pows, exps_i)
            var_pow = jnp.stack([pows[e] for e in exps_i])
            monomials = monomials * var_pow
        return monomials

    def interval_evaluate(self, ix: Interval, method: str = 'horner') -> Interval:
        """Bound the polynomial over the interval ``ix``.

        Delegates to ``natif(partial(self.evaluate, method=method))``, so the
        same evaluation graph used for real inputs is lifted to interval
        arithmetic.  The default ``'horner'`` method organises coefficients
        into a d-way tensor and applies one 1-D Horner sweep per variable,
        capturing cross-variable interactions that termwise bounds would miss.

        Parameters
        ----------
        ix : Interval
            Input box, shape ``(d,)``.
        method : str, optional
            Evaluation method passed to :meth:`evaluate` (default ``'horner'``).

        Returns
        -------
        Interval
            Overapproximation of polynomial range, shape ``(*output_shape,)``.
        """
        from functools import partial
        from immrax.inclusion.nif import natif as _natif
        return _natif(partial(self.evaluate, method=method))(ix)

    def evaluate_structured(self, *args):
        treedef, leaf_shapes, flat_x = pack_pytree(args)
        if treedef != self._domain_treedef or leaf_shapes != self._leaf_shapes:
            raise ValueError(
                "Domain pytree structure must match the polynomial's domain structure,"
                f"got {treedef} and {leaf_shapes} instead of {self._domain_treedef} and {self._leaf_shapes}."
            )
        return self.evaluate(flat_x)

    def __call__(self, *args):
        """Alias for evaluate_structured"""
        return self.evaluate_structured(*args)

    def get_order(self, order: "int | tuple[int, ...]") -> "TaylorPolynomial":
        """Get the specified order term of the Taylor expansion as a new (non-canonical) TaylorPolynomial."""
        leaf_order = normalize_leaf_order(
            order, self._domain_treedef, self._leaf_shapes
        )
        prev_leaf_order = tuple(o - 1 for o in leaf_order)
        # Compute static boolean list for filtering multiindices
        keep_list = [
            _multiindex_in_leaf_bounds(mi, self._leaf_shapes, leaf_order)
            and not _multiindex_in_leaf_bounds(mi, self._leaf_shapes, prev_leaf_order)
            for mi in self.multiindices
        ]
        keep_mask = onp.array(keep_list)
        new_multiindices = MultiIndexArray(
            mi for mi, k in zip(self.multiindices, keep_list) if k
        )
        return TaylorPolynomial(
            self.coeffs[..., keep_mask],
            new_multiindices,
            self.flat_center,
            input_pytree=self.input_pytree,
            output_pytree=self.output_pytree,
            leaf_order=leaf_order,
        )

    def evaluate_order(self, order: "int | tuple[int, ...]", x: ArrayLike) -> Array:
        """Evaluate the p-th term of the Taylor expansion at the point x."""
        return self.get_order(order).evaluate(x)

    def get_to_order(self, order: "int | tuple[int, ...]") -> "TaylorPolynomial":
        leaf_order = normalize_leaf_order(
            order, self._domain_treedef, self._leaf_shapes
        )
        # Compute static boolean list for filtering multiindices
        keep_list = [
            _multiindex_in_leaf_bounds(mi, self._leaf_shapes, leaf_order)
            for mi in self.multiindices
        ]
        keep_mask = onp.array(keep_list)
        new_multiindices = MultiIndexArray(
            mi for mi, k in zip(self.multiindices, keep_list) if k
        )
        return TaylorPolynomial(
            self.coeffs[..., keep_mask],
            new_multiindices,
            self.flat_center,
            input_pytree=self.input_pytree,
            output_pytree=self.output_pytree,
            leaf_order=leaf_order,
        )

    def evaluate_to_order(self, order: "int | tuple[int, ...]", x: ArrayLike) -> Array:
        """Evaluate the Taylor expansion up to and including the specified order at the point x."""
        return self.get_to_order(order).evaluate(x)

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
        exponents = self.multiindices.to_jnp()  # (d, m)

        # Scale coefficients: a_alpha * r^alpha
        log_r = jnp.log(jnp.abs(domain_radius) + 1e-30)
        log_scale = exponents.T @ log_r  # (num_monomials,)
        scale = jnp.exp(log_scale)

        scaled_coeffs = self.coeffs * scale  # (*output_shape, m)

        has_odd = jnp.any(exponents % 2 == 1, axis=0)
        is_constant = jnp.all(exponents == 0, axis=0)

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

        # Determine new leaf_order
        if target_order is not None:
            if isinstance(target_order, int):
                leaf_order = tuple([target_order] * num_leaves)
            elif len(target_order) == num_leaves:
                leaf_order = tuple(target_order)
            else:
                # target_order has wrong length (probably per-variable order),
                # use stored leaf_order to avoid guessing
                leaf_order = self.leaf_order
        else:
            leaf_order = self.leaf_order

        leaf_order = tuple(leaf_order)

        canonical_mia = leaf_total_degree_exponents(leaf_shapes, leaf_order)
        num_canonical = canonical_mia.num_monomials

        # Max order approximation for hashing: take max over all leaves
        max_order_val = max(leaf_order)
        base = max_order_val + 2
        powers = base ** jnp.arange(self.d)

        current_exp = self.multiindices.to_jnp()  # (d, m)
        canonical_exp = canonical_mia.to_jnp()  # (d, num_canonical)

        current_hash = jnp.sum(current_exp * powers[:, None], axis=0)
        canonical_hash = jnp.sum(canonical_exp * powers[:, None], axis=0)

        has_match = check_leaf_bounds(self.multiindices, leaf_shapes, leaf_order)

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
            canonical_mia,
            self.flat_center,
            input_pytree=PyTreeShape(self._domain_treedef, leaf_shapes),
            output_pytree=self.output_pytree,
            leaf_order=leaf_order,
        )

    def reduce_order(self, target_order: "int | tuple[int, ...]") -> "TaylorPolynomial":
        """Reduce polynomial order, silently discarding high-order terms."""
        num_leaves = len(self._leaf_shapes)
        if isinstance(target_order, int):
            target_order = tuple([target_order] * num_leaves)
        else:
            target_order = tuple(target_order)

        # Check against per-leaf bounds
        keep_mask = check_leaf_bounds(
            self.multiindices, self._leaf_shapes, target_order
        )
        new_coeffs = jnp.where(keep_mask, self.coeffs, 0.0)

        return TaylorPolynomial(
            new_coeffs,
            self.multiindices,
            self.flat_center,
            input_pytree=self.input_pytree,
            output_pytree=self.output_pytree,
            leaf_order=target_order,
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
            self.multiindices,
            remainder,
            domain,
            self.flat_center,
            input_pytree=self.input_pytree,
            output_pytree=self.output_pytree,
            leaf_order=self.leaf_order,
        )

    # --- Indexing ---

    def __getitem__(self, idx) -> "TaylorPolynomial":
        c = self.coeffs[idx]
        if c.ndim == 0:
            raise ValueError("Cannot index into monomial dimension directly")
        return TaylorPolynomial(
            c,
            self.multiindices,
            self.flat_center,
            input_pytree=self.input_pytree,
            output_pytree=PyTreeShape.flat(c.shape[:-1]),
            leaf_order=self.leaf_order,
        )

    def __len__(self) -> int:
        if len(self._output_shape) == 0:
            raise TypeError("Scalar TaylorPolynomial has no len()")
        return self._output_shape[0]

    # --- Arithmetic stubs (implemented in pjet.py) ---

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
        in_shapes = (
            self.input_pytree.leaf_shapes if self.input_pytree is not None else None
        )
        out_shapes = (
            self.output_pytree.leaf_shapes
            if self.output_pytree is not None
            else (self._output_shape,)
        )
        in_str = (
            in_shapes[0] if in_shapes is not None and len(in_shapes) == 1 else in_shapes
        )
        out_str = (
            out_shapes[0]
            if out_shapes is not None and len(out_shapes) == 1
            else out_shapes
        )
        return (
            f"TaylorPolynomial(input_shape={in_str}, output_shape={out_str}, "
            f"order={self.order}, monomials={self.num_monomials})"
        )

    def __repr__(self) -> str:
        return (
            f"TaylorPolynomial(coeffs={self.coeffs!r}, multiindices={self.multiindices!r}, "
            f"flat_center={self.flat_center!r})"
        )


def _taylor_polynomial_constant_impl(
    val: ArrayLike,
    flat_center: ArrayLike,
    input_pytree: "PyTreeShape",
    leaf_order: "tuple[int, ...]",
    output_pytree: "PyTreeShape | None" = None,
) -> TaylorPolynomial:
    """Internal implementation for taylor_polynomial_constant."""
    val = jnp.asarray(val)
    output_shape = val.shape
    _leaf_shapes = input_pytree.leaf_shapes

    # Generate multiindices with per-leaf total degree bounds
    multiindices = leaf_total_degree_exponents(_leaf_shapes, leaf_order)

    # Coeffs: constant term is val, all others zero
    coeffs = jnp.zeros((*output_shape, multiindices.num_monomials), dtype=val.dtype)
    coeffs = coeffs.at[..., 0].set(val)

    if output_pytree is None:
        output_pytree = PyTreeShape.flat(output_shape)

    return TaylorPolynomial(
        coeffs,
        multiindices,
        flat_center,
        input_pytree=input_pytree,
        output_pytree=output_pytree,
        leaf_order=leaf_order,
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
    treedef, leaf_shapes, flat_domain = pack_pytree(domain)

    # Infer leaf_order
    num_leaves = len(leaf_shapes)
    if isinstance(order, int):
        leaf_order = tuple([order] * num_leaves)
    else:
        leaf_order = tuple(order)

    if center is not None:
        _, _, flat_center = pack_pytree(center)
    else:
        flat_center = flat_domain.center

    return _taylor_polynomial_constant_impl(
        val,
        flat_center,
        PyTreeShape(treedef, leaf_shapes),
        leaf_order,
    )


def taylor_polynomial_identity(
    center: "ArrayLike | PyTree[ArrayLike]",
    order: "int | PyTree[int]" = 1,
) -> TaylorPolynomial:
    """Create a Taylor polynomial representing the identity function.

    The polynomial is ``p(x) = x``, expanded around ``center``.
    The pytree structure of ``center`` determines the domain layout
    (leaf shapes and per-leaf degree bounds).

    Parameters
    ----------
    center : ArrayLike or PyTree[ArrayLike]
        Expansion point. Pytree structure determines the domain layout.
    order : int or PyTree[int], optional
        Polynomial degree bounds. If a pytree, must match the structure
        of ``center`` for per-leaf degree bounds.

    Returns
    -------
    TaylorPolynomial
        Identity polynomial satisfying ``p(center) == center``.
    """
    center = jnp.asarray(center)
    treedef, leaf_shapes, flat_center = pack_pytree(center)
    leaf_order = normalize_leaf_order(order, treedef, leaf_shapes)
    total_dim = flat_center.shape[0]

    # Generate multiindices with per-leaf total degree bounds
    multiindices = leaf_total_degree_exponents(leaf_shapes, leaf_order)
    exponents = multiindices.to_jnp()  # (d, m)

    # Identify constant monomial (all exponents zero)
    is_constant = jnp.sum(exponents, axis=0) == 0  # (m,)

    # Identify linear monomials (unit vectors e_i)
    eye_n = jnp.eye(total_dim, dtype=jnp.int32)  # (d, d)
    is_linear = jnp.all(exponents[:, None, :] == eye_n[:, :, None], axis=0)  # (d, m)

    # coeffs[i, j] = center[i] for constant monomial, 1.0 for e_i monomial, else 0
    coeffs = jnp.where(is_constant[None, :], flat_center[:, None], 0.0) + jnp.where(
        is_linear, 1.0, 0.0
    )  # (d, m)

    input_pytree = PyTreeShape(treedef, leaf_shapes)
    output_pytree = PyTreeShape(treedef, leaf_shapes)

    return TaylorPolynomial(
        coeffs,
        multiindices,
        flat_center,
        input_pytree=input_pytree,
        output_pytree=output_pytree,
        leaf_order=leaf_order,
    )


def taylor_polynomial_concatenate(
    tps: list["TaylorPolynomial"], axis: int = 0
) -> "TaylorPolynomial":
    """Concatenate multiple TaylorPolynomials along an output dimension.

    Mirrors :func:`taylor_model_concatenate` for TaylorPolynomials.
    All input TaylorPolynomials must share the same domain (multiindices, center).

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

    leaf_order = reduce(
        lambda a, b: tuple(max(ai, bi) for ai, bi in zip(a, b)),
        (tp.leaf_order for tp in tps),
    )

    coeffs = jnp.concatenate([tp.coeffs for tp in tps], axis=axis)

    # Build merged output_pytree from all input TPs' output pytrees
    merged_leaf_shapes = []
    for tp in tps:
        merged_leaf_shapes.extend(tp.output_pytree.leaf_shapes)
    merged_leaf_shapes = tuple(merged_leaf_shapes)
    dummy_leaves = [jnp.zeros(s) for s in merged_leaf_shapes]
    merged_treedef = jax.tree_util.tree_structure(
        dummy_leaves, is_leaf=_is_interval_or_array_leaf
    )
    output_pytree = PyTreeShape(merged_treedef, merged_leaf_shapes)

    return TaylorPolynomial(
        coeffs,
        ref.multiindices,
        ref.flat_center,
        input_pytree=ref.input_pytree,
        output_pytree=output_pytree,
        leaf_order=leaf_order,
    )
