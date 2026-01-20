from typing import Optional, Tuple

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike


@register_pytree_node_class
class Zonotope:
    r"""Defines the set

    .. math::
        Z = \{ \mathring{x} + Gv : v \in [-1,1]^m \}

    where :math:`\mathring{x} \in \mathbb{R}^n` is the center and
    :math:`G \in \mathbb{R}^{n \times m}` is the generator matrix.

    Parameters
    ----------
    ox : ArrayLike
        Center of the zonotope, shape (n,)
    G : ArrayLike
        Generator matrix, shape (n, m) where m is the number of generators
    """

    ox: Array  # Center
    G: Array  # Generator matrix

    def __init__(self, ox: ArrayLike, G: ArrayLike) -> None:
        self.ox = jnp.asarray(ox)
        self.G = jnp.asarray(G)
        if self.G.shape[:-1] != self.ox.shape:
            raise ValueError(
                f"Incompatible shapes: ox has shape {self.ox.shape}, "
                f"G has shape {self.G.shape} (expected G.shape[:-1] = {self.ox.shape})"
            )

    # --- Pytree methods ---

    def tree_flatten(self) -> Tuple[Tuple[Array, Array], None]:
        return ((self.ox, self.G), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Zonotope":
        return cls(*children)

    # --- Properties ---

    @property
    def n(self) -> int:
        """Dimension of the zonotope (state space dimension)."""
        return self.ox.shape[0]

    @property
    def m(self) -> int:
        """Number of generators."""
        return self.G.shape[-1]

    @property
    def order(self) -> float:
        """Order of the zonotope (m / n)."""
        return self.m / self.n

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the center."""
        return self.ox.shape

    @property
    def dtype(self) -> jnp.dtype:
        """Data type of the zonotope arrays."""
        return self.ox.dtype

    @property
    def center(self) -> Array:
        """Center of the zonotope."""
        return self.ox

    @property
    def generators(self) -> Array:
        """Generator matrix."""
        return self.G

    # --- Set operations ---

    def __add__(self, other: "Zonotope") -> "Zonotope":
        """Minkowski sum of two zonotopes.

        Z1 + Z2 = {z1 + z2 : z1 ∈ Z1, z2 ∈ Z2}
                = {c1 + c2 + [G1 | G2] v : v ∈ [-1,1]^(m1+m2)}
        """
        if not isinstance(other, Zonotope):
            raise TypeError("Can only add Zonotope to Zonotope")
        if self.ox.shape != other.ox.shape:
            raise ValueError(
                f"Incompatible dimensions: {self.ox.shape} vs {other.ox.shape}"
            )
        return Zonotope(self.ox + other.ox, jnp.concatenate((self.G, other.G), axis=-1))

    def __sub__(self, other: "Zonotope") -> "Zonotope":
        """Minkowski difference (Pontryagin difference is not closed for zonotopes).

        Z1 - Z2 = Z1 + (-Z2)
        """
        return self + (-other)

    def __neg__(self) -> "Zonotope":
        """Negation of a zonotope: -Z = {-z : z ∈ Z}."""
        return Zonotope(-self.ox, -self.G)

    def __matmul__(self, other: ArrayLike) -> "Zonotope":
        """Right matrix multiplication: Z @ A^T.

        For a zonotope Z and matrix A, computes {A^T z : z ∈ Z}.
        Use this when you want to transform by A^T, or use `linear_map(A, Z)` for A @ Z.
        """
        A = jnp.asarray(other)
        return Zonotope(self.ox @ A, self.G.T @ A).T

    def __rmatmul__(self, other: ArrayLike) -> "Zonotope":
        """Left matrix multiplication: A @ Z.

        For a matrix A ∈ R^{p×n} and zonotope Z ⊂ R^n, computes
        A @ Z = {A @ z : z ∈ Z} = {A @ c + A @ G @ v : v ∈ [-1,1]^m}
        """
        A = jnp.asarray(other)
        return Zonotope(A @ self.ox, A @ self.G)

    def __mul__(self, other: ArrayLike) -> "Zonotope":
        """Scalar multiplication: Z * α = {α * z : z ∈ Z}."""
        alpha = jnp.asarray(other)
        return Zonotope(alpha * self.ox, alpha * self.G)

    def __rmul__(self, other: ArrayLike) -> "Zonotope":
        """Scalar multiplication: α * Z = {α * z : z ∈ Z}."""
        return self.__mul__(other)

    @property
    def T(self) -> "Zonotope":
        """Transpose (for 2D zonotopes)."""
        return Zonotope(self.ox.T, self.G.T)

    # --- Conversion methods ---

    def interval_hull(self) -> "Interval":
        """Compute the interval hull (bounding box) of the zonotope.

        Returns the tightest axis-aligned bounding box containing the zonotope.
        """
        from immrax.inclusion.interval import Interval

        # The interval hull is [c - |G| @ 1, c + |G| @ 1]
        # where |G| is element-wise absolute value
        radius = jnp.sum(jnp.abs(self.G), axis=-1)
        return Interval(self.ox - radius, self.ox + radius)

    def contains(self, x: ArrayLike) -> Array:
        """Check if a point x is contained in the zonotope.

        This checks if there exists v ∈ [-1,1]^m such that x = c + G @ v.
        Returns a boolean array.

        Note: This is a necessary but not sufficient condition when m > n
        (underdetermined system). For exact containment, an LP solver is needed.
        """
        x = jnp.asarray(x)
        # Check if x is within the interval hull first (fast rejection)
        hull = self.interval_hull()
        in_hull = jnp.all((x >= hull.lower) & (x <= hull.upper))

        # For exact check when m <= n, solve G @ v = x - c
        # and verify ||v||_inf <= 1
        # For m > n, this is a relaxation
        diff = x - self.ox
        # Use least squares solution
        v, residuals, rank, s = jnp.linalg.lstsq(self.G, diff)
        in_zonotope = jnp.all(jnp.abs(v) <= 1.0 + 1e-8)

        return in_hull & in_zonotope

    # --- String representation ---

    def __str__(self) -> str:
        return f"Zonotope(center={self.ox}, generators={self.G})"

    def __repr__(self) -> str:
        return f"Zonotope(ox={self.ox!r}, G={self.G!r})"


# --- Helper functions ---


def zonotope(ox: ArrayLike, G: ArrayLike | None = None) -> Zonotope:
    """Create a Zonotope from a center and generator matrix.

    Parameters
    ----------
    ox : ArrayLike
        Center of the zonotope
    G : ArrayLike, optional
        Generator matrix. If None, creates a point zonotope (no generators).

    Returns
    -------
    Zonotope
        The constructed zonotope
    """
    ox = jnp.asarray(ox)
    if G is None:
        # Point zonotope: no generators
        G = jnp.zeros(ox.shape + (0,), dtype=ox.dtype)
    return Zonotope(ox, G)


def zonotope_from_interval(interval: "Interval") -> Zonotope:
    """Create a zonotope from an interval (hyperrectangle).

    The resulting zonotope has n generators (one per dimension),
    each aligned with a coordinate axis.

    Parameters
    ----------
    interval : Interval
        The interval to convert

    Returns
    -------
    Zonotope
        Equivalent zonotope representation
    """
    center = (interval.lower + interval.upper) / 2
    radius = (interval.upper - interval.lower) / 2
    # Generator matrix is diagonal with radii
    G = jnp.diag(radius)
    return Zonotope(center, G)


def zonotope_concatenate(zonotopes: list, axis: int = 0) -> Zonotope:
    """Concatenate zonotopes along an axis (Cartesian product).

    Parameters
    ----------
    zonotopes : list of Zonotope
        Zonotopes to concatenate
    axis : int
        Axis along which to concatenate centers

    Returns
    -------
    Zonotope
        Concatenated zonotope (Cartesian product)
    """
    centers = jnp.concatenate([z.ox for z in zonotopes], axis=axis)
    # For Cartesian product, generators are block diagonal
    n_total = sum(z.n for z in zonotopes)
    m_total = sum(z.m for z in zonotopes)

    G = jnp.zeros((n_total, m_total), dtype=zonotopes[0].dtype)
    row_offset = 0
    col_offset = 0
    for z in zonotopes:
        G = G.at[row_offset : row_offset + z.n, col_offset : col_offset + z.m].set(z.G)
        row_offset += z.n
        col_offset += z.m

    return Zonotope(centers, G)

