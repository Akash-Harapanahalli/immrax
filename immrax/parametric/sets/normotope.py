import jax
import jax.numpy as jnp
import equinox as eqx
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Float
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from ...inclusion import Interval, interval, icentpert, i2centpert, mjacM
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from .polytope import Polytope
from .ellipsoid import Ellipsoid
from ...utils import get_rohn_corners, get_corners

from functools import partial
from math import sqrt
from itertools import product


@register_pytree_node_class
class Normotope(Parametope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\| \leq y}

    where :math:`\|\cdot\|` is a norm, :math:`\ox` is the center, :math:`H` is a shaping matrix, and :math:`y` is the offset.

    Define :math:`h` as the norm in subclasses, and :math:`\mu` as the logarithmic norm associated to :math:`h`.
    """

    def __init__(
        self, ox: ArrayLike, alpha: ArrayLike, y: Float, alpha_inv: Array | None = None
    ):
        super().__init__(ox, alpha, y)
        self._alpha_inv = alpha_inv

    def g(self, x: Array) -> Float:
        return self.norm(jnp.dot(self.alpha, x - self.ox))

    @property
    def alpha_inv(self) -> Array:
        if self._alpha_inv is None:
            self._alpha_inv = jnp.linalg.inv(self.alpha)
        return self._alpha_inv

    @classmethod
    def norm(self, z: Array) -> Float:
        """The norm defining the normotope."""
        raise NotImplementedError("Subclasses must implement the norm method.")

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        """Computes the induced norm of A."""
        raise NotImplementedError("Subclasses must implement the induced_norm method.")

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        """The logarithmic norm associated to h."""
        raise NotImplementedError(
            "Subclasses must implement the logarithmic_norm method."
        )

    @classmethod
    def mu(cls, A: Array) -> Float:
        """Alias for the logarithmic norm."""
        return cls.logarithmic_norm(A)

    @classmethod
    def norm_ball_iover(cls, n: int) -> Interval:
        """An interval overapproximation of the norm ball of radius 1 in R^n."""
        raise NotImplementedError(
            "Subclasses must implement the norm_ball_iover method."
        )

    def iover(self) -> Interval:
        """An interval overapproximation of the normotope, defaults to interval analysis"""
        return (
            interval(self.alpha_inv) @ (self.norm_ball_iover(self.ox.shape[0]) * self.y)
            + self.ox
        )

    def plot_projection(
        self, ax: "Axes", xi: int = 0, yi: int = 1, rescale: bool = False, **kwargs
    ) -> None:
        """Plot the projection of the normotope onto the xi-yi plane."""
        raise NotImplementedError(
            "Subclasses must implement the plot_projection method."
        )

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "Normotope":
        return Normotope(pt.ox, pt.alpha, pt.y)

    def vec(self) -> Array:
        """Vectorizes the normotope into a one dimensional array."""
        return jnp.concatenate(
            (self.ox, self.alpha.reshape(-1), jnp.atleast_1d(self.y))
        )

    @classmethod
    def unvec(cls, vec: Array, n: int | None = None) -> "Normotope":
        """Unvectorizes a vector into a normotope."""
        y = vec[-1]
        N = len(vec) - 1
        if n is None:
            # Assume alpha is nxn, so N = n*n + n = n*(n+1)
            # QF: n^2 + n - N = 0 ==> n = (-1 + sqrt(1 + 4*N)) / 2
            n = int((sqrt(1 + 4 * N) - 1) // 2)
        # if alpha is mxn, N = m*n + n = m*(n+1)

        ox = vec[:n]
        alpha = vec[n:N].reshape(-1, n)
        return cls(ox, alpha, y)


class NormotopeEmbedding(ParametricEmbedding):
    r"""Embedding of a :class:`~immrax.system.System` onto :class:`Normotope`
    dynamics.

    Parameters
    ----------
    sys : System
        The system to embed.
    gsc : Callable[[Interval], list], optional
        Corner-selection strategy mapping the interval Jacobian :math:`[M]` to
        the corner matrices used in the log-norm contraction bound. Defaults to
        ``partial(get_rohn_corners, sign='+')`` (exact for :class:`L2Normotope`);
        other choices include :func:`get_corners` or a :func:`get_sparse_corners`
        instance. Static and construction-time, not inferred from the set.
    kappa : float, optional
        Strength of the soft ``y``-clamp toward an external box ``ix``. When
        ``ix`` is supplied to :meth:`_dynamics`, ``-kappa * relu(y - y_box)`` is
        added to ``y``'s dynamics with ``y_box = max_i ||alpha (x_i - ox)||`` over
        the corners of ``ix``. Active only where ``y > y_box`` — where the static
        enclosure already holds with slack — so the certificate is preserved.
        Defaults to ``0.0`` (no clamp); ignored when ``ix`` is ``None``.
    """

    gsc: Callable = eqx.field(
        static=True, default=partial(get_rohn_corners, sign="+")
    )
    kappa: float = eqx.field(static=True, default=0.0)

    def _initialize(self, nt0: Normotope) -> ArrayLike:
        if not isinstance(nt0, Normotope):
            raise ValueError(f"{nt0=} is not a Normotope needed for NormotopeEmbedding")
        # No auxiliary state to evolve; everything is recomputed in _dynamics.
        return None

    def hypercontrol_shape(self, nt0: Normotope) -> tuple:
        # The control is added to H_dot (the shaping-matrix derivative).
        return nt0.alpha.shape

    def _dynamics(self, t, state, U=None, *, perm=None, adjoint=True, ix=None):
        r"""Normotope embedding dynamics.

        Parameters
        ----------
        ix : Interval, optional
            A box intersected with the normotope's own ``iover()`` for the
            mixed-Jacobian evaluation; tighter ``ix`` only reduces conservatism.
            Defaults to ``None`` (plain ``nt.iover()``). With ``kappa > 0`` it
            also enables the soft ``y``-clamp (see ``kappa``).
        """
        nt, aux = state
        Ut = U.reshape(nt.alpha.shape) if U is not None else jnp.zeros_like(nt.alpha)

        H = nt.alpha
        Hp = nt.alpha_inv
        y = nt.y

        # Derived at call time (not frozen in __init__) so vmap/jit over the
        # system's parameters thread through.
        A = jax.jacfwd(self.sys.f, 1)(0.0, nt.ox)

        if adjoint:
            H_dot = -H @ A + Ut
        else:
            H_dot = Ut

        # Mixed Jacobian over the tightest valid enclosure: the normotope's own
        # hull, optionally intersected with an external box ix.
        box = nt.iover() if ix is None else (ix & nt.iover())

        MM = mjacM(self.sys.f)(
            t, box, center=(jnp.zeros(1), nt.ox), permutation=perm
        )
        Mx = MM[1]

        mus = [nt.mu(H_dot @ Hp + H @ M @ Hp) for M in self.gsc(interval(Mx))]
        c = jnp.max(jnp.asarray(mus))
        y_dot = c * y

        # Soft clamp of y toward the tightened box (see kappa). The box corners
        # are frozen certificate data, so gradients flow only through (H, ox), and
        # the pull is active only where the static enclosure already holds.
        if ix is not None and self.kappa != 0.0:
            corners = get_corners(box)
            y_box = jnp.max(jax.vmap(nt.g)(corners))
            y_dot = y_dot - self.kappa * jnp.maximum(y - y_box, 0.0)

        return nt.__class__(self.sys.f(0.0, nt.ox), H_dot, y_dot), None


@register_pytree_node_class
class LinfNormotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_\infty \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The infinity norm"""
        return jnp.max(jnp.abs(z))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_\infty` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=1))

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the logarithmic :math:`\ell_\infty` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=1))

    @classmethod
    def norm_ball_iover(cls, n: int) -> Interval:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def to_polytope(self) -> Polytope:
        n = self.alpha.shape[0]
        return Polytope(self.ox, self.alpha, jnp.ones(2 * n) * self.y)

    def plot_projection(self, ax: "Axes", xi=0, yi=1, rescale=False, **kwargs) -> None:
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)

    @classmethod
    def from_interval(cls, *args) -> "LinfNormotope":
        cent, pert = i2centpert(interval(*args))
        return LinfNormotope(cent, jnp.diag(1 / pert), 1.0)

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "LinfNormotope":
        return LinfNormotope(pt.ox, pt.alpha, pt.y)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "LinfNormotope":
        return LinfNormotope(nt.ox, nt.alpha, nt.y)


@register_pytree_node_class
class L1Normotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_1 \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The L1 norm"""
        return jnp.sum(jnp.abs(z))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_1` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=0))

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the logarithmic :math:`\ell_1` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=0))

    @classmethod
    def norm_ball_iover(cls, n: int) -> Array:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def to_polytope(self) -> Polytope:
        # S has rows for all sign combinations of length n
        n = self.alpha.shape[0]
        S = jnp.array(list(product(*[[1, -1]] * n)))
        return Polytope(self.ox, S @ self.alpha, jnp.ones(2 * 2**n) * self.y)

    def plot_projection(self, ax: "Axes", xi=0, yi=1, rescale=False, **kwargs) -> None:
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)

    @classmethod
    def from_interval(cls, *args) -> "L1Normotope":
        cent, pert = i2centpert(interval(*args))
        return L1Normotope(cent, jnp.diag(1 / pert), 1.0)

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "L1Normotope":
        return L1Normotope(pt.ox, pt.alpha, pt.y)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "L1Normotope":
        return L1Normotope(nt.ox, nt.alpha, nt.y)


@register_pytree_node_class
class L2Normotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_2 \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The L_2 norm"""
        return jnp.sum(z**2) ** 0.5

    @classmethod
    def norm_ball_iover(cls, n: int) -> Array:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def iover(self) -> Interval:
        """Tightest interval overapproximation of an L2 normotope."""
        Pinv = self.alpha_inv @ self.alpha_inv.T * self.y**2
        return icentpert(self.ox, jnp.sqrt(jnp.diag(Pinv)))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_2` norm of A"""
        return jnp.linalg.norm(A, ord=2)

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the :math:`\ell_2` logarithmic norm of A"""
        return jnp.max(jnp.linalg.eigvalsh((A + A.T) / 2))

    def plot_projection(
        self, ax: "Axes", xi: int = 0, yi: int = 1, rescale: bool = False, **kwargs
    ) -> None:
        Ellipsoid(self.ox, self.alpha / self.y, jnp.array([0.0, 1.0])).plot_projection(
            ax, xi, yi, rescale, **kwargs
        )

    @classmethod
    def from_interval(cls, *args) -> "L2Normotope":
        cent, pert = i2centpert(interval(*args))
        rn = jnp.sqrt(len(cent))
        return L2Normotope(cent, jnp.diag(1 / (rn * pert)), 1.0)

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "L2Normotope":
        return L2Normotope(pt.ox, pt.alpha, pt.y)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "L2Normotope":
        return L2Normotope(nt.ox, nt.alpha, nt.y)

    def sample_boundary(self, key: jax.random.PRNGKey, num_samples: int) -> Array:
        """Samples points uniformly along the boundary of the L2Normotope."""
        gauss = jax.random.multivariate_normal(
            key, jnp.zeros_like(self.ox), jnp.eye(len(self.ox)), shape=(num_samples,)
        )
        unif_Sn = gauss / jnp.linalg.norm(gauss, axis=-1, keepdims=True)
        alpha_inv = jnp.linalg.inv(self.alpha / self.y)
        unif_nt = jax.vmap(lambda x: alpha_inv @ x + self.ox)(unif_Sn)
        return unif_nt
