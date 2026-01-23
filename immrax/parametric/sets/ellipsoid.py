from .affine import hParametope
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ...utils import null_space
import numpy as onp
from jax.tree_util import register_pytree_node_class
from ...inclusion import icentpert
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@register_pytree_node_class
class Ellipsoid(hParametope):
    def __init__(self, ox, alpha, y):
        # ly = jnp.zeros_like(uy) if uy is not None else None
        super().__init__(ox, alpha, y)

    @classmethod
    def from_parametope(cls, pt: hParametope):
        return Ellipsoid(pt.ox, pt.alpha, pt.y)

    def h(self, a: ArrayLike):
        return jnp.array([-a.T @ a, a.T @ a])

    def hinv(self, y):
        # Returns a box containing the preimage of the constraint over iy
        n = len(self.ox)
        yu = y[1]

        # |x|_inf \leq |x|_2 \leq \sqrt{n} |x|_inf
        return icentpert(jnp.zeros(n), jnp.sqrt(yu) * jnp.ones(n))
        # return icentpert(jnp.zeros(n), yu*jnp.ones(n))

    # def iover (self) :
    #     return self.ginv(self.H@interval(self.ly, self.uy))

    @property
    def P(self):
        return self.alpha.T @ self.alpha

    def V(self, x: ArrayLike):
        ax = self.alpha @ (x - self.ox)
        return ax.T @ ax

    def plot_projection(self, ax, xi=0, yi=1, rescale=False, **kwargs):
        P = self.P / self.y[1]
        n = P.shape[0]
        if n == 2:
            _plot_ellipse(P, self.ox, ax, rescale, **kwargs)
            return
        ind = [k for k in range(n) if k not in [xi, yi]]
        Phat = P[ind, :]
        N = null_space(Phat)
        M = N[(xi, yi), :]  # Since M is guaranteed 2x2,
        Minv = (1 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) * jnp.array(
            [[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]
        )
        Q = Minv.T @ N.T @ P @ N @ Minv
        _plot_ellipse(Q, self.ox[(xi, yi),], ax, rescale, **kwargs)

    @staticmethod
    def get_projection_mtx(P, xi=0, yi=1):
        n = P.shape[0]
        if n == 2:
            return P
        ind = [k for k in range(n) if k not in [xi, yi]]
        Phat = P[ind, :]
        N = null_space(Phat)
        M = N[(xi, yi), :]  # Since M is guaranteed 2x2,
        Minv = (1 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) * jnp.array(
            [[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]
        )
        Q = Minv.T @ N.T @ P @ N @ Minv
        return Q


def _plot_ellipse(
    Q: ArrayLike,
    xc: ArrayLike = jnp.zeros(2),
    ax: "Axes | None" = None,
    rescale: bool = False,
    **kwargs,
):
    """
    Parameters
    ----------
    Q : ArrayLike
        PD matrix defining the ellipse
    xc : ArrayLike, optional
        Center of the ellipse, by default jnp.zeros(2)
    ax : Axes | None, optional
        Matplotlib Axes object to plot the ellipse on, plt.gca() if None, by default None
    rescale : bool, optional
        Rescales the axes to fit the ellipse, by default False

    Raises
    ------
    ValueError
        Q must be a 2x2 matrix
    """
    from matplotlib.patches import Ellipse

    n = Q.shape[0]
    if n != 2:
        raise ValueError(
            "Use _plot_ellipse for 2D ellipses, see Ellipsoid.plot_projection"
        )

    S, U = jnp.linalg.eigh(Q)
    Sinv = 1 / S

    kwargs.setdefault("color", "k")
    kwargs.setdefault("fill", False)
    width, height = 2 * jnp.sqrt(Sinv)
    angle = jnp.arctan2(U[1, 0], U[0, 0]) * 180 / jnp.pi
    ellipse = Ellipse(xy=xc, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    if rescale:
        ax.set_xlim(xc[0] - 1.5 * width, xc[0] + 1.5 * width)
        ax.set_ylim(xc[1] - 1.5 * height, xc[1] + 1.5 * height)


def _plot_annulus(
    Q: ArrayLike,
    xc: ArrayLike = jnp.zeros(2),
    inner: float = 0.5,
    ax: "Axes | None" = None,
    rescale: bool = False,
    **kwargs,
):
    """
    Parameters
    ----------
    Q : ArrayLike
        PD matrix defining the ellipse
    xc : ArrayLike, optional
        Center of the ellipse, by default jnp.zeros(2)
    inner: float, optional
        Inner radius of the annulus, by default 0.5
    ax : Axes | None, optional
        Matplotlib Axes object to plot the ellipse on, plt.gca() if None, by default None
    rescale : bool, optional
        Rescales the axes to fit the ellipse, by default False

    Raises
    ------
    ValueError
        Q must be a 2x2 matrix
    """
    n = Q.shape[0]
    if n != 2:
        raise ValueError(
            "Use _plot_ellipse for 2D ellipses, see Ellipsoid.plot_projection"
        )

    S, U = jnp.linalg.eigh(Q)
    Sinv = 1 / S

    kwargs.setdefault("color", "k")
    kwargs.setdefault("fill", False)
    width, height = 2 * jnp.sqrt(Sinv)
    angle = jnp.arctan2(U[1, 0], U[0, 0]) * 180 / jnp.pi
    annulus = _AnnulusP(xy=xc, r=jnp.sqrt(Sinv), width=inner, angle=angle, **kwargs)
    ax.add_patch(annulus)

    if rescale:
        ax.set_xlim(xc[0] - 1.5 * width, xc[0] + 1.5 * width)
        ax.set_ylim(xc[1] - 1.5 * height, xc[1] + 1.5 * height)


def _get_annulus_patch_class():
    """Lazy import and creation of AnnulusP class."""
    from matplotlib.patches import Patch
    from matplotlib.path import Path
    from matplotlib import transforms

    class AnnulusP(Patch):
        """
        An elliptical annulus.
        Most of the following code is from matplotlib.patches.Annulus.
        There are small modification to make the inner ellipse defined as
            a percentage of the outer ellipse---scaling major and minor axes
            as a multiplier of the outer ellipse's major and minor axes instead of additive.
        """

        def __init__(self, xy, r, width, angle=0.0, **kwargs):
            super().__init__(**kwargs)
            self.set_radii(r)
            self.center = xy
            self.width = width
            self.angle = angle
            self._path = None

        def __str__(self):
            if self.a == self.b:
                r = self.a
            else:
                r = (self.a, self.b)
            return "Annulus(xy=(%s, %s), r=%s, width=%s, angle=%s)" % (
                *self.center, r, self.width, self.angle,
            )

        def set_center(self, xy):
            self._center = xy
            self._path = None
            self.stale = True

        def get_center(self):
            return self._center

        center = property(get_center, set_center)

        def set_width(self, width):
            if width > 1 or width < 0:
                raise ValueError("Width of annulus must be a float between 0 and 1.")
            self._width = width
            self._path = None
            self.stale = True

        def get_width(self):
            return self._width

        width = property(get_width, set_width)

        def set_angle(self, angle):
            self._angle = angle
            self._path = None
            self.stale = True

        def get_angle(self):
            return self._angle

        angle = property(get_angle, set_angle)

        def set_semimajor(self, a):
            self.a = float(a)
            self._path = None
            self.stale = True

        def set_semiminor(self, b):
            self.b = float(b)
            self._path = None
            self.stale = True

        def set_radii(self, r):
            if onp.shape(r) == (2,):
                self.a, self.b = r
            elif onp.shape(r) == ():
                self.a = self.b = float(r)
            else:
                raise ValueError("Parameter 'r' must be one or two floats.")
            self._path = None
            self.stale = True

        def get_radii(self):
            return self.a, self.b

        radii = property(get_radii, set_radii)

        def _transform_verts(self, verts, a, b):
            return (
                transforms.Affine2D()
                .scale(*self._convert_xy_units((a, b)))
                .rotate_deg(self.angle)
                .translate(*self._convert_xy_units(self.center))
                .transform(verts)
            )

        def _recompute_path(self):
            arc = Path.arc(0, 360)
            a, b, w = self.a, self.b, self.width
            v1 = self._transform_verts(arc.vertices, a, b)
            v2 = self._transform_verts(arc.vertices[::-1], a * w, b * w)
            v = onp.vstack([v1, v2, v1[0, :], (0, 0)])
            c = onp.hstack(
                [arc.codes, Path.MOVETO, arc.codes[1:], Path.MOVETO, Path.CLOSEPOLY]
            )
            self._path = Path(v, c)

        def get_path(self):
            if self._path is None:
                self._recompute_path()
            return self._path

    return AnnulusP


# Lazy singleton for AnnulusP class
_AnnulusP_class = None

def _AnnulusP(*args, **kwargs):
    """Create an AnnulusP instance with lazy class loading."""
    global _AnnulusP_class
    if _AnnulusP_class is None:
        _AnnulusP_class = _get_annulus_patch_class()
    return _AnnulusP_class(*args, **kwargs)
