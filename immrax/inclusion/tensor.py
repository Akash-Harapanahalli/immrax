import jax
import jax.numpy as jnp

from taylax import FullTensorPolynomial, pjet

from .interval import interval
from .jacobian import standard_permutation
from .nif import natif


def mdit(f, p):
    r"""Mixed Derivative Interval Tensor (MDIT) of order ``p+1``.

    Returns a function ``_mdit(ix, xc, permutation=None)`` giving the order-``p``
    Taylor expansion of ``f`` at ``xc`` together with the order-``(p+1)`` MDIT
    remainder over the box ``ix``. For ``C^{p+1}`` ``f`` and any ``x in ix`` (with
    ``xc in ix``) the inclusion holds:

    .. math::
        f(x) \in f(x') + \sum_{i=1}^p L^i f(x') [(x-x')^{\otimes i}]
                 + M^{p+1}_{x'} f(ix) [(x-x')^{\otimes (p+1)}].

    The remainder coefficient of multi-index ``beta`` (``|beta| = p+1``) is

    .. math::
        M_\beta = \frac{1}{p+1} \sum_{j=1}^n \beta_j \,
                  \frac{\partial^\beta f}{\beta!}(X_1,\dots,X_j,x'_{j+1},\dots,x'_n),

    i.e. the order-``(p+1)`` Taylor coefficient evaluated at the mixed replacement
    points (interval in the first ``j`` coordinates, fixed at ``x'`` in the rest),
    weighted by ``beta_j`` and averaged. The high-order Taylor coefficients are
    propagated with :mod:`taylax`.

    Returns
    -------
    (expansion, M)
        ``expansion`` is the :class:`taylax.FullTensorPolynomial` Taylor expansion
        of ``f`` at ``xc`` to order ``p+1``; ``M`` is the interval array of the
        order-``(p+1)`` MDIT remainder coefficients, shape ``(*f_out, N_{p+1})``.
    """

    def _mdit(ix, xc, permutation=None):
        def f_jet(x):
            return pjet(f)(FullTensorPolynomial.identity(x, order=p + 1))

        def f_jet_pp1(x):
            return f_jet(x).get_order(p + 1).coeffs

        # Natural inclusion function for the (p+1) coeffs of the Taylor expansion.
        f_jet_pp1_nif = natif(f_jet_pp1)

        expansion = f_jet(xc)
        pp1 = expansion.get_order(p + 1)

        permutation = (
            standard_permutation(len(xc)) if permutation is None else permutation
        )[0]

        # Mixed replacement points: row j is the box on the first j (permuted)
        # coordinates and the fixed center xc on the rest.
        _z, z_, zc = ix.lower, ix.upper, xc
        Z = interval(
            jnp.where(
                permutation.mtx,
                jnp.tile(_z, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
            jnp.where(
                permutation.mtx,
                jnp.tile(z_, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
        )

        # Order-(p+1) Taylor coeffs at each mixed point: (*f_out, n, N_{p+1}).
        bounded_coeffs = jax.vmap(f_jet_pp1_nif, out_axes=1)(Z)

        # M_beta = (1/(p+1)) sum_j beta_j * coeff_beta(Z_j). multiindices.to_numpy()
        # is (n, N_{p+1}); broadcast over the output axis and sum over j.
        M = natif(jnp.sum)(
            bounded_coeffs * pp1.multiindices.to_numpy()[None, :], axis=1
        ) / (p + 1)

        return expansion, M

    return _mdit


def dit(f, p):
    pass
