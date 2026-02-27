import math
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from itertools import combinations_with_replacement
from .interval import Interval, interval, isinterval
from .jacobian import Permutation, standard_permutation
from .nif import natif
from ..taylor import TaylorPolynomial, pjet, identijet
from ..taylor.base import MultiIndex


def mdit(f, p):
    """TODO: docstring"""

    def _mdit(ix, xc, permutation=None):
        def f_jet(x):
            return pjet(f)(identijet(x, p + 1))

        def f_jet_pp1(x):
            return f_jet(x).get_order(p + 1).coeffs

        # Natural inclusion function for the (p+1) coeffs of the Taylor expansion
        f_jet_pp1_nif = natif(f_jet_pp1)

        expansion = f_jet(xc)
        pp1 = expansion.get_order(p + 1)

        permutation = (
            standard_permutation(len(xc)) if permutation is None else permutation
        )[0]

        _z = ix.lower
        z_ = ix.upper
        zc = xc
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

        # TODO: a bit of a hack for now, implement a Polynomial class that can handle
        # interval coefficient matrices. For now this will do.

        bounded_coeffs = jax.vmap(f_jet_pp1_nif, out_axes=1)(Z)

        # MDIT: M(α) = (1/k) * sum_{j=1}^n α_j f_jet_pp1_nif(Z_j)

        M = natif(jnp.sum)(
            bounded_coeffs * pp1.multiindices.to_numpy()[None, :], axis=1
        ) / (p + 1)

        return expansion, M

    return _mdit


def dit(f, p):
    pass
