import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import taylor_model, TaylorModel, TaylorPolynomial, nattm, nattp, tm_integrate_variable, taylor_model_concatenate
from .base import TMFlowpipeGenerator

from typing import Tuple


def tx_tm_eval(tm: TaylorModel, t: float) -> TaylorModel:
    """Evaluate a time-dependent Taylor Model at a given time.

    Parameters
    ----------
    tm : TaylorModel
        The time-dependent Taylor Model to evaluate.
    t : float
        The time at which to evaluate the Taylor Model.
    t_order : int
        The order of the Taylor expansion.

    Returns
    -------
    TaylorModel
        The spatial Taylor Model extracted from evaluation at time t.
    """

    # Since t is always the first variable, the exponent matrix is structured
    # in [0,0,0,...,1,1,1,...,t_order,t_order,t_order]
    # i.e., tm.exponents[1:, L * i : L * (i + 1)] is the same for every i

    t_order = tm.exponents[0, -1]
    L = tm.exponents.shape[1] // (t_order + 1)

    exponents = tm.exponents[1:, 0:L]
    coeffs_list = jnp.split(tm.coeffs, t_order + 1, axis=1)
    t_shifted = t - tm.flat_center[0]
    coeffs = jnp.sum(
        jnp.asarray(coeffs_list) * (t_shifted ** jnp.arange(t_order + 1))[:, None, None],
        axis=0,
    )

    return TaylorModel(
        coeffs=coeffs,
        exponents=exponents,
        remainder=tm.remainder,
        flat_domain=tm.flat_domain[1:],
        flat_center=tm.flat_center[1:],
        _domain_treedef=tm._domain_treedef.children()[1],
        _leaf_shapes=tm._leaf_shapes[1:],
        _per_leaf_order=tm._per_leaf_order[1:],
    )


def tps_to_tx(
    polys: list[TaylorPolynomial], remainder, domain, center
) -> TaylorPolynomial:
    """Convert a list of TaylorPolynomials to a single TaylorPolynomial with canonical
    coefficient structure, over the domain

    Parameters
    ----------
    polys : list[TaylorPolynomial]
        List of TaylorPolynomials to convert.

    Returns
    -------
    TaylorPolynomial
        The TaylorPolynomial with canonical coefficient structure.
    """
    t_order = len(polys) - 1
    mon_len = polys[0].exponents.shape[-1]
    coeffs = jnp.concatenate(
        [polys[i].coeffs * inv_fact(i) for i in range(t_order + 1)], axis=1
    )
    exponents_top = jnp.repeat(jnp.arange(t_order + 1), mon_len)
    exponents_bottom = jnp.concatenate([p.exponents for p in polys], axis=1)

    return taylor_model(
        coeffs=coeffs,
        exponents=jnp.vstack((exponents_top, exponents_bottom)),
        remainder=remainder,
        domain=domain,
        center=center,
    )


class BasicTMFlowpipeGenerator(TMFlowpipeGenerator):
    """A basic implementation of a Picard-based flowpipe generator."""

    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps=4096, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""
        self.t_order = t_order
        self._prolonged_f = prolongation(self.sys.f, t_order)

        # Parameters for the eps-delta inflation
        self.eps = kwargs.get("eps", 1e-2)
        self.delta = kwargs.get("delta", 1e-2)


    def _picard(self, tm_tx:TaylorModel) -> TaylorModel :
        """Apply the Picard operator to the TaylorModel tm_tx over the domain of the TaylorModel.

        The Picard operator is defined as:
        K : C([t0,t1], R^n) -> C([t0,t1], R^n),
        K(x, t) = x(t0) + \\int_{t0}^t f(x(s), s) ds

        Applied to a TaylorModel tm_tx, this operator uses TaylorModel arithmetic to compute:
        K(p + E) = (q + J) + \\int_{t0}^t f(p + E, s) ds

        As a special case, if the polynomial part of the TaylorModel is the Taylor series expansion
        of the flow map, the polynomial part is a fixed point (to the corresponding order), meaning
        K(p + E) = p + E'
        where E' contains all the remainders.
        """
        # Initial condition
        tm_ic = tx_tm_eval(tm_tx, tm_tx.domain[0].lower)

        # Construct augmented TM: (t, x0) -> (t, phi_1, ..., phi_n)
        # This is needed because structured_center=True slices the TM's *output*
        # according to leaf_shapes. The flow map tm_tx has output shape (n,) but
        # the domain has total dim 1+n, so we prepend a time-identity TM.
        is_constant = jnp.all(tm_tx.exponents == 0, axis=0)
        t_unit = jnp.zeros(tm_tx.d, dtype=jnp.int32).at[0].set(1)
        is_t_linear = jnp.all(tm_tx.exponents == t_unit[:, None], axis=0)
        t_coeffs = (
            jnp.where(is_constant, tm_tx.flat_center[0], 0.0)
            + jnp.where(is_t_linear, 1.0, 0.0)
        )[None, :]  # shape (1, num_monomials)

        tm_t = TaylorModel(
            t_coeffs,
            tm_tx.exponents,
            interval(jnp.zeros(1)),
            tm_tx.flat_domain,
            tm_tx.flat_center,
            _domain_treedef=tm_tx._domain_treedef,
            _leaf_shapes=tm_tx._leaf_shapes,
            _per_leaf_order=tm_tx._per_leaf_order,
        )
        tm_aug = taylor_model_concatenate([tm_t, tm_tx])

        # Integration
        f_tm_tx = nattm(self.sys.f, structured_center=True)(tm_aug)
        tm_int = tm_integrate_variable(f_tm_tx, var_idx=0, keep_order=True)

        con_sh = tm_ic.coeffs.shape

        coeffs = tm_int.coeffs.at[:con_sh[0], :con_sh[1]].add(tm_ic.coeffs)

        return TaylorModel(
            coeffs=coeffs,
            exponents=tm_int.exponents,
            remainder=tm_int.remainder,
            flat_domain=tm_int.flat_domain,
            flat_center=tm_int.flat_center,
            _domain_treedef=tm_int._domain_treedef,
            _leaf_shapes=tm_int._leaf_shapes,
            _per_leaf_order=tm_int._per_leaf_order,
        )

    def _fixed_picard (self, tm_tx:TaylorModel) -> TaylorModel :
        """A more efficient version of _picard when the polynomial part of the TaylorModel is the Taylor series expansion
        of the flow map, the polynomial part is a fixed point (to the corresponding order), meaning
        K(p + E) = p + E'
        where E' contains all the remainders.
        """

    def _step(self, t: float, tmi: TaylorModel, dt_max: float, **kwargs):
        # Step 1: Compute the Taylor expansion of the flow map to t_order, x_order

        poly_coeffs = nattp(lambda x: self._prolonged_f(t, x))(tmi.polynomial)
        poly = tps_to_tx(
            poly_coeffs,
            tmi.remainder,
            (interval(t, t + dt_max), tmi.domain),
            (t, tmi.center),
        )

        # Step 2: Check the contraction of the Picard operator

        picard_poly = self._picard(poly)
        
        return poly, picard_poly


