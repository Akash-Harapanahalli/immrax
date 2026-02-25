"""Bunger-style flowpipe generator with remainder reduction heuristics.

Implements shrink wrapping, blunting, and QR preconditioning from
Bunger (2020) to reduce the wrapping effect in Taylor model flowpipes.
These heuristics are applied to the spatial TM at the end of each step,
before it is passed as the initial condition for the next step.

References
----------
.. [1] Bunger, F. "A Taylor model toolbox for solving ODEs implemented in
       MATLAB/INTLAB." J. Comput. Appl. Math. 368 (2020): 112511.
"""

import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval, natif, icentpert
from immrax.system import System
from immrax.utils import inv_fact, prolongation, check_containment
from .. import (
    TaylorModel,
    pjetm,
    pjet,
    tm_integrate_variable,
    taylor_model_concatenate,
    taylor_model_identity,
)
from ..taylor_model import _bound_monomials_over_domain
from ..base import PyTreeShape, _get_leaf_total_degree_exponents
from .base import TMFlowpipeGenerator, tx_tm_eval, tps_to_tx

from typing import Tuple


class BungerTMFlowpipeGenerator(TMFlowpipeGenerator):
    """Flowpipe generator with Bunger's remainder reduction heuristics.

    Extends the basic Picard-based approach with post-step heuristics
    to reduce the wrapping effect:

    - **Shrink wrapping**: absorbs the interval remainder into the
      polynomial by slightly enlarging non-constant coefficients.
    - **QR preconditioning**: applies a QR-based coordinate change to
      the linear part of the TM, improving conditioning and reducing
      directional wrapping.

    Parameters
    ----------
    sys : System
        The dynamical system.
    shrink_wrap : bool
        Enable shrink wrapping (default True).
    precondition : bool
        Enable QR preconditioning (default False).
    """

    def __init__(self, sys: System, *, shrink_wrap=True, precondition=False):
        super().__init__(sys)
        self.shrink_wrap = shrink_wrap
        self.precondition = precondition

    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps=4096, **kwargs):
        self.t_order = t_order
        self._prolonged_f = prolongation(self.sys.f, t_order)
        self.eps = kwargs.get("eps", 1e-2)
        self.delta = kwargs.get("delta", 1e-2)

    def _picard(self, tm_tx: TaylorModel, ic_remainder=None, u=None) -> TaylorModel:
        """Apply the Picard operator (identical to BasicTMFlowpipeGenerator)."""
        tm_ic = tx_tm_eval(tm_tx, tm_tx.domain[0].lower, self.t_order)

        tm_id = taylor_model_identity(tm_tx.domain, order=tm_tx._per_leaf_order)
        if u is not None:
            f_tm_tx = pjetm(self.sys.f)(tm_id[0:1], tm_tx, u)
        else:
            f_tm_tx = pjetm(self.sys.f)(tm_id[0:1], tm_tx)
        tm_int = tm_integrate_variable(f_tm_tx, var_idx=0, keep_order=True)

        con_sh = tm_ic.coeffs.shape
        coeffs = tm_int.coeffs.at[: con_sh[0], : con_sh[1]].add(tm_ic.coeffs)

        if ic_remainder is None:
            ic_remainder = tm_ic.remainder

        return TaylorModel(
            coeffs=coeffs,
            exponents=tm_int.exponents,
            remainder=tm_int.remainder + ic_remainder,
            flat_domain=tm_int.flat_domain,
            flat_center=tm_int.flat_center,
            _input_pytree=tm_int._input_pytree,
            _output_pytree=tm_int._output_pytree,
            _per_leaf_order=tm_int._per_leaf_order,
        )

    # ------------------------------------------------------------------
    # Heuristic 1: Shrink Wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _shrink_wrap(tm: TaylorModel) -> TaylorModel:
        r"""Absorb the interval remainder into the polynomial.

        Write the polynomial as ``q(x) = c₀ + q̃(x)`` where ``q̃`` is
        the non-constant part.  Let the non-constant range over the
        domain B be ``[nonconst_lo, nonconst_hi]`` with half-width
        ``poly_radius`` and center ``nonconst_center``.

        We absorb most of the remainder into the polynomial by scaling
        non-constant coefficients and adjusting the constant term.
        After construction, a "verify and repair" step ensures the new
        TM's interval hull rigorously contains the original hull,
        compensating for any floating-point rounding in
        ``_bound_polynomial``.

        Dimensions where the polynomial has negligible range
        (``poly_radius < 1e-10``) are left unchanged.
        """
        poly_bounds = tm._bound_polynomial()  # Interval, shape (n,)
        c0 = tm.constant_term                  # (n,)

        # Non-constant polynomial range (contains 0 since q̃(center) = 0)
        nonconst_lo = poly_bounds.lower - c0   # (n,), ≤ 0
        nonconst_hi = poly_bounds.upper - c0   # (n,), ≥ 0
        poly_radius = (nonconst_hi - nonconst_lo) / 2  # (n,)
        nonconst_center = (nonconst_lo + nonconst_hi) / 2  # (n,)

        rem_pert = tm.remainder.pert      # (n,)
        rem_center = tm.remainder.center  # (n,)

        # Only shrink-wrap dimensions with meaningful polynomial range
        should_wrap = poly_radius > 1e-10  # (n,)
        safe_pr = jnp.maximum(poly_radius, 1e-15)

        total_radius = poly_radius + rem_pert
        scale = jnp.where(should_wrap, total_radius / safe_pr, 1.0)

        # Constant shift: preserve the range center.
        # c_new = c0 + rem_center + (1 - scale) * nonconst_center
        delta_const = jnp.where(
            should_wrap,
            rem_center + (1.0 - scale) * nonconst_center,
            0.0,
        )

        is_constant = jnp.all(tm.exponents == 0, axis=0)  # (m,)

        new_coeffs = jnp.where(
            is_constant,
            tm.coeffs + delta_const[:, None],
            tm.coeffs * scale[:, None],
        )

        # --- Verify and repair: ensure hull is preserved despite FP ---
        target_hull = poly_bounds + tm.remainder  # original hull

        # Build the new TM with zero remainder first to measure poly bounds
        new_tm = TaylorModel(
            coeffs=new_coeffs,
            exponents=tm.exponents,
            remainder=interval(jnp.zeros_like(c0)),
            flat_domain=tm.flat_domain,
            flat_center=tm.flat_center,
            _input_pytree=tm._input_pytree,
            _output_pytree=tm._output_pytree,
            _per_leaf_order=tm._per_leaf_order,
        )
        new_poly_bounds = new_tm._bound_polynomial()

        # Remainder = gap between target hull and new poly bounds.
        # lo: target.lo - new_poly.lo (should be ≤ 0: new_poly reaches lower)
        # hi: target.hi - new_poly.hi (should be ≥ 0: new_poly reaches higher)
        # If FP rounding made the poly not reach far enough, the
        # remainder picks up the slack.
        gap_lo = target_hull.lower - new_poly_bounds.lower  # want ≤ 0
        gap_hi = target_hull.upper - new_poly_bounds.upper  # want ≥ 0
        # Repair: remainder must cover any undershoot
        fix_lo = jnp.minimum(gap_lo, 0.0)  # negative or 0
        fix_hi = jnp.maximum(gap_hi, 0.0)  # positive or 0
        fix_rem = interval(fix_lo, fix_hi)

        # For unwrapped dims, keep original remainder
        final_rem_lo = jnp.where(should_wrap, fix_rem.lower, tm.remainder.lower)
        final_rem_hi = jnp.where(should_wrap, fix_rem.upper, tm.remainder.upper)

        return TaylorModel(
            coeffs=new_coeffs,
            exponents=tm.exponents,
            remainder=interval(final_rem_lo, final_rem_hi),
            flat_domain=tm.flat_domain,
            flat_center=tm.flat_center,
            _input_pytree=tm._input_pytree,
            _output_pytree=tm._output_pytree,
            _per_leaf_order=tm._per_leaf_order,
        )

    # ------------------------------------------------------------------
    # Heuristic 2: QR Preconditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _precondition_qr(tm: TaylorModel) -> TaylorModel:
        r"""QR-based preconditioning of the spatial Taylor model.

        Extracts the linear part *A* of the polynomial, computes
        ``A = Q R``, and applies :math:`Q^{-1} = Q^\top` to all
        coefficients and the remainder.  The result has an upper-
        triangular linear part *R*, which is typically better
        conditioned and reduces directional wrapping.
        """
        n = tm.coeffs.shape[0]  # output dimension
        d = tm.exponents.shape[0]  # domain dimension

        # --- Extract linear coefficient matrix A (n x d) ---
        eye_d = jnp.eye(d, dtype=jnp.int32)
        # is_linear_var[j, k] = True if exponent column k equals e_j
        is_linear_var = jnp.all(
            tm.exponents[None, :, :] == eye_d[:, :, None], axis=1
        )  # (d, m)
        A = tm.coeffs @ is_linear_var.T.astype(tm.coeffs.dtype)  # (n, d)

        # --- QR factorization ---
        Q, _R = jnp.linalg.qr(A)  # Q: (n, n), R: (n, d)

        # Apply Q^T to all coefficients
        new_coeffs = Q.T @ tm.coeffs  # (n, m)

        # Rigorous rotation of the remainder interval:
        # Q^T @ [center ± pert] ⊆ Q^T @ center ± |Q^T| @ pert
        new_rem_center = Q.T @ tm.remainder.center
        new_rem_pert = jnp.abs(Q.T) @ tm.remainder.pert
        new_remainder = icentpert(new_rem_center, new_rem_pert)

        return TaylorModel(
            coeffs=new_coeffs,
            exponents=tm.exponents,
            remainder=new_remainder,
            flat_domain=tm.flat_domain,
            flat_center=tm.flat_center,
            _input_pytree=tm._input_pytree,
            _output_pytree=tm._output_pytree,
            _per_leaf_order=tm._per_leaf_order,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, t: float, tmi: TaylorModel, dt_max: float, **kwargs):
        u = kwargs.get("u", None)

        # Step 1: Taylor expansion of the flow map
        if u is not None:
            poly_coeffs = pjet(lambda x: self._prolonged_f(t, x, u))(tmi.polynomial)
        else:
            poly_coeffs = pjet(lambda x: self._prolonged_f(t, x))(tmi.polynomial)
        poly = tps_to_tx(
            poly_coeffs,
            interval(jnp.zeros_like(tmi.remainder.lower)),
            (interval(t, t + dt_max), tmi.domain),
            (t, tmi.center),
            per_leaf_order=(self.t_order,) + tmi._per_leaf_order,
        )

        # Step 2: eps-inflation Picard contraction check
        def _remainder_inflation(rem):
            return rem * icentpert(1.0, self.eps) + icentpert(0.0, self.delta)

        def _check_picard(i, carry):
            poly, contractive = carry
            poly.remainder = jax.lax.cond(
                contractive,
                lambda rem: rem,
                lambda rem: _remainder_inflation(rem),
                poly.remainder,
            )
            contractive = jax.lax.cond(
                contractive,
                lambda: contractive,
                lambda: (
                    check_containment(
                        self._picard(poly, ic_remainder=tmi.remainder, u=u).remainder,
                        poly.remainder,
                    )
                    == 1
                ),
            )
            return (poly, contractive)

        poly, contractive = jax.lax.fori_loop(0, 1000, _check_picard, (poly, False))

        # FP safety margin: JAX does not support directed rounding, so
        # interval arithmetic may underestimate bounds.  Inflate the
        # validated remainder slightly to compensate.
        poly.remainder = poly.remainder + icentpert(
            jnp.zeros_like(poly.remainder.lower),
            jnp.maximum(poly.remainder.pert * 1e-4, 1e-12),
        )

        # Step 3: Extract spatial TM at t + dt
        tmf = tx_tm_eval(poly, t + dt_max, self.t_order)

        # Step 4: Apply remainder reduction heuristics
        if self.shrink_wrap:
            tmf = self._shrink_wrap(tmf)
        if self.precondition:
            tmf = self._precondition_qr(tmf)

        return dt_max, tmf, poly, contractive
