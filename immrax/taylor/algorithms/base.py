"""Shared utilities for validated ODE integration algorithms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod

from immrax.inclusion import Interval, interval
from immrax.system import System
from immrax.utils import inv_fact, prolongation
from .. import TaylorModel, TaylorPolynomial, taylor_model, _pytree_to_flattened_array
from ..base import PyTreeShape

from typing import Tuple


def _tube_to_data(tm):
    """Extract scan-safe raw arrays from a tube TaylorModel."""
    return (
        tm.coeffs,
        tm.remainder.lower,
        tm.remainder.upper,
        tm.flat_domain.lower,
        tm.flat_domain.upper,
        tm.flat_center,
    )


def tx_tm_eval(tm: TaylorModel, t: float, t_order: int) -> TaylorModel:
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

    # t_order = tm.exponents[0, -1]
    L = tm.exponents.shape[1] // (t_order + 1)

    exponents = tm.exponents[1:, 0:L]
    coeffs_list = jnp.split(tm.coeffs, t_order + 1, axis=1)
    t_shifted = t - tm.flat_center[0]
    coeffs = jnp.sum(
        jnp.asarray(coeffs_list)
        * (t_shifted ** jnp.arange(t_order + 1))[:, None, None],
        axis=0,
    )

    new_input_pytree = PyTreeShape(
        tm._domain_treedef.children()[1],
        tm._leaf_shapes[1:],
    )
    return TaylorModel(
        coeffs=coeffs,
        exponents=exponents,
        remainder=tm.remainder,
        flat_domain=tm.flat_domain[1:],
        flat_center=tm.flat_center[1:],
        _input_pytree=new_input_pytree,
        _output_pytree=tm._output_pytree,
        _per_leaf_order=tm._per_leaf_order[1:],
    )


def tps_to_tx(
    polys: list[TaylorPolynomial],
    remainder,
    domain,
    center,
    *,
    per_leaf_order=None,
) -> TaylorModel:
    """Convert a list of TaylorPolynomials to a single TaylorModel with canonical
    coefficient structure, over the domain.

    Parameters
    ----------
    polys : list[TaylorPolynomial]
        List of TaylorPolynomials to convert.
    remainder : Interval
        Remainder interval.
    domain : PyTree[Interval]
        Domain for the resulting TaylorModel.
    center : PyTree[ArrayLike]
        Expansion center for the resulting TaylorModel.
    per_leaf_order : tuple[int, ...], optional
        Per-leaf polynomial orders. If provided, avoids inferring orders from
        exponents (required for JIT compatibility).

    Returns
    -------
    TaylorModel
        The TaylorModel with canonical coefficient structure.
    """
    t_order = len(polys) - 1
    mon_len = polys[0].exponents.shape[-1]
    coeffs = jnp.concatenate(
        [polys[i].coeffs * inv_fact(i) for i in range(t_order + 1)], axis=1
    )
    exponents_top = jnp.repeat(jnp.arange(t_order + 1), mon_len)
    exponents_bottom = jnp.concatenate([p.exponents for p in polys], axis=1)
    exponents = jnp.vstack((exponents_top, exponents_bottom))

    if per_leaf_order is not None:
        _domain_treedef, _leaf_shapes, flat_domain = _pytree_to_flattened_array(domain)
        _, _, flat_center = _pytree_to_flattened_array(center)
        return TaylorModel(
            coeffs,
            exponents,
            remainder,
            flat_domain,
            flat_center,
            _input_pytree=PyTreeShape(_domain_treedef, _leaf_shapes),
            _output_pytree=PyTreeShape.flat(coeffs.shape[:-1]),
            _per_leaf_order=per_leaf_order,
        )

    return taylor_model(
        coeffs=coeffs,
        exponents=exponents,
        remainder=remainder,
        domain=domain,
        center=center,
    )


@register_pytree_node_class
class TMFlowpipe:
    """Result of validated ODE integration.

    Each TaylorModel in the flowpipe has domain variables (t, x),
    where t is in [t_i, t_{i+1}] and x is expanded around the nominal
    polynomial approximation of the solution.

    Attributes
    ----------
    times : Array
        Time grid of shape ``(nsteps,)``.
    nsteps : int
        Number of real integration steps taken.
    success : bool
        ``True`` if every step was validated.
    """

    def __init__(
        self,
        times,
        tube_data,
        exponents,
        nsteps,
        success,
        *,
        _input_pytree,
        _output_pytree,
        _per_leaf_order,
        # Legacy kwargs
        _domain_treedef=None,
        _leaf_shapes=None,
    ):
        self.times = times
        self._tube_data = tube_data
        self.exponents = exponents
        self.nsteps = nsteps
        self.success = success
        if _input_pytree is not None:
            self._input_pytree = _input_pytree
        elif _domain_treedef is not None and _leaf_shapes is not None:
            self._input_pytree = PyTreeShape(_domain_treedef, _leaf_shapes)
        else:
            raise ValueError(
                "Either _input_pytree or both _domain_treedef and _leaf_shapes must be provided"
            )
        self._output_pytree = (
            _output_pytree if _output_pytree is not None else PyTreeShape.flat(())
        )
        self._per_leaf_order = _per_leaf_order

    def __len__(self):
        return self.nsteps

    def __getitem__(self, idx):
        """Reconstruct the tube TaylorModel at step ``idx``."""
        coeffs, rem_lo, rem_hi, dom_lo, dom_hi, center = jax.tree.map(
            lambda x: x[idx], self._tube_data
        )
        return TaylorModel(
            coeffs,
            self.exponents,
            interval(rem_lo, rem_hi),
            interval(dom_lo, dom_hi),
            center,
            _input_pytree=self._input_pytree,
            _output_pytree=self._output_pytree,
            _per_leaf_order=self._per_leaf_order,
        )

    def __call__(self, t):
        """Evaluate the flowpipe at time ``t``, returning a spatial TaylorModel.

        Uses ``searchsorted`` to find the tube segment containing ``t``,
        reconstructs the tube TaylorModel, and evaluates it at ``t`` via
        ``tx_tm_eval``.

        Parameters
        ----------
        t : float
            Time at which to evaluate.

        Returns
        -------
        TaylorModel
            Spatial TaylorModel at time ``t``.
        """
        idx = jnp.searchsorted(self.times, t, side="left")
        idx = jnp.clip(idx, 0, self.nsteps - 1)
        tube_tm = self[idx]
        t_order = self._per_leaf_order[0]
        return tx_tm_eval(tube_tm, t, t_order)

    def interval_hulls(self) -> Interval:
        def _data_to_interval(data):
            coeffs, rem_lo, rem_hi, dom_lo, dom_hi, center = data
            tm = TaylorModel(
                coeffs,
                self.exponents,
                interval(rem_lo, rem_hi),
                interval(dom_lo, dom_hi),
                center,
                _input_pytree=self._input_pytree,
                _output_pytree=self._output_pytree,
                _per_leaf_order=self._per_leaf_order,
            )
            return tm.interval_hull()

        return jax.vmap(_data_to_interval)(self._tube_data)

    def tree_flatten(self):
        children = (
            self.times,
            self._tube_data,
            self.exponents,
            self.nsteps,
            self.success,
        )
        aux = {
            "_input_pytree": self._input_pytree,
            "_output_pytree": self._output_pytree,
            "_per_leaf_order": self._per_leaf_order,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        times, tube_data, exponents, nsteps, success = children
        return cls(
            times,
            tube_data,
            exponents,
            nsteps,
            success,
            _input_pytree=aux["_input_pytree"],
            _output_pytree=aux["_output_pytree"],
            _per_leaf_order=aux["_per_leaf_order"],
        )


class TMFlowpipeGenerator(ABC):
    """Abstract base class for TaylorModel flowpipe generators.

    These collections of algorithms generically use the following steps to generate a validated flowpipe:

    1. Take in an initial TaylorModel (usually identity), a largest dt, and an interval [t_0, t_f]
    2. The taylor expansion of the solution is computed in (t,x) around (t_i,x_i) using either:
        a. Lie derivatives of the system vector field
        b. Iterated Picard iteration
    3. Verify the contraction of a Picard operator. If not contractive either:
        a. Increase the remainder portion of the TaylorModel
        b. Reduce the step size dt
    """

    sys: System

    def __init__(self, sys: System):
        self.sys = sys

    @abstractmethod
    def _initialize(self, t0, tf, tm0, dt_max, *, t_order, max_steps, **kwargs):
        """Initialize the algorithm before calling generate_flowpipe."""

    @abstractmethod
    def _step(
        self, t, tmi, dt_max, **kwargs
    ) -> Tuple[float, TaylorModel, TaylorModel, bool]:
        """Take one validated step.

        Parameters
        ----------
        tmi : TaylorModel
            Time-independent TaylorModel at the start of the step.
        dt_max : float
            Maximum allowed step size.

        Returns
        -------
        dt : float
            Actual step size taken.
        tmf : TaylorModel
            Time-independent TaylorModel at the end of the step (at t = tmi + dt)
        tm_tube : TaylorModel
            Time-dependent validated TaylorModel over [tmi, tmi + dt]
        success : bool
            ``True`` if the step was validated.
        """

    def generate_flowpipe(
        self, t0, tf, tm0, dt_max, *, t_order=4, max_steps=4096, inputs=None, **kwargs
    ):
        """Generate a validated flowpipe for the given system.

        Parameters
        ----------
        t0 : float
            Initial time.
        tf : float
            Final time.
        tm0 : TaylorModel
            Initial TaylorModel.
        dt_max : float
            Maximum allowed step size.
        max_steps : int, optional
            Maximum number of steps, by default 4096.
        inputs : Array, optional
            Piecewise-constant inputs of shape ``(max_steps, ...)``.
            ``inputs[i]`` is held constant during step ``i``, spaced at
            exactly ``dt_max``.
        **kwargs
            Keyword arguments to pass to the step function.

        Returns
        -------
        TMFlowpipe
            Validated flowpipe.
        """
        t0 = jnp.asarray(t0)
        tf = jnp.asarray(tf)
        dt_max = jnp.asarray(dt_max)

        self._initialize(
            t0, tf, tm0, dt_max, t_order=t_order, max_steps=max_steps, **kwargs
        )

        # Compute tube structure metadata from tm0 without running a step
        mon_len = tm0.polynomial.exponents.shape[1]
        n_out = tm0.coeffs.shape[0]

        tube_exponents = jnp.vstack((
            jnp.repeat(jnp.arange(t_order + 1), mon_len),
            jnp.tile(tm0.polynomial.exponents, (1, t_order + 1)),
        ))

        tube_domain = (interval(t0, t0 + dt_max), tm0.domain)
        tube_center = (t0, tm0.center)
        _domain_treedef, _leaf_shapes, flat_tube_domain = _pytree_to_flattened_array(tube_domain)
        _, _, flat_tube_center = _pytree_to_flattened_array(tube_center)

        tube_input_pytree = PyTreeShape(_domain_treedef, _leaf_shapes)
        tube_output_pytree = PyTreeShape.flat((n_out,))
        tube_per_leaf_order = (t_order,) + tm0._per_leaf_order

        # Dummy tube data for noop branch of lax.cond
        dummy_data = (
            jnp.zeros((n_out, mon_len * (t_order + 1))),
            jnp.zeros_like(tm0.remainder.lower),
            jnp.zeros_like(tm0.remainder.upper),
            jnp.zeros_like(flat_tube_domain.lower),
            jnp.zeros_like(flat_tube_domain.upper),
            jnp.zeros_like(flat_tube_center),
        )

        def scan_fn(carry, step_idx):
            ti, tmi, success, steps_taken = carry
            should_step = jnp.logical_and(success, ti < tf)

            u_i = inputs[step_idx] if inputs is not None else None

            def take_step():
                step_kw = dict(kwargs)
                if u_i is not None:
                    step_kw['u'] = u_i
                dt, tmf, tm_tube, step_success = self._step(ti, tmi, dt_max, **step_kw)
                return (ti + dt, tmf, step_success, steps_taken + 1), (
                    ti + dt,
                    _tube_to_data(tm_tube),
                )

            def noop():
                return (ti, tmi, success, steps_taken), (jnp.nan, dummy_data)

            return jax.lax.cond(should_step, take_step, noop)

        init_carry = (t0, tm0, jnp.bool_(True), jnp.int32(0))
        (
            (final_t, final_tm, final_success, final_steps),
            (scan_times, scan_tube_data),
        ) = jax.lax.scan(scan_fn, init_carry, jnp.arange(max_steps))

        return TMFlowpipe(
            scan_times,
            scan_tube_data,
            tube_exponents,
            final_steps,
            final_success,
            _input_pytree=tube_input_pytree,
            _output_pytree=tube_output_pytree,
            _per_leaf_order=tube_per_leaf_order,
        )
