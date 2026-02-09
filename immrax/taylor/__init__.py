"""Taylor model module for immrax.

This module provides Taylor model and Taylor polynomial representations,
and the natural function transformations (nattm, nattp) for propagating
them through functions, similar to natif for intervals.
"""

from .taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_identity,
    taylor_model_constant,
    taylor_model_from_function,
    taylor_model_concatenate,
    tm_integrate_variable,
    integrate_variable,
    _generate_exponents,
    _get_canonical_exponents,
    _get_leaf_total_degree_exponents,
    _check_per_leaf_bounds,
    _leaf_slice,
    _pytree_to_flattened_array,
    _unflatten_array_to_pytree,
    _normalize_order_pytree,
)

from .taylor_polynomial import TaylorPolynomial

from .nattm import (
    nattm,
    nattm_jaxpr,
    tm_inclusion_registry,
    istaylormodel,
)

from .nattp import (
    nattp,
    nattp_jaxpr,
    tp_inclusion_registry,
    istaylorpolynomial,
)

from .algorithms import (
    TMFlowpipe,
    TMFlowpipeGenerator,
    BasicTMFlowpipeGenerator,
)

__all__ = [
    # Taylor Model class and constructors
    "TaylorModel",
    "taylor_model",
    "taylor_model_identity",
    "taylor_model_constant",
    "taylor_model_from_function",
    "taylor_model_concatenate",
    "integrate_variable",
    # Taylor Polynomial class
    "TaylorPolynomial",
    # Natural Taylor Model function
    "nattm",
    "nattm_jaxpr",
    "tm_inclusion_registry",
    "istaylormodel",
    # Natural Taylor Polynomial function
    "nattp",
    "nattp_jaxpr",
    "tp_inclusion_registry",
    "istaylorpolynomial",
]
