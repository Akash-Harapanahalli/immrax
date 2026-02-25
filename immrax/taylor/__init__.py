"""Taylor model module for immrax.

This module provides Taylor model and Taylor polynomial representations,
and the natural function transformations (pjetm, pjet) for propagating
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
)

identijetm = taylor_model_identity

from .base import (
    PyTreeShape,
    leaf_total_degree_exponents,
    check_leaf_bounds,
    leaf_slice,
    pack_pytree,
    unpack_pytree,
    normalize_leaf_order,
    max_leaf_order,
    compute_leaf_order,
)

from .taylor_polynomial import (
    TaylorPolynomial,
    taylor_polynomial_concatenate,
    taylor_polynomial_constant,
    taylor_polynomial_identity,
)

identijet = taylor_polynomial_identity

from .pjetm import (
    pjetm,
    pjetm_jaxpr,
    tm_inclusion_registry,
    istaylormodel,
)

from .pjet import (
    pjet,
    pjet_jaxpr,
    tp_inclusion_registry,
    istaylorpolynomial,
)

from .algorithms import (
    TMFlowpipe,
    TMFlowpipeGenerator,
    BasicTMFlowpipeGenerator,
    BungerTMFlowpipeGenerator,
)

__all__ = [
    # Taylor Model class and constructors
    "TaylorModel",
    "PyTreeShape",
    "taylor_model",
    "taylor_model_identity",
    "identijetm",
    "taylor_model_constant",
    "taylor_model_from_function",
    "taylor_model_concatenate",
    "integrate_variable",
    # Taylor Polynomial class and constructors
    "TaylorPolynomial",
    "taylor_polynomial_concatenate",
    "taylor_polynomial_constant",
    "taylor_polynomial_identity",
    "identijet",
    # Natural Taylor Model function
    "pjetm",
    "pjetm_jaxpr",
    "tm_inclusion_registry",
    "istaylormodel",
    # Natural Taylor Polynomial function
    "pjet",
    "pjet_jaxpr",
    "tp_inclusion_registry",
    "istaylorpolynomial",
]
