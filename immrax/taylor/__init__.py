"""Taylor model module for immrax.

This module provides Taylor model representations and the natural Taylor model
function transformation (nattm), similar to natif for intervals.
"""

from .taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_from_interval,
    taylor_model_from_function,
    taylor_model_concatenate,
    _generate_exponents,
    _get_canonical_exponents,
)

from .nattm import (
    nattm,
    nattm_jaxpr,
    tm_inclusion_registry,
    istaylormodel,
)

__all__ = [
    # Taylor Model class and constructors
    "TaylorModel",
    "taylor_model",
    "taylor_model_from_interval",
    "taylor_model_from_function",
    "taylor_model_concatenate",
    # Natural Taylor Model function
    "nattm",
    "nattm_jaxpr",
    "tm_inclusion_registry",
    "istaylormodel",
]
