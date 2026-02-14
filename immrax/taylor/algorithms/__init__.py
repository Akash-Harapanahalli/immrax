"""Validated integration algorithms for ODEs using Taylor models.

Two algorithms are provided, both from TaylorModels.jl:

- ``validated_integ``: Picard-Lindelöf iteration for remainder validation.
- ``validated_integ2``: Epsilon-inflation algorithm (Bünger 2019).

Both compute rigorous flowpipe enclosures for ``x' = f(t, x)`` with
interval initial conditions.
"""

from immrax.taylor.algorithms.base import (
    TMFlowpipe,
    TMFlowpipeGenerator,
    tx_tm_eval,
    tps_to_tx,
)
from .basic import BasicTMFlowpipeGenerator
from .bunger import BungerTMFlowpipeGenerator


__all__ = [
    "TMFlowpipe",
    "TMFlowpipeGenerator",
    "BasicTMFlowpipeGenerator",
    "BungerTMFlowpipeGenerator",
    "tx_tm_eval",
    "tps_to_tx",
]
