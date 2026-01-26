"""Reachability algorithms for set-based analysis.

This module provides various algorithms for computing reachable sets
of nonlinear dynamical systems using different set representations.

Available Algorithms
--------------------
LohnerReachability
    Taylor expansion with QR-based reduction
AlthoffGirardReachability
    Conservative linearization with matrix exponential
TaylorGirardReachability
    Taylor expansion with Girard-style reduction

Base Classes
------------
ReachableSets
    Abstract base for reachable set containers
GenericReachSets
    Generic pytree-based container for any set type
ReachableSetGenerator
    Abstract base for reachability algorithms
BaseSetGenerator
    Base implementation with Picard iteration
"""

from .base import (
    # Helper functions
    prolongation,
    # Data structures
    ReachableSets,
    GenericReachSets,
    # Abstract classes
    ReachableSetGenerator,
    BaseSetGenerator,
)

# Re-export fact/inv_fact from utils for backwards compatibility
from ...utils import fact, inv_fact

from .lohner import LohnerReachability
from .althoff_girard import AlthoffGirardReachability
from .taylor_girard import TaylorGirardReachability
from .tm_flowpipe import TMFlowpipeGenerator, tm_flowpipe_step, tm_reachtube

__all__ = [
    # Helper functions
    "fact",
    "inv_fact",
    "prolongation",
    # Data structures
    "ReachableSets",
    "GenericReachSets",
    # Abstract classes
    "ReachableSetGenerator",
    "BaseSetGenerator",
    # Algorithms
    "LohnerReachability",
    "AlthoffGirardReachability",
    "TaylorGirardReachability",
    "TMFlowpipeGenerator",
    "tm_flowpipe_step",
    "tm_reachtube",
]
