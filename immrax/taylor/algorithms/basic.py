import jax
import jax.numpy as jnp

from immrax.inclusion import Interval, interval
from immrax.system import System
from immrax.utils import inv_fact
from .. import TaylorModel
from .base import TMFlowpipeGenerator

from typing import Tuple


class BasicTMFlowpipeGenerator(TMFlowpipeGenerator):
    """A basic implementation of a Picard-based flowpipe generator."""

    def _step(self, tmi, dt_max, **kwargs):
        pass
