import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Zonotope :
    r"""Defines the set
    .. math::
        Z = { \mathring{x} + Gv : v \in [-1,1]^m }
    where :math:`\mathring{x}` is the center and :math:`G` is the generator matrix
    """
    def __init__ (self, ox, G) :
        self.ox = ox
        self.G = G
        if G.shape[:-1] != ox.shape:
            raise ValueError("Incompatible shapes")

    def __add__ (self, other) :
        if not isinstance(other, Zonotope):
            raise TypeError("Can only add Zonotope to Zonotope")
        if self.ox.shape != other.ox.shape :
            raise ValueError("Incompatible dimensions of Zonotope")
        return Zonotope(self.ox + other.ox, jnp.hstack((self.G, other.G)))

