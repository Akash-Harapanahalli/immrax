import jax
import jax.numpy as jnp
import pytest
from immrax.taylor.taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_constant,
)
from immrax.inclusion import interval, icentpert


def test_taylor_model_init_validation():
    """Test that TaylorModel.__init__ enforces non-None metadata."""
    coeffs = jnp.array([[1.0]])
    exponents = jnp.array([[0]])
    remainder = interval(jnp.array([0.0]))
    domain = icentpert(jnp.array([0.0]), jnp.array([1.0]))
    center = jnp.array([0.0])

    # TaylorModel constructor requires explicit metadata
    with pytest.raises(ValueError, match="_domain_treedef cannot be None"):
        TaylorModel(
            coeffs,
            exponents,
            remainder,
            domain,
            center,
            _domain_treedef=None,
            _leaf_shapes=None,
            _per_leaf_order=None,
        )


def test_taylor_model_factory_defaults():
    """Test that factory functions compute defaults."""
    domain = icentpert(jnp.array([0.0]), jnp.array([1.0]))
    coeffs = jnp.array([[1.0]])
    exponents = jnp.array([[0]])

    # taylor_model factory (infers metadata)
    tm = taylor_model(coeffs, exponents, domain=domain)

    assert tm._domain_treedef is not None
    assert tm._leaf_shapes is not None
    assert tm._per_leaf_order is not None
    assert tm._leaf_shapes == ((1,),)
    assert tm._per_leaf_order == (0,)  # exp is 0

    # taylor_model_constant factory (infers metadata)
    tm_const = taylor_model_constant(interval(jnp.array([2.0])), domain, order=2)

    assert tm_const._domain_treedef is not None
    assert tm_const._leaf_shapes is not None
    assert tm_const._per_leaf_order == (2,)


def test_taylor_model_no_domain():
    """Test taylor_model with no domain (should use default)."""
    coeffs = jnp.array([[1.0, 0.0]])  # 2 params
    exponents = jnp.array([[0, 1]])  # 1D

    tm = taylor_model(coeffs, exponents)

    assert tm.domain is not None
    assert tm._domain_treedef is not None
