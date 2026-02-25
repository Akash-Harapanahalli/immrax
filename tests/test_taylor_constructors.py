import jax
import jax.numpy as jnp
import pytest
from immrax.taylor.taylor_model import (
    TaylorModel,
    taylor_model,
    taylor_model_constant,
)
from immrax.inclusion import interval, icentpert


def test_taylor_model_factory_validation():
    """Test that taylor_model() validates shapes."""
    domain = icentpert(jnp.array([0.0]), jnp.array([1.0]))

    # Scalar coeffs (not at least 1D)
    with pytest.raises(ValueError, match="coeffs must be at least 1D"):
        taylor_model(jnp.array(1.0), jnp.array([[0]]), domain=domain)

    # Mismatched coeffs/exponents monomial count
    with pytest.raises(ValueError, match="same number of monomials"):
        taylor_model(jnp.array([[1.0, 2.0]]), jnp.array([[0]]), domain=domain)

    # Mismatched remainder shape
    with pytest.raises(ValueError, match="same output shape"):
        taylor_model(
            jnp.array([[1.0]]),
            jnp.array([[0]]),
            remainder=interval(jnp.array([0.0, 0.0])),
            domain=domain,
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
    assert tm.leaf_order is not None
    assert tm._leaf_shapes == ((1,),)
    assert tm.leaf_order == (0,)  # exp is 0

    # taylor_model_constant factory (infers metadata)
    tm_const = taylor_model_constant(interval(jnp.array([2.0])), domain, order=2)

    assert tm_const._domain_treedef is not None
    assert tm_const._leaf_shapes is not None
    assert tm_const.leaf_order == (2,)


def test_taylor_model_no_domain():
    """Test taylor_model with no domain (should use default)."""
    coeffs = jnp.array([[1.0, 0.0]])  # 2 params
    exponents = jnp.array([[0, 1]])  # 1D

    tm = taylor_model(coeffs, exponents)

    assert tm.domain is not None
    assert tm._domain_treedef is not None
