import jax
import jax.numpy as jnp
import immrax as irx

tm_test_in = irx.taylor_model_identity(irx.icentpert(jnp.zeros(3), 1.), order=4)
tm_cos = irx.nattm(lambda x : jnp.cos(x))(tm_test_in)
tm_sin = irx.nattm(lambda x : jnp.sin(x))(tm_test_in)

print(tm_cos.coeffs)
tm_cos_int = irx.taylor.tm_integrate_variable(tm_cos, 0)

print(tm_cos_int.exponents)
print(tm_sin.exponents)

print(tm_cos_int.coeffs)
print(tm_sin.coeffs)