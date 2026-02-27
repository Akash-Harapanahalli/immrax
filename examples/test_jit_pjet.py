import jax
import jax.numpy as jnp
import immrax as irx

x = irx.identijet([1.0, 0.0], 3)
y = irx.identijet([0.0, 1.0], 3)

print(x * y)
print(jax.jit(lambda x, y: x * y)(x, y))
