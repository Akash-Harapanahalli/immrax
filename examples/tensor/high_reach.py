import jax
import jax.numpy as jnp
import immrax as irx

A = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
print(A)


def f(x):
    # A = jnp.array([[1.0, -1.0], [0.0, 1.0]])
    # return A @ x
    # return jnp.array([-2 * x[0], -x[1]])
    return A @ x


# alpha = irx.identijet(jnp.zeros(2), 2)
alpha = irx.pjet(lambda x: x.T @ x)(irx.identijet(jnp.zeros(2), 3))
print(alpha)
print(alpha.coeffs)
print(alpha.multiindices)

ox = jnp.array([1.0, 0.0])
jet_x = irx.identijet(ox, 5)


def inner(x):
    return jax.jvp(alpha.evaluate, (x,), (f(x),))[1]


outer = irx.pjet(inner)

print("====")
print(alpha.evaluate(ox))
print(inner(ox))
res = outer(jet_x)
print(res)
print(res.coeffs)
print(res.multiindices)

# print(jet_x.evaluate_monomials(ox))
# print(z_dot_tensor(ox))
# print(term(ox))

# print(out)
# print(out.coeffs)
# print(out.multiindices)
