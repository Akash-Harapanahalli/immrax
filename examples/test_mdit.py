import jax
import jax.numpy as jnp
import immrax as irx


def f(x):
    return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1])])
    # return jnp.sin(x[0]) * jnp.cos(x[1])


"""MDIT remainder bound must enclose sin(x₀)cos(x₁) in 2D."""
xc = jnp.array([0.0, 0.0])
ix = irx.icentpert(xc, 0.3)
f_mdit = jax.jit(irx.mdit(f, 2))
result, runtimes = irx.utils.run_times(100, f_mdit, ix, xc)

print(f"{jnp.median(runtimes):.6f}")

for r in result:
    print(r)

test_xs = irx.utils.gen_ics(ix, 200, key=jax.random.PRNGKey(42))


def check_bound(x):
    fc = irx.taylor_approx(result[:-1], xc, x)
    fi = result[-1].contract(x - xc, True)
    bound = fc + fi
    val = f(x)
    # return jnp.logical_and(val >= bound.lower, val <= bound.upper)
    return bound


test_bds = jax.vmap(check_bound)(test_xs)
print(test_bds - jax.vmap(f)(test_xs))
# print(jnp.all(test_bds))
