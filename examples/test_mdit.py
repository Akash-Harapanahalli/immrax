import jax
import jax.numpy as jnp
import immrax as irx


def f(x):
    return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1])])
    # return jnp.sin(x[0]) * jnp.cos(x[1])


"""MDIT remainder bound must enclose sin(x₀)cos(x₁) in 2D."""
xc = jnp.array([0.0, 0.0])
ix = irx.icentpert(xc, 1.0)
perm = irx.Permutation((1, 0))
p = 5
f_mdit = jax.jit(
    lambda *args, **kwargs: irx.mdit(f, p)(*args, **kwargs, permutation=perm)
)
f_dit = jax.jit(irx.dit(f, p))

print(irx.mjacM(f)(ix, center=(xc,)))


def check_bound(x, result):
    fc = irx.taylor_approx(result[:-1], xc, x)
    fi = result[-1].contract(x - xc, True)
    bound = fc + fi
    val = f(x)
    # return jnp.logical_and(val >= bound.lower, val <= bound.upper)
    return bound - val


test_xs = irx.utils.gen_ics(ix, 200, key=jax.random.PRNGKey(42))

print("\n====MDIT====")
result_mdit, runtimes_mdit = irx.utils.run_times(100, f_mdit, ix, xc)
print(f"{jnp.median(runtimes_mdit):.6f}")
for r in result_mdit:
    print(r)
mdit_bds = jax.vmap(check_bound, in_axes=(0, None))(test_xs, result_mdit)
print(sum(mdit_bds))
print(irx.natif(check_bound)(ix, result_mdit))

print("\n====DIT====")
result_dit, runtimes_dit = irx.utils.run_times(100, f_dit, ix, xc)
print(f"{jnp.median(runtimes_dit):.6f}")
for r in result_dit:
    print(r)
dit_bds = jax.vmap(check_bound, in_axes=(0, None))(test_xs, result_dit)
print(sum(dit_bds))
print(irx.natif(check_bound)(ix, result_dit))
