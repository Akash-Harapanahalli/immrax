import jax
import jax.numpy as jnp
import immrax as irx

p = 4


def f(x):
    # return jnp.array([jnp.sin(x[0]) * jnp.cos(x[1]), x[0] + x[1]])
    return jnp.ones((2, 2)) * x[0]


f_mdit = jax.jit(irx.mdit(f, p))

xc = jnp.ones(2)
ix = irx.icentpert(jnp.ones(2), 1.0)

res, runtimes = irx.utils.run_times(100, f_mdit, ix, xc)

print(jnp.median(runtimes))
nom_pp1 = res[0].get_order(p + 1)


def eval_coeffs(coeffs, x):
    tp = irx.TaylorPolynomial(coeffs, nom_pp1.multiindices, xc)
    return tp.evaluate(x)


def eval_mdit(x):
    return res[0].evaluate_to_order(p, x) + irx.natif(eval_coeffs)(res[1], x) - f(x)


mc_xs = irx.utils.gen_ics(ix, 100)

print(jax.vmap(eval_mdit)(mc_xs))


# print(eval_coeffs(nom3.coeffs, jnp.zeros(2), xc))

# def bound_f (x) :


# def f_jet(x):
#     # return irx.pjet(f)(irx.identijet(x, 2)).coeffs
#     return irx.pjet(f)(irx.identijet(x, 2)).get_order(2)
#
#
# print(irx.interval(0.0))
#
# test_jet = f_jet(jnp.ones(2))
# print(test_jet.multiindices)
# ret = irx.natif(f_jet)(irx.icentpert(jnp.ones(2), 0.1))
# print(type(ret))
# print(ret)
