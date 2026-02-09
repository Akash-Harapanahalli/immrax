# %%
import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator
import matplotlib.pyplot as plt

# %%
class TestSystem(irx.System):
    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 2
    def f (self, t, x) :
        return jnp.array([x[1], -x[0]])

sys = TestSystem()
prolonged_f = irx.utils.prolongation(sys.f, 3)


# %%
t0 = jnp.asarray(0.)
tf = jnp.asarray(.1)
ix0 = irx.icentpert(jnp.ones(2), .5)
tm0 = irx.taylor_model_identity((irx.interval(t0), ix0), order=(2, 2))
tmt = irx.taylor_model_identity(irx.interval(t0), order=2)
tmx = irx.taylor_model_identity(ix0, order=2)

print(tm0._domain_treedef.children()[1])
print(jax.tree_util.tree_structure(tuple(tm0._domain_treedef.children()[1:])))
print(tm0._domain_treedef, tm0._leaf_shapes[1:])
print(tmx._domain_treedef, tmx._leaf_shapes)

print(tm0.exponents)
print(tm0.coeffs)

# irx.taylor._get_leaf_total_degree_exponents(((),(2,)), (4,2))

# %%
pr_coeffs = irx.nattp(lambda x : prolonged_f(t0, x))(tmx.polynomial)
print(pr_coeffs)
print(pr_coeffs[0].exponents)
print(pr_coeffs[0].coeffs)
# This simple because of canonical exponent structure.
coeffs = jnp.array([tp.coeffs for tp in pr_coeffs])

conv = irx.taylor.algorithms.basic.tps_to_tx(pr_coeffs, irx.interval(jnp.zeros(2)),
 (irx.interval(t0, tf), tmx.domain), (t0, tmx.center))
print(conv)
print(conv.exponents)
print(conv.coeffs)
print(conv.domain)
print(conv.remainder)
print(conv.center)

# %%


# fp = irx.utils.prolongation(sys.f, 2)
tm1 = irx.nattm(sys.f, structured_center=True)(tm0)
tp1 = irx.nattp(sys.f, structured_center=True)(tm0.polynomial)
tm1_integ = irx.taylor.integrate_variable(tm1, var_idx=0)

tm2 = irx.nattm(sys.f)(tmt, tmx)


# %%
# print((jnp.arange(10).reshape(-1,1) @ jnp.ones((1,4))).reshape(-1))
print(jnp.tile(jnp.arange(10).reshape(2,5), (10,)))

# 
fpg = BasicTMFlowpipeGenerator(sys)
fpg._initialize(t0, tf, tm0, tf - t0, t_order=4)

res = fpg._picard(tm1)
print(res.remainder)

tm_test_in = irx.taylor_model_identity(irx.interval([-1.], [1.]), order=4)
tm_test = irx.nattm(lambda x : jnp.sin(x))(tm_test_in)
tm_cos_test = irx.nattm(lambda x : jnp.cos(x))(tm_test_in)

pr_coeffs = prolonged_f(0., ix0.center)

step_conv, picard_step_conv = fpg._step(t0, tmx, tf)
print(jnp.allclose(step_conv.coeffs, picard_step_conv.coeffs))
print(step_conv.coeffs - picard_step_conv.coeffs)

def pr_poly(t) :
    return irx.taylor.algorithms.basic.tx_tm_eval(step_conv, t).evaluate_polynomial(ix0.center)

fig, ax = plt.subplots()

tt = jnp.linspace(t0, tf, 100)
pr_xx = jax.vmap(pr_poly)(tt)
ax.plot(pr_xx[:,0], pr_xx[:,1])
