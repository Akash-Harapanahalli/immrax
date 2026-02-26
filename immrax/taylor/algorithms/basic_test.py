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
        # Harmonic Oscillator
        # return jnp.array([x[1], -x[0]])

        # # VanderPol Oscillator
        mu = 0.5
        return jnp.array([
            x[1],
            mu*(1 - x[0]**2)*x[1] - x[0]
        ])

sys = TestSystem()
prolonged_f = irx.utils.prolongation(sys.f, 3)


# %%
t0 = jnp.asarray(0.)
tf = jnp.asarray(1.)
ix0 = irx.icentpert([-2, 0.], .1)
tm0 = irx.taylor_model_identity((irx.interval(t0), ix0), order=(2, 2))
tmt = irx.taylor_model_identity(irx.interval(t0), order=2)
tmx = irx.taylor_model_identity(ix0, order=2)

print(tm0._domain_treedef.children()[1])
print(jax.tree_util.tree_structure(tuple(tm0._domain_treedef.children()[1:])))
print(tm0._domain_treedef, tm0._leaf_shapes[1:])
print(tmx._domain_treedef, tmx._leaf_shapes)

print(tm0.multiindices)
print(tm0.coeffs)

# irx.taylor.leaf_total_degree_exponents(((),(2,)), (4,2))

# %%
pr_coeffs = irx.pjet(lambda x : prolonged_f(t0, x))(tmx.polynomial)
print(pr_coeffs)
print(pr_coeffs[0].multiindices)
print(pr_coeffs[0].coeffs)
# This simple because of canonical exponent structure.
coeffs = jnp.array([tp.coeffs for tp in pr_coeffs])

conv = irx.taylor.algorithms.basic.tps_to_tx(pr_coeffs, irx.interval(jnp.zeros(2)),
 (irx.interval(t0, tf), tmx.domain), (t0, tmx.center))
print(conv)
print(conv.multiindices)
print(conv.coeffs)
print(conv.domain)
print(conv.remainder)
print(conv.center)

# %%
fpg = BasicTMFlowpipeGenerator(sys)
fpg._initialize(t0, tf, tmx, tf - t0, t_order=3, delta=1.e-4, eps=1.e-2)

# %%
%matplotlib widget

pr_coeffs = prolonged_f(0., ix0.center)
# pr_poly = lambda t : pr_coeffs[0] + pr_coeffs[1]*t + pr_coeffs[2]*t**2/2 + pr_coeffs[3]*t**3/6
# pr_poly = lambda t : conv.evaluate_polynomial(t, ix0.center)

dt = 0.01

_, _, step_conv, contractive = fpg._step(t0, tmx, dt)
print(step_conv.remainder)
print(fpg._picard(step_conv).remainder)
# print(step_conv.coeffs)
# print(picard_step_conv.coeffs)
# print(picard_step_conv.multiindices)
# print(jnp.allclose(step_conv.coeffs, picard_step_conv.coeffs))
# print(step_conv.remainder, picard_step_conv.remainder)
# print(irx.utils.check_containment(picard_step_conv.remainder,step_conv.remainder))

tube = fpg.generate_flowpipe(t0, tf, tmx, dt)

print(contractive)

def pr_poly(t) :
    return irx.taylor.algorithms.basic.tx_tm_eval(step_conv, t, t_order=3).evaluate_polynomial(ix0.center)

fig, ax = plt.subplots()

tt = jnp.linspace(t0, tf, 100)
# xx = jax.vmap(lambda t : tm1_integ.polynomial.evaluate_structured(t, ix0.center) + ix0.center)(tt)
# ax.plot(xx[:,0], xx[:,1])
pr_xx = jax.vmap(pr_poly)(tt)
ax.plot(pr_xx[:,0], pr_xx[:,1])

# fpg.generate_flowpipe(t0, tf, tm0, dt_max=1e-2)
# fpg._initialize(t0, tf, tm0, dt_max=1e-2)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)


