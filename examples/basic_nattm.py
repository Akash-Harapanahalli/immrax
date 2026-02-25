# %%
import jax
import jax.numpy as jnp
import immrax as irx
import matplotlib.pyplot as plt

# %%

ix0 = irx.icentpert(jnp.zeros(4), 1.)
tm0 = irx.taylor_model_identity(ix0)

fig, ax = plt.subplots()

irx.utils.draw_iarray(ax, tm0.interval_hull())
print(tm0(0.))

# %%
def f (x) :
    return jnp.sin(x) - x

jit_or1 = jax.jit(irx.pjetm(f, max_order=1))
jit_or2 = jax.jit(irx.pjetm(f, max_order=4))

res_or1, times_or1 = irx.utils.run_times(100, jit_or1, tm0)
res_or2, times_or2 = irx.utils.run_times(100, jit_or2, tm0)

print(jnp.mean(times_or1), res_or1, res_or1(0.))
print(jnp.mean(times_or2), res_or2, res_or2(0.))

fig, ax = plt.subplots()
irx.utils.draw_iarray(ax, res_or1(0.))
irx.utils.draw_iarray(ax, res_or2(0.))


