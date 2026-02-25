import jax
import jax.numpy as jnp
import immrax as irx
from immutabledict import immutabledict
import matplotlib.pyplot as plt


class Sys(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x):
        return jnp.array([[-2.0, 5.0], [-5.0, -2.0]]) @ x


sys = Sys()
nt0 = irx.L2Normotope(jnp.array([0.0, 0.0]), jnp.diag(jnp.array([1.0, 2.0])), 1.0)
embsys = irx.NormotopeEmbedding(sys)

traj = embsys.compute_reachset(
    # 0.0, 1.0, nt0, (), 0.01, solver="rk45", f_kwargs=immutabledict({})
    0.0,
    10.0,
    nt0,
    (),
    0.01,
)
tfinite = jnp.where(jnp.isfinite(traj.ts))

fig, ax = plt.subplots(1, 1)

for i, t in enumerate(traj.ts[tfinite]):
    nt = irx.L2Normotope(traj.ys[0].ox[i], traj.ys[0].alpha[i], traj.ys[0].y[i])
    nt.plot_projection(ax)

ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
plt.show()

# print(embtraj.ys[0].y[tfinite])
