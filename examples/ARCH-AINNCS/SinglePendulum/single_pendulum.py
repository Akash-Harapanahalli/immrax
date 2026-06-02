# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Single Pendulum Benchmark
#
# We consider a classical inverted pendulum. A ball of mass $m$ is attached to a massless beam of length $L$. The beam is actuated with a torque $T$, and we assume viscous friction with a friction coefficient of $c$.
#
# This benchmark is taken from the ARCH-COMP25 Category Report: Artificial Intelligence and Neural Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants.

# %%
import argparse
import jax
import jax.numpy as jnp
import immrax as irx
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true', help='save reach-set plot (single_pendulum.png/.pdf)')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## The Single Pendulum Dynamics and Controller
#
# The governing equation of motion can be obtained as:
#
# $$
# \ddot{\theta} = \frac{g}{L} \sin \theta + \frac{1}{mL^2} (T - c \dot{\theta})
# $$
#
# where $\theta$ is the angle of the link concerning the upward vertical axis and $\dot{\theta}$ is the angular velocity. After defining the state variables $x_1 = \theta$ and $x_2 = \dot{\theta}$, the dynamics in state-space form are
#
# $$
# \begin{aligned}
# \dot{x}_1 &= x_2 \\
# \dot{x}_2 &= \frac{g}{L} \sin x_1 + \frac{1}{mL^2} (T - c x_2)
# \end{aligned}
# $$
#
# The model parameters are chosen as $m = 0.5, L = 0.5, c = 0, g = 1$.
#
# Controllers are trained using behavior cloning, a supervised learning approach for training controllers. The time step for the controller and the discrete-time model is $\Delta t = 0.05$.

# %%
class SinglePendulum(irx.System):
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 2  

    def f(self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        # Constants
        l = 0.5
        m = 0.5
        g = 1.0
        c = 0.0

        x1, x2 = x  # theta, theta_dot
        a = u[0]  # applied torque

        dx1 = x2
        dx2 = (g / l) * jnp.sin(x1) + (a - c * x2) / (m * l**2)

        return jnp.array([dx1, dx2])

olsys = SinglePendulum()
net = irx.NeuralNetwork('controller_single_pendulum')
clsys = irx.NNCSystem(olsys, net)

# %% [markdown]
# ## Specification
#
# The initial set is
# $$
# x \in [1.0, 1.175] \times [0.0, 0.2]
# $$
#
# The specification is $\forall t \in [0.5, 1] : \theta \in [0, 1]$ (analogously for $k \in [10, 20]$ in discrete time).

# %%
perm = irx.standard_permutation(1+2+1+1)[0]
ix0 = irx.interval([1.0, 0.0], [1.175, 0.2])
pt0 = irx.Polytope.from_interval(ix0)

t0 = 0.
tf = 1.
dt = 0.01
solver = 'tsit5'
max_steps = round((tf - t0) / dt)

# %% [markdown]
# ## Reachability Analysis
#
# We compute the reachable set using `FastlinAdjointEmbedding`.

# %%
w_map = lambda t, x : irx.izeros(1)
mc_wmap = lambda t, x : jnp.array([0.])

@jax.jit
def jit_reach_set(t0, dt, tf, pt0) :
    fae = irx.FastlinAdjointEmbedding(clsys, jnp.eye(2), jnp.zeros((0,2)), perm, iterated=True)
    mc_traj = clsys.compute_trajectory(t0,tf,ix0.upper,(mc_wmap,),dt,solver=solver)
    RS = fae.compute_reachset(t0, tf, pt0, (w_map,), dt, solver=solver)
    T = round (0.5/dt)
    yy = RS.ys[0]; aux = RS.ys[1]
    ixT = irx.interval(aux[0][T]) @ irx.interval(-yy.y[T][:2], yy.y[T][2:]) + yy.ox[T]
    S1 = irx.utils.check_containment(ixT[0], irx.interval(0.5, 1.))
    S2 = irx.utils.check_containment(irx.interval(mc_traj.ys[round(0.5/dt),0]), irx.interval(0.5, 1.))
    return RS, S1, S2

# JIT Compile
jit_t0 = time()
_ = jax.block_until_ready(jit_reach_set(0., dt, dt, pt0))
jit_tf = time()
print(f'JIT compiled in {jit_tf - jit_t0:.5f} seconds')

# %%
N = 10
(RS, S1, S2), times = irx.utils.run_times(N, jit_reach_set, t0, dt, tf, pt0)
print(f'Computed Reachable Set in {jnp.mean(times):.5f} ± {jnp.std(times):.5f} over {N} runs')

runtime = jnp.mean(times)

if S1 == -1 or S2 == -1 :
    status = 'VIOLATED'
elif S1 == 1 and S2 == 1 :
    status = 'VERIFIED'
else :
    status = 'UNKNOWN'

print(f'SinglePendulum,reach,{status},{runtime},comment')

# %% [markdown]
# ## Plotting

# %%
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pypoman import plot_polygon
    import numpy as onp

    yy = RS.ys[0]
    tfinite = jnp.where(jnp.isfinite(RS.ts))
    tt = RS.ts[tfinite]
    aux = RS.ys[1]

    fig, ax = plt.subplots()

    PLOT_EVERY = 1

    irx.utils.draw_iarray(ax, irx.interval([0.5, 0.], [1., 1.]), color='tab:green', alpha=0.5)

    # Nominal trajectories first so the reach-set band draws on top
    ax.plot(tt, yy.ox[tfinite][:,0], zorder=1)

    def mc_wmap (t, x) :
        return jnp.array([0.])
    for mc_x0 in [ix0.lower, ix0.upper] :
        mc_traj = clsys.compute_trajectory(t0,tf,mc_x0,(mc_wmap,),dt,solver=solver)
        mc_finite = jnp.where(jnp.isfinite(mc_traj.ts))
        if PLOT_EVERY == 1:
            ax.plot(mc_traj.ts[mc_finite], mc_traj.ys[mc_finite][:,0], color='tab:red', zorder=0)
        else :
            ax.scatter(mc_traj.ys[mc_finite][::PLOT_EVERY,0], mc_traj.ys[mc_finite][::PLOT_EVERY,1], color='tab:red', zorder=0, s=1)

    # Reach-set band on top
    mins = []
    maxs = []
    for i in range(0,len(tt),PLOT_EVERY) :
        pti = pt0.__class__.from_parametope(irx.hParametope(yy.ox[i], yy.alpha[i,:], yy.y[i,:]))
        proj = pti.one_d_proj()
        mins.append(onp.min(proj))
        maxs.append(onp.max(proj))
    irx.utils.plot_interval_t(ax, tt, irx.interval(mins, maxs) + yy.ox[tfinite][:,0], zorder=2)

    fig.savefig('single_pendulum.png', dpi=300)
    fig.savefig('single_pendulum.pdf')
    print('Saved single_pendulum.{png,pdf}')
