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
# # Navigation Task (NAV)
#
# The navigation benchmark models a simplified robot navigating to a goal while avoiding an obstacle.

# %%
import argparse
import jax
import jax.numpy as jnp
import immrax as irx
from time import time
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true', help='save reach-set plot (nav.png/.pdf)')
args, _ = parser.parse_known_args()

device = 'cpu'

# %% [markdown]
# ## The System Dynamics
#
# The state $x = (p_x, p_y, \theta, v)^T$ consists of the horizontal and vertical positions $(p_x, p_y)$, velocity $v$, and the angle $\theta$ of the robot. The inputs are $u = (u_1, u_2)^T$ which control the acceleration and the heading rate.
#
# The system dynamics are given by:
# $$
# \begin{aligned}
# \dot{p}_x &= v \cos(\theta) \\
# \dot{p}_y &= v \sin(\theta) \\
# \dot{v} &= u_1 \\
# \dot{\theta} &= u_2
# \end{aligned}
# $$

# %%
class NAV (irx.System):
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 4 

    def f(self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        px, py, v, th = x
        u1, u2 = jnp.tanh(u)
        return jnp.array([
            v*jnp.cos(th),
            v*jnp.sin(th),
            u1,
            u2
        ])

olsys = NAV()
net = irx.NeuralNetwork('nn-nav-set')
# net = irx.NeuralNetwork('nn-nav-point')
clsys = irx.NNCSystem(olsys, net)

# %% [markdown]
# ## Initial Conditions and Specifications
#
# **Controller:** A neural network controller (`nn-nav-set`) is used to guide the robot.
#
# **Initial Set:**
# $$
# \begin{aligned}
# p_x &\in [2.9, 3.1] \\
# p_y &\in [2.9, 3.1] \\
# v &= 0 \\
# \theta &= 0
# \end{aligned}
# $$
#
# **Specifications:**
# -   **Goal:** Reach the box centered at origin $[0,0]$ with size $1\times 1$ ($p_x, p_y \in [-0.5, 0.5]$).
# -   **Obstacle:** Avoid the region $p_x \in [1, 2], p_y \in [1, 2]$.

# %%
perm = irx.standard_permutation(1+4+2+1)[0]
cent = jnp.array([3., 3., 0., 0.])
pert = jnp.array([0.1, 0.1, 0., 0.])
ix0 = irx.icentpert(cent, pert)
pt0 = irx.Polytope.from_interval(ix0)

t0 = 0.
tf = 6.
dt = 0.01
max_steps = round((tf-t0)/dt)
solver = 'tsit5'

goal = irx.icentpert(jnp.zeros(4), [0.5,0.5,jnp.inf,jnp.inf])
obs = irx.interval([1., 1.,-jnp.inf,-jnp.inf], [2., 2., jnp.inf, jnp.inf])

print(goal, obs)

# %% [markdown]
# ## Reachability Analysis
#
# We compute the reachable set using `immrax`'s `FastlinAdjointEmbedding`. We use a partition of the initial set to improve accuracy.

# %%
w_map = lambda t, x : irx.izeros(1)

def reach_set(t0, dt, tf, ipx0, kap) :
    ix0 = irx.iconcatenate((irx.ut2i(ipx0), irx.izeros(2)))
    # ix0 = ipx0
    pt0 = irx.Polytope.from_interval(ix0)
    Hp0 = jnp.eye(4)
    N0 = jnp.zeros((0,4))
    fae = irx.FastlinAdjointEmbedding(clsys, Hp0, N0, perm, kap=kap, forward_mode="ibp", iterated=True)
    RS = fae.compute_reachset(t0, tf, pt0, (w_map,), dt, solver=solver)

    def check_goal (oxi, alphai, yi, alphapi) :
        ix = irx.interval(alphapi) @ irx.interval(-yi[:4], yi[:4]) + oxi
        return irx.utils.check_containment(ix, goal)

    def check_obs (oxi, alphai, yi, alphapi) :
        ix = irx.interval(alphapi) @ irx.interval(-yi[:4], yi[:4]) + oxi
        return irx.utils.check_containment(ix, obs)

    yy = RS.ys[0]
    aux = RS.ys[1]
    last = max_steps
    n = max_steps + 1
    G = check_goal(yy.ox[last], yy.alpha[last], yy.y[last], aux[0][last])
    O = jax.vmap(check_obs, in_axes=(0, 0, 0, 0))(
        yy.ox[:n], yy.alpha[:n], yy.y[:n], aux[0][:n]
    )

    return RS, G, jnp.any(O == 1).astype(int) + 2*jnp.any(O == 0).astype(int)

# partitions = irx.utils.get_partitions_ut(irx.i2ut(irx.icentpert(cent[:2], pert[:2])), 15**2)
partitions = irx.utils.get_partitions_ut(irx.i2ut(irx.icentpert(cent[:2], pert[:2])), 14**2)

@partial(jax.jit, backend=device)
def get_reach_set (t0, dt, tf, partitions, kap) :
    return jax.vmap(reach_set, in_axes=(None, None, None, 0, None))(t0, dt, tf, partitions, kap)

# JIT Compile
jit_t0 = time()
_ = jax.block_until_ready(get_reach_set(0., dt, dt, partitions, 0.1))
jit_tf = time()
print(f'JIT compiled in {jit_tf - jit_t0:.5f} seconds')

# %%
# ix0 = irx.icentpert(cent, 0.125*pert)
# ix0 = irx.icentpert(cent, pert*jnp.array([0.05, 0.05, 0.1, 0.1]))
# pt0 = irx.Polytope.from_interval(ix0)
# dt = 0.01

N = 1
(RS, G, O), times = irx.utils.run_times(N, get_reach_set, t0, dt, tf, partitions, 5.)
print(f'Computed Reachable Set in {jnp.mean(times):.5f} ± {jnp.std(times):.5f} over {N} runs')

runtime = jnp.mean(times)

if jnp.all(G == 1) :
    if jnp.all(O == 0) :
        status = 'VERIFIED'
    elif jnp.all(O == 1) :
        status = 'VIOLATED'
    else :
        status = 'UNKNOWN'
elif jnp.all(G == -1) :
    status = 'VIOLATED'
else :
    status = 'UNKNOWN'

print(f'NAV,robust,{status},{runtime},comment')

# %%
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pypoman import plot_polygon
    import numpy as onp

    yy = RS.ys[0]
    aux = RS.ys[1]

    fig, ax = plt.subplots()

    PLOT_EVERY = 50

    # Goal and Obstacle
    irx.utils.draw_iarray(ax, irx.interval([1., 1.], [2., 2.]), ec='tab:red', fc='tab:red', alpha=0.5)
    irx.utils.draw_iarray(ax, irx.icentpert([0., 0.], [.5, .5]), ec='tab:green', fc='tab:green', alpha=0.5)

    errors = []

    for p in range(len(partitions)) :
        ax.plot(yy.ox[p,:,0], yy.ox[p,:,1], c='blue', alpha=0.1)

        mins = []
        maxs = []

        for i in range(0,max_steps,PLOT_EVERY) :
            pti = pt0.__class__.from_parametope(irx.hParametope(yy.ox[p,i,:], yy.alpha[p,i,:], yy.y[p,i,:]))
            try :
                pti.plot_projection(ax)
            except Exception as e :
                print(f'{p} Plotting error on {i=}')
                print(aux[0][p,i,:])
                errors.append(p)

    fig.savefig('nav.png', dpi=300)
    fig.savefig('nav.pdf')
    print('Saved nav.{png,pdf}')
