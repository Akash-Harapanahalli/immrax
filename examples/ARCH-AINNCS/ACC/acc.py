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
# # Adaptive Cruise Control (ACC)
# The Adaptive Cruise Control (ACC) benchmark is a system that tracks a set velocity and maintains a safe distance from a lead vehicle by adjusting the longitudinal acceleration of an ego vehicle. The neural network computes optimal control actions while satisfying safe distance, velocity, and acceleration constraints using model predictive control (MPC).
#
# This example is from the [ARCH-COMP25 AINNCS category report](https://gitlab.com/goranf/ARCH-COMP).

# %%
import argparse
import jax
import jax.numpy as jnp
import immrax as irx
from time import time
from immutabledict import immutabledict

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true', help='save reach-set plots (acc_plot.png/.pdf)')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## The Vehicle's Dynamics
#
# The ego car is set to travel at a set speed $v_{set} = 30$ and maintains a safe distance $D_{safe}$ from the lead car. The car's dynamics are described by the following equations:
#
# $$
# \begin{aligned}
# \dot{x}_{lead}(t) &= v_{lead}(t), \dot{v}_{lead}(t) = a_{lead}(t), \dot{a}_{lead}(t) = -2a_{lead}(t) + 2a_{c,lead} - \mu v_{lead}(t)^2, \\
# \dot{x}_{ego}(t) &= v_{ego}(t), \dot{v}_{ego}(t) = a_{ego}(t), \dot{a}_{ego}(t) = -2a_{ego}(t) + 2a_{c,ego} - \mu v_{ego}(t)^2,
# \end{aligned}
# $$
#
# where $x_i$ is the position, $v_i$ is the velocity, $a_i$ is the acceleration of the car, $a_{c,i}$ is the acceleration control input applied to the car, and $\mu = 0.0001$ is a coefficient for air drag, where $i \in \{ego, lead\}$.
#
# We implement this in `immrax` as an `OpenLoopSystem` as follows.

# %% [markdown]
# ## The Neural Network Controller
#
# We evaluate a neural network controller with five layers and 20 neurons each. The inputs of the controller are the set speed $v_{set}$, the desired time gap $T_{gap}$, the ego velocity $v_{ego}$, the distance $D_{rel} = x_{lead} - x_{ego}$, as well as the relative velocity $v_{rel}$, and the output is $a_{c,ego}$.

# %%
class ACC(irx.System):
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 6  # 6 state variables: x1 to x6

    def f(self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        mu = 0.0001       # friction parameter
        a_lead = -2       # constant lead car acceleration
        a_ego = u[0]      # ego car acceleration input

        # Unpack state variables
        x1, x2, x3, x4, x5, x6 = x

        # Lead car dynamics
        dx1 = x2
        dx2 = x3
        dx3 = -2 * x3 + 2 * a_lead - mu * x2**2

        # Ego car dynamics
        dx4 = x5
        dx5 = x6
        dx6 = -2 * x6 + 2 * a_ego - mu * x5**2

        return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6])

vset = 30.
Ddefault = 10.
Tgap = 1.4

olsys = ACC()
prenet = irx.NeuralNetwork('controller_5_20')
def net (x) :
    vego = x[4]
    Drel = x[0] - x[3]
    vrel = x[1] - x[4]
    return prenet(jnp.array([vset, Tgap, vego, Drel, vrel]))
net.out_len = 1

clsys = irx.NNCSystem(olsys, net)

# %% [markdown]
# ## Initial Conditions and Specifications
#
# The verification objective of this system is that given a scenario where both cars are driving safely, the lead car suddenly slows down with $a_{c,lead} = -2$. We want to check whether there is a collision in the following 5 s. Formally, this safety specification of the system can be expressed as $D_{rel} \ge D_{safe}$, where $D_{safe} = D_{default} + T_{gap} \cdot v_{ego}$, and $T_{gap} = 1.4$ s and $D_{default} = 10$.

# %%
ix0 = irx.interval([90., 32., 0., 10., 30, 0.], [110., 32.2, 0., 11., 30.2, 0.])
t0 = 0.
dt = 0.05
tf = 5.
max_steps = round((tf - t0) / dt) + 1

# %% [markdown]
# ## Reachability Analysis
#
# We compute the reachable set using `immrax`'s `crown` verifier for the neural network and `tsit5` solver for the embedding system trajectory.

# %%
w_map = lambda t,x : irx.izeros(1)
f_kwargs = immutabledict({
    'permutations': irx.standard_permutation (1+6+1+1),
    'corners': irx.bot_corner(1+6+1+1),
})

@jax.jit
def jit_reach_set (t0, dt, tf, ix0) :
    verifier = irx.crown(net, iterated=True)
    F_ol = irx.natif(olsys.f)

    class EMBSYS (irx.EmbeddingSystem) :
        def __init__ (self) :
            self.sys = clsys
            self.xlen = self.sys.xlen * 2
            self.evolution = 'continuous'
        def E (self, t, x) :
            ix = irx.ut2i(x)
            verifier_res = verifier(ix)
            def F (t, ix) :
                return F_ol(t, ix, verifier_res(ix), irx.izeros(1))
            return irx.embed(F)(t, irx.i2ut(ix))
    
    embsys = EMBSYS()

    traj = embsys.compute_trajectory(t0, tf, irx.i2ut(ix0), dt=dt, solver='tsit5')

    ix_lead = irx.interval(traj.ys[:max_steps,0:3], traj.ys[:max_steps,6:9])
    ix_ego = irx.interval(traj.ys[:max_steps,3:6], traj.ys[:max_steps,9:12])

    Drel = ix_lead[:,0] - ix_ego[:,0]
    Dsafe = ix_ego[:,1]*Tgap + Ddefault
    S = jax.vmap(irx.utils.check_containment, in_axes=(0,None))(Drel - Dsafe, irx.interval(0., jnp.inf))

    return traj, jnp.any(S == -1).astype(int) + 2*jnp.any(S == 0).astype(int)


# JIT Compile
jit_t0 = time()
jax.block_until_ready(jit_reach_set(0., dt, dt, ix0))
jit_tf = time()
print(f'JIT compiled in {jit_tf - jit_t0:.5f} seconds')

# %%
N = 10
(RS, S), times = irx.utils.run_times(N, jit_reach_set, t0, dt, tf, ix0)
print(f'Computed Reachable Set in {jnp.mean(times):.5f} ± {jnp.std(times):.5f} over {N} runs')

print(S)

runtime = jnp.mean(times)

if S == 0 :
    status = 'VERIFIED'
elif S % 2 == 1 :
    status = 'VIOLATED'
else :
    status = 'UNKNOWN'

print(f'ACC,safe-distance,{status},{runtime},comment')

# %%
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()
    ix_lead = irx.interval(RS.ys[:,0:3], RS.ys[:,6:9])
    ix_ego = irx.interval(RS.ys[:,3:6], RS.ys[:,9:12])

    Drel = ix_lead[:,0] - ix_ego[:,0]
    Dsafe = ix_ego[:,1]*Tgap + Ddefault
    irx.utils.plot_interval_t(axs, RS.ts, Drel, color='tab:blue', alpha=0.5)
    irx.utils.plot_interval_t(axs, RS.ts, Dsafe, color='tab:red', alpha=0.5)

    fig.savefig('acc.png', dpi=300)
    fig.savefig('acc.pdf')
    print('Saved acc.{png,pdf}')
