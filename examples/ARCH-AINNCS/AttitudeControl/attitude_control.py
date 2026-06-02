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
# # Attitude Control
#
# We consider the attitude control of a rigid body with six states and three inputs. This benchmark is described in [61, 68] of the ARCH-COMP25 report.

# %%
import argparse
import jax
import jax.numpy as jnp
import immrax as irx
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true', help='save reach-set plot (attitude_control.png/.pdf)')
args, _ = parser.parse_known_args()

# %% [markdown]
# ## The System Dynamics
#
# The state $x = (\omega^T, \psi^T)^T$ consists of the angular velocity vector in a body-fixed frame $\omega \in \mathbb{R}^3$ and the Rodrigues parameter vector $\psi \in \mathbb{R}^3$. The control torque $u \in \mathbb{R}^3$ is updated every 0.1 s by a neural network controller. 
#
# The system dynamics is given by:
# $$
# \begin{aligned}
# \dot{\omega}_1 &= 0.25(u_0 + \omega_2 \omega_3), \\
# \dot{\omega}_2 &= 0.5(u_1 - 3\omega_1 \omega_3), \\
# \dot{\omega}_3 &= u_2 + 2\omega_1 \omega_2, \\
# \dot{\psi}_1 &= 0.5 (\omega_2 (S - \psi_3) + \omega_3 (S + \psi_2) + \omega_1 (S + 1)), \\
# \dot{\psi}_2 &= 0.5 (\omega_1 (S + \psi_3) + \omega_3 (S - \psi_1) + \omega_2 (S + 1)), \\
# \dot{\psi}_3 &= 0.5 (\omega_1 (S - \psi_2) + \omega_2 (S + \psi_1) + \omega_3 (S + 1)),
# \end{aligned}
# $$
# where $S = \psi_1^2 + \psi_2^2 + \psi_3^2$.

# %% [markdown]
# ## The Neural Network Controller
#
# The control torque $u$ is computed by a neural network with three hidden layers, each having 64 neurons. The activations of the hidden layers are sigmoid and identity, respectively.

# %%
class AttitudeControl(irx.System):
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 6

    def f(self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        x1, x2, x3, x4, x5, x6 = x
        u1, u2, u3 = u
        S = x4**2 + x5**2 + x6**2

        dx1 = 0.25 * (u1 + x2 * x3)
        dx2 = 0.5 * (u2 - 3 * x1 * x3)
        dx3 = u3 + 2 * x1 * x2
        dx4 = 0.5 * (x2 * (S - x6) + x3 * (S + x5) + x1 * (S + 1))
        dx5 = 0.5 * (x1 * (S + x6) + x3 * (S - x4) + x2 * (S + 1))
        dx6 = 0.5 * (x1 * (S - x5) + x2 * (S + x4) + x3 * (S + 1))

        return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6])

olsys = AttitudeControl()
net = irx.NeuralNetwork('model')
clsys = irx.NNCSystem(olsys, net)

# %% [markdown]
# ## Initial Conditions and Specifications
#
# The initial state set is:
# $$
# \begin{aligned}
# \omega_1 &\in [-0.45, -0.44], \omega_2 \in [-0.55, -0.54], \omega_3 \in [0.65, 0.66], \\
# \psi_1 &\in [-0.75, -0.74], \psi_2 \in [0.85, 0.86], \psi_3 \in [-0.65, -0.64].
# \end{aligned}
# $$
#
# **Specification:** The system should not reach the following unsafe set in 3 s (30 time steps):
# $$
# \begin{aligned}
# \omega_1 &\in [-0.2, 0], \omega_2 \in [-0.5, -0.4], \omega_3 \in [0, 0.2], \\
# \psi_1 &\in [-0.7, -0.6], \psi_2 \in [0.7, 0.8], \psi_3 \in [-0.4, -0.2].
# \end{aligned}
# $$

# %%
perm = irx.standard_permutation(1+6+3+1)[0]
lower = jnp.array([-0.45, -0.55, 0.65, -0.75, 0.85, -0.65])
ix0 = irx.interval(lower, lower + 0.01)
cent, pert = irx.i2centpert(ix0)
pt0 = irx.Polytope.from_interval(ix0)

t0 = 0.
tf = 3.
dt = 0.01
max_steps = round((tf - t0) / dt) + 1
solver = 'tsit5'

unsafe = irx.interval([-0.2, -0.5, 0., -0.7, 0.7, -0.4], [0., -0.4, 0.2, -0.6, 0.8, -0.2])

# %% [markdown]
# ## Reachability Analysis
#
# We compute the reachable set using `immrax`'s `FastlinAdjointEmbedding` and `tsit5` solver.

# %%
w_map = lambda t, x : irx.izeros(1)

@jax.jit
def jit_reach_set(t0, dt, tf, pt0) :
    fae = irx.FastlinAdjointEmbedding(clsys, jnp.eye(6), jnp.zeros((0,6)), perm, iterated=True)
    RS = fae.compute_reachset(t0, tf, pt0, (w_map,), dt, solver=solver)
    def safe (oxi, alphai, yi, alphapi) :
        pti = pt0.__class__.from_parametope(irx.hParametope(oxi, alphai, yi))
        ix = irx.interval(alphapi) @ pti.iy + oxi
        return irx.utils.check_containment(ix, unsafe)
    yy = RS.ys[0]
    aux = RS.ys[1]
    n = max_steps + 1
    S = jax.vmap(safe, in_axes=(0, 0, 0, 0))(
        yy.ox[:n], yy.alpha[:n], yy.y[:n], aux[0][:n]
    )
    return RS, jnp.any(S == -1).astype(int) + 2*jnp.any(S == 0).astype(int)

# JIT Compile
jit_t0 = time()
_ = jax.block_until_ready(jit_reach_set(0., dt, dt, pt0))
jit_tf = time() 
print(f'JIT compiled in {jit_tf - jit_t0:.5f} seconds')

# %%
N = 10
(RS, S), times = irx.utils.run_times(N, jit_reach_set, t0, dt, tf, pt0)
print(f'Computed Reachable Set in {jnp.mean(times):.5f} ± {jnp.std(times):.5f} over {N} runs')
runtime = jnp.mean(times)

if S == 0 :
    status = 'VERIFIED'
elif S % 2 == 1 :
    status = 'VIOLATED'
else :
    status = 'UNKNOWN'

print(f'AttitudeControl,avoid,{status},{runtime},comment')

# %%
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as onp

    yy = RS.ys[0]
    tfinite = jnp.where(jnp.isfinite(RS.ts))
    tt = RS.ts[tfinite]
    aux = RS.ys[1]

    fig, ax = plt.subplots()

    PLOT_EVERY = 1

    # Unsafe region projected to (ω1, ω2). Add a legend handle via label.
    unsafe_lo = onp.array(unsafe.lower)
    unsafe_hi = onp.array(unsafe.upper)
    ax.add_patch(Rectangle(
        (unsafe_lo[0], unsafe_lo[1]),
        unsafe_hi[0] - unsafe_lo[0], unsafe_hi[1] - unsafe_lo[1],
        ec='tab:red', fc='tab:red', alpha=0.4, label='Unsafe', zorder=2,
    ))

    # Monte Carlo trajectories under the reach set
    def mc_wmap (t, x) :
        return jnp.array([0.])
    for mc_x0 in irx.utils.gen_ics(ix0, 100) :
        mc_traj = clsys.compute_trajectory(t0,tf,mc_x0,(mc_wmap,),dt,solver=solver)
        mc_finite = jnp.where(jnp.isfinite(mc_traj.ts))
        ax.plot(mc_traj.ys[mc_finite][:,0], mc_traj.ys[mc_finite][:,1],
                color='tab:red', alpha=0.3, lw=0.5, zorder=0)

    # Nominal trajectory in (ω1, ω2)
    ax.plot(yy.ox[tfinite][:,0], yy.ox[tfinite][:,1], c='blue', zorder=1)

    # Reach-set parametope projections (green outlined polygons)
    for i in range(0, len(tt), PLOT_EVERY) :
        pti = pt0.__class__.from_parametope(
            irx.hParametope(yy.ox[i], yy.alpha[i,:], yy.y[i,:])
        )
        try :
            pti.plot_projection(ax, xi=0, yi=1, color='tab:green', linewidth=0.5)
        except Exception:
            pass

    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.legend(loc='upper left')

    fig.savefig('attitude_control.png', dpi=300)
    fig.savefig('attitude_control.pdf')
    print('Saved attitude_control.{png,pdf}')
