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
# # Robot-arm reachable-set synthesis with `ReachiLQR`
#
# We synthesize a tight reachable-set enclosure for a 4-state planar 2-link arm
# using iterative LQR over a normotope embedding (`immrax.ReachiLQR`). Unlike the
# Van der Pol example, this runs a single constant penalty $R = 20$ for 20
# iterations — enough to drive the set to the full horizon
# $i_{\text{final}} = 1000$ at terminal log-det cost $\approx -19.5$.

# %%
import os

import jax
import jax.numpy as jnp
import numpy as onp
import time
from typing import ClassVar
from jax import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

import immrax as irx
from immrax.parametric import ReachiLQR

# Output directory for the saved figures (works both as a script and a notebook).
try:
    HERE = os.path.dirname(__file__)
except NameError:
    HERE = os.getcwd()
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## The robot-arm system
#
# A 4-state planar 2-link arm with PD joint control, state
# $x = (x_1, x_2, x_3, x_4)$ (two angles and their velocities):
#
# $$
# \begin{aligned}
# \dot{x}_1 &= x_3, \\
# \dot{x}_2 &= x_4, \\
# \dot{x}_3 &= \frac{-2 m\, x_2 x_3 x_4 - k_{p1} x_1 - k_{d1} x_3 + k_{p1} u_1}
#                  {m\, x_2^2 + \ell/3}, \\
# \dot{x}_4 &= x_2 x_3^2 - \frac{k_{p2} x_2}{m} - \frac{k_{d2} x_4}{m}
#             + \frac{k_{p2} u_2}{m},
# \end{aligned}
# $$
#
# with $m = 1$, $\ell = 3$, gains $k_p = (2, 1)$, $k_d = (2, 1)$, and constant
# setpoint input $u = (k_{p1}, k_{p2})$.

# %%
class Robot(irx.System):
    """Planar 2-link arm (4-state). Parameters from emsoft-code reference."""

    xlen: ClassVar[int] = 4

    def f(self, t, x):
        m, l = 1.0, 3.0
        kp1, kp2 = 2.0, 1.0
        kd1, kd2 = 2.0, 1.0
        u1, u2 = kp1, kp2
        dx1 = x[2]
        dx2 = x[3]
        dx3 = (-2 * m * x[1] * x[2] * x[3] - kp1 * x[0] - kd1 * x[2]) / (
            m * x[1] * x[1] + l / 3
        ) + (kp1 * u1) / (m * x[1] * x[1] + l / 3)
        dx4 = x[1] * x[2] ** 2 - kp2 * x[1] / m - kd2 * x[3] / m + kp2 * u2 / m
        return jnp.array([dx1, dx2, dx3, dx4])


sys = Robot()

# %% [markdown]
# ## Initial set
#
# An $\ell_2$ normotope (ellipsoid) initial set centered at
# $\mathring{x} = (1.505, 1.505, 0.005, 0.005)$, with shape matrix
# $\alpha = P_0^{1/2} / 0.1$ for the covariance-like $P_0$ below. The horizon is
# $t \in [0, 10]$ with $N = 1000$ Euler steps.

# %%
P0 = jnp.array(
    [
        [2.5063, -0.0276, 0.9436, 0.0280],
        [-0.0276, 2.4722, 0.0013, 0.9537],
        [0.9436, 0.0013, 2.2441, 0.0151],
        [0.0280, 0.9537, 0.0151, 2.2845],
    ]
)
x0 = jnp.array([1.505, 1.505, 0.005, 0.005])
alpha0 = jnp.real(jax.scipy.linalg.sqrtm(P0))
nt0 = irx.L2Normotope(x0, alpha0 / 0.1, 1.0)
N = 1000
dt = 0.01

# %% [markdown]
# ## Cost functions
#
# As in the Van der Pol example, the terminal cost is the (negative) log-volume
# of the normotope, $\Phi = -\sum_i \log L_{ii}$ with
# $L = \operatorname{chol}(\alpha^\top\alpha / y^2)$, and the running cost is zero
# (the penalty $R$ acts as Levenberg–Marquardt damping inside `iterate`).

# %%
def terminal_cost(pt):
    M = pt.alpha.T @ pt.alpha / pt.y**2
    L = jnp.linalg.cholesky(M)
    return -jnp.sum(jnp.log(jnp.diag(L)))


def running_cost(t, pt, U):
    return jnp.array(0.0)

# %% [markdown]
# ## Embedding and the iLQR solver

# %%
embedding = irx.NormotopeEmbedding(sys)
ilqr = ReachiLQR(
    embedding=embedding,
    pt0=nt0,
    terminal_cost=terminal_cost,
    running_cost=running_cost,
    t0=0.0,
    tf=N * dt,
    N=N,
    DDP=False,
    solver="euler",
    dt=dt,
)

print(f"initial cost = {terminal_cost(nt0): .4f}")
print(f"Xlen={ilqr.Xlen}  Ulen={ilqr.Ulen}  N={ilqr.N}")

# %% [markdown]
# ## Monte Carlo validation
#
# 100 boundary rollouts of the initial set (uncontrolled) for an inner reference.

# %%
mc_key = jax.random.PRNGKey(0)
mc_x0s = nt0.sample_boundary(mc_key, 100)


@jax.jit
def rollout(x):
    def step(carry, _):
        n = carry + dt * sys.f(0.0, carry)
        return n, n

    _, xs = jax.lax.scan(step, x, None, length=N)
    return jnp.vstack([x, xs])


mc_trajs = jax.vmap(rollout)(mc_x0s)

# %% [markdown]
# ## Solving with `run()`
#
# A constant schedule ($R = 20$, no revert) for 20 iterations. Passing a scalar
# `R_schedule` is wrapped automatically by `constant_schedule`. We call
# `ilqr.setup()` first to JIT-compile one iterate, so the per-iteration times
# recorded by the `on_iterate` callback exclude compile cost.

# %%
R = 20.0
ITERS = 20
gamma = 1.0
Jmax = 2.0

# JIT warmup (compile one iterate) so the recorded iter_times exclude compile cost.
t_jit = time.time()
ilqr.setup()
print(f"JIT compile: {time.time() - t_jit:.1f}s")

states = []  # per-iter info for plotting
iter_times = []
last = [time.time()]  # last per-iter timestamp (mutable for the closure)


def record(i, R_i, res):
    jax.block_until_ready(res.l_traj)
    now = time.time()
    iter_times.append(now - last[0])
    last[0] = now
    ifinal = int(res.ifinal)
    cost = float(res.cost_final)
    states.append(
        {"pert_traj": onp.asarray(res.pert_traj), "ifinal": ifinal, "cost": cost}
    )
    print(f"  k={i:>3d}  ifinal={ifinal:>5d}  cost={cost: .4f}  ({iter_times[-1]:.2f}s)")


last[0] = time.time()
ilqr.run(R_schedule=R, iters=ITERS, gamma=gamma, Jmax=Jmax, on_iterate=record)
print(f"Final: ifinal={states[-1]['ifinal']}  cost={states[-1]['cost']:.4f}")

# %% [markdown]
# ## Reachable-set frames
#
# Four iterates of the projected ellipse tube ($x_1$ vs $x_2$) against the
# Monte-Carlo samples.

# %%
PLOT_EVERY = 50


def plot_i(i, ax):
    ax.clear()
    ax.scatter(mc_trajs[:, ::PLOT_EVERY, 0], mc_trajs[:, ::PLOT_EVERY, 1],
               color="tab:red", s=0.1)
    nt0.plot_projection(ax)
    ifinal = states[i]["ifinal"]
    pert_traj = states[i]["pert_traj"]
    for j in range(PLOT_EVERY, ifinal, PLOT_EVERY):
        nti = irx.L2Normotope.unvec(jnp.asarray(pert_traj[j]))
        nti.plot_projection(ax)
    ax.set_title(f"$i={i + 1}$")
    ax.set_xlim(1.4, 2.2)
    ax.set_ylim(0.9, 1.6)


frames = [0, 3, 4, 19]
fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)
for frame, ax in zip(frames, axs.flatten()):
    plot_i(frame, ax)
axs[0, 0].set_ylabel("$x_2$")
axs[1, 0].set_ylabel("$x_2$")
axs[1, 0].set_xlabel("$x_1$")
axs[1, 1].set_xlabel("$x_1$")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"robot_arm_ilqr_frames.{ext}"), dpi=150)
plt.show()

# %% [markdown]
# ## Cost vs. iteration
#
# Terminal cost per iLQR iteration (markers distinguish iterates that reached the
# horizon from those that did not), with cumulative post-JIT runtime on the right
# axis.

# %%
final_vols = onp.array([s["cost"] for s in states])
pre_x = onp.array([i for i, s in enumerate(states) if s["ifinal"] < N], dtype=int)
post_x = onp.array([i for i, s in enumerate(states) if s["ifinal"] >= N], dtype=int)
ii = onp.arange(1, len(states) + 1)
cum_t = onp.cumsum(iter_times)

fig, ax = plt.subplots(figsize=(7, 3))
color = "blue"
ax.plot(ii, final_vols, color=color)
ax.xaxis.get_major_locator().set_params(integer=True)
ax.set_xlabel("iLQR Iteration $i$")
ax.set_ylabel(r"$\Phi(\mathring{x}(t_f),\alpha(t_f),y(t_f))$", color=color)
ax.tick_params(axis="y", labelcolor=color)
if len(pre_x) > 0:
    ax.scatter(pre_x + 1, final_vols[pre_x], color=color, s=20, marker="x",
               label="Did not reach end")
if len(post_x) > 0:
    ax.scatter(post_x + 1, final_vols[post_x], color=color, s=20, marker="o",
               facecolors="none", label="Reached end")
ax.legend(loc="upper center")

color2 = "red"
ax2 = ax.twinx()
ax2.plot(ii, cum_t, color=color2)
ax2.set_ylabel("Cumulative runtime (s)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.scatter(ii, cum_t, color=color2, s=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"robot_arm_ilqr_cost.{ext}"), dpi=150)
plt.show()
