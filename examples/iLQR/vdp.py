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
# # Van der Pol reachable-set synthesis with `ReachiLQR`
#
# We synthesize a tight, *certified* reachable-set enclosure for the Van der Pol
# oscillator using iterative LQR over a normotope embedding (`immrax.ReachiLQR`),
# with **`iover` tracking**: at every timestep the solver keeps the running
# intersection, across iterations, of the perturbed set's interval hull and feeds
# that tighter (still valid) enclosure into the mixed-Jacobian evaluation. Because
# iLQR rotates the ellipsoid differently each iteration, the cross-iteration
# intersection is far tighter than any single iteration's axis-aligned hull.
#
# The schedule runs $R = 2$ for 1000 iterations (reverting to the best at the
# boundary) then $R = 20$ for 1000, reaching the full horizon
# $i_{\text{final}} = 700$ at terminal log-det cost $\approx -6.13$, and produces a
# certified box several times tighter than any single per-step hull.

# %%
import os
import time

import jax
import jax.numpy as jnp
import numpy as onp
from typing import ClassVar
from jax import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import immrax as irx
from immrax.parametric import ReachiLQR, phased_schedule
from immrax.utils import draw_iarray

# Output directory for the saved figures (works both as a script and a notebook).
try:
    HERE = os.path.dirname(__file__)
except NameError:
    HERE = os.getcwd()
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## The Van der Pol system
#
# The Van der Pol oscillator is the 2-state system
#
# $$ \dot{x}_1 = x_2, \qquad \dot{x}_2 = \mu\,(1 - x_1^2)\,x_2 - x_1, $$
#
# with $\mu = 1$. We define it as an `immrax.System`: only `xlen` needs to be set
# (the evolution defaults to ``"continuous"``), and $\mu$ is an ordinary,
# traceable field.

# %%
class VanDerPol(irx.System):
    xlen: ClassVar[int] = 2
    mu: float = 1.0

    def f(self, t, x):
        return jnp.array([x[1], self.mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = VanDerPol()

# %% [markdown]
# ## Initial set
#
# We take an $\ell_2$ **normotope** initial set
# $\{x : \|\alpha\,(x - \mathring{x})\|_2 \le y\}$ centered at
# $\mathring{x} = (-2, 0)$ — an ellipse with shape matrix $\alpha = I / 0.0125$ and
# radius $y = 1$. The horizon is $t \in [0, 7]$ with $N = 700$ Euler steps.

# %%
x0 = jnp.array([-2.0, 0.0])
nt0 = irx.L2Normotope(x0, jnp.eye(2) / 0.0125, 1.0)
N = 700
dt = 0.01

# %% [markdown]
# ## Cost functions
#
# The terminal cost is the (negative) log-volume of the normotope,
#
# $$ \Phi(\alpha, y) = -\log\det\!\big(\alpha^\top\alpha / y^2\big)^{1/2}
#    = -\sum_i \log L_{ii}, \qquad L = \operatorname{chol}(\alpha^\top\alpha / y^2), $$
#
# which iLQR minimizes (a smaller set $\Rightarrow$ lower cost). The running cost
# is zero; the control penalty $R$ enters as Levenberg–Marquardt damping on
# $Q_{uu}$ inside `iterate`.

# %%
def terminal_cost(pt):
    M = pt.alpha.T @ pt.alpha / pt.y**2
    L = jnp.linalg.cholesky(M)
    return -jnp.sum(jnp.log(jnp.diag(L)))


def running_cost(t, pt, U):
    return jnp.array(0.0)

# %% [markdown]
# ## Embedding and the iLQR solver
#
# `NormotopeEmbedding` lifts the system to normotope dynamics, and `ReachiLQR`
# runs iterative LQR over that embedding. With `track_iover=True`, the running
# cross-iteration intersection box is threaded through `run()` and fed into each
# mixed-Jacobian evaluation, tightening the contraction rate.

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
    track_iover=True,
)

print(f"initial cost = {terminal_cost(nt0): .4f}")
print(f"Xlen={ilqr.Xlen}  Ulen={ilqr.Ulen}  N={ilqr.N}  track_iover={ilqr.track_iover}")

# %% [markdown]
# ## Monte Carlo validation
#
# For reference we roll out 20 trajectories from the boundary of the initial set
# (uncontrolled), giving an inner approximation of the true reachable tube.

# %%
mc_key = jax.random.PRNGKey(0)
mc_x0s = nt0.sample_boundary(mc_key, 20)


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
# `run()` drives the outer loop given an R-schedule. We use a *phased* schedule:
# $R = 2$ for 1000 iterations, reverting `Us` to the best-seen at the boundary
# (the `(2.0, True, 1)` step), then $R = 20$ for 1000. The `on_iterate` callback
# records every iterate's trajectory for the plots below. `ilqr.setup()`
# JIT-compiles one iterate up front so the reported runtime excludes compile time.

# %%
schedule = phased_schedule([(2.0, False, 999), (2.0, True, 1), (20.0, False, 1000)])
n_phase0 = 1000  # iterations at R=2 before switching to R=20
gamma = 1.0
Jmax = -1.75

# `states[i] = {pert_traj, ifinal, cost, R}` per iteration, for plotting.
states = []


def record(i, R, res):
    states.append(
        {
            "pert_traj": onp.asarray(res.pert_traj),
            "ifinal": int(res.ifinal),
            "cost": float(res.cost_final),
            "R": R,
        }
    )
    if i % 100 == 0:
        print(
            f"  i={i:>4d}  R={R:>5}  ifinal={int(res.ifinal):>4d}  "
            f"cost={float(res.cost_final): .4f}"
        )


t_jit = time.time()
ilqr.setup()  # JIT warmup; excluded from the run timing below
print(f"JIT compile: {time.time() - t_jit:.1f}s")

t_start = time.time()
result = ilqr.run(R_schedule=schedule, gamma=gamma, Jmax=Jmax, on_iterate=record)
print(f"Elapsed (excl. JIT): {time.time() - t_start:.1f}s")

# %% [markdown]
# We pick out the best iterate within each $R$-phase (for the convergence frames)
# and the final accumulated intersection box (for the comparison figure).

# %%
def best_in(lo, hi):
    return max(range(lo, hi), key=lambda j: (states[j]["ifinal"], -states[j]["cost"]))


best0 = best_in(0, n_phase0)
best1 = best_in(n_phase0, len(states))
overall_best = best_in(0, len(states))
print(
    f"Overall best: iter {overall_best}  ifinal={states[overall_best]['ifinal']}  "
    f"cost={states[overall_best]['cost']:.4f}"
)

# Final accumulated intersection box (for the comparison plot).
iover_lo = onp.asarray(result.iover.lower)
iover_up = onp.asarray(result.iover.upper)

# %% [markdown]
# ## Convergence frames
#
# Four snapshots of the ellipse tube — the first iteration, an early iterate, and
# the best of each $R$-phase — against the Monte-Carlo samples (red). The set
# shrinks toward the true tube as iLQR converges.

# %%
PLOT_EVERY = 10


def plot_i(i, ax, title=True, color="k"):
    ax.clear()
    ax.scatter(mc_trajs[:, ::PLOT_EVERY, 0], mc_trajs[:, ::PLOT_EVERY, 1],
               color="tab:red", s=0.1)
    nt0.plot_projection(ax, color="tab:blue")
    ifinal = states[i]["ifinal"]
    pert_traj = states[i]["pert_traj"]
    for j in range(PLOT_EVERY, ifinal, PLOT_EVERY):
        nti = irx.L2Normotope.unvec(jnp.asarray(pert_traj[j]))
        nti.plot_projection(ax, color=color)
    if title:
        ax.set_title(f"$i={i + 1}$, $R={states[i]['R']}$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3.0, 3.0)


frames = [0, 99, best0, best1]
colors = ["blue", "orange", "green", "purple"]

fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)
for frame, ax, color in zip(frames, axs.flatten(), colors):
    plot_i(frame, ax, color=color)
    axins = zoomed_inset_axes(ax, zoom=4, loc="center")
    plot_i(frame, axins, title=False, color=color)
    axins.set_xlim(1.75, 2.15)
    axins.set_ylim(-0.5, 0.0)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

axs[0, 0].set_ylabel("$x_2$")
axs[1, 0].set_ylabel("$x_2$")
axs[1, 0].set_xlabel("$x_1$")
axs[1, 1].set_xlabel("$x_1$")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"vdp_ilqr_frames.{ext}"), dpi=150)
plt.show()

# %% [markdown]
# ## Cost along the trajectory
#
# The terminal cost $\Phi(\mathring{x}(t), \alpha(t), y(t))$ evaluated along each
# frame's tube, showing how the set volume evolves over the horizon.

# %%
fig, ax = plt.subplots(figsize=(7, 3))
for frame, color in zip(frames, colors):
    pert_traj = states[frame]["pert_traj"]
    ifinal = states[frame]["ifinal"]
    ts = onp.arange(1, ifinal + 1) * dt
    costs = onp.array(
        [
            float(terminal_cost(irx.L2Normotope.unvec(jnp.asarray(pert_traj[j]))))
            for j in range(1, ifinal + 1)
        ]
    )
    ax.plot(ts, costs, label=f"$i={frame + 1}$", color=color)
ax.legend()
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\Phi(\mathring{x}(t),\alpha(t),y(t))$")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"vdp_ilqr_cost.{ext}"), dpi=150)
plt.show()

# %% [markdown]
# ## `iover` comparison (headline figure)
#
# The payoff of tracking: for the best iterate, we overlay the normotope ellipses
# (purple), each one's per-step axis-aligned hull `iover()` (orange), and the
# **accumulated cross-iteration intersection** (green). The green box is several
# times tighter than the orange hull while still soundly enclosing the
# Monte-Carlo truth (red).

# %%
bp = states[overall_best]
pert = bp["pert_traj"]
ifin = bp["ifinal"]


def draw(ax, every=10):
    ax.scatter(mc_trajs[:, ::every, 0], mc_trajs[:, ::every, 1],
               color="tab:red", s=0.1, zorder=1)
    nt0.plot_projection(ax, color="tab:blue")
    for k in range(0, ifin, every):
        nt = irx.L2Normotope.unvec(jnp.asarray(pert[k]))
        nt.plot_projection(ax, color="tab:purple", alpha=0.45)
        # per-step iover hull (axis-aligned box of this iteration's ellipse)
        draw_iarray(ax, nt.iover(), ec="tab:orange", lw=0.5, alpha=0.7)
        # accumulated cross-iteration intersection box at this timestep
        if onp.isfinite([iover_lo[k, 0], iover_lo[k, 1],
                         iover_up[k, 0], iover_up[k, 1]]).all():
            draw_iarray(ax, irx.interval(iover_lo[k], iover_up[k]), ec="tab:green", lw=1.0)


fig, ax = plt.subplots(figsize=(7, 5))
draw(ax)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(
    handles=[
        Line2D([0], [0], color="tab:red", marker="o", ls="", ms=3, label="MC samples (truth)"),
        Line2D([0], [0], color="tab:purple", label="normotope ellipses"),
        Line2D([0], [0], color="tab:orange", label="per-step iover hull"),
        Line2D([0], [0], color="tab:green", label="accumulated intersection"),
    ],
    loc="upper left",
    fontsize=8,
)
axins = zoomed_inset_axes(ax, zoom=4.5, loc="center")
draw(axins, every=4)
axins.set_xlim(1.7, 2.2)
axins.set_ylim(-0.55, 0.05)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"vdp_iover_compare.{ext}"), dpi=150)
plt.show()
