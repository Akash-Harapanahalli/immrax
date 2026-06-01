"""RobotArm reachable-set synthesis via ReachiLQR.

Mirrors Harapanahalli_ACC2026/RobotArmSynth.ipynb: a 4-state planar arm with an
L2 normotope initial set, log-det terminal cost, R=20, 20 outer iterations.
Produces the notebook's two figures:

* ``robot_arm_ilqr_frames.png`` — 2x2 grid of reachable-set snapshots at
  iterations ``[0, 3, 4, 19]``.
* ``robot_arm_ilqr_cost.png``   — terminal cost ``Phi(x(t_f))`` vs iLQR
  iteration, with markers for whether each iter reached the end.

Run from the repo root with:  python examples/iLQR/robot_arm.py
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as onp
from jax import config

config.update("jax_enable_x64", True)

import immrax as irx
from immrax.parametric import ReachiLQR

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- system

class Robot(irx.System):
    """Planar 2-link arm (4-state). Parameters from emsoft-code reference."""

    def __init__(self):
        self.xlen = 4
        self.evolution = "continuous"

    def f(self, t, x):
        m, l = 1.0, 3.0
        kp1, kp2 = 2.0, 1.0
        kd1, kd2 = 2.0, 1.0
        u1, u2 = kp1, kp2
        dx1 = x[2]
        dx2 = x[3]
        dx3 = (
            -2 * m * x[1] * x[2] * x[3] - kp1 * x[0] - kd1 * x[2]
        ) / (m * x[1] * x[1] + l / 3) + (kp1 * u1) / (m * x[1] * x[1] + l / 3)
        dx4 = x[1] * x[2] ** 2 - kp2 * x[1] / m - kd2 * x[3] / m + kp2 * u2 / m
        return jnp.array([dx1, dx2, dx3, dx4])


sys = Robot()


# ---------------------------------------------------------------- initial set

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


# ---------------------------------------------------------------- costs

def terminal_cost(pt):
    M = pt.alpha.T @ pt.alpha / pt.y ** 2
    L = jnp.linalg.cholesky(M)
    return -jnp.sum(jnp.log(jnp.diag(L)))


def running_cost(t, pt, U):
    return jnp.array(0.0)


# ---------------------------------------------------------------- iLQR

embedding = irx.NormotopeEmbedding(sys)
ilqr = ReachiLQR(
    embedding=embedding,
    pt0=nt0,
    terminal_cost=terminal_cost,
    running_cost=running_cost,
    control_shape=nt0.alpha.shape,
    t0=0.0,
    tf=N * dt,
    N=N,
    DDP=False,
    solver="euler",
    dt=dt,
)


def main():
    print(f"initial cost = {terminal_cost(nt0): .4f}")
    print(f"Xlen={ilqr.Xlen}  Ulen={ilqr.Ulen}  N={ilqr.N}")

    # Monte Carlo from boundary of initial set.
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

    # Outer loop: R=20, 20 iterations (matches the notebook), but save every iterate
    # for plotting the convergence frames.
    R = 20.0
    ITERS = 20
    gamma = 1.0
    Jmax = 2.0

    Us = ilqr.initial_controls()
    l_traj, K_traj = ilqr.initial_gains()

    # JIT warmup so iter_times reflect only the post-JIT compute cost (the JIT
    # graph for ReachiLQR.iterate is a few seconds and would otherwise dominate
    # iter 0's reported runtime).
    print("Warming up JIT...")
    _warm = ilqr.iterate(Us, l_traj, K_traj, R=R, gamma=gamma, Jmax=Jmax)
    jax.block_until_ready(_warm.l_traj)

    states = []  # save per-iter info for plotting
    iter_times = []

    t_start = time.time()
    for k in range(ITERS):
        ti = time.time()
        res = ilqr.iterate(Us, l_traj, K_traj, R=R, gamma=gamma, Jmax=Jmax)
        jax.block_until_ready(res.l_traj)
        iter_times.append(time.time() - ti)
        ifinal = int(res.ifinal)
        cost = float(res.cost_final)
        states.append(
            {
                "pert_traj": onp.asarray(res.pert_traj),
                "ifinal": ifinal,
                "cost": cost,
            }
        )
        print(f"  k={k:>3d}  ifinal={ifinal:>5d}  cost={cost: .4f}  ({iter_times[-1]:.2f}s)")
        Us = res.Us_new
        l_traj = res.l_traj
        K_traj = res.K_traj

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")
    final_ifinal = states[-1]["ifinal"]
    final_cost = states[-1]["cost"]
    print(f"Final: ifinal={final_ifinal}  cost={final_cost:.4f}")

    # --------------------------------------------------------- plotting

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOT_EVERY = 50

        def plot_i(i, ax):
            ax.clear()
            ax.scatter(
                mc_trajs[:, ::PLOT_EVERY, 0],
                mc_trajs[:, ::PLOT_EVERY, 1],
                color="tab:red",
                s=0.1,
            )
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
            out_frames = os.path.join(OUT, f"robot_arm_ilqr_frames.{ext}")
            fig.savefig(out_frames, dpi=150)
            print(f"Saved frames plot to {out_frames}")
        plt.close(fig)

        # Cost vs iteration with markers for "reached the end" vs not.
        final_vols = onp.array([s["cost"] for s in states])
        pre_x, post_x = [], []
        for i, s in enumerate(states):
            if s["ifinal"] < N:
                pre_x.append(i)
            else:
                post_x.append(i)
        pre_x = onp.array(pre_x, dtype=int)
        post_x = onp.array(post_x, dtype=int)
        ii = onp.arange(1, len(states) + 1)
        cum_t = onp.cumsum(iter_times)

        fig, ax = plt.subplots(figsize=(7, 3))
        color = "blue"
        ax.plot(ii, final_vols, color=color)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("iLQR Iteration $i$")
        ax.set_ylabel(
            r"$\Phi(\mathring{x}(t_f),\alpha(t_f),y(t_f))$", color=color
        )
        ax.tick_params(axis="y", labelcolor=color)
        if len(pre_x) > 0:
            ax.scatter(
                pre_x + 1,
                final_vols[pre_x],
                color=color,
                s=20,
                marker="x",
                label="Did not reach end",
            )
        if len(post_x) > 0:
            ax.scatter(
                post_x + 1,
                final_vols[post_x],
                color=color,
                s=20,
                marker="o",
                facecolors="none",
                label="Reached end",
            )
        ax.legend(loc="upper center")

        color2 = "red"
        ax2 = ax.twinx()
        ax2.plot(ii, cum_t, color=color2)
        ax2.set_ylabel("Cumulative runtime (s)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.scatter(ii, cum_t, color=color2, s=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_cost = os.path.join(OUT, f"robot_arm_ilqr_cost.{ext}")
            fig.savefig(out_cost, dpi=150)
            print(f"Saved cost plot to {out_cost}")
        plt.close(fig)
    except Exception as e:
        print(f"(skipping plot: {e})")


if __name__ == "__main__":
    main()
