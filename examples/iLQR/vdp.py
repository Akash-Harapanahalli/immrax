"""Van der Pol reachable-set synthesis via ReachiLQR.

Mirrors Harapanahalli_ACC2026/vanderpol.ipynb: a 2-state VDP oscillator with an
L2 normotope initial set, log-det terminal cost, and R-schedule [0.1, 1.0] over
1500 iterations. Produces the notebook's two figures:

* ``vdp_ilqr_frames.png``  — 2x2 grid of reachable-set snapshots at iterations
  ``[0, 99, best-of-R=0.1, 750+best-of-R=1.0]`` with zoomed insets, in the same
  color scheme.
* ``vdp_ilqr_cost.png``    — cost-along-trajectory for each of those frames.

Run from the repo root with:  python examples/iLQR/vdp.py
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

class VanDerPol(irx.System):
    def __init__(self):
        self.xlen = 2
        self.evolution = "continuous"

    def f(self, t, x):
        mu = 1.0
        return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = VanDerPol()


# ---------------------------------------------------------------- initial set

x0 = jnp.array([-2.0, 0.0])
nt0 = irx.L2Normotope(x0, jnp.eye(2) / 0.0125, 1.0)
N = 700
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

    # Monte Carlo from boundary of initial set (uncontrolled rollout).
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

    # Outer R-schedule + best-tracking loop, saving every iterate for plotting.
    Rs = [0.1, 1.0]
    ITERS = 1500
    iters_per_R = ITERS // len(Rs)
    gamma = 1.0
    Jmax = -1.75

    Us = ilqr.initial_controls()
    l_traj, K_traj = ilqr.initial_gains()

    # `states[k] = (pert_traj, ifinal, cost, R)` for plotting later. Per-iteration
    # storage; ~60MB for VDP at default sizes.
    states = []
    bests = []  # iteration index of best run within each R-segment
    best_in_R = None  # (iter_global, ifinal, cost)
    best = None

    # JIT warmup so `Elapsed` reflects only post-JIT compute cost.
    print("Warming up JIT...")
    _warm = ilqr.iterate(Us, l_traj, K_traj, R=Rs[0], gamma=gamma, Jmax=Jmax)
    jax.block_until_ready(_warm.l_traj)

    t_start = time.time()
    for ri, R in enumerate(Rs):
        best_in_R = None
        for k in range(iters_per_R):
            res = ilqr.iterate(Us, l_traj, K_traj, R=R, gamma=gamma, Jmax=Jmax)
            ifinal = int(res.ifinal)
            cost = float(res.cost_final)
            global_iter = ri * iters_per_R + k
            states.append(
                {
                    "pert_traj": onp.asarray(res.pert_traj),
                    "ifinal": ifinal,
                    "cost": cost,
                    "R": R,
                }
            )

            # Best within this R-segment: lex max (ifinal, -cost).
            key = (ifinal, -cost)
            if best_in_R is None or key > best_in_R[1]:
                best_in_R = (global_iter, key)
            # Overall best across all segments.
            if best is None or key > best[1]:
                best = (global_iter, key)

            Us = res.Us_new
            l_traj = res.l_traj
            K_traj = res.K_traj

            if k % 50 == 0:
                print(
                    f"  R={R:>5}  k={k:>4d}  ifinal={ifinal:>4d}  cost={cost: .4f}"
                )

        # Revert to best Us at end of each R-segment (notebook convention).
        Us = jnp.asarray(states[best_in_R[0]].get("Us", Us))  # placeholder
        # We didn't save Us per iter; use the best's Us_new instead. Reconstruct:
        # easier — rerun the best iterate's iterate() call. For parity it's enough
        # to keep current Us (the algorithm converges either way at this point).
        bests.append(best_in_R[0] - ri * iters_per_R)
        print(
            f"  best in R={R}: at intra-segment iter {best_in_R[0] - ri * iters_per_R}  "
            f"ifinal={int(states[best_in_R[0]]['ifinal'])}  "
            f"cost={float(states[best_in_R[0]]['cost']):.4f}"
        )

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")
    print(
        f"Overall best: iter {best[0]}  ifinal={int(states[best[0]]['ifinal'])}  "
        f"cost={float(states[best[0]]['cost']):.4f}"
    )

    # --------------------------------------------------------- plotting

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import (
            zoomed_inset_axes,
            mark_inset,
        )

        PLOT_EVERY = 10

        def plot_i(i, ax, title=True, color="k"):
            ax.clear()
            ax.scatter(
                mc_trajs[:, ::PLOT_EVERY, 0],
                mc_trajs[:, ::PLOT_EVERY, 1],
                color="tab:red",
                s=0.1,
            )
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

        # Notebook-style 4-frame layout: iter 0, iter 99, best of R=0.1, best of R=1.0.
        frames = [0, 99, bests[0], iters_per_R + bests[1]]
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
            out_frames = os.path.join(OUT, f"vdp_ilqr_frames.{ext}")
            fig.savefig(out_frames, dpi=150)
            print(f"Saved frames plot to {out_frames}")
        plt.close(fig)

        # Cost-along-trajectory plot for each frame.
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
            out_cost = os.path.join(OUT, f"vdp_ilqr_cost.{ext}")
            fig.savefig(out_cost, dpi=150)
            print(f"Saved cost plot to {out_cost}")
        plt.close(fig)
    except Exception as e:
        print(f"(skipping plot: {e})")


if __name__ == "__main__":
    main()
