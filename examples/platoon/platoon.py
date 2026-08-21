"""Vehicle platoon reachability with the symplectic adjoint polytope embedding.

A leader vehicle follows a precomputed MPC trajectory around an obstacle;
followers PD-track their predecessor at a fixed offset. The reach set of the
full platoon (4n states) is computed with :class:`AdjointEmbedding` and the
default symplectic integrator, sweeping the number of vehicles to build a
runtime table.

The leader's nominal control is shipped as ``us.npy``. Regenerate it with
``python platoon.py --regenerate`` (requires casadi + ipopt).

Outputs: platoon_grid.{pdf,svg}, platoon_overview.{pdf,svg} (for SHOW_AGENTS
vehicles), and a LaTeX runtime table on stdout.
"""

# ruff: noqa: E402  (x64 flag must be set before jax.numpy is imported)
import argparse
import pathlib
from typing import ClassVar

import equinox as eqx
import jax

# The adjoint alpha of this contracting platoon grows like e^{||J|| t}
# (cond ~ 1e8 by tf); float32 destroys the certificate. Run in x64.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as onp

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})  # Computer Modern

HERE = pathlib.Path(__file__).parent
US_FILE = HERE / "us.npy"

# ---------------------------------------------------------------- parameters
t0, tf = 0.0, 3.0
du = 0.05  # control discretization
dtdu = 5  # integration steps per control step
dt = du / dtdu
N = round((tf - t0) / dt)

x0_leader = jnp.array([8.0, 7.0, -jnp.sqrt(3.0), -1.0])  # [px, py, vx, vy]
displacement = 0.2 * jnp.array([jnp.sqrt(3.0), 1.0, 0.0, 0.0])
obstacle_center = (4, 4)
obstacle_radius = 2.25
obstacle_padding = 1.3

AGENTS = [3, 6, 9, 12, 15, 18, 21]  # runtime sweep
SHOW_AGENTS = 6  # figures for this platoon size
PERT_VEHICLE = jnp.array([1e-2, 0.0, 1e-2, 0.0])  # initial box per vehicle


# --------------------------------------------------------- leader MPC nominal
def make_nominal():
    """Solve the obstacle-avoidance MPC for the leader; returns us (K, 2)."""
    from casadi import MX, Function, Opti, tanh

    ulim, n_horizon = 5, 20
    x = MX.sym("x", 4, 1)
    u = MX.sym("u", 2, 1)
    xdot = MX(4, 1)
    xdot[0], xdot[1] = x[2], x[3]
    xdot[2] = ulim * tanh(u[0] / ulim)
    xdot[3] = ulim * tanh(u[1] / ulim)
    f = Function("f", [x, u], [xdot])

    xF = x
    for _ in range(dtdu):
        xF = xF + dt * f(xF, u)
    F = Function("F", [x, u], [xF])

    opti = Opti()
    xx = opti.variable(4, n_horizon + 1)
    uu = opti.variable(2, n_horizon)
    xp = opti.parameter(4, 1)
    slack = opti.variable(1, n_horizon)
    opti.subject_to(xx[:, 0] == xp)
    J = 0
    for n in range(n_horizon):
        opti.subject_to(xx[:, n + 1] == F(xx[:, n], uu[:, n]))
        J += xx[0, n] ** 2 + xx[1, n] ** 2 + 0.5 * uu[0, n] ** 2 + 0.5 * uu[1, n] ** 2
        if n > 0:
            J += 5e-3 * (uu[0, n] - uu[0, n - 1]) ** 2 + 5 * (uu[1, n] - uu[1, n - 1])
        J += 1e5 * slack[0, n] ** 2
        opti.subject_to(
            (xx[0, n] - obstacle_center[0]) ** 2 + (xx[1, n] - obstacle_center[1]) ** 2
            >= (obstacle_radius * obstacle_padding) ** 2 - slack[0, n]
        )
    J += 100 * xx[0, n_horizon] ** 2 + 100 * xx[1, n_horizon] ** 2 + xx[3, n_horizon] ** 2
    opti.minimize(J)
    opti.subject_to(opti.bounded(-ulim, uu[0, :], ulim))
    opti.subject_to(opti.bounded(-ulim, uu[1, :], ulim))
    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "sb": "yes", "max_iter": 100000})

    xk = onp.asarray(x0_leader)
    us = []
    for _ in onp.arange(t0, tf, du):
        opti.set_value(xp, xk)
        for n in range(n_horizon + 1):
            opti.set_initial(xx[:, n], xk)
        sol = opti.solve()
        us.append(sol.value(uu[:, 0]))
        xk = onp.asarray(sol.value(xx[:, 1])).ravel()
    return onp.array(us)


# ------------------------------------------------------------------- system
class Platoon(irx.System):
    """Leader + PD-tracking followers; 4 states [px, py, vx, vy] per vehicle."""

    num_agents: int = eqx.field(static=True)
    states_per_vehicle: ClassVar[int] = 4
    ulim: ClassVar[float] = 5.0
    kp: ClassVar[float] = 5.0
    kv: ClassVar[float] = 5.0

    def __init__(self, n: int):
        self.num_agents = n
        self.xlen = self.states_per_vehicle * n

    def f(self, t, x, u, w):
        sv = self.states_per_vehicle
        leader = x[:sv]
        au = self.ulim * jnp.tanh(u / self.ulim)
        dyn = [jnp.array([leader[2], leader[3], au[0] + w[0], au[1] + w[1]])]
        for i in range(1, self.num_agents):
            ahead = x[(i - 1) * sv : i * sv]
            me = x[i * sv : (i + 1) * sv]
            speed = jnp.linalg.norm(ahead[2:4])
            cx = self.kp * (ahead[0] - me[0] - 0.5 * ahead[2] / speed) + self.kv * (ahead[2] - me[2])
            cy = self.kp * (ahead[1] - me[1] - 0.5 * ahead[3] / speed) + self.kv * (ahead[3] - me[3])
            dyn.append(jnp.array([me[2], me[3], cx + w[2 * i], cy + w[2 * i + 1]]))
        return jnp.concatenate(dyn)


# ------------------------------------------------------------- reachability
def platoon_x0(n):
    return jnp.concatenate([x0_leader + i * displacement for i in range(n)])


def mjac_permutation(n):
    # mjacM argument order (t, x, u, w): center positions, then velocities,
    # then control, then disturbances (matches the original study's ordering).
    perm_p = tuple(1 + 4 * i + j for i in range(n) for j in range(2))
    perm_v = tuple(3 + 4 * i + j for i in range(n) for j in range(2))
    perm_u = (4 * n + 1, 4 * n + 2)
    off = 1 + len(perm_p) + len(perm_v) + len(perm_u)
    perm_w = tuple(off + 2 * i + j for i in range(n) for j in range(2))
    return irx.Permutation((0,) + perm_p + perm_v + perm_u + perm_w)


def reach(n, us):
    platoon = Platoon(n)
    x0 = platoon_x0(n)
    pt0 = Polytope.from_interval(irx.icentpert(x0, jnp.tile(PERT_VEHICLE, n)))
    w_bounds = irx.icentpert(jnp.zeros(2 * n), jnp.zeros(2 * n))

    iu = lambda t, x: irx.interval(us[jnp.asarray(t / du).astype(int)])
    iw = lambda t, x: w_bounds

    emb = AdjointEmbedding(platoon, jnp.eye(4 * n), jnp.zeros((0, 4 * n)),
                           permutation=mjac_permutation(n))
    run = jax.jit(lambda p: emb.compute_reachset(t0, tf, p, (iu, iw), dt=dt))
    rs, times = irx.utils.run_times(6, run, pt0)
    return rs, float(jnp.mean(times[1:])), platoon


def draw_box(ax, b2, **kw):
    """draw_iarray with a floor width (shapely rejects degenerate boxes)."""
    lo, hi = onp.asarray(b2.lower), onp.asarray(b2.upper)
    if not (onp.isfinite(lo).all() and onp.isfinite(hi).all()):
        return
    pad = onp.maximum((hi - lo) / 2, 1e-4)
    c = (lo + hi) / 2
    irx.utils.draw_iarray(ax, irx.icentpert(jnp.asarray(c), jnp.asarray(pad)), **kw)


def boxes(rs, n):
    """Interval hulls per step: [H+] [hinv(y)] + ox, using the evolved H+."""
    yy, (alpha_p, _N) = rs.ys
    K = yy.y.shape[1] // 2
    out = []
    for k in range(yy.ox.shape[0]):
        iy = irx.interval(-yy.y[k, :K], yy.y[k, K:])
        out.append(irx.interval(alpha_p[k]) @ iy + yy.ox[k])
    return out


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="re-solve the leader MPC")
    args = ap.parse_args()

    if args.regenerate or not US_FILE.exists():
        us = make_nominal()
        onp.save(US_FILE, us)
        print(f"saved {US_FILE}")
    us = jnp.asarray(onp.load(US_FILE))

    # ------------------------------------------------- runtime sweep + table
    results = {}
    for n in AGENTS:
        rs, t_run, _ = reach(n, us)
        results[n] = (rs, t_run)
        print(f"  n={n}: {t_run * 1e3:.1f} ms")

    print(f"\n% platoon adjoint reachability (symplectic, dt={dt}, tf={tf})")
    print("\\begin{tabular}{rrr}")
    print("\\toprule")
    print("Vehicles & States & Runtime (ms) \\\\")
    print("\\midrule")
    for n in AGENTS:
        print(f"{n} & {4 * n} & {results[n][1] * 1e3:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")

    # ------------------------------------------------------------- figures
    n = SHOW_AGENTS
    rs, _ = results[n]
    yy = rs.ys[0]
    ix = boxes(rs, n)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in onp.linspace(0, 1, n)]
    x0s = onp.asarray(platoon_x0(n)).reshape(-1, 4)[:, :2]

    # per-agent grid
    ncols = 3
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = onp.atleast_1d(axs).reshape(-1)
    for j in range(n):
        ax = axs[j]
        ax.add_patch(patches.Circle(obstacle_center, obstacle_radius, facecolor="salmon"))
        for k in range(0, N + 1, 5):
            draw_box(ax, ix[k][4 * j : 4 * j + 2], ec=colors[j])
        ax.plot(yy.ox[:, 4 * j], yy.ox[:, 4 * j + 1], color=colors[j], alpha=0.25)
        others = onp.ones(n, bool)
        others[j] = False
        ax.scatter(x0s[others, 0], x0s[others, 1], color="k", s=5, alpha=0.1)
        ax.scatter(x0s[j, 0], x0s[j, 1], color=colors[j], s=5)
        ax.set_xlim(0, x0s[-1, 0] + 0.5)
        ax.set_ylim(-2, x0s[-1, 1] + 0.5)
        ax.set_title(f"vehicle {j + 1}", fontsize=9)
    for j in range(n, len(axs)):
        axs[j].set_axis_off()
    fig.suptitle(f"{n}-vehicle platoon, adjoint polytope reach sets (symplectic, dt={dt})")
    fig.tight_layout()
    for ext in ["pdf", "svg"]:
        fig.savefig(HERE / f"platoon_grid.{ext}")
        print(f"saved {HERE}/platoon_grid.{ext}")

    # overview
    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(patches.Circle(obstacle_center, obstacle_radius, facecolor="salmon", label="obstacle"))
    for j in range(n):
        for k in range(0, N + 1, 10):
            draw_box(ax, ix[k][4 * j : 4 * j + 2], ec=colors[j])
        ax.plot(yy.ox[:, 4 * j], yy.ox[:, 4 * j + 1], color=colors[j], lw=0.8,
                label="leader" if j == 0 else None)
    ax.set_xlabel("$p_x$")
    ax.set_ylabel("$p_y$")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.set_title(f"{n}-vehicle platoon reach tube")
    fig2.tight_layout()
    for ext in ["pdf", "svg"]:
        fig2.savefig(HERE / f"platoon_overview.{ext}")
        print(f"saved {HERE}/platoon_overview.{ext}")
    plt.show()
