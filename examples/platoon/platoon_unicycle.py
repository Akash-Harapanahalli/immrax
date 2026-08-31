"""Unicycle platoon reachability, symplectic adjoint embedding.

Each vehicle is a unicycle with a speed state -- 4 states [px, py, theta, v],
2 inputs [a (thrust), omega]:

    px' = v cos(theta),  py' = v sin(theta),  theta' = omega,  v' = a.

The leader executes its own MPC nominal (converted to (a, omega)); followers
are fully decentralized (predecessor-only sensing, no broadcast). The
predecessor enters through raw relative position, its speed state, and a
heading-alignment term; own-heading trig dominates the nonlinearity:

    ex  = cos(thi) dx + sin(thi) dy - d,   ey = -sin(thi) dx + cos(thi) dy
    a_i = ka (v_pred - v_i) + k1 ex
    omega_i = k2 ey + k3 sin(th_pred - thi)

The heading term propagates turns down the chain immediately; cross-track
error alone integrates too slowly and followers drive through the obstacle.
The leader runs the same law against a virtual predecessor -- its own
precomputed nominal state trajectory, not a broadcast -- which caps the seed
width of the chain.

The leader nominal ships as ``us_unicycle.npy``; regenerate with
``python platoon_unicycle.py --regenerate`` (requires casadi + ipopt).

Outputs: platoon_unicycle_<method>_{grid,overview}.{pdf,svg} (for SHOW_AGENTS
vehicles), and a LaTeX runtime table on stdout.
"""

# ruff: noqa: E402  (backend/x64 flags must be set before jax.numpy is imported)
import argparse
import os
import pathlib
import shlex
import time
from typing import ClassVar

# Parsed before JAX is imported: the flags below must be set before the first
# array is created, or they are silently ignored.
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--regenerate", action="store_true", help="re-solve the leader MPC")
_ap.add_argument("--platform", choices=("auto", "gpu", "cpu"), default="auto", help="JAX backend")
_ap.add_argument("--no-show", action="store_true", help="save figures without opening a window")
_ap.add_argument("--precision", choices=("32", "64"), default="32", help="float width")
_ap.add_argument("--method", choices=("stacked", "adjoint", "interval"), default="stacked",
                 help="stacked frame [A;I] (default), plain adjoint, or interval (alpha frozen)")
# When imported rather than run, take flags from $PLATOON_ARGS.
args = (
    _ap.parse_args()
    if __name__ == "__main__"
    else _ap.parse_args(shlex.split(os.environ.get("PLATOON_ARGS", "")))
)

if args.platform != "auto":
    os.environ["JAX_PLATFORMS"] = args.platform

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", args.precision == "64")

import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as onp

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope, StackedAdjointEmbedding

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 12})

HERE = pathlib.Path(__file__).parent
US_FILE = HERE / "us_unicycle.npy"

# ---------------------------------------------------------------- parameters
t0, tf = 0.0, 3.0
du = 0.05
dtdu = 5
dt = du / dtdu
N = round((tf - t0) / dt)

ulim = 5.0
theta0 = float(onp.deg2rad(242.0))  # inside the collision cone (bearing 217, half-angle 27)
v_init = 2.0
follow_dist = 0.4
obstacle_center = (4, 4)
obstacle_radius = 2.25
obstacle_padding = 1.7  # modest extra berth for the followers' corner cut

# (n, pert scale) pairs for the sweep.
AGENTS = [(3, 1.0), (9, 1.0), (15, 1.0), (25, 1.0), (50, 1.0), (125, 1.0), (250, 1.0)]
SHOW_AGENTS = 9
# Uniform initial box on every state of every vehicle.
PERT_VEHICLE = jnp.array([5e-3, 5e-3, 5e-3, 5e-3])


# ------------------------------------------------ leader nominal (a, omega)
def x0_holonomic():
    return onp.array([8.0, 7.0, v_init * onp.cos(theta0), v_init * onp.sin(theta0)])


def make_nominal():
    """Solve the leader's obstacle-avoidance MPC (wide berth); returns us."""
    from casadi import MX, Function, Opti, tanh

    n_horizon = 20
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

    xk = x0_holonomic()
    us = []
    for _ in onp.arange(t0, tf, du):
        opti.set_value(xp, xk)
        for n in range(n_horizon + 1):
            opti.set_initial(xx[:, n], xk)
        sol = opti.solve()
        us.append(sol.value(uu[:, 0]))
        xk = onp.asarray(sol.value(xx[:, 1])).ravel()
    return onp.array(us)


def unicycle_nominal():
    """Convert the holonomic MPC nominal to (a, omega) on the dt grid."""
    us = onp.load(US_FILE)
    x = x0_holonomic()
    ao = onp.zeros((N, 2))
    for k in range(N):
        a = ulim * onp.tanh(us[int(k * dt / du)] / ulim)
        v2 = x[2] ** 2 + x[3] ** 2
        v = onp.sqrt(v2)
        ao[k] = [(x[2] * a[0] + x[3] * a[1]) / v, (x[2] * a[1] - x[3] * a[0]) / v2]
        x = x + dt * onp.array([x[2], x[3], a[0], a[1]])
    return ao


def nominal_state(ao):
    """Leader nominal state [px, py, th, v] on the step grid (the virtual
    predecessor the leader tracks)."""
    xn = onp.zeros((N + 1, 4))
    xn[0] = [8.0, 7.0, theta0, v_init]
    for k in range(N):
        p, th, v = xn[k, :2], xn[k, 2], xn[k, 3]
        xn[k + 1] = [p[0] + dt * v * onp.cos(th), p[1] + dt * v * onp.sin(th),
                     th + dt * ao[k, 1], v + dt * ao[k, 0]]
    return xn


# ------------------------------------------------------------------- system
class UnicyclePlatoon(irx.System):
    """Unicycles with speed state; decentralized predecessor-only followers."""

    num_agents: int = eqx.field(static=True)
    states_per_vehicle: ClassVar[int] = 4
    d: ClassVar[float] = follow_dist
    ka: ClassVar[float] = 0.35  # thrust from predecessor speed error
    k1: ClassVar[float] = 0.15  # thrust from along-track gap error
    k2: ClassVar[float] = 0.3  # omega from cross-track error
    k3: ClassVar[float] = 0.7  # omega from heading alignment

    def __init__(self, n: int):
        self.num_agents = n
        self.xlen = self.states_per_vehicle * n

    def f(self, t, x, u, w):
        # u = [a_nom, om_nom, pxn, pyn, thn, vn]: nominal inputs + nominal
        # state; the leader tracks its nominal as a virtual predecessor.
        # Vectorized over vehicles (predecessor == shifted slice), so the
        # traced graph is the same size for every n.
        sv = self.states_per_vehicle
        X = x.reshape(self.num_agents, sv)
        W = w.reshape(self.num_agents, 2)

        th0, v0 = X[0, 2], X[0, 3]
        dx0 = u[2] - X[0, 0]
        dy0 = u[3] - X[0, 1]
        ex0 = jnp.cos(th0) * dx0 + jnp.sin(th0) * dy0
        ey0 = -jnp.sin(th0) * dx0 + jnp.cos(th0) * dy0
        a0 = u[0] + self.ka * (u[5] - v0) + self.k1 * ex0 + W[0, 0]
        om0 = u[1] + self.k2 * ey0 + self.k3 * jnp.sin(u[4] - th0) + W[0, 1]
        leader = jnp.stack([v0 * jnp.cos(th0), v0 * jnp.sin(th0), om0, a0])

        pr, me, Wf = X[:-1], X[1:], W[1:]
        thp, vp = pr[:, 2], pr[:, 3]
        thi, vi = me[:, 2], me[:, 3]
        dx = pr[:, 0] - me[:, 0]
        dy = pr[:, 1] - me[:, 1]
        ex = jnp.cos(thi) * dx + jnp.sin(thi) * dy - self.d
        ey = -jnp.sin(thi) * dx + jnp.cos(thi) * dy
        ai = self.ka * (vp - vi) + self.k1 * ex + Wf[:, 0]
        omi = self.k2 * ey + self.k3 * jnp.sin(thp - thi) + Wf[:, 1]
        followers = jnp.stack(
            [vi * jnp.cos(thi), vi * jnp.sin(thi), omi, ai], axis=-1
        )
        return jnp.concatenate([leader, followers.reshape(-1)])


# ------------------------------------------------------------- reachability
def platoon_x0(n):
    e0 = onp.array([onp.cos(theta0), onp.sin(theta0)])
    p0 = onp.array([8.0, 7.0])
    x0 = [onp.concatenate([p0 - i * follow_dist * e0, [theta0, v_init]])
          for i in range(n)]
    return jnp.asarray(onp.concatenate(x0))


def mjac_permutation(n):
    # mjacM order (t, x, u, w): speeds, then positions, then headings (leader
    # first within each block), then control, then w. Ordering matters a lot:
    # velocities-first with headings LAST cuts the tail-vehicle extents ~7x vs
    # positions-first (headings-first diverges) -- the trig rows are evaluated
    # with the headings still centered for as long as possible.
    perm_v = tuple(4 + 4 * i for i in range(n))
    perm_p = tuple(1 + 4 * i + j for i in range(n) for j in range(2))
    perm_th = tuple(3 + 4 * i for i in range(n))
    perm_u = tuple(4 * n + 1 + j for j in range(6))
    off = 1 + 4 * n + 6
    perm_w = tuple(off + i for i in range(2 * n))
    return irx.Permutation((0,) + perm_v + perm_p + perm_th + perm_u + perm_w)


def reach(n, ao, xn, pert_scale=1.0, disable_adjoint=False, stacked=False):
    platoon = UnicyclePlatoon(n)
    x0 = platoon_x0(n)
    ix0 = irx.icentpert(x0, jnp.tile(PERT_VEHICLE * pert_scale, n))
    w_bounds = irx.icentpert(jnp.zeros(2 * n), jnp.zeros(2 * n))

    iu = lambda t, x: irx.interval(jnp.concatenate(
        [ao[jnp.asarray(t / dt).astype(int)], xn[jnp.asarray(t / dt).astype(int)]]))
    iw = lambda t, x: w_bounds

    if stacked:
        pt0 = Polytope.stacked_from_interval(ix0)
        emb = StackedAdjointEmbedding(platoon, 4 * n, permutation=mjac_permutation(n))
    else:
        pt0 = Polytope.from_interval(ix0)
        emb = AdjointEmbedding(platoon, jnp.eye(4 * n), jnp.zeros((0, 4 * n)),
                               permutation=mjac_permutation(n),
                               disable_adjoint=disable_adjoint)
    run = jax.jit(lambda p: emb.compute_reachset(t0, tf, p, (iu, iw), dt=dt))
    reps = 6 if n <= 6 else 2  # large runs: one timed repeat after compile
    # Not irx.utils.run_times: it holds the previous result while the next call
    # allocates, which doubles peak device memory (OOM at n=250).
    times, rs = [], None
    for _ in range(reps):
        rs = None  # free the previous reachset before allocating the next
        t = time.perf_counter()
        rs = jax.block_until_ready(run(pt0))
        times.append(time.perf_counter() - t)
    t_run = float(onp.mean(times[1:]))
    return rs, t_run, times[0] - t_run  # times[0] is compile + one run


def draw_box(ax, b2, **kw):
    lo, hi = onp.asarray(b2.lower), onp.asarray(b2.upper)
    if not (onp.isfinite(lo).all() and onp.isfinite(hi).all()):
        return
    pad = onp.maximum((hi - lo) / 2, 1e-4)
    c = (lo + hi) / 2
    irx.utils.draw_iarray(ax, irx.icentpert(jnp.asarray(c), jnp.asarray(pad)), **kw)


def box_at(rs, k):
    yy, (alpha_p, _N) = rs.ys
    K = yy.y.shape[1] // 2
    iy = irx.interval(-yy.y[k, :K], yy.y[k, K:])
    d = yy.alpha.shape[2]
    if yy.alpha.shape[1] == 2 * d:  # stacked frame: meet of the two blocks
        b1 = irx.interval(alpha_p[k]) @ iy[:d]
        b2 = iy[d:]
        return irx.interval(jnp.maximum(b1.lower, b2.lower),
                            jnp.minimum(b1.upper, b2.upper)) + yy.ox[k]
    return irx.interval(alpha_p[k]) @ iy + yy.ox[k]


def boxes(rs):
    return [box_at(rs, k) for k in range(rs.ys[0].ox.shape[0])]


def vehicle_vols(rs, n):
    """Per-vehicle 4-D box volume at tf, in m^2 . rad . m/s."""
    b = box_at(rs, -1)
    w = onp.asarray(b.upper, onp.float64) - onp.asarray(b.lower, onp.float64)
    return w.reshape(n, 4).prod(1)


# --------------------------------------------------------------------- table
HEADERS = ["N", "# States", "Runtime s (JIT s)", "Average vol", "Final vol"]
HEADERS_TEX = ["$N$", "\\# States", "Runtime s (JIT s)", "Average vol", "Final vol"]
# disable_numparse keeps the %.3e volume strings intact (tabulate would rewrite
# 1.200e-05 as 1.2e-05), but it also left-aligns every column, hence colalign.
ALIGN = ("right", "right", "left", "right", "right")


def table_row(n, t_run, t_jit, v_avg, v_fin):
    """(plain row, LaTeX row) for one sweep entry."""
    rt = f"{t_run:.3f} ({t_jit:.0f})"
    cells = [(f"{v:.3e}", f"{v:.3e}") if onp.isfinite(v) else ("inf", "$\\infty$")
             for v in (v_avg, v_fin)]
    return ([n, 4 * n, rt] + [c[0] for c in cells],
            [n, 4 * n, rt] + [c[1] for c in cells])


def print_table(rows, rows_tex, caption):
    print(f"\n{caption}")
    try:
        from tabulate import tabulate
    except ImportError:  # optional dependency; fall back to a hand-built table
        print(" & ".join(HEADERS_TEX) + " \\\\")
        for r in rows_tex:
            print(" & ".join(str(c) for c in r) + " \\\\")
    else:
        print(tabulate(rows, headers=HEADERS, tablefmt="simple",
                       disable_numparse=True, colalign=ALIGN))
        print()
        # latex_raw, not latex_booktabs: the headers carry math ($N$, \#) and
        # booktabs escapes it. latex_booktabs_raw does not exist -- tabulate
        # silently falls back to "simple" for an unknown format name.
        print(tabulate(rows_tex, headers=HEADERS_TEX, tablefmt="latex_raw",
                       disable_numparse=True, colalign=ALIGN))
    print()


# ------------------------------------------------------------------- figures
def _colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in onp.linspace(0, 1, n)]


def _frame(ax, x0s):
    ax.set_xlim(0, x0s[-1, 0] + 0.5)
    ax.set_ylim(-2, x0s[-1, 1] + 0.5)


def _save(fig, stem):
    for ext in ["pdf", "svg"]:
        fig.savefig(HERE / f"{stem}.{ext}")
        print(f"saved {HERE}/{stem}.{ext}")


def grid_figure(rs, n, stem, ncols=3, every=5):
    yy = rs.ys[0]
    ix = boxes(rs)
    colors = _colors(n)
    x0s = onp.asarray(platoon_x0(n)).reshape(-1, 4)[:, :2]
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = onp.atleast_1d(axs).reshape(-1)
    for j in range(n):
        ax = axs[j]
        ax.add_patch(patches.Circle(obstacle_center, obstacle_radius, facecolor="salmon"))
        for k in range(0, N + 1, every):
            draw_box(ax, ix[k][4 * j : 4 * j + 2], ec=colors[j])
        ax.plot(yy.ox[:, 4 * j], yy.ox[:, 4 * j + 1], color=colors[j], alpha=0.25)
        others = onp.ones(n, bool)
        others[j] = False
        ax.scatter(x0s[others, 0], x0s[others, 1], color="k", s=5, alpha=0.1)
        ax.scatter(x0s[j, 0], x0s[j, 1], color=colors[j], s=5)
        _frame(ax, x0s)
    for j in range(n, len(axs)):
        axs[j].set_axis_off()
    fig.tight_layout()
    _save(fig, stem)
    return fig


def overview_figure(rs, n, stem, every=10):
    yy = rs.ys[0]
    ix = boxes(rs)
    colors = _colors(n)
    x0s = onp.asarray(platoon_x0(n)).reshape(-1, 4)[:, :2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(patches.Circle(obstacle_center, obstacle_radius,
                                facecolor="salmon", label="obstacle"))
    for j in range(n):
        for k in range(0, N + 1, every):
            draw_box(ax, ix[k][4 * j : 4 * j + 2], ec=colors[j])
        ax.plot(yy.ox[:, 4 * j], yy.ox[:, 4 * j + 1], color=colors[j], lw=0.8,
                label="leader" if j == 0 else None)
    ax.set_xlabel("$p_x$")
    ax.set_ylabel("$p_y$")
    ax.set_aspect("equal")
    _frame(ax, x0s)
    ax.legend()
    fig.tight_layout()
    _save(fig, stem)
    return fig


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    dev = jax.devices()[0]
    print(f"backend: {jax.default_backend()} ({dev.device_kind}), x64={jax.config.jax_enable_x64}")
    if args.regenerate or not US_FILE.exists():
        onp.save(US_FILE, make_nominal())
        print(f"saved {US_FILE}")
    ao_np = unicycle_nominal()
    ao = jnp.asarray(ao_np)
    xn = jnp.asarray(nominal_state(ao_np))

    results, rows, rows_tex = {}, [], []
    for n, ps in AGENTS:
        rs, t_run, t_jit = reach(n, ao, xn, ps, stacked=args.method == "stacked",
                                 disable_adjoint=args.method == "interval")
        vols = vehicle_vols(rs, n)
        v_avg, v_fin = float(onp.mean(vols)), float(vols[-1])
        # Only SHOW_AGENTS is plotted; holding the rest would pin one
        # (steps, 4n, 4n) adjoint per sweep entry in device memory.
        results[n] = rs if n == SHOW_AGENTS else None
        del rs
        r, r_tex = table_row(n, t_run, t_jit, v_avg, v_fin)
        rows.append(r)
        rows_tex.append(r_tex)
        print(f"  n={n}: {t_run:.3f} s (JIT {t_jit:.0f} s)  "
              f"avg vol {v_avg:.3e}  final vol {v_fin:.3e}")

    print_table(rows, rows_tex, (
        f"% unicycle platoon {args.method} reachability (symplectic, dt={dt}, tf={tf}, "
        f"pert {float(PERT_VEHICLE[0]):g}, fp{args.precision} on "
        f"{jax.default_backend()}: {dev.device_kind})"))

    n = SHOW_AGENTS
    grid_figure(results[n], n, f"platoon_unicycle_{args.method}_grid")
    overview_figure(results[n], n, f"platoon_unicycle_{args.method}_overview")
    if not args.no_show:
        plt.show()
