"""Nonholonomic (Dubins-like) platoon reachability, symplectic adjoint embedding.

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

The heading term propagates turns down the chain immediately (cross-track
error alone integrates too slowly -- followers drive straight through the
obstacle without it). The LEADER runs the same law against a virtual
predecessor: its own precomputed nominal state trajectory (not a broadcast --
it is the leader's own mission plan); this caps the seed width of the whole
chain (an open-loop leader's position tube grows to ~0.16 and poisons every
follower's drift bound). Gains are soft because the certificate's drift pays
cond(alpha) x trig curvature and cond(alpha) grows with the loop contraction
rate (stiff gains -> cond ~1e4, certificate dies mid-horizon).

Certified depth: the mean-value wrap is width-dependent and grows with the
loop gains; with the soft gains below, n = 12 (48 states) certifies to tf
with the FULL 1e-2 box on every state. Deeper chains need a smaller initial
box (the wrap is subcritical below a width threshold) -- re-validate any
such run against the corrected mjacM pairing (2026-08 fix).

String stability is the price of decentralization: under acceleration each
link lags a/ka in speed and the chain stretches, so followers cut inside the
leader's path. The scenario sits at the feasibility frontier of that
trade-off: the initial heading (238 deg) is inside the obstacle's collision
cone (bearing 217, half-angle 27 -- a straight-line platoon would hit it),
and the leader's modest extra berth (padding 1.7) absorbs the followers'
corner cut, leaving min center clearance 2.29 > 2.25.

The leader nominal ships as ``us_dubins.npy``; regenerate with
``python platoon_dubins.py --regenerate`` (requires casadi + ipopt).

Outputs: platoon_dubins_grid.{pdf,svg}, platoon_dubins_overview.{pdf,svg}
(for SHOW_AGENTS vehicles), and a LaTeX runtime table on stdout.
"""

# ruff: noqa: E402  (x64 flag must be set before jax.numpy is imported)
import argparse
import pathlib
from typing import ClassVar

import equinox as eqx
import jax

# The adjoint of the contracting follower chain is exponentially conditioned;
# float32 destroys the certificate. Run in x64.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as onp

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})  # Computer Modern

HERE = pathlib.Path(__file__).parent
US_FILE = HERE / "us_dubins.npy"

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

# AGENTS = [3, 6, 9, 12, 15, 18, 21]
# (n, pert scale) pairs for the sweep.
AGENTS = [(6, 1.0), (12, 1.0)]
SHOW_AGENTS = 6
# Full initial box on every state of every vehicle. (An earlier version
# perturbed only px and theta -- a leftover of the holonomic example's
# px/vx pattern, not a deliberate choice.)
PERT_VEHICLE = jnp.array([1e-2, 1e-2, 1e-2, 1e-2])


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
class DubinsPlatoon(irx.System):
    """Unicycles with speed state; decentralized predecessor-only followers."""

    num_agents: int = eqx.field(static=True)
    states_per_vehicle: ClassVar[int] = 4
    d: ClassVar[float] = follow_dist
    # Soft gains: the certified drift pays cond(alpha) x trig curvature, and
    # cond(alpha) grows with EVERY loop contraction rate -- stiffer gains (and
    # even pure linear damping, e.g. a time-headway term) shorten the
    # certificate's life. These values certify n = 12 with the full 1e-2 box;
    # k3 = 0.8 already sits on a cliff (extents x75). k3 carries the turn.
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
        sv = self.states_per_vehicle
        th0, v0 = x[2], x[3]
        dx0 = u[2] - x[0]
        dy0 = u[3] - x[1]
        ex0 = jnp.cos(th0) * dx0 + jnp.sin(th0) * dy0
        ey0 = -jnp.sin(th0) * dx0 + jnp.cos(th0) * dy0
        a0 = u[0] + self.ka * (u[5] - v0) + self.k1 * ex0 + w[0]
        om0 = u[1] + self.k2 * ey0 + self.k3 * jnp.sin(u[4] - th0) + w[1]
        dyn = [jnp.array([v0 * jnp.cos(th0), v0 * jnp.sin(th0), om0, a0])]
        for i in range(1, self.num_agents):
            pr = x[(i - 1) * sv : i * sv]
            me = x[i * sv : (i + 1) * sv]
            thp, thi, vi, vp = pr[2], me[2], me[3], pr[3]
            dx = pr[0] - me[0]
            dy = pr[1] - me[1]
            ex = jnp.cos(thi) * dx + jnp.sin(thi) * dy - self.d
            ey = -jnp.sin(thi) * dx + jnp.cos(thi) * dy
            ai = self.ka * (vp - vi) + self.k1 * ex + w[2 * i]
            omi = self.k2 * ey + self.k3 * jnp.sin(thp - thi) + w[2 * i + 1]
            dyn.append(jnp.array([vi * jnp.cos(thi), vi * jnp.sin(thi), omi, ai]))
        return jnp.concatenate(dyn)


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


def reach(n, ao, xn, pert_scale=1.0):
    platoon = DubinsPlatoon(n)
    x0 = platoon_x0(n)
    pt0 = Polytope.from_interval(irx.icentpert(x0, jnp.tile(PERT_VEHICLE * pert_scale, n)))
    w_bounds = irx.icentpert(jnp.zeros(2 * n), jnp.zeros(2 * n))

    iu = lambda t, x: irx.interval(jnp.concatenate(
        [ao[jnp.asarray(t / dt).astype(int)], xn[jnp.asarray(t / dt).astype(int)]]))
    iw = lambda t, x: w_bounds

    emb = AdjointEmbedding(platoon, jnp.eye(4 * n), jnp.zeros((0, 4 * n)),
                           permutation=mjac_permutation(n))
    run = jax.jit(lambda p: emb.compute_reachset(t0, tf, p, (iu, iw), dt=dt))
    reps = 6 if n <= 6 else 2  # large runs: one timed repeat after compile
    rs, times = irx.utils.run_times(reps, run, pt0)
    return rs, float(jnp.mean(times[1:]))


def draw_box(ax, b2, **kw):
    lo, hi = onp.asarray(b2.lower), onp.asarray(b2.upper)
    if not (onp.isfinite(lo).all() and onp.isfinite(hi).all()):
        return
    pad = onp.maximum((hi - lo) / 2, 1e-4)
    c = (lo + hi) / 2
    irx.utils.draw_iarray(ax, irx.icentpert(jnp.asarray(c), jnp.asarray(pad)), **kw)


def boxes(rs, n):
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
        onp.save(US_FILE, make_nominal())
        print(f"saved {US_FILE}")
    ao_np = unicycle_nominal()
    ao = jnp.asarray(ao_np)
    xn = jnp.asarray(nominal_state(ao_np))

    results = {}
    for n, ps in AGENTS:
        rs, t_run = reach(n, ao, xn, ps)
        results[n] = (rs, t_run, ps)
        print(f"  n={n} (pert x{ps:g}): {t_run * 1e3:.1f} ms")

    print(f"\n% Dubins platoon adjoint reachability (symplectic, dt={dt}, tf={tf})")
    print("\\begin{tabular}{rrrr}")
    print("\\toprule")
    print("Vehicles & States & Init.\\ pert & Runtime (ms) \\\\")
    print("\\midrule")
    for n, ps in AGENTS:
        print(f"{n} & {4 * n} & {1e-2 * ps:g} & {results[n][1] * 1e3:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")

    # ------------------------------------------------------------- figures
    n = SHOW_AGENTS
    rs, _, _ = results[n]
    yy = rs.ys[0]
    ix = boxes(rs, n)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in onp.linspace(0, 1, n)]
    x0s = onp.asarray(platoon_x0(n)).reshape(-1, 4)[:, :2]

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
    fig.suptitle(f"{n}-vehicle Dubins platoon, adjoint polytope reach sets (symplectic, dt={dt})")
    fig.tight_layout()
    for ext in ["pdf", "svg"]:
        fig.savefig(HERE / f"platoon_dubins_grid.{ext}")
        print(f"saved {HERE}/platoon_dubins_grid.{ext}")

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
    ax.set_title(f"{n}-vehicle Dubins platoon reach tube")
    fig2.tight_layout()
    for ext in ["pdf", "svg"]:
        fig2.savefig(HERE / f"platoon_dubins_overview.{ext}")
        print(f"saved {HERE}/platoon_dubins_overview.{ext}")
    plt.show()
