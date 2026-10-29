"""Robot arm reachability from the box x0 +- pert, five certified methods.

Benchmark: 2-link robot arm under PD control (parameters from
https://bitbucket.org/hurricanesoff/emsoft-code/src), 4 states
(q1, q2, q1dot, q2dot), initial box x0 +- pert around
x0 = (1.505, 1.505, 0.005, 0.005), dt = 0.01, tf = 10; ``--pert`` sets the
half-width (default 0.03; at 0.05 the plain L2 adjoint blows up and only
ReachiLQR reaches tf). The closed loop
contracts to the setpoint (2, 1, 0, 0), so a sound, tight method must track
the transient and stabilize there.

The ellipsoid methods (L2) start from the circumscribed ellipsoid of the
box, so the reference "true set" is the flow of that ellipsoid (lighter
gray), with the flow of the box inside it (darker gray) as the reference
for the box-initialized methods (interval, polytope).

Methods:
  1. Interval           -- Polytope + AdjointEmbedding, adjoint disabled
  2. Polytope adjoint   -- Polytope + AdjointEmbedding
  3. L2 log-norm        -- L2Normotope, mjacM log-norm, alpha frozen (no adjoint)
  4. L2 adjoint         -- L2Normotope, mjacM log-norm + adjoint
  5. L2 + ReachiLQR     -- method 4 plus ReachiLQR iterations on the
                           hypercontrol U (running + terminal log-volume cost)

All methods are run with each symplectic integrator: Euler (the
compute_reachset default), Heun and RK4. The MC truth for each uses the same
explicit tableau. Produces robot_arm_pert<pert>{,_heun,_rk4}.{pdf,svg}
(snapshots on MC truth, full view + zoom at the setpoint),
robot_arm_bloat_pert<pert>{,_heun,_rk4}.{pdf,svg} (percent bloat over time),
robot_arm_ilqr_pert<pert>{,_heun,_rk4}.{pdf,svg} (ReachiLQR volume ratio at the
tube's last step and cumulative runtime per iteration),
and a LaTeX table (method, runtime per integrator,
symplectic Euler 4-D volume and bloat) on stdout.
"""

import argparse
import pathlib
import shutil
from time import perf_counter
from typing import ClassVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from immutabledict import immutabledict
from scipy.linalg import sqrtm
from scipy.spatial import ConvexHull

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope, ReachiLQR
from immrax.parametric.symplectic_rk import SOLVERS, get_tableau

plt.rcParams.update(
    {"text.usetex": shutil.which("latex") is not None, "font.family": "serif"}
)  # Computer Modern


# ---------------------------------------------------------------- benchmark
class RobotArm(irx.System):
    """PD-controlled 2-link arm; m=1, l=3, kp=(2,1), kd=(2,1), setpoint (2,1)."""

    xlen: ClassVar[int] = 4

    def f(self, t, x):
        x1, x2, x3, x4 = x
        den = x2**2 + 1.0
        return jnp.array(
            [
                x3,
                x4,
                (-2.0 * x2 * x3 * x4 - 2.0 * x1 - 2.0 * x3 + 4.0) / den,
                x2 * x3**2 - x2 - x4 + 1.0,
            ]
        )


sys = RobotArm()
# Initial box x0 +- pert (EMSOFT uses ||P^1/2 (x - x0)|| <= 0.01, per-axis
# extents ~0.007; the 0.03 box bounds its level-0.041 ellipsoid). Its minimal
# circumscribed ellipsoid (semi-axes sqrt(n) pert, through the corners) is the
# ball ||x - x0|| <= 2 pert -- L2Normotope.from_interval.
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--pert", type=float, default=0.03, help="initial box half-width")
PERT = _ap.parse_args().pert
TAG = f"_pert{PERT:g}"  # figure name tag
x0 = jnp.array([1.505, 1.505, 0.005, 0.005])
pert = PERT * jnp.ones(4)
ix0 = irx.icentpert(x0, pert)
dt, tf = 0.01, 10.0
N = int(round(tf / dt))
snap_ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
snap_ks = [int(round(t / dt)) for t in snap_ts]
ZOOM_TMIN = 5.0  # the zoom panel shows the snapshots from this time on
ZOOM_DT = 0.25  # ... sampled this finely; the tail is where the methods separate
# A snapshot up to LABEL_TMAX is still big enough on the full view to carry its
# t label inside the set; past it the sets are dots at the setpoint, so their
# labels are fanned out around it on a connector.
LABEL_TMAX = 5.0
LABEL_ARC = (30.0, 100.0)  # degrees, fan for the off-to-the-side labels
LABEL_R = (0.20, 0.30)  # fan radii, as fractions of the (x, y) axis span
WMAX = 1.0  # a set wider than this is treated as blown up
perm = irx.Permutation((0, 3, 4, 1, 2))  # velocities first (EMSOFT notebook)
# (solver, table column, figure suffix)
INTEGRATORS = [
    ("symplectic", "Symplectic Euler", ""),
    ("symplectic_heun", "Symplectic Heun", "_heun"),
    ("symplectic_rk4", "Symplectic RK4", "_rk4"),
]


def timings(f, *args):
    """(steady-state runtime, JIT time) in seconds; run 0 carries the compile."""
    _, ts = irx.utils.run_times(4, f, *args)
    t_run = float(jnp.mean(ts[1:]))
    return t_run, float(ts[0]) - t_run


# ---------------------------------------------------------------- MC truth
# Two nested initial sets: the box (interval / polytope methods) and its
# circumscribed ball (L2 methods). Boundary samples flow to the true set's
# boundary (drawn in the figures). True 4-D volumes use interior samples:
# vol phi(X0) = vol X0 * E_x |det D phi(x)|, with D phi accumulated as the
# product of step Jacobians. Each integrator's truth uses its own explicit
# tableau (make_step).
key = jax.random.PRNGKey(0)
M_MC = 8192
u = jax.random.normal(key, (M_MC, 4))
u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
ell_bnd = x0 + 2.0 * pert * u  # semi-axes sqrt(4) pert, through the corners

kf_, ks_ = jax.random.split(key)
face = jax.random.randint(kf_, (M_MC,), 0, 4)
sgn = jnp.where(jax.random.bernoulli(ks_, shape=(M_MC,)), 1.0, -1.0)
w = jax.random.uniform(key, (M_MC, 4), minval=-1.0, maxval=1.0)
w = w.at[jnp.arange(M_MC), face].set(sgn)
box_bnd = x0 + pert * w

ki_, kr_ = jax.random.split(ks_)
box_int = x0 + pert * jax.random.uniform(ki_, (M_MC, 4), minval=-1.0, maxval=1.0)
r = jax.random.uniform(kr_, (M_MC, 1)) ** 0.25  # uniform in the unit 4-ball
ell_int = x0 + 2.0 * pert * r * u
VOL0 = {
    "box": float(onp.prod(2.0 * onp.asarray(pert))),
    "ell": onp.pi**2 / 2 * float(2.0 * pert[0]) ** 4,
}


def make_step(system, tableau):
    A, b, _ = get_tableau(tableau)

    def step(c):
        ks = []
        for i in range(len(b)):
            xi = c
            for j in range(i):
                if A[i, j] != 0.0:
                    xi = xi + dt * A[i, j] * ks[j]
            ks.append(system.f(0.0, xi))
        return c + dt * sum(b[i] * ks[i] for i in range(len(b)))

    return step


def make_roll(system, tableau):
    step = make_step(system, tableau)

    @jax.jit
    def roll(x):
        _, xs = jax.lax.scan(lambda c, _: (step(c),) * 2, x, None, length=N)
        return jnp.vstack([x, xs])

    return roll


def true_vols(system, tableau, xs, vol0):
    """True volume of the flowed set at every step, (N + 1,)."""
    step = make_step(system, tableau)

    def logdets(x):
        def body(c, _):
            return step(c), jnp.linalg.slogdet(jax.jacfwd(step)(c))[1]

        _, ld = jax.lax.scan(body, x, None, length=N)
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(ld)])

    ld = jax.jit(jax.vmap(logdets))(xs)
    return vol0 * onp.asarray(jnp.mean(jnp.exp(ld), axis=0))


# Initial sets and embeddings, shared by all integrators.
pt0 = Polytope.from_interval(ix0)
emb_poly = AdjointEmbedding(
    sys, jnp.linalg.pinv(pt0.alpha), jnp.zeros((0, 4)), permutation=perm
)
emb_box = AdjointEmbedding(  # alpha frozen at I: the interval embedding
    sys,
    jnp.linalg.pinv(pt0.alpha),
    jnp.zeros((0, 4)),
    permutation=perm,
    disable_adjoint=True,
)
nt0 = irx.L2Normotope.from_interval(ix0)  # minimal circumscribed ellipsoid
emb_l2 = irx.NormotopeEmbedding(sys)
emb_l2._initialize(nt0)  # resolve gsc outside jit
kw_noadj = immutabledict({"adjoint": False, "perm": perm})
kw_adj = immutabledict({"perm": perm})
ILQR_ITERS = 50
ILQR_W = 2.0  # running log-volume weight (per unit time)
T_AREA = 2.0  # volumes reported at this time (mid-transient)
kf = int(round(T_AREA / dt))


def logvol(pt):
    M = pt.alpha.T @ pt.alpha / pt.y**2
    return -jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(M))))


def run(solver, suffix):
    """All certified methods with one integrator: saves its figures and returns
    the table rows (plain name, LaTeX name, runtime, volume, bloat) for it."""
    mc_ell = onp.asarray(jax.vmap(make_roll(sys, SOLVERS[solver]))(ell_bnd))
    mc_box = onp.asarray(jax.vmap(make_roll(sys, SOLVERS[solver]))(box_bnd))
    truth_t = {
        "box": true_vols(sys, SOLVERS[solver], box_int, VOL0["box"]),
        "ell": true_vols(sys, SOLVERS[solver], ell_int, VOL0["ell"]),
    }

    # ------------------------------------------------------------ 1. interval
    t_box = timings(
        lambda p: emb_box.compute_reachset(0.0, tf, p, dt=dt, solver=solver).ys[0].y,
        pt0,
    )
    rs_box = emb_box.compute_reachset(0.0, tf, pt0, dt=dt, solver=solver)
    # alpha = I, so the box is [ox - ly, ox + uy]; stored as (lower, upper).
    oy, ox_ = onp.asarray(rs_box.ys[0].y), onp.asarray(rs_box.ys[0].ox)
    box_ys = onp.concatenate([ox_ - oy[:, :4], ox_ + oy[:, 4:]], axis=1)

    # ---------------------------------------------------------------- 2. polytope
    t_poly = timings(
        lambda p: emb_poly.compute_reachset(0.0, tf, p, dt=dt, solver=solver).ys[0].y,
        pt0,
    )
    rs_poly = emb_poly.compute_reachset(0.0, tf, pt0, dt=dt, solver=solver)

    # ------------------------------------------------------- 3/4. L2 normotopes
    def reach_l2(p, kw):
        return emb_l2.compute_reachset(0.0, tf, p, dt=dt, solver=solver, f_kwargs=kw)

    t_l2n = timings(lambda p: reach_l2(p, kw_noadj).ys[0].y, nt0)
    t_l2a = timings(lambda p: reach_l2(p, kw_adj).ys[0].y, nt0)
    rs_l2n = reach_l2(nt0, kw_noadj)
    rs_l2a = reach_l2(nt0, kw_adj)

    # ------------------------------------------------------ 5. L2 + ReachiLQR
    # Method 4 plus ReachiLQR on the hypercontrol U (same integrator, same
    # perm), minimizing log-volume along the whole tube (running cost) plus at
    # tf. The terminal cost alone barely separates from the plain adjoint here;
    # the running cost is what drives the drastic tightening. The certified tube
    # is the best iterate's perturbed rollout. Diminishing returns past ~60
    # iterations; R=5 or 100+ iterations start to oscillate.
    ilqr = ReachiLQR(
        embedding=emb_l2,
        pt0=nt0,
        terminal_cost=logvol,
        running_cost=lambda t, pt, U: ILQR_W * logvol(pt),
        t0=0.0,
        tf=tf,
        N=N,
        dt=dt,
        f_kwargs=kw_adj,
        solver=solver,
    )
    _t0 = perf_counter()
    ilqr.setup()
    _tjit = perf_counter() - _t0
    # Per iterate: whether its tube reaches tf, its volume over the true volume
    # at its last valid step, and the cumulative solver time with this
    # host-side bookkeeping excluded.
    it_log, skip = [], [0.0]

    def record(i, R, res):
        t_cb = perf_counter()
        k = int(res.ifinal)
        nt = ilqr._unflatten(res.pert_traj[k])[0]
        v = onp.pi**2 / 2 * float(nt.y) ** 4 / abs(float(jnp.linalg.det(nt.alpha)))
        it_log.append((k == N, v / truth_t["ell"][k], t_cb - _t0 - skip[0]))
        skip[0] += perf_counter() - t_cb

    _t0 = perf_counter()
    rr_ilqr = ilqr.run(R_schedule=20.0, iters=ILQR_ITERS, on_iterate=record)
    # One timed run of the full iLQR loop (not the repeated-run mean of timings()).
    t_ilqr = (perf_counter() - _t0 - skip[0], _tjit)
    if int(rr_ilqr.best.ifinal) < N:  # the table reports its blow-up time
        print(f"warning: {solver} ReachiLQR best iterate did not reach tf")

    # ------------------------------------------------------------- set accessors
    def poly_at(k):
        s = rs_poly.ys[0]
        return Polytope(s.ox[k], s.alpha[k], s.y[k])

    def l2_at(rs, k):
        s = rs.ys[0]
        return irx.L2Normotope(s.ox[k], s.alpha[k], s.y[k])

    def ilqr_at(k):
        return ilqr._unflatten(rr_ilqr.best.pert_traj[k])[0]

    def box_iover_at(k):
        return irx.interval(jnp.asarray(box_ys[k, :4]), jnp.asarray(box_ys[k, 4:]))

    def poly_iover_at(k):
        p = poly_at(k)
        return irx.interval(jnp.linalg.inv(p.alpha)) @ p.hinv(p.y) + p.ox

    def l2_proj(nt):
        """Center and 2x2 shape matrix of the (q1, q2) projection ellipse."""
        Ez = float(nt.y) ** 2 * onp.linalg.inv(onp.asarray(nt.alpha.T @ nt.alpha))
        return onp.asarray(nt.ox[:2]), Ez[:2, :2]

    def poly_vol(p):  # parallelotope: widths in alpha coordinates / |det alpha|
        y = onp.asarray(p.y)
        return float(onp.prod(y[:4] + y[4:]) / abs(onp.linalg.det(p.alpha)))

    def l2_vol(nt):  # unit 4-ball volume pi^2 / 2
        return float(onp.pi**2 / 2 * nt.y**4 / abs(onp.linalg.det(nt.alpha)))

    def ellipse_boundary(c, E2, npts=200):
        th = onp.linspace(0, 2 * onp.pi, npts)
        B = onp.real(sqrtm(E2 + 1e-30 * onp.eye(2)))
        return (c[:, None] + B @ onp.stack([onp.cos(th), onp.sin(th)])).T

    def alive(io):
        """Set with interval hull io is finite and not blown up."""
        lo, hi = onp.asarray(io.lower), onp.asarray(io.upper)
        return bool(
            onp.isfinite(lo).all() and onp.isfinite(hi).all() and (hi - lo < WMAX).all()
        )

    def death_time(iover_at):
        for k in range(N + 1):
            if not alive(iover_at(k)):
                return k * dt
        return None

    # --------------------------------------------------------------------- table
    # Each method's bloat is measured against the MC truth for ITS OWN initial
    # set: box flow for interval/polytope, ellipsoid flow for L2.
    # (plain name, LaTeX name, (runtime, jit), iover_at, vol_at, truth key)
    methods = [
        (
            "Interval (adjoint off)",
            "Interval (adjoint off)",
            t_box,
            box_iover_at,
            lambda k: float(onp.prod(box_ys[k, 4:] - box_ys[k, :4])),
            "box",
        ),
        (
            "Polytope + adjoint",
            "Polytope + adjoint",
            t_poly,
            poly_iover_at,
            lambda k: poly_vol(poly_at(k)),
            "box",
        ),
        (
            "L2 normotope (log-norm, no adjoint)",
            "L2 normotope (log-norm, no adjoint)",
            t_l2n,
            lambda k: l2_at(rs_l2n, k).iover(),
            lambda k: l2_vol(l2_at(rs_l2n, k)),
            "ell",
        ),
        (
            "L2 normotope (log-norm + adjoint)",
            "L2 normotope (log-norm + adjoint)",
            t_l2a,
            lambda k: l2_at(rs_l2a, k).iover(),
            lambda k: l2_vol(l2_at(rs_l2a, k)),
            "ell",
        ),
        (
            f"L2 + adjoint + ReachiLQR ({ILQR_ITERS} iters)",
            f"L2 + adjoint + ReachiLQR ({ILQR_ITERS} iters)",
            t_ilqr,
            lambda k: ilqr_at(k).iover(),
            lambda k: l2_vol(ilqr_at(k)),
            "ell",
        ),
    ]

    rows = []
    for name, name_tex, rt, iover_at, vol_at, ref in methods:
        td = death_time(iover_at)
        if td is not None and td <= T_AREA:
            a_s, b_s = f"inf (blows up t ~ {td:.2f})", "--"
            a_tex, b_tex = f"$\\infty$ (blows up $t \\approx {td:.2f}$)", "--"
        else:
            a = vol_at(kf)
            bloat = 100.0 * (a / truth_t[ref][kf] - 1.0)
            a_s = a_tex = f"{a:.3e}"
            b_s, b_tex = f"{bloat:+.1f}%", f"{bloat:+.1f}\\%"
        rows.append((name, name_tex, rt, (a_s, a_tex), (b_s, b_tex)))

    # ---------------------------------------------------------------------- plot
    XLIM0, YLIM0 = (1.4, 2.2), (0.9, 1.6)
    xspan, yspan = XLIM0[1] - XLIM0[0], YLIM0[1] - YLIM0[0]
    eq = onp.array([2.0, 1.0])  # closed-loop setpoint
    zoom_ts = [
        ZOOM_TMIN + i * ZOOM_DT
        for i in range(int(round((tf - ZOOM_TMIN) / ZOOM_DT)) + 1)
    ]
    zoom_ks = [int(round(t / dt)) for t in zoom_ts]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, ks in [(axs[0], snap_ks), (axs[1], zoom_ks)]:
        for k in ks:
            hb = mc_box[ConvexHull(mc_box[:, k, :2]).vertices, k, :2]
            he = mc_ell[ConvexHull(mc_ell[:, k, :2]).vertices, k, :2]
            ax.fill(he[:, 0], he[:, 1], color="0.85", zorder=0)
            ax.fill(hb[:, 0], hb[:, 1], color="0.68", zorder=0.5)
    axs[0].fill([], [], color="0.85", label="true set (ellipsoid init)")
    axs[0].fill([], [], color="0.68", label="true set (box init $\\subset$ ellipsoid)")

    lbl_kw = dict(
        color="0.25",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
    )
    far = []
    for k in snap_ks:
        cen = mc_ell[:, k, :2].mean(0)
        if k * dt <= LABEL_TMAX:
            # above the fan's connectors, which run out through this cluster
            axs[0].annotate(f"t={k * dt:g}", cen, zorder=7, **lbl_kw)
        else:
            far.append((k, cen))
    # the late snapshots crowd into the setpoint: fan their labels around it
    for i, (k, cen) in enumerate(far):
        th = onp.deg2rad(
            LABEL_ARC[0] + (LABEL_ARC[1] - LABEL_ARC[0]) * i / max(len(far) - 1, 1)
        )
        tip = eq + onp.array(LABEL_R) * onp.array(
            [xspan * onp.cos(th), yspan * onp.sin(th)]
        )
        axs[0].annotate(
            f"t={k * dt:g}",
            cen,
            xytext=tip,
            zorder=6,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, shrinkA=1, shrinkB=1),
            **lbl_kw,
        )

    for ax, ks in [(axs[0], snap_ks), (axs[1], zoom_ks)]:
        for k in ks:
            if alive(box_iover_at(k)):
                lo, hi = box_ys[k, :2], box_ys[k, 4:6]
                ax.plot(
                    [lo[0], hi[0], hi[0], lo[0], lo[0]],
                    [lo[1], lo[1], hi[1], hi[1], lo[1]],
                    color="tab:purple",
                    lw=0.9,
                    label="interval" if k == 0 and ax is axs[0] else None,
                )
        for k in ks:
            if not alive(poly_iover_at(k)):
                continue
            V = onp.asarray(poly_at(k).get_vertices())[:, :2]
            V = V[ConvexHull(V).vertices]
            V = onp.vstack([V, V[0]])
            ax.plot(
                V[:, 0],
                V[:, 1],
                color="tab:blue",
                lw=0.9,
                label="polytope + adjoint" if k == 0 and ax is axs[0] else None,
            )
        for nt_at, color, al, lab in [
            (lambda k: l2_at(rs_l2n, k), "tab:olive", 0.45, "L2 log-norm (no adjoint)"),
            (lambda k: l2_at(rs_l2a, k), "tab:green", 1.0, "L2 log-norm + adjoint"),
            (ilqr_at, "tab:red", 1.0, f"L2 + ReachiLQR ({ILQR_ITERS} iters)"),
        ]:
            for k in ks:
                nt = nt_at(k)
                if not alive(nt.iover()):
                    continue
                B = ellipse_boundary(*l2_proj(nt))
                ax.plot(
                    B[:, 0],
                    B[:, 1],
                    color=color,
                    lw=0.9,
                    alpha=al,
                    label=lab if k == 0 and ax is axs[0] else None,
                )

    axs[0].set_xlim(*XLIM0)
    axs[0].set_ylim(*YLIM0)
    # Zoom window: fit the alive adjoint certificates at the zoomed snapshots
    # (the diverging no-adjoint rings would dwarf everything; let them clip).
    zlo, zhi = [], []
    for k in zoom_ks:
        for io in [
            box_iover_at(k),
            poly_iover_at(k),
            l2_at(rs_l2a, k).iover(),
            ilqr_at(k).iover(),
        ]:
            if alive(io):
                zlo.append(onp.asarray(io.lower)[:2])
                zhi.append(onp.asarray(io.upper)[:2])
    if not zlo:  # every certificate is dead by the zoom window: fit the truth
        pts = mc_ell[:, zoom_ks, :2].reshape(-1, 2)
        zlo, zhi = [pts.min(0)], [pts.max(0)]
    zlo, zhi = onp.min(zlo, axis=0), onp.max(zhi, axis=0)
    zpad = 0.08 * (zhi - zlo).max()
    axs[1].set_xlim(zlo[0] - zpad, zhi[0] + zpad)
    axs[1].set_ylim(zlo[1] - zpad, zhi[1] + zpad)
    for ax in axs:
        ax.set_xlabel("$q_1$")
        ax.set_ylabel("$q_2$")
    axs[0].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ["pdf", "svg"]:
        out = pathlib.Path(__file__).parent / f"robot_arm{TAG}{suffix}.{ext}"
        fig.savefig(out)
        print(f"saved {out}")

    # ------------------------------------------------------------ bloat over time
    # Percent bloat = 100 (volume / own-init true volume - 1), while alive.
    plot_ks = list(range(0, N + 1, 8))
    styles = {
        "Interval (adjoint off)": ("tab:purple", "interval"),
        "Polytope + adjoint": ("tab:blue", "polytope + adjoint"),
        "L2 normotope (log-norm, no adjoint)": (
            "tab:olive",
            "L2 log-norm (no adjoint)",
        ),
        "L2 normotope (log-norm + adjoint)": ("tab:green", "L2 log-norm + adjoint"),
        f"L2 + adjoint + ReachiLQR ({ILQR_ITERS} iters)": (
            "tab:red",
            f"L2 + ReachiLQR ({ILQR_ITERS} iters)",
        ),
    }
    fig2, axb = plt.subplots(figsize=(6, 4.5))
    for name, name_tex, t, iover_at, vol_at, ref in methods:
        color, lab = styles[name]
        bl = []
        for k in plot_ks:
            if not alive(iover_at(k)):
                break
            bl.append(100.0 * (vol_at(k) / truth_t[ref][k] - 1.0))
        tsb = onp.array(plot_ks[: len(bl)]) * dt
        axb.semilogy(
            tsb, onp.maximum(onp.array(bl), 1e-1), color=color, lw=1.2, label=lab
        )
    axb.set_xlabel("$t$")
    axb.set_ylabel(r"bloat over own-init true volume (\%)")
    axb.set_title(r"Certificate bloat over time (clipped below at 0.1\%)")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    for ext in ["pdf", "svg"]:
        out = pathlib.Path(__file__).parent / f"robot_arm_bloat{TAG}{suffix}.{ext}"
        fig2.savefig(out)
        print(f"saved {out}")

    # ------------------------------------------------ ReachiLQR per iteration
    done = onp.array([d for d, _, _ in it_log])
    ratio = onp.array([r for _, r, _ in it_log])
    cum = onp.array([c for _, _, c in it_log])
    its = onp.arange(1, len(it_log) + 1)
    fig3, axv = plt.subplots(figsize=(6.5, 3))
    axv.semilogy(its, ratio, color="tab:blue", lw=1.2)
    axv.scatter(
        its[done],
        ratio[done],
        marker="o",
        s=18,
        facecolors="none",
        edgecolors="tab:blue",
        label="reached $t_f$",
    )
    if (~done).any():
        axv.scatter(
            its[~done],
            ratio[~done],
            marker="x",
            s=24,
            color="tab:blue",
            label="did not reach $t_f$",
        )
    axv.set_ylim(bottom=1.0)  # certified volume >= true volume
    axv.set_xlabel("ReachiLQR iteration $i$")
    axv.set_ylabel("certified / true volume", color="tab:blue")
    axv.tick_params(axis="y", colors="tab:blue")
    axr = axv.twinx()
    axr.plot(its, cum, color="tab:red", marker=".", lw=1.2)
    axr.set_ylim(bottom=0.0)
    axr.set_ylabel("cumulative runtime (s)", color="tab:red")
    axr.tick_params(axis="y", colors="tab:red")
    axv.legend(loc="upper center", fontsize=9)
    fig3.tight_layout()
    for ext in ["pdf", "svg"]:
        out = pathlib.Path(__file__).parent / f"robot_arm_ilqr{TAG}{suffix}.{ext}"
        fig3.savefig(out)
        print(f"saved {out}")

    return rows


# --------------------------------------------------------------------- table
results = {solver: run(solver, suffix) for solver, _, suffix in INTEGRATORS}


def fmt_rt(rt):
    return f"{rt[0]:.4f} ({rt[1]:.1f})"


# Runtimes for every integrator; volume and bloat for symplectic Euler (the
# per-integrator volumes agree to ~1%; see each integrator's figures).
base = results["symplectic"]
rows, rows_tex = [], []
for i, (name, name_tex, _, (a_s, a_tex), (b_s, b_tex)) in enumerate(base):
    rts = [fmt_rt(results[s][i][2]) for s, _, _ in INTEGRATORS]
    rows.append([name, *rts, a_s, b_s])
    rows_tex.append([name_tex, *rts, a_tex, b_tex])

headers = [
    "Method",
    *(f"{lab} s (JIT s)" for _, lab, _ in INTEGRATORS),
    f"Vol at t={T_AREA:g}",
    "Bloat",
]
headers_tex = [
    "Method",
    *(f"{lab} s (JIT s)" for _, lab, _ in INTEGRATORS),
    f"Vol at $t={T_AREA:g}$",
    "Bloat",
]
print(
    f"\n% robot arm, box x0 +- {PERT:g} / its circumscribed ball "
    f"(radius {2 * PERT:g}), "
    f"dt={dt}, tf={tf}; 4-D volume and bloat (symplectic Euler) vs own-initial-set "
    f"MC truth at t={T_AREA:g}"
)
try:
    from tabulate import tabulate
except ImportError:  # optional dependency; fall back to a hand-built table
    print(" & ".join(headers_tex) + " \\\\")
    for r in rows_tex:
        print(" & ".join(r) + " \\\\")
else:
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()
    # latex_raw, not latex_booktabs: the cells carry math ($\infty$, $m=2$, \%)
    # and booktabs escapes it.
    print(tabulate(rows_tex, headers=headers_tex, tablefmt="latex_raw"))
print()
