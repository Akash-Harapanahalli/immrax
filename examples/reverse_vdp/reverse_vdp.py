"""Reverse Van der Pol reachability from the 0.12 box, five certified methods.

Benchmark: reverse-time Van der Pol (mu=1), initial box (1.10, 0.55) +- 0.12,
dt = 0.005, tf = 8. The center is INSIDE the limit cycle, so the reverse flow
spirals into the origin and a sound, tight method must stabilize there; from
outside the cycle the true reverse flow escapes to infinity and no method can
contract.

The ellipsoid methods (L2, Chebyshev) start from the circumscribed ellipsoid
of the box, so the reference "true set" is the flow of that ellipsoid (light
gray), with the flow of the box inside it (darker gray) as the reference for
the box-initialized methods (interval, polytope).

Methods:
  1. Interval           -- natural inclusion embedding (natemb)
  2. Polytope adjoint   -- Polytope + AdjointEmbedding
  3. L2 log-norm        -- L2Normotope, mjacM log-norm, alpha frozen (no adjoint)
  4. L2 adjoint         -- L2Normotope, mjacM log-norm + adjoint
  5. Chebyshev m=3      -- ChebyshevNormotope, Carleman lift + adjoint + interval drift

All adjoint methods use the symplectic integrator (the compute_reachset default).
Produces examples/reverse_vdp.{pdf,svg} (snapshots on MC truth),
examples/reverse_vdp_bloat.{pdf,svg} (percent bloat over time), and a LaTeX table
(method, runtime, reach set area at tf) on stdout.
"""

import pathlib
from typing import ClassVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from immutabledict import immutabledict

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})  # Computer Modern

# ---------------------------------------------------------------- benchmark
class ReverseVanDerPol(irx.System):
    xlen: ClassVar[int] = 2
    mu: float = 1.0

    def f(self, t, x):
        x1, x2 = x
        return jnp.array([-x2, -self.mu * (1.0 - x1**2) * x2 + x1])


sys = ReverseVanDerPol()
ox0 = jnp.array([1.10, 0.55])
pert = jnp.array([0.12, 0.12])
ix0 = irx.icentpert(ox0, pert)
dt, tf = 0.005, 8.0
N = int(round(tf / dt))
snap_ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
snap_ks = [int(round(t / dt)) for t in snap_ts]
LABEL_TMAX = 3.0  # sets past this are too small near the origin to label
WMAX = 20.0  # a set wider than this is treated as blown up


def time_ms(f, *args):
    """Steady-state jitted runtime (ms): 6 runs, drop the compile call."""
    _, ts = irx.utils.run_times(6, f, *args)
    return float(jnp.mean(ts[1:])) * 1e3


def shoelace(V):
    V = onp.asarray(V)
    c = V.mean(0)
    V = V[onp.argsort(onp.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(onp.dot(x, onp.roll(y, -1)) - onp.dot(y, onp.roll(x, -1)))


# ---------------------------------------------------------------- MC truth
# Two nested initial sets: the box, and its circumscribed ellipsoid (the common
# initial set of the L2 / Chebyshev methods).
edge = jnp.linspace(0.0, 1.0, 100, endpoint=False)[:, None]
corners = jnp.array([[ix0.lower[0], ix0.lower[1]], [ix0.lower[0], ix0.upper[1]],
                     [ix0.upper[0], ix0.upper[1]], [ix0.upper[0], ix0.lower[1]]])
box_bnd = jnp.concatenate(
    [(1 - edge) * corners[i] + edge * corners[(i + 1) % 4] for i in range(4)]
)
theta = jnp.linspace(0.0, 2 * jnp.pi, 400, endpoint=False)
ell_bnd = ox0 + jnp.sqrt(2.0) * pert * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)


@jax.jit
def roll(x):
    def step(c, _):
        nx = c + dt * sys.f(0.0, c)
        return nx, nx

    _, xs = jax.lax.scan(step, x, None, length=N)
    return jnp.vstack([x, xs])


mc_box = onp.asarray(jax.vmap(roll)(box_bnd))  # (400, N+1, 2), ordered polygons
mc_ell = onp.asarray(jax.vmap(roll)(ell_bnd))


# ---------------------------------------------------------------- 1. interval
natemb = irx.natemb(sys)
z0 = jnp.concatenate([ix0.lower, ix0.upper])
reach_box = jax.jit(
    lambda z: natemb.compute_trajectory(0.0, tf, z, dt=dt, solver="euler").ys
)
box_ys = onp.asarray(reach_box(z0))
t_box = time_ms(reach_box, z0)

# ---------------------------------------------------------------- 2. polytope
pt0 = Polytope.from_interval(ix0)
emb_poly = AdjointEmbedding(sys, jnp.linalg.pinv(pt0.alpha), jnp.zeros((0, 2)))
rs_poly = emb_poly.compute_reachset(0.0, tf, pt0, dt=dt)
t_poly = time_ms(lambda p: emb_poly.compute_reachset(0.0, tf, p, dt=dt).ys[0].y, pt0)

# ------------------------------------------------------- 3/4. L2 normotopes
nt0 = irx.L2Normotope.from_interval(ix0)
emb_l2 = irx.NormotopeEmbedding(sys)
emb_l2._initialize(nt0)  # resolve gsc outside jit
noadj = immutabledict({"adjoint": False})
rs_l2n = emb_l2.compute_reachset(0.0, tf, nt0, dt=dt, f_kwargs=noadj)
rs_l2a = emb_l2.compute_reachset(0.0, tf, nt0, dt=dt)
t_l2n = time_ms(
    lambda p: emb_l2.compute_reachset(0.0, tf, p, dt=dt, f_kwargs=noadj).ys[0].y, nt0
)
t_l2a = time_ms(lambda p: emb_l2.compute_reachset(0.0, tf, p, dt=dt).ys[0].y, nt0)

# ---------------------------------------------------------- 5. Chebyshev m=3
cpt0 = irx.ChebyshevNormotope.from_interval(ix0, m=3)
emb_ch = irx.ChebyshevNormotopeEmbedding(sys, drift="interval")
emb_ch._initialize(cpt0)  # resolve deg_f outside jit
rs_ch = emb_ch.compute_reachset(0.0, tf, cpt0, dt=dt)
t_ch = time_ms(lambda p: emb_ch.compute_reachset(0.0, tf, p, dt=dt).ys[0].y, cpt0)


# ------------------------------------------------------------- set accessors
def poly_at(k):
    s = rs_poly.ys[0]
    return Polytope(s.ox[k], s.alpha[k], s.y[k])


def l2_at(rs, k):
    s = rs.ys[0]
    return irx.L2Normotope(s.ox[k], s.alpha[k], s.y[k])


def cheb_at(k):
    s = rs_ch.ys[0]
    return irx.ChebyshevNormotope(s.ox[k], s.T[k], s.alpha[k], s.y[k])


def box_iover_at(k):
    return irx.interval(jnp.asarray(box_ys[k, :2]), jnp.asarray(box_ys[k, 2:]))


def poly_iover_at(k):
    p = poly_at(k)
    return irx.interval(jnp.linalg.inv(p.alpha)) @ p.hinv(p.y) + p.ox


def alive(io):
    """Set with interval hull io is finite and not blown up."""
    lo, hi = onp.asarray(io.lower), onp.asarray(io.upper)
    return bool(onp.isfinite(lo).all() and onp.isfinite(hi).all()
                and (hi - lo < WMAX).all())


def death_time(iover_at):
    for k in range(N + 1):
        if not alive(iover_at(k)):
            return k * dt
    return None


def ellipse_boundary(nt, npts=200):
    th = onp.linspace(0, 2 * onp.pi, npts)
    circ = onp.stack([onp.cos(th), onp.sin(th)], axis=0)
    Hinv = onp.linalg.inv(onp.asarray(nt.alpha))
    return (onp.asarray(nt.ox)[:, None] + float(nt.y) * Hinv @ circ).T


def cheb_area(pt, ngrid=400):
    """Grid-membership area of {g <= y} inside a padded iover window."""
    io = pt.iover()
    lo, hi = onp.asarray(io.lower), onp.asarray(io.upper)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xs = onp.linspace(lo[0], hi[0], ngrid)
    ys = onp.linspace(lo[1], hi[1], ngrid)
    X, Y = onp.meshgrid(xs, ys)
    pts = jnp.asarray(onp.stack([X.ravel(), Y.ravel()], axis=1), jnp.float32)
    inside = onp.asarray(jax.vmap(pt.g)(pts)) <= float(pt.y)
    cell = (hi[0] - lo[0]) * (hi[1] - lo[1]) / ngrid**2
    return inside.sum() * cell


# --------------------------------------------------------------------- table
T_AREA = 2.0  # areas reported at this time (a set visible in the figure)
kf = int(round(T_AREA / dt))
# Each method's bloat is measured against the MC truth for ITS OWN initial
# set: box flow for interval/polytope, ellipsoid flow for L2/Chebyshev.
truth = {"box": shoelace(mc_box[:, kf, :]), "ell": shoelace(mc_ell[:, kf, :])}
methods = [
    ("Interval (natural embedding)", t_box, box_iover_at,
     lambda k: float(onp.prod(box_ys[k, 2:] - box_ys[k, :2])), "box"),
    ("Polytope + adjoint", t_poly, poly_iover_at,
     lambda k: shoelace(poly_at(k).get_vertices()), "box"),
    ("L2 normotope (log-norm, no adjoint)", t_l2n, lambda k: l2_at(rs_l2n, k).iover(),
     lambda k: onp.pi * float(l2_at(rs_l2n, k).y) ** 2
     / abs(onp.linalg.det(onp.asarray(l2_at(rs_l2n, k).alpha))), "ell"),
    ("L2 normotope (log-norm + adjoint)", t_l2a, lambda k: l2_at(rs_l2a, k).iover(),
     lambda k: onp.pi * float(l2_at(rs_l2a, k).y) ** 2
     / abs(onp.linalg.det(onp.asarray(l2_at(rs_l2a, k).alpha))), "ell"),
    ("Chebyshev normotope $m=3$ (adjoint + interval drift)", t_ch,
     lambda k: cheb_at(k).iover(), lambda k: cheb_area(cheb_at(k)), "ell"),
]

print(f"\n% reverse VDP, box (1.10, 0.55) +- 0.12 / circumscribed ellipsoid, "
      f"dt={dt}, tf={tf}; area and bloat vs own-initial-set MC truth at t={T_AREA:g}")
print("\\begin{tabular}{lrrr}")
print("\\toprule")
print(f"Method & Runtime (ms) & Area at $t={T_AREA:g}$ & Bloat \\\\")
print("\\midrule")
for name, t, iover_at, area_at, ref in methods:
    td = death_time(iover_at)
    if td is not None and td <= T_AREA:
        astr, bstr = f"$\\infty$ (blows up $t \\approx {td:.2f}$)", "--"
    else:
        a = area_at(kf)
        astr = f"{a:.3e}"
        bstr = f"{100.0 * (a / truth[ref] - 1.0):+.1f}\\%"
    print(f"{name} & {t:.2f} & {astr} & {bstr} \\\\")
print("\\bottomrule")
print("\\end{tabular}\n")

# ---------------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7, 7))
for k in snap_ks:
    ax.fill(mc_ell[:, k, 0], mc_ell[:, k, 1], color="0.85", zorder=0)
    ax.fill(mc_box[:, k, 0], mc_box[:, k, 1], color="0.68", zorder=0.5)
    cen = mc_box[:, k, :].mean(0)
    if k * dt <= LABEL_TMAX:
        ax.annotate(f"t={k * dt:g}", cen, color="0.25", fontsize=9,
                    ha="center", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
    else:
        # Tiny sets near the origin: put the label off to the side (radially
        # outward) with a thin connector so the set stays visible.
        u = cen / max(float(onp.linalg.norm(cen)), 1e-9)
        ax.annotate(f"t={k * dt:g}", cen, xytext=cen + 0.22 * u,
                    color="0.25", fontsize=9, ha="center", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5,
                                    shrinkA=2, shrinkB=2),
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
ax.fill([], [], color="0.85", label="true set (ellipsoid init)")
ax.fill([], [], color="0.68", label="true set (box init $\\subset$ ellipsoid)")

for k in snap_ks:
    if not alive(box_iover_at(k)):
        continue
    lo, hi = box_ys[k, :2], box_ys[k, 2:]
    bx = [lo[0], hi[0], hi[0], lo[0], lo[0]]
    by = [lo[1], lo[1], hi[1], hi[1], lo[1]]
    ax.plot(bx, by, color="tab:purple", lw=0.9,
            label="interval" if k == 0 else None)

for k in snap_ks:
    if not alive(poly_iover_at(k)):
        continue
    V = onp.asarray(poly_at(k).get_vertices())
    c = V.mean(0)
    V = V[onp.argsort(onp.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]
    V = onp.vstack([V, V[0]])
    ax.plot(V[:, 0], V[:, 1], color="tab:blue", lw=0.9,
            label="polytope + adjoint" if k == 0 else None)

for rs, color, ls, lw, al, lab in [
    (rs_l2n, "tab:olive", "-", 0.9, 0.45, "L2 log-norm (no adjoint)"),
    (rs_l2a, "tab:green", "-", 0.9, 1.0, "L2 log-norm + adjoint"),
]:
    for k in snap_ks:
        nt = l2_at(rs, k)
        if not alive(nt.iover()):
            continue
        B = ellipse_boundary(nt)
        ax.plot(B[:, 0], B[:, 1], color=color, ls=ls, lw=lw, alpha=al,
                label=lab if k == 0 else None)

for k in snap_ks:
    pk = cheb_at(k)
    io = pk.iover()
    if not alive(io):
        continue
    lo, hi = onp.asarray(io.lower), onp.asarray(io.upper)
    pad = 0.15 * (hi - lo)
    xs = onp.linspace(lo[0] - pad[0], hi[0] + pad[0], 220)
    ys = onp.linspace(lo[1] - pad[1], hi[1] + pad[1], 220)
    X, Y = onp.meshgrid(xs, ys)
    pts = jnp.asarray(onp.stack([X.ravel(), Y.ravel()], axis=1), jnp.float32)
    Z = onp.asarray(jax.vmap(pk.g)(pts)).reshape(X.shape) - float(pk.y)
    ax.contour(X, Y, Z, levels=[0.0], colors=["tab:red"], linewidths=0.9)
ax.plot([], [], color="tab:red", lw=0.9, label="Chebyshev m=3")

# Frame the MC tube (square window), not the whole limit cycle.
mid = 0.5 * (mc_ell.reshape(-1, 2).min(0) + mc_ell.reshape(-1, 2).max(0))
half = 0.5 * (mc_ell.reshape(-1, 2).max(0) - mc_ell.reshape(-1, 2).min(0)).max() + 0.25
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="best", fontsize=9, framealpha=0.9)
ax.set_title(
    f"Reverse Van der Pol, box $(1.10, 0.55) \\pm 0.12$ and its circumscribed "
    f"ellipsoid\n(dt={dt}, snapshots to t={tf:g})"
)
fig.tight_layout()
for ext in ["pdf", "svg"]:
    out = pathlib.Path(__file__).parent / f"reverse_vdp.{ext}"
    fig.savefig(out)
    print(f"saved {out}")

# ------------------------------------------------------------ bloat over time
# Percent bloat = 100 (area / own-init MC truth area - 1), while alive.
plot_ks = list(range(0, N + 1, 8))
styles = {
    "Interval (natural embedding)": ("tab:purple", "interval"),
    "Polytope + adjoint": ("tab:blue", "polytope + adjoint"),
    "L2 normotope (log-norm, no adjoint)": ("tab:olive", "L2 log-norm (no adjoint)"),
    "L2 normotope (log-norm + adjoint)": ("tab:green", "L2 log-norm + adjoint"),
    "Chebyshev normotope $m=3$ (adjoint + interval drift)": ("tab:red", "Chebyshev m=3"),
}
truth_t = {
    "box": onp.array([shoelace(mc_box[:, k, :]) for k in plot_ks]),
    "ell": onp.array([shoelace(mc_ell[:, k, :]) for k in plot_ks]),
}
fig2, axb = plt.subplots(figsize=(6, 4.5))
for name, t, iover_at, area_at, ref in methods:
    color, lab = styles[name]
    bl = []
    for i, k in enumerate(plot_ks):
        if not alive(iover_at(k)):
            break
        bl.append(100.0 * (area_at(k) / truth_t[ref][i] - 1.0))
    tsb = onp.array(plot_ks[: len(bl)]) * dt
    axb.semilogy(tsb, onp.maximum(onp.array(bl), 1e-1), color=color, lw=1.2, label=lab)
axb.set_xlabel("$t$")
axb.set_ylabel(r"bloat over own-init true area (\%)")
axb.set_title(r"Certificate bloat over time (clipped below at 0.1\%)")
axb.legend(fontsize=8)
axb.grid(alpha=0.3, which="both")
fig2.tight_layout()
for ext in ["pdf", "svg"]:
    out = pathlib.Path(__file__).parent / f"reverse_vdp_bloat.{ext}"
    fig2.savefig(out)
    print(f"saved {out}")
plt.show()
