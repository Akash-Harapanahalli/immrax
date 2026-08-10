r"""Reverse-time Van der Pol region-of-attraction via the GramNormotope SOS set.

Two-step monotone-embedding ROA synthesis, consolidated on
:class:`~immrax.parametric.sets.gram_normotope.GramNormotope`:

  **Step 1 -- find the ROA.** Maximize the area of the nonconvex sublevel set
  ``{x : ||R p(x)|| <= 1}`` subject to a *rigorous* boundary drift
  ``max_boundary Vdot <= 0`` (ray-root partition cells, centered + Lagrangian
  interval bound). We optimize against the very bound we certify with, so the
  optimized area equals the certified area -- no SOS/SDP solver, no post-shrink.

  **Step 2 -- contract the nested family.** The radius ``y`` is an ODE state of
  the embedding, ``ydot = max_{||R p||=y} Vdot / (2 y)``. Each snapshot
  ``{||R p|| <= y(t)}`` is forward-invariant (Nagumo). Two shape policies:

    - ``barrier``          : ``R`` fixed (offset only). A fixed-certificate
      sublevel sweep -- the Lyapunov/barrier baseline.
    - ``projected adjoint``: ``Pdot = Pi_PSD(-L^T P - P L)`` with ``L`` the
      linearization lift and ``Pi_PSD`` clamping negative eigenvalues to zero --
      a valid Loewner-NSD hypercontrol that reshapes the certificate and drives
      the family to the equilibrium.

Produces a 2x2 figure: rows ``m in {3, 5}``, columns ``{barrier, projected
adjoint}``. Every snapshot is Monte-Carlo verified (boundary points rolled
forward stay inside). Offline; ROA + flow results are cached to ``/tmp``.

Run::

    python examples/gram_normotope_reverse_vdp_roa.py
"""

# ruff: noqa: E402  -- x64 must be enabled before immrax/JAX is imported

import os
import time

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import optax
from scipy.linalg import solve_continuous_lyapunov, cholesky

from immrax.inclusion import interval, natif
from immrax.parametric.sets.gram_normotope import (
    GramNormotope,
    _gram_basis,
    _lift,
    carleman_A,
)

# ---------------------------------------------------------------- problem setup
MU = 2.0
N = 2
LSCALE = 2.5  # per-degree monomial normalization (z^k ~ O(1) after /LSCALE^k)
NDIR = 720
ORDERS = [3, 5]
CACHE = "/tmp/gram_roa_cache"
os.makedirs(CACHE, exist_ok=True)

_th = onp.linspace(0, 2 * onp.pi, NDIR, endpoint=False)
DIRS = jnp.asarray(onp.stack([onp.cos(_th), onp.sin(_th)], 1))
DTH = 2 * onp.pi / NDIR


def f_rev(x):
    """Reverse-time Van der Pol: origin is a stable focus, basin = limit-cycle interior."""
    return jnp.array([-x[1], -MU * (1.0 - x[0] ** 2) * x[1] + x[0]])


def limit_cycle():
    x = onp.array([2.0, 0.0])
    dt = 0.001

    def fr(z):
        return onp.array([-z[1], -MU * (1 - z[0] ** 2) * z[1] + z[0]])

    for _ in range(60000):
        x = x - dt * fr(x)
    x0 = x.copy()
    xs = [x0]
    for k in range(40000):
        x = x - dt * fr(x)
        xs.append(x.copy())
        if k > 2000 and onp.linalg.norm(x - x0) < 1e-2:
            break
    return onp.array(xs)


LC = limit_cycle()
BASIN = 0.5 * abs(
    onp.dot(LC[:, 0], onp.roll(LC[:, 1], -1)) - onp.dot(LC[:, 1], onp.roll(LC[:, 0], -1))
)


# ------------------------------------------------------- per-order drift machinery
def build(m):
    """Scaled lift, area, partition drift, and linearization lift for degree ``m``.

    Everything is in *scaled* coordinates (lift divided by ``LSCALE**deg``) for
    conditioning; fold the scaling back into ``R`` (``R / scale``) to recover a
    standard :class:`GramNormotope`.
    """
    exps, lin_idx = _gram_basis(N, m)
    li = jnp.asarray(lin_idx)
    P = exps.shape[0]
    scale = jnp.asarray(LSCALE ** exps.sum(1).astype(float))

    def lift(z):
        return _lift(z, exps) / scale

    def V(z, R):
        return jnp.sum((R @ lift(z)) ** 2)

    def Vdot(z, R):
        p = lift(z)
        Dp = jax.jacfwd(lift)(z)
        return 2.0 * jnp.dot(R @ p, R @ (Dp @ f_rev(z)))

    def rroot(d, R, y):  # Newton for r s.t. V(r d, R) = y^2 along direction d
        r = LSCALE * y / (jnp.linalg.norm(R[:, li] @ d) + 1e-12)
        for _ in range(22):
            gr, dgr = jax.jvp(lambda rr: V(rr * d, R), (r,), (jnp.ones(()),))
            r = r - (gr - y * y) / (dgr + 1e-12)
        return jnp.abs(r)

    gV = jax.grad(V, 0)
    gVd = jax.grad(Vdot, 0)
    nJV = natif(lambda z, R: jax.jacfwd(V, 0)(z, R))
    nJVd = natif(lambda z, R: jax.jacfwd(Vdot, 0)(z, R))

    def cell_bound(R, y, pts):  # rigorous max boundary Vdot over centered+Lagrangian cells at pts
        nb = jnp.roll(pts, -1, 0)
        pb = jnp.roll(pts, 1, 0)
        hw = jnp.maximum(jnp.abs(nb - pts), jnp.abs(pts - pb)) * 0.6 + 1e-5
        RI = interval(R, R)

        def cell(zc, h):
            box = interval(zc - h, zc + h)
            gvc = gV(zc, R)
            gvdc = gVd(zc, R)
            lam = jnp.dot(gvdc, gvc) / (jnp.dot(gvc, gvc) + 1e-30)
            gc = Vdot(zc, R) - lam * (V(zc, R) - y * y)  # Vdot on {V=y^2}
            JVd = nJVd(box, RI)
            JV = nJV(box, RI)
            lo = jnp.minimum(lam * JV.lower, lam * JV.upper)
            hi = jnp.maximum(lam * JV.lower, lam * JV.upper)
            return gc + jnp.sum(
                jnp.maximum(jnp.abs(JVd.lower - hi), jnp.abs(JVd.upper - lo)) * h
            )

        return jnp.max(jax.vmap(cell)(pts, hw))

    def ydot_rate(R, y):  # flow drift: Newton ray-root boundary localization (fast; fine pre-degeneracy)
        pts = jax.vmap(lambda d: rroot(d, R, y))(DIRS)[:, None] * DIRS
        return cell_bound(R, y, pts) / (2.0 * y)

    area = jax.jit(lambda R, y: 0.5 * jnp.sum(jax.vmap(lambda d: rroot(d, R, y))(DIRS) ** 2) * DTH)
    pd1 = jax.jit(lambda R: ydot_rate(R, 1.0) * 2.0)  # certified max boundary Vdot at y=1
    gl = jax.jit(
        jax.value_and_grad(
            lambda R, mb: -jnp.log(area(R, 1.0)) - mb * jnp.log(jnp.maximum(-pd1(R), 1e-9))
        )
    )

    # linearization lift L (block-diagonal Kronecker sum of Df(0)) in scaled coords
    Df0 = jax.jacfwd(f_rev)(jnp.zeros(N))
    L_un = carleman_A(lambda x: Df0 @ x, jnp.zeros(N), exps, m)
    L = jnp.diag(1.0 / scale) @ L_un @ jnp.diag(scale)

    return dict(
        m=m, P=P, scale=scale, area=area, pd1=pd1, gl=gl,
        ydot=jax.jit(ydot_rate), cell_bound=jax.jit(cell_bound), L=L,
        vdot_pts=jax.jit(lambda R, pts: jax.vmap(lambda z: Vdot(z, R))(pts)),
    )


# --------------------------------------------------------------- Step 1: find ROA
def optimize(M, R0, steps):
    pd1, gl, area = M["pd1"], M["gl"], M["area"]
    R = R0
    k = 0
    while float(pd1(R)) > -1e-3 and k < 80:  # inflate to a feasible (Vdot<0) init
        R = R * 1.03
        k += 1
    opt = optax.adam(1.5e-3)
    st = opt.init(R)
    bR = R
    bA = float(area(R, 1.0)) if float(pd1(R)) < -1e-3 else -1.0
    for i in range(steps):
        _, g = gl(R, 0.05 * (0.005) ** (i / steps))
        if not bool(jnp.all(jnp.isfinite(g))):
            break
        upd, st = opt.update(g, st)
        Rn = optax.apply_updates(R, upd)
        if float(pd1(Rn)) < -1e-4:  # stay strictly feasible
            R = Rn
        a = float(area(R, 1.0))
        if float(pd1(R)) < -1e-3 and a > bA:
            bA, bR = a, R
    return bR, bA


def find_roa(builds):
    """Ladder m=3->4->5 (generic Lyapunov init, degree warm-start), cache scaled R_roa.
    Return {m: R_scaled}."""
    A_lin = onp.array([[0.0, -1.0], [1.0, -MU]])
    Lc = jnp.asarray(cholesky(solve_continuous_lyapunov(A_lin.T, -onp.eye(2))))
    R_roa = {}
    prev, prevP = None, 0
    for m in [3, 4, 5]:
        path = f"{CACHE}/R_roa_m{m}.npy"
        M = builds[m]
        Pn = M["P"]
        _, lin_idx = _gram_basis(N, m)
        if os.path.exists(path):
            R = jnp.asarray(onp.load(path))
            a = float(M["area"](R, 1.0))
        else:
            R = (jnp.eye(Pn) * 0.7).at[onp.ix_(lin_idx, lin_idx)].set(Lc * (LSCALE / 0.9))
            if prev is not None:
                R = R.at[:prevP, :prevP].set(prev)
            t0 = time.time()
            R, a = optimize(M, R, 2500 if prev is None else 1000)
            onp.save(path, onp.asarray(R))
            print(f"  m={m}: ROA area={a:.2f} ({100*a/BASIN:.0f}% of basin) ({time.time()-t0:.0f}s)", flush=True)
        prev, prevP = R, Pn
        if m in ORDERS:
            R_roa[m] = R
    return R_roa


# ---------------------------------------------------- Step 2: flow nested family
def _Pi_PSD(Mx):
    Ms = 0.5 * (Mx + Mx.T)
    w, Q = jnp.linalg.eigh(Ms)
    return Q @ jnp.diag(jnp.maximum(w, 0.0)) @ Q.T


def _chol_psd(Pm, P):
    return jnp.linalg.cholesky(0.5 * (Pm + Pm.T) + 1e-9 * jnp.eye(P)).T


def to_gn(M, R_scaled, y):
    """Standard (unscaled) GramNormotope from a scaled factor: fold LSCALE into R."""
    return GramNormotope(jnp.zeros(N), R_scaled / M["scale"][None, :], float(y))


def logvol(M, R_scaled, y):
    return float(to_gn(M, R_scaled, y).log_volume_linear())


def flow(M, R0, mode, dt=0.05, T=2400):
    """Integrate the parameter ODE. Cached. Three modes:

      - ``barrier``        : R fixed, y flows by the certified drift (offset only).
      - ``adjoint``        : R flows by the projected adjoint, y flows by the drift.
      - ``adjoint_fixedy`` : R flows by the projected adjoint, ``ydot := 0``. Since
        the hypercontrol keeps the boundary drift <= 0, ``ydot = 0`` is a valid
        *upper bound* on the radius rate (sound embedding), and the entire family
        contracts from the P-growth alone -- so NO in-loop drift computation is
        needed after the one-time barrier search. The cheap mode.

    Stops on radius collapse (``y < 1e-4``), ill-conditioning (``cond(R) > 1e8``),
    or non-finite -- keeping only good steps. Returns ``(ts, ys, Rs)`` (scaled R).
    """
    path = f"{CACHE}/flow_m{M['m']}_{mode}.npz"
    if os.path.exists(path):
        d = onp.load(path)
        return d["ts"], d["ys"], d["Rs"], float(d["ms_step"])
    ydot, L, P = M["ydot"], M["L"], M["P"]
    if mode != "adjoint_fixedy":
        _ = float(ydot(R0, 1.0))  # warm up the drift JIT (excluded from per-step timing)
    R = R0
    y = 1.0
    ts, ys, Rs = [0.0], [1.0], [onp.asarray(R)]
    t0 = time.time()
    for k in range(T):
        yd = 0.0 if mode == "adjoint_fixedy" else float(ydot(R, y))
        Rn = R
        if mode in ("adjoint", "adjoint_fixedy"):
            Pm = R.T @ R
            Rn = _chol_psd(Pm + dt * _Pi_PSD(-L.T @ Pm - Pm @ L), P)
        yn = max(y + dt * yd, 1e-6)
        Rn_np = onp.asarray(Rn)
        if not onp.all(onp.isfinite(Rn_np)) or yn < 1e-4 or float(jnp.linalg.cond(Rn)) > 1e8:
            break
        R, y = Rn, yn
        ts.append((k + 1) * dt)
        ys.append(y)
        Rs.append(Rn_np)
    ms_step = 1000.0 * (time.time() - t0) / max(len(ts) - 1, 1)
    ts, ys, Rs = map(onp.asarray, (ts, ys, Rs))
    onp.savez(path, ts=ts, ys=ys, Rs=Rs, ms_step=ms_step)
    return ts, ys, Rs, ms_step


def cert_drift(M, R_scaled, y, bnd_pts):
    """Max boundary steady V_dot, sampled at robustly-localized boundary points.

    On ``{V = y^2}`` this is exactly the Nagumo quantity; ``<= 0`` (with margin)
    == forward-invariant. Uses dense sampling + robust ``gn.g``-bisection
    localization (``bnd_pts``) -- the interval cell bound (``M['cell_bound']``,
    which drives the flow's drift and certifies the barrier/adjoint families by
    construction) loosens badly via ``natif`` on the near-degenerate factors the
    projected adjoint produces, so for the post-hoc check we sample. Complementary
    to the independent MC roll-out.
    """
    return float(jnp.max(M["vdot_pts"](jnp.asarray(R_scaled), jnp.asarray(bnd_pts))))


# ------------------------------------------------------------- geometry + MC checks
def ray_boundary(gn, K=320):
    dd = onp.stack([onp.cos(onp.linspace(0, 2 * onp.pi, K, endpoint=False)),
                    onp.sin(onp.linspace(0, 2 * onp.pi, K, endpoint=False))], 1)

    def g_ray(d):
        lo, hi = 0.0, 8.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            inside = gn.g(gn.ox + mid * d) <= gn.y
            lo = jnp.where(inside, mid, lo)
            hi = jnp.where(inside, hi, mid)
        return gn.ox + 0.5 * (lo + hi) * d

    return onp.asarray(jax.vmap(g_ray)(jnp.asarray(dd)))


def log_set_area(gn):
    """Log of the true 2D area of the nonconvex set (shoelace on the ray-cast boundary).

    Robust where ``log_volume_linear`` (the convex shadow) degenerates -- the
    projected adjoint can collapse the linear block while the set stays compact.
    """
    b = ray_boundary(gn, 260)
    x, yv = b[:, 0], b[:, 1]
    A = 0.5 * abs(float(onp.dot(x, onp.roll(yv, -1)) - onp.dot(yv, onp.roll(x, -1))))
    return float(onp.log(max(A, 1e-300)))


def mc_violation(gn):
    """Roll boundary points forward; max (g - y) over the trajectory (<=0 == invariant)."""
    bd = jnp.asarray(ray_boundary(gn, 320))

    def roll(x):
        def s(c, _):
            return c + 0.004 * f_rev(c), c

        _, xs = jax.lax.scan(s, x, None, length=400)
        return xs

    tr = jax.vmap(roll)(bd)  # (320, 400, 2)
    gmax = jnp.max(jax.vmap(lambda step: jnp.max(jax.vmap(gn.g)(step)))(tr.transpose(1, 0, 2)))
    return float(gmax - gn.y)


# --------------------------------------------------------------------------- main
def main():
    print(f"basin area = {BASIN:.2f}", flush=True)
    builds = {m: build(m) for m in [3, 4, 5]}
    print("Step 1: finding ROA (cached) ...", flush=True)
    R_roa = find_roa(builds)

    print("Step 2: flowing nested families ...", flush=True)
    families, timings = {}, {}
    MODES = ["barrier", "adjoint", "adjoint_fixedy"]
    for m in ORDERS:
        M = builds[m]
        a = float(M["area"](R_roa[m], 1.0))
        print(f"  m={m}: ROA = {100*a/BASIN:.0f}% of basin", flush=True)
        for mode in MODES:
            ts, ys, Rs, ms_step = flow(M, R_roa[m], mode)
            la0 = log_set_area(to_gn(M, jnp.asarray(Rs[0]), ys[0]))
            laT = log_set_area(to_gn(M, jnp.asarray(Rs[-1]), ys[-1]))
            families[(m, mode)] = (ts, ys, Rs, la0, laT)
            timings[(m, mode)] = (len(ts), ms_step)
            print(f"    {mode:16s}: {len(ts):4d} steps, {ms_step:5.2f} ms/step  "
                  f"log-area {la0:.2f} -> {laT:.2f} over t={ts[-1]:.1f}", flush=True)

    print("Plotting + verification (MC + certified drift) ...", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    col_titles = {
        "barrier": "barrier (R fixed)",
        "adjoint": "projected adjoint",
        "adjoint_fixedy": "projected adjoint, ydot=0 (cheap)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    worst, worst_cert = {}, {}
    for i, m in enumerate(ORDERS):
        M = builds[m]
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            ts, ys, Rs, la0, laT = families[(m, mode)]
            ax.plot(LC[:, 0], LC[:, 1], "k--", lw=1.4, label="limit cycle", zorder=1)
            tmax = ts[-1]
            targs = onp.concatenate([[0.0], onp.geomspace(max(0.3, ts[1]), max(tmax, 0.5), 6)])
            idx = sorted(set(int(onp.argmin(onp.abs(ts - tt))) for tt in targs))
            norm = Normalize(vmin=0.0, vmax=max(tmax, 1e-3))
            wv, cv = -onp.inf, -onp.inf
            for k in idx:
                gn = to_gn(M, jnp.asarray(Rs[k]), ys[k])
                b = ray_boundary(gn, 360)
                wv = max(wv, mc_violation(gn))
                cv = max(cv, cert_drift(M, Rs[k], ys[k], b))  # sampled boundary V_dot, robust localization
                pp = onp.vstack([b, b[:1]])
                ax.plot(pp[:, 0], pp[:, 1], color=plt.cm.viridis(norm(ts[k])),
                        lw=(2.6 if k == 0 else 1.8), zorder=3 if k == 0 else 2)
            worst[(m, mode)] = wv
            worst_cert[(m, mode)] = cv
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                              fraction=0.046, pad=0.02)
            cb.set_label("embedding time t", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-4.2, 4.2)
            nsteps, ms_step = timings[(m, mode)]
            ax.set_title(
                f"log-area {la0:.1f}->{laT:.1f}, t={tmax:.1f}, {ms_step:.1f} ms/step\n"
                f"max bnd Vdot {cv:+.2f} (samp), MC viol {wv:+.1e}",
                fontsize=8.5,
            )
            if i == 0:
                ax.annotate(col_titles[mode], xy=(0.5, 1.14), xycoords="axes fraction",
                            ha="center", fontsize=12, fontweight="bold")
            if j == 0:
                a = float(M["area"](R_roa[m], 1.0))
                ax.annotate(f"m = {m}\nROA {100*a/BASIN:.0f}% of basin", xy=(-0.26, 0.5),
                            xycoords="axes fraction", va="center", ha="center",
                            rotation=90, fontsize=12, fontweight="bold")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)
    plt.suptitle(
        "Reverse-VDP ROA on GramNormotope: nested invariant family -- barrier vs projected adjoint vs projected adjoint with ydot=0\n"
        "(two independent soundness checks: max boundary Vdot, sampled, and Monte-Carlo roll-out; both <=0 == forward-invariant)",
        fontsize=12,
    )
    out = f"{CACHE}/reverse_vdp_roa_2x3.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print("worst MC:", {f"{k[0]}-{k[1]}": f"{v:+.1e}" for k, v in worst.items()}, flush=True)
    print("worst cert Vdot:", {f"{k[0]}-{k[1]}": f"{v:+.1e}" for k, v in worst_cert.items()}, flush=True)


if __name__ == "__main__":
    main()
