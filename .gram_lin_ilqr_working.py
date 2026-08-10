"""Synthesis probe v3: SCALED-coord drift (the tightness fix).

State R is the SCALED factor R_s (R_banana = R_s/scale); the drift uses the scaled
lift ps = _lift/scale so degree-6 monomials stay O(1) and the natif bound is tight
(memory gotcha #1). Everything else as v2:
  N = <R_s ps, R_s (Dps f_err) + U ps>,  ydot = max_{V=y^2} N/y  (ray-root, 720 dirs)
  R_dot = U  (linear, no freeze),  pt0 = banana warm-start.
"""

import os, time
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
from typing import ClassVar

import immrax as irx
from immrax.parametric import ReachiLQR
from immrax.parametric.sets.gram_normotope import (
    GramNormotope, GramNormotopeEmbedding, _gram_basis, _lift,
)
from immrax.inclusion import interval, natif

MU = 2.0
LSCALE = 2.5
N, M = 2, 3
exps, lin_idx = _gram_basis(N, M)
scale = jnp.asarray(LSCALE ** exps.sum(1).astype(float))

NDIR = int(os.environ.get("NDIR", "720"))
_th = onp.linspace(0, 2 * onp.pi, NDIR, endpoint=False)
DIRS = jnp.asarray(onp.stack([onp.cos(_th), onp.sin(_th)], 1))


class ReverseVDP(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([-x[1], -MU * (1 - x[0] ** 2) * x[1] + x[0]])


sys = ReverseVDP()


def linear_drift(f, ox, Rs, y, U):
    ps = lambda z: _lift(z, exps) / scale          # SCALED lift (O(1) monomials)
    fe = lambda z: f(ox + z) - f(ox)

    def Nfun(z):
        Rp = Rs @ ps(z)
        esc = jax.jvp(ps, (z,), (fe(z),))[1]
        return jnp.dot(Rp, Rs @ esc + U @ ps(z))

    def Vfun(z):
        return jnp.sum((Rs @ ps(z)) ** 2)

    nJN, nJV = natif(jax.jacfwd(Nfun)), natif(jax.jacfwd(Vfun))
    gN, gV = jax.grad(Nfun), jax.grad(Vfun)
    li = jnp.asarray(lin_idx)

    def rroot(d):
        r = LSCALE * y / (jnp.linalg.norm(Rs[:, li] @ d) + 1e-12)
        for _ in range(10):
            gr, dgr = jax.jvp(lambda rr: Vfun(rr * d), (r,), (jnp.ones(()),))
            r = r - (gr - y * y) / (dgr + 1e-12)
        return jnp.abs(r)

    # stop_gradient: don't differentiate the boundary localization through the
    # iLQR backward pass (the Newton unroll x NDIR is the compile killer); the
    # drift VALUE stays exact, only the boundary-motion gradient is dropped.
    pts = jax.lax.stop_gradient(jax.vmap(rroot)(DIRS)[:, None] * DIRS)
    nb, pb = jnp.roll(pts, -1, 0), jnp.roll(pts, 1, 0)
    hw = jnp.maximum(jnp.abs(nb - pts), jnp.abs(pts - pb)) * 0.6 + 1e-5

    def cell(zc, h):
        bx = interval(zc - h, zc + h)
        gNc, gVc = gN(zc), gV(zc)
        lam = jnp.dot(gNc, gVc) / (jnp.dot(gVc, gVc) + 1e-30)
        gc = Nfun(zc) - lam * (Vfun(zc) - y * y)
        JN, JV = nJN(bx), nJV(bx)
        lo = jnp.minimum(lam * JV.lower, lam * JV.upper)
        hi = jnp.maximum(lam * JV.lower, lam * JV.upper)
        return gc + jnp.sum(jnp.maximum(jnp.abs(JN.lower - hi), jnp.abs(JN.upper - lo)) * h)

    return jnp.max(jax.vmap(cell)(pts, hw)) / y


class GramTubeEmbedding(GramNormotopeEmbedding):
    def _dynamics(self, t, state, *args, U=None, adjoint=True, ix=None):
        pt, _aux = state          # pt.alpha holds the SCALED factor R_s
        Rs, y = pt.alpha, pt.y
        f = lambda x: self.sys.f(t, x)
        Up = jnp.zeros_like(Rs) if U is None else U
        R_dot = Up
        y_dot = linear_drift(f, pt.ox, Rs, y, Up) * y
        return pt.__class__(self.sys.f(t, pt.ox), R_dot, y_dot), None


Rs0 = jnp.asarray(onp.load("/tmp/gram_roa_cache/R_roa_m3.npy"))   # SCALED banana factor
Y0 = float(os.environ.get("Y0", "0.92"))
pt0 = GramNormotope(jnp.zeros(2), Rs0, Y0)                        # alpha = R_s
to_banana = lambda pt: GramNormotope(pt.ox, pt.alpha / scale[None, :], pt.y)
print(f"banana cond(R_s)={float(jnp.linalg.cond(Rs0)):.0f}", flush=True)

emb = GramTubeEmbedding(sys, G=1)
d0, _ = emb._dynamics(0.0, (pt0, None), U=None)
print(f"banana (y=1) U=0 ydot [scaled] = {float(d0.y):+.5f}  (example frozen ~ -0.001)", flush=True)


def terminal_cost(pt):
    return pt.log_volume_linear()


def running_cost(t, pt, U):
    return jnp.array(0.0)


Nctl = int(os.environ.get("NCTL", "150"))
dt = float(os.environ.get("DT", "0.08"))
ilqr = ReachiLQR(emb, pt0, terminal_cost, running_cost, 0.0, Nctl * dt, Nctl,
                 DDP=False, dt=dt, track_iover=True)
print(f"Xlen={ilqr.Xlen} Ulen={ilqr.Ulen} N={ilqr.N} tf={Nctl*dt:.1f}", flush=True)
t0 = time.time(); ilqr.setup(); print(f"JIT {time.time()-t0:.1f}s", flush=True)

res0 = ilqr.iterate(ilqr.initial_controls(), *ilqr.initial_gains(), 50.0, Jmax=20.0)
print(f"U=0 baseline: ifinal={int(res0.ifinal)}/{Nctl} cost={float(res0.cost_final):.3f} (init {float(terminal_cost(pt0)):.3f})", flush=True)

R = 30.0; gamma = 0.4; ITERS = int(os.environ.get("ITERS", "150")); Jmax = 20.0
Us = ilqr.initial_controls(); l_traj, K_traj = ilqr.initial_gains(); iover = ilqr.initial_iover()
best = {"ifinal": int(res0.ifinal), "cost": float(res0.cost_final), "Us": res0.Us_new, "sv": onp.asarray(res0.pert_traj)}
t0 = time.time()
for i in range(ITERS):
    res = ilqr.iterate(Us, l_traj, K_traj, R, gamma=gamma, Jmax=Jmax, iover=iover)
    iover = res.iover
    f, c = int(res.ifinal), float(res.cost_final)
    if (f, -c) > (best["ifinal"], -best["cost"]):
        best = {"ifinal": f, "cost": c, "Us": res.Us_new, "sv": onp.asarray(res.pert_traj)}
    Us = best["Us"] if (i % 5 == 4) else res.Us_new
    l_traj, K_traj = res.l_traj, res.K_traj
    if i % 20 == 0:
        print(f"  i={i:>3d} ifinal={f:>4d} cost={c: .3f}", flush=True)
print(f"elapsed {time.time()-t0:.1f}s  best ifinal={best['ifinal']}/{Nctl} cost={best['cost']:.3f}", flush=True)

sv = best["sv"]; ifin = best["ifinal"]


def ray_boundary(pt, K=240):
    th = onp.linspace(0, 2 * onp.pi, K, endpoint=False)
    dd = jnp.asarray(onp.stack([onp.cos(th), onp.sin(th)], 1))

    def g_ray(d):
        lo, hi = 0.0, 6.0
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            inside = pt.g(pt.ox + mid * d) <= pt.y
            lo = jnp.where(inside, mid, lo); hi = jnp.where(inside, hi, mid)
        return pt.ox + 0.5 * (lo + hi) * d
    return onp.asarray(jax.vmap(g_ray)(dd))


bd0 = jnp.asarray(ray_boundary(to_banana(pt0), 240))
nsub = 12; sub = dt / nsub


def roll_one(x):
    out, c = [], x
    for _ in range(max(ifin, 1)):
        for _ in range(nsub):
            c = c + sub * sys.f(0.0, c)
        out.append(c)
    return jnp.stack(out)


rolled = jax.vmap(roll_one)(bd0)
worst_tube = -onp.inf
for k in range(1, ifin + 1):
    pt, _ = ilqr._unflatten(jnp.asarray(sv[k]))
    ptb = to_banana(pt)
    pts_k = rolled[:, k - 1, :]
    worst_tube = max(worst_tube, float(jnp.max(jax.vmap(lambda x: ptb.g(x) - ptb.y)(pts_k))))
print(f"worst MOVING-TUBE MC violation = {worst_tube:+.2e}  (<=0 == sound)", flush=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable; from matplotlib.colors import Normalize
fig, ax = plt.subplots(figsize=(7, 7))
idx = sorted(set(onp.linspace(0, ifin, 9).astype(int)))
norm = Normalize(0, ifin if ifin > 0 else 1)
for k in idx:
    pt, _ = ilqr._unflatten(jnp.asarray(sv[k]))
    b = ray_boundary(to_banana(pt), 300); b = onp.vstack([b, b[:1]])
    ax.plot(b[:, 0], b[:, 1], color=plt.cm.viridis(norm(k)), lw=2.4 if k == 0 else 1.6)
for k in idx[1:]:
    ax.scatter(onp.asarray(rolled[:, k - 1, 0]), onp.asarray(rolled[:, k - 1, 1]), s=0.5, color="r", alpha=0.4)
ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-4, 4)
ax.set_title(f"GramNormotope linear-control iLQR moving tube from banana (scaled drift)\n"
             f"ifinal={ifin}/{Nctl}, cost {float(terminal_cost(pt0)):.2f}->{best['cost']:.2f}, tube MC {worst_tube:+.1e}", fontsize=9)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.046, label="embedding step")
out = "/tmp/gram_lin3.png"; plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}", flush=True)
