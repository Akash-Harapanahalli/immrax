"""Van der Pol (ARCH-COMP, mu = 1) with a Chebyshev polytope.

Initial box x in [1.25, 1.55], y in [2.35, 2.45]; spec y < 2.75 up to t = 7.
Degree-3 Chebyshev faces, residual bounds on a 32 x 32 cell partition of the chart,
Heun discretization of the embedding with dt = 0.1. The true set is the flowed
boundary of the box under the same Heun map.

Produces vdp.{pdf,png}: the sets every 0.2 (every step in the zoom), and the
certified bound on y against the spec.
"""

import pathlib
import time
from typing import ClassVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

import immrax as irx

jax.config.update("jax_enable_x64", True)
HERE = pathlib.Path(__file__).parent


class VanDerPol(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([x[1], (1 - x[0] ** 2) * x[1] - x[0]])


dt, tf, spec = 0.1, 7.0, 2.75
N = int(round(tf / dt))
sys = VanDerPol()
ix0 = irx.icentpert(jnp.array([1.4, 2.4]), jnp.array([0.15, 0.05]))
pt0 = irx.ChebyshevPolytope.from_interval(ix0, m=3)
emb = irx.ChebyshevPolytopeEmbedding(sys, bound="cells", G=32)
emb._initialize(pt0)
reach = jax.jit(
    lambda p: emb.compute_reachset(0.0, tf, p, dt=dt, solver="symplectic_heun").ys[0]
)

t0 = time.time()
jax.block_until_ready(reach(pt0))
t_first = time.time() - t0
t0 = time.time()
ys = jax.block_until_ready(reach(pt0))
t_run = time.time() - t0
pt_at = lambda k: irx.ChebyshevPolytope(ys.ox[k], ys.H[k], ys.alpha[k], ys.y[k])

# true set: the flowed boundary of the box
f = lambda x: sys.f(0.0, x)
F = lambda x: x + dt / 2 * (f(x) + f(x + dt * f(x)))
s = jnp.linspace(0, 1, 500, endpoint=False)[:, None]
c = (
    jnp.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * (ix0.upper - ix0.lower) / 2
    + (ix0.upper + ix0.lower) / 2
)
bnd = jnp.concatenate([(1 - s) * c[i] + s * c[(i + 1) % 4] for i in range(4)])
B = onp.asarray(
    jax.vmap(
        lambda x: jnp.vstack(
            [x, jax.lax.scan(lambda z, _: (F(z),) * 2, x, None, length=N)[1]]
        )
    )(bnd)
)

y_cert = onp.array([float(pt_at(k).iover().upper[1]) for k in range(N + 1)])
y_true = B[:, :, 1].max(0)
g = jax.jit(lambda pt, X: jax.vmap(pt.g)(X))

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(17, 7.5), gridspec_kw=dict(width_ratios=[1.35, 1])
)
axz = ax.inset_axes([0.47, 0.41, 0.31, 0.34])
ZX, ZY = (1.72, 2.18), (-0.78, 0.22)
for k in range(N + 1):
    t = round(k * dt, 1)
    Bk = B[:, k]
    main = k % 2 == 0
    zoom = ZX[0] < Bk[:, 0].mean() < ZX[1] and ZY[0] < Bk[:, 1].mean() < ZY[1]
    axes = ([ax] if main else []) + ([axz] if zoom else [])
    if not axes:
        continue
    io = pt_at(k).iover()
    lo, hi = onp.asarray(io.lower), onp.asarray(io.upper)
    pad = 0.05 * (hi - lo)
    Xg, Yg = onp.meshgrid(
        *(onp.linspace(lo[i] - pad[i], hi[i] + pad[i], 1000) for i in range(2))
    )
    Z = onp.asarray(
        g(pt_at(k), jnp.asarray(onp.stack([Xg.ravel(), Yg.ravel()], 1)))
    ).reshape(Xg.shape)
    for a in axes:
        a.contour(Xg, Yg, Z, levels=[0.0], colors=["tab:red"], linewidths=0.9, zorder=4)
        a.fill(Bk[:, 0], Bk[:, 1], color="0.2", alpha=0.5, lw=0, zorder=3)
    if zoom and main:
        axz.annotate(
            f"t={t:g}",
            Bk[Bk[:, 0].argmin()],
            xytext=(-5, -3),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="0.3",
        )
    if main and t == int(t):
        ax.annotate(
            f"t={t:g}",
            Bk.mean(0),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=9,
            color="0.25",
        )

ax.axhline(spec, color="tab:purple", ls="--", lw=1.2)
axz.set_xlim(*ZX)
axz.set_ylim(*ZY)
axz.tick_params(labelsize=7)
axz.grid(alpha=0.3)
ax.indicate_inset_zoom(axz, edgecolor="0.4")
ax.plot([], [], color="tab:red", lw=1.2, label="certified set")
ax.fill([], [], color="0.2", alpha=0.5, label="true set")
ax.plot([], [], color="tab:purple", ls="--", label=f"spec $y < {spec}$")
ax.set_xlim(-2.35, 2.25)
ax.set_ylim(-2.95, 3.0)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc="center", bbox_to_anchor=(0.25, 0.5), fontsize=9)
ax.grid(alpha=0.3)
ax.set_title(
    f"Van der Pol ($\\mu=1$): Chebyshev polytope, m=3, 32$^2$ cells, Heun dt={dt}"
)

tt = onp.arange(N + 1) * dt
ax2.plot(tt, y_cert, color="tab:red", label="certified bound on $y$")
ax2.plot(tt, y_true, color="k", label="true max $y$")
ax2.axhline(spec, color="tab:purple", ls="--", lw=1.2, label=f"spec {spec}")
ax2.set_xlabel("$t$")
ax2.set_ylabel("$y$")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)
ins = ax2.inset_axes([0.55, 0.12, 0.42, 0.35])
peak = (tt > 6.2) & (tt < 6.9)
ins.plot(tt[peak], y_cert[peak], color="tab:red")
ins.plot(tt[peak], y_true[peak], color="k")
ins.axhline(spec, color="tab:purple", ls="--")
ins.grid(alpha=0.3)
ins.set_title("near the peak", fontsize=8)
ax2.set_title(
    f"certified max y = {y_cert.max():.3f} (true {y_true.max():.3f}); JIT {t_first - t_run:.1f}s, run {t_run:.2f}s"
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"vdp.{ext}", dpi=110)
print(
    f"certified max y {y_cert.max():.4f} (spec {spec}, true {y_true.max():.4f}); JIT {t_first - t_run:.1f}s, run {t_run:.2f}s"
)
