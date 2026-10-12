"""Coupled Van der Pol (ARCH-COMP CVDP20, mu = 1) with a Chebyshev polytope.

Two oscillators coupled through x1 - x2; initial x_i in [1.25, 1.55], y_i in
[2.35, 2.45]; spec y_i < 2.75 up to t = 7. Degree-2 Chebyshev faces; every bound
is a support function over the set, certified by validated degree-4 SOS (needs
MOSEK: ``pip install immrax[sos]``); Heun discretization with dt = 0.1. The true
set is Monte Carlo under the same Heun map.

Produces coupled_vdp.{pdf,png}: projections onto (x1, y1) and (x1, x2) every 0.2,
and the certified max y_i (SOS over the set) against the spec.
"""

import pathlib
import time
from typing import ClassVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

import immrax as irx
from immrax.parametric.sets.chebyshev_polytope import cheb_eval, cheb_indices

jax.config.update("jax_enable_x64", True)
HERE = pathlib.Path(__file__).parent


class CoupledVanDerPol(irx.System):
    xlen: ClassVar[int] = 4

    def f(self, t, s):
        x1, y1, x2, y2 = s
        return jnp.array(
            [y1, (1 - x1**2) * y1 - 2 * x1 + x2, y2, (1 - x2**2) * y2 - 2 * x2 + x1]
        )


dt, tf, spec = 0.1, 7.0, 2.75
N = int(round(tf / dt))
sys = CoupledVanDerPol()
ix0 = irx.icentpert(
    jnp.array([1.4, 2.4, 1.4, 2.4]), jnp.array([0.15, 0.05, 0.15, 0.05])
)
pt0 = irx.ChebyshevPolytope.from_interval(ix0, m=2)
emb = irx.ChebyshevPolytopeEmbedding(sys, bound="sos")
emb._initialize(pt0)

t0 = time.time()
ys = jax.block_until_ready(
    emb.compute_reachset(0.0, tf, pt0, dt=dt, solver="symplectic_heun").ys[0]
)
t_run = time.time() - t0
pt_at = lambda k: irx.ChebyshevPolytope(ys.ox[k], ys.H[k], ys.alpha[k], ys.y[k])

# certified max y_i: SOS support function over each set
C = jnp.zeros((2, 4)).at[0, 1].set(1.0).at[1, 3].set(1.0)
y_cert = onp.array([float(jnp.max(emb.support(pt_at(k), C))) for k in range(N + 1)])

# true set: Monte Carlo under the same Heun map
f = lambda x: sys.f(0.0, x)
F = lambda x: x + dt / 2 * (f(x) + f(x + dt * f(x)))
X0 = ix0.lower + (ix0.upper - ix0.lower) * jax.random.uniform(
    jax.random.PRNGKey(0), (20000, 4)
)
MC = onp.asarray(
    jax.vmap(
        lambda x: jnp.vstack(
            [x, jax.lax.scan(lambda z, _: (F(z),) * 2, x, None, length=N)[1]]
        )
    )(X0)
)
y_true = MC[:, :, [1, 3]].max((0, 2))

# certified set: samples of its chart box kept by the faces
exps = cheb_indices(4, 2)
Z = jax.random.uniform(jax.random.PRNGKey(1), (4_000_000, 4), minval=-1, maxval=1)


@jax.jit
def set_samples(pt):
    u = jax.vmap(lambda z: pt.alpha @ cheb_eval(z, exps))(Z)
    return pt.ox + Z @ jnp.linalg.inv(pt.H).T, jnp.all((u >= pt.ly) & (u <= pt.uy), 1)


def region(ax, P, color, edge, z):
    """Filled projection of a point cloud (occupied cells of a fine histogram)."""
    lo, hi = P.min(0), P.max(0)
    pad = 0.04 * (hi - lo).max() + 1e-3
    h, xe, ye = onp.histogram2d(
        P[:, 0],
        P[:, 1],
        bins=220,
        range=[[lo[0] - pad, hi[0] + pad], [lo[1] - pad, hi[1] + pad]],
    )
    xc, yc, occ = (
        (xe[1:] + xe[:-1]) / 2,
        (ye[1:] + ye[:-1]) / 2,
        (h.T > 0).astype(float),
    )
    ax.contourf(xc, yc, occ, levels=[0.5, 1.5], colors=[color], zorder=z)
    ax.contour(xc, yc, occ, levels=[0.5], colors=[edge], linewidths=0.8, zorder=z)


fig, axs = plt.subplots(
    1, 3, figsize=(20, 6.8), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1])
)
for k in range(0, N + 1, 2):
    t = round(k * dt, 1)
    S, keep = set_samples(pt_at(k))
    S = onp.asarray(S)[onp.asarray(keep)]
    for ax, (i, j) in [(axs[0], (0, 1)), (axs[1], (0, 2))]:
        region(ax, S[:, [i, j]], "#f6b6b6", "tab:red", 1)
        region(ax, MC[:, k][:, [i, j]], "0.15", "0.15", 2)
        if t == int(t):
            ax.annotate(
                f"t={t:g}",
                MC[:, k, [i, j]].mean(0),
                xytext=(12, 8),
                textcoords="offset points",
                fontsize=8,
                color="0.3",
            )

axs[0].set_ylim(-2.9, 3.0)
for ax, (xl, yl) in [(axs[0], ("$x_1$", "$y_1$")), (axs[1], ("$x_1$", "$x_2$"))]:
    ax.set_xlim(-2.3, 3.0)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
axs[1].set_ylim(-2.3, 3.0)
axs[0].axhline(spec, color="tab:purple", ls="--", lw=1.2)
axs[0].fill([], [], color="#f6b6b6", ec="tab:red", label="certified set (projection)")
axs[0].fill([], [], color="0.15", label="true set")
axs[0].plot([], [], color="tab:purple", ls="--", label=f"spec $y_i < {spec}$")
axs[0].legend(loc="center", fontsize=8)
axs[0].set_title("oscillator 1 (oscillator 2 identical)")
axs[1].set_title("coupling projection $(x_1, x_2)$")

tt = onp.arange(N + 1) * dt
axs[2].plot(tt, y_cert, color="tab:red", label="certified max $y_i$ over the set (SOS)")
axs[2].plot(tt, y_true, color="k", label="true max $y_i$")
axs[2].axhline(spec, color="tab:purple", ls="--", lw=1.2, label=f"spec {spec}")
axs[2].set_xlabel("$t$")
axs[2].grid(alpha=0.3)
axs[2].legend(fontsize=8, loc="lower left")
ins = axs[2].inset_axes([0.55, 0.1, 0.42, 0.35])
peak = (tt > 6.0) & (tt < 7.0)
ins.plot(tt[peak], y_cert[peak], color="tab:red")
ins.plot(tt[peak], y_true[peak], color="k")
ins.axhline(spec, color="tab:purple", ls="--")
ins.grid(alpha=0.3)
ins.set_title("near the peak", fontsize=8)
axs[2].set_title(
    f"certified max y = {y_cert.max():.3f} (true {y_true.max():.3f}); run {t_run:.0f}s",
    fontsize=10,
)
fig.suptitle(
    f"Coupled Van der Pol (ARCH CVDP20, $\\mu=1$): Chebyshev polytope, m=2, SOS bounds, Heun dt={dt}"
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"coupled_vdp.{ext}", dpi=110)
print(
    f"certified max y {y_cert.max():.4f} (spec {spec}, true {y_true.max():.4f}); run {t_run:.1f}s"
)
