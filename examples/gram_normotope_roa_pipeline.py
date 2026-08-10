r"""Reverse-VDP ROA on GramNormotope -- the full THREE-step pipeline in one file.

The nonconvex region-of-attraction and its nested invariant family are built in
three distinct operations (two are the embedding system, one is a gradient solve):

  **Step 1 -- SEED (embedding, backward).** Run the embedding BACKWARD from a tiny
  Lyapunov ball at the origin with the shape held fixed: ``dy/dtau = -ydot`` grows
  the radius while the boundary drift stays ``<= 0``, asymptoting at the ``E = 0``
  cap (the largest invariant level of that fixed shape). Cheap, and it lands a
  *feasible* set -- but convex-ish, so only ~35% of the basin.

  **Step 2 -- AREA-MAX (gradient optimizer, NOT the embedding).** Maximize the
  set's *true area* over the shape ``R`` subject to a rigorous boundary drift
  ``max_{V=1} Vdot < 0`` (ray-root partition bound), via Adam interior-point on
  ``-log(area) - mu*log(-pd1)``. This is where the nonconvexity is born: the lift's
  high-degree monomials let the level set bend with the limit cycle where an
  ellipse cannot -> the 57% banana. It *must* be warm-started off the convex
  manifold (Step 1 / a degree ladder) -- the convex shape is a flat critical point
  of the area objective, so no flow or local step reaches the banana from it.

  **Step 3 -- FAMILY (embedding, forward).** From the banana, run the embedding
  FORWARD with the projected-Carleman-adjoint hypercontrol
  ``Pdot = Pi_PSD(-L^T P - P L)`` (``L`` = Carleman lift of ``Df(0)``): the
  reshaping control that contracts the nested invariant family to the origin
  (true-area logvol 2.3 -> -65 vs -16 for a fixed shape). Every snapshot is
  Monte-Carlo verified forward-invariant.

So the embedding does NOT find the ROA (Step 2, an optimizer, does -- forced by the
flat-critical-point obstruction); the embedding's roles are seeding the search
(Step 1) and turning the found ROA into the family (Step 3).

Run (Step 2/3 reuse the main example's caches)::

    python -m examples.gram_normotope_reverse_vdp_roa     # writes R_roa_m*.npy + flow caches
    python -m examples.gram_normotope_roa_pipeline
"""

# ruff: noqa: E402

import os

from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp

from examples.gram_normotope_reverse_vdp_roa import (
    build, optimize, to_gn, ray_boundary, mc_violation, flow, log_set_area,
    BASIN, LC, CACHE,
)
from examples.gram_normotope_backward_reachability import lyapunov_ball, grow_family

M3 = int(os.environ.get("ORDER", "3"))


def main():
    M = build(M3)

    # ---- Step 1: SEED -- embedding backward, fixed shape, radius grows to the cap
    Rseed = lyapunov_ball(M)
    tausB, famB = grow_family(M, Rseed)
    Rcap, ycap = jnp.asarray(famB[-1][0]), famB[-1][1]
    aSeed = float(M["area"](Rcap, ycap))
    wvSeed = max(mc_violation(to_gn(M, jnp.asarray(famB[j][0]), famB[j][1]))
                 for j in onp.linspace(0, len(famB) - 1, 6).astype(int))
    print(f"Step 1 (seed): backward grow -> {100*aSeed/BASIN:.0f}% feasible (worst MC {wvSeed:+.0e})", flush=True)

    # ---- Step 2: AREA-MAX -- gradient optimizer finds the nonconvex ROA
    # (cached by the main example via the degree ladder; else solve from the Step-1
    #  seed -- both are off-convex warm starts, the only way past the flat critical point).
    roa_path = f"{CACHE}/R_roa_m{M3}.npy"
    if os.path.exists(roa_path):
        R_roa = jnp.asarray(onp.load(roa_path))
    else:
        R_roa, _ = optimize(M, (Rcap / ycap) * 1.03, 2500)
        onp.save(roa_path, onp.asarray(R_roa))
    aROA = float(M["area"](R_roa, 1.0))
    pd_roa = float(M["pd1"](R_roa))
    print(f"Step 2 (area-max): nonconvex ROA -> {100*aROA/BASIN:.0f}% of basin, certified max bnd Vdot {pd_roa:+.4f}", flush=True)

    # ---- Step 3: FAMILY -- embedding forward, projected-Carleman-adjoint
    ts, ys, Rs, ms = flow(M, R_roa, "adjoint")
    laF, mcF = [], -onp.inf
    idxF = onp.unique(onp.linspace(0, len(Rs) - 1, 9).astype(int))
    for k in range(len(Rs)):
        gn = to_gn(M, jnp.asarray(Rs[k]), ys[k])
        laF.append(log_set_area(gn))
        if k in idxF:
            mcF = max(mcF, mc_violation(gn))
    print(f"Step 3 (family): {len(Rs)} steps, true log-area {laF[0]:.1f} -> {laF[-1]:.1f}, worst MC {mcF:+.1e}", flush=True)

    # --------------------------------------------------------------- figure (2x2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 13))

    def frame(ax):
        ax.plot(LC[:, 0], LC[:, 1], "k--", lw=1.3, label="limit cycle", zorder=1)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-4.2, 4.2)

    # Step 1: backward-grown seed family
    frame(ax1)
    idxB = onp.unique(onp.linspace(0, len(famB) - 1, 8).astype(int))
    for c, j in zip(plt.cm.plasma(onp.linspace(0.05, 0.9, len(idxB))), idxB):
        b = ray_boundary(to_gn(M, jnp.asarray(famB[j][0]), famB[j][1]), 220)
        pp = onp.vstack([b, b[:1]])
        ax1.plot(pp[:, 0], pp[:, 1], color=c, lw=1.4)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"Step 1 -- SEED (embedding, backward)\norigin ball grows, fixed shape -> {100*aSeed/BASIN:.0f}% feasible (convex-ish)")

    # Step 2: seed -> nonconvex ROA
    frame(ax2)
    b = ray_boundary(to_gn(M, Rcap, ycap), 240)
    pp = onp.vstack([b, b[:1]])
    ax2.plot(pp[:, 0], pp[:, 1], color="0.6", lw=1.6, label=f"Step-1 seed ({100*aSeed/BASIN:.0f}%)")
    b = ray_boundary(to_gn(M, R_roa, 1.0), 360)
    pp = onp.vstack([b, b[:1]])
    ax2.plot(pp[:, 0], pp[:, 1], "r-", lw=2.8, label=f"nonconvex ROA ({100*aROA/BASIN:.0f}%)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title(f"Step 2 -- AREA-MAX (gradient optimizer)\nmax area s.t. boundary Vdot<0 -> banana, cert Vdot {pd_roa:+.3f}")

    # Step 3: forward projected-adjoint nested family
    frame(ax3)
    norm = Normalize(0, ts[-1] if ts[-1] > 0 else 1)
    for k in idxF:
        b = ray_boundary(to_gn(M, jnp.asarray(Rs[k]), ys[k]), 360)
        pp = onp.vstack([b, b[:1]])
        ax3.plot(pp[:, 0], pp[:, 1], color=plt.cm.viridis(norm(ts[k])), lw=2.6 if k == 0 else 1.6)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax3, fraction=0.046, pad=0.02, label="embedding time t")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title(f"Step 3 -- FAMILY (embedding, forward)\nprojected-adjoint nested family -> origin (worst MC {mcF:+.0e})")

    # Step 3 contraction curve + the Step-1/2 area levels for reference
    ax4.plot(ts, laF, "C0-", lw=2.2, label="Step 3: forward contraction")
    ax4.axhline(onp.log(BASIN), color="0.5", ls=":", lw=1)
    ax4.annotate("basin", xy=(ts[-1] * 0.85, onp.log(BASIN)), fontsize=8, va="bottom", color="0.4")
    ax4.axhline(onp.log(aROA), color="r", ls=":", lw=1.2)
    ax4.annotate(f"Step 2 ROA ({100*aROA/BASIN:.0f}%)", xy=(ts[-1] * 0.5, onp.log(aROA)), fontsize=8, va="bottom", color="r")
    ax4.axhline(onp.log(aSeed), color="0.6", ls=":", lw=1.2)
    ax4.annotate(f"Step 1 seed ({100*aSeed/BASIN:.0f}%)", xy=(ts[-1] * 0.5, onp.log(aSeed)), fontsize=8, va="top", color="0.5")
    ax4.set_xlabel("forward embedding time t")
    ax4.set_ylabel("true set log-area")
    ax4.legend(fontsize=8, loc="lower left")
    ax4.grid(alpha=0.3)
    ax4.set_title("Step 3 contraction (true area), with Step 1/2 levels")

    plt.suptitle("Reverse-VDP ROA -- three steps: SEED (backward) -> AREA-MAX (optimize) -> FAMILY (forward)\n"
                 "the embedding seeds and flows; the optimizer finds the nonconvex ROA", fontsize=13)
    out = f"{CACHE}/roa_pipeline_m{M3}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
