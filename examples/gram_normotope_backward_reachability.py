r"""Backward reachability for the reverse-VDP ROA on GramNormotope (section figure).

Running the embedding BACKWARD grows the invariant set instead of contracting it:
``dy/dtau = -ydot`` grows the radius while the boundary drift ``E = max_bnd Vdot <= 0``;
growth asymptotes at the ``E=0`` cap (the ROA boundary) -- the dual of the forward flow
asymptoting at the equilibrium. Three panels:

  (A) **No barrier search.** Grow a feasible set from a tiny Lyapunov ball (offset only)
      straight to a usable ROA -- no optimization at all.
  (B) **The two embedding equilibria.** With a fixed shape, forward contracts to the
      origin (y->0) and backward grows to the ROA boundary (y->cap); both asymptote.
  (C) **Backward reach as a warm start.** Folding the grown set into the barrier solver
      beats the generic init -- bigger, more consistent ROA.

Run (after the main example has cached R_roa)::

    python examples/gram_normotope_reverse_vdp_roa.py        # writes R_roa_m*.npy
    python examples/gram_normotope_backward_reachability.py
"""

# ruff: noqa: E402  -- x64 must be enabled before immrax/JAX is imported

import os

from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
from scipy.linalg import solve_continuous_lyapunov, cholesky

import examples.gram_normotope_reverse_vdp_roa as ex
from examples.gram_normotope_reverse_vdp_roa import (
    build, to_gn, ray_boundary, mc_violation, optimize,
    BASIN, LC, N, LSCALE, CACHE, MU,
)

M3 = 3


def lyapunov_ball(M):
    """Tiny feasible seed: Lyapunov metric in the linear block (placed at the taylax lin_idx
    rows/cols), unit higher blocks. A literal identity ball is marginal here (mu_2(Df(0))=0,
    non-normal origin), so the seed uses the contracting Lyapunov metric."""
    Pn = M["P"]
    _, lin_idx = ex._gram_basis(N, M["m"])
    Lc = jnp.asarray(cholesky(solve_continuous_lyapunov(onp.array([[0., -1.], [1., -MU]]).T, -onp.eye(2))))
    return (jnp.eye(Pn) * 1.0).at[onp.ix_(lin_idx, lin_idx)].set(Lc * LSCALE)


def grow_family(M, R, y0=0.03, dtau=0.1, T=6000):
    """Offset-only backward grow of a fixed shape; record (taus, [(R,y),...]) to the E=0 cap."""
    y = y0
    fam = [(R, y)]
    taus = [0.0]
    for k in range(T):
        yd = float(M["ydot"](R, y))
        if yd * 2.0 * y > 1e-6 or (-yd) < 1e-6:
            break
        y = y + dtau * (-yd)
        fam.append((R, y))
        taus.append((k + 1) * dtau)
    return onp.array(taus), fam


def forward_contract(M, R, dt=0.05, T=4000):
    """Offset-only forward contraction of a fixed shape from y=1; record (taus, ys)."""
    y = 1.0
    taus = [0.0]
    ys = [1.0]
    for k in range(T):
        yd = float(M["ydot"](R, y))
        if y < 1e-3:
            break
        y = max(y + dt * yd, 1e-4)
        taus.append((k + 1) * dt)
        ys.append(y)
    return onp.array(taus), onp.array(ys)


def main():
    M = build(M3)

    # (A) backward reach with no barrier search: grow the Lyapunov ball
    Rseed = lyapunov_ball(M)
    tsA, famA = grow_family(M, Rseed)
    aA = float(M["area"](jnp.asarray(famA[-1][0]), famA[-1][1]))
    wvA = max(mc_violation(to_gn(M, jnp.asarray(famA[j][0]), famA[j][1]))
              for j in onp.linspace(0, len(famA) - 1, 6).astype(int))

    # (B) duality with the SAME fixed Lyapunov shape (offset-only both ways, both clean):
    #     backward (reuse panel A) grows to the E=0 cap; forward (from y=1 < cap) contracts to origin.
    tb, yb = tsA, onp.array([f[1] for f in famA])
    ycap = float(yb[-1])
    tf, yf = forward_contract(M, Rseed)

    # (C) warm-start comparison (self-contained: optimize both inits, cache each).
    #   generic init: Lyapunov + 0.7I + shrink-to-feasibility (what optimize() does internally)
    gpath = f"{CACHE}/R_generic_m{M3}.npy"
    if os.path.exists(gpath):
        Rgen = jnp.asarray(onp.load(gpath))
    else:
        _, lin_idx = ex._gram_basis(N, M3)
        Lc = jnp.asarray(cholesky(solve_continuous_lyapunov(onp.array([[0., -1.], [1., -MU]]).T, -onp.eye(2))))
        Rg0 = (jnp.eye(M["P"]) * 0.7).at[onp.ix_(lin_idx, lin_idx)].set(Lc * (LSCALE / 0.9))
        Rgen, _ = optimize(M, Rg0, 2500)
        onp.save(gpath, onp.asarray(Rgen))
    #   backward-reach init: grow the Lyapunov ball, fold y into R, then optimize
    wpath = f"{CACHE}/R_warmstart_m{M3}.npy"
    if os.path.exists(wpath):
        Rwarm = jnp.asarray(onp.load(wpath))
    else:
        Rw0 = (Rseed / ycap) * 1.03  # ycap = the Lyapunov ball's backward-grow cap (from panel B)
        Rwarm, _ = optimize(M, Rw0, 2500)
        onp.save(wpath, onp.asarray(Rwarm))
    ag = float(M["area"](Rgen, 1.0))
    ar = float(M["area"](Rwarm, 1.0))
    print(f"(A) no-barrier-search {100*aA/BASIN:.0f}% (worst MC {wvA:+.0e})  "
          f"(C) generic init {100*ag/BASIN:.0f}% vs backward-reach init {100*ar/BASIN:.0f}%", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 6))

    axA.plot(LC[:, 0], LC[:, 1], "k--", lw=1.4, label="limit cycle")
    idx = onp.unique(onp.linspace(0, len(famA) - 1, 9).astype(int))
    for c, j in zip(plt.cm.plasma(onp.linspace(0, 1, len(idx))), idx):
        b = ray_boundary(to_gn(M, jnp.asarray(famA[j][0]), famA[j][1]), 240)
        pp = onp.vstack([b, b[:1]])
        axA.plot(pp[:, 0], pp[:, 1], color=c, lw=1.5)
    axA.set_aspect("equal")
    axA.set_xlim(-3, 3)
    axA.set_ylim(-4.2, 4.2)
    axA.legend(fontsize=8)
    axA.set_title(f"(A) backward reach, NO barrier search\ngrow from a tiny ball -> {100*aA/BASIN:.0f}% of basin (MC {wvA:+.0e})")

    axB.plot(tb, yb, "C2-", lw=2, label="backward grow: origin -> ROA boundary")
    axB.plot(tf, yf, "C0-", lw=2, label="forward contract: ROA -> equilibrium")
    axB.axhline(ycap, color="0.6", ls=":", lw=1)
    axB.axhline(0.0, color="0.6", ls=":", lw=1)
    axB.annotate("ROA boundary (E=0)", xy=(0.5, ycap), fontsize=7, va="bottom", color="0.4")
    axB.annotate("equilibrium", xy=(0.5, 0.0), fontsize=7, va="bottom", color="0.4")
    axB.set_xlabel("embedding time tau")
    axB.set_ylabel("radius y (fixed Lyapunov shape)")
    axB.set_xlim(0, max(tb[-1], tf[-1]) * 1.02)
    axB.set_ylim(-0.05, ycap * 1.1)
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)
    axB.set_title("(B) the two embedding equilibria\nforward -> origin, backward -> ROA cap (offset only)")

    axC.plot(LC[:, 0], LC[:, 1], "k--", lw=1.4, label="limit cycle")
    for R, col, lab in [(Rgen, "C1", f"generic init -> {100*ag/BASIN:.0f}%"),
                        (Rwarm, "C3", f"backward-reach init -> {100*ar/BASIN:.0f}%")]:
        b = ray_boundary(to_gn(M, R, 1.0), 300)
        pp = onp.vstack([b, b[:1]])
        axC.plot(pp[:, 0], pp[:, 1], col, lw=2, label=lab)
    axC.set_aspect("equal")
    axC.set_xlim(-3, 3)
    axC.set_ylim(-4.2, 4.2)
    axC.legend(fontsize=8)
    axC.set_title("(C) backward reach as a warm start\nbigger, more consistent ROA (no shrink-to-feasibility, no ladder)")

    plt.suptitle(f"Backward reachability for the reverse-VDP ROA (m={M3}): construction, duality, and as a warm start", fontsize=13)
    out = f"{CACHE}/backward_reachability.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
