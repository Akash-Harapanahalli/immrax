"""Symplectic vs non-symplectic integration of the adjoint embedding.

Each system flows a box under the polytope AdjointEmbedding, integrated four
ways: explicit Euler vs symplectic Euler (left panel) and classic RK4 vs
symplectic RK4 (right panel). Boundary samples of the box are pushed through
the same explicit map the center follows.

The symplectic frame is the exact discrete adjoint of the state map. Integrating
alpha' = -alpha A with the same explicit scheme instead gives alpha_{k+1} =
alpha_k R(-hA), off the exact adjoint alpha_k R(hA)^{-1} by the matrix
D = R(hA) R(-hA): I - h^2 A^2 for Euler, I + O(h^6) for RK4.

Systems (one figure each, symplectic_drift_<name>.{pdf,svg}):
  oscillator  x'' = -x. A^2 = -I, so D is a scalar: the non-symplectic set is
              only rescaled (Euler shrinks: unsound; RK4 grows: conservative).
  damped      x'' = -x - c x'. A^2 is not scalar: D also rotates the frame.
  node        a non-normal stable node. D shears the frame.
  pendulum    theta'' = -sin(theta). Nonlinear: mjacM is no longer exact, so
              even the symplectic sets only enclose the samples.

For linear systems the symplectic set is exactly the image of the box: the
samples sit on its boundary to machine precision. The true (continuous-time)
flow is drawn in gray: the annulus the oscillator's box sweeps, else the image
of the box boundary under a finely resolved RK4 flow. A table of worst sample
violations per snapshot is printed on stdout.
"""

import pathlib
import shutil
from typing import ClassVar

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.patches import Annulus
from scipy.special import ellipk

import immrax as irx
from immrax.parametric import AdjointEmbedding, Polytope
from immrax.parametric.symplectic_rk import get_tableau

jax.config.update("jax_enable_x64", True)
plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,
        "font.family": "serif",  # Computer Modern
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 18,
    }
)

HERE = pathlib.Path(__file__).parent


class RK4(diffrax.AbstractERK):
    """Classic 4-stage RK4 (fixed step only: no error estimate)."""

    tableau: ClassVar[diffrax.ButcherTableau] = diffrax.ButcherTableau(
        a_lower=(onp.array([0.5]), onp.array([0.0, 0.5]), onp.array([0.0, 0.0, 1.0])),
        b_sol=onp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        b_error=onp.zeros(4),
        c=onp.array([0.5, 0.5, 1.0]),
    )
    interpolation_cls: ClassVar = (
        diffrax.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 4


# ------------------------------------------------------------------ systems
class Oscillator(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([x[1], -x[0]])


C_DAMP = 0.1


class Damped(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([x[1], -x[0] - C_DAMP * x[1]])


A_NODE = onp.array([[-0.5, 2.0], [0.0, -1.0]])


class Node(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.asarray(A_NODE) @ x


class Pendulum(irx.System):
    xlen: ClassVar[int] = 2

    def f(self, t, x):
        return jnp.array([x[1], -jnp.sin(x[0])])


THETA0 = 1.0
# Rotating systems overlay revolutions at fixed phases: `period` sets the
# revolution, `per_rev` the (Euler, RK4) steps per revolution. The node has no
# revolution: `steps` and `snap_ts` instead. The steps differ between panels
# because the pairing drift is O(h) per unit time for Euler but O(h^5) for RK4:
# each panel uses a step where its drift is visible.
SYSTEMS = [
    dict(
        name="oscillator",
        sys=Oscillator(),
        center=[1.0, 0.0],
        half=[0.25, 0.25],
        annulus=True,
        period=2 * onp.pi,
        period_tex=r"2\pi",
        per_rev=(120, 6),
        revs=3,
    ),
    dict(
        name="damped",
        sys=Damped(),
        center=[1.0, 0.0],
        half=[0.25, 0.25],
        period=2 * onp.pi / onp.sqrt(1 - C_DAMP**2 / 4),
        period_tex="T",
        per_rev=(120, 6),
        revs=3,
    ),
    dict(
        name="node",
        sys=Node(),
        center=[0.0, 1.0],
        half=[0.25, 0.25],
        steps=(0.05, 0.5),
        snap_ts=[0.0, 0.5, 1.0, 1.5, 2.0],
    ),
    # Small box, two revolutions: the pendulum's shear outruns the box frame,
    # and symplectic Euler's enclosure blows up in the third revolution.
    dict(
        name="pendulum",
        sys=Pendulum(),
        center=[THETA0, 0.0],
        half=[0.02, 0.02],
        period=4 * float(ellipk(onp.sin(THETA0 / 2) ** 2)),
        period_tex="T",
        per_rev=(120, 6),
        revs=2,
    ),
]
PHASES = 6  # snapshots per revolution, 60 deg apart; revolutions overlay in place
REV_ALPHA = [0.25, 0.55, 1.0]  # later revolutions more opaque (last len(revs))
FILL = 0.3  # fill opacity of a revolution-3 box; earlier ones scale by REV_ALPHA
TRUE_SUB = 20  # RK4 substeps per step for the true flow


# ------------------------------------------------------------------ helpers
def box_boundary(center, half, per_side):
    """Counterclockwise samples of the box boundary (corners included)."""
    s = onp.linspace(-1.0, 1.0, per_side, endpoint=False)
    one = onp.ones_like(s)
    unit = onp.concatenate(
        [
            onp.stack([s, -one], 1),
            onp.stack([one, s], 1),
            onp.stack([-s, one], 1),
            onp.stack([-one, -s], 1),
        ]
    )
    return jnp.asarray(center + half * unit)


def rk_map(f, x, h, tab):
    A, b, _ = get_tableau(tab)
    ks = []
    for i in range(len(b)):
        xi = x
        for j in range(i):
            if A[i, j] != 0.0:
                xi = xi + h * A[i, j] * ks[j]
        ks.append(f(0.0, xi))
    return x + h * sum(b[i] * ks[i] for i in range(len(b)))


def roll(f, X, h, tab, N, sub=1):
    """States of the points X under N steps of the explicit map (sub substeps
    each), stacked (len(X), N+1, 2)."""

    def one(x):
        def step(c, _):
            for _ in range(sub):
                c = rk_map(f, c, h / sub, tab)
            return c, c

        _, xs = jax.lax.scan(step, x, None, length=N)
        return jnp.vstack([x, xs])

    return onp.asarray(jax.jit(jax.vmap(one))(X))


def box_at(ys, k):
    """Vertices (closed loop) and polytope of the 2D box-frame set at step k."""
    pt = Polytope(ys.ox[k], ys.alpha[k], ys.y[k])
    lo, up = pt.ly, pt.uy
    Z = jnp.array([[lo[0], lo[1]], [up[0], lo[1]], [up[0], up[1]], [lo[0], up[1]]])
    V = pt.ox + Z @ jnp.linalg.inv(pt.alpha).T
    return onp.vstack([V, V[:1]]), pt


def area(V):
    """Shoelace area of a closed polygon."""
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(onp.dot(x[:-1], y[1:]) - onp.dot(x[1:], y[:-1]))


def violation(pt, X):
    """Worst constraint violation of points X, relative to the set's width."""
    z = (X - pt.ox) @ pt.alpha.T
    return float(jnp.max(jnp.maximum(pt.ly - z, z - pt.uy) / (pt.uy - pt.ly)))


# ------------------------------------------------------------------- figures
def run(cfg):
    sys = cfg["sys"]
    center, half = onp.asarray(cfg["center"]), onp.asarray(cfg["half"])
    ix0 = irx.icentpert(jnp.asarray(center), jnp.asarray(half))
    pt0 = Polytope.from_interval(ix0)
    emb = AdjointEmbedding(sys, jnp.linalg.inv(pt0.alpha), jnp.zeros((0, 2)))
    samples0 = box_boundary(center, half, 12)
    dense0 = box_boundary(center, half, 100)
    rotating = "period" in cfg

    panels = [
        ("Euler", "euler", "symplectic", "euler"),
        ("RK4", RK4(), "symplectic_rk4", "rk4"),
    ]
    # equal aspect: a wide, short system would leave a tall gap under the title
    fig, axs = plt.subplots(1, 2, figsize=(12, 7 if rotating else 5.5))
    print(
        f"\n=== {cfg['name']}: worst sample violation, as a fraction of the set "
        "width\n(> 0: samples escape the set; < 0: the set is strictly larger)"
    )
    for i, (ax, (name, explicit, symplectic, tab)) in enumerate(zip(axs, panels)):
        if rotating:
            per_rev = cfg["per_rev"][i]
            h = float(cfg["period"]) / per_rev  # python float: jit-static
            N = per_rev * cfg["revs"]
            # (step, opacity) per snapshot: PHASES per revolution, overlaid
            snaps = [
                ((r * PHASES + j) * (per_rev // PHASES), al)
                for r, al in enumerate(REV_ALPHA[-cfg["revs"] :])
                for j in range(PHASES)
            ]
            dt_tex = rf"{cfg['period_tex']}/{per_rev}"
            span_tex = (
                rf"{cfg['revs']}{cfg['period_tex']}"
                if cfg["period_tex"] != r"2\pi"
                else rf"{2 * cfg['revs']}\pi"
            )
        else:
            h = cfg["steps"][i]
            N = int(round(max(cfg["snap_ts"]) / h))
            snaps = [(int(round(t / h)), 1.0) for t in cfg["snap_ts"]]
            dt_tex, span_tex = f"{h:g}", f"{max(cfg['snap_ts']):g}"
        TF = float(N * h)

        explicit_ys = emb.compute_reachset(0.0, TF, pt0, dt=h, solver=explicit).ys[0]
        sympl_ys = emb.compute_reachset(0.0, TF, pt0, dt=h, solver=symplectic).ys[0]
        xs = roll(sys.f, samples0, h, tab, N)

        if cfg.get("annulus"):
            # The rotation sweeps the box through the annulus between its
            # nearest point and its farthest corner.
            lo0, up0 = center - half, center + half
            r_in = float(onp.linalg.norm(onp.clip(0.0, lo0, up0)))
            r_out = float(
                onp.max(onp.linalg.norm(box_boundary(center, half, 1), axis=1))
            )
            ax.add_patch(
                Annulus(
                    (0.0, 0.0),
                    r_out,
                    r_out - r_in,
                    color="0.5",
                    alpha=0.2,
                    lw=0,
                    zorder=0,
                )
            )
        else:
            true = roll(sys.f, dense0, h, "rk4", N, sub=TRUE_SUB)
            for k, al in snaps:
                ax.fill(
                    true[:, k, 0], true[:, k, 1], color="0.5", alpha=0.2, lw=0, zorder=0
                )

        rows = []
        for k, al in snaps:
            Vs, ps = box_at(sympl_ys, k)
            Ve, pe = box_at(explicit_ys, k)
            # larger set underneath, so the smaller one stays visible on top
            pair = sorted(
                [(Vs, "tab:blue"), (Ve, "tab:orange")], key=lambda p: -area(p[0])
            )
            for z, (V, color) in enumerate(pair, start=1):
                ax.fill(V[:, 0], V[:, 1], color=color, alpha=FILL * al, lw=0, zorder=z)
                ax.plot(V[:, 0], V[:, 1], color=color, lw=0.8, alpha=al, zorder=z)
            ax.plot(xs[:, k, 0], xs[:, k, 1], ".", color="k", ms=3, alpha=al, zorder=3)
            if not rotating:
                ax.annotate(
                    f"$t={k * h:g}$",
                    xs[:, k].mean(0),
                    color="0.3",
                    fontsize=9,
                    ha="center",
                    va="center",
                    zorder=4,
                )
            rows.append((k * h, violation(ps, xs[:, k]), violation(pe, xs[:, k])))

        per = (
            rf", $T = {cfg['period']:.3f}$"
            if rotating and cfg["period_tex"] == "T"
            else ""
        )
        ax.set_title(rf"{name}, $\Delta t = {dt_tex}$, $t \in [0, {span_tex}]${per}")
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$" if cfg["name"] != "pendulum" else r"$\theta$")
        ax.set_ylabel("$x_2$" if cfg["name"] != "pendulum" else r"$\dot\theta$")

        print(
            f"\n{name} (dt={h:.4g})\n{'t':>7}  {'symplectic':>11}  "
            f"{'non-symplectic':>14}"
        )
        for t, vs, ve in rows:
            print(f"{t:7.2f}  {vs:11.1e}  {ve:14.1e}")

    # One legend for both panels: the titles carry the integrator names. Proxy
    # artists, so it survives whatever each panel happened to draw.
    handles = [
        axs[0].fill([], [], color="0.5", alpha=0.2, lw=0)[0],
        axs[0].fill([], [], fc=(0.12, 0.47, 0.71, FILL), ec="tab:blue", lw=0.8)[0],
        axs[0].fill([], [], fc=(1.0, 0.5, 0.05, FILL), ec="tab:orange", lw=0.8)[0],
        axs[0].plot([], [], ".", color="k", ms=5)[0],
    ]
    labels = [
        "true flow (continuous time)",
        "symplectic",
        "non-symplectic",
        "sampled points",
    ]
    if rotating:
        for r, al in enumerate(REV_ALPHA[-cfg["revs"] :]):
            handles.append(axs[0].plot([], [], color="0.2", lw=4, alpha=al)[0])
            labels.append(f"revolution {r + 1}")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        framealpha=0.9,
    )

    fig.tight_layout()
    for ext in ["pdf", "svg"]:
        out = HERE / f"symplectic_drift_{cfg['name']}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)


for cfg in SYSTEMS:
    run(cfg)
