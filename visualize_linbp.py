#!/usr/bin/env python3
"""
Visualization of linbp forward linear bound propagation for univariate primitives.

Two-phase animation per primitive:
  Phase 1 — fast sweep across the domain at a constant interval width.
  Phase 2 — sweep again while the interval width oscillates (widens/shrinks).

Each panel shows:
  - Black curve     : true function f(x)
  - Blue segment    : lower linear bound  α·x + β_l  on [l, u]  (marker ticks at endpoints)
  - Red segment     : upper linear bound  α·x + β_u  on [l, u]  (marker ticks at endpoints)
  - Purple fill     : region between the two bound segments
  - Gold highlight  : current interval [l, u]

Run with:
    JAX_PLATFORMS=cpu python visualize_linbp.py

Saves:  linbp_animation.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.artist import Artist
import jax.numpy as jnp
from jax import lax

from immrax.inclusion.linbp import LinearBound, linbp_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_identity_lb(l: float, u: float) -> LinearBound:
    """1-D identity LinearBound for x ∈ [l, u]."""
    return LinearBound(
        lA=jnp.array([[1.0]]),
        lb=jnp.zeros(1),
        uA=jnp.array([[1.0]]),
        ub=jnp.zeros(1),
        l=jnp.array([l]),
        u=jnp.array([u]),
    )


def get_linear_bounds(handler, l: float, u: float, **extra_params):
    """Return (alpha_l, beta_l, alpha_u, beta_u) by calling the handler directly."""
    x = make_identity_lb(l, u)
    result = handler(x, relu_mode="adaptive", **extra_params)
    return (
        float(result.lA[0, 0]),
        float(result.lb[0]),
        float(result.uA[0, 0]),
        float(result.ub[0]),
    )


# ---------------------------------------------------------------------------
# Primitive definitions
# ---------------------------------------------------------------------------
# (label, numpy_fn, handler, (domain_l, domain_r), base_interval_width,
#  (y_lo, y_hi), extra_kwargs_for_handler)

def _relu_handler(x, *, relu_mode, **kwargs):
    """Wrap max_p as a univariate ReLU handler: max(x, 0)."""
    return linbp_registry[lax.max_p](x, jnp.zeros(1), relu_mode=relu_mode)


PRIMITIVES = [
    (
        "relu(x) = max(x,0)",
        lambda x: np.maximum(x, 0.0),
        _relu_handler,
        (-4.0, 4.0), 2.0,
        (-0.3, 4.3), {},
    ),
    (
        "sigmoid σ(x)",
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        linbp_registry[lax.logistic_p],
        (-6.0, 6.0), 2.5,
        (-0.15, 1.15), {},
    ),
    (
        "tanh(x)",
        np.tanh,
        linbp_registry[lax.tanh_p],
        (-4.0, 4.0), 2.0,
        (-1.3, 1.3), {},
    ),
    (
        "sin(x)",
        np.sin,
        linbp_registry[lax.sin_p],
        (-2 * np.pi, 2 * np.pi), np.pi * 0.85,
        (-1.6, 1.6), {},
    ),
    (
        "cos(x)",
        np.cos,
        linbp_registry[lax.cos_p],
        (-2 * np.pi, 2 * np.pi), np.pi * 0.85,
        (-1.6, 1.6), {},
    ),
    (
        "tan(x)",
        np.tan,
        linbp_registry[lax.tan_p],
        # Domain strictly inside (-π/2, π/2) ≈ (-1.5708, 1.5708)
        (-1.35, 1.35), 0.7,
        (-8.0, 8.0), {},
    ),
    (
        "x²  (integer_pow n=2)",
        lambda x: x ** 2,
        linbp_registry[lax.integer_pow_p],
        (-3.0, 3.0), 1.5,
        (-0.5, 9.5), {"y": 2},
    ),
    (
        "x³  (integer_pow n=3)",
        lambda x: x ** 3,
        linbp_registry[lax.integer_pow_p],
        (-2.5, 2.5), 1.5,
        (-16.0, 16.0), {"y": 3},
    ),
    (
        "exp(x)",
        np.exp,
        linbp_registry[lax.exp_p],
        (-3.0, 3.0), 1.5,
        (-1.0, 22.0), {},
    ),
    (
        "log1p(x)",
        np.log1p,
        linbp_registry[lax.log1p_p],
        (-0.5, 4.0), 1.2,
        (-1.5, 2.5), {},
    ),
]

# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------

N_SWEEP   = 50   # frames for phase 1: fast constant-width sweep (left → right)
N_WOBBLE  = 100  # frames for phase 2: oscillating width sweep (right → left)
N_FRAMES  = N_SWEEP + N_WOBBLE

FPS         = 30          # interactive window frame rate
SAVE_FPS    = FPS // 2    # saved mp4 is 2× slower
INTERVAL_MS = int(1000 / FPS)

OUTPUT_MP4 = "linbp_animation.mp4"


def _interval_for_frame(frame: int, w: float, dl: float, dr: float):
    """Return (l, u) for the current frame.

    Phase 1 (frame < N_SWEEP):   constant width w, sweeps left → right.
    Phase 2 (frame >= N_SWEEP):  width oscillates, centre sweeps left → right.
    """
    domain_span = dr - dl

    if frame < N_SWEEP:
        t = frame / N_SWEEP          # 0 → 1  (left → right)
        w_cur = w
    else:
        s = (frame - N_SWEEP) / N_WOBBLE   # 0 → 1  (progress through wobble)
        t = 1.0 - s                         # position: 1 → 0  (right → left)
        # Width starts at w (s=0 → sin=0) and oscillates with 2.5 cycles.
        amplitude = w * 0.80
        w_cur = w + amplitude * np.sin(2 * np.pi * 2.5 * s)
        w_cur = float(np.clip(w_cur, w * 0.12, min(w * 2.0, domain_span * 0.75)))

    w_cur = min(w_cur, domain_span * 0.92)   # never wider than 92 % of domain
    travel = domain_span - w_cur
    centre = dl + w_cur / 2 + t * travel
    return centre - w_cur / 2, centre + w_cur / 2


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def build_figure():
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle(
        "linbp Forward Linear Bound Propagation — Univariate Primitives\n"
        "blue = lower bound  |  red = upper bound  |  gold = current interval",
        fontsize=12,
    )
    axes_flat = axes.flatten()

    panel_data = []
    for i, (label, np_fn, _handler, (dl, dr), _w, (yl, yu), _ekw) in enumerate(PRIMITIVES):
        ax = axes_flat[i]
        ax.set_title(label, fontsize=10, pad=3)
        ax.set_xlim(dl, dr)
        ax.set_ylim(yl, yu)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="k", lw=0.4, alpha=0.35, zorder=0)
        ax.axvline(0, color="k", lw=0.4, alpha=0.35, zorder=0)

        # Static function curve — NaN outside y-limits to avoid ugly clipping
        xs = np.linspace(dl, dr, 600)
        ys = np_fn(xs)
        ys_display = np.where((ys > yl - 0.05) & (ys < yu + 0.05), ys, np.nan)
        ax.plot(xs, ys_display, color="black", lw=1.8, zorder=3)

        # Animated bound segments: marker='|' draws a tick at each endpoint
        lower_line, = ax.plot([], [], color="steelblue", lw=2.0, zorder=5,
                              marker="|", markersize=9, markeredgewidth=2.0,
                              label="lower bound")
        upper_line, = ax.plot([], [], color="tomato", lw=2.0, zorder=5,
                              marker="|", markersize=9, markeredgewidth=2.0,
                              label="upper bound")

        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

        panel_data.append({
            "ax":          ax,
            "lower_line":  lower_line,
            "upper_line":  upper_line,
            "fill":        None,
            "span":        None,
            "label_text":  None,
        })

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, panel_data


# ---------------------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------------------

def animate(frame: int, panel_data: list) -> list[Artist]:
    for i, (_label, _np_fn, handler, (dl, dr), w, (yl, yu), ekw) in enumerate(PRIMITIVES):
        d = panel_data[i]
        ax = d["ax"]

        # Remove previous frame's disposable artists
        for key in ("fill", "span", "label_text"):
            if d[key] is not None:
                try:
                    d[key].remove()
                except Exception:
                    pass
                d[key] = None

        l, u = _interval_for_frame(frame, w, dl, dr)

        # Gold interval highlight
        d["span"] = ax.axvspan(l, u, alpha=0.18, color="gold", zorder=1)

        try:
            al, bl, au, bu = get_linear_bounds(handler, l, u, **ekw)

            # Bound segments exactly on [l, u] — no extension
            xs_seg = np.array([l, u])
            lower_y = al * xs_seg + bl
            upper_y = au * xs_seg + bu

            d["lower_line"].set_data(xs_seg, lower_y)
            d["upper_line"].set_data(xs_seg, upper_y)

            # Fill between bounds
            xs_fill = np.linspace(l, u, 100)
            lower_fill = np.clip(al * xs_fill + bl, yl, yu)
            upper_fill = np.clip(au * xs_fill + bu, yl, yu)
            d["fill"] = ax.fill_between(
                xs_fill, lower_fill, upper_fill,
                alpha=0.22, color="mediumpurple", zorder=2,
            )

            # Phase label + interval annotation
            phase = "sweep" if frame < N_SWEEP else "wobble"
            d["label_text"] = ax.text(
                0.02, 0.97,
                f"[{l:.2f}, {u:.2f}]  α={al:.3f}  [{phase}]",
                transform=ax.transAxes,
                fontsize=6, va="top", color="dimgray",
            )

        except Exception:
            d["lower_line"].set_data([], [])
            d["upper_line"].set_data([], [])

    return [d["lower_line"] for d in panel_data] + [d["upper_line"] for d in panel_data]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    fig, panel_data = build_figure()
    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(panel_data,),
        frames=N_FRAMES,
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    print(f"Saving animation to {OUTPUT_MP4} ...")
    anim.save(
        OUTPUT_MP4,
        writer=animation.FFMpegWriter(fps=SAVE_FPS, bitrate=1800),
        dpi=120,
    )
    print("Saved.")

    plt.show()


if __name__ == "__main__":
    main()
