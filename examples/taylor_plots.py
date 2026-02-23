"""
Visualise Taylor polynomial approximations computed via ltdiff / taylor_approx.

Two figures are produced:

  Figure 1 – sin(x) on [-2π, 2π], orders p = 1, 3, 5, 7 (expansion at 0).
  Figure 2 – sin(x₀)·cos(x₁) on a 2-D grid (π/4, π/4 ± δ), orders p = 1, 3, 5,
             shown as 1-D slices along the diagonal x₀ = x₁.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import immrax as irx

jax.config.update("jax_enable_x64", True)

ORDERS = [1, 3, 5, 7]
COLORS = cm.plasma(np.linspace(0.15, 0.85, len(ORDERS)))

# ── Figure 1: sin(x) ──────────────────────────────────────────────────────────

f1 = lambda x: jnp.sin(x[0:1])
xc1 = jnp.array([0.0])
xs = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

ax = axes[0]
ax.plot(xs, jnp.sin(xs), "k-", lw=2, label="sin(x)", zorder=5)

for p, color in zip(ORDERS, COLORS):
    tensors = irx.ltdiff(f1, p)(xc1)
    approx = np.array([float(irx.taylor_approx(tensors, xc1, jnp.array([xi]))[0]) for xi in xs])
    ax.plot(xs, approx, "-", color=color, lw=1.5, label=f"p = {p}")

ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(float(xs[0]), float(xs[-1]))
ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_title("Taylor approximations of sin(x) around x = 0")
ax.legend(loc="upper right")
ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
ax.set_xticklabels(["-2π", "-π", "0", "π", "2π"])

# Error plot (log scale)
ax2 = axes[1]
for p, color in zip(ORDERS, COLORS):
    tensors = irx.ltdiff(f1, p)(xc1)
    approx = np.array([float(irx.taylor_approx(tensors, xc1, jnp.array([xi]))[0]) for xi in xs])
    error = np.abs(approx - np.sin(np.array(xs)))
    ax2.semilogy(xs, np.clip(error, 1e-16, None), "-", color=color, lw=1.5, label=f"p = {p}")

ax2.set_xlim(float(xs[0]), float(xs[-1]))
ax2.set_xlabel("x")
ax2.set_ylabel("|error|")
ax2.set_title("Approximation error (log scale)")
ax2.legend(loc="upper right")
ax2.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
ax2.set_xticklabels(["-2π", "-π", "0", "π", "2π"])

fig.tight_layout()
fig.savefig("examples/taylor_sin.png", dpi=150)
print("Saved examples/taylor_sin.png")

# ── Figure 2: sin(x₀)·cos(x₁) diagonal slice ─────────────────────────────────

f2 = lambda x: jnp.stack([jnp.sin(x[0]) * jnp.cos(x[1])])
xc2 = jnp.array([jnp.pi / 4, jnp.pi / 4])
ts = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 400)   # offset from xc2
xs2 = jnp.stack([xc2[0] + ts, xc2[1] + ts], axis=1)  # diagonal x₀ = x₁ = π/4 + t

true2 = np.array([float(f2(xi)[0]) for xi in xs2])

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

ax3 = axes2[0]
ax3.plot(ts, true2, "k-", lw=2, label="true", zorder=5)

for p, color in zip(ORDERS, COLORS):
    tensors = irx.ltdiff(f2, p)(xc2)
    approx = np.array([float(irx.taylor_approx(tensors, xc2, xi)[0]) for xi in xs2])
    ax3.plot(ts, approx, "-", color=color, lw=1.5, label=f"p = {p}")

ax3.axvline(0, color="gray", lw=0.5, ls="--")
ax3.set_xlim(float(ts[0]), float(ts[-1]))
ax3.set_ylim(-1.4, 1.4)
ax3.set_xlabel("t  (x₀ = x₁ = π/4 + t)")
ax3.set_title("Taylor approx of sin(x₀)·cos(x₁) along diagonal, xc = (π/4, π/4)")
ax3.legend(loc="upper right")

ax4 = axes2[1]
for p, color in zip(ORDERS, COLORS):
    tensors = irx.ltdiff(f2, p)(xc2)
    approx = np.array([float(irx.taylor_approx(tensors, xc2, xi)[0]) for xi in xs2])
    error = np.abs(approx - true2)
    ax4.semilogy(ts, np.clip(error, 1e-16, None), "-", color=color, lw=1.5, label=f"p = {p}")

ax4.set_xlim(float(ts[0]), float(ts[-1]))
ax4.set_xlabel("t")
ax4.set_ylabel("|error|")
ax4.set_title("Approximation error (log scale)")
ax4.legend(loc="upper right")

fig2.tight_layout()
fig2.savefig("examples/taylor_sin_cos_2d.png", dpi=150)
print("Saved examples/taylor_sin_cos_2d.png")
