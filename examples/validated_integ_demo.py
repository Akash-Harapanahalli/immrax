"""Demo of validated_integ and validated_integ2 on simple ODE systems."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from immrax.inclusion import icentpert
from immrax.taylor.algorithms import validated_integ, validated_integ2
from immrax.utils import run_times


# ── Systems ──────────────────────────────────────────────────────────────

def pendulum(t, x):
    """ẋ₁ = x₂, ẋ₂ = -sin(x₁)"""
    return jnp.array([x[1], -jnp.sin(x[0])])


def vanderpol(t, x):
    """ẋ₁ = x₂, ẋ₂ = μ(1 - x₁²)x₂ - x₁,  μ = 1"""
    mu = 1.0
    return jnp.array([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]])


# ── Configuration ────────────────────────────────────────────────────────

systems = [
    ("Pendulum",    pendulum,  icentpert(jnp.array([1.0, 0.0]), jnp.array([0.05, 0.05])), 3.0),
    ("Van der Pol", vanderpol, icentpert(jnp.array([2.0, 0.0]), jnp.array([0.05, 0.05])), 7.0),
]

static_vi  = ["f", "order_time", "adaptive", "maxsteps"]
static_vi2 = ["f", "order_time", "adaptive", "maxsteps",
               "epsilon", "delta", "validatesteps"]

algorithms = [
    ("Picard-Lindelof", validated_integ,  static_vi,
     dict(order_time=6, abstol=1e-10, maxsteps=5000)),
    ("eps-inflation",   validated_integ2, static_vi2,
     dict(order_time=6, abstol=1e-10, maxsteps=5000,
          epsilon=1e-10, delta=1e-6)),
]

N_BENCH = 10   # number of timing runs

# ── Run, benchmark, and plot ─────────────────────────────────────────────

fig, axes = plt.subplots(len(systems), 2, figsize=(14, 5 * len(systems)))

for row, (sys_name, f, x0, tmax) in enumerate(systems):
    for col, (alg_name, alg_fn, snames, kwargs) in enumerate(algorithms):
        ax = axes[row, col]
        print(f"Running {sys_name} with {alg_name}...")

        jit_alg = jax.jit(alg_fn, static_argnames=snames)
        sol, times = run_times(N_BENCH, jit_alg, f, x0, 0.0, tmax, **kwargs)

        nsteps = int(sol.nsteps)
        compile_ms = float(times[0]) * 1000
        warm_avg_ms = float(times[1:].mean()) * 1000
        warm_std_ms = float(times[1:].std()) * 1000
        print(f"  {nsteps} steps, success={bool(sol.success)}")
        print(f"  compile+run: {compile_ms:.0f}ms")
        print(f"  warm avg:    {warm_avg_ms:.1f} +/- {warm_std_ms:.1f} ms"
              f"  ({N_BENCH - 1} runs)")

        # Draw interval boxes in phase space (x1 vs x2)
        for i in range(nsteps + 1):
            lo = sol.flowpipe_lo[i]
            hi = sol.flowpipe_hi[i]
            rect = mpatches.Rectangle(
                (float(lo[0]), float(lo[1])),
                float(hi[0] - lo[0]),
                float(hi[1] - lo[1]),
                linewidth=0.5, edgecolor="tab:blue", facecolor="tab:blue",
                alpha=0.3,
            )
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"{sys_name} -- {alg_name}")
        ax.set_aspect("equal", adjustable="datalim")
        status = "OK" if bool(sol.success) else "FAILED"
        ax.annotate(
            f"{nsteps} steps, {status}\n"
            f"compile: {compile_ms:.0f}ms, warm: {warm_avg_ms:.1f}ms",
            xy=(0.02, 0.96), xycoords="axes fraction", va="top",
            fontsize=8, color="green" if sol.success else "red",
        )

plt.tight_layout()
plt.savefig("validated_integ_demo.png", dpi=150)
print("\nSaved validated_integ_demo.png")
plt.show()
