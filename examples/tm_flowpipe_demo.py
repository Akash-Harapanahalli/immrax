"""Demo of TaylorModel flowpipe integration (validated_integ2).

Plots the TM flowpipe for the Van der Pol oscillator and the pendulum,
showing interval-hull boxes in phase space and the time traces for each
state component.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from immrax.inclusion import icentpert
from immrax.taylor.algorithms import validated_integ2


# -- Systems -----------------------------------------------------------------

def pendulum(t, x):
    """dx1 = x2,  dx2 = -sin(x1)"""
    return jnp.array([x[1], -jnp.sin(x[0])])


def vanderpol(t, x):
    """dx1 = x2,  dx2 = mu*(1 - x1^2)*x2 - x1,  mu = 1"""
    mu = 1.0
    return jnp.array([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]])


# -- Configuration ----------------------------------------------------------

systems = [
    ("Van der Pol",
     vanderpol,
     icentpert(jnp.array([2.0, 0.0]), jnp.array([0.05, 0.05])),
     2.0),
    ("Pendulum",
     pendulum,
     icentpert(jnp.array([1.0, 0.0]), jnp.array([0.05, 0.05])),
     2.0),
]

common_kwargs = dict(
    order_time=5,
    order_space=2,
    abstol=1e-8,
    maxsteps=500,
    epsilon=1e-10,
    delta=1e-6,
)

# -- Plotting helpers --------------------------------------------------------

def draw_hull_boxes(ax, sol, color="tab:blue", alpha=0.25):
    """Draw interval-hull rectangles in phase space (x1 vs x2)."""
    for tm in sol.flowpipe:
        hull = tm.interval_hull()
        lo, hi = hull.lower, hull.upper
        w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        rect = mpatches.Rectangle(
            (float(lo[0]), float(lo[1])), w, h,
            linewidth=0.4, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)


def draw_time_traces(ax, sol, dim, color="tab:blue", alpha=0.3):
    """Fill-between the lower/upper bounds of state component `dim` vs time."""
    times = sol.times[:sol.nsteps + 1]
    lo_vals = []
    hi_vals = []
    for tm in sol.flowpipe:
        hull = tm.interval_hull()
        lo_vals.append(float(hull.lower[dim]))
        hi_vals.append(float(hull.upper[dim]))
    ax.fill_between(times, lo_vals, hi_vals, color=color, alpha=alpha)


# -- Run and plot ------------------------------------------------------------

fig, axes = plt.subplots(len(systems), 3, figsize=(18, 5 * len(systems)))

for row, (sys_name, f, x0, tmax) in enumerate(systems):
    print(f"Integrating {sys_name}...")
    sol = validated_integ2(f, x0, 0.0, tmax, **common_kwargs)
    nsteps = sol.nsteps
    status = "OK" if sol.success else "FAILED"
    print(f"  {nsteps} steps, success={sol.success}")

    # Phase portrait
    ax_phase = axes[row, 0]
    draw_hull_boxes(ax_phase, sol)
    ax_phase.autoscale_view()
    ax_phase.set_xlabel("$x_1$")
    ax_phase.set_ylabel("$x_2$")
    ax_phase.set_title(f"{sys_name} — phase portrait")
    ax_phase.set_aspect("equal", adjustable="datalim")
    ax_phase.annotate(
        f"{nsteps} steps, {status}",
        xy=(0.02, 0.96), xycoords="axes fraction", va="top", fontsize=8,
        color="green" if sol.success else "red",
    )

    # x1 vs time
    ax_x1 = axes[row, 1]
    draw_time_traces(ax_x1, sol, 0)
    ax_x1.set_xlabel("$t$")
    ax_x1.set_ylabel("$x_1$")
    ax_x1.set_title(f"{sys_name} — $x_1(t)$")

    # x2 vs time
    ax_x2 = axes[row, 2]
    draw_time_traces(ax_x2, sol, 1)
    ax_x2.set_xlabel("$t$")
    ax_x2.set_ylabel("$x_2$")
    ax_x2.set_title(f"{sys_name} — $x_2(t)$")

plt.tight_layout()
plt.savefig("tm_flowpipe_demo.png", dpi=150)
print("\nSaved tm_flowpipe_demo.png")
plt.show()
