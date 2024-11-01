import pickle
from typing import Literal

from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pypoman import (
    plot_polygon,
)  # TODO: Add pypoman to requirements. Need pycddlib==2.1.8.post1 for some reason
from scipy.spatial import HalfspaceIntersection

import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.inclusion import interval, mjacif
from immrax.utils import draw_iarray, run_times

# Read papers about contraction and stability

vdp_mu = 1

x0_int = irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
sim_len = 1.56
plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
# Number of subdivisions of [0, pi] to make aux vars for
# Certain values of this are not good choices, as they will generate angles theta=0 or theta=pi/2
# This will introduce dependence in the aux vars, causing problems with the JAX LP solver
N = 6
aux_vars = jnp.array(
    [
        [jnp.cos(n * jnp.pi / (N + 1)), jnp.sin(n * jnp.pi / (N + 1))]
        for n in range(1, N + 1)
    ]
)

plt.figure()


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Harmonic Oscillator"

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


class VanDerPolOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Van der Pol Oscillator"

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([vdp_mu * (x1 - 1 / 3 * x1**3 - x2), x1 / vdp_mu])


# Trajectory of unrefined system
sys = HarmOsc()  # Can use an arbitrary system here
embsys = TransformEmbedding(sys)
traj = embsys.compute_trajectory(
    0.0,
    sim_len,
    irx.i2ut(x0_int),
)

# Clean up and display results
tfinite = jnp.where(jnp.isfinite(traj.ts))
ys_clean = traj.ys[tfinite]
y_int = [irx.ut2i(y) for y in ys_clean]
for bound in y_int:
    draw_iarray(plt.gca(), bound, alpha=0.4)
plt.gcf().suptitle(f"{sys.name} with Uncertainty (No Refinement)")


def plot_refined_traj(mode: Literal["sample", "linprog"]):
    fig, axs = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
    fig.suptitle(f"{sys.name} with Uncertainty ({mode} Refinement)")
    axs = axs.reshape(-1)

    H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    for i in range(len(aux_vars)):
        # Add new refinement
        print(f"Adding auxillary variable {aux_vars[i]}")
        H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
        lifted_x0_int = interval(H) @ x0_int

        # Compute sample refined trajectory
        auxsys = AuxVarEmbedding(
            sys, H, mode=mode, num_samples=10 ** (i + 1), if_transform=mjacif
        )
        traj, time = run_times(
            1,
            auxsys.compute_trajectory,
            0.0,
            sim_len,
            irx.i2ut(lifted_x0_int),
        )
        tfinite = jnp.where(jnp.isfinite(traj.ts))
        ys_clean = traj.ys[tfinite]
        ys_int = [irx.ut2i(y) for y in ys_clean]
        print(f"\t{mode} for {i+1} aux vars took: {time}")
        print(f"\tFinal bound: \n{ys_int[-1][:2]}")
        pickle.dump(ys_int, open(f"{mode}_traj_{i}.pkl", "wb"))

        # Clean up and display results
        plt.sca(axs[i])
        axs[i].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N+1}}}$")
        for bound in ys_int:
            cons = onp.hstack(
                (
                    onp.vstack((-H, H)),
                    onp.concatenate((bound.lower, -bound.upper)).reshape(-1, 1),
                )
            )
            hs = HalfspaceIntersection(cons, bound.center[0:2])
            vertices = hs.intersections

            plot_polygon(vertices, fill=False, resize=True, color="tab:blue")


plot_refined_traj("sample")
plot_refined_traj("linprog")

print("Plotting finished")
plt.show()
