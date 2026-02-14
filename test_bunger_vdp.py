"""Compare BasicTMFlowpipeGenerator vs BungerTMFlowpipeGenerator on VDP."""

import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator, BungerTMFlowpipeGenerator
from immrax.taylor.algorithms.base import tx_tm_eval


class VDP(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x):
        mu = 1.0
        return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = VDP()
t0 = 0.0
tf = 7.0
ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=2)

dt = 0.05
delta = 1e-4
eps = 1e-2

# --- Basic ---
fpg_basic = BasicTMFlowpipeGenerator(sys)

@jax.jit
def gen_basic(t0, tf, tmx, dt, delta, eps):
    return fpg_basic.generate_flowpipe(
        t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps
    )

fp_basic = gen_basic(t0, tf, tmx, dt, delta, eps)
print(f"Basic:       nsteps={fp_basic.nsteps}, success={fp_basic.success}")

# --- Bunger (shrink wrap only) ---
fpg_sw = BungerTMFlowpipeGenerator(sys, shrink_wrap=True, precondition=False)

@jax.jit
def gen_sw(t0, tf, tmx, dt, delta, eps):
    return fpg_sw.generate_flowpipe(
        t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps
    )

fp_sw = gen_sw(t0, tf, tmx, dt, delta, eps)
print(f"Bunger (SW): nsteps={fp_sw.nsteps}, success={fp_sw.success}")

# --- Soundness check: every 10th tube step ---
mc_x0s = irx.utils.gen_ics(ix0, 20)

def mc_sim(x0):
    return sys.compute_trajectory(t0, tf, x0, (), dt)

mc_trajs = jax.vmap(mc_sim)(mc_x0s).to_convenience()

nsteps = int(fp_sw.nsteps)
n_checked = 0
n_contained = 0
check_every = 5

for step_i in range(0, nsteps, check_every):
    tube_tm = fp_sw[step_i]
    t_end = float(fp_sw.times[step_i])
    if jnp.isnan(t_end):
        break
    spatial_tm = tx_tm_eval(tube_tm, t_end, fp_sw._per_leaf_order[0])
    hull = spatial_tm.interval_hull()
    # MC trajectories at corresponding time index
    mc_step = int(round(t_end / dt))
    if mc_step >= mc_trajs.ys.shape[1]:
        continue
    pts = mc_trajs.ys[:, mc_step, :]
    for pt in pts:
        n_checked += 1
        if jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper):
            n_contained += 1

print(f"\nSoundness (SW): {n_contained}/{n_checked} MC points contained in spatial hulls")
if n_contained < n_checked:
    print("WARNING: Some MC points not contained - potential soundness issue!")
else:
    print("All MC points contained - soundness verified.")
