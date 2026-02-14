"""Targeted soundness test: check every step, verify MC time alignment."""

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

# Generate Bunger flowpipe
fpg_sw = BungerTMFlowpipeGenerator(sys, shrink_wrap=True, precondition=False)

@jax.jit
def gen_sw(t0, tf, tmx, dt, delta, eps):
    return fpg_sw.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps)

fp = gen_sw(t0, tf, tmx, dt, delta, eps)
print(f"Bunger (SW): nsteps={fp.nsteps}, success={fp.success}")

# Generate a SINGLE MC trajectory from the leftmost IC
x0_left = jnp.array([-2.1, 0.0])
traj_left = sys.compute_trajectory(t0, tf, x0_left, (), dt).to_convenience()
print(f"MC traj shape: {traj_left.ys.shape}, ts shape: {traj_left.ts.shape}")
print(f"First 5 times: {traj_left.ts[:5]}")
print(f"MC traj at t=0: {traj_left.ys[0]}")
print(f"MC traj at t=dt: {traj_left.ys[1]}")

# Check containment at every step
nsteps = int(fp.nsteps)
first_fail = None
for step_i in range(nsteps):
    t_end = float(fp.times[step_i])
    if jnp.isnan(t_end):
        break
    mc_step = int(round(t_end / dt))
    if mc_step >= traj_left.ys.shape[0]:
        break

    # Verify time alignment
    mc_time = float(traj_left.ts[mc_step])

    tube_tm = fp[step_i]
    spatial_tm = tx_tm_eval(tube_tm, t_end, fp._per_leaf_order[0])
    hull = spatial_tm.interval_hull()
    pt = traj_left.ys[mc_step]

    inside = jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper)
    if not inside and first_fail is None:
        first_fail = step_i
        print(f"\nFIRST FAILURE at step {step_i}:")
        print(f"  fp time = {t_end:.6f}, mc time = {mc_time:.6f}")
        print(f"  mc pt = {pt}")
        print(f"  hull lo = {hull.lower}")
        print(f"  hull hi = {hull.upper}")
        print(f"  hull width = {hull.upper - hull.lower}")
        print(f"  remainder = [{spatial_tm.remainder.lower}, {spatial_tm.remainder.upper}]")

        # Also check the TUBE hull
        tube_hull = tube_tm.interval_hull()
        inside_tube = jnp.all(pt >= tube_hull.lower) and jnp.all(pt <= tube_hull.upper)
        print(f"  tube hull lo = {tube_hull.lower}")
        print(f"  tube hull hi = {tube_hull.upper}")
        print(f"  inside tube hull? {inside_tube}")

        # Check previous step
        if step_i > 0:
            prev_tube = fp[step_i - 1]
            prev_t = float(fp.times[step_i - 1])
            prev_spatial = tx_tm_eval(prev_tube, prev_t, fp._per_leaf_order[0])
            prev_hull = prev_spatial.interval_hull()
            prev_mc_step = int(round(prev_t / dt))
            prev_pt = traj_left.ys[prev_mc_step]
            prev_inside = jnp.all(prev_pt >= prev_hull.lower) and jnp.all(prev_pt <= prev_hull.upper)
            print(f"\n  PREV step {step_i-1}:")
            print(f"    t = {prev_t:.6f}")
            print(f"    mc pt = {prev_pt}")
            print(f"    hull = [{prev_hull.lower}, {prev_hull.upper}]")
            print(f"    inside? {prev_inside}")
            print(f"    remainder = [{prev_spatial.remainder.lower}, {prev_spatial.remainder.upper}]")

if first_fail is None:
    print(f"\nAll {nsteps} steps contain the leftmost MC trajectory. Soundness OK!")
else:
    print(f"\nFirst failure at step {first_fail}")

    # Check a center trajectory too
    x0_center = jnp.array([-2.0, 0.0])
    traj_center = sys.compute_trajectory(t0, tf, x0_center, (), dt).to_convenience()
    mc_step = int(round(float(fp.times[first_fail]) / dt))
    pt_c = traj_center.ys[mc_step]
    hull = tx_tm_eval(fp[first_fail], float(fp.times[first_fail]), fp._per_leaf_order[0]).interval_hull()
    print(f"\n  Center traj at failure step: {pt_c}")
    print(f"  Inside hull? {jnp.all(pt_c >= hull.lower) and jnp.all(pt_c <= hull.upper)}")
