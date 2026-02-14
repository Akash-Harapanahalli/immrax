"""Debug containment failures."""

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

# Generate Basic and Bunger flowpipes
fpg_basic = BasicTMFlowpipeGenerator(sys)
fpg_sw = BungerTMFlowpipeGenerator(sys, shrink_wrap=True, precondition=False)

@jax.jit
def gen_basic(t0, tf, tmx, dt, delta, eps):
    return fpg_basic.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps)

@jax.jit
def gen_sw(t0, tf, tmx, dt, delta, eps):
    return fpg_sw.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps)

fp_basic = gen_basic(t0, tf, tmx, dt, delta, eps)
fp_sw = gen_sw(t0, tf, tmx, dt, delta, eps)

# Check containment for Basic first
mc_x0s = irx.utils.gen_ics(ix0, 20)

def mc_sim(x0):
    return sys.compute_trajectory(t0, tf, x0, (), dt)

mc_trajs = jax.vmap(mc_sim)(mc_x0s).to_convenience()

print("=== Basic flowpipe containment ===")
for step_i in range(0, min(int(fp_basic.nsteps), 49), 10):
    tube_tm = fp_basic[step_i]
    t_end = float(fp_basic.times[step_i])
    if jnp.isnan(t_end):
        break
    spatial_tm = tx_tm_eval(tube_tm, t_end, fp_basic._per_leaf_order[0])
    hull = spatial_tm.interval_hull()
    mc_step = int(round(t_end / dt))
    if mc_step >= mc_trajs.ys.shape[1]:
        continue
    pts = mc_trajs.ys[:, mc_step, :]
    contained = sum(1 for pt in pts if jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper))
    print(f"  step={step_i:3d}, t={t_end:.2f}, mc_step={mc_step}, contained={contained}/{len(pts)}, hull_width={hull.upper - hull.lower}")

print("\n=== Bunger (SW) flowpipe containment ===")
for step_i in range(0, int(fp_sw.nsteps), 10):
    tube_tm = fp_sw[step_i]
    t_end = float(fp_sw.times[step_i])
    if jnp.isnan(t_end):
        break
    spatial_tm = tx_tm_eval(tube_tm, t_end, fp_sw._per_leaf_order[0])
    hull = spatial_tm.interval_hull()
    mc_step = int(round(t_end / dt))
    if mc_step >= mc_trajs.ys.shape[1]:
        continue
    pts = mc_trajs.ys[:, mc_step, :]
    contained = sum(1 for pt in pts if jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper))
    if contained < len(pts):
        # Show failing points
        for j, pt in enumerate(pts):
            if not (jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper)):
                dist_lo = pt - hull.lower
                dist_hi = hull.upper - pt
                print(f"  step={step_i:3d}, t={t_end:.2f}, FAIL pt[{j}]={pt}, hull=[{hull.lower}, {hull.upper}], dist_lo={dist_lo}, dist_hi={dist_hi}")
                break  # just first failure
    else:
        print(f"  step={step_i:3d}, t={t_end:.2f}, contained={contained}/{len(pts)}, hull_width={hull.upper - hull.lower}")

# Also check: use tube hull (full time range) instead of spatial TM at endpoint
print("\n=== Bunger (SW) tube hull containment ===")
for step_i in [0, 20, 40, 60, 80, 100, 120, 130]:
    if step_i >= int(fp_sw.nsteps):
        break
    tube_tm = fp_sw[step_i]
    tube_hull = tube_tm.interval_hull()
    t_end = float(fp_sw.times[step_i])
    if jnp.isnan(t_end):
        break
    mc_step = int(round(t_end / dt))
    if mc_step >= mc_trajs.ys.shape[1]:
        continue
    pts = mc_trajs.ys[:, mc_step, :]
    contained = sum(1 for pt in pts if jnp.all(pt >= tube_hull.lower) and jnp.all(pt <= tube_hull.upper))
    print(f"  step={step_i:3d}, t={t_end:.2f}, contained={contained}/{len(pts)}, hull_width={tube_hull.upper - tube_hull.lower}")
