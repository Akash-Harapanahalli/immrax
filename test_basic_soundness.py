"""Check if BasicTMFlowpipeGenerator has the same spatial hull issue."""

import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator
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

fpg = BasicTMFlowpipeGenerator(sys)

@jax.jit
def gen(t0, tf, tmx, dt, delta, eps):
    return fpg.generate_flowpipe(t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps)

fp = gen(t0, tf, tmx, dt, 1e-4, 1e-2)
print(f"Basic: nsteps={fp.nsteps}, success={fp.success}")

# Check with leftmost trajectory
x0_left = jnp.array([-2.1, 0.0])
traj = sys.compute_trajectory(t0, tf, x0_left, (), dt).to_convenience()

nsteps = int(fp.nsteps)
for step_i in range(nsteps):
    t_end = float(fp.times[step_i])
    if jnp.isnan(t_end):
        break
    mc_step = int(round(t_end / dt))
    if mc_step >= traj.ys.shape[0]:
        break

    tube = fp[step_i]
    spatial = tx_tm_eval(tube, t_end, fp._per_leaf_order[0])
    hull = spatial.interval_hull()
    pt = traj.ys[mc_step]

    inside = jnp.all(pt >= hull.lower) and jnp.all(pt <= hull.upper)
    if not inside:
        gap_lo = pt - hull.lower
        gap_hi = hull.upper - pt
        print(f"Step {step_i} (t={t_end:.3f}): FAIL! pt={pt}, hull=[{hull.lower}, {hull.upper}], gap_lo={gap_lo}, gap_hi={gap_hi}")
        # Also check tube hull
        tube_hull = tube.interval_hull()
        inside_tube = jnp.all(pt >= tube_hull.lower) and jnp.all(pt <= tube_hull.upper)
        print(f"  tube hull inside? {inside_tube}")
        break

print("Done checking Basic spatial hull containment.")
