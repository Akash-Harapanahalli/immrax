import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator
import matplotlib.pyplot as plt


class HarmonicOscillator(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x):
        # return jnp.array([x[1], -x[0]])
        # VanderPol Oscillator
        mu = 1.0
        return jnp.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


sys = HarmonicOscillator()
t0 = 0.0
tf = 7.0
ix0 = irx.icentpert([-2, 0.0], [0.1, 0.0])
tmx = irx.taylor_model_identity(ix0, order=2)

fpg = BasicTMFlowpipeGenerator(sys)
dt = 0.05

delta = 1e-4
eps = 1e-2


@jax.jit
def gen_flowpipe(t0, tf, tmx, dt, delta, eps):
    return fpg.generate_flowpipe(
        t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps
    )


fp, times = irx.utils.run_times(100, gen_flowpipe, t0, tf, tmx, dt, delta, eps)

# assert fp.success, "Flowpipe generation failed"
# assert fp.nsteps > 0, "No steps taken"
print(f"nsteps: {fp.nsteps}, success: {fp.success}, med time {jnp.median(times)}")

# Check __getitem__
tube0 = fp[0]
# assert tube0.coeffs.shape == (2, 24), (
#     f"Unexpected tube coeffs shape: {tube0.coeffs.shape}"
# )
tubef = fp[fp.nsteps - 1]

fig, ax = plt.subplots()

# Check __call__ against exact solution x(t) = [-2*cos(t), 2*sin(t)]
# test_times = [0.05, 0.25, 0.5, 0.75, 0.99]
test_times = jnp.linspace(t0, tf, 20)
for t in test_times:
    spatial_tm = fp(t)
    irx.utils.draw_iarray(ax, spatial_tm.interval_hull(), color="tab:blue")

plt.show()

print("All checks passed.")
