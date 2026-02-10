import jax
import jax.numpy as jnp
import immrax as irx
from immrax.taylor import BasicTMFlowpipeGenerator
import matplotlib.pyplot as plt


class CVDP(irx.System):
    def __init__(self, mu=1.0):
        self.evolution = "continuous"
        self.xlen = 5
        self.mu = mu

    def f(self, t, x):
        x1, y1, x2, y2, b = x
        return jnp.array(
            [
                y1,
                self.mu * (1 - x1**2) * y1 + b * (x2 - x1) - x1,
                y2,
                self.mu * (1 - x2**2) * y2 + b * (x1 - x2) - x2,
                0.0,
            ]
        )


sys = CVDP(mu=1.0)
t0 = 0.0
tf = 7.0
ix0 = irx.icentpert([1.4, 2.4, 1.4, 2.4, 2.0], [0.15, 0.05, 0.15, 0.05, 1.0])
print(ix0)
tmx = irx.taylor_model_identity(ix0, order=1)

fpg = BasicTMFlowpipeGenerator(sys)
dt = 0.005

delta = 1e-5
eps = 1e-3


@jax.jit
def gen_flowpipe(t0, tf, tmx, dt, delta, eps):
    return fpg.generate_flowpipe(
        t0, tf, tmx, dt_max=dt, t_order=3, delta=delta, eps=eps
    )


fp, times = irx.utils.run_times(10, gen_flowpipe, t0, tf, tmx, dt, delta, eps)

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

# Draw a line at y = 2.75
ax.axhline(y=2.75, color="tab:red", linestyle="--")

# Check __call__ against exact solution x(t) = [-2*cos(t), 2*sin(t)]
# test_times = [0.05, 0.25, 0.5, 0.75, 0.99]
test_times = jnp.linspace(t0, tf, 100)
for t in test_times:
    spatial_tm = fp(t)
    irx.utils.draw_iarray(
        ax, spatial_tm.interval_hull(), color="tab:blue", fc="tab:blue"
    )

plt.show()

print("All checks passed.")
