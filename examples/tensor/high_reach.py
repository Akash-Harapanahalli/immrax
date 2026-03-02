# %%
import jax
import jax.numpy as jnp
import immrax as irx
import matplotlib.pyplot as plt

print(jax.devices())

# %%

p = 2

# A = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
# Generate a random stable matrix

# stable eigs
L = jnp.diag(jax.random.uniform(jax.random.PRNGKey(0), (2,), minval=-10.0, maxval=-1.0))
# random matrix
U = jax.random.normal(jax.random.PRNGKey(1), (2, 2))
A = U @ L @ jnp.linalg.inv(U)


class VanDerPol(irx.System):
    def __init__(self):
        # self.mu = 0.5
        self.mu = 1.0

    def f(self, t, x):
        # x1, x2 = x
        # return jnp.array([x2, self.mu * (1 - x1**2) * x2 - x1])
        # return jnp.array([x2, -x1])
        return A @ x


P = jnp.array([[1.0, 0.0], [0.0, 10.0]])
# P = jnp.array([[1.0, 0.0], [0.0, 1.0]])
sample_alpha = irx.pjet(lambda x: x.T @ P @ x)(irx.identijet(jnp.zeros(2), p))
# ox = jnp.array([1.0, 0.0])
ox = jnp.array([2.0, 0.0])
# ix = irx.icentpert(ox, 0.5)


@jax.tree_util.register_pytree_node_class
class PolyParametope(irx.Parametope):
    def __init__(self, ox, alpha, y):
        super().__init__(ox, alpha, y)

    @property
    def poly(self):
        return irx.TaylorPolynomial(self.alpha, sample_alpha.multiindices, self.ox)

    @property
    def poly_uncentered(self):
        return irx.TaylorPolynomial(
            self.alpha, sample_alpha.multiindices, jnp.zeros_like(self.ox)
        )

    def g(self, x):
        # print('g', self.ox)
        return self.poly.evaluate(x)

    def plot_projection(self, ax, xi=0, yi=1):
        aa = jnp.linspace(-3, 3, 501)
        xx, yy = jnp.meshgrid(aa, aa)
        # evaluate g on the grid
        sh = xx.shape
        gg = jax.vmap(self.g)(jnp.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1))
        gg = gg.reshape(xx.shape)

        # mask = jnp.logical_not(jnp.logical_and(gg <= self.y, gg >= 0.0))
        # # xx = jnp.where(mask, xx, jnp.nan)
        # # yy = jnp.where(mask, yy, jnp.nan)
        # xx = (xx.reshape(-1)[mask]).reshape(sh)
        # yy = (yy.reshape(-1)[mask]).reshape(sh)

        ax.contour(xx, yy, gg, levels=[self.y], cmap="viridis")
        return gg

    def iover(self):
        # sampling based iover for now
        aa = jnp.linspace(-3, 3, 101)
        xx, yy = jnp.meshgrid(aa, aa)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # evaluate g on the grid
        gg = jax.vmap(self.g)(jnp.stack((xx, yy), axis=-1))
        mask = gg <= self.y * 1.1
        xl = jnp.min(jnp.where(mask, xx, jnp.inf))
        xu = jnp.max(jnp.where(mask, xx, -jnp.inf))
        yl = jnp.min(jnp.where(mask, yy, jnp.inf))
        yu = jnp.max(jnp.where(mask, yy, -jnp.inf))
        # return irx.interval(jnp.array([xl, yl]), jnp.array([xu, yu])).scale(1.1)
        return irx.interval(self.ox) + irx.icentpert(0.0, 0.05)

    @classmethod
    def from_parametope(cls, pt):
        return PolyParametope(pt.ox, pt.alpha, pt.y)


sys = VanDerPol()
pt0 = PolyParametope(ox, sample_alpha.coeffs, jnp.array(0.1))
# nt0 = irx.L2Normotope(ox, P, 0.1)
print(jax.tree_util.tree_flatten(pt0))

# plot the initial parametope
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
gg = pt0.plot_projection(ax)
irx.utils.draw_iarray(ax, pt0.iover())

# %%


class PolyParametopeEmbedding(irx.ParametricEmbedding):
    def _initialize(self, pt0):
        # if not isinstance(pt0, PolyParametope):
        #     raise ValueError(
        #         "PolyParametopeEmbedding requires a PolyParametope as the initial set"
        #     )
        return None

    def _dynamics(self, t, state):
        pt, aux = state

        ix = pt.iover()
        f_ox = self.sys.f(t, pt.ox)

        def inner(x):
            prod = jax.jvp(
                pt.poly_uncentered.evaluate,
                (x - pt.ox,),
                (self.sys.f(t, x) - f_ox,),
            )[1]
            return jnp.atleast_1d(prod)

        outer = irx.mdit(inner, p)
        res = outer(ix, pt.ox)

        def eval_pp1(coeffs, x):
            tp = irx.TaylorPolynomial(
                coeffs, res[0].get_order(p + 1).multiindices, pt.ox
            )
            # print(coeffs)
            # print(tp.evaluate_monomials(x))
            return tp.evaluate(x)

        def eval_R(x):
            return jnp.atleast_1d(irx.natif(eval_pp1)(res[1], x).upper)

        ox_dot = self.sys.f(t, pt.ox)
        # High order adjoint cancellation

        alpha_dot = -res[0].get_to_order(p).coeffs.flatten().at[0].set(0.0)
        y_dot = irx.natif(eval_R)(ix).upper[0]
        # y_dot = 0.0

        return PolyParametope(ox_dot, alpha_dot, y_dot), None


embsys = PolyParametopeEmbedding(sys)
jit_dyn = jax.jit(embsys._dynamics)
print(jit_dyn(0.0, (pt0, None)))

# %%

traj = embsys.compute_reachset(0.0, 6.0, pt0, (), dt=0.01)
# print(traj)


# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

tfinite = traj.ts[jnp.isfinite(traj.ts)]
print(len(tfinite))
yy = traj.ys[0]
#
# ax.plot(yy.ox[:, 0], yy.ox[:, 1], "k-", label="center")

# for i in range(0, 300, 10):
for i in range(0, len(tfinite), 10):
    # for i in [15, 30, 50] :
    ppt = PolyParametope(yy.ox[i], yy.alpha[i], yy.y[i])
    # ppt = traj[i]
    print(ppt)
    ppt.plot_projection(ax, xi=0, yi=1)

# %%
