"""Iterative LQR / DDP for norm-based reachable-set synthesis over a ParametricEmbedding.

The class :class:`ReachiLQR` lifts the notebook-style ``iterate()`` from
Harapanahalli_ACC2026 into a reusable, generic component that

* operates on any :class:`~immrax.parametric.embedding.ParametricEmbedding`,
* drives both the forward state sweep and the backward costate sweep with a
  fixed-step Euler ``jax.lax.scan`` (the backward pass integrates the costate
  and emits the per-step gains in a single fused scan),
* accepts piecewise-constant control inputs along a fixed control grid,
* supports both iLQR (Gauss-Newton) and DDP (full second-order) backward passes,
* threads R (the control penalty) as a runtime argument so the outer R-schedule
  does not trigger recompilation.

The backward sweep linearizes at the *perturbed* trajectory (the trajectory that
becomes the next iteration's nominal) under the *nominal* control, matching the
Harapanahalli_ACC2026 notebooks.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float
from immutabledict import immutabledict

from ..inclusion import interval
from .embedding import ParametricEmbedding
from .parametope import Parametope


class IterateResult(NamedTuple):
    state_traj: Array  # (N+1, Xlen) flat nominal trajectory at control grid
    pert_traj: Array   # (N+1, Xlen) flat perturbed trajectory at control grid
    l_traj: Array      # (N, Ulen)
    K_traj: Array      # (N, Ulen, Xlen)
    ts: Array          # (N+1,) control grid
    ifinal: Array      # int scalar, last valid grid index (<= N)
    Us_new: Array      # (N, *control_shape) Us - gamma*l - K*(X - Xnom) per step
    cost_final: Array  # scalar terminal cost at state_traj[ifinal]
    # Per-timestep tightest interval enclosure of the reachable set seen so far:
    # the running intersection, across iterations, of the perturbed set's
    # ``iover()`` box at each grid point. Shape (N+1, n) where n is the state
    # dimension. When ``track_iover=False`` these pass through unchanged.
    iover_lower: Array  # (N+1, n)
    iover_upper: Array  # (N+1, n)


class RunResult(NamedTuple):
    best: IterateResult
    Us: Array
    history: list  # list of dicts with per-iteration {R, ifinal, cost}


class ReachiLQR:
    """Iterative LQR / DDP over a :class:`ParametricEmbedding`.

    Parameters
    ----------
    embedding
        A configured :class:`ParametricEmbedding`. Its ``_initialize(pt0)`` is
        called eagerly to set up any internal state.
    pt0
        The initial parametope. Defines the pytree structure used internally for
        flatten/unflatten and the initial state of the forward sweep.
    terminal_cost
        ``phi(pt) -> scalar``. Evaluated at the perturbed trajectory's final
        parametope (and used for the Jmax early-termination event).
    running_cost
        ``L(t, pt, U) -> scalar``. Optional user Lagrangian; autodiffed for the
        gain computation. To match the original Harapanahalli_ACC2026 notebooks,
        pass ``running_cost = lambda t, pt, U: 0.0`` and let the ``R`` runtime
        arg of :meth:`iterate` apply ``R * I`` as Levenberg-Marquardt-style
        damping on ``Quu``. R as a true Lagrangian term (``0.5 R ||U||^2``) is
        more textbook-correct but couples the gain to the current ``Us``,
        which can oscillate without a line search.
    control_shape
        Shape of each control input ``U_k``. Total control dimension is
        ``Ulen = prod(control_shape)``.
    t0, tf
        Time horizon endpoints.
    N
        Number of piecewise-constant control steps. The control grid has
        ``N+1`` time points ``t0, t0 + dt_ctl, ..., tf`` where
        ``dt_ctl = (tf - t0) / N``.
    DDP
        If True, include second-order ``s' @ fxx``-type terms in the Q-function
        (differential dynamic programming). If False, use iLQR (Gauss-Newton).
    solver
        Integration scheme for both sweeps. Only ``"euler"`` is supported: the
        fixed-step Euler scan with ``dt = dt_ctl`` is what gives notebook-parity
        runtime, and a fixed step is required to fuse the backward costate +
        gain pass into a single scan.
    dt
        Nominal step size. The scan always uses the control-grid spacing
        ``dt_ctl = (tf - t0) / N``; ``dt`` is retained for API compatibility.
    """

    def __init__(
        self,
        embedding: ParametricEmbedding,
        pt0: Parametope,
        terminal_cost: Callable[[Parametope], Float],
        running_cost: Callable[[Float, Parametope, Array], Float],
        control_shape: tuple,
        t0: Float,
        tf: Float,
        N: int,
        *,
        DDP: bool = False,
        solver: str = "euler",
        dt: float = 0.01,
        track_iover: bool = True,
        f_kwargs: immutabledict = immutabledict({}),
    ):
        if solver != "euler":
            raise NotImplementedError(
                f"ReachiLQR only supports solver='euler' (got {solver!r}); the "
                "fused fixed-step scan requires a fixed Euler step."
            )
        self.embedding = embedding
        self.terminal_cost = terminal_cost
        self.running_cost = running_cost
        self.control_shape = tuple(int(s) for s in control_shape)
        self.Ulen = int(jnp.prod(jnp.array(self.control_shape)))
        self.t0 = float(t0)
        self.tf = float(tf)
        self.N = int(N)
        self.DDP = bool(DDP)
        self.solver = solver
        self.dt = float(dt)
        self.track_iover = bool(track_iover)
        self.f_kwargs = f_kwargs

        # Eagerly set up the embedding's per-pt0 state (e.g. NormotopeEmbedding caches NT, gsc).
        self.aux0 = self.embedding._initialize(pt0)
        x0_flat, self._unflatten = ravel_pytree((pt0, self.aux0))
        self.Xlen = int(x0_flat.size)
        self.x0_flat = x0_flat
        self.pt0 = pt0
        # State dimension n (for the per-timestep iover boxes). Requires the
        # parametope to support ``iover()`` — only needed when tracking is on.
        self.n = int(pt0.iover().shape[0]) if self.track_iover else 0
        # Cached flatten that just concatenates leaves (no unflatten closure rebuild).
        # The output structure of `_dynamics` matches (pt, aux), so this is always valid.
        self._flatten = lambda pt_aux: jnp.concatenate(
            [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(pt_aux)]
        )

        self.ts_grid = jnp.linspace(self.t0, self.tf, self.N + 1)
        self.dt_ctl = (self.tf - self.t0) / self.N

        # Compile the inner workhorse once.
        self._iterate_jit = jax.jit(self._iterate_impl)

    # ---------------------------------------------------------------- internals

    def _split(self, doubled: Array) -> tuple[Array, Array]:
        return doubled[: self.Xlen], doubled[self.Xlen :]

    def _join(self, x_pert: Array, x_nom: Array) -> Array:
        return jnp.concatenate([x_pert, x_nom])

    def _f_flat(self, t: Float, x_flat: Array, U_flat: Array, ix=None) -> Array:
        """Flat-vector wrapper around ``embedding._dynamics``.

        If ``ix`` (an :class:`~immrax.inclusion.Interval`) is supplied, it is
        passed to ``_dynamics`` as the box over which the mixed Jacobian is
        evaluated, in place of the parametope's own ``iover()``. ReachiLQR uses
        this to feed the running cross-iteration intersection box (a tighter, but
        still valid, enclosure of the reachable set) and shave conservatism off
        the contraction rate.
        """
        pt, aux = self._unflatten(x_flat)
        U = U_flat.reshape(self.control_shape)
        kw = dict(self.f_kwargs)
        if ix is not None:
            kw["ix"] = ix
        dpt_daux = self.embedding._dynamics(t, (pt, aux), U, **kw)
        return self._flatten(dpt_daux)

    def _iover_lu(self, x_flat: Array) -> tuple[Array, Array]:
        """``(lower, upper)`` of the parametope's interval hull from a flat state."""
        pt, _ = self._unflatten(x_flat)
        ix = pt.iover()
        return ix.lower, ix.upper

    def _terminal_flat(self, x_flat: Array) -> Float:
        pt, _ = self._unflatten(x_flat)
        return self.terminal_cost(pt)

    def _running_flat(
        self, t: Float, x_flat: Array, U_flat: Array
    ) -> Float:
        pt, _ = self._unflatten(x_flat)
        return self.running_cost(t, pt, U_flat.reshape(self.control_shape))

    # ----------------------------------------------------- forward + backward

    def _iterate_impl(
        self,
        Us: Array,
        l_traj: Array,
        K_traj: Array,
        R: Float,
        gamma: Float,
        Jmax: Float,
        iover_lower_in: Array,
        iover_upper_in: Array,
    ) -> IterateResult:
        Xlen = self.Xlen
        Ulen = self.Ulen
        N = self.N
        dt = self.dt_ctl
        DDP = self.DDP
        track = self.track_iover

        def alive_fn(y_flat):
            x_pert, _ = self._split(y_flat)
            pt, _ = self._unflatten(x_pert)
            cost_val = self.terminal_cost(pt)
            return (
                (cost_val <= Jmax)
                & jnp.isfinite(cost_val)
                & jnp.all(jnp.isfinite(y_flat))
            )

        # ---- forward sweep: fixed-step Euler scan (freezes derivative on bad cost) ----
        # Euler with dt == dt_ctl saves exactly one state per control-grid point.
        def forward_step(y_flat, k):
            t = self.ts_grid[k]
            idx = jnp.minimum(k, N - 1)  # control-grid index at ts_grid[k]
            x_pert, x_nom = self._split(y_flat)
            U_nom = Us[idx].reshape(-1)
            U_pert = U_nom - gamma * l_traj[idx] - K_traj[idx] @ (x_pert - x_nom)
            # Optional: feed the running cross-iteration intersection box at this
            # grid point to the mixed-Jacobian evaluation (tighter contraction).
            ix_k = interval(iover_lower_in[k], iover_upper_in[k]) if track else None
            # lax.cond (not where) so the two dynamics evals are SKIPPED once the
            # set blows up — most steps are dead on early iterations, and this is
            # what gives notebook-parity runtime.
            deriv = jax.lax.cond(
                alive_fn(y_flat),
                lambda: self._join(
                    self._f_flat(t, x_pert, U_pert, ix_k),
                    self._f_flat(t, x_nom, U_nom, ix_k),
                ),
                lambda: jnp.zeros_like(y_flat),
            )
            return y_flat + dt * deriv, y_flat

        y0 = self._join(self.x0_flat, self.x0_flat)
        y_last, ys_part = jax.lax.scan(forward_step, y0, jnp.arange(N))
        sol_ys = jnp.concatenate([ys_part, y_last[None]], axis=0)  # (N+1, 2*Xlen)

        pert_traj = sol_ys[:, :Xlen]
        state_traj = sol_ys[:, Xlen:]  # nominal

        # ifinal = last grid index where the perturbed cost is within Jmax and
        # the state is finite (cumulative-and: once dead, stays dead).
        alive_grid = jax.vmap(alive_fn)(sol_ys)
        alive_cum = jnp.cumprod(alive_grid.astype(jnp.int32))
        ifinal = jnp.maximum(jnp.sum(alive_cum) - 1, 0)

        # Per-step perturbed control. Pad with nominal Us[i] past ifinal to keep
        # Us_new finite (matters because Us_new becomes next iter's input).
        def per_step_U(i):
            U_nom = Us[i].reshape(-1)
            U_pert = U_nom - gamma * l_traj[i] - K_traj[i] @ (pert_traj[i] - state_traj[i])
            return jnp.where(i < ifinal, U_pert.reshape(self.control_shape), Us[i])

        Us_new = jax.vmap(per_step_U)(jnp.arange(N))

        # ---- update the running per-timestep intersection box ----
        # Each grid point's perturbed normotope is a valid enclosure of the true
        # reachable set R(t_k), so intersecting its interval hull into the box
        # keeps a valid (and tighter) enclosure. Only update where the step is
        # alive (k <= ifinal); past ifinal the set has blown up and the frozen
        # state is not a valid enclosure. The box accumulates monotonically and
        # is fed back into the next iteration's mixed-Jacobian evaluation.
        if track:
            lo_new, up_new = jax.vmap(self._iover_lu)(pert_traj)  # (N+1, n) each
            box_lo = jnp.maximum(iover_lower_in, lo_new)
            box_up = jnp.minimum(iover_upper_in, up_new)
            alive_k = (jnp.arange(N + 1) <= ifinal)[:, None]
            iover_lower_out = jnp.where(alive_k, box_lo, iover_lower_in)
            iover_upper_out = jnp.where(alive_k, box_up, iover_upper_in)
        else:
            iover_lower_out = iover_lower_in
            iover_upper_out = iover_upper_in

        # ---- backward sweep: single fused Euler scan over the costate (sig, s, S),
        # emitting per-step gains (l_i, K_i) in the same pass. The Jacobians fx, fu
        # are therefore evaluated once per step, not twice. Linearization is at the
        # perturbed trajectory under the nominal control (matches the notebook).
        x_final = pert_traj[ifinal]
        sig_T = self._terminal_flat(x_final)
        s_T = jax.grad(self._terminal_flat)(x_final)
        S_T = jax.hessian(self._terminal_flat)(x_final)

        def backward_step(carry, U_p):
            sig, s, S, i = carry  # i = state index of this step (control index + 1)
            ii = jnp.clip(i, 0, N)
            mask = i < ifinal
            t = self.ts_grid[ii]
            x_flat = pert_traj[ii]
            U_flat = U_p.reshape(-1)

            # The backward gain computation linearizes against the plain
            # per-iteration dynamics (no intersection box): the box tightens the
            # forward contraction rate, but the gains must be consistent with the
            # dynamics the rollout actually follows.
            f_flat_bwd = lambda t_, x_, u_: self._f_flat(t_, x_, u_)

            # lax.cond (not where) so the expensive fx/fu Jacobians are SKIPPED on
            # masked steps (everything past ifinal). On early iterations most of
            # the horizon is masked; this is the dominant runtime saving.
            def real_branch():
                fx = jax.jacfwd(f_flat_bwd, 1)(t, x_flat, U_flat).T
                fu = jax.jacfwd(f_flat_bwd, 2)(t, x_flat, U_flat).T

                L = lambda x_, u_: self._running_flat(t, x_, u_)
                l_val = L(x_flat, U_flat)
                lx = jax.grad(L, 0)(x_flat, U_flat)
                lu = jax.grad(L, 1)(x_flat, U_flat)
                lxx = jax.hessian(L, 0)(x_flat, U_flat)
                luu = jax.hessian(L, 1)(x_flat, U_flat)
                lux = jax.jacfwd(jax.jacrev(L, 1), 0)(x_flat, U_flat)

                Qu = lu + fu @ s
                if not DDP:
                    Qxx = lxx + S @ fx.T + fx @ S
                    Qux = lux + fu @ S
                    Quu = luu
                else:
                    fxx = jax.hessian(f_flat_bwd, 1)(t, x_flat, U_flat)
                    fux = jax.jacfwd(jax.jacrev(f_flat_bwd, 2), 1)(t, x_flat, U_flat)
                    fuu = jax.hessian(f_flat_bwd, 2)(t, x_flat, U_flat)
                    Qxx = lxx + S @ fx.T + fx @ S + jnp.tensordot(s, fxx, axes=1)
                    Qux = lux + fu @ S + jnp.tensordot(s, fux, axes=1)
                    Quu = luu + jnp.tensordot(s, fuu, axes=1)

                Quu_inv = jnp.linalg.inv(Quu + (R + 1e-10) * jnp.eye(Ulen))
                l_i = Quu_inv @ Qu
                K_i = Quu_inv @ Qux

                sig_dot = -(l_val - 0.5 * Qu @ Quu_inv @ Qu)
                s_dot = -(lx + fx @ s - Qux.T @ Quu_inv @ Qu)
                S_dot = Qxx - Qux.T @ Quu_inv @ Qux
                return (sig - dt * sig_dot, s - dt * s_dot, S - dt * S_dot, l_i, K_i)

            def masked_branch():
                return (sig, s, S, jnp.zeros(Ulen), jnp.zeros((Ulen, Xlen)))

            sig_n, s_n, S_n, l_i, K_i = jax.lax.cond(mask, real_branch, masked_branch)
            return (
                jnp.nan_to_num(sig_n),
                jnp.nan_to_num(s_n),
                jnp.nan_to_num(S_n),
                i - 1,
            ), (jnp.nan_to_num(l_i), jnp.nan_to_num(K_i))

        (_, _, _, _), (l_new, K_new) = jax.lax.scan(
            backward_step, (sig_T, s_T, S_T, N), Us, length=N, reverse=True
        )

        # Cost reported on the perturbed trajectory at ifinal (the trajectory that
        # becomes the next iteration's nominal). Matches the notebook's reporting.
        cost_final = self._terminal_flat(pert_traj[ifinal])

        return IterateResult(
            state_traj=state_traj,
            pert_traj=pert_traj,
            l_traj=l_new,
            K_traj=K_new,
            ts=self.ts_grid,
            ifinal=ifinal,
            Us_new=Us_new,
            cost_final=cost_final,
            iover_lower=iover_lower_out,
            iover_upper=iover_upper_out,
        )

    # --------------------------------------------------------- public surface

    def initial_gains(self) -> tuple[Array, Array]:
        """Convenience: zero-initialized ``(l_traj, K_traj)`` of the right shape."""
        l_traj = jnp.zeros((self.N, self.Ulen))
        K_traj = jnp.zeros((self.N, self.Ulen, self.Xlen))
        return l_traj, K_traj

    def initial_controls(self) -> Array:
        """Convenience: zero ``Us`` of shape ``(N, *control_shape)``."""
        return jnp.zeros((self.N, *self.control_shape))

    def initial_iover(self) -> tuple[Array, Array]:
        """Convenience: the unbounded ``(lower, upper)`` per-timestep box.

        Initialized to ``[-inf, +inf]`` so the first iteration's intersection is
        just that iteration's own ``iover()``. Shapes are ``(N+1, n)``; when
        ``track_iover`` is off, ``n == 0`` and these are inert placeholders.
        """
        lo = jnp.full((self.N + 1, self.n), -jnp.inf)
        up = jnp.full((self.N + 1, self.n), jnp.inf)
        return lo, up

    def iterate(
        self,
        Us: Array,
        l_traj: Array,
        K_traj: Array,
        R: Float,
        *,
        gamma: Float = 1.0,
        Jmax: Float = jnp.inf,
        iover_lower: Array | None = None,
        iover_upper: Array | None = None,
    ) -> IterateResult:
        """Run one outer iLQR/DDP iteration.

        When ``track_iover`` is enabled, pass the previous iteration's
        ``iover_lower``/``iover_upper`` (from the returned :class:`IterateResult`)
        to accumulate the running cross-iteration intersection box; if omitted,
        the box is (re)initialized to ``[-inf, +inf]``.
        """
        if iover_lower is None or iover_upper is None:
            iover_lower, iover_upper = self.initial_iover()
        return self._iterate_jit(
            Us,
            l_traj,
            K_traj,
            jnp.asarray(R),
            jnp.asarray(gamma),
            jnp.asarray(Jmax),
            iover_lower,
            iover_upper,
        )

    def run(
        self,
        Us0: Array | None = None,
        *,
        R_schedule: Sequence[Float] = (1.0,),
        iters_per_R: int = 100,
        gamma: Float = 1.0,
        Jmax: Float = jnp.inf,
        verbose: bool = False,
    ) -> RunResult:
        """Outer R-schedule loop with best-tracking and revert-on-regression.

        Mirrors the notebook's outer loop: for each ``R`` in ``R_schedule``, run
        ``iters_per_R`` ``iterate`` calls, lexicographically maximizing
        ``(ifinal, -cost)``, and reverting the working ``Us`` to the best seen so
        far whenever it gets worse.
        """
        Us = self.initial_controls() if Us0 is None else Us0
        l_traj, K_traj = self.initial_gains()
        # The intersection box accumulates across ALL iterations (and across
        # R-segments) and is never reverted: every iterate contributes a valid
        # enclosure regardless of whether its cost regressed.
        iover_lower, iover_upper = self.initial_iover()

        best = None
        history: list = []

        for R in R_schedule:
            R = float(R)
            for k in range(iters_per_R):
                res = self.iterate(
                    Us,
                    l_traj,
                    K_traj,
                    R,
                    gamma=gamma,
                    Jmax=Jmax,
                    iover_lower=iover_lower,
                    iover_upper=iover_upper,
                )
                iover_lower = res.iover_lower
                iover_upper = res.iover_upper
                ifinal = int(res.ifinal)
                cost = float(res.cost_final)
                history.append({"R": R, "iter": k, "ifinal": ifinal, "cost": cost})
                if verbose:
                    print(
                        f"R={R:>8.3g}  it={k:>4d}  ifinal={ifinal:>5d}  cost={cost: .4f}"
                    )

                if best is None or (ifinal, -cost) > (
                    int(best.best.ifinal),
                    -float(best.best.cost_final),
                ):
                    best = RunResult(best=res, Us=res.Us_new, history=[])

                Us = res.Us_new
                l_traj = res.l_traj
                K_traj = res.K_traj

            if verbose and best is not None:
                print(
                    f"  reverting to best  ifinal={int(best.best.ifinal)}  cost={float(best.best.cost_final): .4f}"
                )
            if best is not None:
                Us = best.Us

        assert best is not None
        return RunResult(best=best.best, Us=best.Us, history=history)
