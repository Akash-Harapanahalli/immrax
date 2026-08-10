"""Iterative LQR / DDP for norm-based reachable-set synthesis over a ParametricEmbedding.

:class:`ReachiLQR` operates on any :class:`ParametricEmbedding`, drives both the
forward state sweep and backward costate sweep with a fixed-step ``jax.lax.scan``
(explicit Euler or the embedding's semi-implicit ``_symplectic_step``; the
backward pass integrates the costate and emits per-step gains in one fused
scan), supports iLQR (Gauss-Newton) and DDP (full second-order) modes, and
threads R as a runtime argument so the outer R-schedule does not recompile. The backward sweep linearizes at the perturbed trajectory
(next iteration's nominal) under the nominal control.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float
from immutabledict import immutabledict

from ..inclusion import Interval, interval
from .embedding import ParametricEmbedding
from .parametope import Parametope


class IterateResult(NamedTuple):
    state_traj: Array  # (N+1, Xlen) flat nominal trajectory at control grid
    pert_traj: Array   # (N+1, Xlen) flat perturbed trajectory at control grid
    l_traj: Array      # (N, Ulen)
    K_traj: Array      # (N, Ulen, Xlen)
    ts: Array          # (N+1,) control grid
    ifinal: Array      # int scalar, last valid grid index (<= N)
    Us_new: Array      # (N, *hypercontrol_shape) Us - gamma*l - K*(X - Xnom) per step
    cost_final: Array  # scalar terminal cost at state_traj[ifinal]
    # Running cross-iteration intersection of the perturbed set's iover() box at
    # each grid point (an Interval of shape (N+1, n)); passes through unchanged
    # when track_iover=False.
    iover: Interval
    # t0 costate s(0) = d(terminal_cost)/dx0 propagated along the trajectory under
    # the current control. Enables a free-initial-state gradient step on x0
    # (x0 <- x0 - beta * (costate0 + grad_initial_cost(x0))) by the caller.
    costate0: Array


class RunResult(NamedTuple):
    best: IterateResult  # best iterate by (ifinal, -cost)
    Us: Array            # best.Us_new
    history: list        # per-iteration {iter, R, ifinal, cost} dicts
    iover: Interval      # final accumulated intersection box


def constant_schedule(R: Float, revert_best: bool = False) -> Callable[[int], tuple]:
    """R-schedule factory: constant ``(R, revert_best)`` every iteration."""
    R, revert_best = float(R), bool(revert_best)

    def schedule(i: int) -> tuple:
        return R, revert_best

    return schedule


def phased_schedule(phases: Sequence[tuple]) -> Callable[[int], tuple]:
    """R-schedule factory from ``phases = [(R, revert_best, n_iters), ...]``.

    Returns a callable ``i -> (R, revert_best)`` carrying a ``.total`` attribute
    (the summed iteration count), which :meth:`ReachiLQR.run` uses as the default
    ``iters``. For example, "R=2 for 750 iterations (reverting to the best on the
    last one), then R=20 for 750" is::

        phased_schedule([(2.0, False, 749), (2.0, True, 1), (20.0, False, 750)])
    """
    bounds, total = [], 0
    for R, revert_best, n in phases:
        total += int(n)
        bounds.append((total, float(R), bool(revert_best)))

    def schedule(i: int) -> tuple:
        for b, R, revert_best in bounds:
            if i < b:
                return R, revert_best
        return bounds[-1][1], bounds[-1][2]

    schedule.total = total
    return schedule


class ReachiLQR:
    """Iterative LQR / DDP over a :class:`ParametricEmbedding`.

    Parameters
    ----------
    embedding
        A configured :class:`ParametricEmbedding`. Its ``_initialize(pt0)`` is
        called eagerly to set up any internal state, and its
        ``hypercontrol_shape(pt0)`` defines the shape of each control input
        ``U_k`` (``Ulen = prod(hypercontrol_shape)``).
    pt0
        The initial parametope. Defines the pytree structure used internally for
        flatten/unflatten and the initial state of the forward sweep.
    terminal_cost
        ``phi(pt) -> scalar``. Evaluated at the perturbed trajectory's final
        parametope (and used for the Jmax early-termination event).
    running_cost
        ``L(t, pt, U) -> scalar``. Autodiffed for the gain computation. Passing
        ``lambda t, pt, U: 0.0`` lets the ``R`` runtime arg of :meth:`iterate`
        act as Levenberg-Marquardt damping ``R * I`` on ``Quu``; a true
        Lagrangian term ``0.5 R ||U||^2`` is more textbook-correct but couples
        the gain to the current ``Us`` and can oscillate without a line search.
    t0, tf
        Time horizon endpoints.
    N
        Number of piecewise-constant control steps; the control grid is
        ``t0, t0 + dt_ctl, ..., tf`` with ``dt_ctl = (tf - t0) / N``.
    DDP
        If True, include second-order ``s' @ fxx`` terms (DDP); else iLQR.
    solver
        ``"symplectic"`` (default) or ``"euler"``. Symplectic rolls the forward
        sweep with ``embedding._symplectic_step`` (semi-implicit adjoint step)
        and linearizes the backward sweep at the discrete-equivalent rate
        ``(step(x, U) - x) / dt``, so the gains match the map the rollout
        actually follows. Requires the embedding to implement
        ``_symplectic_step``; pass ``solver="euler"`` for embeddings without one
        (e.g. GramNormotopeEmbedding).
    dt
        Retained for API compatibility; the scan uses ``dt_ctl``.
    """

    def __init__(
        self,
        embedding: ParametricEmbedding,
        pt0: Parametope,
        terminal_cost: Callable[[Parametope], Float],
        running_cost: Callable[[Float, Parametope, Array], Float],
        t0: Float,
        tf: Float,
        N: int,
        *,
        DDP: bool = False,
        solver: str = "symplectic",
        dt: float = 0.01,
        track_iover: bool = True,
        f_kwargs: immutabledict = immutabledict({}),
    ):
        if solver not in ("euler", "symplectic"):
            raise NotImplementedError(
                f"ReachiLQR only supports solver='euler' or 'symplectic' (got "
                f"{solver!r}); the fused fixed-step scan requires a fixed step."
            )
        if (
            solver == "symplectic"
            and type(embedding)._symplectic_step
            is ParametricEmbedding._symplectic_step
        ):
            raise NotImplementedError(
                f"{type(embedding).__name__} does not implement _symplectic_step; "
                "use solver='euler'."
            )
        self.embedding = embedding
        self.terminal_cost = terminal_cost
        self.running_cost = running_cost
        self.hypercontrol_shape = tuple(
            int(s) for s in embedding.hypercontrol_shape(pt0)
        )
        self.Ulen = int(jnp.prod(jnp.array(self.hypercontrol_shape)))
        self.t0 = float(t0)
        self.tf = float(tf)
        self.N = int(N)
        self.DDP = bool(DDP)
        self.solver = solver
        self.dt = float(dt)
        self.track_iover = bool(track_iover)
        self.f_kwargs = f_kwargs

        self.aux0 = self.embedding._initialize(pt0)
        x0_flat, self._unflatten = ravel_pytree((pt0, self.aux0))
        self.Xlen = int(x0_flat.size)
        self.x0_flat = x0_flat
        self.pt0 = pt0
        # State dimension for the per-timestep iover boxes; only needed when tracking.
        self.n = (
            int(self.embedding.iover((pt0, self.aux0)).shape[0])
            if self.track_iover
            else 0
        )
        # _dynamics output matches the (pt, aux) structure, so a flat concatenate
        # of leaves is always a valid inverse of _unflatten.
        self._flatten = lambda pt_aux: jnp.concatenate(
            [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(pt_aux)]
        )

        self.ts_grid = jnp.linspace(self.t0, self.tf, self.N + 1)
        self.dt_ctl = (self.tf - self.t0) / self.N

        self._iterate_jit = jax.jit(self._iterate_impl)

    # ---------------------------------------------------------------- internals

    def _split(self, doubled: Array) -> tuple[Array, Array]:
        return doubled[: self.Xlen], doubled[self.Xlen :]

    def _join(self, x_pert: Array, x_nom: Array) -> Array:
        return jnp.concatenate([x_pert, x_nom])

    def _f_flat(self, t: Float, x_flat: Array, U_flat: Array, ix=None) -> Array:
        """Flat-vector wrapper around ``embedding._dynamics``. ``ix``, if given,
        is the box for the mixed-Jacobian evaluation (the running intersection)."""
        pt, aux = self._unflatten(x_flat)
        U = U_flat.reshape(self.hypercontrol_shape)
        kw = dict(self.f_kwargs)
        if ix is not None:
            kw["ix"] = ix
        dpt_daux = self.embedding._dynamics(t, (pt, aux), U=U, **kw)
        return self._flatten(dpt_daux)

    def _step_flat(self, t: Float, x_flat: Array, U_flat: Array, ix=None) -> Array:
        """Flat-vector wrapper around ``embedding._symplectic_step`` with
        ``dt = dt_ctl``: the discrete map ``x_k -> x_{k+1}``."""
        pt, aux = self._unflatten(x_flat)
        U = U_flat.reshape(self.hypercontrol_shape)
        kw = dict(self.f_kwargs)
        if ix is not None:
            kw["ix"] = ix
        nxt = self.embedding._symplectic_step(t, self.dt_ctl, (pt, aux), U=U, **kw)
        return self._flatten(nxt)

    def _iover_box(self, x_flat: Array) -> Interval:
        """Interval hull of the reachable set from a flat ``(pt, aux)`` state.

        Delegates to ``embedding.iover``, which can use the aux (e.g. a maintained
        ``H+``) — not just the parametope.
        """
        state = self._unflatten(x_flat)
        return self.embedding.iover(state)

    def _terminal_flat(self, x_flat: Array) -> Float:
        pt, _ = self._unflatten(x_flat)
        return self.terminal_cost(pt)

    def _running_flat(
        self, t: Float, x_flat: Array, U_flat: Array
    ) -> Float:
        pt, _ = self._unflatten(x_flat)
        return self.running_cost(t, pt, U_flat.reshape(self.hypercontrol_shape))

    # ----------------------------------------------------- forward + backward

    def _iterate_impl(
        self,
        Us: Array,
        l_traj: Array,
        K_traj: Array,
        R: Float,
        gamma: Float,
        Jmax: Float,
        iover_in: Interval,
        x0_flat: Array,
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

        # ---- forward sweep: fixed-step scan, dt == dt_ctl ----
        symplectic = self.solver == "symplectic"

        def forward_step(y_flat, k):
            t = self.ts_grid[k]
            idx = jnp.minimum(k, N - 1)
            x_pert, x_nom = self._split(y_flat)
            U_nom = Us[idx].reshape(-1)
            U_pert = U_nom - gamma * l_traj[idx] - K_traj[idx] @ (x_pert - x_nom)
            ix_k = iover_in[k] if track else None
            # lax.cond (not where) skips both dynamics evals once the set blows up;
            # most steps are dead on early iterations, which is the runtime win.
            if symplectic:
                y_next = jax.lax.cond(
                    alive_fn(y_flat),
                    lambda: self._join(
                        self._step_flat(t, x_pert, U_pert, ix_k),
                        self._step_flat(t, x_nom, U_nom, ix_k),
                    ),
                    lambda: y_flat,
                )
            else:
                deriv = jax.lax.cond(
                    alive_fn(y_flat),
                    lambda: self._join(
                        self._f_flat(t, x_pert, U_pert, ix_k),
                        self._f_flat(t, x_nom, U_nom, ix_k),
                    ),
                    lambda: jnp.zeros_like(y_flat),
                )
                y_next = y_flat + dt * deriv
            return y_next, y_flat

        y0 = self._join(x0_flat, x0_flat)
        y_last, ys_part = jax.lax.scan(forward_step, y0, jnp.arange(N))
        sol_ys = jnp.concatenate([ys_part, y_last[None]], axis=0)  # (N+1, 2*Xlen)

        pert_traj = sol_ys[:, :Xlen]
        state_traj = sol_ys[:, Xlen:]  # nominal

        # ifinal = last grid index still alive (once dead, stays dead).
        alive_grid = jax.vmap(alive_fn)(sol_ys)
        alive_cum = jnp.cumprod(alive_grid.astype(jnp.int32))
        ifinal = jnp.maximum(jnp.sum(alive_cum) - 1, 0)

        # Perturbed control per step, padded with nominal Us[i] past ifinal so
        # Us_new (next iter's input) stays finite.
        def per_step_U(i):
            U_nom = Us[i].reshape(-1)
            U_pert = U_nom - gamma * l_traj[i] - K_traj[i] @ (pert_traj[i] - state_traj[i])
            return jnp.where(i < ifinal, U_pert.reshape(self.hypercontrol_shape), Us[i])

        Us_new = jax.vmap(per_step_U)(jnp.arange(N))

        # ---- update the running per-timestep intersection box ----
        # Each alive grid point's perturbed set is a valid enclosure of R(t_k),
        # so intersecting its hull keeps a valid (tighter) box. Skip steps past
        # ifinal (blown up). The box only shrinks and feeds the next iteration.
        if track:
            new_box = jax.vmap(self._iover_box)(pert_traj)  # Interval (N+1, n)
            box_lo = jnp.maximum(iover_in.lower, new_box.lower)
            box_up = jnp.minimum(iover_in.upper, new_box.upper)
            alive_k = (jnp.arange(N + 1) <= ifinal)[:, None]
            iover_out = interval(
                jnp.where(alive_k, box_lo, iover_in.lower),
                jnp.where(alive_k, box_up, iover_in.upper),
            )
        else:
            iover_out = iover_in

        # ---- backward sweep: fused Euler scan over the costate (sig, s, S),
        # emitting per-step gains (l_i, K_i) in the same pass. Linearization is at
        # the perturbed trajectory under the nominal control.
        x_final = pert_traj[ifinal]
        sig_T = self._terminal_flat(x_final)
        s_T = jax.grad(self._terminal_flat)(x_final)
        S_T = jax.hessian(self._terminal_flat)(x_final)

        def backward_step(carry, U_p):
            sig, s, S, i = carry  # i = state index of this step
            ii = jnp.clip(i, 0, N)
            mask = i < ifinal
            t = self.ts_grid[ii]
            x_flat = pert_traj[ii]
            U_flat = U_p.reshape(-1)

            # No intersection box here: the gains must match the dynamics the
            # rollout actually follows, not the tightened forward contraction.
            # Symplectic: linearize the discrete-equivalent rate (step(x,U) - x)/dt,
            # so fx/fu (and DDP terms) are exact for the map the rollout follows.
            if symplectic:
                f_flat_bwd = lambda t_, x_, u_: (self._step_flat(t_, x_, u_) - x_) / dt
            else:
                f_flat_bwd = lambda t_, x_, u_: self._f_flat(t_, x_, u_)

            # lax.cond skips the expensive fx/fu Jacobians on masked steps (past
            # ifinal); most of the horizon is masked early on.
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

        # The final reverse-scan carry's costate is s(0) = d(terminal_cost)/dx0
        # propagated to t0 (used for a free-initial-state update by the caller).
        (_, costate0, _, _), (l_new, K_new) = jax.lax.scan(
            backward_step, (sig_T, s_T, S_T, N), Us, length=N, reverse=True
        )

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
            iover=iover_out,
            costate0=costate0,
        )

    # --------------------------------------------------------- public surface

    def initial_gains(self) -> tuple[Array, Array]:
        """Convenience: zero-initialized ``(l_traj, K_traj)`` of the right shape."""
        l_traj = jnp.zeros((self.N, self.Ulen))
        K_traj = jnp.zeros((self.N, self.Ulen, self.Xlen))
        return l_traj, K_traj

    def initial_controls(self) -> Array:
        """Convenience: zero ``Us`` of shape ``(N, *hypercontrol_shape)``."""
        return jnp.zeros((self.N, *self.hypercontrol_shape))

    def initial_iover(self) -> Interval:
        """The unbounded per-timestep box ``[-inf, +inf]`` (an Interval of shape
        ``(N+1, n)``), so the first iteration's intersection is just its own
        ``iover()``."""
        # Pin the dtype: a Python-scalar fill (``jnp.full(..., -jnp.inf)``) yields
        # a weak_type array, whereas the box returned by ``iterate`` is strong;
        # the mismatch would force a recompile on the first fed-back iteration.
        dtype = self.x0_flat.dtype
        return interval(
            jnp.full((self.N + 1, self.n), -jnp.inf, dtype=dtype),
            jnp.full((self.N + 1, self.n), jnp.inf, dtype=dtype),
        )

    def setup(self) -> "ReachiLQR":
        """Warm up the JIT: run one ``iterate`` and block until it finishes, so
        the one-time compile cost is paid here and excluded from any subsequently
        timed run. The runtime values of ``R``/``gamma``/``Jmax`` do not affect
        the compiled program, so placeholders are used; the only effect is
        populating the jit cache. Returns ``self`` for chaining.
        """
        res = self.iterate(self.initial_controls(), *self.initial_gains(), 1.0)
        jax.block_until_ready(res)
        return self

    def iterate(
        self,
        Us: Array,
        l_traj: Array,
        K_traj: Array,
        R: Float,
        *,
        gamma: Float = 1.0,
        Jmax: Float = jnp.inf,
        iover: Interval | None = None,
        x0: Array | None = None,
    ) -> IterateResult:
        """Run one outer iLQR/DDP iteration.

        When ``track_iover`` is enabled, pass the previous iteration's ``iover``
        (from the returned :class:`IterateResult`) to accumulate the running
        cross-iteration intersection box; if omitted, the box is (re)initialized
        to ``[-inf, +inf]``.

        ``x0`` (flat ``(pt, aux)`` state) overrides the initial condition for the
        forward sweep; defaults to the fixed ``pt0`` from construction. Pass the
        running value to co-optimize the initial set via a free-initial-state step
        using the returned :attr:`IterateResult.costate0`.
        """
        if iover is None:
            iover = self.initial_iover()
        if x0 is None:
            x0 = self.x0_flat
        return self._iterate_jit(
            Us,
            l_traj,
            K_traj,
            jnp.asarray(R),
            jnp.asarray(gamma),
            jnp.asarray(Jmax),
            iover,
            x0,
        )

    def run(
        self,
        Us0: Array | None = None,
        *,
        R_schedule: Callable[[int], tuple] | Float,
        iters: int | None = None,
        gamma: Float = 1.0,
        Jmax: Float = jnp.inf,
        on_iterate: Callable[[int, Float, IterateResult], None] | None = None,
        verbose: bool = False,
    ) -> RunResult:
        """Outer R-schedule loop with best-tracking.

        ``R_schedule`` is a callable ``i -> (R, revert_best)`` (e.g. from
        :func:`phased_schedule`); a scalar is wrapped via
        :func:`constant_schedule`. ``revert_best`` resets the working ``Us`` to
        the best-seen-so-far after that iteration (used at phase boundaries).
        ``iters`` defaults to the schedule's ``.total`` when present.
        ``on_iterate(i, R, result)``, if given, is called each iteration with the
        raw :class:`IterateResult` (e.g. to record trajectories for plotting).
        """
        schedule = (
            R_schedule if callable(R_schedule) else constant_schedule(R_schedule)
        )
        if iters is None:
            iters = getattr(schedule, "total", None)
        if iters is None:
            raise ValueError(
                "run() needs `iters` (or an R_schedule with a `.total`, e.g. from "
                "phased_schedule)."
            )
        iters = int(iters)

        Us = self.initial_controls() if Us0 is None else Us0
        l_traj, K_traj = self.initial_gains()
        # The intersection box accumulates across all iterations and is never
        # reverted: every iterate contributes a valid enclosure.
        iover = self.initial_iover()

        best_res = None
        best_Us = None
        history: list = []

        for i in range(iters):
            R, revert_best = schedule(i)
            R = float(R)
            res = self.iterate(
                Us, l_traj, K_traj, R, gamma=gamma, Jmax=Jmax, iover=iover
            )
            iover = res.iover
            ifinal = int(res.ifinal)
            cost = float(res.cost_final)
            history.append({"iter": i, "R": R, "ifinal": ifinal, "cost": cost})

            if best_res is None or (ifinal, -cost) > (
                int(best_res.ifinal),
                -float(best_res.cost_final),
            ):
                best_res, best_Us = res, res.Us_new

            if on_iterate is not None:
                on_iterate(i, R, res)
            if verbose:
                print(f"i={i:>5d}  R={R:>8.3g}  ifinal={ifinal:>5d}  cost={cost: .4f}")

            # Advance Us, reverting to the best-seen at a phase boundary.
            Us = best_Us if revert_best else res.Us_new
            l_traj = res.l_traj
            K_traj = res.K_traj

        assert best_res is not None
        return RunResult(best=best_res, Us=best_Us, history=history, iover=iover)
