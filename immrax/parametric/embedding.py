import jax
import jax.numpy as jnp
from jaxtyping import Integer, Float, ArrayLike
from typing import Any, Union, List, Callable, Literal
from abc import abstractmethod
import equinox as eqx
from ..system import System
from ..system.system import LegacyAttrModule
from .parametope import Parametope
from immutabledict import immutabledict
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
import warnings


class ReachsetSolution(eqx.Module):
    """Fixed-step scan solution mirroring the diffrax ``Solution`` fields callers use.

    ``ts`` is all-finite with length ``N+1``; ``ys`` is the ``(pt, aux)`` pytree
    with a leading step axis (``ys[0].alpha[k]`` etc.).
    """

    ts: jax.Array
    ys: Any


class ParametricEmbedding(LegacyAttrModule):
    """Base class for embeddings that lift a :class:`~immrax.system.System` to
    dynamics over a parametric set representation (:class:`Parametope`).

    ``ParametricEmbedding`` is an :class:`equinox.Module`, so a configured
    embedding is a JAX pytree: the wrapped ``sys`` (and any array-valued fields
    declared by a subclass) are pytree leaves, enabling ``jit``/``vmap`` over the
    embedding as a whole.

    Subclasses implement :meth:`_initialize` (build the auxiliary state evolved
    alongside the parametope, for a particular initial set ``pt0``) and
    :meth:`_dynamics` (the embedding right-hand side). :meth:`_initialize` must be
    a pure function of ``pt0`` returning ``aux0`` — it must not mutate ``self``.
    """

    sys: System

    @abstractmethod
    def _initialize(self, pt0: Parametope) -> ArrayLike:
        """Initialize the embedding for a particular initial set ``pt0``.

        This must be a pure function: it returns the auxiliary state ``aux0`` to
        evolve alongside the parametope and must *not* mutate ``self`` (the
        embedding is an immutable :class:`equinox.Module`).

        Parameters
        ----------
        pt0 : Parametope
            The initial set.

        Returns
        -------
        ArrayLike
            ``aux0``: auxiliary states to evolve with the embedding system.
        """

    @abstractmethod
    def _dynamics(self, t, state, *args):
        """Embedding right-hand side: the dynamics of ``(parametope, aux)``."""

    def _symplectic_step(self, t, dt, state, *args, U=None, **kwargs):
        """One semi-implicit (symplectic) Euler step ``state_k -> state_{k+1}``.

        ``args`` are per-step input values already evaluated at ``(t, state)``;
        ``U`` is a constant hypercontrol; extra kwargs forward to ``_dynamics``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support solver='symplectic'; "
            "pass a diffrax solver instead, e.g. solver='tsit5' or 'euler'."
        )

    def hypercontrol_shape(self, pt0: Parametope) -> tuple:
        """Shape of the control input :meth:`_dynamics` accepts for ``pt0``.

        The control enters :meth:`_dynamics` in an embedding-specific shape that
        is not otherwise discoverable from the generic ``_dynamics(t, state,
        *args)`` signature. Consumers such as :class:`ReachiLQR` use this to size
        their gains and to reshape their internal flat control vector back to what
        ``_dynamics`` expects. Subclasses that support a control override this;
        the default signals that the embedding exposes no control.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define hypercontrol_shape; it "
            "exposes no control input for synthesis."
        )

    def iover(self, state):
        """Interval overapproximation of the set for embedding state ``(pt, aux)``.

        The default delegates to the parametope's own ``iover()``. Embeddings
        whose enclosure needs auxiliary state (e.g. a maintained inverse ``H+``)
        override this. Consumers such as :class:`ReachiLQR` call it as
        ``embedding.iover((pt, aux))``.
        """
        pt, _aux = state
        return pt.iover()

    @eqx.filter_jit
    def compute_reachset(
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        pt0: Parametope,
        inputs: List[Callable[[int, jax.Array], jax.Array]] = [],
        dt: float = 0.01,
        *,
        solver: Union[
            Literal["symplectic", "euler", "rk45", "tsit5"], AbstractSolver
        ] = "symplectic",
        f_kwargs: immutabledict = immutabledict({}),
        **kwargs,
    ):
        """Flow the embedding from ``pt0`` over ``[t0, tf]``.

        The default ``solver="symplectic"`` runs a fixed-step semi-implicit
        Euler scan: explicit Euler on the center, an implicit adjoint step
        ``alpha_{k+1} (I + dt A_k) = alpha_k + dt U_k`` with ``A_k`` at the old
        center (the exact discrete adjoint of the Euler state map, so the
        alpha/state pairing is preserved — machine-exact for linear systems),
        and the offset bound evaluated with the discrete rate
        ``(alpha_{k+1} - alpha_k)/dt``. It requires concrete ``t0, tf, dt``
        (grid ``N = round((tf-t0)/dt)`` hits ``tf`` exactly; effective step may
        differ slightly from ``dt``) and returns a :class:`ReachsetSolution`
        with all-finite ``ts`` of length ``N+1``. Embeddings without a
        ``_symplectic_step`` (Gram/Polynomial) raise ``NotImplementedError``;
        pass a diffrax solver there. Diffrax solvers return the diffrax
        ``Solution`` with inf-padded ``ts`` as before.
        """
        aux0 = self._initialize(pt0)

        if solver == "symplectic":
            if kwargs:
                raise TypeError(
                    f"solver='symplectic' does not accept diffrax kwargs: {sorted(kwargs)}"
                )
            try:
                t0_, tf_, dt_ = float(t0), float(tf), float(dt)
            except TypeError as e:
                raise TypeError(
                    "solver='symplectic' requires concrete (non-traced) t0, tf, dt"
                ) from e
            N = max(1, int(round((tf_ - t0_) / dt_)))
            h = (tf_ - t0_) / N
            ts = t0_ + h * jnp.arange(N + 1)
            state0 = jax.tree.map(jnp.asarray, (pt0, aux0))

            def body(state, tk):
                args = [u(tk, state) for u in inputs]
                new = self._symplectic_step(tk, h, state, *args, **f_kwargs)
                return new, new

            _, tail = jax.lax.scan(body, state0, ts[:-1])
            ys = jax.tree.map(
                lambda x0, xs: jnp.concatenate([x0[None], xs], axis=0), state0, tail
            )
            return ReachsetSolution(ts=ts, ys=ys)

        def func(t, x, args):
            return self._dynamics(t, x, *[u(t, x) for u in inputs], **f_kwargs)

        term = ODETerm(func)
        if solver == "euler":
            solver = Euler()
        elif solver == "rk45":
            solver = Dopri5()
        elif solver == "tsit5":
            solver = Tsit5()
        elif isinstance(solver, AbstractSolver):
            pass
        else:
            raise Exception(f"{solver=} is not a valid solver")

        saveat = SaveAt(t0=True, t1=True, steps=True)
        return diffeqsolve(
            term, solver, t0, tf, dt, (pt0, aux0), saveat=saveat, **kwargs
        )


class ParametopeEmbedding(ParametricEmbedding):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Class 'ParametopeEmbedding' is deprecated. Use 'ParametricEmbedding' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
