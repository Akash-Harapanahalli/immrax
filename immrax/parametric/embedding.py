import jax
from jaxtyping import Integer, Float, ArrayLike
from typing import Union, List, Callable, Literal
from abc import abstractmethod
import equinox as eqx
from ..system import System
from ..system.system import LegacyAttrModule
from .parametope import Parametope
from immutabledict import immutabledict
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
import warnings


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

    @eqx.filter_jit
    def compute_reachset(
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        pt0: Parametope,
        inputs: List[Callable[[int, jax.Array], jax.Array]] = [],
        dt: float = 0.01,
        *,
        solver: Union[Literal["euler", "rk45", "tsit5"], AbstractSolver] = "tsit5",
        f_kwargs: immutabledict = immutabledict({}),
        **kwargs,
    ):
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

        aux0 = self._initialize(pt0)

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
