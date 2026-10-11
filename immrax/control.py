import abc
import jax
from jaxtyping import Integer, Float
from typing import Union
from immrax.system import System, OpenLoopSystem

__all__ = [
    "Control",
    "ControlledSystem",
    "LinearControl",
]


class Control(abc.ABC):
    """Control
    A feedback controller of the form :math:`u:\\mathbb{R}\\times\\mathbb{R}^n\\to\\mathbb{R}^p`.
    """

    @abc.abstractmethod
    def u(self, t: Union[Integer, Float], x: jax.Array) -> jax.Array:
        """Feedback Control Output

        Parameters
        ----------
        t:Union[Integer, Float] :

        x:jax.Array :


        Returns
        -------

        """


class LinearControl(Control):
    K: jax.Array

    def __init__(self, K: jax.Array) -> None:
        self.K = K

    def u(self, t: Union[Integer, Float], x: jax.Array) -> jax.Array:
        return self.K @ x


class ControlledSystem(System):
    """ControlledSystem
    A closed-loop nonlinear dynamical system of the form

    .. math::

        \\dot{x} = f^{\\textsf{c}}(x,w) = f(x,N(x),w),

    where :math:`N:\\mathbb{R}^n \\to \\mathbb{R}^p`.
    """

    olsystem: OpenLoopSystem
    control: Control

    def __init__(self, olsystem: OpenLoopSystem, control: Control) -> None:
        self.olsystem = olsystem
        self.control = control
        self.evolution = olsystem.evolution
        self.xlen = olsystem.xlen

    def f(self, t: Union[Integer, Float], x: jax.Array, *args) -> jax.Array:
        """Returns the value of the closed loop system

        Parameters
        ----------
        t : Union[Integer, Float]
            time value
        x : jax.Array
            state value
        w : jax.Array
            disturbance value

        Returns
        -------
        jax.Array
            :math:`f^{\\textsf{c}}(x,w) = f(x,N(x),w)`

        """
        return self.olsystem.f(t, x, self.control.u(t, x), *args)
