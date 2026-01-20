import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import jet
from abc import ABC, abstractmethod
from ..system import System
from .parametope import Parametope
from functools import partial
from typing import Callable

class ReachableTube (ABC) :
    @abstractmethod
    def __call__ (self, t) -> Parametope :
        """Slice the reachable tube at t"""
        pass

def fact (n) :
    return lax.exp(lax.lgamma(n + 1.))

def inv_fact (n) :
    return lax.exp(-lax.lgamma(n + 1.))

class ReachableSets (ABC) :
    ts: jax.Array
    sets: list
    @abstractmethod
    def __call__ (self, t) :
        """Get the reachable set at closest time index to t"""
        i = jnp.searchsorted(self.ts, t) - 1
        return jnp.where(t - self.ts[i] < self.ts[i+1] - t, self.sets[i], self.sets[i+1])
        

class PolyReachableTube (ReachableTube) :
    ts: jax.Array
    ox_coeffs: jax.Array
    alpha_coeffs: jax.Array
    y_coeffs: jax.Array
    ox_order: int
    alpha_order: int
    y_order: int

    def __init__ (self, ts, ox_coeffs, alpha_coeffs, y_coeffs) :
        self.ts = ts
        self.ox_coeffs = ox_coeffs
        self.alpha_coeffs = alpha_coeffs
        self.y_coeffs = y_coeffs
        self.ox_order = len(ox_coeffs)
        self.alpha_order = len(alpha_coeffs)
        self.y_order = len(y_coeffs)

    def __call__ (self, t) :
        i = jnp.searchsorted(self.ts, t) - 1
        h = t - i
        ox_nn = jnp.arange(self.ox_order)
        alpha_nn = jnp.arange(self.alpha_order)
        y_nn = jnp.arange(self.y_order)
        ox = jnp.sum(self.ox_coeffs * inv_fact(ox_nn) * h**ox_nn, axis=0)
        alpha = jnp.sum(self.alpha_coeffs * inv_fact(alpha_nn) * h**alpha_nn, axis=0)
        y = jnp.sum(self.y_coeffs * inv_fact(y_nn) * h**y_nn, axis=0)
        return Parametope(ox, alpha, y)

def prolongation (f:Callable, p:int) -> Callable :
    # Iteratively call jet with a growing series
    @jax.jit
    def f_prolonged (t, x, *args) :
        # Any remaining args at this stage are treated as constants for the Taylor series
        def _f (t, x) : f(t, x, *args)
        t_series = [t, 1.]
        x_series = [x, _f(t, x)]
        for k in range (p) :
            out1, out2 = jet(_f, (t_series[0], x_series[0]), (t_series[1:], x_series[1:]))
            t_series.append(0.)
            x_series.append(out2[-1])
        return x_series
    return f_prolonged

class ReachableSetGenerator (ABC) :
    sys: System

    @abstractmethod
    def step (self, t:float, set0:Parametope, f_args) -> Parametope :
        """Compute the next reachable set from set0 at time t"""
        pass

    @abstractmethod
    def compute_reach_sets (self, t0:float, tf:float, set0:Parametope, f_args) -> ReachableSets :
        """Compute the reachable sets over [t0, tf] starting from set0"""
        pass

class PolyReachableTubeGenerator :
    sys: System

    def __init__ (self, sys: System, dt:float, ox_order:int, alpha_order:int=0, y_order:int=1) :
        self.sys = sys
        self.dt = dt
        self.ox_order = ox_order
        self.alpha_order = alpha_order
        self.y_order = y_order
        self.prolonged_f = prolongation(sys.f, ox_order)

    def get_rough_enclosure (self, pt:Parametope) :
        # Rough enclosure is a function of parametope and dt
        # TODO: Implement actual rough enclosure
        return pt.iover()

    def step (self, t:float, pt0:Parametope, f_args) :
        rough_enclosure = self.get_rough_enclosure(pt0)
        series = self.prolonged_f(t, pt0.ox, *f_args)
        def f_pp1 (t, x) : 
            return jet(self.sys.f, (t, x), ([1.] + [0.]*(len(series)-1), series))[-1]
        

    # def compute_reachable_tube (self) -> PolyReachableTube :

