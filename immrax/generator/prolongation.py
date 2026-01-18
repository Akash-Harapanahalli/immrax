import jax
import jax.numpy as jnp
from jax.experimental import jet
from typing import Callable

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
