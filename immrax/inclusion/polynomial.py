import jax
import jax.numpy as jnp
from immrax.inclusion.interval import interval, Interval
from immrax.inclusion import nif, custom_if


@custom_if
def polynomial(a, x):
    return jnp.polyval(a, x)


@polynomial.defif
def polynomial_inclusion(a, x):
    """Tight inclusion for constant coefficients ``a``: the min/max over the
    endpoints of ``x`` and the real critical points inside it. Interval
    coefficients fall back to the natural inclusion function."""
    if isinstance(a, Interval):
        return nif.natif(jnp.polyval)(a, x)

    ad_roots = jnp.roots(jnp.polyder(a), strip_zeros=False)
    crit_vals = jnp.real(jax.vmap(jnp.polyval, in_axes=(None, 0))(a, ad_roots))
    lo, hi = jnp.atleast_1d(x.lower), jnp.atleast_1d(x.upper)
    crit_in_x = jax.vmap(lambda c: (c > lo) & (c < hi) & jnp.isreal(c))(ad_roots)
    end_vals = jnp.array([jnp.polyval(a, lo), jnp.polyval(a, hi)])
    l_vals = jnp.concatenate(
        (end_vals, jnp.where(crit_in_x, crit_vals[:, None], jnp.inf))
    )
    u_vals = jnp.concatenate(
        (end_vals, jnp.where(crit_in_x, crit_vals[:, None], -jnp.inf))
    )
    return interval(jnp.min(l_vals, axis=0), jnp.max(u_vals, axis=0))
