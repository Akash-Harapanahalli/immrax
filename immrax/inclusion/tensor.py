import math
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from itertools import combinations_with_replacement
from .interval import Interval, interval, isinterval
from .jacobian import Permutation, standard_permutation
from .nif import natif


def get_multiindices(n, p):
    """Yield all multi-indices α with |α| = p over n variables."""
    for idx in combinations_with_replacement(range(n), p):
        counts = [0] * n
        for i in idx:
            counts[i] += 1
        yield MultiIndex(*counts)


class MultiIndex(tuple):
    """Multi-index α = (α₁, ..., αₙ) with αᵢ ∈ ℕ₀.

    Stores the count vector so that standard multi-index notation applies:
      |α| = Σαᵢ,  α! = Π(αᵢ!),  x^α = Πxᵢ^αᵢ, etc.
    """

    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __repr__(self):
        return f"MultiIndex{super().__repr__()}"

    # --- Component-wise arithmetic ---

    def __add__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return MultiIndex(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return MultiIndex(*(a - b for a, b in zip(self, other)))

    # --- Component-wise ordering (overrides tuple's lexicographic order) ---

    def __le__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a <= b for a, b in zip(self, other))

    def __lt__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a < b for a, b in zip(self, other))

    def __ge__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a >= b for a, b in zip(self, other))

    def __gt__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return all(a > b for a, b in zip(self, other))

    # --- Multi-index operations ---

    def __abs__(self):
        """Order of the multi-index: |α| = Σαᵢ."""
        return sum(self)

    def factorial(self):
        """α! = α₁! · α₂! · ... · αₙ!"""
        result = 1
        for a in self:
            result *= math.factorial(a)
        return result

    def inv_factorial(self):
        """1/α! = exp(-Σᵢ lgamma(αᵢ + 1)), numerically stable via log-gamma."""
        return math.exp(-sum(math.lgamma(a + 1) for a in self))

    def binomial(self, other):
        """Binomial coefficient C(α, β) = α! / (β! · (α−β)!), requires β ≤ α."""
        if not (other <= self):
            raise ValueError(f"β={other} must be ≤ α={self} component-wise")
        diff = self - other
        return self.factorial() // (other.factorial() * diff.factorial())

    def multinomial(self):
        """Multinomial coefficient: |α|! / α!"""
        return math.factorial(abs(self)) // self.factorial()

    def power(self, x):
        """x^α = x₁^α₁ · x₂^α₂ · ... · xₙ^αₙ, requires len(x) == len(α)."""
        if len(x) != len(self):
            raise ValueError(f"len(x)={len(x)} must equal len(α)={len(self)}")
        result = 1
        for xi, ai in zip(x, self):
            result = result * xi**ai
        return result


@register_pytree_node_class
class SparseLowerTriangularTensor:
    """Sparse representation of a lower triangular tensor.

    A lower triangular tensor of order p over R^n stores entries only for sorted
    index tuples i₁ ≤ i₂ ≤ ... ≤ iₚ. Every such sorted tuple corresponds
    bijectively to a multi-index α = (α₁, ..., αₙ) where αⱼ counts how many times
    index j appears, so we can index entries by MultiIndex keys. Backed by a dict
    mapping MultiIndex → jax.Array.
    """

    def __init__(self, p: int, m: tuple, data: dict | None = None):
        self.p = p
        self.m = m
        self.data: dict[MultiIndex, jax.Array] = {} if data is None else data

    def __setitem__(self, key: MultiIndex, value: jax.Array):
        if abs(key) != self.p:
            raise ValueError(
                f"expected multi-index of order {self.p}, got |α|={abs(key)}"
            )
        self.data[key] = value

    def __getitem__(self, key: MultiIndex) -> jax.Array:
        if abs(key) != self.p:
            raise ValueError(
                f"expected multi-index of order {self.p}, got |α|={abs(key)}"
            )
        if key not in self.data:
            return jnp.zeros(self.m)
        return self.data[key]

    def __contains__(self, key: MultiIndex) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"SparseLowerTriangularTensor(p={self.p}, m={self.m}, {self.data!r})"

    def contract(self, v, scale=False):
        """Contract with v^⊗p: Σ_{|α|=p} data[α] · v^α.

        Parameters
        ----------
        v : array or Interval
            Vector to contract against.
        scale : bool
            If True, scale each term by 1/α! — i.e. compute the Taylor
            monomial Σ_{|α|=p} data[α]/α! · v^α. The inverse factorials are
            derived from static keys at trace time and fold into XLA constants.

        Uses natif to support both real and Interval inputs, as well as
        Interval-valued data entries.
        """
        if not self.data:
            return jnp.zeros(self.m)

        keys = list(self.data.keys())  # static, captured into closure
        inv_facts = (
            jnp.array([alpha.inv_factorial() for alpha in keys]) if scale else None
        )
        data_has_interval = any(isinterval(val) for val in self.data.values())
        vals = (
            jnp.stack(list(self.data.values()))
            if not data_has_interval
            else Interval(
                jnp.stack([interval(val).lower for val in self.data.values()]),
                jnp.stack([interval(val).upper for val in self.data.values()]),
            )
        )

        def _contract(vals, v):
            # alpha.power(v) expands to scalar ops v[0]**α₀ * v[1]**α₁ * ...
            # keeping each pow scalar, working around a shape bug in natif's
            # array pow handler.
            v_powers = jnp.stack([alpha.power(v) for alpha in keys])  # (K,)
            if inv_facts is not None:
                v_powers = v_powers * inv_facts
            return jnp.einsum("k,k...->...", v_powers, vals)

        if data_has_interval or isinterval(v):
            return natif(_contract)(vals, v)
        return _contract(vals, v)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def tree_flatten(self):
        keys = tuple(self.data.keys())
        return [self.data[k] for k in keys], (self.p, self.m, keys)

    @classmethod
    def tree_unflatten(cls, aux, children):
        p, m, keys = aux
        return cls(p, m, dict(zip(keys, children)))


def _last_nz_idx(alpha):
    """Index of the last (largest) nonzero component of a MultiIndex."""
    for i in reversed(range(len(alpha))):
        if alpha[i] > 0:
            return i
    raise ValueError(f"zero multi-index {alpha} has no nonzero index")


def ltdiff(f, p):
    """Function transform computing all partial derivative LTTs of f up to order p.

    Uses a bottom-up approach: a dict of functions {MultiIndex -> callable} is
    maintained at each level. For each α of order k, exactly one jax.jvp call is
    made to its parent function g_β in direction eⱼ (j = last nonzero index of α),
    giving T_k[α] = ∂g_β/∂xⱼ. No full dense Jacobian is ever formed.

    Parameters
    ----------
    f : Callable
        Function to differentiate, mapping R^n -> R^m.
    p : int
        Maximum derivative order.

    Returns
    -------
    Callable
        A function x -> list of p+1 SparseLowerTriangularTensors, where index k
        holds all k-th order partial derivatives of f at x.
    """

    def _ltdiff(x):
        n = x.shape[0]
        e = jnp.eye(n)
        f0 = f(x)
        m = f0.shape

        # Order 0: the function value, stored under the zero multi-index
        t0 = SparseLowerTriangularTensor(0, m)
        t0[MultiIndex(*([0] * n))] = f0
        result = [t0]

        # prev_fns: MultiIndex of order k-1 -> callable R^n -> R^m
        prev_fns = {MultiIndex(*([0] * n)): f}

        for k in range(1, p + 1):
            curr_fns = {}
            t_k = SparseLowerTriangularTensor(k, m)

            for alpha in get_multiindices(n, k):
                j = _last_nz_idx(alpha)
                beta = MultiIndex(*(alpha[i] - (1 if i == j else 0) for i in range(n)))

                # One JVP call in direction eⱼ only — no dense Jacobian formed
                def g_alpha(y, g=prev_fns[beta], ej=e[j]):
                    return jax.jvp(g, (y,), (ej,))[1]

                curr_fns[alpha] = g_alpha
                t_k[alpha] = g_alpha(x)

            result.append(t_k)
            prev_fns = curr_fns

        return result

    return _ltdiff


def taylor_approx(tensors, xc, x):
    """Evaluate the Taylor polynomial defined by a list of LTTs at x around xc.

    Parameters
    ----------
    tensors : list of SparseLowerTriangularTensor
        Output of ltdiff(f, p)(xc): tensors[k] holds all order-k partial
        derivatives of f at xc.
    xc : array
        The expansion center.
    x : array or Interval
        The point(s) at which to evaluate the polynomial.

    Returns
    -------
    array or Interval
        Σ_{k=0}^{p} Σ_{|α|=k} (∂^α f(xc) / α!) · (x - xc)^α
    """
    dx = x - xc
    return sum(t.contract(dx, scale=True) for t in tensors)


def mdit(f, p):
    """Function transform computing all partial derivative LTTs of f up to order p,
    as well as the Mixed Derivative Interval Tensor bounding the remainder of the
    Taylor expansion.

    Uses a bottom-up approach: a dict of functions {MultiIndex -> callable} is
    maintained at each level. For each α of order k, exactly one jax.jvp call is
    made to its parent function g_β in direction eⱼ (j = last nonzero index of α),
    giving T_k[α] = ∂g_β/∂xⱼ. No full dense Jacobian is ever formed.

    Finally, the MDIT is computed through mixed interval/fixed point evaluations of
    (p+1)-th order derivatives.

    Parameters
    ----------
    f : Callable
        Function to differentiate, mapping R^n -> R^m.
    p : int
        Maximum derivative order.

    Returns
    -------
    Callable
        A function x -> list of p+1 SparseLowerTriangularTensors, where index k
        holds all k-th order partial derivatives of f at x.
    """

    def _mdit(ix, xc, permutation=None):
        n = xc.shape[0]
        e = jnp.eye(n)
        f0 = f(xc)
        m = f0.shape

        if permutation is None:
            permutation = standard_permutation(n)[0]

        ## First compute the LTTs to order p

        # Order 0: the function value, stored under the zero multi-index
        t0 = SparseLowerTriangularTensor(0, m)
        t0[MultiIndex(*([0] * n))] = f0
        result = [t0]

        # prev_fns: MultiIndex of order k-1 -> callable R^n -> R^m
        prev_fns = {MultiIndex(*([0] * n)): f}

        for k in range(1, p + 1):
            curr_fns = {}
            t_k = SparseLowerTriangularTensor(k, m)

            for alpha in get_multiindices(n, k):
                j = _last_nz_idx(alpha)
                beta = MultiIndex(*(alpha[i] - (1 if i == j else 0) for i in range(n)))

                # One JVP call in direction eⱼ only — no dense Jacobian formed
                def g_alpha(y, g=prev_fns[beta], ej=e[j]):
                    return jax.jvp(g, (y,), (ej,))[1]

                curr_fns[alpha] = g_alpha
                t_k[alpha] = g_alpha(xc)

            result.append(t_k)
            prev_fns = curr_fns

        ## Now compute the MDIT.

        _z = ix.lower
        z_ = ix.upper
        zc = xc

        Z = interval(
            jnp.where(
                permutation.mtx,
                jnp.tile(_z, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
            jnp.where(
                permutation.mtx,
                jnp.tile(z_, (len(permutation), 1)),
                jnp.tile(zc, (len(permutation), 1)),
            ),
        )

        M = SparseLowerTriangularTensor(p + 1, m)

        for alpha in get_multiindices(n, p + 1):
            j = _last_nz_idx(alpha)
            beta = MultiIndex(*(alpha[i] - (1 if i == j else 0) for i in range(n)))

            def g_alpha(y, g=prev_fns[beta], ej=e[j]):
                return jax.jvp(g, (y,), (ej,))[1]

            # Vmap g_alpha over Z, then weighted sum by alpha(i)/(p+1).
            partials = jax.vmap(natif(g_alpha))(Z)
            res = natif(jnp.sum)((jnp.asarray(alpha) / (p + 1)) * partials)
            M[alpha] = res

        result.append(M)

        return result

    return _mdit
