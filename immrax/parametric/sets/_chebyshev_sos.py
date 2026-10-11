r"""Validated SOS bounds over a Chebyshev polytope (MOSEK).

For each objective ``p`` and sign ``s`` this solves

.. math::
    \min \gamma \ \text{s.t.}\ \gamma - s p = \sigma_0
    + \sum_j \sigma_j^+ (u_j - \alpha_j b) + \sigma_j^- (\alpha_j b - \ell_j)
    + \sum_k \tau_k (1 - w_k^2),

with SOS ``sigma``, ``tau`` written as Gram matrices in the Chebyshev basis and
the identity matched coefficient-wise (exact Chebyshev product rules). The
returned bound is validated: ``gamma`` plus the coefficient residual and the
negative-eigenvalue slack of each Gram matrix, each bounded on the set using
``|T_beta| <= 1``.
"""

import functools
import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as onp

from .chebyshev_polytope import cheb_indices


def _prod(a, b):
    """``T_a T_b = 2^-n sum_signs T_{|a +- b|}``."""
    n, out = len(a), {}
    for signs in itertools.product((1, -1), repeat=n):
        e = tuple(abs(a[k] + s * b[k]) for k, s in enumerate(signs))
        out[e] = out.get(e, 0.0) + 0.5**n
    return out


def _gram_map(E, idx, mult):
    """``(NC, L, L)``: coefficients of ``mult * sum_ab Q_ab T_a T_b`` (symmetrized)."""
    L = len(E)
    S = onp.zeros((len(idx), L, L))
    for i, a in enumerate(E):
        for j, b in enumerate(E):
            for e, c in _prod(a, b).items():
                for e2, c2 in mult.items():
                    for e3, c3 in _prod(e, e2).items():
                        if e3 in idx:
                            S[idx[e3], i, j] += c * c2 * c3
    return 0.5 * (S + S.transpose(0, 2, 1))


def _lower(T):
    kk, ll = onp.tril_indices(T.shape[-1])
    return T[..., kk, ll], kk, ll


def _unpack(v, L):
    """MOSEK's lower triangle, column-major -> symmetric matrix."""
    rows = onp.concatenate([onp.arange(j, L) for j in range(L)])
    cols = onp.concatenate([onp.full(L - j, j) for j in range(L)])
    M = onp.zeros((L, L))
    M[rows, cols] = v
    return M + onp.tril(M, -1).T


@functools.lru_cache(maxsize=None)
def solver(n: int, m: int) -> "_Solver":
    return _Solver(n, m)


class _Solver:
    """One single-threaded MOSEK task per (objective, sign), solved in parallel."""

    def __init__(self, n: int, m: int, workers: int = 12):
        import mosek

        self.mosek, self.n, self.m, D = mosek, n, m, 2 * m
        faces = cheb_indices(n, m)
        self.P = len(faces)
        ED = cheb_indices(n, D)
        idx = {tuple(e): i for i, e in enumerate(ED)}
        self.NC, self.z = len(ED), idx[(0,) * n]
        z = (0,) * n
        e0, e1, e2 = (cheb_indices(n, d) for d in (D // 2, (D - m) // 2, (D - 2) // 2))
        self.S0 = _gram_map(e0, idx, {z: 1.0})
        box = lambda k: {
            z: 0.5,
            tuple(2 * (i == k) for i in range(n)): -0.5,
        }  # 1 - w_k^2
        self.Sb = onp.stack([_gram_map(e2, idx, box(k)) for k in range(n)])
        self.Pi = onp.stack([_gram_map(e1, idx, {tuple(b): 1.0}) for b in faces])
        self.sizes = [len(e0)] + [len(e1)] * (2 * self.P) + [len(e2)] * n

        V0, k0, l0 = _lower(self.S0)
        Vb, kb, lb = _lower(self.Sb)
        c0, t0 = onp.nonzero(V0)
        stat = [(c0, onp.zeros(len(c0), int), k0[t0], l0[t0], -V0[c0, t0])]
        for k in range(n):
            c, t = onp.nonzero(Vb[k])
            stat.append(
                (c, onp.full(len(c), 1 + 2 * self.P + k), kb[t], lb[t], -Vb[k][c, t])
            )
        self.static = [onp.concatenate(a) for a in zip(*stat)]
        # face blocks: always write every entry any face can touch (zeros included),
        # since reused tasks keep entries that are not overwritten
        Vp = _lower(self.Pi)[0]
        self.face_ct = onp.nonzero(onp.any(Vp != 0, axis=0))
        self.env, self.tasks = mosek.Env(), []
        self.pool = ThreadPoolExecutor(workers)

    def _task(self):
        mk = self.mosek
        t = self.env.Task()
        t.appendvars(1)
        t.appendcons(self.NC)
        t.appendbarvars(self.sizes)
        t.putvarbound(0, mk.boundkey.fr, -1e30, 1e30)
        t.putcj(0, 1.0)
        t.putaij(self.z, 0, 1.0)
        t.putobjsense(mk.objsense.minimize)
        t.putintparam(mk.iparam.num_threads, 1)
        return t

    def solve(self, alpha, lo, hi, Pc) -> onp.ndarray:
        """``Pc (NC, Q)`` objective coefficients -> validated ``[sup p, sup -p]`` ``(2, Q)``."""
        mk, P, NC = self.mosek, self.P, self.NC
        alpha, lo, hi, Pc = (onp.asarray(a, dtype=float) for a in (alpha, lo, hi, Pc))
        Q = Pc.shape[1]
        while len(self.tasks) < 2 * Q:
            self.tasks.append(self._task())

        A = onp.einsum("jb,bcxy->jcxy", alpha, self.Pi)
        one = self.Pi[0][None]  # the T_0 map
        Mf = onp.stack(
            [hi[:, None, None, None] * one - A, A - lo[:, None, None, None] * one], 1
        )
        Mf = Mf.reshape(2 * P, NC, *A.shape[2:])
        Vf, kf, lf = _lower(Mf)
        c, t_ = (onp.tile(a, 2 * P) for a in self.face_ct)
        q = onp.repeat(onp.arange(2 * P), len(self.face_ct[0]))
        I, J, K, L, V = (
            onp.concatenate([s, x])
            for s, x in zip(self.static, (c, 1 + q, kf[t_], lf[t_], -Vf[q, c, t_]))
        )
        I, J, K, L = (a.astype(onp.int32) for a in (I, J, K, L))
        width = onp.repeat(hi - lo, 2)
        ntot = sum(s * (s + 1) // 2 for s in self.sizes)
        ev = lambda M: max(0.0, -onp.linalg.eigvalsh(M)[0])

        def run(s):
            sign, p = (1.0 if s < Q else -1.0), Pc[:, s % Q]
            t = self.tasks[s]
            t.putbarablocktriplet(I, J, K, L, V)
            b = (sign * p).tolist()
            t.putconboundslice(0, NC, [mk.boundkey.fx] * NC, b, b)
            t.optimize()
            g = t.getxx(mk.soltype.itr)[0]
            flat = onp.array(t.getbarxslice(mk.soltype.itr, 0, len(self.sizes), ntot))
            X = onp.split(
                flat, onp.cumsum([s_ * (s_ + 1) // 2 for s_ in self.sizes])[:-1]
            )
            X = [_unpack(x, L_) for x, L_ in zip(X, self.sizes)]
            E = (
                -sign * p
                - onp.einsum("cij,ij->c", self.S0, X[0])
                - onp.einsum("qcij,qij->c", Mf, onp.stack(X[1 : 1 + 2 * P]))
                - onp.einsum("kcij,kij->c", self.Sb, onp.stack(X[1 + 2 * P :]))
            )
            E[self.z] += g
            pen = onp.abs(E).sum() + ev(X[0]) * self.sizes[0]
            pen += sum(ev(X[1 + j]) * self.sizes[1] * width[j] for j in range(2 * P))
            pen += sum(ev(X[1 + 2 * P + k]) * self.sizes[-1] for k in range(self.n))
            return g + pen

        out = onp.array(list(self.pool.map(run, range(2 * Q))))
        return out.reshape(2, Q)
