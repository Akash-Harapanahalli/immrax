r"""Symplectic (partitioned) Runge--Kutta helpers for adjoint embeddings.

The nominal state runs the explicit tableau ``(A, b)``; the frame ``alpha``
runs its symplectic partner ``(Ahat, b)``, ``b_i Ahat_ij + b_j A_ji = b_i b_j``,
so ``alpha`` stays the exact discrete adjoint of the RK state map.
"""

import jax.numpy as jnp
import numpy as onp

TABLEAUS = {
    "euler": ([[0.0]], [1.0], [0.0]),
    "heun": ([[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5], [0.0, 1.0]),
    "rk4": (
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        [0.0, 0.5, 0.5, 1.0],
    ),
}

# compute_reachset / ReachiLQR solver string -> tableau name
SOLVERS = {
    "symplectic": "euler",
    "symplectic_heun": "heun",
    "symplectic_rk4": "rk4",
}


def get_tableau(name: str):
    """Static ``(A, b, c)`` numpy arrays for a named explicit tableau."""
    if name not in TABLEAUS:
        raise ValueError(
            f"unknown tableau {name!r}; expected one of {sorted(TABLEAUS)}"
        )
    return tuple(onp.asarray(v, dtype=float) for v in TABLEAUS[name])


def symplectic_partner(A, b):
    """``Ahat_ij = b_j - b_j A_ji / b_i`` (requires ``b_i > 0``)."""
    A, b = onp.asarray(A), onp.asarray(b)
    return b[None, :] - b[None, :] * A.T / b[:, None]


def solve_alpha_stages(alpha, Js, Us, Ahat, dt):
    r"""All implicit stages at once:
    ``alpha^i + dt sum_j Ahat_ij alpha^j J^j = alpha + dt sum_j Ahat_ij U^j``.

    Transposed, this is one ``(s n) x (s n)`` solve with ``K`` right-hand sides.
    Returns ``(s, K, n)``.
    """
    s, n = len(Js), alpha.shape[1]
    Js, Us = jnp.stack(Js), jnp.stack(Us)
    Ahat = jnp.asarray(Ahat, dtype=alpha.dtype)
    blocks = Ahat[:, :, None, None] * jnp.swapaxes(Js, 1, 2)[None]  # (s, s, n, n)
    L = jnp.eye(s * n, dtype=alpha.dtype) + dt * blocks.transpose(0, 2, 1, 3).reshape(
        s * n, s * n
    )
    R = alpha[None] + dt * jnp.einsum("ij,jkn->ikn", Ahat, Us)  # (s, K, n)
    X = jnp.linalg.solve(L, jnp.swapaxes(R, 1, 2).reshape(s * n, -1))
    return jnp.swapaxes(X.reshape(s, n, -1), 1, 2)


def stage_sensitivities(Js, Us, A, b, dt):
    r"""Point derivative of the RK state map and the discrete forcing.

    ``K^i = I + dt sum_{j<i} A_ij J^j K^j``, ``Phi = I + dt sum_i b_i J^i K^i``
    and ``G = dt sum_i b_i U^i K^i``, so ``alpha_next Phi = alpha + G``.
    """
    Id = jnp.eye(Js[0].shape[0], dtype=Js[0].dtype)
    Ks = []
    for i in range(len(b)):
        Ki = Id
        for j in range(i):
            if A[i, j] != 0.0:
                Ki = Ki + dt * A[i, j] * (Js[j] @ Ks[j])
        Ks.append(Ki)
    Phi = Id + dt * sum(b[i] * (Js[i] @ Ks[i]) for i in range(len(b)))
    G = dt * sum(b[i] * (Us[i] @ Ks[i]) for i in range(len(b)))
    return Phi, G


def aux_update(Phi, G, alpha, alpha_p, N, forced: bool):
    """alpha_p / N updates preserving alpha_p @ alpha = I and N @ alpha = 0.

    ``alpha_next = (alpha + G) Phi^-1``. Without forcing (G == 0) the updates
    are exact without an inverse. Square ``alpha + G`` uses inv (LU): cheaper
    and better-behaved under the iLQR backward pass.
    """
    if not forced:
        return Phi @ alpha_p, N
    F = alpha + G
    Fp = jnp.linalg.inv(F) if F.shape[0] == F.shape[1] else jnp.linalg.pinv(F)
    return Phi @ Fp, N - (N @ G) @ Fp
