# CLAUDE.md

immrax: interval analysis and reachability in JAX. Everything it computes is a
guaranteed over-approximation; soundness is the invariant that matters.

## Setup and checks

```bash
uv sync                      # dev env; system deps: libcdd-dev libgmp-dev (pypoman)
uv sync --extra sos          # MOSEK, for ChebyshevPolytopeEmbedding(bound="sos")
git config core.hooksPath .githooks   # ruff check --fix + format on commit
uv run pytest                # ~2 min; CI also runs examples/getting_started/compare.py
uv run ruff check . && uv run ruff format .
```

Both jax 0.9 and 0.10 must work (the conda env is 0.9, uv resolves 0.10).

## How the core works

- `natif` and `linbp` are jaxpr interpreters. A primitive is supported only if it
  has a rule: `inclusion_registry` (nif.py), `linbp_registry` and
  `backward_registry` (linbp.py). New jax versions add parameters (`out_dtype` on
  `mul`) and primitives (`stack`); route version differences through
  `inclusion/_jax_compat.py`, never `get_bind_params`/`pjit_p` directly.
- Scalar activations in linbp have one relaxation `_relax_<f>(l, u) -> (alpha,
  beta_l, beta_u)` in `_ACTIVATIONS`, shared by the forward and backward passes.
  Add new ones there, not as separate forward/backward handlers.
- Rigorous mode (ULP widening) is off by default; `rigorous=True`, the
  `rigorous()` context, or `set_rigorous`.
- Parametric sets are `Parametope`s evolved by a `ParametricEmbedding`.
  `compute_reachset` defaults to `solver="symplectic"`: the frame alpha is the
  exact discrete adjoint of the state map. Embeddings implement the `_rk_*` hooks
  or `_symplectic_step`.
- `System` is an equinox Module: declare fields; undeclared attributes set in
  `__init__` still work but warn and are static (not traced).

## Conventions

- `import immrax as irx`, `jax.numpy as jnp`, `numpy as onp`.
- Offsets `y = (-ly, uy)` stacked, with `.ly`/`.uy` properties (`Polytope`,
  `ChebyshevPolytope`); scaled chart coordinates are `z`; `l, u` are interval
  bounds.
- Terse comments and docstrings; no commented-out code.
- Tests are soundness properties over sampled points (allow float32 round-off,
  ~1e-5). Add a case to an existing table (`CASES` in test_linbp.py, `FNS` in
  test_rigorous.py) rather than a new test function.
- Examples live in `examples/<topic>/`, write their figures next to the script
  (tracked, used by the docs), and have a page in `docs/source/examples_*.rst`.
  Notebooks are standalone `.ipynb`.
