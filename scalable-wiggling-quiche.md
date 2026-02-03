# Plan: Multi-Argument Taylor Models with Per-Argument Total Degree Bounds

## Summary

Refactor `taylor/taylor_model.py` to support functions `f(*args)` where each argument can have arbitrary shape, with polynomial order specified as per-argument total degree bounds.

**Example**: For `f(t, x)` where `t: ()` (scalar) and `x: (2,)`:
- `max_order = (2, 3)` means `|alpha_t| <= 2` and `|alpha_x| <= 3`
- Generates monomials `t^a * x1^b1 * x2^b2` where `a <= 2` and `b1 + b2 <= 3`

## Current vs Desired

| Aspect | Current | Desired |
|--------|---------|---------|
| Input | Single `x: (d,)` | `*args` with arbitrary shapes |
| Order | Per-variable `(k_0, ..., k_{d-1})` | Per-argument total degree `(k_arg0, k_arg1, ...)` |
| Exponents | Cartesian product: `x_i^{e_i}` where `e_i <= k_i` | Per-arg total degree: `sum(e_i for i in arg) <= k_arg` |

## Implementation Phases

### Phase 1: Add ArgumentStructure Data Class

**File:** `immrax/taylor/taylor_model.py` (add near top, after imports)

```python
@dataclass(frozen=True)
class ArgumentStructure:
    """Metadata mapping domain variables to function arguments."""
    arg_shapes: tuple[tuple[int, ...], ...]  # Shape of each argument

    @cached_property
    def arg_sizes(self) -> tuple[int, ...]:
        return tuple(math.prod(s) if s else 1 for s in self.arg_shapes)

    @cached_property
    def arg_starts(self) -> tuple[int, ...]:
        starts = [0]
        for size in self.arg_sizes[:-1]:
            starts.append(starts[-1] + size)
        return tuple(starts)

    @property
    def num_args(self) -> int:
        return len(self.arg_shapes)

    @property
    def total_dim(self) -> int:
        return sum(self.arg_sizes)

    def arg_slice(self, arg_idx: int) -> slice:
        start = self.arg_starts[arg_idx]
        return slice(start, start + self.arg_sizes[arg_idx])
```

### Phase 2: New Exponent Generation for Total Degree

**File:** `immrax/taylor/taylor_model.py` (add after existing `_generate_exponents_impl`)

```python
def _enumerate_total_degree(dim: int, max_deg: int) -> list[tuple[int, ...]]:
    """Generate all multi-indices in R^dim with total degree <= max_deg."""
    if dim == 0:
        return [()]
    if dim == 1:
        return [(k,) for k in range(max_deg + 1)]
    result = []
    for first in range(max_deg + 1):
        for rest in _enumerate_total_degree(dim - 1, max_deg - first):
            result.append((first,) + rest)
    return result

@lru_cache(maxsize=128)
def _get_arg_total_degree_exponents(
    arg_structure: ArgumentStructure,
    per_arg_order: tuple[int, ...]
) -> Array:
    """Generate exponents with per-argument total degree bounds."""
    arg_indices = []
    for size, max_ord in zip(arg_structure.arg_sizes, per_arg_order):
        arg_indices.append(_enumerate_total_degree(size, max_ord))

    all_exponents = []
    for combo in product(*arg_indices):
        flat_exp = [e for arg_exp in combo for e in arg_exp]
        all_exponents.append(flat_exp)

    return np.array(all_exponents, dtype=np.int32).T
```

### Phase 3: Extend TaylorModel Class

**File:** `immrax/taylor/taylor_model.py`

1. **Add new fields to `__init__`:**
```python
def __init__(
    self,
    coeffs, exponents, remainder, domain,
    center=None,
    _static_order=None,
    _arg_structure: "ArgumentStructure | None" = None,  # NEW
    _per_arg_order: "tuple[int, ...] | None" = None,    # NEW
) -> None:
    # ... existing code ...
    self._arg_structure = _arg_structure
    self._per_arg_order = _per_arg_order
```

2. **Update `tree_flatten`/`tree_unflatten`** to include new fields in aux_data

3. **Add properties:**
```python
@property
def arg_structure(self) -> "ArgumentStructure | None":
    return self._arg_structure

@property
def per_arg_order(self) -> "tuple[int, ...] | None":
    return self._per_arg_order
```

### Phase 4: Add Multi-Argument Factory Functions

**File:** `immrax/taylor/taylor_model.py` (add at end)

```python
def taylor_model_multiarg_identity(
    *arg_domains: Interval,
    per_arg_order: tuple[int, ...],
) -> TaylorModel:
    """Create identity Taylor model for multiple arguments."""
    arg_shapes = tuple(d.lower.shape for d in arg_domains)
    arg_structure = ArgumentStructure(arg_shapes)

    # Flatten domains
    flat_lower = jnp.concatenate([d.lower.reshape(-1) for d in arg_domains])
    flat_upper = jnp.concatenate([d.upper.reshape(-1) for d in arg_domains])
    flat_domain = interval(flat_lower, flat_upper)

    # Generate exponents
    exponents = _get_arg_total_degree_exponents(arg_structure, per_arg_order)

    # Build identity coefficients (center + linear terms)
    # ... (full implementation in code)

    return TaylorModel(
        coeffs, exponents, remainder, flat_domain,
        center=center_flat,
        _static_order=per_var_order,
        _arg_structure=arg_structure,
        _per_arg_order=per_arg_order,
    )
```

### Phase 5: Add Per-Argument Bounds Checking

**File:** `immrax/taylor/taylor_model.py`

```python
def _check_per_arg_bounds(
    exponents: Array,
    arg_structure: ArgumentStructure,
    per_arg_order: tuple[int, ...]
) -> Array:
    """Return boolean mask for monomials within per-argument bounds."""
    masks = []
    for arg_idx in range(arg_structure.num_args):
        slc = arg_structure.arg_slice(arg_idx)
        arg_total = jnp.sum(exponents[slc, :], axis=0)
        masks.append(arg_total <= per_arg_order[arg_idx])
    return jnp.all(jnp.stack(masks), axis=0)
```

### Phase 6: Update Arithmetic Operations

**File:** `immrax/taylor/nattm.py`

Update `_truncate_product` and `_tm_mul_p` to use per-argument bounds when `_arg_structure` is present:

```python
def _truncate_product(coeffs, exponents, max_order, shifted_domain,
                      arg_structure=None, per_arg_order=None):
    if arg_structure is not None and per_arg_order is not None:
        keep_mask = _check_per_arg_bounds(exponents, arg_structure, per_arg_order)
    else:
        # existing per-variable logic
        target_arr = jnp.array(max_order, dtype=jnp.int32)[:, None]
        keep_mask = jnp.all(exponents <= target_arr, axis=0)
    # ... rest unchanged
```

### Phase 7: Update `to_canonical`

**File:** `immrax/taylor/taylor_model.py`

Add branch for per-argument mode:

```python
def to_canonical(self, target_order=None, method=None):
    if self._arg_structure is not None and self._per_arg_order is not None:
        return self._to_canonical_per_arg(target_order, method)
    # ... existing per-variable logic
```

### Phase 8: Mirror Changes in TaylorPolynomial

**File:** `immrax/taylor/taylor_polynomial.py`

Add same `_arg_structure` and `_per_arg_order` fields and update methods.

## Files to Modify

| File | Changes |
|------|---------|
| `immrax/taylor/taylor_model.py` | ArgumentStructure, new exponent gen, TaylorModel fields, factory functions, canonicalization |
| `immrax/taylor/nattm.py` | Update `_truncate_product`, `_tm_mul_p` for per-arg bounds |
| `immrax/taylor/taylor_polynomial.py` | Mirror TaylorModel changes |
| `immrax/taylor/nattp.py` | Update truncation logic |

## Backward Compatibility

- When `_arg_structure` is `None`, all behavior is identical to current
- Existing code continues to work unchanged
- New factory functions enable multi-argument mode

## Verification

1. **Unit tests:**
   ```bash
   pytest tests/test_taylor.py -v
   ```

2. **Manual verification:**
   ```python
   from immrax.taylor import taylor_model_multiarg_identity
   from immrax import icentpert
   import jax.numpy as jnp

   t_dom = icentpert(jnp.array([0.0]), jnp.array([1.0]))
   x_dom = icentpert(jnp.zeros(2), jnp.ones(2))

   tm = taylor_model_multiarg_identity(t_dom, x_dom, per_arg_order=(2, 3))

   # Check monomial count: (2+1) * C(2+3,3) = 3 * 10 = 30
   assert tm.num_monomials == 30
   assert tm._arg_structure.num_args == 2
   assert tm._per_arg_order == (2, 3)
   ```

3. **Soundness test:**
   ```python
   # Evaluate at random points and verify containment
   for _ in range(100):
       t = jax.random.uniform(key, (1,), minval=-1, maxval=1)
       x = jax.random.uniform(key, (2,), minval=-1, maxval=1)
       flat_pt = jnp.concatenate([t, x])
       assert tm.contains(flat_pt)  # or verify with evaluate()
   ```
