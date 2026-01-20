# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**immrax** is a Python library for interval analysis and mixed monotone reachability analysis in JAX. It provides composable interval function transformations that work with JAX's automatic differentiation, parallelization, and GPU acceleration for reachable set estimation in control theory applications.

## Common Commands

**Installation:**
```bash
pip install .              # Standard install
pip install .[cuda]        # With CUDA support
pip install .[examples]    # With example dependencies
```

**System Dependencies (required for pypoman/pycddlib):**
```bash
# Ubuntu
apt-get install -y libcdd-dev libgmp-dev
# Arch
pacman -S cddlib
```

**Running Tests:**
```bash
pytest                           # Run all tests
pytest tests/test_inclusion.py   # Run specific test file
pytest -k "test_natif"           # Run tests matching pattern
```

**Linting:**
```bash
ruff check .                     # Check for lint errors
ruff format .                    # Format code
```

**Verify Installation:**
```bash
python examples/compare.py       # Benchmarks different inclusion functions
```

## Architecture

### Core Modules

- **inclusion/** - Interval analysis and inclusion functions (natif, jacif, mjacif)
- **system/** - Dynamical systems (ContinuousSystem, DiscreteSystem)
- **embedding.py** - Embedding transformations that lift dynamics from R^n to T^{2n}
- **control.py** - Control feedback systems
- **neural.py** - Neural network verification (CROWN, Fastlin methods)
- **parametric/** - Parametric set representations (Parametope, Ellipsoid, Polytope, Normotope)
- **generator/** - Reachability set generation (in development)
- **refinement/** - Set refinement strategies

### Key Design Patterns

**JAX Pytree Registration:** All custom classes (Interval, Parametope, etc.) are registered as JAX pytrees, enabling seamless composition with `jit`, `grad`, `vmap`.

**Inclusion Function Framework:** Functions that compute interval overapproximations:
- `natif` - Natural Interval Extension
- `jacif` - Jacobian-based Interval Function
- `mjacif` - Mixed Jacobian Interval Function
- `custom_if` - User-defined interval functions

**System Hierarchy:**
```
System (abstract)
├── ContinuousSystem / DiscreteSystem
├── ReversedSystem
├── LinearTransformedSystem
├── LiftedSystem
└── OpenLoopSystem / ControlledSystem
```

**Parametric Sets:** Geometric representations for reachable sets:
```
Parametope (base)
├── AffineParametope
├── Ellipsoid / Polytope
├── Normotope (L∞, L1, L2)
└── Annulus
```

### Vendored Dependencies

The `_vendor/jax_verify` directory contains a git submodule. Initialize with:
```bash
git submodule update --init --recursive
```

## Code Conventions

- Lambda functions are allowed (ruff E731 is ignored)
- Use `jnp` (JAX numpy) for numerical operations, not standard numpy
- Import convention: `import immrax as irx`
- Tests use pytest with fixtures; see `tests/utils.py` for validation helpers
