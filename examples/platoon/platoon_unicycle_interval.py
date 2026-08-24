"""Unicycle platoon reachability, interval baseline (adjoint disabled).

Same system, seed, dt and tf as ``platoon_unicycle.py``, but the parametope's
alpha is frozen at the identity (``disable_adjoint=True``), so the set stays an
axis-aligned box and the first-order cancellation is dropped -- plain interval
analysis, for comparison against the adjoint polytope numbers.

Outputs: platoon_unicycle_interval_grid.{pdf,svg},
platoon_unicycle_interval_overview.{pdf,svg}, and a one-row LaTeX table.
"""

# ruff: noqa: E402  (backend/x64 flags must be set before jax.numpy is imported)
import argparse
import os
import pathlib
import sys

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--platform", choices=("auto", "gpu", "cpu"), default="auto", help="JAX backend")
_ap.add_argument("--no-show", action="store_true", help="save figures without opening a window")
_ap.add_argument("--precision", choices=("32", "64"), default="32", help="float width")
args = _ap.parse_args()

# platoon_unicycle sets the backend and x64 from $PLATOON_ARGS when imported.
os.environ["PLATOON_ARGS"] = (
    f"--platform {args.platform} --precision {args.precision}"
    + (" --no-show" if args.no_show else "")
)
if args.platform != "auto":
    os.environ["JAX_PLATFORMS"] = args.platform

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import platoon_unicycle as pu

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

def divergence_time(rs):
    """First time the raw parametope state goes non-finite, or None."""
    yy = rs.ys[0]
    ok = (onp.isfinite(onp.asarray(yy.ox)).all(1)
          & onp.isfinite(onp.asarray(yy.y)).all(1))
    return None if ok.all() else int(onp.argmin(ok)) * pu.dt


if __name__ == "__main__":
    dev = jax.devices()[0]
    print(f"backend: {jax.default_backend()} ({dev.device_kind}), x64={jax.config.jax_enable_x64}")
    ao_np = pu.unicycle_nominal()
    ao = jnp.asarray(ao_np)
    xn = jnp.asarray(pu.nominal_state(ao_np))

    results, rows, rows_tex = {}, [], []
    for n, ps in pu.AGENTS:
        rs, t_run, t_jit = pu.reach(n, ao, xn, ps, disable_adjoint=True)
        vols = pu.vehicle_vols(rs, n)
        v_avg, v_fin = float(onp.mean(vols)), float(vols[-1])
        td = divergence_time(rs)
        results[n] = rs if n == pu.SHOW_AGENTS else None
        del rs
        r, r_tex = pu.table_row(n, t_run, t_jit, v_avg, v_fin)
        rows.append(r)
        rows_tex.append(r_tex)
        print(f"  n={n}: {t_run:.3f} s (JIT {t_jit:.0f} s)  "
              f"avg vol {v_avg:.3e}  final vol {v_fin:.3e}"
              + (f"  DIVERGED at t={td:.3f}" if td is not None else ""))

    pu.print_table(rows, rows_tex, (
        f"% unicycle platoon interval reachability (alpha frozen, dt={pu.dt}, "
        f"tf={pu.tf}, pert {float(pu.PERT_VEHICLE[0]):g}, fp{args.precision} on "
        f"{jax.default_backend()}: {dev.device_kind})"))

    n = pu.SHOW_AGENTS
    pu.grid_figure(results[n], n, "platoon_unicycle_interval_grid")
    pu.overview_figure(results[n], n, "platoon_unicycle_interval_overview")
    if not args.no_show:
        plt.show()
