#!/usr/bin/env python3
"""A box moved to a cell has the volume element of a box created there.

The volume element dv is the product of the grid spacings in an orthogonal cell
and V/M in an oblique one, whichever constructor made the box and whether it was
created at its cell or moved there (set_lx, set_lattice_parameters); a cell change
once took V/M in an orthogonal cell as well, one ulp from the product. Without a
space group, for 3D, 2D and 1D orthogonal cells (with and without a mask) and for
3D and 2D oblique cells, a solver created at one cell, computed, moved to another
cell and computed again is compared with solvers created at the new cell, without
angles and with right angles: dv at every grid point, the accessible volume and the
cell volume must agree bit for bit on every platform, and Q and the concentrations
of both chain models (fields of standard deviation 5) bit for bit on cpu-mkl and
CUDA (cpu-fftw plans each transform anew, so there to 1e-13).
"""

import contextlib
import io

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing

# (nx, cell, new cell, angles of an oblique cell or None)
CASES = [
    ([12, 10, 8], [2.0, 1.8, 1.6], [2.1, 1.8, 1.55], None),
    ([12, 10, 8], [2.0, 1.8, 1.6], [2.1, 1.7, 1.65], None),
    ([24, 20], [3.0, 2.7], [3.3, 2.6], None),
    ([40], [4.1], [3.7], None),
    ([12, 10, 8], [2.0, 1.8, 1.6], [2.1, 1.8, 1.55], [90.0, 90.0, 110.0]),
    ([24, 20], [3.0, 2.7], [3.3, 2.6], [100.0]),
]
BITWISE_PLATFORMS = ("cpu-mkl", "cuda")
TOL_FFTW = 1e-13


def solver(platform, chain_model, nx, lx, angles, mask):
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lx, ds=0.025, bond_lengths={"A": 1.0, "B": 1.2}, bc=["periodic"] * (2 * len(nx)),
                             chain_model=chain_model, numerical_method="rqm4" if chain_model == "continuous" else "ps",
                             platform=platform, mask=mask, angles=angles)
        s.add_polymer(1.0, [["A", 0.45, 0, 1], ["B", 0.55, 1, 2]])
    return s


def results(s, w):
    with contextlib.redirect_stdout(io.StringIO()):
        s.compute_propagators(w)
        s.compute_concentrations()
    box = s.computation_box
    return {"dv": np.array([box.get_dv(i) for i in range(s.total_grid)]),
            "accessible volume": np.array([box.get_accessible_volume()]),
            "volume": np.array([box.get_volume()]),
            "Q": np.array([s.get_partition_function(0)]),
            "phi": np.concatenate([np.asarray(s.get_concentration(m), float) for m in "AB"])}


def differences(platform, moved, fresh):
    """Names of the quantities that differ, with their largest difference"""
    out = []
    for key in moved:
        if platform in BITWISE_PLATFORMS or key in ("dv", "accessible volume", "volume"):
            if moved[key].tobytes() != fresh[key].tobytes():
                out.append(f"{key} {np.max(np.abs(moved[key] - fresh[key])):.1e}")
        elif np.max(np.abs(moved[key] - fresh[key])) > TOL_FFTW * np.max(np.abs(fresh[key])):
            out.append(f"{key} {np.max(np.abs(moved[key] - fresh[key])):.1e}")
    return out


def main():
    rng = np.random.default_rng(11)
    failures = checks = 0
    for platform in PlatformSelector.avail_platforms():
        for nx, lx0, lx1, oblique in CASES:
            M = int(np.prod(nx))
            w = {m: rng.normal(0.0, 5.0, M) for m in "AB"}
            right = [90.0] * (3 if len(nx) == 3 else 1)
            for masked in (False, True):
                mask = None
                if masked:
                    mask = np.ones(M)
                    mask[: M // 7] = 0.0
                for chain_model in ("continuous", "discrete"):
                    if oblique is None:
                        # (how the solver moves, its angles, the angles of the fresh solvers)
                        moves = [("set_lx", None, [None, right]), ("set_lattice_parameters", right, [None, right])]
                    else:
                        moves = [("set_lattice_parameters", oblique, [oblique])]
                    for how, angles, fresh_angles in moves:
                        s = solver(platform, chain_model, nx, lx0, angles, mask)
                        results(s, w)
                        if how == "set_lx":
                            s.set_lx(lx1)
                        else:
                            s.set_lattice_parameters(lx1, angles)
                        moved = results(s, w)
                        for created in fresh_angles:
                            diff = differences(platform, moved, results(solver(platform, chain_model, nx, lx1, created,
                                                                                mask), w))
                            checks += 1
                            failures += bool(diff)
                            print(f"[{'FAIL' if diff else 'PASS'}] {platform} {nx} mask={int(masked)} {chain_model}: "
                                  f"{how} (angles {angles}) against a box created with angles {created}"
                                  + (": " + ", ".join(diff) if diff else ""))
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
