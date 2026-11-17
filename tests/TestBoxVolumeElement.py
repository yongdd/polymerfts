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
CUDA (cpu-fftw plans each transform anew, so there to 1e-13). Every one of these
boxes, created or moved, must also have that volume element itself: the product of
its grid spacings (or V/M) times the mask at every grid point, and as
PropagatorSolver.get_dv(). The cells include ones where V/M differs from the
product, so a box that took the other one is caught in each kind of box.

With a space group, the reduced-basis integrals take the same volume element: on
Pmmm grids, created at a cell and moved to another (orthogonal cells where V/M
differs from the product), integral, inner_product, inner_product_inverse_weight
and multi_inner_product of fields of ones give n M dv exactly, whatever the order
of the sum, and mean gives M dv / V.
"""

import contextlib
import io

import numpy as np

from polymerfts import _core, PlatformSelector, PropagatorSolver
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
# Cases whose new cell also has V/M other than the product of the spacings (fields from their own
# generator, so that the cases above keep theirs)
MORE_CASES = [
    ([24, 20], [3.0, 2.7], [3.1, 2.6], None),
]
# (nx, cell, new cell) of the space-group integrals, V/M not the product of the spacings in both
SPACE_GROUP_CASES = [
    ([12, 10, 8], [2.0, 1.8, 1.6], [2.1, 1.8, 1.55]),
    ([16, 12, 8], [2.0, 1.8, 1.6], [2.1, 1.8, 1.55]),
]
BITWISE_PLATFORMS = ("cpu-mkl", "cuda")
TOL_FFTW = 1e-13


def solver(platform, chain_model, nx, lx, angles, mask, space_group=None):
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lx, ds=0.025, bond_lengths={"A": 1.0, "B": 1.2}, bc=["periodic"] * (2 * len(nx)),
                             chain_model=chain_model, numerical_method="rqm4" if chain_model == "continuous" else "ps",
                             platform=platform, mask=mask, angles=angles, space_group=space_group)
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


def volume_element(box, oblique):
    """The volume element of a grid point: the product of the grid spacings (in their order, as the box
    multiplies them) in an orthogonal cell, V/M in an oblique one"""
    if oblique:
        return box.get_volume() / box.get_total_grid()
    dv = 1.0
    for dx in box.get_dx():
        dv *= dx
    return dv


def volume_element_errors(s, oblique, mask):
    """What differs from the volume element of the box's cell: dv at the grid points (times the mask) and
    PropagatorSolver.get_dv()"""
    box = s.computation_box
    expected = volume_element(box, oblique)
    dv = np.array([box.get_dv(i) for i in range(s.total_grid)])
    expected_dv = np.full(s.total_grid, expected) if mask is None else expected * mask
    errors = []
    if dv.tobytes() != expected_dv.tobytes():
        errors.append(f"dv {np.max(np.abs(dv - expected_dv)):.1e} from the {'V/M' if oblique else 'product'}")
    if s.get_dv() != expected:
        errors.append(f"get_dv() {s.get_dv() - expected:.1e} from the {'V/M' if oblique else 'product'}")
    return errors


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


def report(failures, label, errors):
    print(f"[{'FAIL' if errors else 'PASS'}] {label}" + (": " + ", ".join(errors) if errors else ""))
    return failures + bool(errors)


def space_group_integral_errors(s):
    """The reduced-basis integrals of fields of ones against n M dv (sums of integers, exact in any order)"""
    box = s.computation_box
    dv = volume_element(box, False)
    M = s.total_grid
    ones = np.ones(box.get_n_basis())
    values = {"integral": (box.integral(ones), M * dv),
              "mean": (box.mean(ones), M * dv / box.get_volume()),
              "inner_product": (box.inner_product(ones, ones), M * dv),
              "inner_product_inverse_weight": (box.inner_product_inverse_weight(ones, ones, ones), M * dv),
              "multi_inner_product": (box.multi_inner_product(2, np.ones(2 * ones.size), np.ones(2 * ones.size)),
                                      2 * M * dv)}
    return [f"{name} {value - expected:.1e}" for name, (value, expected) in values.items() if value != expected]


def main():
    rng, rng_more = np.random.default_rng(11), np.random.default_rng(12)
    failures = checks = 0
    for platform in PlatformSelector.avail_platforms():
        for nx, lx0, lx1, oblique, generator in [case + (rng,) for case in CASES] + \
                                                [case + (rng_more,) for case in MORE_CASES]:
            M = int(np.prod(nx))
            w = {m: generator.normal(0.0, 5.0, M) for m in "AB"}
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
                        label = f"{platform} {nx} mask={int(masked)} {chain_model}"
                        s = solver(platform, chain_model, nx, lx0, angles, mask)
                        results(s, w)
                        checks += 1
                        failures = report(failures, f"{label}: created at {lx0} with angles {angles}",
                                          volume_element_errors(s, oblique, mask))
                        if how == "set_lx":
                            s.set_lx(lx1)
                        else:
                            s.set_lattice_parameters(lx1, angles)
                        moved = results(s, w)
                        checks += 1
                        failures = report(failures, f"{label}: {how} to {lx1} (angles {angles})",
                                          volume_element_errors(s, oblique, mask))
                        for created in fresh_angles:
                            fresh = solver(platform, chain_model, nx, lx1, created, mask)
                            diff = differences(platform, moved, results(fresh, w))
                            checks += 2
                            failures = report(failures, f"{label}: created at {lx1} with angles {created}",
                                              volume_element_errors(fresh, oblique, mask))
                            failures = report(failures, f"{label}: {how} (angles {angles}) against a box created "
                                              f"with angles {created}", diff)
        for nx, lx0, lx1 in SPACE_GROUP_CASES:
            s = solver(platform, "continuous", nx, lx0, None, None, space_group=_core.SpaceGroup(nx, "Pmmm"))
            checks += 1
            failures = report(failures, f"{platform} Pmmm {nx} created at {lx0}: reduced-basis integrals",
                              space_group_integral_errors(s))
            s.set_lx(lx1)
            checks += 1
            failures = report(failures, f"{platform} Pmmm {nx} moved to {lx1}: reduced-basis integrals",
                              space_group_integral_errors(s))
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
