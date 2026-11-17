#!/usr/bin/env python3
"""CrysFFT solvers after a cell change: updated solver against a fresh one.

A solver whose lattice changes (set_lattice_parameters) rebuilds its CrysFFT
tables, including the k^2 tables that the stress uses. For each platform, each
case computes Q, the concentrations and the stress after one or two cell changes
and compares them with a solver created at the new cell (they agree to round-off;
the CUDA solvers share and recycle their tables). The changes include one axis
alone, so a cell comparison that ignored Ly or Lz would keep stale tables. CUDA
is also compared with the CPU, stress included. Cases (fields symmetrized and
rescaled to a standard deviation of 5, continuous and discrete chains):
- Pmmm 16x12x8, Pmmm physical basis and irreducible basis: the Pmmm DCT
  (nz/2 = 4); Ly alone, then Lx, Ly and Lz;
- Pmmm 32x24x16, irreducible basis: the recursive 3m CrysFFT (nz/2 = 8); Ly
  alone, then Lz alone;
- Im-3m 16^3, m3 physical basis: the recursive 3m CrysFFT on a cubic cell;
- Im-3m 18^3, m3 physical basis: an odd basis size (9^3), so the second propagator
  of a stress pair lies at an odd offset that cuFFT cannot read directly.
All of them take the Pmmm DCT or the recursive 3m CrysFFT with the stress fast path,
which leaves the full grid unused. After the first cell change and its propagators,
the off-diagonal stress flag is turned on (set_force_off_diagonal_stress), which
takes the stress to the full grid: that stress must equal the one of a solver
created at the cell with the flag.

A solver chooses its CrysFFT engine once, for the angles of its box: the Pmmm DCT
and the recursive 3m CrysFFT need an orthogonal cell, the ObliqueZ CrysFFT alpha =
beta = 90 degrees. The space group fixes these angles to the orthogonality test of
the box (1e-10 rad). For the ObliqueZ (P6/mmm 12x12x16, and monoclinic P2/m with
unique axis c 12x10x8, which fixes alpha and beta only) and the Pmmm DCT (Pmmm
16x12x8), irreducible basis, each fixed angle is moved by 1e-9 degrees (inside the
test: the changed solver gives the Q of a fresh one), by 1e-8 and by 9e-5 degrees
(outside it, 9e-5 within a relative 1e-6 of 90: refused before the box changes, and
refused for a fresh solver). The solver then follows a valid change.
"""

import contextlib
import io

import numpy as np
from polymerfts import _core, PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing

CASES = [
    ("Pmmm", None, [16, 12, 8], "pmmm", [[2.0, 1.8, 1.6], [2.0, 1.9, 1.6], [2.1, 1.95, 1.5]]),
    ("Pmmm", None, [16, 12, 8], "irreducible", [[2.0, 1.8, 1.6], [2.0, 1.9, 1.6], [2.1, 1.95, 1.5]]),
    ("Pmmm", None, [32, 24, 16], "irreducible", [[3.0, 2.4, 2.0], [3.0, 2.6, 2.0], [3.0, 2.6, 2.2]]),
    ("Im-3m", 529, [16, 16, 16], "m3", [[1.9] * 3, [2.0] * 3]),
    ("Im-3m", 529, [18, 18, 18], "m3", [[1.9] * 3, [2.0] * 3]),
]


def space_group(symbol, hall, nx, basis):
    sg = _core.SpaceGroup(nx, symbol, hall) if hall else _core.SpaceGroup(nx, symbol)
    if basis == "pmmm":
        sg.enable_pmmm_physical_basis()
    if basis == "m3":
        sg.enable_m3_physical_basis()
    return sg


def solver(platform, chain, case, lx, fields):
    """A solver of the case at lx and the fields in its basis"""
    symbol, hall, nx, basis, _ = case
    sg = space_group(symbol, hall, nx, basis)
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lx, ds=0.025, bond_lengths={"A": 1.0, "B": 1.2}, bc=["periodic"] * 6,
                             chain_model=chain, numerical_method="rqm4" if chain == "continuous" else None,
                             platform=platform, space_group=sg)
        s.add_polymer(1.0, [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]])
    return s, {m: sg.to_reduced_basis(fields[m]) for m in "AB"}


def results(platform, chain, case, lattices, fields):
    """Q, phi_A and the stress after each lattice of `lattices` (one solver, updated in turn)."""
    symbol, _, _, basis, _ = case
    out = []
    s, w = solver(platform, chain, case, lattices[0], fields)
    with contextlib.redirect_stdout(io.StringIO()):
        for i, lx in enumerate(lattices):
            if i:
                s.set_lattice_parameters(lx, [90.0, 90.0, 90.0])
            s.compute_propagators(w)
            s.compute_concentrations()
            s.compute_stress()
            assert s.check_total_partition(), (platform, chain, symbol, basis, lx)
            phi = np.asarray(s.get_concentration("A")) + np.asarray(s.get_concentration("B"))
            assert abs(s.mean(phi) - 1.0) < 1e-10, (platform, chain, symbol, basis, lx)
            out.append((s.get_partition_function(0), np.asarray(s.get_concentration("A"), float).copy(),
                        np.asarray(s.get_stress(), float).copy()))
    return out


def flag_turned_on(platform, chain, case, lattices, fields):
    """Stress after a computation at lattices[1] (a cell change from lattices[0]) when the off-diagonal stress
    flag is turned on only then, and the stress of a solver created at lattices[1] with the flag"""
    s, w = solver(platform, chain, case, lattices[0], fields)
    fresh, w_fresh = solver(platform, chain, case, lattices[1], fields)
    fresh.set_force_off_diagonal_stress(True)
    with contextlib.redirect_stdout(io.StringIO()):
        s.compute_propagators(w)
        s.compute_concentrations()
        s.compute_stress()
        s.set_lattice_parameters(lattices[1], [90.0, 90.0, 90.0])
        s.compute_propagators(w)
        s.compute_concentrations()
        s.set_force_off_diagonal_stress(True)
        s.compute_stress()
        fresh.compute_propagators(w_fresh)
        fresh.compute_concentrations()
        fresh.compute_stress()
    return np.asarray(s.get_stress(), float), np.asarray(fresh.get_stress(), float)


# (engine, Hall number or symbol, nx, lx, a valid change of lx, angles, the angles the group fixes);
# irreducible basis
ANGLE_CASES = [
    ("ObliqueZ", 485, [12, 12, 16], [4.0, 4.0, 3.0], [4.1, 4.1, 3.1], [90.0, 90.0, 120.0], (0, 1, 2)),
    ("ObliqueZ P2/m", 58, [12, 10, 8], [1.5, 1.3, 1.2], [1.6, 1.4, 1.25], [90.0, 90.0, 100.0], (0, 1)),
    ("Pmmm DCT", "Pmmm", [16, 12, 8], [2.0, 1.8, 1.6], [2.1, 1.95, 1.5], [90.0, 90.0, 90.0], (0, 1, 2)),
]
# A fixed angle moved by these degrees, and whether the group accepts it (1e-10 rad is 5.7e-9 degrees)
ANGLE_OFFSETS = [(1e-9, True), (1e-8, False), (9e-5, False)]


def fixed_angles(platform, case):
    """(ok, line) for each angle the group fixes, moved by each offset: an accepted change gives the Q of a
    solver created at the new cell; a refused one leaves the box as it was, and a solver created at that cell
    is refused too. Then a valid change of lx against a fresh solver. Fields of standard deviation 5."""
    engine, group, nx, lx, moved, angles, fixed = case
    f = _core.SpaceGroup(nx, group).symmetrize(np.random.default_rng(7).normal(0.0, 1.0, int(np.prod(nx))))

    def solver(box, cell_angles):
        sg = _core.SpaceGroup(nx, group)
        s = PropagatorSolver(nx=nx, lx=box, ds=0.05, bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                             chain_model="continuous", numerical_method="rqm4", platform=platform,
                             space_group=sg, angles=cell_angles)
        s.add_polymer(1.0, [["A", 1.0, 0, 1]])
        return s, {"A": sg.to_reduced_basis(5.0 * f / np.std(f))}

    def against_fresh(s, w, box, cell_angles):
        """|Q / Q_fresh - 1| of s against a solver created at the cell"""
        s.compute_propagators(w)
        fresh, w_fresh = solver(box, cell_angles)
        fresh.compute_propagators(w_fresh)
        return abs(s.get_partition_function(0) / fresh.get_partition_function(0) - 1.0)

    def fresh_solver(box, cell_angles):
        s, w = solver(box, cell_angles)
        s.compute_propagators(w)   # the first computation attaches the group to the box

    def refused(call):
        try:
            call()
        except Exception as error:   # the native RuntimeError, or an error raised from it
            return "crystal system" in str(error)
        return False

    lines = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in fixed:
            for offset, accepted in ANGLE_OFFSETS:
                off = list(angles)
                off[i] += offset
                s, w = solver(lx, angles)
                if accepted:
                    s.set_lattice_parameters(lx, off)
                    d = against_fresh(s, w, lx, off)
                    ok, result = d < 1e-12, f"accepted, Q against a fresh solver {d:.1e}"
                else:
                    ok = (refused(lambda: s.set_lattice_parameters(lx, off))
                          and np.allclose(s.get_angles_degrees(), angles, rtol=0.0, atol=1e-12)
                          and refused(lambda: fresh_solver(lx, off)))
                    result = "refused, also for a fresh solver"
                lines.append((ok, f"{platform} {engine}: angle {i} moved by {offset:g} degrees {result}"))
        s, w = solver(lx, angles)
        s.set_lattice_parameters(moved, angles)
        d = against_fresh(s, w, moved, angles)
        lines.append((d < 1e-12, f"{platform} {engine}: lx {moved}, Q against a fresh solver {d:.1e}"))
    return lines


def max_rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(b)), 1e-300))


def main():
    platforms = PlatformSelector.avail_platforms()
    cpu = [p for p in platforms if p.startswith("cpu")]
    failures = checks = 0
    for case in CASES:
        symbol, hall, nx, basis, lattices = case
        full = space_group(symbol, hall, nx, "irreducible")
        rng = np.random.default_rng(5)
        fields = {}
        for m in "AB":
            f = full.symmetrize(rng.normal(0.0, 1.0, int(np.prod(nx))))
            fields[m] = 5.0 * f / np.std(f)
        for chain in ("continuous", "discrete"):
            reference = {}
            for platform in platforms:
                updated = results(platform, chain, case, lattices, fields)
                for k in range(1, len(lattices)):
                    fresh = results(platform, chain, case, [lattices[k]], fields)[0]
                    d = max(abs(updated[k][0] - fresh[0]) / abs(fresh[0]), max_rel(updated[k][1], fresh[1]),
                            max_rel(updated[k][2], fresh[2]))
                    ok = d < 1e-12
                    checks += 1
                    failures += not ok
                    print(f"[{'PASS' if ok else 'FAIL'}] {platform} {symbol} {basis} {nx} {chain}: lattice {lattices[k]} "
                          f"updated vs fresh {d:.1e}")
                reference[platform] = updated
                walked, fresh = flag_turned_on(platform, chain, case, lattices, fields)
                d = max_rel(walked, fresh) if walked.shape == fresh.shape else np.inf
                ok = d < 1e-12
                checks += 1
                failures += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] {platform} {symbol} {basis} {nx} {chain}: off-diagonal flag turned on "
                      f"after a cell change, stress vs a solver created with it {d:.1e}")
            if "cuda" in reference and cpu:
                d = max(max_rel(g[2], c[2]) for g, c in zip(reference["cuda"], reference[cpu[0]]))
                ok = d < 1e-10
                checks += 1
                failures += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] cuda vs {cpu[0]} {symbol} {basis} {nx} {chain}: stress {d:.1e}")
    for platform in platforms:
        for case in ANGLE_CASES:
            for ok, line in fixed_angles(platform, case):
                checks += 1
                failures += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] {line}")
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
