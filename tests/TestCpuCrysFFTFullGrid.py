#!/usr/bin/env python3
"""CPU CrysFFT paths against the full grid, with unequal bond lengths, local contour steps and a multiplicity.

The CrysFFT steps look up their Boltzmann tables by the step coefficient b^2 ds / 6 of the
monomer type and contour step of each block, and the stress fast path its multipliers by b^2
(full and half bonds). Equal bond lengths and blocks that are multiples of ds would hide a
table built or looked up with the coefficient of another monomer type or contour step. Here
three monomer types have bond lengths 1, 1.25 and 0.8; an AB diblock has blocks that are not
multiples of ds = 0.02 (0.413/0.587 continuous: local contour steps; 0.42/0.58 discrete); a
star carries a block multiplicity (2.5 continuous, 3 discrete). For every CPU platform a
solver with a space group is compared with a full-grid solver without one, on the same fields
(symmetrized under the group, mean zero, standard deviation 5):
- Pmmm 16x12x8 (nz/2 = 4): the Pmmm DCT, irreducible basis and Pmmm physical basis;
- Pmmm 32x24x16, irreducible basis: the recursive 3m CrysFFT;
- Im-3m 16^3, m3 physical basis and irreducible basis, and Ia-3d 16^3, m3 physical basis:
  the recursive 3m CrysFFT;
- P6/mmm 12x12x16 (gamma = 120 degrees), irreducible basis and z-mirror physical basis:
  the ObliqueZ CrysFFT (no stress fast path);
for rqm4 and rk2 (continuous) and ps (discrete), with full and reduced memory: Q of both
polymers (relative 1e-11), phi of every monomer type (absolute 1e-11), the stress (relative to
its largest component, 1e-10), check_total_partition() and material conservation (the mean of
the summed concentrations is one to 1e-11). Then both solvers move to another cell
(set_lattice_parameters) and are compared again. The two agree to round-off (a few 1e-14).
"""

import contextlib
import io

import numpy as np
from polymerfts import _core, PlatformSelector, PropagatorSolver

DS = 0.02
BOND_LENGTHS = {"A": 1.0, "B": 1.25, "C": 0.8}
TOL_Q = 1e-11
TOL_PHI = 1e-11
TOL_STRESS = 1e-10
TOL_MEAN = 1e-11

# (label, group, Hall number, nx, basis, cell, cell after the change)
CASES = [
    ("Pmmm DCT", "Pmmm", None, [16, 12, 8], "irreducible", [2.0, 1.8, 1.6], [2.1, 1.7, 1.65]),
    ("Pmmm DCT", "Pmmm", None, [16, 12, 8], "pmmm", [2.0, 1.8, 1.6], [2.1, 1.7, 1.65]),
    ("recursive 3m", "Pmmm", None, [32, 24, 16], "irreducible", [3.0, 2.4, 2.0], [3.1, 2.3, 2.1]),
    ("recursive 3m", "Im-3m", 529, [16, 16, 16], "m3", [1.9] * 3, [2.05] * 3),
    ("recursive 3m", "Im-3m", 529, [16, 16, 16], "irreducible", [1.9] * 3, [2.05] * 3),
    ("recursive 3m", "Ia-3d", 530, [16, 16, 16], "m3", [3.3] * 3, [3.5] * 3),
    ("ObliqueZ", "P6/mmm", 485, [12, 12, 16], "irreducible", [2.0, 2.0, 1.7], [2.1, 2.1, 1.6]),
    ("ObliqueZ", "P6/mmm", 485, [12, 12, 16], "z-mirror", [2.0, 2.0, 1.7], [2.1, 2.1, 1.6]),
]
ANGLES = {"P6/mmm": [90.0, 90.0, 120.0]}


def polymers(chain_model):
    """An AB diblock whose blocks are not multiples of ds, and a star with a block multiplicity"""
    if chain_model == "continuous":
        return [(0.5, [["A", 0.413, 0, 1], ["B", 0.587, 1, 2]]),
                (0.5, [["C", 0.3, 0, 1, 2.5], ["A", 0.2, 1, 2], ["B", 0.25, 0, 3]])]
    return [(0.5, [["A", 0.42, 0, 1], ["B", 0.58, 1, 2]]),
            (0.5, [["C", 0.3, 0, 1, 3], ["A", 0.2, 1, 2], ["B", 0.24, 0, 3]])]


def space_group(group, hall, nx, basis=None):
    with contextlib.redirect_stdout(io.StringIO()):
        sg = _core.SpaceGroup(nx, group, hall) if hall else _core.SpaceGroup(nx, group)
        if basis == "pmmm":
            sg.enable_pmmm_physical_basis()
        elif basis == "m3":
            sg.enable_m3_physical_basis()
        elif basis == "z-mirror":
            sg.enable_z_mirror_physical_basis()
    return sg


def run(platform, chain_model, method, sg, nx, cells, angles, fields, reduce_memory):
    """Q, phi (full grid) and the stress at each cell of `cells`, one solver moved from cell to cell"""
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=cells[0], ds=DS, bond_lengths=BOND_LENGTHS, bc=["periodic"] * 6,
                             chain_model=chain_model, numerical_method=method, platform=platform,
                             space_group=sg, angles=angles, reduce_memory=reduce_memory)
        for volume_fraction, blocks in polymers(chain_model):
            s.add_polymer(volume_fraction, blocks)
        w = {m: (sg.to_reduced_basis(f) if sg is not None else f) for m, f in fields.items()}
        for i, lx in enumerate(cells):
            if i:
                s.set_lattice_parameters(lx, angles)
            s.compute_propagators(w)
            s.compute_concentrations()
            s.compute_stress()
            phi = {}
            for m in BOND_LENGTHS:
                p = np.asarray(s.get_concentration(m), float)
                phi[m] = sg.from_reduced_basis(p) if sg is not None else p.copy()
            out.append({"Q": [s.get_partition_function(p) for p in range(2)], "phi": phi,
                        "stress": np.asarray(s.get_stress(), float).copy(), "partition": s.check_total_partition(),
                        "mean": float(np.mean(sum(phi.values())))})
    return out


def main():
    platforms = [p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")]
    if not platforms:
        print("SKIPPED: this build has no CPU platform")
        return 77
    rng = np.random.default_rng(2026)
    failures = checks = 0
    for label, group, hall, nx, basis, lx0, lx1 in CASES:
        angles = ANGLES.get(group, [90.0, 90.0, 90.0])
        full_group = space_group(group, hall, nx)
        fields = {}
        for m in BOND_LENGTHS:
            f = full_group.symmetrize(rng.standard_normal(int(np.prod(nx))))
            fields[m] = 5.0 * (f - f.mean()) / np.std(f)
        for platform in platforms:
            for chain_model, method in (("continuous", "rqm4"), ("continuous", "rk2"), ("discrete", "ps")):
                reference = run(platform, chain_model, method, None, nx, [lx0, lx1], angles, fields, False)
                for reduce_memory in (False, True):
                    sg = space_group(group, hall, nx, basis)
                    result = run(platform, chain_model, method, sg, nx, [lx0, lx1], angles, fields, reduce_memory)
                    for stage, (r, f) in enumerate(zip(result, reference)):
                        dq = max(abs(r["Q"][p] / f["Q"][p] - 1.0) for p in range(2))
                        dphi = max(float(np.max(np.abs(r["phi"][m] - f["phi"][m]))) for m in BOND_LENGTHS)
                        dstress = float(np.max(np.abs(r["stress"] - f["stress"])) / np.max(np.abs(f["stress"])))
                        ok = (dq < TOL_Q and dphi < TOL_PHI and dstress < TOL_STRESS and r["partition"]
                              and abs(r["mean"] - 1.0) < TOL_MEAN)
                        checks += 1
                        failures += not ok
                        print(f"[{'PASS' if ok else 'FAIL'}] {platform} {label} {group} {basis} {method} "
                              f"reduce_memory={int(reduce_memory)} cell {stage}: Q {dq:.1e} phi {dphi:.1e} "
                              f"stress {dstress:.1e} partition {r['partition']} mean-1 {r['mean'] - 1.0:+.1e}")
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
