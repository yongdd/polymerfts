#!/usr/bin/env python3
"""CUDA CrysFFT solvers with four streams give bit for bit the results of one stream.

A CUDA solver keeps one CrysFFT object per stream (plans and work buffers) and one
k-space table per coefficient for all streams, which it rebuilds in place when the
cell changes. With OMP_NUM_THREADS = 4 the propagators of the polymers below run
on four streams at once (a branched polymer and a diblock); with 1 they run one
after another. For the Pmmm DCT (Pmmm 16x12x8, Pmmm physical basis) and the
recursive 3m CrysFFT (Im-3m 16^3, m3 physical basis), continuous (rqm4) and
discrete chains, one solver per stream count goes through three cells (two cell
changes); after each, the partition functions, the concentrations and the stress
of four streams must equal those of one stream exactly. Fields symmetrized and
rescaled to a standard deviation of 5. Exit code 77 (skipped) in a build without
the CUDA backend; exit code 1 in a CUDA build without a visible CUDA device.
"""

import contextlib
import io
import os

import numpy as np
from polymerfts import _core, PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing

CASES = [
    ("Pmmm DCT", "Pmmm", None, [16, 12, 8], "pmmm",
     [[2.0, 1.8, 1.6], [2.1, 1.8, 1.55], [1.9, 1.85, 1.7]]),
    ("recursive 3m", "Im-3m", 529, [16, 16, 16], "m3",
     [[1.9] * 3, [2.0] * 3, [1.85] * 3]),
]
CHAINS = [("continuous", "rqm4"), ("discrete", "ps")]


def space_group(symbol, hall, nx, basis):
    sg = _core.SpaceGroup(nx, symbol, hall) if hall else _core.SpaceGroup(nx, symbol)
    getattr(sg, f"enable_{basis}_physical_basis")()
    return sg


def run(n_streams, case, chain, method, fields):
    """(Q, phi_A, phi_B, stress) after each cell of the case, one solver updated in turn"""
    _, symbol, hall, nx, basis, lattices = case
    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(n_streams)   # read when the solver creates its computation
    try:
        sg = space_group(symbol, hall, nx, basis)
        out = []
        with contextlib.redirect_stdout(io.StringIO()):
            s = PropagatorSolver(nx=nx, lx=lattices[0], ds=0.025, bond_lengths={"A": 1.0, "B": 1.2},
                                 bc=["periodic"] * 6, chain_model=chain, numerical_method=method,
                                 platform="cuda", space_group=sg)
            s.add_polymer(0.6, [["A", 0.4, 0, 1], ["B", 0.3, 1, 2], ["B", 0.3, 1, 3], ["A", 0.2, 1, 4]])
            s.add_polymer(0.4, [["B", 0.5, 0, 1], ["A", 0.5, 1, 2]])
            w = {m: sg.to_reduced_basis(fields[m]) for m in "AB"}
            for i, lx in enumerate(lattices):
                if i:
                    s.set_lattice_parameters(lx, [90.0, 90.0, 90.0])
                s.compute_propagators(w)
                s.compute_concentrations()
                s.compute_stress()
                out.append((np.array([s.get_partition_function(0), s.get_partition_function(1)]),
                            np.asarray(s.get_concentration("A"), float).copy(),
                            np.asarray(s.get_concentration("B"), float).copy(),
                            np.asarray(s.get_stress(), float).copy()))
        return out
    finally:
        if previous is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = previous


def main():
    if "cuda" not in PlatformSelector.avail_platforms():
        if cuda_part_missing():
            return 1
        print("This build has no CUDA backend; skipped")
        return 77
    failures = checks = 0
    rng = np.random.default_rng(11)
    for case in CASES:
        engine, symbol, hall, nx, basis, lattices = case
        sg = space_group(symbol, hall, nx, basis)
        fields = {}
        for m in "AB":
            f = sg.symmetrize(rng.normal(0.0, 1.0, int(np.prod(nx))))
            fields[m] = 5.0 * f / np.std(f)
        for chain, method in CHAINS:
            one = run(1, case, chain, method, fields)
            four = run(4, case, chain, method, fields)
            for i, lx in enumerate(lattices):
                same = all(np.array_equal(a, b) for a, b in zip(one[i], four[i]))
                checks += 1
                failures += not same
                print(f"[{'PASS' if same else 'FAIL'}] {engine} {chain} cell {i} {lx}: "
                      f"4 streams {'equal' if same else 'differ from'} 1 stream bit for bit")
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
