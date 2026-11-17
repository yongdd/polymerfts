#!/usr/bin/env python3
"""Partition functions that are not positive and finite.

With real fields Q is a sum of positive Boltzmann weights. A discrete chain grafted
next to an absorbing wall in a strong field (TODO.md, item 29: N = 50, 704 cells,
Lz = 6, W = 100 for z < 3) leaves its physical propagator so far below the largest
value on the grid that the spectral transforms return round-off noise: Q comes out
near -1e-19, against an independent reference of +4.67e-28. The native library only
reports values; the simulation loops reject accepted states:

- compute_propagators() returns without raising, since intermediate SCFT and L-FTS
  iterates can legitimately pass through Q <= 0. W = 1e5 removes the whole propagator
  on every platform: its partition error is infinite and check_total_partition()
  fails. partition_errors() and check_total_partition() print nothing.
- At W = 100 a result is never accepted silently: either Q is not positive, which
  check_partition_functions() rejects, or check_total_partition() fails. The sign of
  the round-off noise depends on the FFT plan.
- W = 10 (Q = 8.46e-7) passes both checks, with the same Q on every platform.
- SCFT final fields and an L-FTS saddle point with a non-positive Q (simulated by
  negating Q) raise NumericalError naming the partition function, a numerical failure
  rather than an input error; its fields are the monomer fields of that state.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import SCFT, LFTS, PropagatorSolver, PlatformSelector, NumericalError
from polymerfts.utils.common import check_partition_functions
from cuda_part import cuda_part_missing
from fts_checks import diblock_fields, diblock_parameters, silenced

N, NZ, LZ = 50, 704, 6.0
Z = (np.arange(NZ) + 0.5) * LZ / NZ
GRAFT = np.exp(-(Z - 1 / np.sqrt(N)) ** 2 / (2 * 0.2 ** 2))


def solve(platform, strength):
    """The solver after computing the propagators of the grafted chain."""
    with silenced():
        solver = PropagatorSolver(nx=[NZ], lx=[LZ], ds=1 / N, bond_lengths={"A": 1.0}, bc=["absorbing"] * 2,
                                  chain_model="discrete", platform=platform)
        solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
        solver.compute_propagators({"A": np.where(Z < 3.0, strength, 0.0)}, q_init={"G": GRAFT})
    return solver


def accepted(solver):
    """Whether check_partition_functions accepts the state."""
    try:
        check_partition_functions(solver, "the test state")
    except NumericalError as exc:
        assert "partition function of polymer 0" in str(exc) and exc.fields is None, str(exc)
        return False
    return True


def child_output(platform, checks):
    """Output of a separate process that solves W = 1e5, with or without the checks."""
    result = subprocess.run([sys.executable, __file__, "--child", platform, str(int(checks))],
                            capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


def negate_partition_functions(sim):
    native = sim.prop_solver.get_partition_function
    sim.prop_solver.get_partition_function = lambda polymer: -native(polymer)


def expect_partition_error(run, state):
    """run() raises a NumericalError for the partition function at state, with the 16-point
    monomer fields of that state."""
    try:
        with np.errstate(invalid="ignore"), silenced():
            run()
    except NumericalError as exc:
        fields = exc.fields
        return ("partition function of polymer 0" in str(exc) and state in str(exc)
                and isinstance(fields, dict) and sorted(fields) == ["A", "B"]
                and all(np.shape(fields[m]) == (16,) and np.all(np.isfinite(fields[m])) for m in fields))
    return False


def check_simulations(root):
    """SCFT and L-FTS reject a non-positive Q at their accepted states."""
    failures = []
    params = diblock_parameters(root / "lfts")
    with silenced():
        lfts = LFTS(params, random_seed=11)
    negate_partition_functions(lfts)
    if not expect_partition_error(lambda: lfts.run(diblock_fields()), "the initial saddle point"):
        failures.append("L-FTS accepted a saddle point with a negative Q")

    for key in ("langevin", "compressor", "saddle", "recording", "verbose_level"):   # SCFT refuses them
        params.pop(key)
    params.update(optimizer={"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
                  max_iter=3, tolerance=1e-8, relax_cell=False)
    with silenced():
        scft = SCFT(params)
    negate_partition_functions(scft)
    if not expect_partition_error(lambda: scft.run(diblock_fields()), "the end of the SCFT iteration"):
        failures.append("SCFT accepted final fields with a negative Q")
    return failures


def main():
    if sys.argv[1:2] == ["--child"]:
        solver = solve(sys.argv[2], 1e5)
        if sys.argv[3] == "1":
            solver.partition_errors()
            solver.check_total_partition()
        return 0

    failures = []
    valid = {}
    for platform in PlatformSelector.avail_platforms():
        solver = solve(platform, 1e5)
        if np.isfinite(solver.partition_errors()[0]) or solver.check_total_partition() or accepted(solver):
            failures.append(f"{platform}: Q = {solver.get_partition_function(0)!r} at W = 1e5 was accepted")
        if child_output(platform, True) != child_output(platform, False):
            failures.append(f"{platform}: partition_errors() or check_total_partition() printed output")

        solver = solve(platform, 100.0)
        if accepted(solver) and solver.check_total_partition():
            failures.append(f"{platform}: the round-off result Q = {solver.get_partition_function(0):.3e} "
                            "at W = 100 passed both checks")

        solver = solve(platform, 10.0)
        valid[platform] = solver.get_partition_function(0)
        if not (accepted(solver) and solver.check_total_partition()):
            failures.append(f"{platform}: the valid result at W = 10 was rejected")
    values = list(valid.values())
    if np.ptp(values) > 1e-10 * max(values):
        failures.append(f"W = 10 differs between platforms: {valid}")

    if any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)  # any stray output stays in the temporary directory
            try:
                failures += check_simulations(Path(directory))
            finally:
                os.chdir(previous)

    for failure in failures:
        print("[FAIL]", failure)
    if failures or cuda_part_missing():
        return 1
    print(f"Partition guard: OK (platforms: {', '.join(valid)}; Q at W = 10: {values[0]:.6e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
