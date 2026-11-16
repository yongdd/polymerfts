#!/usr/bin/env python3
"""Absent grafted chains must not invalidate a converged canonical SCFT state.

A zero graft gives Q = 0 without extreme fields. A species of exactly zero
population contributes neither density nor log Q to the free energy, while
every positive population retains the positive, finite partition requirement.
The SCFT checks are Python: the first CPU platform runs them, and
TestGceZeroPartition.py checks the propagators of such a chain on every platform.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from polymerfts import PlatformSelector, SCFT, ValidationError
from cuda_part import cuda_part_missing
from polymerfts.utils.common import check_partition_functions
from fts_checks import quiet


def check_guard_policy():
    """No tolerance may reinterpret a tiny positive population as absent."""
    for fraction in (0.0, np.nextafter(0.0, 1.0), 1e-300, 0.25):
        for partition in (0.0, -1.0, np.nan, np.inf, -np.inf):
            solver = SimpleNamespace(
                field_type="real", get_n_polymer_types=lambda: 1,
                get_polymer=lambda p: SimpleNamespace(get_volume_fraction=lambda: fraction),
                get_partition_function=lambda p: partition)
            try:
                check_partition_functions(solver, "the test state")
            except ValidationError as error:
                assert fraction > 0.0, (fraction, partition, error)
                assert "partition function of polymer 0" in str(error)
            else:
                assert fraction == 0.0, (fraction, partition)
        for partition in (np.nextafter(0.0, 1.0), 1.0, np.finfo(float).max):
            check_partition_functions(solver, "the positive-partition state")
    # Complex-field positivity remains outside this real-field guard.
    solver.field_type = "complex"
    for partition in (0.0, -1.0, 1j):
        check_partition_functions(solver, "the complex-field state")


def parameters(platform, model, reduced, fraction=0.0, include_graft=True):
    polymers = []
    if include_graft:
        polymers.append({"volume_fraction": fraction, "initial_conditions": {0: "G"},
                         "blocks": [{"type": "B", "length": 1.0, "v": 0, "u": 1}]})
    polymers.append({"volume_fraction": 1.0 - fraction,
                     "blocks": [{"type": "A", "length": 1.0}]})
    return {
        "nx": [4], "lx": [2.0], "ds": 0.1, "chain_model": model,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 5.0},
        "polymers": polymers, "platform": platform, "reduce_memory": reduced,
        "relax_cell": False, "max_iter": 1, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                      "mix_min": 0.1, "mix_init": 0.1},
    }


def run_case(platform, model, reduced, profile):
    sim = quiet(SCFT, parameters(platform, model, reduced, include_graft=profile is not None))
    fields = {name: np.zeros(4) for name in ("A", "B")}
    profiles = None if profile is None else {"G": profile.copy()}
    result = quiet(sim.run, fields, q_init=profiles, return_result=True)
    assert result.converged
    assert quiet(sim.prop_solver.check_total_partition)
    np.testing.assert_allclose(result.phi["A"], 1.0, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(result.phi["B"], np.zeros(4))
    np.testing.assert_allclose(sim.prop_solver.mean(result.phi["A"] + result.phi["B"]),
                               1.0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(result.free_energy_absolute, -1.0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(result.free_energy, 0.0, rtol=0, atol=2e-12)
    if profile is not None:
        np.testing.assert_array_equal(profiles["G"], profile)
    return sim, result


def compare_state(actual, expected):
    for kind in ("phi", "w"):
        for name in ("A", "B"):
            np.testing.assert_allclose(getattr(actual, kind)[name], getattr(expected, kind)[name],
                                       rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual.free_energy_absolute, expected.free_energy_absolute,
                               rtol=0, atol=2e-12)


def check_scft(root, platform, model, reduced):
    sim, result = run_case(platform, model, reduced, np.zeros(4))
    assert result.partition_functions[0] == 0.0
    _, normal = run_case(platform, model, reduced, np.ones(4))
    _, omitted = run_case(platform, model, reduced, None)
    assert normal.partition_functions[0] > 0.0
    compare_state(result, normal)
    compare_state(result, omitted)

    # A successful run exposes a serializable result, including the genuine zero Q.
    root.mkdir(parents=True)
    for suffix in ("json", "mat"):
        path = root / ("result." + suffix)
        quiet(sim.save_results, path)
        saved = json.loads(path.read_text()) if suffix == "json" else loadmat(path, simplify_cells=True)
        assert saved["partition_functions"][0] == 0.0
        np.testing.assert_array_equal(saved["phi_B"], np.zeros(4))
        np.testing.assert_allclose(saved["free_energy_absolute"], -1.0, rtol=0, atol=2e-12)

    occupied = quiet(SCFT, parameters(platform, model, reduced, fraction=0.25))
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            quiet(occupied.run, {name: np.zeros(4) for name in ("A", "B")},
                  q_init={"G": np.zeros(4)}, return_result=True)
        except ValidationError:
            pass
        else:
            raise AssertionError("SCFT accepted a zero partition for an occupied polymer")
    return result


def main():
    check_guard_policy()
    # A fix in Python: one platform, a CPU one where the build has it
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return int(cuda_part_missing()) or 77
    platforms = [next((name for name in platforms if name.startswith("cpu")), platforms[0])]
    platform = platforms[0]
    with tempfile.TemporaryDirectory(prefix="polymerfts_absent_scft_") as directory:
        root = Path(directory)
        for model in ("continuous", "discrete"):
            full = check_scft(root / model / "full", platform, model, False)
            reduced = check_scft(root / model / "reduced", platform, model, True)
            for name in ("A", "B"):
                assert full.phi[name].tobytes() == reduced.phi[name].tobytes()
                assert full.w[name].tobytes() == reduced.w[name].tobytes()
            assert np.asarray(full.partition_functions).tobytes() == np.asarray(reduced.partition_functions).tobytes()
        print(f"[PASS] {platform}: absent grafts, occupied partitions and SCFT result serialization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
