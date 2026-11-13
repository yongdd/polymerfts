#!/usr/bin/env python3
"""Random-block fractions must preserve the canonical monomer mass.

Reject rounded sums outside the same absolute tolerance used for total polymer
fractions. Exact and roundoff-sized inputs stay unchanged, give the homogeneous
SCFT state and preserve mass in nonuniform native propagator calculations.
"""

import contextlib
import copy
import io
import itertools
import os
from pathlib import Path
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from polymerfts import SCFT, LFTS, CLFTS, PlatformSelector, ValidationError
from polymerfts.theory.chains import process_random_copolymer, read_chains
from polymerfts.utils.validation import validate_system_params
from cuda_part import cuda_part_missing


def quiet(call):
    with contextlib.redirect_stdout(io.StringIO()):
        return call()


def parameters(platform, fractions, directory, model="continuous"):
    return {
        "nx": [16], "lx": [4.0], "chain_model": model, "ds": 0.05,
        "segment_lengths": {name: 1.0 + 0.2 * i for i, name in enumerate(sorted(fractions))},
        "chi_n": {",".join(pair): 15.0 for pair in itertools.combinations(sorted(fractions), 2)},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "R", "length": 1.0, "monomer_fractions": dict(fractions)}]}],
        "platform": platform, "relax_cell": False, "max_iter": 1, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                      "mix_min": 0.1, "mix_init": 0.1},
        "langevin": {"nbar": 10000.0, "dt": 0.1, "max_step": 0},
        "compressor": {"name": "am", "max_hist": 5, "start_error": 0.5,
                       "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 100, "tolerance": 1e-8},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 1},
        "verbose_level": 0,
    }


def expect_sum_error(call):
    try:
        quiet(call)
    except ValidationError as error:
        assert "monomer_fractions" in str(error) and "sum to 1" in str(error), str(error)
    else:
        raise AssertionError("Accepted random fractions whose sum is outside the composition tolerance")


def check_rejected(platform, directory):
    for fractions in (
        {"A": 0.5, "B": 0.500009},
        {"A": 0.33333, "B": 0.33333, "C": 0.33333},
        {"A": 0.5, "B": 0.5000000002},
        {"A": 0.5, "B": 0.4999999998},
    ):
        params = parameters(platform, fractions, directory)
        before = copy.deepcopy(params)
        expect_sum_error(lambda: validate_system_params(params))
        expect_sum_error(lambda: process_random_copolymer(params["polymers"][0], params["segment_lengths"]))
        expect_sum_error(lambda: read_chains(params))
        for cls in (SCFT, LFTS, CLFTS):
            expect_sum_error(lambda: cls(params))
        assert params == before, "Rejected input was modified"


def check_scft(platform, directory):
    accepted = (
        {"A": 0.3, "B": 0.7},
        {"A": 0.1 + 0.2, "B": 0.7},
        {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0},
        {"A": 0.5, "B": 0.5 + 5e-11},
    )
    for model in ("continuous", "discrete"):
        equivalent = None
        for index, fractions in enumerate(accepted):
            params = parameters(platform, fractions, directory, model)
            before = copy.deepcopy(params)
            validate_system_params(params)
            random = process_random_copolymer(params["polymers"][0], params["segment_lengths"])
            assert random[2] == fractions
            sim = quiet(lambda: SCFT(params))
            assert sim.random_fraction["R"] == fractions
            assert sim.params["polymers"][0]["blocks"][0]["monomer_fractions"] == fractions
            fields = {name: np.zeros(16) for name in fractions}
            result = quiet(lambda: sim.run(fields, return_result=True))
            assert result.converged and result.error_level < 1e-12
            assert quiet(sim.prop_solver.check_total_partition)
            np.testing.assert_allclose(result.partition_functions, [1.0], rtol=0, atol=2e-13)
            np.testing.assert_allclose(result.free_energy, 0.0, rtol=0, atol=2e-12)
            for name, fraction in fractions.items():
                np.testing.assert_allclose(result.phi[name], fraction, rtol=0, atol=2e-13)
            np.testing.assert_allclose(sum(np.mean(value) for value in result.phi.values()),
                                       1.0, rtol=0, atol=1e-10)
            if index == 0:
                equivalent = result
            elif index == 1:
                for name in fractions:
                    np.testing.assert_allclose(result.phi[name], equivalent.phi[name], rtol=0, atol=2e-13)
                np.testing.assert_allclose(result.free_energy_absolute, equivalent.free_energy_absolute,
                                           rtol=0, atol=2e-13)

            # Strong nonuniform monomer fields exercise the actual random-chain
            # propagators and the conversion back to each monomer concentration.
            rng = np.random.default_rng(31)
            w = rng.normal(0.0, 5.0, (len(fractions), 16))
            phi, _ = quiet(lambda: sim.compute_concentrations(w))
            native = sim.prop_solver.get_concentration("R")
            assert quiet(sim.prop_solver.check_total_partition)
            np.testing.assert_allclose(sim.prop_solver.mean(native), 1.0, rtol=0, atol=2e-11)
            for name, fraction in fractions.items():
                np.testing.assert_allclose(phi[name], fraction * native, rtol=0, atol=2e-13)
                np.testing.assert_allclose(sim.prop_solver.mean(phi[name]), fraction, rtol=0, atol=2e-11)
            np.testing.assert_allclose(sim.prop_solver.mean(sum(phi[name] for name in fractions)),
                                       sum(fractions.values()), rtol=0, atol=2e-11)
            assert params == before, "Accepted input was modified"
        print(f"[PASS] {platform} {model}: random fractions, SCFT and native mass conservation")


def check_fts_constructors(platform, directory):
    for cls in (LFTS, CLFTS):
        for fractions in ({"A": 0.3, "B": 0.7}, {"A": 0.5, "B": 0.5 + 5e-11}):
            params = parameters(platform, fractions, directory / cls.__name__)
            before = copy.deepcopy(params)
            sim = quiet(lambda: cls(params, random_seed=17))
            assert sim.random_fraction["R"] == fractions
            assert sim.params["polymers"][0]["blocks"][0]["monomer_fractions"] == fractions
            assert params == before, "FTS constructor modified the input"
            del sim


def main():
    with tempfile.TemporaryDirectory(prefix="polymerfts-random-fractions-") as scratch:
        directory = Path(scratch)
        for platform in PlatformSelector.avail_platforms():
            check_rejected(platform, directory)
            check_scft(platform, directory)
            check_fts_constructors(platform, directory)
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
