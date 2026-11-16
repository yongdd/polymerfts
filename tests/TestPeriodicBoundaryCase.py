#!/usr/bin/env python3
"""Pressure accepts the same case-insensitive periodic names as the solver.

Lowercase and mixed-case CLFTS runs record identical pressure and fields with
LM and ETD, and each restarts bit for bit with pending samples. Gibbs boxes also
agree, while reflecting/absorbing boxes remain invalid for pressure and ETD.
Boundary spelling in input and checkpoint settings must not be rewritten. The
checks of the names are Python: the first CPU platform runs them.
"""

import copy
import json
import os
from pathlib import Path
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, GibbsEnsemble, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing
from fts_checks import compare_files, exact_restart, quiet


LOWER = ["periodic"] * 4
MIXED = ["Periodic", "periodic", "PERIODIC", "PeRiOdIc"]


def parameters(platform, directory, bc, steps=3, method="lm"):
    return {
        "nx": [4, 4], "lx": [1.0, 1.0], "ds": 0.05, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 3.0}, "zeta_n": 50.0,
        "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 1.0}]},
                     {"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 1.0}]}],
        "bc": list(bc), "platform": platform,
        "langevin": {"nbar": 1e4, "dt": 0.01, "max_step": steps, "integrator": method},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 3, "pressure": True},
        "verbose_level": 0,
    }


def check_settings(directory, step, bc):
    checkpoint = loadmat(directory / f"fields_{step:06d}.mat", squeeze_me=True)
    settings = json.loads(str(checkpoint["checkpoint_settings_json"]))
    assert settings["bc"] == bc, "boundary spelling changed in checkpoint settings"


def compare_clfts(platform, root, method):
    x, y = np.meshgrid(np.arange(4) / 4, np.arange(4) / 4, indexing="ij")
    field = (0.5 * np.cos(2 * np.pi * x) + 0.3 * np.sin(2 * np.pi * y)).ravel()
    for label, bc in (("lower", LOWER), ("mixed", MIXED)):
        def make(name, steps, seed):
            params = parameters(platform, root / label / name, bc, steps, method)
            before = copy.deepcopy(params)
            sim = quiet(CLFTS, params, random_seed=seed)
            sim.draw_architectures = False
            assert params == before and sim.prop_solver.bc == bc
            return sim

        def start(sim):
            status = sim.run({"A": field, "B": -field})
            assert sim.prop_solver.bc == bc
            return status

        def inspect(checkpoint):
            assert np.size(loadmat(checkpoint, squeeze_me=True)["record_pressure_history"]) == 1

        sims = exact_restart(make, start, 3, 1, {
            "fields_000003.mat": ("w_", "normal_noise", "random_state"),
            "dH_000003.mat": ("H_history", "pressure_history", "chemical_potential_history")}, inspect)
        assert sims["continued"].prop_solver.bc == bc
        for name, step in (("full", 3), ("partial", 1), ("continued", 3)):
            check_settings(root / label / name, step, bc)
        full_data = loadmat(root / label / "full" / "dH_000003.mat")
        for name in ("pressure_history_real", "pressure_history_imag"):
            assert np.size(full_data[name]) == 3 and np.all(np.isfinite(full_data[name]))
    compare_files(root / "lower" / "full", root / "mixed" / "full", "fields_000003.mat",
                  ("w_A_", "w_B_", "normal_noise_prev"))
    compare_files(root / "lower" / "full", root / "mixed" / "full", "dH_000003.mat", ("pressure_history_",))


def gibbs_params(platform, directory, bc):
    params = parameters(platform, directory, bc)
    params["gibbs_ensemble"] = {
        "boxes": [{"volume": 1.0, "volume_fractions": [0.7, 0.3]},
                  {"volume": 1.0, "volume_fractions": [0.3, 0.7]}],
        "cycles": 1, "cycle_steps": 2, "skip_steps": 0, "exchange_rate": 0.01,
    }
    return params


def compare_gibbs(platform, root):
    histories = []
    for label, bc in (("lower", LOWER), ("mixed", MIXED)):
        params = gibbs_params(platform, root / f"gibbs_{label}", bc)
        before = copy.deepcopy(params)
        sim = quiet(GibbsEnsemble, params, random_seed=12)
        quiet(sim.run)
        assert params == before
        assert len(sim.history["cycles"]) == 1
        histories.append(sim.history["cycles"])
        np.testing.assert_allclose(np.sum(sim.amounts, axis=0), np.sum(sim.initial_amounts, axis=0),
                                   rtol=0, atol=2e-15)
    assert histories[0] == histories[1], "case changed Gibbs pressure or particle exchange"


def reject_walls(platform, root):
    for wall in ("Reflecting", "ABSORBING"):
        etd = parameters(platform, root / f"{wall}_etd", [wall] * 4, method="etd")
        # Exercise ETD's guard independently of the pressure observable.
        etd["recording"]["pressure"] = False
        for make in (lambda: CLFTS(parameters(platform, root / wall, [wall] * 4), random_seed=1),
                     lambda: GibbsEnsemble(gibbs_params(platform, root / wall, [wall] * 4), random_seed=1),
                     lambda: CLFTS(etd, random_seed=1)):
            try:
                quiet(make)
            except ValidationError as error:
                assert "periodic" in str(error), str(error)
            else:
                raise AssertionError(f"Pressure or ETD accepted {wall} boundaries")


def main():
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return int(cuda_part_missing()) or 77
    # The only CL-FTS restart (LM and ETD) that also runs on CUDA
    for platform in platforms:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("lm", "etd"):
                compare_clfts(platform, root / method, method)
            compare_gibbs(platform, root)
            reject_walls(platform, root)
            print(f"[PASS] {platform}: periodic boundary case, LM/ETD pressure restart and Gibbs ensemble")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
