#!/usr/bin/env python3
"""LFTS's noninteracting canonical limit has no auxiliary fields or saddle search.

Check analytic mixture densities and H with each compressor, actual LM/Euler
runs, checkpointed RNG/noise/pending observations, and an intermediate-force
extension. Finite compressibility and incompressibility keep their pressure
saddle and serve as controls. The ideal limit is decided in Python, but every
platform runs it: the L-FTS runs with the LR and LRAM compressors run on CUDA too.
"""

import copy
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import LFTS, PlatformSelector
from cuda_part import cuda_part_missing
from fts_checks import exact_restart, quiet


FRACTIONS = {"A": 0.3, "B": 0.7}


def parameters(directory, platform, compressor="am", method="lm", steps=4, zeta=0.0):
    return {
        "nx": [4], "lx": [2.0], "ds": 0.25, "chain_model": "continuous",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 0.0}, "zeta_n": zeta,
        "polymers": [{"volume_fraction": FRACTIONS[name], "blocks": [{"type": name, "length": length}]}
                     for name, length in (("A", 1.0), ("B", 0.5))],
        "platform": platform,
        "langevin": {"nbar": 10000.0, "dt": 0.01, "max_step": steps, "integrator": method},
        "compressor": {"name": compressor, "max_hist": 10, "start_error": 0.1,
                       "mix_min": 0.1, "mix_init": 0.1},
        "saddle": {"max_iter": 100, "tolerance": 1e-9},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 4},
        "verbose_level": 0,
    }


class Observations:
    """Stateful accepted-step observer: its pending history must also survive a restart."""
    settings = {}

    def __init__(self):
        self.steps = []
        self.masses = []

    def __call__(self, sample):
        for name, fraction in FRACTIONS.items():
            np.testing.assert_allclose(sample["phi"][name], fraction, rtol=0, atol=2e-12)
        mass = float(np.mean(sum(sample["phi"].values())))
        np.testing.assert_allclose(mass, 1.0, rtol=0, atol=2e-12)
        np.testing.assert_allclose(sample["hamiltonian"], 0.0, rtol=0, atol=2e-12)
        assert sample["drift_max"] == 0.0
        self.steps.append(int(sample["step"]))
        self.masses.append(mass)

    def state_dict(self):
        return {"steps": list(self.steps), "masses": list(self.masses)}

    def load_state_dict(self, state):
        self.steps = list(state["steps"])
        self.masses = list(state["masses"])


def ideal_runs(root, platform, compressor, method):
    observers, simulations, initial_rng = {}, {}, {}

    def make(name, steps, seed):
        observers[name] = Observations()
        sim = quiet(LFTS, parameters(root / name, platform, compressor, method, steps),
                    random_seed=seed, callbacks={"observations": observers[name]})
        assert sim.mpt.R == sim.mpt.I == 0
        assert len(sim.mpt.aux_fields_dropped_idx) == 2
        if name == "full":
            initial_rng.update(copy.deepcopy(sim.random_bg.state))
        simulations[name] = sim
        return sim

    # All supplied potentials belong to dropped modes and must be pinned to the ideal reference.
    rng = np.random.default_rng(19)
    initial = {name: rng.normal(0.0, 5.0, 4) for name in FRACTIONS}

    def start(sim):
        status = sim.run(initial)
        assert status["status"] == "completed" and status["saddle_failures"] == 0
        assert sim.saddle.iterations == 0, "An empty saddle must need no compression iterations"
        assert sim.random_bg.state == initial_rng, "There are no active fields to draw noise for"
        for p in range(2):
            np.testing.assert_allclose(sim.prop_solver.get_partition_function(p), 1.0, rtol=0, atol=2e-12)
        return status

    def inspect(checkpoint):
        pending = loadmat(checkpoint, squeeze_me=True)
        assert pending["record_sample_count"] == 2
        assert np.size(pending["record_H_history"]) == 2
        assert np.size(pending["normal_noise_prev"]) == 0
        np.testing.assert_array_equal(pending["w_aux"], np.zeros((2, 4)))

    fields_keys = ("w_", "phi_", "normal_noise", "record_", "rng_state", "extension_state")
    exact_restart(make, start, 4, 2, {
        "fields_000003.mat": fields_keys, "fields_000004.mat": fields_keys,
        "dH_000004.mat": ("H_", "dH_", "drift_"),
        "structure_function_000004.mat": ("structure_function_",)}, inspect)
    assert simulations["continued"].random_bg.state == initial_rng
    assert observers["full"].state_dict() == observers["continued"].state_dict()
    assert observers["continued"].steps == [1, 2, 3, 4]
    recorded = loadmat(root / "full" / "dH_000004.mat", squeeze_me=True)
    assert recorded["sample_count"] == 4
    np.testing.assert_allclose(recorded["H_history"], np.zeros(4), rtol=0, atol=2e-12)
    np.testing.assert_array_equal(recorded["drift_max_history"], np.zeros(4))

    # A new run restores the observer's constructor state, and empty noise still leaves RNG untouched.
    if compressor == "am" and method == "lm":
        quiet(simulations["full"].run, initial)
        assert observers["full"].steps == [1, 2, 3, 4]
        assert simulations["full"].random_bg.state == initial_rng


def pressure_controls(root, platform):
    # These retain one pressure field, so their compressors and convergence checks still operate.
    for zeta in (None, 1.0):
        for compressor in ("am", "lr", "lram"):
            observer = Observations()
            sim = quiet(LFTS, parameters(root / f"{compressor}_{zeta}", platform, compressor, steps=1, zeta=zeta),
                        random_seed=17, callbacks={"observations": observer})
            assert sim.mpt.R == 0 and sim.mpt.I == 1
            status = quiet(sim.run, {name: np.zeros(4) for name in FRACTIONS})
            assert status["status"] == "completed"
            assert sim.saddle.iterations >= 1
            assert observer.steps == [1]
            assert quiet(sim.prop_solver.check_total_partition)


def active_real_without_saddle(root, platform):
    # The positive eigenvalues fall below the existing dropped-mode threshold,
    # while the negative eigenvalue is retained. No imaginary field does not
    # imply that the supplied real field, its densities or its energy are zero.
    fractions = {"A": 0.2, "B": 0.3, "C": 0.5}
    for compressor in ("am", "lr", "lram"):
        params = parameters(root / compressor, platform, compressor)
        params.update(segment_lengths={name: 1.0 for name in fractions},
                      chi_n={"A,B": -7.5e-9, "A,C": -7.5e-9, "B,C": -7.5e-9},
                      polymers=[{"volume_fraction": fraction, "blocks": [{"type": name, "length": 1.0}]}
                                for name, fraction in fractions.items()])
        sim = quiet(LFTS, params, random_seed=17)
        assert sim.mpt.R == 1 and sim.mpt.I == 0
        assert len(sim.mpt.aux_fields_dropped_idx) == 2
        fields = np.zeros((3, 4))
        fields[sim.mpt.aux_fields_real_idx[0]] = np.array([2.0, -3.0, 4.0, -1.0]) * 1e-5
        before = fields.copy()
        expected_phi, _ = quiet(sim.compute_concentrations, fields)
        expected_phi = {name: values.copy() for name, values in expected_phi.items()}
        partitions = [sim.prop_solver.get_partition_function(p) for p in range(3)]
        expected_h = sim.mpt.compute_hamiltonian(
            sim.prop_solver.molecules, fields, partitions, sim.cb,
            include_const_term=True, phi=sim.smearing.apply_to_dict(expected_phi))
        assert np.isfinite(expected_h) and abs(expected_h) > 1e-5
        phi, _, hamiltonian, iterations, error = quiet(sim.saddle.solve, fields)
        np.testing.assert_array_equal(fields, before)
        assert iterations == 0 and error == 0.0
        np.testing.assert_allclose(hamiltonian, expected_h, rtol=0, atol=2e-13)
        for name, fraction in fractions.items():
            np.testing.assert_allclose(phi[name], expected_phi[name], rtol=0, atol=2e-13)
            np.testing.assert_allclose(np.mean(phi[name]), fraction, rtol=0, atol=2e-12)
        assert quiet(sim.prop_solver.check_total_partition)


class ProbeEmptyForce:
    name = "probe-empty-force"
    settings = {}

    def step(self, fields, force, *, evaluate_force, **kwargs):
        assert fields.shape == force.shape == (0, 4)
        intermediate = evaluate_force(fields.copy())
        assert intermediate.shape == fields.shape and intermediate.size == 0
        return fields.copy()


def intermediate_force(root, platform):
    observer = Observations()
    sim = quiet(LFTS, parameters(root, platform, steps=1), random_seed=17,
                integrator=ProbeEmptyForce(), callbacks={"observations": observer})
    status = quiet(sim.run, {name: np.zeros(4) for name in FRACTIONS})
    assert status["status"] == "completed" and observer.steps == [1]
    assert sim.saddle.iterations == 0   # the final and the intermediate saddle points need no iteration


def main():
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return int(cuda_part_missing()) or 77
    # Also the L-FTS runs with the LR and LRAM compressors on CUDA
    for platform in platforms:
        with tempfile.TemporaryDirectory(prefix="polymerfts_lfts_ideal_") as directory:
            root = Path(directory)
            for compressor in ("am", "lr", "lram"):
                for method in ("lm", "euler"):
                    ideal_runs(root / f"{compressor}_{method}", platform, compressor, method)
            pressure_controls(root / "controls", platform)
            active_real_without_saddle(root / "active_real", platform)
            intermediate_force(root / "intermediate", platform)
            print(f"[PASS] {platform}: zero-field LFTS, compressor controls, LM/Euler restarts and extensions")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
