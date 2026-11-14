"""A monomer named aux must coexist with the checkpoint's full auxiliary matrix.

Exercise actual LM/Euler runs, public MAT readers and exact continuation with
both checkpoint layouts. A malformed new matrix must never fall back to an old
alias. Run on every available backend, with ordinary monomer names as controls.
"""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import LFTS, CLFTS, PlatformSelector, ValidationError
from polymerfts.utils.checkpoint import auxiliary_field_key, extension_state
from polymerfts.utils.common import atomic_savemat
from polymerfts.utils.mat_fields import read_monomer_fields
from cuda_part import cuda_part_missing


PRIMARY = "checkpoint_aux_fields"


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def parameters(directory, platform, name, cls, method, steps):
    return {
        "nx": [8], "lx": [3.0], "ds": 0.05, "chain_model": "continuous",
        "segment_lengths": {name: 1.0, "B": 1.0}, "chi_n": {name + ",B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": name, "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": platform, "verbose_level": 0,
        "langevin": {"nbar": 10000.0, "dt": 0.001, "max_step": steps, "integrator": method},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5,
                       "mix_min": 0.01, "mix_init": 0.01},
        "saddle": {"max_iter": 500, "tolerance": 1e-8},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 3,
                      "pressure": cls is CLFTS, "concentrations": cls is CLFTS},
    }


class Observer:
    def __init__(self, real_fields):
        self.real_fields = real_fields
        self.history = []

    @property
    def settings(self):
        return {"real_fields": self.real_fields}

    def __call__(self, sample):
        np.testing.assert_allclose(np.mean(sum(sample["phi"].values())), 1.0, rtol=0, atol=5e-8)
        assert np.isfinite(sample["hamiltonian"])
        if self.real_fields:
            assert sample["error_level"] < 1e-8
        value = complex(sample["hamiltonian"])
        self.history.append([sample["step"], value.real, value.imag])

    def state_dict(self):
        return {"history": copy.deepcopy(self.history)}

    def load_state_dict(self, state):
        self.history = copy.deepcopy(state["history"])


def read(path):
    return {key: value for key, value in loadmat(path, squeeze_me=True).items() if not key.startswith("__")}


def compare_files(left, right, filename, prefixes):
    a, b = read(left / filename), read(right / filename)
    keys = {key for key in a if key.startswith(prefixes)}
    assert keys == {key for key in b if key.startswith(prefixes)} and keys
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


def check_checkpoint_fields(sim, data, name, cls):
    assert data["checkpoint_format_version"] == 3
    assert auxiliary_field_key(data) == PRIMARY
    auxiliary = np.asarray(data[PRIMARY])
    assert auxiliary.shape == (2, 8)
    expected = sim.mpt.to_monomer_fields(auxiliary)
    fields = read_monomer_fields(data)
    for monomer, values in zip(sim.monomer_types, expected):
        np.testing.assert_array_equal(fields[monomer], values)
    if cls is LFTS and name == "aux":
        assert np.size(data["w_aux"]) == 8
        np.testing.assert_array_equal(data["w_aux"], fields["aux"])
    else:
        # Existing consumers keep their matrix alias whenever it is unambiguous.
        np.testing.assert_array_equal(data["w_aux"], auxiliary)


def reject_before_restore(sim, path, data):
    atomic_savemat(path, data, long_field_names=True)
    rng = copy.deepcopy(sim.random_bg.state)
    extensions = copy.deepcopy(extension_state(sim))
    metadata = copy.deepcopy(sim.metadata)
    try:
        quiet(sim.continue_run, path)
    except ValidationError:
        pass
    else:
        raise AssertionError("A malformed auxiliary-field checkpoint was accepted")
    assert sim.random_bg.state == rng
    assert extension_state(sim) == extensions
    assert sim.metadata == metadata


def check_run(root, platform, name, cls, method):
    paths = {label: root / label for label in ("full", "partial", "continued")}
    simulations, observers = {}, {}
    for label, steps, seed in (("full", 5, 17), ("partial", 2, 17), ("continued", 5, 99)):
        observer = Observer(cls is LFTS)
        sim = quiet(cls, parameters(paths[label], platform, name, cls, method, steps), random_seed=seed,
                    callbacks={"observer": observer})
        sim.draw_architectures = False
        simulations[label], observers[label] = sim, observer
    initial = dict(zip((name, "B"), np.random.default_rng(19).normal(0.0, 5.0, (2, 8))))
    for label in ("full", "partial"):
        status = quiet(simulations[label].run, initial)
        assert status["status"] == "completed"
        assert quiet(simulations[label].prop_solver.check_total_partition)

    checkpoint = paths["partial"] / "fields_000002.mat"
    pending = read(checkpoint)
    check_checkpoint_fields(simulations["partial"], pending, name, cls)
    assert pending["record_sample_count"] == 2 and np.size(pending["record_H_history"]) == 2
    assert json.loads(pending["extension_state_json"])["callbacks"]["observer"]["history"] == observers["partial"].history

    continued, full = simulations["continued"], simulations["full"]

    def compare_continuation(path):
        status = quiet(continued.continue_run, path)
        assert status["status"] == "completed" and status["last_completed_step"] == 5
        assert quiet(continued.prop_solver.check_total_partition)
        assert continued.random_bg.state == full.random_bg.state
        assert observers["continued"].history == observers["full"].history
        if cls is LFTS:
            assert status["noise_resamples"] == full.run_status["noise_resamples"] == 0
            assert status["saddle_failures"] == full.run_status["saddle_failures"]
        for step in (3, 4, 5):
            filename = f"fields_{step:06d}.mat"
            data = read(paths["continued"] / filename)
            check_checkpoint_fields(continued, data, name, cls)
            compare_files(paths["full"], paths["continued"], filename,
                          (PRIMARY, "w_", "phi_", "normal_noise", "rng_state", "record_", "extension_state"))
        compare_files(paths["full"], paths["continued"], "dH_000003.mat",
                      ("H_", "dH_", "drift_", "chemical_potential_", "pressure_"))
        compare_files(paths["full"], paths["continued"], "structure_function_000003.mat",
                      ("structure_function_", "concentration_"))
        assert read(paths["continued"] / "fields_000005.mat")["record_sample_count"] == 2

    compare_continuation(checkpoint)
    # Older version-3 files have only w_aux. They remain exact continuations when
    # that key really holds the matrix; an already-damaged aux file cannot recover it.
    historical = copy.deepcopy(pending)
    historical.pop(PRIMARY)
    path = root / "historical.mat"
    if cls is LFTS and name == "aux":
        reject_before_restore(continued, path, historical)
    else:
        assert auxiliary_field_key(historical) == "w_aux"
        atomic_savemat(path, historical, long_field_names=True)
        compare_continuation(path)

    for label, value in (("shape", np.zeros(8)), ("nonfinite", np.full((2, 8), np.nan))):
        broken = copy.deepcopy(pending)
        broken[PRIMARY] = value
        # For ordinary names (and CLFTS) the old alias is still valid. It must
        # not conceal corruption in the authoritative new matrix.
        reject_before_restore(continued, root / (label + ".mat"), broken)
    if cls is LFTS:
        broken = copy.deepcopy(pending)
        broken[PRIMARY] = broken[PRIMARY].astype(np.complex128) + 1j
        reject_before_restore(continued, root / "complex.mat", broken)
    print(f"[PASS] {platform} {cls.__name__} {method} {name}/B: fields, records and exact restart")


def main():
    with tempfile.TemporaryDirectory(prefix="polymerfts_aux_checkpoint_") as directory:
        root = Path(directory)
        for platform in PlatformSelector.avail_platforms():
            for cls in (LFTS, CLFTS):
                for method in ("lm", "euler"):
                    for name in ("A", "aux"):
                        check_run(root / platform / cls.__name__ / method / name, platform, name, cls, method)
            # The new metadata key is also an allowed monomer name: its public
            # potential uses w_checkpoint_aux_fields and cannot overwrite it.
            check_run(root / platform / "new_key_name", platform, PRIMARY, LFTS, "lm")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
