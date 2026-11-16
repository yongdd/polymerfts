#!/usr/bin/env python3
"""Extended NumPy scalars in metadata, SCFT output and FTS checkpoints.

Ordinary scalar encodings stay identical. Long-double inputs terminate, retain
finite values at Python float precision and reject unrepresentable magnitudes.
Actual runs verify checkpoint compatibility and exact same-backend continuation.
The encodings and SCFT outputs are Python: one platform checks them. The L-FTS
and CL-FTS restarts run on every platform, which checks the exact continuation of
runs with real dynamics on CUDA too.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat
import yaml

from polymerfts import SCFT, LFTS, CLFTS, PlatformSelector
from cuda_part import cuda_part_missing
from polymerfts.utils.common import input_hash, json_ready, mat_ready, settings_json
from TestSmallFieldCoefficients import parameters
from fts_checks import exact_restart, python_platform, quiet


def scalar_encodings():
    ordinary = {"bool": np.bool_(True), "integer": np.int64(7),
                "float32": np.float32(0.25), "float64": np.float64(1.5),
                "complex": np.complex128(1.5 + 0.25j), "array": np.array([1.0, 2.0])}
    expected = {"bool": True, "integer": 7, "float32": 0.25, "float64": 1.5,
                "complex": {"real": 1.5, "imag": 0.25}, "array": [1.0, 2.0]}
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert settings_json(ordinary) == encoded
    assert input_hash(ordinary) == hashlib.sha256(encoded.encode()).hexdigest()
    extended = {"scalar": np.longdouble("10000"),
                "complex": np.clongdouble(1.5 + 0.25j),
                "array": np.array([[1.5, 2.0]], dtype=np.longdouble),
                "zero_dim": np.array(1.5, dtype=np.longdouble),
                "nested": [np.longdouble("0.125"), np.clongdouble(-2j)]}
    expected = {"scalar": 10000.0, "complex": {"real": 1.5, "imag": 0.25},
                "array": [[1.5, 2.0]], "zero_dim": 1.5,
                "nested": [0.125, {"real": -0.0, "imag": -2.0}]}
    assert json_ready(extended) == expected
    assert input_hash(extended) == input_hash(expected)
    assert isinstance(extended["scalar"], np.longdouble)
    assert extended["array"].dtype == np.dtype(np.longdouble)
    assert np.signbit(json_ready(extended)["nested"][1]["real"])
    assert np.signbit(json_ready(np.longdouble("-0.0")))
    assert mat_ready({"x": np.longdouble(1.5)}) == {"x": 1.5}
    for value, tag in ((np.inf, "inf"), (-np.inf, "-inf"), (np.nan, "nan")):
        assert json_ready(np.longdouble(value)) == {"nonfinite_float": tag}
    assert json.loads(settings_json(np.clongdouble(complex(np.inf, np.nan)))) == {
        "real": {"nonfinite_float": "inf"}, "imag": {"nonfinite_float": "nan"}}
    # Platforms where longdouble is just double have no extra exponent range.
    if np.finfo(np.longdouble).maxexp > np.finfo(float).maxexp:
        for value in (np.finfo(np.longdouble).max, np.finfo(np.longdouble).tiny):
            for convert in (json_ready, mat_ready):
                for item in (value, -value, np.clongdouble(value),
                             np.clongdouble(1) + np.clongdouble(1j) * value):
                    try:
                        convert(item)
                    except ValueError as error:
                        assert "float range" in str(error)
                    else:
                        raise AssertionError("Silently changed an out-of-range finite scalar")


def scft_outputs(root, platform):
    params = parameters(root, platform, zeta=None)
    params.update(relax_cell=False, max_iter=1, tolerance=np.longdouble("1e-8"),
                  optimizer={"name": "am", "max_hist": 5, "start_error": 0.01,
                             "mix_min": 0.1, "mix_init": 0.1})
    before = copy.deepcopy(params)
    sim = quiet(SCFT, params)
    result = quiet(sim.run, {name: np.zeros(4) for name in ("A", "B")}, return_result=True)
    assert result.converged and quiet(sim.prop_solver.check_total_partition)
    root.mkdir(parents=True, exist_ok=True)
    for extension in ("json", "yaml", "mat"):
        path = root / ("scft." + extension)
        quiet(sim.save_results, path)
        if extension == "mat":
            saved = loadmat(path, simplify_cells=True)
        else:
            saved = (json.loads if extension == "json" else yaml.safe_load)(path.read_text())
        assert saved["initial_params"]["tolerance"] == float(params["tolerance"])
        for name in ("A", "B"):
            np.testing.assert_allclose(saved["phi_" + name], result.phi[name], rtol=0, atol=0)
    assert params == before and isinstance(sim.params["tolerance"], np.longdouble)


def fts_restart(root, platform):
    initial = {name: values for name, values in zip(
        ("A", "B"), np.random.default_rng(19).normal(0.0, 5.0, (2, 4)))}
    for cls in (LFTS, CLFTS):
        def make(name, steps, seed):
            params = parameters(root / cls.__name__ / name, platform, zeta=100.0, steps=steps)
            params["langevin"]["nbar"] = np.longdouble(10000.0)
            before = copy.deepcopy(params)
            sim = quiet(cls, params, random_seed=seed)
            sim.draw_architectures = False
            control = copy.deepcopy(sim.params)
            control["langevin"]["nbar"] = 10000.0
            assert sim.metadata["input_hash"] == input_hash(control)
            assert params == before and isinstance(params["langevin"]["nbar"], np.longdouble)
            return sim

        def inspect(checkpoint):
            data = loadmat(checkpoint, squeeze_me=True)
            assert data["nbar"] == 10000.0 and data["record_sample_count"] == 2
            stored = json.loads(str(data["checkpoint_settings_json"]))
            assert stored["nbar"] == 10000.0

        fields_keys = ("w_", "phi_", "normal_noise", "record_", "rng_state", "extension_state")
        exact_restart(make, lambda sim: sim.run(initial), 4, 2, {
            "fields_000003.mat": fields_keys, "fields_000004.mat": fields_keys,
            "dH_000004.mat": ("H_", "dH_", "drift_", "chemical_potential_"),
            "structure_function_000004.mat": ("structure_function_",)}, inspect)
        for step in (3, 4):
            saved = loadmat(root / cls.__name__ / "full" / f"fields_{step:06d}.mat", squeeze_me=True)
            phi = (saved["phi_A"] + saved["phi_B"] if cls is LFTS else
                   saved["phi_A_real"] + saved["phi_B_real"]
                   + 1j * (saved["phi_A_imag"] + saved["phi_B_imag"]))
            np.testing.assert_allclose(np.mean(phi),
                                       1.0, rtol=0, atol=5e-8)


def main():
    scalar_encodings()
    platform = python_platform()
    with tempfile.TemporaryDirectory(prefix="polymerfts_numpy_scalars_") as directory:
        root = Path(directory)
        scft_outputs(root / "scft", platform)
        print(f"[PASS] {platform}: NumPy scalar metadata and SCFT outputs")
        for platform in PlatformSelector.avail_platforms():
            fts_restart(root / "fts" / platform, platform)
            print(f"[PASS] {platform}: exact L-FTS and CL-FTS restarts")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
