"""Failure semantics, restart compatibility, pending samples, initial fields, previous noise and atomic writes.

A failed computation raises NumericalError, of which SaddlePointError is a subclass; L-FTS records a
SaddlePointError as aborted_saddle_failure, CL-FTS every error as failed with its type and message.
"""
import copy
import json
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import (LFTS, CLFTS, PlatformSelector, LeimkuhlerMatthews, ValidationError, SaddlePointError,
                        NumericalError)
from polymerfts.utils.common import atomic_savemat
from polymerfts.utils.validation import validate_lfts_params
from fts_checks import (FailingLFTS, diblock_fields, diblock_parameters, exact_restart, expect_error,
                        silenced)


def expect_refused(sim, fields, *messages, **run_options):
    """sim.run(fields, **run_options) raises a ValidationError whose message contains messages."""
    try:
        sim.run(fields, **run_options)
    except ValidationError as error:
        assert all(message in str(error) for message in messages), str(error)
    else:
        raise AssertionError(f"{type(sim).__name__} accepted invalid input: expected {messages}")


def expect_numerical(call, message):
    """call() raises a NumericalError, not an error of the input (ValidationError), whose message
    contains message."""
    try:
        call()
    except NumericalError as error:
        assert not isinstance(error, ValidationError) and message in str(error), str(error)
    else:
        raise AssertionError(f"Expected NumericalError: {message}")


def numerical_failures(root):
    """Failed computations raise NumericalError: a partition consistency check that fails, a Hamiltonian or
    a force that is not finite at an intermediate saddle point of L-FTS, intermediate CL-FTS fields or
    forces that are not finite, and concentrations that are not finite after a CL-FTS step."""
    for cls, dtype in ((LFTS, np.float64), (CLFTS, np.complex128)):
        sim = cls(diblock_parameters(root / (cls.__name__ + "_partition"), 1, cls), random_seed=5)
        sim._partition_checked = False
        with patch("polymerfts.theory.chains.report_partition_consistency", return_value=False):
            expect_numerical(lambda: sim.compute_concentrations(np.zeros((2, 16), dtype)), "Partition consistency")

    # The forces at the intermediate fields of an integrator in L-FTS (SaddlePointSearch.evaluate_force)
    lfts = LFTS(diblock_parameters(root / "LFTS_intermediate", 1), random_seed=5)
    solve = lfts.saddle.solve

    def non_finite_hamiltonian(w_aux):
        phi, _, iterations, error_level = solve(w_aux)
        return phi, np.nan, iterations, error_level
    lfts.saddle.solve = non_finite_hamiltonian
    expect_numerical(lambda: lfts.saddle.evaluate_force(np.zeros((1, 16)), np.zeros((2, 16)), 1),
                     "Non-finite Hamiltonian at an intermediate saddle point of step 1")
    lfts.saddle.solve = solve
    derivative, real = lfts.mpt.compute_func_deriv, list(lfts.mpt.aux_fields_real_idx)
    lfts.mpt.compute_func_deriv = lambda w_aux, phi, indices: (
        derivative(w_aux, phi, indices) * (np.nan if list(indices) == real else 1.0))
    expect_numerical(lambda: lfts.saddle.evaluate_force(np.zeros((1, 16)), np.zeros((2, 16)), 1),
                     "Non-finite intermediate LFTS force")

    # CL-FTS: intermediate fields and forces (CLFTS.evaluate_force), and the concentrations after a step
    clfts = CLFTS(diblock_parameters(root / "CLFTS_intermediate", 1, CLFTS), random_seed=5)
    fields = np.zeros((2, 16), np.complex128)
    fields[0, 3] = np.nan
    expect_numerical(lambda: clfts.evaluate_force(fields), "Intermediate CLFTS fields have non-finite values")
    derivative = clfts.mpt.compute_func_deriv
    clfts.mpt.compute_func_deriv = lambda w_aux, phi, indices: derivative(w_aux, phi, indices) * np.nan
    expect_numerical(lambda: clfts.evaluate_force(np.zeros((2, 16), np.complex128)),
                     "Non-finite intermediate CLFTS force")
    directory = root / "CLFTS_concentrations"
    clfts = CLFTS(diblock_parameters(directory, 1, CLFTS), random_seed=5)
    compute, calls = clfts.compute_concentrations, []

    def non_finite_after_the_first(w_aux):
        phi, elapsed_time = compute(w_aux)
        calls.append(len(calls))
        if len(calls) > 1:
            phi = {m: np.full_like(values, np.nan) for m, values in phi.items()}
        return phi, elapsed_time
    clfts.compute_concentrations = non_finite_after_the_first
    expect_numerical(lambda: clfts.run(diblock_fields()), "Non-finite complex fields or concentrations at step 1")
    status = json.loads((directory / "run_status.json").read_text())
    assert status["status"] == "failed" and status["reason"].startswith("NumericalError"), status


class FailedSolveStep(LeimkuhlerMatthews):
    """An integrator whose step fails like a saddle-point solve inside it."""
    name = "test-failed-solve"

    def step(self, fields, force, **kwargs):
        raise SaddlePointError("Test failure of a solve inside the step")


class NonFiniteStep(LeimkuhlerMatthews):
    """An integrator whose step leaves non-finite fields."""
    name = "test-non-finite"

    def step(self, fields, force, **kwargs):
        return np.full_like(fields, np.nan)


def failed_clfts_run(directory, kind, integrator="lm", partition=None):
    """The error of kind that stops a CL-FTS run, which run_status.json records as failed with
    its type and message; partition replaces every partition function."""
    sim = CLFTS(diblock_parameters(directory, cls=CLFTS), random_seed=5, integrator=integrator)
    if partition is not None:
        sim.prop_solver.get_partition_function = lambda polymer: partition
    try:
        sim.run(diblock_fields())
    except kind as error:
        status = json.loads((directory / "run_status.json").read_text())
        assert status["status"] == "failed" and status["reason"] == f"{kind.__name__}: {error}", status
        assert status["last_completed_step"] == 0, status
        return error
    raise AssertionError(f"Expected {kind.__name__}")


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        previous = Path.cwd()
        os.chdir(root)  # any stray output stays in the temporary directory
        try:
            with silenced():
                for cls in (LFTS, CLFTS):
                    exact_restart(
                        lambda name, steps, seed: cls(diblock_parameters(root / (cls.__name__ + name), steps, cls),
                                                      random_seed=seed),
                        lambda sim: sim.run(diblock_fields()), 4, 1, {
                            "fields_000004.mat": ("w_", "phi_"),
                            "dH_000002.mat": ("H_history", "dH_history", "drift_max_history"),
                            "structure_function_000002.mat": ("structure_function",)})
                    # The run draws the chain architectures into its recording directory, and
                    # neither the constructor nor the run writes into the current directory
                    assert (root / (cls.__name__ + "full") / "polymer_0.png").is_file(), cls.__name__
                    assert not (root / "polymer_0.png").exists(), cls.__name__
                    checkpoint = root / (cls.__name__ + "partial") / "fields_000001.mat"
                    equivalent_params = diblock_parameters(root / (cls.__name__ + "equivalent"), cls=cls)
                    equivalent_params["langevin"]["nbar"] = 10000.0
                    equivalent_params["chi_n"] = {"A,B": 15.0}
                    equivalent = cls(equivalent_params, random_seed=99)
                    assert equivalent.continue_run(checkpoint)["status"] == "completed"
                    for key in ("dt", "nbar"):
                        params = diblock_parameters(root / "incompatible", cls=cls)
                        params["langevin"][key] *= 2
                        bad = cls(params, random_seed=13)
                        rng_before = copy.deepcopy(bad.random_bg.state)
                        expect_error(lambda: bad.continue_run(checkpoint), ValidationError)
                        assert bad.random_bg.state == rng_before

                normal = LFTS(diblock_parameters(root / "normal"), random_seed=5)
                normal.run(diblock_fields())
                FailingLFTS.fail_calls = (3,)
                recovered = FailingLFTS(diblock_parameters(root / "recovered"), random_seed=5)
                result = recovered.run(diblock_fields())
                assert result["saddle_failures"] == 1 and result["noise_resamples"] == 0
                assert recovered.random_bg.state == normal.random_bg.state
                params = diblock_parameters(root / "resampling")
                params["saddle"]["failure_policy"] = "resample"
                resampling = FailingLFTS(params, random_seed=5)
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always")
                    result = resampling.run(diblock_fields())
                assert result["noise_resamples"] == 1
                assert any("sampled distribution" in str(message.message) for message in messages)
                assert resampling.random_bg.state != normal.random_bg.state
                params = diblock_parameters(root / "abort")
                params["saddle"]["failure_policy"] = "abort"
                aborting = FailingLFTS(params, random_seed=5)
                expect_error(lambda: aborting.run(diblock_fields()), SaddlePointError)
                assert aborting.calls == 3 and aborting.run_status["last_completed_step"] == 1
                FailingLFTS.fail_calls = tuple(range(1, 20))
                initial = FailingLFTS(diblock_parameters(root / "initial_failure"), random_seed=5)
                rng_before = copy.deepcopy(initial.random_bg.state)
                expect_error(lambda: initial.run(diblock_fields()), SaddlePointError)
                assert initial.run_status["last_completed_step"] == 0
                assert initial.random_bg.state == rng_before
                FailingLFTS.fail_calls = tuple(range(3, 20))
                failing = FailingLFTS(diblock_parameters(root / "failure"), random_seed=5)
                try:
                    failing.run(diblock_fields())
                except SaddlePointError as error:
                    # A numerical failure with the monomer fields of the failed step
                    assert isinstance(error, NumericalError) and not isinstance(error, ValidationError)
                    assert sorted(error.fields) == ["A", "B"], error.fields
                    assert all(np.shape(w) == (16,) and np.all(np.isfinite(w)) for w in error.fields.values())
                else:
                    raise AssertionError("Expected SaddlePointError")
                status = json.loads((root / "failure" / "run_status.json").read_text())
                assert status["status"] == "aborted_saddle_failure" and status["last_completed_step"] == 1
                assert not (root / "failure" / "fields_000002.mat").exists()

                # CL-FTS solves no saddle points: a SaddlePointError of its integrator fails the run like
                # a step that leaves non-finite fields and a non-finite Hamiltonian, whose error holds
                # the complex monomer fields of the step
                failed_clfts_run(root / "CLFTS_saddle", SaddlePointError, FailedSolveStep())
                error = failed_clfts_run(root / "CLFTS_non_finite", NumericalError, NonFiniteStep())
                assert "Integrator fields must be finite" in str(error) and error.fields is None
                with np.errstate(invalid="ignore"):
                    error = failed_clfts_run(root / "CLFTS_hamiltonian", NumericalError, partition=np.nan)
                assert "Non-finite Hamiltonian at step 1" in str(error), str(error)
                assert sorted(error.fields) == ["A", "B"], error.fields
                assert all(np.shape(w) == (16,) and np.iscomplexobj(w) and np.all(np.isfinite(w))
                           for w in error.fields.values())

                # Explicit legacy opt-in and corrupt pending state are checked before RNG mutation.
                checkpoint = root / "LFTSpartial" / "fields_000001.mat"
                data = loadmat(checkpoint, squeeze_me=True)
                data["record_sample_count"] = 2
                atomic_savemat(root / "corrupt.mat", {k: v for k, v in data.items() if not k.startswith("__")})
                candidate = LFTS(diblock_parameters(root / "corrupt"), random_seed=21)
                rng_before = copy.deepcopy(candidate.random_bg.state)
                expect_error(lambda: candidate.continue_run(root / "corrupt.mat"), ValidationError)
                assert candidate.random_bg.state == rng_before
                data = loadmat(checkpoint, squeeze_me=True)
                data.pop("checkpoint_format_version")
                atomic_savemat(root / "legacy.mat", {k: v for k, v in data.items() if not k.startswith("__")})
                expect_error(lambda: candidate.continue_run(root / "legacy.mat"), ValidationError)

                # MATLAB squeezes a one-species field basis to a scalar.
                for cls in (LFTS, CLFTS):
                    params = diblock_parameters(root / (cls.__name__ + "_homopolymer"), 1, cls)
                    params["segment_lengths"] = {"A": 1.0}
                    params["chi_n"] = {}
                    params["polymers"] = [{"volume_fraction": 1.0,
                                                       "blocks": [{"type": "A", "length": 1.0}]}]
                    homogeneous = cls(params, random_seed=5)
                    homogeneous.run({"A": diblock_fields()["A"]})
                    params["langevin"]["max_step"] = 2
                    continued = cls(params, random_seed=99)
                    result = continued.continue_run(Path(params["recording"]["dir"]) / "fields_000001.mat")
                    assert result["last_completed_step"] == 2

                # Options given as None (their defaults) are left out of the saved parameters,
                # which MAT files could not hold
                for cls, extra in ((LFTS, {"zeta_n": None, "wtmd": None}), (CLFTS, {"electrostatics": None})):
                    params = diblock_parameters(root / (cls.__name__ + "_none"), 2, cls)
                    params.update(smearing=None, angles=None, bc=None, **extra)
                    assert cls(params, random_seed=5).run(diblock_fields())["status"] == "completed"
                    for name in ("fields_000002.mat", "structure_function_000002.mat"):
                        saved = loadmat(root / (cls.__name__ + "_none") / name, squeeze_me=True)["initial_params"]
                        assert "nx" in saved.dtype.names and "smearing" not in saved.dtype.names, (cls.__name__, name)

                # Initial fields that are not finite numbers at float64 precision (strings, ragged
                # lists, NaN, values beyond float64 and, in L-FTS, complex numbers), of another size or
                # without a monomer type are refused before the run changes any state
                zeros = np.zeros(16)
                for cls in (LFTS, CLFTS):
                    directory = root / (cls.__name__ + "_fields")
                    sim = cls(diblock_parameters(directory, 1, cls), random_seed=5)
                    rng_before = copy.deepcopy(sim.random_bg.state)
                    refused = [(np.full(16, "0.5"), "finite"), ([[0.0] * 15, [0.0]], "finite"),
                               (np.full(16, np.nan), "finite"), (np.full(16, np.nan * 1j), "finite"),
                               ([10**400] + [0] * 15, "finite"), (zeros[:15], "needs 16 values")]
                    if cls is LFTS:
                        refused.append((zeros + 1j, "real, finite"))
                    for values, message in refused:
                        expect_refused(sim, {"A": values, "B": zeros}, "initial_fields['A']", message)
                    expect_refused(sim, {"A": zeros}, "missing monomer 'B'")
                    # The previous noise of a run continued by hand: real, finite numbers, one per real
                    # (L-FTS) or auxiliary (CL-FTS) field and grid point
                    rows = 1 if cls is LFTS else 2
                    for noise, message in ((np.full((rows, 16), np.nan), "real, finite"),
                                           (np.zeros((rows, 16)) + 1j, "real, finite"),
                                           (np.full((rows, 16), "0.5"), "real, finite"),
                                           ([[0.0] * 16, [0.0]], "real, finite"),
                                           (np.zeros(5), f"needs {rows} x 16 values")):
                        expect_refused(sim, diblock_fields(), "normal_noise_prev", message,
                                       normal_noise_prev=noise, start_langevin_step=2)
                    assert sim.random_bg.state == rng_before and sim.run_status["status"] == "initialized"
                    assert not (directory / "run_status.json").exists(), cls.__name__
                # CL-FTS takes complex initial fields
                clfts = CLFTS(diblock_parameters(root / "CLFTS_complex", 1, CLFTS), random_seed=5)
                assert clfts.run({name: values + 0.5j for name, values in diblock_fields().items()})["status"] \
                    == "completed"

                numerical_failures(root)

            for complex_fields in (False, True):
                for section, key, value in ((None, "ds", np.nan), ("langevin", "dt", np.inf),
                                            ("langevin", "nbar", -1), ("langevin", "max_step", True),
                                            ("recording", "sample_interval", 0)):
                    params = diblock_parameters(root, cls=CLFTS if complex_fields else LFTS)
                    (params if section is None else params[section])[key] = value
                    expect_error(lambda: validate_lfts_params(params, complex_fields=complex_fields), ValidationError)

            target = root / "atomic.mat"
            target.write_bytes(b"previous checkpoint")
            with patch("scipy.io.savemat", side_effect=OSError("injected write failure")):
                expect_error(lambda: atomic_savemat(target, {"x": 1}), OSError)
            assert target.read_bytes() == b"previous checkpoint"
            assert not list(root.glob(".atomic.mat.*"))
        finally:
            os.chdir(previous)
    print("Test passed, FTS reliability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
