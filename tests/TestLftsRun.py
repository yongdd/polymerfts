#!/usr/bin/env python3
"""Short L-FTS and CL-FTS runs on the CPU (small 2D grid).

Checks that the run loops work for every compressor and verbose level, that the
recorded files are written, that a run continued from a checkpoint reproduces
the uninterrupted run exactly, that the CL-FTS forces carry the dynamical
stabilization, and that invalid options are rejected.
"""

import os
import sys
import atexit
import shutil
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, ValidationError, PlatformSelector, LeimkuhlerMatthews
from fts_checks import exact_restart, quiet


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return cpu[0] if cpu else None


def params_for(work_dir, max_step, compressor="am", verbose_level=1):
    return {
        "nx": [16, 16],
        "lx": [3.0, 3.0],
        "chain_model": "continuous",
        "ds": 1/20,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
        "platform": cpu_platform(),
        "langevin": {"nbar": 10000, "dt": 8.0, "max_step": max_step},
        "recording": {"dir": work_dir, "checkpoint_interval": 2, "sample_interval": 1, "statistics_save_interval": 2},
        "saddle": {"max_iter": 200, "tolerance": 1e-4},
        "compressor": {"name": compressor, "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01},
        "verbose_level": verbose_level,
    }


def initial_fields(seed=3):
    rng = np.random.default_rng(seed)
    return {"A": rng.normal(0.0, 0.5, 256), "B": rng.normal(0.0, 0.5, 256)}


class RecordingSteps(LeimkuhlerMatthews):
    """Leimkuhler-Matthews steps that keep the fields and the forces each step starts from."""
    name = "test-recording-lm"

    def __init__(self):
        super().__init__()
        self.steps = []

    def step(self, fields, force, **kwargs):
        self.steps.append((fields.copy(), force.copy()))
        return super().step(fields, force, **kwargs)


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        return 77  # ctest SKIP_RETURN_CODE
    ok = True
    root = tempfile.mkdtemp(prefix="polymerfts_lfts_")
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    # ---- every compressor and verbose level completes and records its files
    for compressor in ("am", "lr", "lram"):
        for verbose_level in (0, 1, 2):
            work_dir = os.path.join(root, f"{compressor}_{verbose_level}")
            sim = quiet(LFTS, params_for(work_dir, 4, compressor, verbose_level), random_seed=5)
            quiet(sim.run, initial_fields=initial_fields())
            for name in ("fields_000002.mat", "fields_000004.mat", "dH_000002.mat",
                         "dH_000004.mat", "structure_function_000002.mat"):
                if not os.path.exists(os.path.join(work_dir, name)):
                    print(f"FAILED: {compressor} verbose {verbose_level}: {name} not written")
                    ok = False
            data = loadmat(os.path.join(work_dir, "dH_000004.mat"), squeeze_me=True)
            if not np.all(np.isfinite(data["H_history"])):
                print(f"FAILED: {compressor} verbose {verbose_level}: Hamiltonian not finite")
                ok = False

    # ---- continuing from a checkpoint reproduces the uninterrupted run, in the directory
    # where the shorter run stopped
    exact_restart(lambda name, steps, seed: quiet(LFTS, params_for(os.path.join(root, name), steps), random_seed=seed),
                  lambda sim: sim.run(initial_fields=initial_fields()), 4, 2, {"fields_000004.mat": ("w_",)},
                  same_directory=True)

    # ---- CL-FTS run (complex fields, compressible)
    work_dir = os.path.join(root, "clfts")
    params = params_for(work_dir, 4)
    del params["compressor"], params["saddle"]
    params["dynamic_stabilization"] = 0.05
    sim = quiet(CLFTS, params, random_seed=5)
    quiet(sim.run, initial_fields=initial_fields())
    data = loadmat(os.path.join(work_dir, "dH_000004.mat"), squeeze_me=True)
    if not (os.path.exists(os.path.join(work_dir, "fields_000004.mat"))
            and np.all(np.isfinite(data["H_history_real"]))):
        print("FAILED: CL-FTS run")
        ok = False

    # ---- the forces of every CL-FTS step carry the dynamical stabilization i alpha_ds Im[w] of the
    # real-type fields (fields with imaginary parts from the start)
    work_dir = os.path.join(root, "clfts_stabilization")
    params = params_for(work_dir, 2)
    del params["compressor"], params["saddle"]
    params["dynamic_stabilization"] = 0.05
    integrator = RecordingSteps()
    sim = quiet(CLFTS, params, random_seed=5, integrator=integrator)
    rng = np.random.default_rng(7)
    quiet(sim.run, initial_fields={name: values + 1j * rng.normal(0.0, 0.5, 256)
                                   for name, values in initial_fields().items()})
    for step, (fields, force) in enumerate(integrator.steps, start=1):
        phi, _ = quiet(sim.compute_concentrations, fields)
        expected = sim.mpt.compute_func_deriv(fields, sim.smearing.apply_to_dict(phi), range(len(sim.monomer_types)))
        stabilization = np.zeros_like(expected)
        for i in sim.mpt.aux_fields_real_idx:
            stabilization[i] = 1j * 0.05 * np.imag(fields[i])
        expected += stabilization
        difference = np.max(np.abs(force - expected))
        if not (len(integrator.steps) == 2 and difference <= 1e-10 * np.max(np.abs(expected))
                and np.max(np.abs(stabilization)) > 1e-3):
            print(f"FAILED: CL-FTS forces of step {step} with dynamical stabilization differ by {difference:.2e} "
                  f"(stabilization {np.max(np.abs(stabilization)):.2e})")
            ok = False

    # ---- invalid options
    for bad, what in ((("compressor", "name", "foo"), "unknown compressor"),
                      (("verbose_level", None, 3), "verbose_level 3")):
        params = params_for(os.path.join(root, "bad"), 4)
        key, sub, value = bad
        if sub is None:
            params[key] = value
        else:
            params[key][sub] = value
        try:
            quiet(LFTS, params, random_seed=1)
            print(f"FAILED: {what} accepted")
            ok = False
        except ValidationError:
            pass
    params = params_for(os.path.join(root, "bad"), 4)
    params["wtmd"] = {"ell": 4}
    try:
        quiet(LFTS, params, random_seed=1)
        print("FAILED: WTMD accepted on a 2D grid")
        ok = False
    except ValidationError:
        pass

    if ok:
        print("Test passed, L-FTS and CL-FTS runs")
        return 0
    print("Test failed, L-FTS and CL-FTS runs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
