#!/usr/bin/env python3
"""Every example builds its simulation with the current code and input checks.

Each script under examples/ runs in its own process, in its directory of a
temporary copy of the examples tree (it finds its input files there, also in a
neighbouring directory, and writes nothing into the source tree). Warnings of the
UserWarning category, such as the one about the keys of another method, are errors.
The SCFT, LFTS and CLFTS classes are wrapped: the first run() or continue_run() stops
the script, so the example is checked up to the start of its simulation (inputs,
parameter validation, construction) on the platform it selects. With a visible CUDA
device the first run() instead runs briefly (SCFT max_iter 6, L-FTS and CL-FTS
max_step 5) and the script continues, so that the code after it, such as a second
construction, is checked too; the second run() or a continue_run() stops the script
then. On the CPU platforms these runs would more than double the time of the test, so
a CPU-only build checks the construction only. The examples in FIRST_RUN_STOPS need a
complete first run for what follows it and always stop at their first run(). An
example that iterates a PropagatorSolver itself stops the same way at its first
compute_propagators() outside a simulation. The examples of tools that build no
simulation (WITHOUT_SIMULATION) run to the end instead and check their own results.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cuda_part import cuda_part_missing

ROOT = Path(__file__).resolve().parents[1]

# Examples that build no SCFT/LFTS/CLFTS simulation, and modules of examples; they
# must finish without an error
WITHOUT_SIMULATION = {"examples/tools/FourierTransform.py",
                      "examples/block_copolymer/scft/phase_diagram/phase_diagram.py",
                      "examples/block_copolymer/scft/seeds.py"}

# Examples whose code after the first run() needs a complete run: a converged
# solution (AngleScan2D), or the records of the full number of steps (GibbsEnsemble
# reads them between its runs)
FIRST_RUN_STOPS = {"examples/block_copolymer/scft/crystal_systems/AngleScan2D.py",
                   "examples/blend/clfts/GibbsEnsemble.py",
                   "examples/blend/clfts/HomopolymerSolutionBook.py",
                   "examples/charged/clfts/CoacervateGibbs.py",
                   "examples/charged/clfts/ComplexCoacervate.py",
                   "examples/charged/clfts/CounterionsBetweenPlates.py",
                   "examples/charged/clfts/SaltReservoir.py",
                   "examples/charged/clfts/SmearedSaltSelfEnergy.py"}

DRIVER = r"""
import copy, os, runpy, sys, warnings
warnings.simplefilter("error", UserWarning)
import polymerfts

# The process output, kept for the messages below: an example may redirect file
# descriptor 1 around run(), e.g. to silence the native library
output = os.dup(1)
short = sys.argv[2] == "short"

runs = {"count": 0, "inside": 0}


def stop(name):
    # Straight to the process output: an example may redirect sys.stdout around run()
    os.write(output, ("EXAMPLE STOPPED AT " + name + "\n").encode())
    os._exit(0)


def inside(method):
    # A simulation computes propagators while it is built and while it runs
    def call(*args, **kwargs):
        runs["inside"] += 1
        try:
            return method(*args, **kwargs)
        finally:
            runs["inside"] -= 1
    return call


def wrap(cls):
    init, run = cls.__init__, cls.run

    @inside
    def __init__(self, params, *args, **kwargs):
        params = copy.deepcopy(params)
        if short and cls is polymerfts.SCFT:
            params["max_iter"] = 6
        elif short and isinstance(params.get("langevin"), dict):
            params["langevin"]["max_step"] = min(params["langevin"].get("max_step", 5), 5)
        init(self, params, *args, **kwargs)

    def brief_run(self, *args, **kwargs):
        if not short or runs["count"] > 0:
            stop(cls.__name__ + ".run")
        runs["count"] += 1
        result = inside(run)(self, *args, **kwargs)
        os.write(output, ("EXAMPLE RAN " + cls.__name__ + ".run\n").encode())
        return result

    def continue_run(self, *args, **kwargs):
        stop(cls.__name__ + ".continue_run")

    cls.__init__ = __init__
    cls.run = brief_run
    cls.continue_run = continue_run


for cls in (polymerfts.SCFT, polymerfts.LFTS, polymerfts.CLFTS):
    wrap(cls)

compute_propagators = polymerfts.PropagatorSolver.compute_propagators


def stop_chains(self, *args, **kwargs):
    if not runs["inside"]:
        stop("PropagatorSolver.compute_propagators")
    return compute_propagators(self, *args, **kwargs)


polymerfts.PropagatorSolver.compute_propagators = stop_chains
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
print("EXAMPLE FINISHED", flush=True)
"""


def main():
    scripts = sorted(p for p in (ROOT / "examples").rglob("*.py") if "__pycache__" not in p.parts)
    if not scripts:
        print("No examples found")
        return 1
    from polymerfts import _core
    brief_runs = "cuda" in _core.PlatformSelector.avail_platforms()
    print("The first run() of an example runs briefly (CUDA)" if brief_runs else
          "The examples are checked up to the start of their simulation (no CUDA device)")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "examples"
        shutil.copytree(ROOT / "examples", tree, ignore=shutil.ignore_patterns("__pycache__", "*.png"))
        for script in scripts:
            copy = tree / script.relative_to(ROOT / "examples")
            name = script.relative_to(ROOT).as_posix()
            mode = "short" if brief_runs and name not in FIRST_RUN_STOPS else "construct"
            child = subprocess.run([sys.executable, "-c", DRIVER, str(copy), mode], cwd=copy.parent,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=600, env=dict(os.environ, MPLBACKEND="Agg"))
            simulated = "EXAMPLE RAN" in child.stdout or "EXAMPLE STOPPED" in child.stdout
            ended = "EXAMPLE STOPPED" in child.stdout or "EXAMPLE FINISHED" in child.stdout
            ok = child.returncode == 0 and ended and simulated != (name in WITHOUT_SIMULATION)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                tail = (child.stdout + child.stderr).strip().splitlines()[-6:]
                print("    " + "\n    ".join(tail))
                failures.append(name)
    print(f"\n{len(scripts) - len(failures)} passed, {len(failures)} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
