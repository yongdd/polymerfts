#!/usr/bin/env python3
"""Every tutorial notebook runs with the current code.

The code cells of each notebook under tutorials/ run in order in their own process,
in the notebook's directory of a temporary copy of the tutorials tree (a notebook
finds its input files there and writes nothing into the source tree), with the
non-interactive Agg backend of matplotlib. Warnings of the UserWarning category are
errors, except the one of show() under Agg, where a notebook would draw its figure
inline. As in TestExamples, the LFTS and CLFTS classes are wrapped: a run stops
after at most MAX_STEP Langevin steps, and its checkpoint interval is cut to that
number of steps, so the cells after it find a fields file. A notebook that runs its
own Langevin loop is shortened by the text replacements in SHORTENED, which must
each match once. An animation calls its frame function for every frame, but draws
no frame: encoding the frames would take half of the run time and tests nothing of
polymerfts.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cuda_part import cuda_part_missing

ROOT = Path(__file__).resolve().parents[1]
MAX_STEP = 5

# NaiveLFTS writes out its own Langevin loop (200 steps, about two minutes): 5 of them
# reach every line of the loop and the cells after it
SHORTENED = {"tutorials/LangevinFieldTheoreticSimulation/NaiveLFTS.ipynb": [("nsteps = 200 ", "nsteps = 5 ")]}

DRIVER = r"""
import copy, json, numbers, sys, types, warnings
warnings.simplefilter("error", UserWarning)
warnings.filterwarnings("ignore", message=".*non-interactive", category=UserWarning)
import matplotlib.animation
import polymerfts

max_step = int(sys.argv[2])


def wrap(cls):
    init = cls.__init__

    def __init__(self, params, *args, **kwargs):
        params = copy.deepcopy(params)
        if isinstance(params.get("langevin"), dict):
            params["langevin"]["max_step"] = min(params["langevin"].get("max_step", max_step), max_step)
            if isinstance(params.get("recording"), dict) and params["langevin"]["max_step"] > 0:
                params["recording"]["checkpoint_interval"] = min(
                    params["recording"].get("checkpoint_interval", max_step), params["langevin"]["max_step"])
        init(self, params, *args, **kwargs)

    cls.__init__ = __init__


for cls in (polymerfts.LFTS, polymerfts.CLFTS):
    wrap(cls)


def frames_only(fig, func, frames=None, **kwargs):
    # In place of FuncAnimation: the frame function runs for every frame, nothing is drawn
    for frame in (range(frames) if isinstance(frames, numbers.Integral) else frames):
        func(frame)
    return types.SimpleNamespace(to_jshtml=lambda *args, **kwargs: "")


matplotlib.animation.FuncAnimation = frames_only

namespace = {"__name__": "__main__"}
for number, code in enumerate(json.load(sys.stdin), 1):
    exec(compile(code, f"{sys.argv[1]}, code cell {number}", "exec"), namespace)
print("TUTORIAL FINISHED", flush=True)
"""


def code_cells(notebook, name):
    """The code of each code cell, with the replacements of SHORTENED; None if one does not match once."""
    cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
             if cell["cell_type"] == "code"]
    for old, new in SHORTENED.get(name, []):
        if sum(code.count(old) for code in cells) != 1:
            return None
        cells = [code.replace(old, new) for code in cells]
    return cells


def main():
    notebooks = sorted(p for p in (ROOT / "tutorials").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    if not notebooks:
        print("No tutorials found")
        return 1
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tutorials"
        # Outputs of runs in the checkout (ignored by git) stay behind: a notebook reads only its own
        shutil.copytree(ROOT / "tutorials", tree, ignore=shutil.ignore_patterns(
            ".ipynb_checkpoints", "data_simulation*", "fields_*", "fields.*", "*.png"))
        for notebook in notebooks:
            name = notebook.relative_to(ROOT).as_posix()
            cells = code_cells(notebook, name)
            if cells is None:
                print(f"[FAIL] {name}\n    a replacement of SHORTENED does not match once: update it")
                failures.append(name)
                continue
            copy = tree / notebook.relative_to(ROOT / "tutorials")
            # Each tutorial sets OMP_NUM_THREADS=1 before it imports polymerfts, which the driver imports first
            child = subprocess.run([sys.executable, "-c", DRIVER, name, str(MAX_STEP)],
                                   input=json.dumps(cells), cwd=copy.parent, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=600,
                                   env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1"))
            ok = child.returncode == 0 and "TUTORIAL FINISHED" in child.stdout
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                tail = (child.stdout + child.stderr).strip().splitlines()[-8:]
                print("    " + "\n    ".join(tail))
                failures.append(name)
    print(f"\n{len(notebooks) - len(failures)} passed, {len(failures)} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
