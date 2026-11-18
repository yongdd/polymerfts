#!/usr/bin/env python3
"""Parameter files: load_config, ConfigError and SCFT/LFTS/CLFTS.from_file.

Writes small YAML and JSON files and checks that numpy expressions are
evaluated, that nothing outside the allowed expressions is executed, that bad
files raise ConfigError, that a UTF-8 file with Greek letters loads under an ASCII
locale, and that the three simulation classes build from a
file with keyword overrides applied: one file serves the three, each warning once,
at the line that calls from_file, about the options of the other methods it
ignores; a key given as None is absent, also as an override, and a key no method
reads is refused.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

from polymerfts import SCFT, LFTS, CLFTS, PlatformSelector, load_config, ConfigError, ValidationError


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


YAML = """
nx: [8, 8, 8]
lx: ["4.0*np.sqrt(3)/2", "np.pi", 4.0]
chain_model: continuous
ds: 0.1
segment_lengths: {A: 1.0, B: 1.0}
chi_n: {"A,B": 15.0}
polymers:
  - volume_fraction: 1.0
    blocks: [{type: A, length: 0.5}, {type: B, length: 0.5}]
platform: PLATFORM
relax_cell: false
optimizer: {name: am, max_hist: 20, start_error: 0.01, mix_min: 0.1, mix_init: 0.1}
langevin: {nbar: 10000, dt: 8.0, max_step: 10}
compressor: {name: am, max_hist: 20, start_error: 0.1, mix_min: 0.1, mix_init: 0.1}
saddle: {max_iter: 100, tolerance: 1.0e-4}
recording: {dir: data, checkpoint_interval: 10, sample_interval: 1, statistics_save_interval: 10}
verbose_level: 1
note: "np.sqrt(4) + np.save('INJECTED', 0)"
complex: "(-8)**0.5*np.pi"
"""


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    failed = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            with open("p.yaml", "w") as f:
                f.write(YAML.replace("PLATFORM", cpu_platform()))

            # Numpy expressions are evaluated; anything else is kept as text and not executed
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p = load_config("p.yaml")
            if not np.allclose(p["lx"], [2.0*np.sqrt(3.0), np.pi, 4.0], rtol=0, atol=1e-15):
                failed.append(f"lx expressions evaluated to {p['lx']}")
            # (the former eval with numpy in scope ran np.save and wrote INJECTED.npy)
            if os.path.exists("INJECTED.npy") or not isinstance(p["note"], str) or len(caught) < 2:
                failed.append("an expression outside the allowed set was executed or not reported")
            if not isinstance(p["complex"], str):
                failed.append(f"a complex value was accepted: {p['complex']!r}")

            # JSON gives the same dictionary
            with open("p.json", "w") as f:
                json.dump({k: v for k, v in p.items() if k not in ("note", "complex")}, f)
            q = load_config("p.json")
            if q["chi_n"] != p["chi_n"] or q["lx"] != list(p["lx"]):
                failed.append("JSON and YAML differ")

            # Bad files
            for name, text in (("p.txt", "nx: [8]"), ("empty.yaml", ""), ("list.yaml", "- 1\n- 2\n")):
                with open(name, "w") as f:
                    f.write(text)
                try:
                    load_config(name)
                    failed.append(f"{name} accepted")
                except ConfigError:
                    pass
            try:
                load_config("missing.yaml")
                failed.append("missing file accepted")
            except FileNotFoundError:
                pass

            # YAML and JSON files are UTF-8: a comment with Greek letters loads also where the locale
            # encoding is another one (ASCII here; cp949 or cp1252 on Windows raised UnicodeDecodeError)
            with open("utf8.yaml", "w", encoding="utf-8") as f:
                f.write("# ρ₀ is the reference density\nnx: [8]   # χN\n")
            env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C")
            loaded = subprocess.run([sys.executable, "-c", "from polymerfts import load_config; "
                                     "print(load_config('utf8.yaml')['nx'])"],
                                    env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if loaded.returncode != 0 or loaded.stdout.strip().splitlines()[-1:] != ["[8]"]:
                failed.append(f"a UTF-8 file under an ASCII locale: {loaded.stdout} {loaded.stderr[-300:]}")

            # The simulation classes build from one file, with overrides applied, each with one warning
            # about the options of the other methods that it ignores, at the line that calls from_file; a
            # key given as None (~) in the file is absent, and an override given as None removes the key
            # of the file; a key that no method reads is refused
            with open("shared.yaml", "w") as f:
                f.writelines(line for line in YAML.replace("PLATFORM", cpu_platform()).splitlines(keepends=True)
                             if not line.startswith(("note:", "complex:")))
                f.write("dynamic_stabilization: ~\nzeta_n: ~\n")
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scft = SCFT.from_file("shared.yaml", max_iter=7)
                lfts = LFTS.from_file("shared.yaml", random_seed=1, ds=0.05)
                clfts = CLFTS.from_file("shared.yaml", random_seed=1)
                ignored = [w for w in caught if "ignores the options of other methods" in str(w.message)]
                for method, key in (("SCFT", "langevin (L-FTS and CL-FTS)"), ("L-FTS", "relax_cell (SCFT)"),
                                    ("CL-FTS", "compressor (L-FTS)")):
                    texts = [str(w.message) for w in ignored if str(w.message).startswith(method + " ")]
                    if len(texts) != 1 or key not in texts[0]:
                        failed.append(f"{method} gave not one warning that it ignores {key}: {texts}")
                for w in ignored:
                    if os.path.realpath(w.filename) != os.path.realpath(__file__):
                        failed.append(f"the warning '{w.message}' points at {w.filename}:{w.lineno}")
                try:
                    LFTS.from_file("p.yaml")
                    failed.append("LFTS.from_file accepted the key 'note'")
                except ValidationError as error:
                    if "L-FTS does not take a 'note' parameter" not in str(error):
                        failed.append(f"LFTS.from_file refused p.yaml with {error}")
                # The file gives relax_cell: false; the override None removes it, and SCFT requires it
                try:
                    SCFT.from_file("shared.yaml", relax_cell=None)
                    failed.append("SCFT.from_file kept the relax_cell of the file under an override None")
                except ValidationError as error:
                    if "relax_cell" not in str(error):
                        failed.append(f"SCFT.from_file with relax_cell=None refused with {error}")
            if clfts.alpha_ds != 0.0 or clfts.zeta_n != 100.0 or lfts.zeta_n is not None:
                failed.append("an option given as None in the file is not absent")
            if scft.max_iter != 7:
                failed.append("SCFT.from_file override")
            if abs(lfts.ds - 0.05) > 1e-15:
                failed.append("LFTS.from_file override")
            if not np.allclose(clfts.prop_solver.get_lx(), p["lx"]):
                failed.append("CLFTS.from_file box")
        finally:
            os.chdir(cwd)

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("Parameter files: OK")


if __name__ == "__main__":
    main()
