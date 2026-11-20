#!/usr/bin/env python3
"""
The OpenMP defaults of the package: one thread and a 1 GB thread stack unless the user sets them.

The OpenMP runtime reads OMP_NUM_THREADS and OMP_STACKSIZE when it loads, so the package sets them
(src/python/__init__.py) before it imports its native module, which loads the runtime. A child process
without any OMP_ variable imports polymerfts: every OpenMP runtime in the process must then run one
thread. Imported before the defaults are set, the runtime takes all CPUs. A value the user sets (3
threads and a 64 MB stack in a second child) is kept. The stack size is also checked as libgomp took it
when it loaded (OMP_DISPLAY_ENV): a default set after the runtime loaded would be in the environment
while libgomp keeps its own. Other runtimes print another format; their stack size is not checked.
Linux only: the runtimes are found in /proc/self/maps.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

CHILD = r'''
import ctypes, json, os
import polymerfts
paths = set()
with open("/proc/self/maps") as maps:
    for line in maps:
        fields = line.split()
        if len(fields) == 6 and any(name in os.path.basename(fields[5]) for name in ("libgomp", "libomp", "libiomp5")):
            paths.add(fields[5])
threads = {path: ctypes.CDLL(path).omp_get_max_threads() for path in sorted(paths)}
print(json.dumps({"threads": threads, "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE")}))
'''


def child(extra):
    """Import polymerfts in a child process without OpenMP variables, plus extra; returns its report."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("OMP_", "GOMP_", "KMP_"))}
    env.update(extra)
    # libgomp prints the stack size it took when it loaded
    env["OMP_DISPLAY_ENV"] = "TRUE"
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([sys.executable, "-c", CHILD], cwd=tmp, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"the child exited with {result.returncode}:\n{result.stderr}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    report["libgomp stack"] = re.findall(r"OMP_STACKSIZE = '(\d+)'", result.stderr)
    return report


def main():
    if not sys.platform.startswith("linux"):
        print("SKIP: the OpenMP runtimes are found in /proc/self/maps (Linux)")
        return 77
    failures = []
    for extra, threads, stack, size in (({}, 1, "1G", 2 ** 30),
                                        ({"OMP_NUM_THREADS": "3", "OMP_STACKSIZE": "64M"}, 3, "64M", 2 ** 26)):
        report = child(extra)
        print(f"OpenMP variables {extra}: {report}")
        if not report["threads"]:
            failures.append("no OpenMP runtime found in the process")
        if any(value != threads for value in report["threads"].values()) or report["OMP_STACKSIZE"] != stack:
            failures.append(f"with {extra} the runtimes run {report['threads']} threads and OMP_STACKSIZE is "
                            f"{report['OMP_STACKSIZE']} (expected {threads} and {stack})")
        if any("libgomp" in os.path.basename(path) for path in report["threads"]) and \
                report["libgomp stack"] != [str(size)]:
            failures.append(f"with {extra} libgomp took the stack size {report['libgomp stack']} (expected {size})")
    for failure in failures:
        print("FAIL:", failure)
    if failures:
        return 1
    print("OpenMP defaults of the package (one thread, unless the user sets it): PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
