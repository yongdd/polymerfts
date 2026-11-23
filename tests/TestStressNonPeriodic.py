#!/usr/bin/env python3
"""
Stress with non-periodic (reflecting/absorbing) boundary conditions, for discrete chains and
the continuous methods RQM4 and RK2 (which has its own stress code on the GPU).

1. Finite-difference validation on the first CPU platform and on CUDA: stress[d] must match
   the central difference of -lnQ with respect to L_d.
   - DISCRETE chains: the stress formula is an exact derivative of the discretized lnQ, so
     agreement is limited only by the FD truncation (tolerance 1e-6).
   - RQM4: the stress is a contour-quadrature approximation: at ds = 0.01 the reflecting
     mismatch is ~2e-4, like the periodic baseline (tolerance 1e-3); with absorbing walls it
     is ~3e-7 in 1D (tolerance 1e-5) and ~1e-5 for the 3D mixed walls (tolerance 1e-4).
     Until 2026-10 the full-storage stress dropped the chain-end segments, whose q(r, 0) = 1
     is not stress-free at an absorbing wall; that gave absorbing walls an O(ds) error of
     ~5e-3, which these tolerances catch. Do NOT loosen them.
   - RK2: the stress differs from the finite difference of its own Q by the O(ds^2) splitting
     error: 1.7e-2 at ds = 0.01 for the reflecting case, 4e-3 at 0.005, 1e-3 at 0.0025; the
     same holds for periodic boundaries.

2. CPU-vs-CUDA equivalence for MIXED reflecting/absorbing boundaries per axis: Q to 1e-12 and
   the stress to 1e-10, relative. This guards the per-dimension BC plumbing (a squeeze bug here
   once built the CUDA Boltzmann/stress tables with the wrong axis assignment - wrong Q by
   O(1) - while all pure-BC cases passed). The CPU side of RK2 shares its stress code with
   RQM4, which the finite differences validate.

The discrete and RQM4 runs compute two propagators at once (OMP_NUM_THREADS 2), the RK2 runs
one (OMP_NUM_THREADS 1); a computation reads the variable when it is created. The number of
checks is printed and checked.
"""

import os
import sys

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing

DS = 0.01
F_A = 0.36
BOND = {"A": 1.0, "B": 1.1}

REFLECT_1D = ([48], [2.3], ["reflecting"] * 2)
ABSORB_1D = ([48], [2.3], ["absorbing"] * 2)
R_A_2D = ([24, 20], [2.1, 1.7], ["reflecting", "reflecting", "absorbing", "absorbing"])
A_R_2D = ([24, 20], [2.1, 1.7], ["absorbing", "absorbing", "reflecting", "reflecting"])
R_A_R_3D = ([24, 20, 16], [2.1, 1.7, 1.4], ["reflecting", "reflecting", "absorbing", "absorbing", "reflecting", "reflecting"])
A_R_R_3D = ([24, 20, 16], [2.1, 1.7, 1.4], ["absorbing", "absorbing", "reflecting", "reflecting", "reflecting", "reflecting"])
R_R_A_3D = ([24, 20, 16], [2.1, 1.7, 1.4], ["reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"])

# (chain model, numerical method, OMP_NUM_THREADS of its runs,
#  finite-difference cases (name, box, tolerance), cross-platform cases (name, box))
METHODS = [
    ("discrete", None, "2",
     [("1D reflect", REFLECT_1D, 1e-6), ("1D absorb", ABSORB_1D, 1e-6), ("3D mixed r/a/r", R_A_R_3D, 1e-6)],
     [("mixed r/a/r", R_A_R_3D), ("mixed a/r/r", A_R_R_3D), ("mixed r/r/a", R_R_A_3D)]),
    ("continuous", "rqm4", "2",
     [("1D reflect", REFLECT_1D, 1e-3), ("1D absorb", ABSORB_1D, 1e-5), ("3D mixed r/a/r", R_A_R_3D, 1e-4)],
     [("mixed r/a/r", R_A_R_3D), ("mixed a/r/r", A_R_R_3D), ("mixed r/r/a", R_R_A_3D)]),
    ("continuous", "rk2", "1",
     [("1D reflect", REFLECT_1D, 2.5e-2), ("1D absorb", ABSORB_1D, 1.5e-2), ("2D reflect/absorb", R_A_2D, 1.5e-2)],
     [("1D reflect", REFLECT_1D), ("1D absorb", ABSORB_1D), ("2D r/a", R_A_2D), ("2D a/r", A_R_2D),
      ("3D r/a/r", R_A_R_3D), ("3D a/r/r", A_R_R_3D)]),
]

n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[PASS] {msg}")
    else:
        n_fail += 1
        print(f"[FAIL] {msg}")


def smooth_fields(nx, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(2, *nx))
    for c in range(2):
        w[c] = gaussian_filter(w[c], sigma=2.0, mode="nearest")
        w[c] *= 5.0 / max(w[c].std(), 1e-12)
    return w.reshape(2, -1)


def make_solver(nx, lx, bc, model, method, platform):
    solver = PropagatorSolver(
        nx=list(nx), lx=list(lx), ds=DS,
        bond_lengths=dict(BOND), bc=list(bc),
        chain_model=model, numerical_method=method,
        platform=platform)
    solver.add_polymer(1.0, [["A", F_A, 0, 1], ["B", 1.0 - F_A, 1, 2]])
    solver._initialize_solver()
    return solver


def run_fd_case(name, nx, lx, bc, model, method, platform, tol):
    w = smooth_fields(nx, 1234 + len(nx))
    wd = {"A": w[0], "B": w[1]}
    solver = make_solver(nx, lx, bc, model, method, platform)
    solver.compute_propagators(wd)
    solver.compute_stress()
    stress = np.array(solver.get_stress()).real

    def neg_lnq(lx_use):
        s2 = make_solver(nx, lx_use, bc, model, method, platform)
        s2.compute_propagators(wd)
        return -np.log(s2.get_partition_function(0))

    worst = 0.0
    for d in range(len(nx)):
        dl = lx[d] * 2e-4
        lx_p = list(lx); lx_p[d] += 0.5 * dl
        lx_m = list(lx); lx_m[d] -= 0.5 * dl
        fd = (neg_lnq(lx_p) - neg_lnq(lx_m)) / dl
        rel = abs(stress[d] - fd) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
    check(worst < tol,
          f"{name} [{method or model}] ({platform}): max FD mismatch {worst:.3e} < {tol}")


def run_cross_platform_case(name, nx, lx, bc, model, method, cpu):
    w = smooth_fields(nx, 777)
    wd = {"A": w[0], "B": w[1]}
    results = {}
    for platform in [cpu, "cuda"]:
        s = make_solver(nx, lx, bc, model, method, platform)
        s.compute_propagators(wd)
        s.compute_stress()
        results[platform] = (s.get_partition_function(0),
                             np.array(s.get_stress()).real[:len(nx)])
    q_cpu, st_cpu = results[cpu]
    q_gpu, st_gpu = results["cuda"]
    rel_q = abs(q_gpu - q_cpu) / abs(q_cpu)
    rel_s = np.max(np.abs(st_gpu - st_cpu)
                   / np.maximum(np.abs(st_cpu), 1e-12))
    check(rel_q < 1e-12,
          f"{name} [{method or model}]: Q cpu-vs-cuda {rel_q:.3e} < 1e-12")
    check(rel_s < 1e-10,
          f"{name} [{method or model}]: stress cpu-vs-cuda {rel_s:.3e} < 1e-10")


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpu = next((p for p in platforms if p.startswith("cpu")), None)
    has_cuda = "cuda" in platforms
    if cpu is None and not has_cuda:
        print("SKIP: no computational platform available in this build.")
        sys.exit(77)

    test_platforms = ([cpu] if cpu else []) + (["cuda"] if has_cuda else [])
    expected = 0
    for model, method, threads, fd_cases, cross_cases in METHODS:
        os.environ["OMP_NUM_THREADS"] = threads
        # --- FD validation ---
        for platform in test_platforms:
            for name, (nx, lx, bc), tol in fd_cases:
                run_fd_case(name, nx, lx, bc, model, method, platform, tol)
        expected += len(test_platforms) * len(fd_cases)
        # --- Cross-platform equivalence for mixed BCs ---
        if has_cuda and cpu is not None:
            for name, (nx, lx, bc) in cross_cases:
                run_cross_platform_case(name, nx, lx, bc, model, method, cpu)
            expected += 2 * len(cross_cases)

    print(f"\n{n_pass} passed, {n_fail} failed, {n_pass + n_fail} checks (expected {expected})")
    sys.exit(1 if cuda_part_missing() or n_fail or n_pass + n_fail != expected else 0)


if __name__ == "__main__":
    main()
