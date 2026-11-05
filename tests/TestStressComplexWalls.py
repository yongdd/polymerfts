#!/usr/bin/env python3
"""
Stress of complex fields with reflecting and absorbing walls.

The pseudo-spectral methods transform the real and imaginary parts of the propagators with the
cosine and sine transforms and multiply the coefficients as complex numbers; rs sums on the grid.
Fields: AB diblock, random real parts of standard deviation 5 and imaginary parts of 1.

1. A reflecting box of length L is the periodic box of length 2L with the field mirrored (the
   even extension of the DCT-II), so Q is the same and -d ln Q / dL is twice the periodic value,
   which takes the Fourier path of complex fields. Real fields check the relation itself. 1D and
   2D, continuous chains (rqm4, rk2) and discrete chains (ps, rs): relative 1e-11.
2. Absorbing walls, which have no such periodic image: the stress of discrete chains is the exact
   derivative of their ln Q and matches the central difference of -ln Q (relative 1e-7), and ps
   equals rs where the grid resolves the half bond (relative 1e-10, as for real fields in
   TestRealSpaceDiscrete). The stress of continuous chains carries the error of their contour
   quadrature, so they are checked through the complex step instead: the stress is holomorphic
   in the fields, so for the fields w + i eps v its real part is the stress at w and its
   imaginary part over eps the derivative along v, the central difference of the stress of real
   fields (rqm4 and rk2 on the CPU, relative 1e-8 at eps = 1e-4).
3. CUDA equals the CPU backends for ps and rs (relative 1e-11).

Exit code 1 in a CUDA build without a visible CUDA device, after the CPU parts.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing

FAILURES = []
DIBLOCK = [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]]
BOND = {"A": 1.0, "B": 1.2}
CHAIN_MODEL = {"rqm4": "continuous", "rk2": "continuous", "ps": "discrete", "rs": "discrete"}


def check(condition, message):
    print(f"    [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def fields(rng, nx, field_type):
    w = {}
    for m in "AB":
        parts = [rng.normal(0.0, 1.0, nx), rng.normal(0.0, 1.0, nx)]
        parts = [p / p.std() for p in parts]
        w[m] = 5.0 * parts[0] + (1j * parts[1] if field_type == "complex" else 0.0)
    return w


def solve(method, platform, nx, lx, bc, w, ds=0.05):
    """Q and the stress -d ln Q / dL of each axis."""
    field_type = "complex" if np.iscomplexobj(w["A"]) else "real"
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(nx=list(nx), lx=list(lx), ds=ds, bond_lengths=BOND, bc=list(bc),
                                  chain_model=CHAIN_MODEL[method], numerical_method=method, platform=platform,
                                  field_type=field_type)
        solver.add_polymer(1.0, DIBLOCK)
        solver.compute_propagators({m: w[m].flatten().copy() for m in w})
        solver.compute_stress()
    return solver.get_partition_function(0), np.asarray(solver.get_stress())[:len(nx)]


def mirrored(w):
    """The even extension of a field on a box with walls to the periodic box of twice the size."""
    for axis in range(w.ndim):
        w = np.concatenate([w, np.flip(w, axis=axis)], axis=axis)
    return w


def periodic_image(platforms, rng):
    print("  reflecting box of L against the periodic box of 2L with the field mirrored")
    for nx, lx in (([40], [2.0]), ([24, 20], [1.5, 1.25])):
        for field_type in ("real", "complex"):
            w = fields(rng, nx, field_type)
            w2 = {m: mirrored(w[m]) for m in w}
            for method in ("rqm4", "rk2", "ps", "rs"):
                for platform in platforms:
                    if platform == "cuda" and field_type == "complex" and CHAIN_MODEL[method] == "continuous":
                        continue   # the CUDA continuous-chain solvers take no complex fields with walls
                    q, stress = solve(method, platform, nx, lx, ["reflecting"] * (2 * len(nx)), w)
                    q2, stress2 = solve(method, platform, [2 * n for n in nx], [2 * l for l in lx],
                                        ["periodic"] * (2 * len(nx)), w2)
                    dq = abs(q - q2) / abs(q2)
                    ds = np.max(np.abs(stress - 2.0 * stress2)) / np.max(np.abs(2.0 * stress2))
                    check(dq < 1e-11 and ds < 1e-11, f"{nx} {field_type} {method} {platform}: Q {dq:.1e}, stress {ds:.1e}")


def central_difference(method, platform, nx, lx, bc, w, ds, h=1e-5):
    derivative = []
    for a in range(len(nx)):
        q = []
        for sign in (1, -1):
            box = list(lx)
            box[a] *= 1 + sign * h
            q.append(solve(method, platform, nx, box, bc, w, ds)[0])
        derivative.append(-(np.log(q[0]) - np.log(q[1])) / (2 * h * lx[a]))
    return np.array(derivative)


def absorbing(platforms, rng):
    print("  absorbing walls")
    # Discrete chains: the stress against the central difference on a coarse grid
    for nx, lx, bc in (([16], [2.0], ["absorbing"] * 2),
                       ([12, 10], [2.4, 2.0], ["absorbing", "absorbing", "reflecting", "reflecting"])):
        w = fields(rng, nx, "complex")
        for platform in platforms:
            for method in ("ps", "rs"):
                stress = solve(method, platform, nx, lx, bc, w)[1]
                derivative = central_difference(method, platform, nx, lx, bc, w, 0.05)
                d = np.max(np.abs(stress - derivative)) / np.max(np.abs(derivative))
                check(d < 1e-7, f"{nx} {bc[::2]} {method} {platform}: stress against -d ln Q / dL {d:.1e}")

    # ps against rs where the grid resolves the half bond (2.9 grid spacings for A)
    for nx, lx, bc in (([128], [4.0], ["absorbing"] * 2),
                       ([80, 64], [2.5, 2.0], ["reflecting", "reflecting", "absorbing", "absorbing"])):
        w = fields(rng, nx, "complex")
        for platform in platforms:
            (q_ps, s_ps), (q_rs, s_rs) = solve("ps", platform, nx, lx, bc, w), solve("rs", platform, nx, lx, bc, w)
            dq = abs(q_ps - q_rs) / abs(q_rs)
            ds = np.max(np.abs(s_ps - s_rs)) / np.max(np.abs(s_rs))
            check(dq < 1e-12 and ds < 1e-10, f"{nx} {bc[::2]} {platform}: ps against rs, Q {dq:.1e}, stress {ds:.1e}")

    # Continuous chains through the complex step (the CUDA continuous-chain solvers take no complex fields
    # with walls)
    eps = 1e-4
    for nx, lx, bc in (([48], [2.3], ["absorbing"] * 2),
                       ([24, 20], [1.5, 1.25], ["absorbing", "absorbing", "reflecting", "reflecting"])):
        w = fields(rng, nx, "real")
        v = {m: rng.normal(0.0, 1.0, nx) for m in w}
        for platform in [p for p in platforms if p.startswith("cpu")]:
            for method in ("rqm4", "rk2"):
                stress = solve(method, platform, nx, lx, bc, {m: w[m] + 1j * eps * v[m] for m in w})[1]
                real = solve(method, platform, nx, lx, bc, w)[1]
                plus = solve(method, platform, nx, lx, bc, {m: w[m] + eps * v[m] for m in w})[1]
                minus = solve(method, platform, nx, lx, bc, {m: w[m] - eps * v[m] for m in w})[1]
                derivative = (plus - minus) / (2 * eps)
                d_re = np.max(np.abs(stress.real - real)) / np.max(np.abs(real))
                d_im = np.max(np.abs(stress.imag / eps - derivative)) / np.max(np.abs(derivative))
                check(d_re < 1e-8 and d_im < 1e-8,
                      f"{nx} {bc[::2]} {method} {platform}: complex step, real part {d_re:.1e}, imaginary part {d_im:.1e}")


def cuda_against_cpu(cpus, rng):
    print("  CUDA against the CPU backends")
    for nx, lx, bc in (([128], [4.0], ["reflecting"] * 2),
                       ([80, 64], [2.5, 2.0], ["reflecting", "reflecting", "absorbing", "absorbing"]),
                       ([24, 20, 16], [1.5, 1.25, 1.0], ["absorbing", "absorbing", "reflecting", "reflecting",
                                                         "absorbing", "absorbing"])):
        w = fields(rng, nx, "complex")
        for method in ("ps", "rs"):
            q_gpu, s_gpu = solve(method, "cuda", nx, lx, bc, w)
            for cpu in cpus:
                q_cpu, s_cpu = solve(method, cpu, nx, lx, bc, w)
                dq = abs(q_gpu - q_cpu) / abs(q_cpu)
                ds = np.max(np.abs(s_gpu - s_cpu)) / np.max(np.abs(s_cpu))
                check(dq < 1e-11 and ds < 1e-11, f"{nx} {bc[::2]} {method} cuda vs {cpu}: Q {dq:.1e}, stress {ds:.1e}")


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpus = [p for p in platforms if p.startswith("cpu")]
    print(f"Available platforms: {platforms}")
    if not platforms:
        print("No platform available: skipped")
        return 77
    periodic_image(platforms, np.random.default_rng(5))
    absorbing(platforms, np.random.default_rng(6))
    if "cuda" in platforms and cpus:
        cuda_against_cpu(cpus, np.random.default_rng(7))
    if cuda_part_missing():
        return 1
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
