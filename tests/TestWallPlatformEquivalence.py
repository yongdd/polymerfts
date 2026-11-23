#!/usr/bin/env python3
"""
Boxes with walls give the same propagators on every platform.

Every pair of the available real platforms (cpu-mkl, cpu-fftw and cuda; a CPU-only build
compares cpu-mkl with cpu-fftw) runs two lists of boxes on the same fields; the first platform
of a pair, in this order, is the reference of the relative differences.

- Walls on every axis, all reflecting or all absorbing, in 1D (32 points), 2D (16x16), 3D
  (16x16x16) and 48x48x48, more than the 65536 threads of one stride of the CUDA wall
  kernels: a homopolymer (ds 1/50) of continuous and discrete chains (their default methods)
  in a field of standard deviation 2, whose Nyquist content exposed the CUDA discrete path that
  used the periodic cuFFT plans, CUDA wall kernels without grid-stride loops (NaN beyond 65536
  points) and an MKL DST-III that dropped the highest mode. The partition function agrees to
  1e-9 relative.
- Odd reflecting axes (1D 33, 2D 15x12, 3D 9x8x7), an odd absorbing axis (8x7) and absorbing
  axes of 6144 points (2x6144, 6144x2, 2x2x6144), which the earlier CUDA sine transform
  refused: an AB diblock (ds 0.02) of continuous (rqm4) and discrete chains in fields of
  standard deviation 5. The partition function agrees to 1e-10 relative, the concentrations to
  1e-10 absolute and the stress to 1e-8 relative.

platform="auto" chooses CUDA for every 2D box of these kinds, complex fields of continuous
chains with walls included (in a build with CUDA). The number of comparisons is printed and
checked. Exit code 77 (skipped) with fewer than two real platforms; exit code 1 in a CUDA
build without a visible CUDA device, after the CPU comparisons.
"""

import contextlib
import io
import itertools
import math
import sys

import numpy as np

from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
import checks
from checks import check, summary
from cuda_part import cuda_part_missing

checks.indent = "    "   # the check lines print indented
# The order of the platforms in a pair: the first is the reference of a relative difference
ORDER = ("cpu-mkl", "cpu-fftw", "cuda")


def mt19937_normal(seed, n, stddev):
    """The first n values of std::normal_distribution(0, stddev) on std::mt19937(seed) as
    libstdc++ draws them, bit for bit: Marsaglia's polar method on generate_canonical<double, 53>,
    which takes two 32-bit outputs per number, the first as the low part. These are the fields
    of the C++ test the first list comes from."""
    bits = np.random.RandomState(seed).randint(0, 2**32, size=4 * n + 64, dtype=np.uint32).astype(float)
    uniform = (bits[0::2] + bits[1::2] * 2.0**32) / 2.0**64
    values = []
    for x, y in zip(2.0 * uniform[0::2] - 1.0, 2.0 * uniform[1::2] - 1.0):
        r2 = x * x + y * y
        if r2 > 1.0 or r2 == 0.0:
            continue
        mult = math.sqrt(-2.0 * math.log(r2) / r2)
        values += [y * mult * stddev, x * mult * stddev]
        if len(values) >= n:
            return np.array(values[:n])
    raise RuntimeError("too few uniform numbers drawn")


def partition_function(platform, chain_model, nx, lx, bc, w):
    """Q of a homopolymer A of length 1 (ds 1/50) in the field w."""
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(nx=list(nx), lx=list(lx), ds=1.0 / 50.0, bond_lengths={"A": 1.0}, bc=list(bc),
                                  chain_model=chain_model, platform=platform)
        solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
        solver.compute_propagators({"A": w.copy()})
    return solver.get_partition_function(0)


def diblock(platform, chain_model, nx, lx, bc, w):
    """Q, the concentrations and the stress of an AB diblock (ds 0.02) in the fields w."""
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=list(nx), lx=list(lx), ds=0.02, bond_lengths={"A": 1.0, "B": 1.0}, bc=list(bc),
            chain_model=chain_model, numerical_method="rqm4" if chain_model == "continuous" else None,
            platform=platform)
        solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        solver.compute_propagators({m: w[m].copy() for m in "AB"})
        solver.compute_concentrations()
        phi = {m: np.asarray(solver.get_concentration(m)).flatten() for m in "AB"}
        solver.compute_stress()
        stress = np.asarray(solver.get_stress()).astype(float)[:len(nx)]
    return solver.get_partition_function(0), phi, stress


def main():
    platforms = [p for p in ORDER if p in _core.PlatformSelector.avail_platforms()]
    print(f"Platforms compared: {platforms}")
    if len(platforms) < 2:
        if cuda_part_missing():
            return 1
        print("Fewer than two real platforms: skipped")
        return 77
    pairs = list(itertools.combinations(platforms, 2))
    n_compared = 0

    # Walls on every axis: a homopolymer, the field of std::mt19937(12345) drawn anew for each
    # box (the first M values of one stream)
    print("\nWalls on every axis, a homopolymer: Q")
    field = mt19937_normal(12345, 48 * 48 * 48, 2.0)
    homopolymer_boxes = [([32], [3.0]), ([16, 16], [3.0, 3.0]), ([16, 16, 16], [3.0, 3.0, 3.0]),
                         ([48, 48, 48], [3.0, 3.0, 3.0])]
    for chain_model in ("continuous", "discrete"):
        for wall in ("reflecting", "absorbing"):
            for nx, lx in homopolymer_boxes:
                w = field[:int(np.prod(nx))]
                q = {p: partition_function(p, chain_model, nx, lx, [wall] * (2 * len(nx)), w) for p in platforms}
                for a, b in pairs:
                    rel_q = abs(q[b] - q[a]) / abs(q[a])
                    check(rel_q < 1e-9, f"{chain_model} {nx} {wall}: {b} vs {a}: Q {q[a]:.12f}, "
                                        f"relative diff {rel_q:.2e}")
                    n_compared += 1

    # Odd wall axes and absorbing axes of 6144 points: an AB diblock
    print("\nOdd wall axes and absorbing axes of 6144 points, an AB diblock: Q, concentrations, stress")
    cases = [
        ([33], [3.0], ["reflecting", "reflecting"]),
        ([15, 12], [2.0, 1.6], ["reflecting", "reflecting", "absorbing", "absorbing"]),
        ([9, 8, 7], [1.5, 1.4, 1.3], ["reflecting", "reflecting", "absorbing", "absorbing", "reflecting", "reflecting"]),
    ]
    # Boxes the earlier CUDA sine transform refused: an odd absorbing axis, and absorbing axes
    # of 6144 points (a line of nx + 2 values did not fit the shared memory of a block)
    absorbing_6144 = []
    for nx in ([2, 6144], [6144, 2], [2, 2, 6144]):
        bc = ["reflecting"] * (2*len(nx))
        bc[2*nx.index(6144):2*nx.index(6144) + 2] = ["absorbing", "absorbing"]
        absorbing_6144.append((nx, [1.0] * len(nx), bc))
    cases += [([8, 7], [1.0, 1.0], ["reflecting", "reflecting", "absorbing", "absorbing"])] + absorbing_6144
    rng = np.random.default_rng(31)
    for nx, lx, bc in cases:
        w = {m: rng.normal(0.0, 5.0, int(np.prod(nx))) for m in "AB"}
        for chain_model in ("continuous", "discrete"):
            results = {p: diblock(p, chain_model, nx, lx, bc, w) for p in platforms}
            for a, b in pairs:
                (q_a, phi_a, s_a), (q_b, phi_b, s_b) = results[a], results[b]
                tag = f"{nx} {bc[::2]} {chain_model}: {b} vs {a}"
                rel_q = abs(q_b - q_a) / abs(q_a)
                dphi = max(np.max(np.abs(phi_b[m] - phi_a[m])) for m in "AB")
                rel_s = np.max(np.abs(s_b - s_a) / np.abs(s_a))
                check(rel_q < 1e-10, f"{tag}: Q relative diff {rel_q:.2e}")
                check(dphi < 1e-10, f"{tag}: phi max abs diff {dphi:.2e}")
                check(rel_s < 1e-8, f"{tag}: stress relative diff {rel_s:.2e}")
                n_compared += 1

    # platform="auto" runs every 2D box on CUDA
    auto_rows = []
    if "cuda" in platforms:
        print("\nplatform=\"auto\"")
        walls = ["reflecting", "reflecting", "absorbing", "absorbing"]
        auto_rows = [
            ([8, 7], walls, "continuous", "real", "cuda"), ([8, 7], walls, "discrete", "real", "cuda"),
            ([8, 8], walls, "discrete", "real", "cuda"), ([8, 8], walls, "continuous", "complex", "cuda"),
            ([8, 8], walls, "discrete", "complex", "cuda"), ([8, 7], ["periodic"]*4, "continuous", "complex", "cuda"),
            ([2, 6144], walls, "continuous", "real", "cuda"), ([2, 6142], walls, "discrete", "real", "cuda"),
            ([6144, 2], ["reflecting"]*4, "continuous", "real", "cuda")]
    for nx, bc, chain_model, field_type, platform in auto_rows:
        with contextlib.redirect_stdout(io.StringIO()):
            chosen = PropagatorSolver(nx=nx, lx=[1.0, 1.0], ds=0.02, bond_lengths={"A": 1.0}, bc=bc,
                                      chain_model=chain_model, field_type=field_type).platform
        check(chosen.startswith(platform), f"auto platform for {nx} {bc[::2]} {chain_model} {field_type}: {chosen}")
        n_compared += 1

    expected = (2 * 2 * len(homopolymer_boxes) + 2 * len(cases)) * len(pairs) + len(auto_rows)
    print(f"\n{n_compared} comparisons (expected {expected}) of {len(pairs)} platform pairs")
    check(n_compared == expected, "every comparison ran")
    return summary() | cuda_part_missing()


if __name__ == "__main__":
    sys.exit(main())
