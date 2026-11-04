#!/usr/bin/env python3
"""
Discrete chains with the bond convolution summed in real space (numerical_method "rs").

1. The transform of the sampled Gaussian differs from the bond function exp(-k^2 sigma^2/2)
   of the FFT solver "ps" by about exp(-pi^2 sigma^2 / 2 dx^2) near the largest wavenumber
   of the grid. Here the standard deviation of a half bond spans 2.9 grid spacings, which
   makes that 1e-18, so both give the same partition function (relative 1e-12) and
   concentrations (1e-12 of the largest) in random fields (std 5): an AB diblock with
   different bond lengths (half bonds at the junction) in 1D, 2D and 3D, periodic,
   reflecting and absorbing walls, a mask, and complex fields; with them the stress (relative
   1e-10; complex fields in periodic boxes, as PS rejects their stress with walls).
2. Boundaries "ps" cannot take: an axis absorbing at one end and reflecting at the
   other equals the first half of the doubled axis absorbing at both ends in the
   mirrored field; a box periodic in x and reflecting in y with a field along y
   equals the 1D box in y.
3. A grafted chain at a reflecting wall in a field that drops by 350 kT per chain
   (a charged brush): every concentration is positive and the partition function is
   the same from every segment.
4. On coarse grids, where the width of the kernels is corrected to keep their variance, the
   stress is still the derivative of -ln Q with respect to the box lengths (central differences,
   relative 1e-7).
5. CUDA equals the CPU backends (relative 1e-11) on these cases, the stress included.

SCFT relaxes the box lengths with this stress; skewed cells, the relaxation of angles and
continuous-chain names are refused. Exit code 1 in a CUDA build without a
visible CUDA device, after the CPU parts.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import _core, scft
from polymerfts.theory.propagator_solver import PropagatorSolver
from polymerfts.utils.validation import ValidationError
from cuda_part import cuda_part_missing

FAILURES = []
DIBLOCK = [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]]


def check(condition, message):
    print(f"    [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def solve(method, platform, nx, lx, bc, w, ds=0.05, blocks=DIBLOCK, mask=None, field_type="real",
          grafting=None, q_init=None):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(nx=list(nx), lx=list(lx), ds=ds, bond_lengths={"A": 1.0, "B": 1.2}, bc=list(bc),
                                  chain_model="discrete", numerical_method=method, platform=platform, mask=mask,
                                  field_type=field_type)
        solver.add_polymer(1.0, blocks, grafting_points=grafting)
        solver.compute_propagators({m: w[m].copy() for m in w}, q_init=q_init)
        solver.compute_concentrations()
    phi = {m: np.asarray(solver.get_concentration(m)).flatten() for m in w}
    return solver, solver.get_partition_function(0), phi


def stress_of(solved):
    solver = solved[0]
    with contextlib.redirect_stdout(io.StringIO()):
        solver.compute_stress()
    return np.asarray(solver.get_stress())[:len(solver.nx)]


def compare_stress(tag, a, b, tolerance):
    sa, sb = stress_of(a), stress_of(b)
    d = np.max(np.abs(sa - sb)) / np.max(np.abs(sb))
    check(d < tolerance, f"{tag}: stress {d:.1e}")


def compare(tag, a, b, tolerance):
    (_, q_a, phi_a), (_, q_b, phi_b) = a, b
    top = max(np.max(np.abs(phi_b[m])) for m in phi_b)
    dq = abs(q_a - q_b) / abs(q_b)
    dphi = max(np.max(np.abs(phi_a[m] - phi_b[m])) for m in phi_b) / top
    check(dq < tolerance and dphi < tolerance, f"{tag}: Q {dq:.1e}, phi {dphi:.1e}")


def fields(rng, n, field_type="real"):
    w = {m: rng.normal(0.0, 5.0, n) for m in "AB"}
    if field_type == "complex":
        w = {m: w[m] + 1j * rng.normal(0.0, 1.0, n) for m in "AB"}
    return w


# Grid spacing 1/32: the half bond of A (ds = 0.05) has the standard deviation 0.091
CASES = [
    ([128], [4.0], ["periodic"] * 2),
    ([128], [4.0], ["reflecting"] * 2),
    ([128], [4.0], ["absorbing"] * 2),
    ([80, 64], [2.5, 2.0], ["reflecting", "reflecting", "absorbing", "absorbing"]),
    ([32, 32, 32], [1.0, 1.0, 1.0], ["periodic"] * 6),
    ([32, 28, 24], [1.0, 0.875, 0.75], ["absorbing", "absorbing", "reflecting", "reflecting", "absorbing", "absorbing"]),
    # A kernel of 2x270+1 points along the first axis, longer than the box: folded again and again, and
    # on CUDA too long for the tiles in shared memory
    ([256, 128], [2.0, 4.0], ["absorbing", "absorbing", "reflecting", "reflecting"]),
]


def disk(nx, lx):
    x, y = np.meshgrid(*[(np.arange(n) + 0.5) * l / n for n, l in zip(nx, lx)], indexing="ij")
    return np.where((x - 0.5 * lx[0]) ** 2 + (y - 0.5 * lx[1]) ** 2 < 0.4 ** 2, 0.0, 1.0).flatten()


def against_ps(platform, rng):
    print(f"  rs against ps on {platform}")
    for nx, lx, bc in CASES:
        for field_type in ("real", "complex"):
            w = fields(rng, int(np.prod(nx)), field_type)
            rs = solve("rs", platform, nx, lx, bc, w, field_type=field_type)
            ps = solve("ps", platform, nx, lx, bc, w, field_type=field_type)
            compare(f"{nx} {bc[::2]} {field_type}", rs, ps, 1e-12)
            if field_type == "real" or all(b == "periodic" for b in bc):
                compare_stress(f"{nx} {bc[::2]} {field_type}", rs, ps, 1e-10)
            check(rs[0].check_total_partition(), f"{nx} {bc[::2]} {field_type}: partition function from every segment")
            total = sum(rs[2][m] for m in "AB")
            check(abs(np.mean(total) - 1.0) < 1e-12, f"{nx} {bc[::2]} {field_type}: mean total concentration 1")
    nx, lx = [96, 80], [3.0, 2.5]
    mask = disk(nx, lx)
    w = fields(rng, int(np.prod(nx)))
    compare("mask in a periodic box", solve("rs", platform, nx, lx, ["periodic"] * 4, w, mask=mask),
            solve("ps", platform, nx, lx, ["periodic"] * 4, w, mask=mask), 1e-12)


def mixed_boundaries(platform, rng):
    print(f"  mixed boundaries on {platform}")
    # Absorbing at x = 0, reflecting at x = L: the first half of an absorbing box of 2L in the mirrored field
    w = fields(rng, 96)
    doubled = {m: np.concatenate([w[m], w[m][::-1]]) for m in w}
    half = solve("rs", platform, [96], [3.0], ["absorbing", "reflecting"], w)
    full = solve("ps", platform, [192], [6.0], ["absorbing", "absorbing"], doubled)
    full = (full[0], full[1], {m: full[2][m][:96] for m in full[2]})
    compare("absorbing-reflecting against the doubled absorbing box", half, full, 1e-12)

    # Periodic in x, reflecting in y, a field along y only: the 1D box in y
    wy = fields(rng, 80)
    box = solve("rs", platform, [24, 80], [0.75, 2.5], ["periodic", "periodic", "reflecting", "reflecting"],
                {m: np.tile(wy[m], 24) for m in wy})
    line = solve("ps", platform, [80], [2.5], ["reflecting", "reflecting"], wy)
    box = (box[0], box[1], {m: box[2][m].reshape(24, 80).mean(axis=0) for m in box[2]})
    compare("periodic x reflecting box against the 1D box", box, line, 1e-12)


def strong_field(platform):
    print(f"  grafted chain in a field dropping by 350 kT per chain on {platform}")
    nx, lx = 1600, 20.0
    z = (np.arange(nx) + 0.5) * lx / nx
    w = {"A": 350.0 / (1.0 + np.exp((z - 8.0) / 0.5))}
    graft = np.exp(-0.5 * (z / 0.05) ** 2)
    for method in ("rs", "ps"):
        solver, _, phi = solve(method, platform, [nx], [lx], ["reflecting", "reflecting"], w, ds=0.01,
                               blocks=[["A", 1.0, 0, 1]], grafting={0: "G"}, q_init={"G": graft})
        ratio = phi["A"].min() / phi["A"].max()
        if method == "rs":
            check(ratio > 0.0, f"rs: smallest concentration / largest {ratio:.1e}")
            check(solver.check_total_partition(), f"rs: partition function from every segment, "
                                                  f"error {max(solver.partition_errors()):.1e}")
        else:
            print(f"    (ps: smallest concentration / largest {ratio:.1e})")
    return solve("rs", platform, [nx], [lx], ["reflecting", "reflecting"], w, ds=0.01,
                 blocks=[["A", 1.0, 0, 1]], grafting={0: "G"}, q_init={"G": graft})


def stress_derivative(platform, rng):
    print(f"  stress against differences of -ln Q on coarse grids on {platform}")
    for nx, lx, bc in (([16], [4.0], ["periodic"] * 2), ([12, 10], [2.4, 2.0], ["reflecting"] * 4)):
        w = fields(rng, int(np.prod(nx)))
        stress = stress_of(solve("rs", platform, nx, lx, bc, w))
        h = 1e-5
        for a in range(len(nx)):
            q = []
            for sign in (1, -1):
                box = list(lx)
                box[a] *= 1 + sign * h
                q.append(solve("rs", platform, nx, box, bc, w)[1])
            derivative = -(np.log(q[0]) - np.log(q[1])) / (2 * h * lx[a])
            d = abs(stress[a] - derivative) / abs(derivative)
            check(d < 1e-7, f"{nx} {bc[0]} axis {a}: stress {stress[a]:.8f}, -dlnQ/dL {derivative:.8f}, {d:.1e}")


def refusals(platform):
    print(f"  refusals on {platform}")
    # The Gaussian factorizes along the axes of orthogonal cells only
    try:
        PropagatorSolver(nx=[8, 8, 8], lx=[1.0] * 3, ds=0.05, bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                         chain_model="discrete", numerical_method="rs", platform=platform, angles=[90, 90, 120])
        check(False, "rs refuses a skewed cell")
    except ValueError:
        check(True, "rs refuses a skewed cell")
    box, _, _ = solve("rs", platform, [8, 8, 8], [1.0] * 3, ["periodic"] * 6, fields(np.random.default_rng(2), 512))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            box.set_lattice_parameters([1.0] * 3, [90.0, 90.0, 100.0])
        check(False, "rs refuses to skew its cell")
    except Exception as exc:
        check("orthogonal" in str(exc), "rs refuses to skew its cell")
    # SCFT relaxes the box lengths with the stress of rs, but not the angles of a skewed crystal system
    params = {"nx": [8, 8, 8], "lx": [2.0] * 3, "ds": 0.05, "chain_model": "discrete", "numerical_method": "rs",
              "platform": platform, "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15},
              "relax_cell": True, "max_iter": 3, "tolerance": 1e-8,
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.4}, {"type": "B", "length": 0.6}]}],
              "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1}}
    w = {"A": np.random.default_rng(4).normal(0.0, 1.0, 512), "B": np.zeros(512)}
    with contextlib.redirect_stdout(io.StringIO()):
        calc = scft.SCFT(params=params)
        result = calc.run(initial_fields=w, return_result=True)
    check(max(abs(l - 2.0) for l in result.lx) > 1e-6, f"SCFT with rs relaxes the box lengths: {np.round(result.lx, 6)}")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scft.SCFT(params=dict(params, crystal_system="Monoclinic"))
        check(False, "SCFT with rs refuses to relax the angles")
    except ValidationError:
        check(True, "SCFT with rs refuses to relax the angles")
    for chain_model, method in (("discrete", "rqm4"), ("continuous", "rs")):
        try:
            PropagatorSolver(nx=[16], lx=[1.0], ds=0.05, bond_lengths={"A": 1.0}, bc=["periodic"] * 2,
                             chain_model=chain_model, numerical_method=method, platform=platform)
            check(False, f"{chain_model} chains refuse {method}")
        except ValueError:
            check(True, f"{chain_model} chains refuse {method}")


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpus = [p for p in platforms if p.startswith("cpu")]
    print(f"Available platforms: {platforms}")
    reference = {}
    for platform in cpus:
        rng = np.random.default_rng(7)
        against_ps(platform, rng)
        mixed_boundaries(platform, rng)
        reference[platform] = strong_field(platform)
        stress_derivative(platform, rng)
        refusals(platform)

    if "cuda" in platforms:
        rng = np.random.default_rng(7)
        against_ps("cuda", rng)
        mixed_boundaries("cuda", rng)
        gpu = strong_field("cuda")
        stress_derivative("cuda", rng)
        refusals("cuda")
        print("  CUDA against the CPU backends")
        rng = np.random.default_rng(11)
        for nx, lx, bc in CASES + [([24, 80], [0.75, 2.5], ["periodic", "periodic", "reflecting", "reflecting"])]:
            for field_type in ("real", "complex"):
                w = fields(rng, int(np.prod(nx)), field_type)
                cuda = solve("rs", "cuda", nx, lx, bc, w, field_type=field_type)
                for cpu in cpus:
                    host = solve("rs", cpu, nx, lx, bc, w, field_type=field_type)
                    compare(f"{nx} {bc[::2]} {field_type}: cuda vs {cpu}", cuda, host, 1e-11)
                    if field_type == "real" or all(b == "periodic" for b in bc):
                        compare_stress(f"{nx} {bc[::2]} {field_type}: cuda vs {cpu}", cuda, host, 1e-11)
        for cpu in cpus:
            compare(f"strong field: cuda vs {cpu}", gpu, reference[cpu], 1e-11)
    elif cuda_part_missing():
        return 1

    if not cpus and "cuda" not in platforms:
        print("No platform available: skipped")
        return 77
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
