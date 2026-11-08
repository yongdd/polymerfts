#!/usr/bin/env python3
"""
Block multiplicities: m copies of a block and of everything on its u side.

A block (v, u) with multiplicity m stands for m copies of its branch attached at
v; for real m this continues an integer number of identical arms (an A_n B_n
star with real n). This test checks, for each available platform, both chain
models, propagator aggregation on and off and reduced memory on and off:

  1. Integer multiplicities give the same chains as the explicit arms: partition
     functions (relative difference < 1e-13), concentrations and stress
     (< 1e-12), and each block's concentration equals the sum over its explicit
     copies. Covered: an A3B2 miktoarm star, an (AB)3 star, nested copies (two
     arms ending in three sub-arms), a homopolymer star given by one block with
     its copies at a chain end, in a blend with a diblock, and a pom-pom whose two
     copied ends share one computation block.
  2. Real multiplicities (A_n B_n, n = 5.6 and 1.5): check_total_partition(),
     material conservation (< 1e-12), full and reduced memory agree bit for bit
     (Q, concentrations and stress; if not, the test prints whether the stress
     repeats and at which steps the chain propagators differ), the stress
     equals the central differences of -ln Q / alpha in the box lengths
     (discrete < 1e-7; continuous < 3e-4 at
     ds = 0.0125, where its contour quadrature is accurate enough), platforms agree
     (< 1e-10), the uniform-field limit
     Q = exp(-sum copies * length * w) (< 1e-12), ln Q is convex in n, and the
     real power at n = 5 + 1e-9 joins the integer power at n = 5 (< 1e-7).
  3. Input errors: multiplicity < 1, two blocks replicating each other, copies at
     a grafted end, a non-integer multiplicity with complex fields (every platform
     and chain model, and CL-FTS validation, which also refuses integers above
     10^6), while an integer one reproduces the explicit arms in complex fields; a
     real power of a negative propagator (Crank-Nicolson next to absorbing walls)
     stops with an error naming the multiplicity on every platform, a near-integer
     one with all its digits; blocks numbered only in part give a
     ValidationError; the Python copy count equals the native one, and alpha and
     the segment total count every copy.
  4. SCFT of a blend of an A3B2 star with a diblock, given with multiplicities and
     with explicit arms: the same free energy, concentrations and chemical
     potentials (the Python layer weights blocks by their copies).
  5. L-FTS, six Langevin steps from the same fields and noise: an A3B2 star given
     with multiplicities follows the explicit arms (fields < 1e-9); an A_n B_n star
     at n = 5 + 1e-9 (real powers) follows n = 5 (< 1e-6); at n = 5.6 the run
     stays finite, conserves material (< 1e-12) and draws the chain.

Fields are smooth random fields with standard deviation 5. Runs in seconds.
Exit code 0 on success, 1 on any failure.
"""

import sys

import numpy as np

from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
from polymerfts.utils.validation import ValidationError, polymer_block_copies, validate_lfts_params
from cuda_part import cuda_part_missing

NX = [12, 10, 8]
LX = [2.0, 1.8, 1.5]
DS = 0.05
BONDS = {"A": 1.0, "B": 1.2}

FAILURES = []


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def smooth_field(rng):
    x = [np.arange(n) * l / n for n, l in zip(NX, LX)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    f = np.zeros(NX)
    for _ in range(4):
        kx, ky, kz = rng.integers(1, 3, size=3)
        f += rng.normal() * (np.cos(2 * np.pi * kx * X / LX[0] + rng.uniform(0, 2 * np.pi))
                             * np.cos(2 * np.pi * ky * Y / LX[1] + rng.uniform(0, 2 * np.pi))
                             * np.cos(2 * np.pi * kz * Z / LX[2] + rng.uniform(0, 2 * np.pi)))
    f = f - f.mean()
    return (f / f.std() * 5.0).flatten()


def make_solver(platform, chain_model, aggregate=True, reduce_memory=False, field_type="real", lx=LX, ds=DS):
    return PropagatorSolver(
        nx=NX, lx=lx, ds=ds, bond_lengths=BONDS, bc=["periodic"] * 6, chain_model=chain_model,
        numerical_method="rqm4" if chain_model == "continuous" else None, platform=platform,
        reduce_memory=reduce_memory, aggregate_propagator_computation=aggregate, field_type=field_type)


def solve(polymers, w, platform, chain_model, aggregate=True, reduce_memory=False, stress=False):
    solver = make_solver(platform, chain_model, aggregate, reduce_memory)
    for volume_fraction, blocks in polymers:
        solver.add_polymer(volume_fraction, blocks)
    solver.compute_propagators({key: value.copy() for key, value in w.items()})
    solver.compute_concentrations()
    result = {
        "solver": solver,
        "Q": [solver.get_partition_function(p) for p in range(len(polymers))],
        "phi": {key: np.asarray(solver.get_concentration(key)).flatten() for key in w},
        "partition_ok": solver.check_total_partition(),
    }
    if stress:
        solver.compute_stress()
        result["stress"] = np.asarray(solver.get_stress())
    if not aggregate:
        result["blocks"] = [np.asarray(solver._propagator_computation.get_block_concentration(p))
                            for p in range(len(polymers))]
    return result


# (name, compact polymers, explicit polymers, explicit blocks of each compact block per polymer)
ARCHITECTURES = [
    ("A3B2 miktoarm star",
     [(1.0, [["A", 0.3, 0, 1, 3], ["B", 0.4, 0, 2, 2]])],
     [(1.0, [["A", 0.3, 0, 1], ["A", 0.3, 0, 2], ["A", 0.3, 0, 3], ["B", 0.4, 0, 4], ["B", 0.4, 0, 5]])],
     [[[0, 1, 2], [3, 4]]]),
    ("(AB)3 star",
     [(1.0, [["A", 0.2, 0, 1, 3], ["B", 0.3, 1, 2]])],
     [(1.0, [["A", 0.2, 0, 1], ["A", 0.2, 0, 2], ["A", 0.2, 0, 3],
             ["B", 0.3, 1, 4], ["B", 0.3, 2, 5], ["B", 0.3, 3, 6]])],
     [[[0, 1, 2], [3, 4, 5]]]),
    ("nested copies",
     [(1.0, [["A", 0.2, 0, 1, 2], ["B", 0.2, 1, 2, 3], ["A", 0.1, 0, 3]])],
     [(1.0, [["A", 0.2, 0, 1], ["A", 0.2, 0, 2],
             ["B", 0.2, 1, 3], ["B", 0.2, 1, 4], ["B", 0.2, 1, 5],
             ["B", 0.2, 2, 6], ["B", 0.2, 2, 7], ["B", 0.2, 2, 8], ["A", 0.1, 0, 9]])],
     [[[0, 1], [2, 3, 4, 5, 6, 7], [8]]]),
    ("star at a chain end, blended",
     [(0.6, [["A", 0.4, 0, 1, 4]]), (0.4, [["A", 0.3, 0, 1], ["B", 0.3, 1, 2]])],
     [(0.6, [["A", 0.4, 0, 1], ["A", 0.4, 0, 2], ["A", 0.4, 0, 3], ["A", 0.4, 0, 4]]),
      (0.4, [["A", 0.3, 0, 1], ["B", 0.3, 1, 2]])],
     [[[0, 1, 2, 3]], [[0], [1]]]),
    ("pom-pom, copies at both ends",
     [(1.0, [["A", 0.6, 0, 1], ["B", 0.2, 0, 2, 3], ["B", 0.2, 1, 3, 3]])],
     [(1.0, [["A", 0.6, 0, 1], ["B", 0.2, 0, 2], ["B", 0.2, 0, 3], ["B", 0.2, 0, 4],
             ["B", 0.2, 1, 5], ["B", 0.2, 1, 6], ["B", 0.2, 1, 7]])],
     [[[0], [1, 2, 3], [4, 5, 6]]]),
]


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def check_integer_multiplicities(w, platforms):
    print("1. Integer multiplicities against explicit arms")
    for name, compact, explicit, copies in ARCHITECTURES:
        for platform in platforms:
            for chain_model in ["continuous", "discrete"]:
                for aggregate in [True, False]:
                    for reduce_memory in [False, True]:
                        tag = (f"{name}, {platform}/{chain_model}, aggregation {aggregate}, "
                               f"reduce_memory {reduce_memory}")
                        c = solve(compact, w, platform, chain_model, aggregate, reduce_memory, stress=True)
                        e = solve(explicit, w, platform, chain_model, aggregate, reduce_memory, stress=True)
                        dq = max(abs(qc / qe - 1.0) for qc, qe in zip(c["Q"], e["Q"]))
                        dphi = max(max_abs(c["phi"][key], e["phi"][key]) for key in w)
                        dstress = max_abs(c["stress"], e["stress"])
                        ok = (c["partition_ok"] and e["partition_ok"] and dq < 1e-13
                              and dphi < 1e-12 and dstress < 1e-12)
                        if not aggregate:
                            for p, block_copies in enumerate(copies):
                                for b, explicit_blocks in enumerate(block_copies):
                                    total = c["blocks"][p][b]
                                    expected = sum(e["blocks"][p][j] for j in explicit_blocks)
                                    ok = ok and max_abs(total, expected) < 1e-12
                        check(ok, f"{tag}: |dQ/Q| {dq:.1e}, |dphi| {dphi:.1e}, |dstress| {dstress:.1e}")


def star(n):
    return [(1.0, [["A", 0.5, 0, 1, n], ["B", 0.5, 0, 2, n]])]


# Stress against the central differences of -ln Q / alpha. Discrete chains: the stress is the exact
# derivative of the discrete ln Q, so only the differencing error remains (4e-9). Continuous chains:
# the stress is a contour quadrature whose error falls about 13-fold per halving of ds, alike for
# real and integer n and for a diblock (6e-3 at ds = 0.05, 5e-5 at 0.0125); it is checked at 0.0125.
STRESS_DS = {"discrete": DS, "continuous": 0.0125}
STRESS_TOLERANCE = {"discrete": 1e-7, "continuous": 3e-4}


def stress_mismatch(n, w, platform, chain_model):
    """Largest difference between the first three stress components and the central differences of
    -ln Q / alpha in the box lengths at fixed grid fields, relative to the largest difference. For a
    melt of one polymer get_stress gives the derivatives of -sum_p phi_p/alpha_p ln Q_p."""
    def solver_at(lx):
        solver = make_solver(platform, chain_model, lx=lx, ds=STRESS_DS[chain_model])
        solver.add_polymer(1.0, star(n)[0][1])
        solver.compute_propagators({key: value.copy() for key, value in w.items()})
        return solver

    solver = solver_at(LX)
    solver.compute_stress()
    stress = np.asarray(solver.get_stress())[:3]
    derivatives = []
    for d in range(3):
        values = []
        for sign in (1.0, -1.0):
            lx = list(LX)
            lx[d] += sign * 1e-4 * LX[d]
            shifted = solver_at(lx)
            values.append(-np.log(shifted.get_partition_function(0)) / shifted.get_polymer(0).get_alpha())
        derivatives.append((values[0] - values[1]) / (2e-4 * LX[d]))
    derivatives = np.array(derivatives)
    return np.max(np.abs(stress - derivatives)) / np.max(np.abs(derivatives))


def print_memory_mode_differences(n, w, platform, chain_model, full, reduced):
    """Where full and reduced memory part when they do not agree bit for bit"""
    def digits(stress):
        return [f"{x:.17g}" for x in stress]

    results = {"full": full, "reduced": reduced}
    lines = [f"stress, {name}: {digits(result['stress'])}" for name, result in results.items()]
    try:
        # Both repeats run before a new solver changes the heap
        again = {}
        for name, result in results.items():
            result["solver"].compute_stress()
            again[name] = np.asarray(result["solver"].get_stress())
        fresh = {name: solve(star(n), w, platform, chain_model, reduce_memory=(name == "reduced"),
                             stress=True)["stress"] for name in results}
        for name, result in results.items():
            lines.append(f"{name}, compute_stress() again: {digits(again[name])}, the same: "
                         f"{np.array_equal(again[name], result['stress'])}")
            lines.append(f"{name}, a new solver: {digits(fresh[name])}, the same: "
                         f"{np.array_equal(fresh[name], result['stress'])}")
        lines.append(f"new solvers, full and reduced agree bit for bit: "
                     f"{np.array_equal(fresh['full'], fresh['reduced'])}")
        first_step = 0 if chain_model == "continuous" else 1
        if chain_model == "discrete":
            lines.append("propagators (the half-bond states at the junction are not compared):")
        for v, u in [(0, 1), (1, 0), (0, 2), (2, 0)]:
            steps, largest = [], 0.0
            for step in range(first_step, round(0.5 / DS) + 1):
                a = np.asarray(full["solver"].get_propagator(0, v, u, step))
                b = np.asarray(reduced["solver"].get_propagator(0, v, u, step))
                if not np.array_equal(a, b):
                    steps.append(step)
                    largest = max(largest, max_abs(a, b))
            lines.append(f"propagator ({v},{u}): " + (f"differs at steps {steps}, max {largest:.1e}"
                                                     if steps else "the same at every step"))
    except Exception as exc:
        lines.append(f"the comparison stopped: {exc!r}")
    for line in lines:
        print(f"        {line}")


def check_real_multiplicities(w, platforms):
    print("2. Real multiplicities (A_n B_n stars)")
    for n in [5.6, 1.5]:
        reference = {}
        for platform in platforms:
            for chain_model in ["continuous", "discrete"]:
                tag = f"n = {n}, {platform}/{chain_model}"
                full = solve(star(n), w, platform, chain_model, stress=True)
                reduced = solve(star(n), w, platform, chain_model, reduce_memory=True, stress=True)
                mass = abs(np.mean(full["phi"]["A"] + full["phi"]["B"]) - 1.0)
                check(full["partition_ok"] and reduced["partition_ok"], f"{tag}: check_total_partition()")
                check(mass < 1e-12, f"{tag}: |mean(phi_A + phi_B) - 1| = {mass:.1e}")
                same = (full["Q"] == reduced["Q"] and all(np.array_equal(full["phi"][k], reduced["phi"][k]) for k in w)
                        and np.array_equal(full["stress"], reduced["stress"]))
                # The differences name the quantity that broke the bitwise agreement
                q_diff = max(abs(a / b - 1.0) for a, b in zip(full["Q"], reduced["Q"]))
                phi_diff = max(max_abs(full["phi"][k], reduced["phi"][k]) for k in w)
                stress_diff = max_abs(full["stress"], reduced["stress"])
                check(same, f"{tag}: full and reduced memory agree bit for bit (|dQ/Q| {q_diff:.1e}, "
                            f"|dphi| {phi_diff:.1e}, |dstress| {stress_diff:.1e})")
                if not same:
                    print_memory_mode_differences(n, w, platform, chain_model, full, reduced)
                dstress = stress_mismatch(n, w, platform, chain_model)
                check(dstress < STRESS_TOLERANCE[chain_model],
                      f"{tag}: stress against the differences of -ln Q / alpha at ds = "
                      f"{STRESS_DS[chain_model]} (relative {dstress:.1e})")
                if chain_model in reference:
                    ref = reference[chain_model]
                    dq = abs(full["Q"][0] / ref["Q"][0] - 1.0)
                    dphi = max(max_abs(full["phi"][k], ref["phi"][k]) for k in w)
                    check(dq < 1e-10 and dphi < 1e-10,
                          f"{tag}: agrees with {ref['platform']} (|dQ/Q| {dq:.1e}, |dphi| {dphi:.1e})")
                else:
                    reference[chain_model] = dict(full, platform=platform)

    # Uniform fields: Q = exp(-sum over block copies of length * w), phi_A = phi_B = 1/2
    uniform = {"A": np.full(np.prod(NX), 0.7), "B": np.full(np.prod(NX), -0.3)}
    n = 5.6
    expected = np.exp(-(n * 0.5 * 0.7 + n * 0.5 * -0.3))
    for platform in platforms:
        for chain_model in ["continuous", "discrete"]:
            r = solve(star(n), uniform, platform, chain_model)
            dq = abs(r["Q"][0] / expected - 1.0)
            dphi = max(max_abs(r["phi"]["A"], 0.5), max_abs(r["phi"]["B"], 0.5))
            check(dq < 1e-12 and dphi < 1e-12,
                  f"uniform fields, {platform}/{chain_model}: |Q/exp(-sum n L w) - 1| {dq:.1e}, |phi - 1/2| {dphi:.1e}")

    # ln Q = ln <q_c^n> is convex in n; the real power continues the integer powers
    for platform in platforms:
        for chain_model in ["continuous", "discrete"]:
            log_q = {n: np.log(solve(star(n), w, platform, chain_model)["Q"][0]) for n in [5, 5.5, 6, 5 + 1e-9]}
            check(log_q[5.5] < 0.5 * (log_q[5] + log_q[6]),
                  f"{platform}/{chain_model}: ln Q(5.5) < (ln Q(5) + ln Q(6))/2")
            joined = abs(log_q[5 + 1e-9] - log_q[5])
            check(joined < 1e-7, f"{platform}/{chain_model}: |ln Q(5 + 1e-9) - ln Q(5)| = {joined:.1e}")


def raises(function):
    try:
        function()
    except Exception as exc:  # native errors arrive as RuntimeError or ValueError
        return str(exc)
    return None


def check_input_errors(w, platforms):
    print("3. Input errors and copy counts")
    platform = platforms[0]

    def add(blocks, grafting_points=None):
        solver = make_solver(platform, "continuous")
        solver.add_polymer(1.0, blocks, grafting_points)

    message = raises(lambda: add([["A", 0.5, 0, 1, 0.5]]))
    check(message is not None and "multiplicity" in message, f"multiplicity < 1 is rejected ({message})")
    message = raises(lambda: add([["A", 0.2, 0, 1, 2], ["A", 0.2, 1, 2], ["A", 0.2, 3, 2, 2]]))
    check(message is not None and "u side" in message, f"blocks replicating each other are rejected ({message})")
    message = raises(lambda: add([["A", 0.5, 0, 1, 2]], {0: "G"}))
    check(message is not None and "grafted" in message, f"copies at a grafted end are rejected ({message})")

    complex_w = {"A": w["A"] + 0.3j * w["B"], "B": w["B"] - 0.3j * w["A"]}
    explicit_a2b2 = [(1.0, [["A", 0.5, 0, 1], ["A", 0.5, 0, 2], ["B", 0.5, 0, 3], ["B", 0.5, 0, 4]])]

    def complex_solve(polymers, platform, chain_model):
        solver = make_solver(platform, chain_model, field_type="complex")
        for volume_fraction, blocks in polymers:
            solver.add_polymer(volume_fraction, blocks)
        solver.compute_propagators({key: value.copy() for key, value in complex_w.items()})
        solver.compute_concentrations()
        return solver

    for platform_c in platforms:
        for chain_model in ["continuous", "discrete"]:
            tag = f"{platform_c}/{chain_model}"
            message = raises(lambda: complex_solve(star(2.5), platform_c, chain_model))
            check(message is not None and "non-integer" in message, f"{tag}: complex fields reject a real power")
            c = complex_solve(star(2), platform_c, chain_model)
            e = complex_solve(explicit_a2b2, platform_c, chain_model)
            dq = abs(c.get_partition_function(0) / e.get_partition_function(0) - 1.0)
            dphi = max(max_abs(c.get_concentration(k), e.get_concentration(k)) for k in complex_w)
            check(dq < 1e-13 and dphi < 1e-12,
                  f"{tag}: complex fields, multiplicity 2 against explicit arms (|dQ/Q| {dq:.1e}, |dphi| {dphi:.1e})")

    # Crank-Nicolson turns the propagator negative in the cell next to an absorbing wall
    def walls_star(n, platform):
        solver = PropagatorSolver(nx=[256], lx=[2.0], ds=0.01, bond_lengths={"A": 1.0, "B": 1.0},
                                  bc=["absorbing", "absorbing"], chain_model="continuous",
                                  numerical_method="cn-adi2", platform=platform)
        solver.add_polymer(1.0, star(n)[0][1])
        w_a = 6.0 * np.cos(2 * np.pi * (np.arange(256) + 0.5) / 256)
        solver.compute_propagators({"A": w_a, "B": -w_a})
        return solver

    for platform_w in platforms:
        message = raises(lambda: walls_star(5.6, platform_w))
        check(message is not None and "multiplicity" in message and "negative" in message,
              f"{platform_w}: a real power of a negative propagator stops with an error ({message})")
        check(raises(lambda: walls_star(5, platform_w)) is None,
              f"{platform_w}: the same chain with an integer multiplicity runs")
    # The message gives every digit, so a multiplicity next to an integer does not print as one
    message = raises(lambda: walls_star(5.0000001, platforms[0]))
    check(message is not None and "multiplicity of 5.0000001:" in message,
          f"the error names a near-integer multiplicity in full ({message})")

    params = {"nx": [16], "lx": [2.0], "ds": 0.05, "chain_model": "continuous",
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 10.0},
              "polymers": [{"volume_fraction": 1.0, "blocks": [
                  {"type": "A", "length": 0.5, "v": 0, "u": 1, "multiplicity": 2.5},
                  {"type": "B", "length": 0.5, "v": 0, "u": 2, "multiplicity": 2.5}]}],
              "langevin": {"nbar": 1000, "dt": 0.1, "max_step": 1},
              "recording": {"dir": "unused", "checkpoint_interval": 1, "sample_interval": 1,
                            "statistics_save_interval": 1},
              "verbose_level": 1}
    # The backends take integer powers up to 10^6 only, so CL-FTS refuses larger integers too
    for multiplicity in [2.5, 2000000]:
        for block in params["polymers"][0]["blocks"]:
            block["multiplicity"] = multiplicity
        try:
            validate_lfts_params(params, complex_fields=True)
            message = None
        except ValidationError as exc:
            message = str(exc)
        check(message is not None and "multiplicity" in message,
              f"CL-FTS validation rejects the multiplicity {multiplicity} ({message})")

    try:
        polymer_block_copies({"blocks": [{"type": "A", "length": 0.5, "v": 0, "u": 1, "multiplicity": 2},
                                         {"type": "B", "length": 0.5}]})
        error = None
    except Exception as exc:  # a bare KeyError would mean the numbering is not checked
        error = exc
    check(isinstance(error, ValidationError) and "Index v" in str(error),
          f"blocks numbered only in part give a ValidationError ({error!r})")

    # Without aggregation the two copied ends of the pom-pom share one computation block, which
    # holds the copies of both; each block takes the share of its own copies
    pom_pom = solve(ARCHITECTURES[4][1], w, platform, "continuous", aggregate=False)["solver"]
    shared = sorted(block.n_repeated for block in pom_pom.propagator_optimizer.get_computation_blocks().values())
    check(6.0 in shared, f"the copied ends of the pom-pom share one computation block (n_repeated {shared})")

    blocks = ARCHITECTURES[2][1][0][1]
    solver = make_solver(platform, "continuous")
    solver.add_polymer(1.0, blocks)
    native = [block.n_copies for block in solver.get_polymer(0).get_blocks()]
    python = polymer_block_copies({"blocks": [dict(zip(["type", "length", "v", "u", "multiplicity"], block))
                                              for block in blocks]})
    check(list(python) == native and native == [2.0, 6.0, 1.0],
          f"copies of the nested blocks: Python {list(python)}, native {native}")
    alpha = solver.get_polymer(0).get_alpha()
    check(abs(alpha - (2 * 0.2 + 6 * 0.2 + 0.1)) < 1e-14, f"alpha counts all copies ({alpha})")
    n_total = solver.get_polymer(0).get_n_segment_total()
    check(n_total == 2 * 4 + 6 * 4 + 2, f"the segment total counts all copies ({n_total})")
    solver = make_solver(platform, "continuous")
    solver.add_polymer(1.0, star(5.6)[0][1])
    n_total = solver.get_polymer(0).get_n_segment_total()
    check(abs(n_total - 5.6 * 20) < 1e-12, f"the segment total of a real multiplicity ({n_total})")


def check_scft(platform):
    print("4. SCFT with multiplicities against explicit arms")
    from polymerfts import scft
    star_compact = [{"type": "A", "length": 0.2, "v": 0, "u": 1, "multiplicity": 3},
                    {"type": "B", "length": 0.3, "v": 0, "u": 2, "multiplicity": 2}]
    star_explicit = ([{"type": "A", "length": 0.2, "v": 0, "u": u} for u in (1, 2, 3)]
                     + [{"type": "B", "length": 0.3, "v": 0, "u": u} for u in (4, 5)])
    diblock = [{"type": "A", "length": 0.4, "v": 0, "u": 1}, {"type": "B", "length": 0.6, "v": 1, "u": 2}]
    x = (np.arange(32) + 0.5) / 32
    results = []
    for star_blocks in (star_compact, star_explicit):
        params = {"platform": platform, "nx": [32], "lx": [1.5], "relax_cell": False, "ds": 0.02, "chain_model": "continuous",
                  "segment_lengths": {"A": 1.0, "B": 1.2}, "chi_n": {"A,B": 16.0},
                  "polymers": [{"volume_fraction": 0.6, "blocks": star_blocks},
                               {"volume_fraction": 0.4, "blocks": diblock}],
                  "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
                  "max_iter": 2000, "tolerance": 1e-9}
        calculation = scft.SCFT(params=params)
        calculation.run(initial_fields={"A": 2.0 * np.cos(2 * np.pi * x), "B": -2.0 * np.cos(2 * np.pi * x)})
        results.append(calculation)
    c, e = results
    df = abs(c.free_energy - e.free_energy)
    dphi = max(max_abs(c.phi[k], e.phi[k]) for k in ("A", "B"))
    dmu = max_abs(c.chemical_potentials, e.chemical_potentials)
    check(c.error_level < 1e-9 and df < 1e-10 and dphi < 1e-9 and dmu < 1e-9,
          f"{platform}: |dF| {df:.1e}, |dphi| {dphi:.1e}, |dmu| {dmu:.1e} (F = {c.free_energy:.8f})")


def check_lfts(platform):
    print("5. L-FTS with multiplicities")
    import contextlib
    import io
    import os
    import tempfile
    from scipy.io import loadmat
    from polymerfts import lfts

    rng = np.random.default_rng(5)
    w0 = {"A": rng.normal(0.0, 0.5, 512), "B": rng.normal(0.0, 0.5, 512)}

    def run(root, name, blocks):
        params = {"platform": platform, "nx": [8, 8, 8], "lx": [2.0, 2.0, 2.0], "chain_model": "continuous",
                  "ds": 0.05, "segment_lengths": {"A": 1.0, "B": 1.2}, "chi_n": {"A,B": 15.0},
                  "polymers": [{"volume_fraction": 1.0, "blocks": blocks}],
                  "langevin": {"nbar": 10000, "dt": 8.0, "max_step": 6},
                  "recording": {"dir": os.path.join(root, name), "checkpoint_interval": 6, "sample_interval": 1,
                                "statistics_save_interval": 6},
                  "saddle": {"max_iter": 200, "tolerance": 1e-6},
                  "compressor": {"name": "am", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01,
                                 "mix_init": 0.01},
                  "verbose_level": 1}
        with contextlib.redirect_stdout(io.StringIO()):
            calculation = lfts.LFTS(params, random_seed=1)
            calculation.run(initial_fields={key: value.copy() for key, value in w0.items()})
        return loadmat(os.path.join(root, name, "fields_000006.mat"))

    def star_n(n):
        return [{"type": "A", "length": 0.1, "v": 0, "u": 1, "multiplicity": n},
                {"type": "B", "length": 0.1, "v": 0, "u": 2, "multiplicity": n}]

    star_compact = [{"type": "A", "length": 0.2, "v": 0, "u": 1, "multiplicity": 3},
                    {"type": "B", "length": 0.3, "v": 0, "u": 2, "multiplicity": 2}]
    star_explicit = ([{"type": "A", "length": 0.2, "v": 0, "u": u} for u in (1, 2, 3)]
                     + [{"type": "B", "length": 0.3, "v": 0, "u": u} for u in (4, 5)])
    with tempfile.TemporaryDirectory() as root:
        c, e = run(root, "compact", star_compact), run(root, "explicit", star_explicit)
        dw = max(max_abs(c["w_" + key], e["w_" + key]) for key in "AB")
        moved = max_abs(c["w_A"].flatten(), w0["A"])
        check(dw < 1e-9 and moved > 0.1,
              f"{platform}: A3B2 star with multiplicities against explicit arms (|dw| {dw:.1e}, "
              f"fields moved by {moved:.2f})")

        integer, joined = run(root, "n5", star_n(5)), run(root, "n5_eps", star_n(5 + 1e-9))
        dw = max(max_abs(integer["w_" + key], joined["w_" + key]) for key in "AB")
        check(dw < 1e-6, f"{platform}: n = 5 + 1e-9 (real powers) follows n = 5 (|dw| {dw:.1e})")

        real = run(root, "n5.6", star_n(5.6))
        finite = all(np.all(np.isfinite(real[key])) for key in ("w_A", "w_B", "phi_A", "phi_B"))
        total = float(np.mean(real["phi_A"] + real["phi_B"]))
        drawn = os.path.exists(os.path.join(root, "n5.6", "polymer_0.png"))
        check(finite and abs(total - 1.0) < 1e-12 and drawn,
              f"{platform}: n = 5.6 runs (mean phi_A + phi_B - 1 = {total - 1.0:.1e}, drawing {drawn})")


def main():
    rng = np.random.default_rng(11)
    w = {"A": smooth_field(rng), "B": smooth_field(rng)}
    platforms = _core.PlatformSelector.avail_platforms()
    check_integer_multiplicities(w, platforms)
    check_real_multiplicities(w, platforms)
    check_input_errors(w, platforms)
    check_scft(platforms[0])
    check_lfts(platforms[0])

    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        return 1
    if cuda_part_missing():
        return 1
    print("All block-multiplicity checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
