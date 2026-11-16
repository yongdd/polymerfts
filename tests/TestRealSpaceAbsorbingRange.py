#!/usr/bin/env python3
"""Absorbing rs bonds retain small positive modes and their cell derivatives.

The analytic ground mode tests the original narrow-box failure without relying
on another solver. Mixed walls, strided passes, half bonds, complex fields and
operator replacement also exercise the folded kernels on every backend, and a
signed image sum checks the leading retained modes, their lattice aliases and the
Nyquist mode just inside the folded regime.
"""

import numpy as np
from scipy.optimize import brentq

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def solver(platform, reduced, nx, lx, bc, field_type="real", blocks=None, method="rs"):
    result = PropagatorSolver(nx, lx, 0.1, {"A": 1.0, "B": 1.2}, bc=bc,
                             chain_model="discrete", numerical_method=method,
                             platform=platform, reduce_memory=reduced, field_type=field_type)
    result.add_polymer(1.0, blocks or [["A", 0.2, 0, 1]], {0: "G"})
    return result


def evaluate(calc, fields, graft):
    calc.compute_propagators(fields, q_init={"G": graft})
    calc.compute_concentrations()
    calc.compute_stress()
    assert calc.check_total_partition(), calc.partition_errors()
    phi = np.array([calc.get_concentration(name) for name in fields])
    np.testing.assert_allclose(calc.mean(phi.sum(axis=0)), 1.0, rtol=0, atol=3e-12)
    return calc.get_partition_function(0), phi, np.asarray(calc.get_stress())


def same_memory_modes(full, reduced):
    for first, second in zip(full, reduced):
        assert np.asarray(first).tobytes() == np.asarray(second).tobytes()


def ground_mode(platform, nx, lx, bc, field_type):
    axes, k_squared = [], []
    alias = 1.0
    for d, (n, length) in enumerate(zip(nx, lx)):
        x = (np.arange(n) + 0.5) / n
        walls = bc[2*d:2*d+2]
        if walls == ["absorbing", "absorbing"]:
            axes.append(np.sin(np.pi*x))
            k_squared.append((np.pi/length)**2)
            # At n=1 the sampled Gaussian has two equal Nyquist aliases.
            # Here sigma/dx > 3.6, so its variance correction is below 1e-100.
            if n == 1:
                alias *= 2.0
        elif walls == ["absorbing", "reflecting"]:
            axes.append(np.sin(0.5*np.pi*x))
            k_squared.append((0.5*np.pi/length)**2)
        elif walls == ["reflecting", "absorbing"]:
            axes.append(np.cos(0.5*np.pi*x))
            k_squared.append((0.5*np.pi/length)**2)
        else:
            axes.append(np.ones(n))
            k_squared.append(0.0)
    graft = np.prod(np.array(np.meshgrid(*axes, indexing="ij")), axis=0).ravel()
    field = 0.0 if field_type == "real" else 0.3 + 0.7j
    if field_type == "complex":
        graft = graft * (1.0 + 0.4j)
    fields = {"A": np.full(graft.size, field)}
    expected = graft * alias * np.exp(-0.1*sum(k_squared)/6.0 - 0.2*field)
    stress = np.zeros(6)
    stress[:len(nx)] = -0.1*np.asarray(k_squared)/(3.0*np.asarray(lx)*0.2)
    results = []
    for reduced in (False, True):
        calc = solver(platform, reduced, nx, lx, bc, field_type)
        state = evaluate(calc, fields, graft)
        np.testing.assert_allclose(calc.get_propagator(0, 0, 1, 2), expected, rtol=3e-12, atol=0)
        np.testing.assert_allclose(state[0], np.mean(expected), rtol=3e-12, atol=0)
        np.testing.assert_allclose(state[2], stress, rtol=3e-11, atol=2e-11)
        assert np.min(state[1].real) > 0.0
        np.testing.assert_allclose(state[1].imag, 0.0, rtol=0, atol=3e-12)
        results.append(state)
    same_memory_modes(*results)


def branched_half_bonds(platform, field_type):
    # Both different-monomer junctions and branching require half bonds.
    blocks = [["A", 0.1, 0, 1], ["B", 0.2, 1, 2], ["A", 0.2, 1, 3]]
    n, length = 16, 0.08
    rng = np.random.default_rng(13019)
    fields = {name: rng.normal(0.0, 5.0, n) for name in "AB"}
    graft = rng.uniform(0.4, 1.6, n)
    if field_type == "complex":
        fields = {name: value + 1j*rng.normal(0.0, 1.0, n) for name, value in fields.items()}
        graft = graft * (1.0 + 0.2j)
    states = []
    for reduced in (False, True):
        calc = solver(platform, reduced, [n], [length], ["absorbing"]*2, field_type, blocks)
        state = evaluate(calc, fields, graft)
        reference = evaluate(solver(platform, reduced, [n], [length], ["absorbing"]*2,
                                    field_type, blocks, method="ps"), fields, graft)
        for actual, expected in zip(state, reference):
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=0)
        if field_type == "real":
            assert np.min(state[1]) > 0.0
        # Independent derivative of the canonical -log(Q)/alpha term.
        h = 1e-6
        partitions = []
        for sign in (1, -1):
            calc.set_lx([length*(1.0+sign*h)])
            calc.compute_propagators(fields, q_init={"G": graft})
            partitions.append(calc.get_partition_function(0))
        alpha = calc.get_polymer(0).get_alpha()
        derivative = -(np.log(partitions[0])-np.log(partitions[1]))/(2*h*length*alpha)
        np.testing.assert_allclose(state[2][0], derivative, rtol=2e-7, atol=0)
        # Switch to ordinary kernels, then restore the folded kernels.
        calc.set_lx([1.0])
        evaluate(calc, fields, graft)
        calc.set_lx([length])
        restored = evaluate(calc, fields, graft)
        same_memory_modes(state, restored)
        states.append(state)
    same_memory_modes(*states)


def coarse_sampled_bond(platform):
    # On a single cell the full Gaussian aliases matter. Calibrate the width
    # independently, then sum its alternating images directly in a regime
    # where the answer is O(0.1) and that summation is well conditioned.
    distances = np.arange(-128, 129)
    variance = 0.1/3.0

    def eigenvalue(length, theta):
        def weights(width):
            values = np.exp(-0.5*(distances*length/width)**2)
            return values/values.sum()
        width = brentq(lambda value: np.sum(weights(value)*(distances*length)**2)-variance,
                       0.01*np.sqrt(variance), np.sqrt(variance)+length,
                       xtol=1e-15, rtol=1e-14)
        return np.sum(weights(width)*np.cos(theta*distances))

    for length, bc, theta in ((0.3, ["absorbing"]*2, np.pi),
                              (0.18, ["absorbing", "reflecting"], np.pi/2),
                              (0.18, ["reflecting", "absorbing"], np.pi/2)):
        expected = eigenvalue(length, theta)
        h = 1e-5
        derivative = -(np.log(eigenvalue(length*(1+h), theta))
                       - np.log(eigenvalue(length*(1-h), theta)))/(2*h*length*0.2)
        states = []
        for reduced in (False, True):
            calc = solver(platform, reduced, [1], [length], bc)
            state = evaluate(calc, {"A": np.zeros(1)}, np.ones(1))
            np.testing.assert_allclose(state[0], expected, rtol=2e-12, atol=0)
            np.testing.assert_allclose(state[2][0], derivative, rtol=2e-8, atol=0)
            states.append(state)
        same_memory_modes(*states)


def image_sum_bond(n, length, walls, variance):
    """One bond on a cell-centered axis as a signed sum of lattice images.

    The sampled Gaussian is calibrated to the bond variance as in
    coarse_sampled_bond. An absorbing wall mirrors with a minus sign, and one
    absorbing wall with one reflecting wall makes the extension antiperiodic.
    """
    dx = length/n
    reach = 64*n
    distances = np.arange(-reach, reach + 1)

    def weights(width):
        values = np.exp(-0.5*(distances*dx/width)**2)
        return values/values.sum()
    width = brentq(lambda value: np.sum(weights(value)*(distances*dx)**2)-variance,
                   0.01*np.sqrt(variance), np.sqrt(variance)+dx, xtol=1e-15, rtol=1e-14)
    w = weights(width)
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    low, high = walls
    mirror = -1.0 if low == "absorbing" else 1.0
    bond = np.zeros((n, n))
    for k in range(-31, 32):
        period = 1.0 if low == high else (-1.0)**k
        bond += period*(w[i - j - 2*k*n + reach] + mirror*w[i + j + 1 - 2*k*n + reach])
    return bond


def higher_folded_modes(platform, field_type):
    # Just inside the folded regime ten modes (two absorbing walls) or five (one)
    # are retained. A graft and fields that are not eigenfunctions excite all of
    # them; their eigenvalues fall off as exp(-m^2), so at rtol 1e-11 the comparison
    # resolves the modes above about 1e-11 of the first: five with two absorbing
    # walls, three with one. The stress needs the derivatives of the same modes.
    # With two absorbing walls on two or three cells the bond is about one cell
    # wide: the higher modes carry lattice aliases, and the last one is the Nyquist
    # mode, which has two equal aliases and its own norm. One absorbing wall needs a
    # bond twice as many cells wide; there the aliases change the propagator by at
    # most about 1e-10, near the tolerance.
    ds = 0.1
    rng = np.random.default_rng(29417)
    for n, factor, bc in ((16, 1.0, ["absorbing"]*2), (16, 0.5, ["absorbing", "reflecting"]),
                          (16, 0.5, ["reflecting", "absorbing"]), (2, 1.0, ["absorbing"]*2),
                          (3, 1.0, ["absorbing"]*2), (2, 0.5, ["absorbing", "reflecting"]),
                          (2, 0.5, ["reflecting", "absorbing"])):
        length = factor*np.pi*np.sqrt(ds/6.0)*(1-1e-3)
        field = rng.normal(0.0, 5.0, n)
        graft = rng.uniform(0.4, 1.6, n)
        if field_type == "complex":
            field = field + 1j*rng.normal(0.0, 1.0, n)
            graft = graft * (1.0 + 0.2j)
        boltzmann = np.exp(-ds*field)
        expected = boltzmann*(image_sum_bond(n, length, bc, ds/3.0) @ (boltzmann*graft))
        states = []
        for reduced in (False, True):
            calc = solver(platform, reduced, [n], [length], bc, field_type)
            state = evaluate(calc, {"A": field}, graft)
            np.testing.assert_allclose(calc.get_propagator(0, 0, 1, 2), expected, rtol=1e-11, atol=0)
            np.testing.assert_allclose(state[0], np.mean(expected), rtol=1e-11, atol=0)
            h = 1e-6
            partitions = []
            for sign in (1, -1):
                calc.set_lx([length*(1.0+sign*h)])
                calc.compute_propagators({"A": field}, q_init={"G": graft})
                partitions.append(calc.get_partition_function(0))
            alpha = calc.get_polymer(0).get_alpha()
            derivative = -(np.log(partitions[0])-np.log(partitions[1]))/(2*h*length*alpha)
            np.testing.assert_allclose(state[2][0], derivative, rtol=2e-7, atol=0)
            states.append(state)
        same_memory_modes(*states)


def main():
    platforms = PlatformSelector.avail_platforms()
    cases = [
        ([16], [0.05], ["absorbing"]*2),
        ([16], [0.2], ["absorbing"]*2),
        ([16], [np.pi*np.sqrt(0.1/6.0)*(1-1e-6)], ["absorbing"]*2),
        ([16], [np.pi*np.sqrt(0.1/6.0)*(1+1e-6)], ["absorbing"]*2),
        ([1], [0.05], ["absorbing"]*2),
        ([16], [0.025], ["absorbing", "reflecting"]),
        ([16], [0.025], ["reflecting", "absorbing"]),
        ([8, 12], [0.05, 0.08], ["absorbing"]*4),
        ([6, 8, 10], [0.05, 0.04, 1.0],
         ["absorbing", "absorbing", "reflecting", "absorbing", "periodic", "periodic"]),
    ]
    for platform in platforms:
        for field_type in ("real", "complex"):
            for nx, lx, bc in cases:
                ground_mode(platform, nx, lx, bc, field_type)
            branched_half_bonds(platform, field_type)
            higher_folded_modes(platform, field_type)
        coarse_sampled_bond(platform)
        print(f"PASS {platform}: absorbing modes, retained higher modes, half bonds, stress and memory modes", flush=True)
    if cuda_part_missing():
        return 1
    if not platforms:
        print("No platform available: skipped")
        return 77
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
