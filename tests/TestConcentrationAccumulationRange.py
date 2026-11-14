#!/usr/bin/env python3
"""Contour sums retain range until canonical/GCE normalization on every backend.

The -706 field has finite propagators and Q, but its 100-term raw block
integral exceeds double. Positive 720/740 fields guard against fixing that by
downscaling every term, which would lose the tiny-Q densities. At 740 the
partition has only about 85 subnormal ulps; an extended-precision contour quadrature
checks accumulation accurately despite the partition's intrinsic rounding.
"""

from decimal import Decimal, localcontext

import numpy as np

from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing


def make_solver(platform, model, reduced, complex_fields=False, population=1.0):
    solver = PropagatorSolver(
        nx=[8], lx=[1.0], ds=0.01, bond_lengths={"A": 1.0},
        bc=["periodic"] * 2, chain_model=model,
        numerical_method="rk2" if model == "continuous" else "ps",
        platform=platform, reduce_memory=reduced,
        aggregate_propagator_computation=False,
        field_type="complex" if complex_fields else "real")
    solver.add_polymer(population, [["A", 1.0, 0, 1]])
    return solver


def solve(platform, model, reduced, offset, complex_fields=False, population=1.0):
    solver = make_solver(platform, model, reduced, complex_fields, population)
    field = offset + (0.7j if complex_fields else 0.0)
    solver.compute_propagators({"A": np.full(8, field)})
    partition = solver.get_partition_function(0)
    tolerance = 0.1 if offset == 740.0 else 2e-9
    assert np.isfinite(partition) and partition != 0.0, partition
    assert abs(partition / np.exp(-field) - 1.0) < tolerance, partition
    assert solver.check_total_partition(tolerance), "inconsistent contour partitions"
    solver.compute_concentrations()
    phi = solver.get_concentration("A")
    block = solver._propagator_computation.get_block_concentration(0)
    species = solver._propagator_computation.get_total_concentration(0, "A")
    for density in (phi, block, species):
        assert np.all(np.isfinite(density)), density
        assert np.max(np.abs(density - population)) < tolerance, density
    if offset == 740.0:
        # Compare the accumulated density itself at full precision, without
        # confusing rounding of Q with lost contour contributions.
        with localcontext() as context:
            context.prec = 60
            reference = [Decimal(0) for _ in range(8)]
            divisor = Decimal.from_float(float(np.exp(-offset * 0.01)))
            first = 0 if model == "continuous" else 1
            for n in range(first, 101):
                left_step = 100 - n if model == "continuous" else 101 - n
                left = solver.get_propagator(0, 0, 1, left_step)
                right = solver.get_propagator(0, 1, 0, n)
                weight = 1.0 / 3.0 if n in (0, 100) else (4.0 / 3.0 if n % 2 else 2.0 / 3.0)
                for i in range(8):
                    term = Decimal.from_float(float(left[i])) * Decimal.from_float(float(right[i]))
                    reference[i] += term * Decimal.from_float(weight) if model == "continuous" else term / divisor
            scale = Decimal.from_float(0.01 * population) / Decimal.from_float(float(partition))
            reference = np.array([float(value * scale) for value in reference])
        assert np.max(np.abs(phi - reference)) < 2e-12, (phi, reference)
    assert np.array_equal(solver.get_concentration_gce(0.0, 0, "A"), np.zeros(8))
    gce = None
    if offset == -706.0:
        gce = solver.get_concentration_gce(np.exp(-706.0), 0, "A")
        expected = np.exp(-0.7j) if complex_fields else 1.0
        assert np.max(np.abs(gce - expected)) < 2e-11, gce
    # Reset both mantissas and exponents on the next concentration computation.
    solver.compute_concentrations()
    assert np.array_equal(phi, solver.get_concentration("A")), "accumulator was not reset"
    return partition, phi, block, gce


def zero_partition_gce(platform, model, reduced, complex_fields):
    """Spatial cancellation makes Q exactly zero without removing its GCE density."""
    solver = PropagatorSolver(
        nx=[8], lx=[1.0], ds=0.01, bond_lengths={"A": 0.01},
        bc=["periodic"] * 2, chain_model=model,
        numerical_method="rk2" if model == "continuous" else "ps",
        platform=platform, reduce_memory=reduced,
        aggregate_propagator_computation=False,
        field_type="complex" if complex_fields else "real")
    solver.add_polymer(0.0, [["A", 1.0, 0, 1]], {0: "G"})
    sign = np.tile([1.0, -1.0], 4) * (1j if complex_fields else 1.0)
    solver.compute_propagators({"A": np.full(8, -706.0)}, {"G": sign})
    assert solver.get_partition_function(0) == 0.0
    assert solver.check_total_partition()
    solver.compute_concentrations()
    assert np.array_equal(solver.get_concentration("A"), np.zeros(8))
    density = solver.get_concentration_gce(np.exp(-706.0), 0, "A")
    decay = 0.01**2 / 6.0 * (2.0 * np.pi * 4.0)**2
    if model == "continuous":
        weights = np.full(101, 2.0 / 3.0)
        weights[1::2] = 4.0 / 3.0
        weights[[0, -1]] = 1.0 / 3.0
        amplitude = 0.01 * np.sum(weights * np.exp(-decay * 0.01 * np.arange(101)))
    else:
        amplitude = 0.01 * np.sum(np.exp(-decay * 0.01 * np.arange(100)))
    assert np.max(np.abs(density - amplitude * sign)) < 2e-11, density
    return density


def main():
    for platform in _core.PlatformSelector.avail_platforms():
        for model in ("continuous", "discrete"):
            for offset, complex_fields, population in (
                (-706.0, False, 1.0), (-706.0, True, 1.0),
                (720.0, False, 1.0), (740.0, False, 1.0),
                (-706.0, False, 0.0), (-706.0, True, 0.0),
            ):
                full = solve(platform, model, False, offset, complex_fields, population)
                reduced = solve(platform, model, True, offset, complex_fields, population)
                assert all(np.array_equal(a, b) for a, b in zip(full, reduced)), (
                    platform, model, offset, complex_fields, population, "memory modes differ")
                print("PASS", platform, model, offset, "complex" if complex_fields else "real", population)
            for complex_fields in (False, True):
                full = zero_partition_gce(platform, model, False, complex_fields)
                reduced = zero_partition_gce(platform, model, True, complex_fields)
                assert np.array_equal(full, reduced), "zero-Q GCE memory modes differ"
                print("PASS", platform, model, "zero-Q GCE", "complex" if complex_fields else "real")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
