#!/usr/bin/env python3
"""Absent solvents contribute zero density while retaining their actual partition.

A complex solvent can have Q = 0 by cancellation with finite fields. Its
zero-population concentration must remain exactly zero, as in the no-solvent
control. Real Boltzmann underflow exercises the same case for real fields.
Occupied solvents retain their normalization, including the Q = 0 singularity.
"""

import sys

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def solve(platform, reduced, model, field_type, solvent_field,
          volume_fraction=0.0, include_solvent=True):
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([4], [4.0])
    molecules = factory.create_molecules_information(model, 1.0, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(1.0 - volume_fraction, [["B", 1.0, 0, 1]])
    if include_solvent:
        molecules.add_solvent(volume_fraction, "A")
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(
        box, molecules, optimizer, "ps" if model == "discrete" else "rqm4")
    dtype = np.complex128 if field_type == "complex" else np.float64
    solver.compute_propagators({"A": np.asarray(solvent_field, dtype=dtype),
                               "B": np.zeros(4, dtype=dtype)})
    solver.compute_concentrations()
    phi_a = solver.get_total_concentration("A")
    phi_b = solver.get_total_concentration("B")
    np.testing.assert_allclose(solver.get_total_partition(0), 1.0, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(phi_b, 1.0 - volume_fraction, rtol=0.0, atol=2e-14)
    assert solver.check_total_partition()  # The polymer partition remains regular.
    partition = solver.get_solvent_partition(0) if include_solvent else None
    if include_solvent:
        np.testing.assert_array_equal(solver.get_solvent_concentration(0), phi_a)
    return phi_a, phi_b, partition, box.mean(phi_a + phi_b)


def test_zero_partition(platform, reduced, model, field_type):
    field = (np.array([0.0, 0.0, 1j * np.pi, -1j * np.pi])
             if field_type == "complex" else np.full(4, 1000.0))
    control = solve(platform, reduced, model, field_type, field, include_solvent=False)
    absent = solve(platform, reduced, model, field_type, field)
    assert absent[2] == 0.0
    for actual, expected in zip(absent[:2], control[:2]):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(absent[0], np.zeros(4))
    np.testing.assert_allclose(absent[3], 1.0, rtol=0.0, atol=2e-14)

    # The absent-species branch must not hide an occupied species' singularity.
    occupied = solve(platform, reduced, model, field_type, field, volume_fraction=0.25)
    assert occupied[2] == 0.0
    assert not np.isfinite(occupied[0]).all()
    return absent


def test_finite_partition(platform, reduced, model, field_type):
    field = np.array([-0.7, 0.1, 0.4, 0.9])
    if field_type == "complex":
        field = field + 1j * np.array([0.2, -0.1, 0.3, -0.4])
    boltzmann = np.exp(-field)
    expected_partition = np.mean(boltzmann)
    results = []
    for fraction in (0.0, 0.25):
        result = solve(platform, reduced, model, field_type, field, volume_fraction=fraction)
        np.testing.assert_allclose(result[2], expected_partition, rtol=2e-14, atol=2e-14)
        if fraction == 0.0:
            np.testing.assert_array_equal(result[0], np.zeros(4))
        else:
            np.testing.assert_allclose(result[0], fraction * boltzmann / expected_partition,
                                       rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result[3], 1.0, rtol=0.0, atol=2e-14)
        results.append(result)
    return results


def test_nonfinite_partition(platform, reduced, model, field_type):
    # The concentration must be written as zero, not multiplied by zero: the
    # latter would turn overflowing Boltzmann factors into NaNs.
    result = solve(platform, reduced, model, field_type, np.full(4, -1000.0))
    assert not np.isfinite(result[2])
    np.testing.assert_array_equal(result[0], np.zeros(4))
    np.testing.assert_allclose(result[3], 1.0, rtol=0.0, atol=2e-14)


def main():
    for platform in PlatformSelector.avail_platforms():
        for model in ("continuous", "discrete"):
            for field_type in ("real", "complex"):
                modes = []
                for reduced in (False, True):
                    zero = test_zero_partition(platform, reduced, model, field_type)
                    finite = test_finite_partition(platform, reduced, model, field_type)
                    test_nonfinite_partition(platform, reduced, model, field_type)
                    modes.append([zero, *finite])
                for full, reduced in zip(*modes):
                    for actual, expected in zip(full, reduced):
                        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes(), (
                            platform, model, field_type, "checkpoint mismatch")
        print(f"Absent solvent: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
