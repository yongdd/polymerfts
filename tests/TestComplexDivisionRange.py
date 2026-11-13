#!/usr/bin/env python3
"""Discrete complex quotients retain finite weights under uniform field shifts.

For one bead, q = exp(-w), Q = mean(mask*q), and phi = mask*q/Q.
At w = +/-250, q, q*q and Q are representable, but forming q*q*conj(q)
inside a complex division overflows or underflows. Spatially varying complex
fields also exercise both components of the scaled division.
"""

import sys

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def solve(platform, reduced, method, field_type, profile, offset, mask):
    dtype = np.complex128 if field_type == "complex" else np.float64
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([4], [4.0], mask=mask)
    molecules = factory.create_molecules_information("discrete", 1.0, {"A": 1.0})
    molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(box, molecules, optimizer, method)

    field = np.asarray(profile + offset, dtype=dtype)
    accessible = np.ones(4, dtype=bool) if mask is None else mask != 0.0
    # Excluded zero divisors must still be skipped by the masked kernel.
    field[~accessible] = 1000.0
    solver.compute_propagators({"A": field})
    q = solver.get_chain_propagator(0, 0, 1, 1)
    assert np.isfinite(q).all()
    assert np.isfinite(q * q).all()
    assert np.all(np.abs((q * q)[accessible]) > 0.0)

    partition = solver.get_total_partition(0)
    assert np.isfinite(partition) and partition != 0.0
    assert solver.check_total_partition()
    errors = np.asarray(solver.compute_partition_errors())
    assert np.isfinite(errors).all()
    solver.compute_concentrations()
    phi = solver.get_total_concentration("A")
    block = solver.get_block_concentration(0)[0]
    gce = solver.get_total_concentration_gce(0.4, 0, "A")
    assert all(np.isfinite(values).all() for values in (phi, block, gce))

    # Remove the known gauge factor before comparing Q: an absolute tolerance
    # would otherwise accept an erroneous zero when Q is around 1e-109.
    scale = np.exp(-offset)
    weight = np.exp(-np.asarray(profile, dtype=dtype))
    weight[~accessible] = 0.0
    expected_partition = np.mean(weight)
    expected_phi = weight / expected_partition
    np.testing.assert_allclose(partition / scale, expected_partition, rtol=2e-13, atol=0.0)
    np.testing.assert_allclose(q / scale, weight, rtol=2e-13, atol=0.0)
    for concentration in (phi, block):
        np.testing.assert_allclose(concentration, expected_phi, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(box.mean(concentration), 1.0, rtol=0.0, atol=2e-13)
        np.testing.assert_array_equal(concentration[~accessible], np.zeros(np.count_nonzero(~accessible)))
    np.testing.assert_allclose(gce / scale, 0.4 * weight, rtol=2e-13, atol=0.0)
    np.testing.assert_allclose(box.mean(gce) / scale, 0.4 * expected_partition, rtol=2e-13, atol=0.0)

    return {
        "partition": np.asarray(partition),
        "q": q,
        "phi": phi,
        "block": block,
        "gce": gce,
        "errors": errors,
    }


def test_configuration(platform, method, field_type, profile, mask):
    reference = None
    for offset in (0.0, -10.0, 10.0, -250.0, 250.0):
        modes = [solve(platform, reduced, method, field_type, profile, offset, mask)
                 for reduced in (False, True)]
        for key in modes[0]:
            assert modes[0][key].tobytes() == modes[1][key].tobytes(), (
                platform, method, field_type, offset, key, "full/reduced mismatch")
        if reference is None:
            reference = modes[0]
        else:
            for key in ("phi", "block"):
                np.testing.assert_allclose(modes[0][key], reference[key], rtol=2e-13, atol=2e-14)


def main():
    masks = (None, np.ones(4), np.array([1.0, 1.0, 1.0, 0.0]))
    for platform in PlatformSelector.avail_platforms():
        for method in ("ps", "rs"):
            for field_type in ("real", "complex"):
                varying = np.array([-1.2, 0.7, 1.4, -0.4])
                if field_type == "complex":
                    varying = varying + 1j * np.array([0.3, 1.2, -2.1, 2.8])
                for profile in (np.zeros(4), varying):
                    for mask in masks:
                        test_configuration(platform, method, field_type, profile, mask)
        print(f"Complex division range: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
