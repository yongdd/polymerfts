#!/usr/bin/env python3
"""Shared-bead products retain representable Q and densities at large fields."""

import sys

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def solve(platform, reduced, method, field_type, offset, population):
    dtype = np.complex128 if field_type == "complex" else np.float64
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([4], [4.0])
    molecules = factory.create_molecules_information("discrete", 1.0, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(population, [["A", 1.0, 0, 1]])
    molecules.add_polymer(1.0 - population, [["B", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(box, molecules, optimizer, method)

    profile = np.array([-1.2, 0.7, 1.4, -0.4], dtype=dtype)
    if field_type == "complex":
        profile += 1j * np.array([0.3, 1.2, -2.1, 2.8])
    solver.compute_propagators({"A": profile + offset, "B": np.zeros(4, dtype=dtype)})
    q = solver.get_chain_propagator(0, 0, 1, 1)
    assert np.isfinite(q).all() and np.all(np.abs(q) > 0.0)
    # The operands and final quotient are representable, although the naive
    # intermediate product is not. This is distinct from a zero propagator.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = q * q
    if offset == 400.0:
        assert np.all(product == 0.0)
    elif offset == -400.0:
        assert not np.isfinite(product).all()

    weight = np.exp(-profile)
    scale = np.exp(-offset)
    partition = np.asarray(solver.get_total_partition(0))
    np.testing.assert_allclose(partition / scale, np.mean(weight), rtol=3e-13, atol=0.0)
    assert solver.check_total_partition()
    errors = np.asarray(solver.compute_partition_errors())
    assert np.isfinite(errors).all() and np.max(errors) < 1e-13
    solver.compute_concentrations()
    phi = solver.get_total_concentration("A")
    block = solver.get_block_concentration(0)[0]
    expected = population * weight / np.mean(weight)
    for concentration in (phi, block):
        np.testing.assert_allclose(concentration, expected, rtol=3e-13, atol=2e-14)
    total = phi + solver.get_total_concentration("B")
    np.testing.assert_allclose(box.mean(total), 1.0, rtol=0.0, atol=3e-13)

    # Opposite chemical-potential/field shifts cancel exactly in the analytic
    # density; a comparison against an order-one target cannot accept zero.
    gce = solver.get_total_concentration_gce(float(np.exp(offset)), 0, "A")
    np.testing.assert_allclose(gce, weight, rtol=3e-13, atol=0.0)
    zero = solver.get_total_concentration_gce(0.0, 0, "A")
    np.testing.assert_array_equal(zero, np.zeros(4))
    return {"q": q, "partition": partition, "errors": errors,
            "phi": phi, "block": block, "gce": gce, "zero": zero}


def check_absent_species(platform, reduced, method, field_type):
    """Zero fugacity stays zero even if an unused raw density is nonfinite."""
    dtype = np.complex128 if field_type == "complex" else np.float64
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([4], [4.0])
    molecules = factory.create_molecules_information("discrete", 1.0, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(0.0, [["A", 1.0, 0, 1]])
    molecules.add_polymer(1.0, [["B", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(box, molecules, optimizer, method)
    solver.compute_propagators({"A": np.full(4, 1000.0, dtype=dtype),
                                "B": np.zeros(4, dtype=dtype)})
    solver.compute_concentrations()
    np.testing.assert_array_equal(solver.get_total_concentration_gce(0.0, 0, "A"), np.zeros(4))
    np.testing.assert_allclose(solver.get_total_concentration("B"), 1.0, rtol=0.0, atol=2e-14)
    # The fast path must retain argument validation.
    for polymer, monomer in ((2, "A"), (0, "unknown")):
        try:
            solver.get_total_concentration_gce(0.0, polymer, monomer)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Zero fugacity bypassed polymer/monomer validation")


def main():
    for platform in PlatformSelector.avail_platforms():
        for method in ("ps", "rs"):
            for field_type in ("real", "complex"):
                for population in (0.0, 1.0):
                    reference = None
                    for offset in (0.0, 400.0, -400.0):
                        modes = [solve(platform, reduced, method, field_type, offset, population)
                                 for reduced in (False, True)]
                        for key in modes[0]:
                            assert modes[0][key].tobytes() == modes[1][key].tobytes(), (
                                platform, method, field_type, population, offset, key)
                        if reference is None:
                            reference = modes[0]
                        else:
                            for key in ("phi", "block", "gce"):
                                np.testing.assert_allclose(modes[0][key], reference[key],
                                                           rtol=3e-13, atol=2e-14)
                for reduced in (False, True):
                    check_absent_species(platform, reduced, method, field_type)
        print(f"Discrete product range: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
