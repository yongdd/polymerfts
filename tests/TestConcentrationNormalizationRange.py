#!/usr/bin/env python3
"""Finite canonical/GCE densities must not require a representable scalar prefactor."""

import sys

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def partition_ratio(value, reference):
    # NumPy's complex division can overflow its reciprocal for subnormal Q.
    # Scale both operands componentwise before forming this relative check.
    value, reference = complex(value), complex(reference)
    scale = max(abs(reference.real), abs(reference.imag))
    assert scale > 0.0
    return (complex(value.real / scale, value.imag / scale)
            / complex(reference.real / scale, reference.imag / scale))


def make_solver(platform, reduced, method, field_type):
    return PropagatorSolver(
        nx=[4], lx=[4.0], ds=1.0, bond_lengths={"A": 1.0, "B": 1.0},
        bc=["periodic"] * 2,
        chain_model="discrete" if method in ("ps", "rs") else "continuous",
        numerical_method=method, platform=platform, reduce_memory=reduced,
        field_type=field_type, aggregate_propagator_computation=False)


def canonical(platform, reduced, method, field_type, offset, solvent, population):
    solver = make_solver(platform, reduced, method, field_type)
    if solvent:
        solver.add_polymer(1.0 - population, [["B", 1.0, 0, 1]])
        solver.molecules.add_solvent(population, "A")
    else:
        solver.add_polymer(population, [["A", 1.0, 0, 1]])
        solver.add_polymer(1.0 - population, [["B", 1.0, 0, 1]])
    field = offset + (0.37j if field_type == "complex" else 0.0)
    dtype = complex if field_type == "complex" else float
    solver.compute_propagators({"A": np.full(4, field, dtype=dtype), "B": np.zeros(4, dtype=dtype)})
    solver.compute_concentrations()
    native = solver.propagator_computation
    q = native.get_solvent_partition(0) if solvent else solver.get_partition_function(0)
    expected_q = np.exp(-field)
    assert np.isfinite(q) and q != 0.0
    # Subnormal propagators lose absolute units of denorm_min at intermediate
    # contour operations. Account for that quantization without an absolute
    # floor that could accept an erroneously zero partition or density.
    tolerance = 5e-13 + 128 * (np.nextafter(0.0, 1.0) / abs(expected_q))
    np.testing.assert_allclose(partition_ratio(q, expected_q), 1.0, rtol=tolerance, atol=0.0)
    assert solver.check_total_partition()

    a, b = solver.get_concentration("A"), solver.get_concentration("B")
    values = {"partition": np.asarray(q), "A": a, "B": b}
    if solvent:
        values["solvent"] = native.get_solvent_concentration(0)
    else:
        values["polymer"] = native.get_total_concentration(0, "A")
        values["block"] = native.get_block_concentration(0)[0]
    for key in ("A", "solvent") if solvent else ("A", "polymer", "block"):
        assert np.isfinite(values[key]).all(), (platform, method, field_type, offset, key)
        np.testing.assert_allclose(values[key], population, rtol=tolerance, atol=0.0)
    np.testing.assert_allclose(b, 1.0 - population, rtol=5e-13, atol=0.0)
    np.testing.assert_allclose(solver.mean(a + b), 1.0, rtol=tolerance, atol=0.0)
    return values, tolerance


def grand_canonical(platform, reduced, method, field_type, fugacity):
    solver = make_solver(platform, reduced, method, field_type)
    solver.add_polymer(0.0, [["A", 1.0, 0, 1], ["A", 1.0, 0, 2], ["A", 1.0, 0, 3]])
    solver.add_polymer(1.0, [["B", 1.0, 0, 1]])
    dtype = complex if field_type == "complex" else float
    field = 236.0 + (0.1j if field_type == "complex" else 0.0)
    solver.compute_propagators({"A": np.full(4, field, dtype=dtype), "B": np.zeros(4, dtype=dtype)})
    solver.compute_concentrations()
    copies = [block.n_repeated for key, block in solver.propagator_optimizer.get_computation_blocks().items()
              if key[0] == 0]
    assert copies == [3], copies
    expected_q = np.exp(-3.0 * field)
    q = solver.get_partition_function(0)
    assert np.isfinite(q) and abs(q) >= np.finfo(float).tiny
    np.testing.assert_allclose(partition_ratio(q, expected_q), 1.0, rtol=3e-12, atol=0.0)
    assert solver.check_total_partition()
    density = solver.get_concentration_gce(fugacity, 0, "A")
    assert np.isfinite(density).all()
    # Three beads per molecule. Multiplying 3*fugacity first would overflow
    # for the upper case even though the partition is normal and density ~10.
    expected = 3.0 * (fugacity * expected_q)
    np.testing.assert_allclose(density, expected, rtol=3e-12, atol=0.0)
    np.testing.assert_array_equal(solver.get_concentration("A"), np.zeros(4))
    np.testing.assert_array_equal(solver.get_concentration_gce(0.0, 0, "A"), np.zeros(4))
    return {"partition": np.asarray(q), "density": density}


def main():
    reference = {}
    for platform in PlatformSelector.avail_platforms():
        for method in ("ps", "rs", "rqm4", "rk2"):
            for field_type in ("real", "complex"):
                for solvent in (False, True):
                    for population in (0.0, 0.5):
                        for offset in (0.0, 700.0, 720.0):
                            key = method, field_type, offset, solvent, population
                            full, tolerance = canonical(platform, False, *key)
                            reduced, _ = canonical(platform, True, *key)
                            for name in full:
                                assert full[name].tobytes() == reduced[name].tobytes(), (platform, key, name)
                            if key in reference:
                                for name in full:
                                    # Divide before comparing tiny Q, so an absolute tolerance cannot hide zero.
                                    if name == "partition":
                                        np.testing.assert_allclose(partition_ratio(full[name], reference[key][name]), 1.0,
                                                                   rtol=2 * tolerance, atol=0.0)
                                    else:
                                        np.testing.assert_allclose(full[name], reference[key][name],
                                                                   rtol=2 * tolerance, atol=0.0)
                            else:
                                reference[key] = full
                if method in ("ps", "rs"):
                    for fugacity in (1e307, 1e308):
                        full = grand_canonical(platform, False, method, field_type, fugacity)
                        reduced = grand_canonical(platform, True, method, field_type, fugacity)
                        for name in full:
                            assert full[name].tobytes() == reduced[name].tobytes(), (platform, method, name)
        print(f"Concentration normalization range: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
