#!/usr/bin/env python3
"""A zero grand-canonical partition is regular; canonical zero Q remains invalid.

A canonical polymer of zero population with Q = 0 has exactly zero density, the
GCE density of the same chain is regular, and an occupied one fails the partition
check, with real and complex fields on every platform and in both memory modes;
SCFT (TestScftAbsentPolymer.py) and CL-FTS (TestClftsGrandCanonical.py) check their
own observables of these states on one CPU platform. A zero fugacity gives exactly
zero density, also when the unused raw density of the chain is not finite, and
still validates its polymer and monomer arguments.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def solve(platform, reduced, model, field_type, canonical_weight=0.0):
    # Complex weights of a discrete chain cancel in Q; otherwise a zero graft
    # gives Q = 0 at every contour position without cancellation.
    cancellation = model == "discrete" and field_type == "complex"
    dtype = complex if field_type == "complex" else float
    w = np.array([0.0, 0.0, 1j * np.pi, -1j * np.pi]) if cancellation else np.zeros(4, dtype=dtype)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=[4], lx=[2.0], bc=["periodic"] * 2,
            ds=1.0 if model == "discrete" else 0.1,
            bond_lengths={"A": 1.0, "B": 1.0}, chain_model=model,
            platform=platform, reduce_memory=reduced, field_type=field_type,
            aggregate_propagator_computation=False,
        )
        solver.add_polymer(canonical_weight, [["A", 1.0, 0, 1]],
                           None if cancellation else {0: "G"})
        solver.add_polymer(1.0 - canonical_weight, [["B", 1.0, 0, 1]])
        solver.compute_propagators(
            {"A": w, "B": np.zeros(4, dtype=dtype)},
            None if cancellation else {"G": np.zeros(4, dtype=dtype)},
        )
        assert solver.get_partition_function(0) == 0.0
        if canonical_weight:
            assert not solver.check_total_partition()
            return None
        solver.compute_concentrations()
        canonical = solver.get_concentration("A")
        np.testing.assert_array_equal(canonical, np.zeros(4))
        np.testing.assert_array_equal(solver.propagator_computation.get_total_concentration(0, "A"), np.zeros(4))
        np.testing.assert_array_equal(solver.propagator_computation.get_block_concentration(0), np.zeros((1, 4)))
        np.testing.assert_allclose(solver.get_concentration("B"), 1.0, rtol=0.0, atol=2e-14)
        gce = solver.get_concentration_gce(0.4, 0, "A")
        expected = 0.4 * np.exp(-w) if cancellation else np.zeros(4)
        np.testing.assert_allclose(gce, expected, rtol=2e-14, atol=2e-14)
        assert solver.check_total_partition()
        np.testing.assert_array_equal(solver.partition_errors(), np.zeros(2))
        solver.compute_stress()
        stress = np.asarray(solver.get_stress())
        stress_gce = np.asarray(solver.propagator_computation.get_stress_gce([0.4, 0.0]))
        np.testing.assert_allclose(stress, 0.0, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(stress_gce, 0.0, rtol=0.0, atol=2e-14)
    return canonical, gce, stress, stress_gce


def solve_cancellation(platform, reduced, real_offset, model):
    """Nonzero spatial integrands cancel, leaving finite contour roundoff."""
    w = real_offset + 1j * (np.array([0.0, 0.0, np.pi, np.pi]) + 0.7)
    graft = None
    if model == "continuous":
        w = np.full(4, real_offset, dtype=complex)
        graft = {"G": np.exp(0.7j) * np.array([1.0, 1.0, -1.0, -1.0])}
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=[4], lx=[2.0], bc=["periodic"] * 2,
            ds=1.0 if model == "discrete" else 0.1,
            bond_lengths={"A": 1.0, "B": 1.0}, chain_model=model,
            platform=platform, reduce_memory=reduced, field_type="complex",
        )
        solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
        solver.add_polymer(0.0, [["B", 3.0, 0, 1]], {0: "G"} if graft else None)
        solver.compute_propagators({"A": np.zeros(4, dtype=complex), "B": w}, graft)
        solver.compute_concentrations()
        phi = solver.get_concentration_gce(1.0, 1, "B")
        assert np.isfinite(phi).all()
        # The absent canonical chain has no density, whatever its contour round-off
        np.testing.assert_array_equal(solver.get_concentration("B"), np.zeros(4))
        assert solver.check_total_partition()
        errors = np.asarray(solver.partition_errors())
        assert np.isfinite(errors).all() and np.max(errors) < 1e-13
        solver.compute_stress()
        stress = np.asarray(solver.propagator_computation.get_stress_gce([0.0, 1.0]))
        assert np.isfinite(stress).all()
        partition = np.asarray(solver.get_partition_function(1))
        # The length-three chain multiplies by exp(-3*offset) under a uniform shift.
        # Scaling back also checks that the diagnostic does not impose an absolute
        # numerical floor on a small or large partition.
        scale = np.exp(3.0 * real_offset)
        assert abs(partition * scale) < 1e-14
        if model == "discrete" and real_offset == 0.0:
            # The GCE density of the three beads integrates to 3 Q on the weight scale
            np.testing.assert_allclose(np.mean(phi), 3.0 * partition, rtol=0, atol=2e-14)
    return partition, phi, stress, errors


def zero_fugacity(platform, reduced, method, field_type):
    """Zero fugacity stays zero even if an unused raw density is nonfinite."""
    dtype = np.complex128 if field_type == "complex" else np.float64
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([4], [4.0])
    molecules = factory.create_molecules_information("discrete", 1.0, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(0.0, [["A", 1.0, 0, 1]])
    molecules.add_polymer(1.0, [["B", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(box, molecules, optimizer, method)
    # exp(-1000) underflows: the raw density of A divides zero by zero
    solver.compute_propagators({"A": np.full(4, 1000.0, dtype=dtype),
                                "B": np.zeros(4, dtype=dtype)})
    solver.compute_concentrations()
    np.testing.assert_array_equal(solver.get_total_concentration_gce(0.0, 0, "A"), np.zeros(4))
    np.testing.assert_allclose(solver.get_total_concentration("B"), 1.0, rtol=0.0, atol=2e-14)
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
                for reduced in (False, True):
                    zero_fugacity(platform, reduced, method, field_type)
        for model in ("continuous", "discrete"):
            for field_type in ("real", "complex"):
                full = solve(platform, False, model, field_type)
                reduced = solve(platform, True, model, field_type)
                for a, b in zip(full, reduced):
                    assert a.tobytes() == b.tobytes(), (platform, model, field_type, "checkpoint mismatch")
                for reduced in (False, True):
                    solve(platform, reduced, model, field_type, canonical_weight=1.0)
        for model in ("continuous", "discrete"):
            reference = None
            for offset in (0.0, -2.0, 2.0):
                full = solve_cancellation(platform, False, offset, model)
                reduced = solve_cancellation(platform, True, offset, model)
                for a, b in zip(full, reduced):
                    assert a.tobytes() == b.tobytes(), (platform, model, offset, "cancellation checkpoint mismatch")
                scaled = full[1] * np.exp(3.0 * offset)
                if reference is None:
                    reference = scaled
                else:
                    np.testing.assert_allclose(scaled, reference, rtol=2e-13, atol=2e-14)
        print(f"Zero-partition GCE: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
