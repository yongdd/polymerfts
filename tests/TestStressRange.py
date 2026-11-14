#!/usr/bin/env python3
"""Stress keeps its range through FFT products, contour sums and ensemble normalization.

A uniform field shift leaves canonical stress unchanged. The shifted cases have
finite Q and propagators but raw Fourier products or contour integrals outside
double range. Signed grafts additionally check GCE stress at exactly zero Q.
"""

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver, SpaceGroup
from cuda_part import cuda_part_missing


def make_solver(platform, model, reduced, field_type, geometry="periodic", population=1.0):
    nx, lx, mask, group = [32], [3.0], None, None
    bc = ["periodic"] * 2
    if geometry in ("reflecting", "absorbing"):
        bc = [geometry] * 2
    elif geometry == "mask":
        mask = np.ones(32)
        mask[:4] = 0.0
        mask[-4:] = 0.0
    elif geometry in ("space_group", "space_group_generic"):
        nx, lx = [8, 8, 8], [3.0, 3.0, 3.0]
        bc = ["periodic"] * 6
        group = SpaceGroup(nx, 227)  # Pmmm
        if geometry == "space_group":
            group.enable_pmmm_physical_basis()  # CrysFFT; the other case expands to the full grid
    solver = PropagatorSolver(
        nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0}, bc=bc,
        chain_model=model, numerical_method="rqm4" if model == "continuous" else "ps",
        platform=platform, reduce_memory=reduced, field_type=field_type,
        mask=mask, space_group=group, aggregate_propagator_computation=False)
    solver.add_polymer(population, [["A", 1.0, 0, 1]])
    coordinates = np.meshgrid(*[2.0 * np.pi * (np.arange(n) + 0.5) / n for n in nx], indexing="ij")
    field = sum(2.0 / len(nx) * np.cos(x) for x in coordinates).flatten()
    if field_type == "complex":
        field = field + 0.2j * np.cos(2.0 * coordinates[0]).flatten() + 0.37j
    if group is not None:
        field = solver.to_reduced_basis(field)
    return solver, field


def shifted_stress(platform, model, reduced, field_type, geometry="periodic"):
    solver, field = make_solver(platform, model, reduced, field_type, geometry)
    # The larger 3-D FFT also has to propagate q itself, before computing stress.
    shifts = (0.0, -695.0) if geometry.startswith("space_group") else (0.0, -703.0, 700.0)
    reference = None
    results = []
    for shift in shifts:
        solver.compute_propagators({"A": field + shift})
        q = solver.get_partition_function(0)
        assert np.isfinite(q) and q != 0.0, (platform, model, geometry, shift, q)
        assert solver.check_total_partition(), (platform, model, geometry, shift)
        solver.compute_concentrations()
        density = solver.get_concentration("A")
        assert np.isfinite(density).all()
        np.testing.assert_allclose(solver.mean(density), 1.0, rtol=2e-10, atol=2e-10)
        solver.compute_stress()
        stress = np.asarray(solver.get_stress())
        gce = np.asarray(solver.get_stress_gce([np.exp(shift)]))
        assert np.isfinite(stress).all() and np.isfinite(gce).all()
        np.testing.assert_array_equal(solver.get_stress_gce([0.0]), np.zeros(6))
        if reference is None:
            reference = q, density, stress, gce
            assert np.max(np.abs(stress)) > 1e-5, "the reference must exercise nonzero stress"
        else:
            np.testing.assert_allclose(q * np.exp(shift), reference[0], rtol=2e-10, atol=0.0)
            np.testing.assert_allclose(density, reference[1], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(stress, reference[2], rtol=2e-10, atol=2e-12)
            np.testing.assert_allclose(gce, reference[3], rtol=2e-10, atol=2e-12)
        # A second stress computation must reset both values and exponents.
        solver.compute_stress()
        np.testing.assert_array_equal(stress, solver.get_stress())
        results.append((stress, gce))
    return results


def zero_partition_gce(platform, model, reduced, field_type):
    phase = 1j if field_type == "complex" else 1.0
    sign = phase * np.array([1.0, -1.0])
    outputs = []
    reference_density = None
    for shift in (0.0, -703.0):
        solver = PropagatorSolver(
            nx=[2], lx=[1.0], ds=0.05, bond_lengths={"A": 0.1},
            bc=["periodic"] * 2, chain_model=model,
            numerical_method="rqm4" if model == "continuous" else "ps",
            platform=platform, reduce_memory=reduced, field_type=field_type,
            aggregate_propagator_computation=False)
        solver.add_polymer(0.0, [["A", 1.0, 0, 1]], {0: "G"})
        field = np.full(2, shift, dtype=complex if field_type == "complex" else float)
        solver.compute_propagators({"A": field}, {"G": sign})
        # Odd graft against the free end: exact cancellation on the two-point
        # grid. Q stays zero at every contour cut, without a zero propagator.
        assert solver.get_partition_function(0) == 0.0, solver.get_partition_function(0)
        assert solver.check_total_partition()
        solver.compute_concentrations()
        np.testing.assert_array_equal(solver.get_concentration("A"), np.zeros(2))
        density = solver.get_concentration_gce(np.exp(shift), 0, "A")
        assert np.isfinite(density).all() and np.max(np.abs(density)) > 0.1
        np.testing.assert_allclose(solver.mean(density), 0.0, rtol=0.0, atol=2e-12)
        if reference_density is None:
            reference_density = density
        else:
            np.testing.assert_allclose(density, reference_density, rtol=2e-10, atol=2e-12)
        solver.compute_stress()
        np.testing.assert_array_equal(solver.get_stress(), np.zeros(6))
        np.testing.assert_array_equal(solver.get_stress_gce([0.0]), np.zeros(6))
        stress = np.asarray(solver.get_stress_gce([np.exp(shift)]))
        # Q is identically zero as L varies, hence its derivative is zero.
        # Dividing the stress integral by Q anywhere would produce NaN.
        np.testing.assert_array_equal(stress, np.zeros(6))
        outputs.append(stress)
    return outputs


def component_range(platform, model, reduced):
    phase = np.array([1.0, -1.0])
    outputs = []
    # The last case crosses the subnormal threshold along the contour, so a
    # block contains both ordinary and split-component stress pairs.
    for small, bond in ((1e-200, 0.1), (1e-109, 0.1), (1e-107, 1.0)):
        results = []
        for wide in (False, True):
            solver = PropagatorSolver(
                nx=[2], lx=[1.0], ds=0.05, bond_lengths={"A": bond},
                bc=["periodic"] * 2, chain_model=model,
                numerical_method="rqm4" if model == "continuous" else "ps",
                platform=platform, reduce_memory=reduced, field_type="complex",
                aggregate_propagator_computation=False)
            solver.add_polymer(1.0, [["A", 1.0, 0, 1]], {0: "Z", 1: "A"})
            if wide:
                left = np.empty(2, dtype=complex)
                left.real, left.imag = 1e200, small * phase
                right = np.empty(2, dtype=complex)
                right.real, right.imag = 1e100 * phase, 1e-200
                fugacity = 1.0 / (small * 1e100)
            else:
                left, right = 1j * phase, phase.astype(complex)
                fugacity = 1.0
            solver.compute_propagators({"A": np.zeros(2, dtype=complex)}, {"Z": left, "A": right})
            q = solver.get_partition_function(0)
            assert np.isfinite(q) and q != 0.0
            assert solver.check_total_partition()
            solver.compute_concentrations()
            density = solver.get_concentration("A")
            assert np.isfinite(density).all()
            np.testing.assert_allclose(solver.mean(density), 1.0, rtol=2e-10, atol=2e-10)
            solver.compute_stress()
            stress = np.asarray(solver.get_stress_gce([fugacity]))
            assert np.isfinite(stress).all()
            results.append(stress)
        assert np.max(np.abs(results[0])) > 1e-5
        # Constants contribute no stress. Only the small imaginary alternating
        # component times the large real alternating component contributes.
        np.testing.assert_allclose(results[1], results[0], rtol=2e-10, atol=0.0)
        outputs.append(results[1])
    return outputs


def main():
    references = {}
    for platform in PlatformSelector.avail_platforms():
        for model in ("continuous", "discrete"):
            for field_type in ("real", "complex"):
                full = shifted_stress(platform, model, False, field_type)
                reduced = shifted_stress(platform, model, True, field_type)
                for (a, ga), (b, gb) in zip(full, reduced):
                    np.testing.assert_array_equal(a, b, err_msg=f"{platform} {model} {field_type}: canonical memory modes")
                    np.testing.assert_array_equal(ga, gb, err_msg=f"{platform} {model} {field_type}: GCE memory modes")
                key = model, field_type
                if key in references:
                    np.testing.assert_allclose(full, references[key], rtol=2e-10, atol=2e-12)
                else:
                    references[key] = full
                full = zero_partition_gce(platform, model, False, field_type)
                reduced = zero_partition_gce(platform, model, True, field_type)
                np.testing.assert_array_equal(full, reduced)
            full = component_range(platform, model, False)
            reduced = component_range(platform, model, True)
            np.testing.assert_array_equal(full, reduced)
            for geometry in ("reflecting", "absorbing", "mask", "space_group", "space_group_generic"):
                full = shifted_stress(platform, model, False, "real", geometry)
                reduced = shifted_stress(platform, model, True, "real", geometry)
                np.testing.assert_array_equal(full, reduced, err_msg=f"{platform} {model} {geometry}: memory modes")
        print("Stress range:", platform, "passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
