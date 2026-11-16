#!/usr/bin/env python3
"""Excluded beads must not divide by an underflowed Boltzmann factor.

Changing only the field inside a binary mask leaves propagators, partitions,
canonical and grand-canonical concentrations unchanged. The absent canonical
species also exercises the absolute-integrand partition diagnostic. Accessible
zero divisors retain their singular behavior. All platforms, both memory modes,
ps/rs, real/complex fields and space-group masks are covered.
"""

import sys

import numpy as np

from polymerfts import PlatformSelector, SpaceGroup
from cuda_part import cuda_part_missing


def solve(platform, reduced, method, field_type, nx, mask, field, segments,
          canonical_weight, group=None, regular=True):
    dtype = np.complex128 if field_type == "complex" else np.float64
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box(nx, [4.0] * len(nx), mask=mask)
    molecules = factory.create_molecules_information("discrete", 1.0, {"A": 1.0, "B": 1.2})
    molecules.add_polymer(canonical_weight, [["A", float(segments), 0, 1]])
    molecules.add_polymer(1.0 - canonical_weight, [["B", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    solver = factory.create_propagator_computation(box, molecules, optimizer, method, group)
    full_field = np.asarray(field, dtype=dtype)
    basis = (lambda values: values) if group is None else group.to_reduced_basis
    solver.compute_propagators({"A": basis(full_field), "B": basis(np.zeros_like(full_field))})
    partition = solver.get_total_partition(0)
    errors = np.asarray(solver.compute_partition_errors())
    assert solver.check_total_partition() == regular
    solver.compute_concentrations()
    phi_a = solver.get_total_concentration("A")
    phi_b = solver.get_total_concentration("B")
    phi_gce = solver.get_total_concentration_gce(0.4, 0, "A")
    block_a = solver.get_block_concentration(0)[0]
    np.testing.assert_array_equal(phi_a, solver.get_total_concentration(0, "A"))
    # Block normalization runs on the host; total concentration is summed on the
    # backend. Compare each getter with itself across the field/memory changes,
    # since host and CUDA complex multiplications can round differently.
    result = {
        "partition": np.asarray(partition),
        "errors": errors,
        "propagator": solver.get_chain_propagator(0, 0, 1, segments),
        "phi_a": phi_a,
        "block_a": block_a,
        "phi_b": phi_b,
        "phi_gce": phi_gce,
    }
    if regular:
        assert all(np.isfinite(value).all() for value in result.values())
        np.testing.assert_allclose(box.mean(phi_a + phi_b), 1.0, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(box.mean(block_a), canonical_weight, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(box.mean(phi_gce), 0.4 * segments * partition,
                                   rtol=2e-12, atol=2e-12)
        if canonical_weight == 0.0:
            np.testing.assert_array_equal(phi_a, np.zeros_like(phi_a))
            np.testing.assert_array_equal(block_a, np.zeros_like(block_a))
        if segments == 1:
            boltzmann = np.exp(-full_field)
            if mask is not None:
                boltzmann[mask == 0.0] = 0.0
            expected_partition = np.mean(boltzmann)
            expected_phi = canonical_weight * basis(boltzmann) / expected_partition
            np.testing.assert_allclose(partition, expected_partition, rtol=2e-14, atol=2e-14)
            for value in (phi_a, block_a):
                np.testing.assert_allclose(value, expected_phi, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(phi_gce, 0.4 * basis(boltzmann), rtol=2e-14, atol=2e-14)
        if mask is not None:
            excluded = basis(mask) == 0.0
            for value in (phi_a, block_a, phi_b, phi_gce):
                np.testing.assert_array_equal(value[excluded], np.zeros_like(value[excluded]))

            # The public host-buffer box operation must also ignore excluded 0/0.
            numerator = np.ones(box.get_n_basis(), dtype=dtype)
            divisor = numerator.copy()
            numerator[excluded] = 0.0
            divisor[excluded] = 0.0
            np.testing.assert_allclose(box.inner_product_inverse_weight(numerator, numerator, divisor),
                                       box.integral(np.ones_like(numerator)), rtol=2e-14, atol=2e-14)
    return result


def compare_bitwise(left, right, label):
    for key in left:
        assert left[key].tobytes() == right[key].tobytes(), (label, key)


def masked_case(platform, method, field_type, nx, mask, field, segments, canonical_weight, group=None):
    field = field.copy()
    field[mask == 0.0] = 0.0
    underflow = field.copy()
    underflow[mask == 0.0] = 1000.0
    modes = []
    for reduced in (False, True):
        control = solve(platform, reduced, method, field_type, nx, mask, field,
                        segments, canonical_weight, group)
        changed = solve(platform, reduced, method, field_type, nx, mask, underflow,
                        segments, canonical_weight, group)
        for key in control:
            np.testing.assert_array_equal(changed[key], control[key], err_msg=key)
        modes.append(changed)
    compare_bitwise(*modes, (platform, method, field_type, "checkpoint", segments, canonical_weight))


def test_full_grid(platform, method, field_type):
    nx = [8]
    mask = np.array([1.0] * 6 + [0.0] * 2)
    rng = np.random.default_rng(20261113)
    random_field = rng.normal(0.0, 5.0, 8)
    if field_type == "complex":
        random_field = random_field + 1j * rng.normal(0.0, 1.0, 8)
    for segments in (1, 3):
        field = np.zeros_like(random_field) if segments == 1 else random_field
        for fraction in (0.0, 1.0):
            masked_case(platform, method, field_type, nx, mask, field, segments, fraction)

    for reduced in (False, True):
        # An all-accessible mask uses the new kernel but must preserve ordinary arithmetic.
        plain = solve(platform, reduced, method, field_type, nx, None, random_field, 3, 1.0)
        ones = solve(platform, reduced, method, field_type, nx, np.ones(8), random_field, 3, 1.0)
        compare_bitwise(plain, ones, (platform, method, field_type, "all-accessible mask"))

        # A zero divisor at an accessible point must not be silently zeroed.
        singular = np.zeros_like(random_field)
        singular[0] = 1000.0
        result = solve(platform, reduced, method, field_type, nx, mask, singular, 1, 1.0, regular=False)
        assert not np.isfinite(result["partition"])
        assert not np.isfinite(result["phi_a"]).all()


def test_space_group(platform, method):
    nx = [8, 8, 8]
    for physical in (False, True):
        group = SpaceGroup(nx, "Im-3m", 529)
        orbit = np.asarray(group.get_full_to_reduced_map())
        mask = (orbit != orbit[0]).astype(float)
        rng = np.random.default_rng(41)
        field = group.symmetrize(rng.normal(0.0, 1.0, np.prod(nx)))
        field = 5.0 * (field - field.mean()) / field.std()
        if physical:
            group.enable_m3_physical_basis()
        for fraction in (0.0, 1.0):
            masked_case(platform, method, "real", nx, mask, field, 3, fraction, group)


def main():
    for platform in PlatformSelector.avail_platforms():
        for method in ("ps", "rs"):
            for field_type in ("real", "complex"):
                test_full_grid(platform, method, field_type)
            test_space_group(platform, method)
        print(f"Masked discrete weights: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
