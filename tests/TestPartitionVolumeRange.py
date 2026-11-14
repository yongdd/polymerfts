#!/usr/bin/env python3
"""A finite partition must not overflow or underflow through its volume integral.

Uniform field shifts leave canonical density unchanged. These cases keep every
propagator and Q representable while V*Q is outside double range; they cover
both chain models, memory modes, masks, aggregation, complex fields and solvent.
"""

import sys

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver, SpaceGroup
from cuda_part import cuda_part_missing


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all(), (label, actual)
    # Comparing ratios also checks tiny, nonzero partitions without an absolute floor.
    np.testing.assert_allclose(actual / expected, 1.0, rtol=3e-11, atol=3e-11, err_msg=label)


def tiny_volume_mean(platform, field_type):
    # The integral is subnormal, but dividing by V makes the result look
    # ordinary again. Test the box alone: no diffusion on this tiny geometry.
    factory = PlatformSelector.create_factory(platform, False, field_type)
    box = factory.create_computation_box([8], [1e-300])
    expected = 1e-20 * (1.0 + 0.5j if field_type == "complex" else 1.0)
    field = np.full(8, expected)
    close(box.mean(field), expected, "mean must not retain subnormal-integral rounding")


def make_solver(platform, model, reduced, field_type, geometry, length, aggregate=False):
    nx, lx, mask, group = [8], [length], None, None
    if geometry == "mask":
        mask = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    elif geometry.startswith("space_group"):
        nx, lx = [8, 8, 8], [length] * 3
        group = SpaceGroup(nx, 227)
        if geometry == "space_group_physical":
            group.enable_pmmm_physical_basis()
    solver = PropagatorSolver(
        nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0, "B": 1.2, "S": 1.0},
        bc=["periodic"] * (2 * len(nx)), chain_model=model,
        numerical_method="rqm4" if model == "continuous" else "ps",
        platform=platform, reduce_memory=reduced, field_type=field_type,
        mask=mask, space_group=group, aggregate_propagator_computation=aggregate)
    if aggregate:
        blocks = [["A", 0.2, 0, 1], ["A", 0.2, 1, 2]]
        for node in range(3):
            blocks += [["B", 0.1, node, 10 + 2 * node], ["A", 0.1, 10 + 2 * node, 11 + 2 * node]]
    else:
        blocks = [["A", 1.0, 0, 1]]
    solver.add_polymer(0.7, blocks)
    solver._molecules.add_solvent(0.3, "S")
    n = int(np.prod(nx)) if group is None else group.get_n_reduced_basis()
    return solver, n


def shifted(platform, model, reduced, field_type, geometry="bulk", length=8.0, shift=-708.0, aggregate=False):
    solver, n = make_solver(platform, model, reduced, field_type, geometry, length, aggregate)
    dtype = complex if field_type == "complex" else float
    phase = 0.37j if field_type == "complex" else 0.0
    outputs = []
    baseline = None
    for offset in (0.0, shift):
        fields = {m: np.full(n, offset + phase, dtype=dtype) for m in "AB"}
        # Solvent field uses ds rather than the polymer's contour length.
        fields["S"] = np.full(n, (offset + phase) / 0.05, dtype=dtype)
        solver.compute_propagators(fields)
        for v, u in ((0, 1), (1, 0)) if not aggregate else ():
            propagator = solver.get_propagator(0, v, u, 20)
            assert np.isfinite(propagator).all(), (platform, model, field_type, geometry, offset, v, u)
        q = solver.get_partition_function(0)
        assert np.isfinite(q) and q != 0.0, (platform, model, geometry, offset, q)
        assert solver.check_total_partition(), (platform, model, geometry, offset, solver.partition_errors())
        solver.compute_concentrations()
        phi = np.asarray([solver.get_concentration(m) for m in "ABS"])
        assert np.isfinite(phi).all()
        np.testing.assert_allclose(solver.mean(phi.sum(axis=0)), 1.0, rtol=3e-11, atol=3e-11)
        solvent_q = solver.propagator_computation.get_solvent_partition(0)
        if baseline is None:
            baseline = q, solvent_q, phi
        else:
            scale = np.exp(-offset)
            close(q / scale, baseline[0], "polymer field-shift normalization")
            close(solvent_q / scale, baseline[1], "solvent field-shift normalization")
            np.testing.assert_allclose(phi, baseline[2], rtol=3e-11, atol=3e-11)
        outputs.append((np.asarray(q), np.asarray(solvent_q), phi))
    if aggregate:
        assert any(b.n_segment_right and b.n_copies / b.n_repeated > 1
                   for b in solver.propagator_optimizer.get_computation_blocks().values())
    return outputs


def signed_graft(platform, model, reduced, field_type):
    dtype = complex if field_type == "complex" else float
    phase = 1j if field_type == "complex" else 1.0
    solver = PropagatorSolver(
        nx=[2], lx=[1000.0], ds=0.05, bond_lengths={"A": 0.1},
        bc=["periodic"] * 2, chain_model=model, platform=platform,
        numerical_method="rqm4" if model == "continuous" else "ps",
        reduce_memory=reduced, field_type=field_type, aggregate_propagator_computation=False)
    solver.add_polymer(0.0, [["A", 1.0, 0, 1]], {0: "G"})
    results = []
    for sign in (np.array([1.0, -1.0]), np.array([-1.0, -1.0])):
        solver.compute_propagators({"A": np.full(2, -705.0, dtype=dtype)}, {"G": phase * sign})
        q = solver.get_partition_function(0)
        assert np.isfinite(q)
        assert solver.check_total_partition(), solver.partition_errors()
        if sign[1] != sign[0]:
            assert q == 0.0, q
        else:
            close(q / np.exp(705.0), -phase, "signed partition")
        results.append(np.asarray(q))
    return results


def main():
    reference = {}
    for platform in PlatformSelector.avail_platforms():
        for field_type in ("real", "complex"):
            tiny_volume_mean(platform, field_type)
        for model in ("continuous", "discrete"):
            for field_type in ("real", "complex"):
                # Keep the propagators themselves finite on every backend.
                # CUDA complex RQM4 applies FFT normalization after a complex
                # product; -705 avoids overflow there while V*Q still overflows.
                large_length, large_shift = (1000.0, -705.0) if field_type == "complex" else (8.0, -708.0)
                cases = [("bulk", large_length, large_shift, False), ("bulk", 1e-20, 708.0, False),
                         ("mask", large_length, large_shift, False), ("bulk", 1e6, -700.0, True)]
                if field_type == "real":
                    cases += [("space_group", 2.0, -708.0, False),
                              ("space_group_physical", 2.0, -708.0, False)]
                for geometry, length, shift, aggregate in cases:
                    key = model, field_type, geometry, length, shift, aggregate
                    full = shifted(platform, model, False, field_type, geometry, length, shift, aggregate)
                    reduced = shifted(platform, model, True, field_type, geometry, length, shift, aggregate)
                    for a, b in zip(full, reduced):
                        for left, right in zip(a, b):
                            assert left.tobytes() == right.tobytes(), (platform, key, "checkpoint mismatch")
                    if key in reference:
                        for a, b in zip(full, reference[key]):
                            close(a[0], b[0], "partition backend agreement")
                            close(a[1], b[1], "solvent backend agreement")
                            np.testing.assert_allclose(a[2], b[2], rtol=3e-11, atol=3e-11)
                    else:
                        reference[key] = full
                full = signed_graft(platform, model, False, field_type)
                reduced = signed_graft(platform, model, True, field_type)
                for a, b in zip(full, reduced):
                    assert a.tobytes() == b.tobytes(), (platform, model, field_type, "signed checkpoint mismatch")
        print(f"Partition volume range: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
