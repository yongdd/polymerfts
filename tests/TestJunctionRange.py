"""Whole junction products retain finite propagators when individual powers overflow.

Stars of identical arms (aggregated or not, with integer and real multiplicities) in
fields of +-500 per arm, and a star of five arms with distinct initial conditions whose
partial products overflow: its junctions take several sources, which the reduced-memory
CUDA path multiplies over several launches. Full and reduced memory agree bit for bit.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def solve(platform, model, reduced, aggregate, field_type, offset, fractional=False):
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([8], [3.0])
    molecules = factory.create_molecules_information(model, 0.05, {"A": 1.0, "B": 1.0})
    if fractional:
        blocks = [["A", 1.0, 0, 1, 2.5], ["B", 1.0, 0, 2, 2.5]]
    else:
        blocks = [["A", 1.0, 0, 1], ["A", 1.0, 0, 2],
                  ["B", 1.0, 0, 3], ["B", 1.0, 0, 4]]
    molecules.add_polymer(1.0, blocks)
    optimizer = factory.create_propagator_computation_optimizer(molecules, aggregate)
    method = "rqm4" if model == "continuous" else "ps"
    solver = factory.create_propagator_computation(box, molecules, optimizer, method)
    dtype = np.complex128 if field_type == "complex" else np.float64
    potential = offset + (0.125j if field_type == "complex" else 0.0)
    solver.compute_propagators({"A": np.full(8, -potential, dtype=dtype),
                                "B": np.full(8, potential, dtype=dtype)})
    context = (platform, model, reduced, aggregate, field_type, offset, fractional)
    partition = np.asarray(solver.get_total_partition(0))
    np.testing.assert_allclose(partition, 1.0, rtol=3e-11, atol=3e-13, err_msg=str(context))
    assert solver.check_total_partition(), context
    errors = np.asarray(solver.compute_partition_errors())
    assert np.isfinite(errors).all() and np.max(errors) < 3e-11, (context, errors)
    solver.compute_concentrations()
    phi_a = solver.get_total_concentration("A")
    phi_b = solver.get_total_concentration("B")
    for phi in (phi_a, phi_b):
        np.testing.assert_allclose(phi, 0.5, rtol=3e-11, atol=3e-13, err_msg=str(context))
    np.testing.assert_allclose(box.mean(phi_a + phi_b), 1.0, rtol=0.0, atol=3e-11, err_msg=str(context))
    return {"partition": partition, "errors": errors, "phi_a": phi_a, "phi_b": phi_b}


def check_signed_and_zero(platform, model, reduced, aggregate, field_type):
    """Initial propagators may have signs or be zero; neither is a positive-density clamp."""
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box([8], [3.0])
    molecules = factory.create_molecules_information(model, 0.05, {"A": 1.0})
    molecules.add_polymer(1.0, [["A", 1.0, 0, leaf] for leaf in (1, 2, 3, 4)],
                          {leaf: str(leaf) for leaf in (1, 2, 3, 4)})
    optimizer = factory.create_propagator_computation_optimizer(molecules, aggregate)
    solver = factory.create_propagator_computation(box, molecules, optimizer,
                                                   "rqm4" if model == "continuous" else "ps")
    dtype = np.complex128 if field_type == "complex" else np.float64
    for first in (-2.0, 0.0):
        amplitudes = (first, 3.0, 4.0, 5.0)
        initial = {str(i): np.full(8, amplitude, dtype=dtype) for i, amplitude in enumerate(amplitudes, 1)}
        solver.compute_propagators({"A": np.zeros(8, dtype=dtype)}, q_init=initial)
        np.testing.assert_allclose(solver.get_total_partition(0), np.prod(amplitudes),
                                   rtol=3e-12, atol=1e-13)


def distinct_amplitudes(field_type, many):
    """Uniform initial conditions of a star whose amplitudes multiply to 3 (zero field).

    Five arms: e^400, e^400, e^-400, e^-400 and 3. Many arms: 20 x e^100, 20 x e^-100 and 3,
    whose 40 sources per junction take two CUDA launches also in full memory. Every junction
    takes several sources and its partial products overflow.
    """
    phase = np.exp(0.3j) if field_type == "complex" else 1.0
    if not many:
        return [np.exp(400.0) * phase, np.exp(400.0), np.exp(-400.0), np.exp(-400.0) / phase, 3.0]
    big = [np.exp(100.0) * (phase if i % 3 == 0 else 1.0) for i in range(20)]
    small = [np.exp(-100.0) / (phase if i % 3 == 0 else 1.0) for i in range(20)]
    return [big[0], small[0], *big[1:], *small[1:], 3.0]


def check_distinct_sources(platform, model, field_type, amplitudes):
    n_arms = len(amplitudes)
    dtype = np.complex128 if field_type == "complex" else np.float64
    results = []
    for reduced in (False, True):
        factory = PlatformSelector.create_factory(platform, reduced, field_type)
        box = factory.create_computation_box([8], [3.0])
        molecules = factory.create_molecules_information(model, 0.05, {"A": 1.0})
        molecules.add_polymer(1.0, [["A", 1.0, 0, leaf] for leaf in range(1, n_arms + 1)],
                              {leaf: str(leaf) for leaf in range(1, n_arms + 1)})
        optimizer = factory.create_propagator_computation_optimizer(molecules, False)
        solver = factory.create_propagator_computation(box, molecules, optimizer,
                                                       "rqm4" if model == "continuous" else "ps")
        initial = {str(i): np.full(8, amplitude, dtype=dtype) for i, amplitude in enumerate(amplitudes, 1)}
        solver.compute_propagators({"A": np.zeros(8, dtype=dtype)}, q_init=initial)
        partition = np.asarray(solver.get_total_partition(0))
        # The product of the amplitudes is 3, although partial products overflow
        np.testing.assert_allclose(partition, 3.0, rtol=3e-12, atol=0.0,
                                   err_msg=str((platform, model, field_type, n_arms, reduced)))
        solver.compute_concentrations()
        phi = solver.get_total_concentration("A")
        assert np.isfinite(phi).all(), (platform, model, field_type, n_arms, reduced)
        np.testing.assert_allclose(box.mean(phi), 1.0, rtol=0.0, atol=3e-11)
        results.append((partition, phi))
    for a, b in zip(*results):
        assert a.tobytes() == b.tobytes(), (platform, model, field_type, n_arms, "full and reduced memory differ")


def main():
    for platform in PlatformSelector.avail_platforms():
        for model in ("continuous", "discrete"):
            for aggregate in (False, True):
                for field_type in ("real", "complex"):
                    for offset in (0.0, 500.0, -500.0):
                        modes = [solve(platform, model, reduced, aggregate, field_type, offset)
                                 for reduced in (False, True)]
                        for key in modes[0]:
                            assert modes[0][key].tobytes() == modes[1][key].tobytes(), (
                                platform, model, aggregate, field_type, offset, key)
                    for reduced in (False, True):
                        check_signed_and_zero(platform, model, reduced, aggregate, field_type)
                for offset in (500.0, -500.0):
                    modes = [solve(platform, model, reduced, aggregate, "real", offset, fractional=True)
                             for reduced in (False, True)]
                    for key in modes[0]:
                        assert modes[0][key].tobytes() == modes[1][key].tobytes(), (
                            platform, model, aggregate, "fractional", offset, key)
            for field_type in ("real", "complex"):
                for many in (False, True):
                    check_distinct_sources(platform, model, field_type, distinct_amplitudes(field_type, many))
        print(f"Junction range: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
