"""compute_concentrations_and_stress() gives the results of compute_concentrations() and
compute_stress(), bit for bit.

With reduced memory one walk along each block gives the concentrations and the stress, which
must equal those of the full-memory computation. Every platform, both chain models, every method
that computes the stress, real and complex fields, with and without propagator aggregation; the
polymers have junction half bonds (discrete chains), sliced aggregated blocks, a grafted end and
a solvent, and blocks longer than the checkpoint interval, which reduced memory recomputes in
several pieces. Random fields with standard deviation 5. A method without stress refuses the call.
"""

import sys

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing

DS = 0.01
NX = [6, 8, 10]
LX = [3.0, 3.3, 3.6]
OBLIQUE = [80.0, 95.0, 100.0]
# 40 and 60 segments; the checkpoint interval of reduced memory, 2√(total segments), is about 45
DIBLOCK = [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]]
# A backbone of three blocks with side chains of 60, 60, 40 and 40 segments: aggregation slices
# the side chains of 60 segments at the length of the others
COMB = ([["A", 0.1, i, i+1] for i in range(3)]
        + [["B", 0.6, 0, 4], ["B", 0.6, 1, 5], ["B", 0.4, 2, 6], ["B", 0.4, 3, 7]])
# A backbone of 1, 2, 3 and 1 segments with a B2-A1 side chain on every backbone node: the sliced
# keys are merged again
BRUSH = ([["A", DS*n, i, i+1] for i, n in enumerate([1, 2, 3, 1])]
         + [block for i in range(5) for block in (["B", DS*2, i, 100+2*i], ["A", DS*1, 100+2*i, 101+2*i])])
STRESS_METHODS = {"continuous": ("rqm4", "rk2"), "discrete": ("ps", "rs")}


def results(computation, n_polymers, aggregate):
    """Every concentration getter and both stresses, as bytes."""
    out = []
    for m in "AB":
        out.append(computation.get_total_concentration(m))
        for p in range(n_polymers):
            out.append(computation.get_total_concentration(p, m))
            out.append(computation.get_total_concentration_gce(1.5, p, m))
    if not aggregate:
        for p in range(n_polymers):
            out.append(computation.get_block_concentration(p))
    out.append(computation.get_solvent_concentration(0))
    out.append(np.asarray([computation.get_solvent_partition(0)]))
    out.append(np.asarray(computation.get_stress()))
    out.append(np.asarray(computation.get_stress_gce([0.5, 1.0, 2.0, 0.7])))
    return [np.asarray(x).tobytes() for x in out]


def solve(platform, reduced, chain_model, method, field_type, aggregate, angles, fields):
    """The results of the two calls and of the combined call on one computation."""
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    box = factory.create_computation_box(NX, LX, angles=angles)
    molecules = factory.create_molecules_information(chain_model, DS, {"A": 1.0, "B": 1.2})
    molecules.add_polymer(0.3, DIBLOCK, {0: "G"})
    molecules.add_polymer(0.3, COMB)
    molecules.add_polymer(0.2, BRUSH)
    molecules.add_polymer(0.1, DIBLOCK)
    molecules.add_solvent(0.1, "A")
    optimizer = factory.create_propagator_computation_optimizer(molecules, aggregate)
    computation = factory.create_propagator_computation(box, molecules, optimizer, method)

    w, q_init = fields
    dtype = np.complex128 if field_type == "complex" else np.float64
    w = {m: w[m].astype(dtype) for m in w}
    q_init = {"G": q_init.astype(dtype)}
    computation.compute_propagators(w, q_init)
    # Right after the propagators, so that the call also sets what it computed
    computation.compute_concentrations_and_stress()
    combined = results(computation, 4, aggregate)
    computation.compute_concentrations()
    computation.compute_stress()
    separate = results(computation, 4, aggregate)
    return combined, separate


def check_no_stress(platform, reduced, fields):
    """A method without stress refuses the call before computing anything."""
    factory = PlatformSelector.create_factory(platform, reduced, "real")
    box = factory.create_computation_box(NX, LX)
    molecules = factory.create_molecules_information("continuous", DS, {"A": 1.0, "B": 1.2})
    molecules.add_polymer(1.0, DIBLOCK)
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    computation = factory.create_propagator_computation(box, molecules, optimizer, "cn-adi2")
    computation.compute_propagators(fields[0])
    try:
        computation.compute_concentrations_and_stress()
    except RuntimeError as exc:
        assert "does not support stress" in str(exc), str(exc)
    else:
        raise AssertionError(f"cn-adi2 computed the stress ({platform}, reduced={reduced})")
    try:
        computation.get_total_concentration("A")
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"cn-adi2 computed concentrations in a refused call ({platform}, reduced={reduced})")


def main():
    rng = np.random.default_rng(2026)
    n = int(np.prod(NX))
    fields = ({m: rng.normal(0, 5, n) for m in "AB"}, np.exp(rng.normal(0, 1, n)))
    for platform in PlatformSelector.avail_platforms():
        for chain_model, methods in STRESS_METHODS.items():
            for method in methods:
                # rs needs an orthogonal cell
                for angles in ([None] if method == "rs" else [None, OBLIQUE]):
                    for field_type in ("real", "complex"):
                        for aggregate in (False, True):
                            case = (platform, chain_model, method, field_type, aggregate, angles)
                            full = solve(platform, False, chain_model, method, field_type, aggregate, angles, fields)
                            reduced = solve(platform, True, chain_model, method, field_type, aggregate, angles, fields)
                            # One call equals the two; reduced memory equals full memory
                            for name, result in (("full combined", full[0]), ("reduced combined", reduced[0]),
                                                 ("reduced separate", reduced[1])):
                                for i, (a, b) in enumerate(zip(result, full[1])):
                                    assert a == b, (case, name, "differs from full separate in output", i)
        for reduced in (False, True):
            check_no_stress(platform, reduced, fields)
        print(f"Concentrations and stress: {platform} passed")
    if cuda_part_missing():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
