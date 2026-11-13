#!/usr/bin/env python3
"""SCFT validates original initial fields before casting or resetting the solver.

Complex and nonfinite values, invalid sizes and missing monomers fail before
the optimizer or propagators are touched, including axis and space-group
conversions. Valid real arrays remain unchanged and preserve their first
iteration fields; a homogeneous control converges on every backend.
"""

import contextlib
import copy
import io
import os
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import SCFT, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing


def quiet(call):
    with contextlib.redirect_stdout(io.StringIO()):
        return call()


def parameters(platform, group=False):
    params = {
        "nx": [8, 8, 8] if group else [4, 6],
        "lx": [2.0, 2.0, 2.0] if group else [1.5, 2.0],
        "chain_model": "continuous", "ds": 0.1,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 3.0},
        "polymers": [{"volume_fraction": 1.0,
                      "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": platform, "relax_cell": False, "max_iter": 1, "tolerance": 1e-9,
        "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                      "mix_min": 0.1, "mix_init": 0.1},
    }
    if group:
        params["space_group"] = {"symbol": "Im-3m", "hall_number": 529}
    return params


def snapshot(fields):
    return {name: (value.shape, value.dtype, value.flags.writeable, value.tobytes())
            for name, value in fields.items()}


def invalid(sim, fields, *messages):
    before = snapshot(fields)
    # The native optimizer's methods are read-only, so replace its owner for
    # this validation check. A rejected run must leave the existing history.
    with patch.object(sim, "field_optimizer") as optimizer, \
            patch.object(sim, "compute_concentrations") as compute:
        optimizer.reset_count.side_effect = AssertionError("reset optimizer for invalid fields")
        compute.side_effect = AssertionError("computed concentrations for invalid fields")
        try:
            quiet(lambda: sim.run(fields, return_result=True))
        except ValidationError as error:
            assert all(message in str(error) for message in messages), str(error)
        else:
            raise AssertionError(f"Accepted invalid initial fields: expected {messages!r}")
        optimizer.reset_count.assert_not_called()
        compute.assert_not_called()
    assert snapshot(fields) == before, "validation changed a caller's array"


def check_inputs(sim, shape):
    zeros = np.zeros(shape)
    bad_values = []
    for imaginary in (0.0, 5.0, np.nan):
        field = zeros.astype(np.complex128)
        field.imag.flat[0] = imaginary
        bad_values.append((field, "real, finite"))
    for value in (np.nan, np.inf, -np.inf):
        field = zeros.copy()
        field.flat[0] = value
        bad_values.append((field, "real, finite"))
    bad_values += [(np.zeros(zeros.size - 1), "needs"),
                   (np.full(shape, "not a number"), "real, finite")]
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        bad_values.append((np.full(shape, np.finfo(np.longdouble).max, dtype=np.longdouble),
                           "float64 precision"))
    for bad, message in bad_values:
        fields = {"A": bad, "B": zeros.copy()}
        for value in fields.values():
            value.setflags(write=False)
        invalid(sim, fields, "initial_fields['A']", message)
    invalid(sim, {"B": zeros.copy()}, "missing monomer 'A'")


def valid_run(sim, fields):
    for value in fields.values():
        value.setflags(write=False)
    before = snapshot(fields)
    result = quiet(lambda: sim.run(fields, return_result=True))
    assert snapshot(fields) == before, "SCFT changed a caller's initial field"
    assert quiet(sim.prop_solver.check_total_partition)
    np.testing.assert_allclose(np.mean(result.phi["A"] + result.phi["B"]), 1.0, rtol=0, atol=2e-11)
    return result


def check_branch(platform, group, permute):
    standard = parameters(platform, group)
    shape = tuple(standard["nx"])
    params = copy.deepcopy(standard)
    order = tuple(reversed(range(len(shape)))) if permute else tuple(range(len(shape)))
    if permute:
        params["axis_ordering"] = [["a", "b", "c"][i] for i in order]
        for key in ("nx", "lx"):
            params[key] = [standard[key][i] for i in order]
    sim = quiet(lambda: SCFT(params))
    check_inputs(sim, tuple(params["nx"]))

    # A nonuniform real start survives conversion and only loses its mean,
    # as required by the incompressible pressure convention. Use a profile
    # invariant under Im-3m so symmetry averaging cannot hide lost values.
    axes = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in shape], indexing="ij")
    field = sum(np.cos(4 * np.pi * axis) for axis in axes)
    field = 5.0 * field / np.std(field)
    expected = field - np.mean(field)
    user_field = field.transpose(order).copy()
    result = valid_run(sim, {"A": user_field, "B": -user_field})
    np.testing.assert_allclose(result.w["A"].reshape(shape), expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(result.w["B"].reshape(shape), -expected, rtol=0, atol=2e-12)

    # Ordinary integer and boolean guesses retain their previous acceptance.
    # Constant fields become zero under the same gauge normalization and give
    # the analytic homogeneous state with Q=1 and phi_A=phi_B=1/2.
    homogeneous = valid_run(sim, {"A": np.ones(params["nx"], dtype=np.int64),
                                  "B": np.ones(int(np.prod(params["nx"])), dtype=np.bool_)})
    assert homogeneous.converged
    np.testing.assert_allclose(homogeneous.partition_functions, [1.0], rtol=0, atol=2e-12)
    for name in ("A", "B"):
        np.testing.assert_allclose(homogeneous.phi[name], 0.5, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(homogeneous.w[name].reshape(shape), np.zeros(shape))
    print(f"[PASS] {platform}: initial fields (space group={group}, axis permutation={permute})")


def main():
    for platform in PlatformSelector.avail_platforms():
        for group, permute in ((False, False), (False, True), (True, False), (True, True)):
            check_branch(platform, group, permute)
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
