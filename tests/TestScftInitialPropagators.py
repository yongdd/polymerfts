#!/usr/bin/env python3
"""SCFT graft profiles on full and reduced grids, with input axis permutations.

Nonuniform profiles with the symmetry of Pmmm, Im-3m and P-1 give the same
one-iteration state as the existing reduced-input interface and a full-grid
solver. Test each available backend, including physical and irreducible bases.
Full profiles that break the group symmetry are rejected instead of changing
the graft distribution. No conversion may modify the caller's arrays.
"""

import contextlib
import copy
import io
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import SCFT, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing


def quiet(call):
    with contextlib.redirect_stdout(io.StringIO()):
        return call()


def parameters(platform, symbol, nx, angles):
    return {
        "nx": nx, "lx": [2.0] * 3 if symbol == "Im-3m" else [1.7, 2.0, 2.3],
        "angles": angles, "chain_model": "continuous", "ds": 0.05,
        "segment_lengths": {"A": 1.0, "B": 1.2}, "chi_n": {"A,B": 15.0},
        "polymers": [{"volume_fraction": 1.0, "initial_conditions": {0: "G"},
                      "blocks": [{"type": "A", "length": 0.4, "v": 0, "u": 1},
                                 {"type": "B", "length": 0.6, "v": 1, "u": 2}]}],
        "space_group": {"symbol": symbol}, "platform": platform,
        "relax_cell": False, "max_iter": 1, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                      "mix_min": 0.1, "mix_init": 0.1},
    }


def permuted(params, fields, profile, order, reduced=False):
    """Express a standard-axis input in the user's order, preserving reduced q."""
    params = copy.deepcopy(params)
    shape = tuple(params["nx"])
    params["axis_ordering"] = [["a", "b", "c"][i] for i in order]
    for name in ("nx", "lx"):
        params[name] = [params[name][i] for i in order]
    angle_index = {(1, 2): 0, (0, 2): 1, (0, 1): 2}
    params["angles"] = [params["angles"][angle_index[tuple(sorted((order[i], order[j])))]]
                        for i, j in ((1, 2), (0, 2), (0, 1))]
    fields = {name: value.reshape(shape).transpose(order).copy() for name, value in fields.items()}
    profile = profile.copy() if reduced else profile.reshape(shape).transpose(order).copy()
    return params, fields, profile


def run(sim, fields, profile):
    before = profile.copy()
    profile.setflags(write=False)
    result = quiet(lambda: sim.run(fields, q_init={"G": profile}, return_result=True))
    np.testing.assert_array_equal(profile, before)
    assert quiet(sim.prop_solver.check_total_partition), "forward/backward partition mismatch"
    # Grafting changes the spatial profile, but not canonical chain amounts.
    np.testing.assert_allclose([np.mean(result.phi[name]) for name in ("A", "B")],
                               [0.4, 0.6], rtol=0, atol=2e-11)
    return result


def same_state(actual, expected):
    for kind in ("phi", "w"):
        for name in ("A", "B"):
            np.testing.assert_allclose(getattr(actual, kind)[name], getattr(expected, kind)[name],
                                       rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual.partition_functions, expected.partition_functions,
                               rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual.free_energy_absolute, expected.free_energy_absolute,
                               rtol=2e-10, atol=2e-10)


def expect_invalid(sim, fields, profile, message):
    before = profile.copy()
    try:
        quiet(lambda: sim.run(fields, q_init={"G": profile}, return_result=True))
    except ValidationError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"q_init accepted: expected {message!r}")
    np.testing.assert_array_equal(profile, before)


def check_group(platform, symbol, nx, angles):
    params = parameters(platform, symbol, nx, angles)
    control = quiet(lambda: SCFT(params))
    sg = control.sg
    assert control.prop_solver.n_grid < control.prop_solver.total_grid
    if symbol == "P-1":
        assert sg.physical_basis_name() == "irreducible"
    else:
        assert sg.physical_basis_name() == "pmmm"
    if symbol == "Im-3m":
        assert sg.get_n_reduced_basis() > sg.get_n_reduced_basis_full()

    coordinates = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in nx], indexing="ij")
    profile = 1.0 + sum(weight * np.cos(2 * np.pi * axis)
                        for weight, axis in zip((0.15, 0.1, 0.05), coordinates))
    profile += 0.2 * np.cos(4 * np.pi * coordinates[0])
    profile = sg.symmetrize(profile.ravel())
    assert np.std(profile) > 0.01 and np.min(profile) > 0.0
    rng = np.random.default_rng(40)
    field = sg.symmetrize(rng.normal(size=int(np.prod(nx))))
    field *= 5.0 / np.std(field)
    fields = {"A": field, "B": -field}
    reduced = sg.to_reduced_basis(profile)
    expected = run(control, fields, reduced.copy())

    plain = copy.deepcopy(params)
    plain.pop("space_group")
    same_state(run(quiet(lambda: SCFT(plain)), fields, profile.copy()), expected)
    same_state(run(quiet(lambda: SCFT(params)), fields, profile.copy()), expected)

    # Identity axis_ordering used to fail for both full and reduced q. The cycle
    # permutation also catches transposing a reduced physical basis a second time.
    for order in ((0, 1, 2), (2, 0, 1)):
        for use_reduced in (False, True):
            p, w, q = permuted(params, fields, reduced if use_reduced else profile, order, use_reduced)
            sim = quiet(lambda: SCFT(p))
            same_state(run(sim, w, q), expected)

    # Invalid full profiles must not be sampled or averaged into another model.
    broken = profile.copy()
    broken[0] += 0.25
    expect_invalid(control, fields, broken, "symmetry")
    expect_invalid(control, fields, np.ones(control.prop_solver.n_grid + 1), "needs")
    expect_invalid(control, fields, profile.astype(complex) + 0.1j, "real, finite")
    nonfinite = profile.copy()
    nonfinite[0] = np.nan
    expect_invalid(control, fields, nonfinite, "real, finite")
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        overflowing = np.full(profile.shape, np.finfo(np.longdouble).max, dtype=np.longdouble)
        expect_invalid(control, fields, overflowing, "float64 precision")
    print(f"[PASS] {platform} {symbol}: full/reduced graft profiles and axis ordering")


def main():
    for platform in PlatformSelector.avail_platforms():
        for symbol, nx, angles in (
            ("Pmmm", [8, 10, 12], [90.0, 90.0, 90.0]),
            ("Im-3m", [8, 8, 8], [90.0, 90.0, 90.0]),
            ("P-1", [8, 10, 6], [80.0, 100.0, 110.0]),
        ):
            check_group(platform, symbol, nx, angles)
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
