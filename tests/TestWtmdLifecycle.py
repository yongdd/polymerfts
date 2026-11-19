"""WTMD state across fresh runs, manual continuations, checkpoint restarts and a bias set on sim.wtmd.

Every key of the input "wtmd" reaches WTMD, built directly and by LFTS. A fresh run (from step 1) keeps
the bias and statistics the object holds, also a bias set on sim.wtmd before the first run or between two
runs, and drops the samples an earlier run left pending. The updates of the bias keep U' the derivative
of U. The WTMD state is Python: the first CPU platform runs the simulations.
"""
import copy
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import LFTS
from polymerfts.fts.wtmd import WTMD
from polymerfts.utils.checkpoint import validate_checkpoint
from fts_checks import compare_files, python_platform, quiet


def state(wtmd):
    return {"u": wtmd.u.copy(), "up": wtmd.up.copy(), "I0": wtmd.I0.copy(),
            "I1": {key: value.copy() for key, value in wtmd.I1.items()},
            "samples": list(wtmd.order_parameter_history),
            "dH": {key: list(values) for key, values in wtmd.dH_history.items()}}


def assert_state(wtmd, expected):
    for name in ("u", "up", "I0"):
        np.testing.assert_array_equal(getattr(wtmd, name), expected[name])
    assert wtmd.I1.keys() == expected["I1"].keys()
    for key in wtmd.I1:
        np.testing.assert_array_equal(wtmd.I1[key], expected["I1"][key])
    assert wtmd.order_parameter_history == expected["samples"]
    assert wtmd.dH_history == expected["dH"]


def constructor_state():
    wtmd = quiet(WTMD, {"psi_max": 2.0, "dpsi": 0.1}, [4, 4, 4], [1.3, 1.7, 2.1], 17.0, np.array([-15.0, 0.0]), [0])
    assert_state(wtmd, {"u": np.zeros(20), "up": np.zeros(20), "I0": np.zeros(20), "I1": {}, "samples": [], "dH": {}})
    # A fresh run drops pending samples and keeps whatever bias and statistics the object holds
    wtmd.I1["A,B"] = np.zeros(wtmd.bins)
    wtmd.u[:] = 7.0
    wtmd.up[:] = 8.0
    wtmd.I0[:] = 9.0
    wtmd.I1["A,B"][:] = 10.0
    wtmd.I1["another pair"] = np.ones(wtmd.bins)
    wtmd.order_parameter_history.append(0.4)
    wtmd.dH_history["A,B"] = [0.5]
    expected = dict(state(wtmd), samples=[], dH={})
    wtmd.start_fresh_run()
    assert_state(wtmd, expected)


def bias_updates():
    """After a few updates U' is the derivative of U: a hill adds its derivative times the well-tempering
    factor e^(-CV U/ΔT) and its height times the derivative of that factor, -CV U'/ΔT (update_statistics)."""
    wtmd = quiet(WTMD, {"psi_max": 2.0, "delta_t": 5.0, "sigma_psi": 0.16}, [4, 4, 4], [2.0] * 3, 10000.0,
                 np.array([-15.0, 0.0]), [0])
    random = np.random.default_rng(1)
    for _ in range(3):
        for psi in random.normal(1.0, 0.15, 500):
            wtmd.store_order_parameter(psi, {"A,B": 1.0})
        quiet(wtmd.update_statistics)
    inner = slice(50, -50)   # np.gradient is of first order at the ends
    error = np.max(np.abs(wtmd.up - np.gradient(wtmd.u, wtmd.dpsi))[inner]) / np.max(np.abs(wtmd.up))
    assert error < 1e-4, f"U' differs from dU/dpsi by {error:.2e} of max |U'|"


# Every key of the input "wtmd" with a value other than its default (two of them are the attributes l and dT)
WTMD_INPUT = {"ell": 2.5, "kc": 4.5, "delta_t": 3.0, "sigma_psi": 0.2, "psi_min": 0.5, "psi_max": 2.5,
              "dpsi": 0.05, "update_freq": 7, "recording_period": 21}
WTMD_ATTRIBUTES = {"ell": "l", "delta_t": "dT"}


def check_input(wtmd, label):
    """wtmd holds the values of WTMD_INPUT, and its settings, which checkpoints and files store, are those."""
    expected = {WTMD_ATTRIBUTES.get(key, key): value for key, value in WTMD_INPUT.items()}
    held = {name: getattr(wtmd, name) for name in expected}
    assert held == expected, (label, held)
    assert {**wtmd.settings(), "recording_period": wtmd.recording_period} == {**expected, "bins": 40}, label


def input_keys(root, platform):
    check_input(quiet(WTMD, WTMD_INPUT, [4, 4, 4], [1.3, 1.7, 2.1], 17.0, np.array([-15.0, 0.0]), [0]), "WTMD")
    params = parameters(root / "input_keys", platform)
    params["wtmd"] = dict(WTMD_INPUT)
    check_input(quiet(LFTS, params, random_seed=1).wtmd, "LFTS")


def parameters(directory, platform, steps=2):
    return {"nx": [4, 4, 4], "lx": [2.0, 2.0, 2.0], "ds": 0.05, "chain_model": "continuous",
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
            "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                {"type": "B", "length": 0.5}]}],
            "platform": platform, "langevin": {"nbar": 10000, "dt": 0.2, "max_step": steps},
            "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
            "saddle": {"max_iter": 200, "tolerance": 1e-6}, "verbose_level": 0,
            "recording": {"dir": str(directory), "checkpoint_interval": 1,
                          "sample_interval": 1, "statistics_save_interval": 2},
            "wtmd": {"update_freq": 3, "recording_period": 100, "dpsi": 0.1}}


def fields():
    random = np.random.default_rng(11)
    return {name: random.normal(0.0, 5.0, 64) for name in ("A", "B")}


def checkpoint(sim, step):
    return Path(sim.recording["dir"]) / f"fields_{step:06d}.mat"


def compare_checkpoints(left, right, step):
    compare_files(left.recording["dir"], right.recording["dir"], f"fields_{step:06d}.mat",
                  ("w_", "phi_", "normal_noise", "record_", "rng_state", "wtmd_"), exclude=("wtmd_params",))


def same_start(sim, directory, platform, steps):
    """A new simulation with the WTMD bias and statistics and the noise stream of sim."""
    other = quiet(LFTS, parameters(directory, platform, steps), random_seed=99)
    for name in ("u", "up", "I0"):
        setattr(other.wtmd, name, getattr(sim.wtmd, name).copy())
    other.wtmd.I1 = {key: value.copy() for key, value in sim.wtmd.I1.items()}
    other.random_bg.state = copy.deepcopy(sim.random_bg.state)
    return other


def repeated_runs(root, platform):
    """Fresh runs from step 1 (start_langevin_step None or 1), each by a new object."""
    for first_step in (None, 1):
        sim = quiet(LFTS, parameters(root / f"run_{first_step}", platform), random_seed=1)
        fresh = same_start(sim, root / f"fresh_{first_step}", platform, 2)   # the noise stream, not just the seed
        quiet(sim.run, fields(), start_langevin_step=first_step)
        quiet(fresh.run, fields())
        assert len(sim.wtmd.order_parameter_history) == 2
        assert_state(sim.wtmd, state(fresh.wtmd))
        assert sim.random_bg.state == fresh.random_bg.state
        for step in (1, 2):
            data = loadmat(checkpoint(sim, step), squeeze_me=True)
            validate_checkpoint(sim, data)
            assert np.size(data["wtmd_order_parameter_history"]) == step
            compare_checkpoints(sim, fresh, step)
    return sim


def manual_continuation(sim):
    data = loadmat(checkpoint(sim, 2), squeeze_me=True)
    before = state(sim.wtmd)
    update_statistics = sim.wtmd.update_statistics
    pending_at_update = []

    def update():
        pending_at_update.append(len(sim.wtmd.order_parameter_history))
        for name in ("u", "up", "I0"):
            np.testing.assert_array_equal(getattr(sim.wtmd, name), before[name])
        update_statistics()

    sim.langevin["max_step"] = 3
    with patch.object(sim.wtmd, "update_statistics", side_effect=update):
        quiet(sim.run, {name: data["w_" + name] for name in sim.monomer_types},
              normal_noise_prev=data["normal_noise_prev"], start_langevin_step=3)
    assert pending_at_update == [3], "The hill must include both pending samples of the preceding run"
    assert not sim.wtmd.order_parameter_history
    assert not sim.wtmd.dH_history
    assert np.any(sim.wtmd.u != before["u"])
    # A manual continuation with no remaining steps must also keep the accumulated bias.
    before = state(sim.wtmd)
    rng = copy.deepcopy(sim.random_bg.state)
    data = loadmat(checkpoint(sim, 3), squeeze_me=True)
    quiet(sim.run, {name: data["w_" + name] for name in sim.monomer_types},
          normal_noise_prev=data["normal_noise_prev"], start_langevin_step=4)
    assert_state(sim.wtmd, before)
    assert sim.random_bg.state == rng


def checkpoint_restart(root, platform):
    full = quiet(LFTS, parameters(root / "full", platform, 4), random_seed=13)
    quiet(full.run, fields())
    partial = quiet(LFTS, parameters(root / "partial", platform, 2), random_seed=13)
    quiet(partial.run, fields())
    continued = quiet(LFTS, parameters(root / "continued", platform, 4), random_seed=99)
    quiet(continued.run, fields())   # the recipient already has a bias and pending samples
    quiet(continued.continue_run, checkpoint(partial, 2))
    assert_state(continued.wtmd, state(full.wtmd))
    assert continued.random_bg.state == full.random_bg.state
    compare_checkpoints(continued, full, 4)

    # A fresh run after the restart keeps the restored bias and statistics, without the pending sample
    assert len(continued.wtmd.order_parameter_history) == 1 and np.any(continued.wtmd.I0 != 0.0)
    fresh = same_start(continued, root / "fresh_after_restart", platform, 4)
    quiet(continued.run, fields())
    quiet(fresh.run, fields())
    assert_state(continued.wtmd, state(fresh.wtmd))
    assert continued.random_bg.state == fresh.random_bg.state
    compare_checkpoints(continued, fresh, 4)


def preset_bias(root, platform):
    """A bias set on sim.wtmd before the first run is used by that run and kept by a later fresh run."""
    sim = quiet(LFTS, parameters(root / "preset", platform, 2), random_seed=5)
    unbiased = quiet(LFTS, parameters(root / "unbiased", platform, 2), random_seed=5)
    bins = sim.wtmd.bins
    sim.wtmd.u = np.linspace(0.0, 2.0, bins) / sim.wtmd.CV
    sim.wtmd.up = np.full(bins, 40.0) / sim.wtmd.CV
    preset = state(sim.wtmd)
    rng = copy.deepcopy(sim.random_bg.state)
    quiet(sim.run, fields())
    quiet(unbiased.run, fields())
    # update_freq 3 > 2 steps: the run deposits no hill, and its bias is the preset one
    for name in ("u", "up", "I0"):
        np.testing.assert_array_equal(getattr(sim.wtmd, name), preset[name])
    first = loadmat(checkpoint(sim, 2), squeeze_me=True)
    np.testing.assert_array_equal(first["wtmd_unscaled_up"], preset["up"])
    assert np.any(first["w_A"] != loadmat(checkpoint(unbiased, 2), squeeze_me=True)["w_A"]), \
        "The run did not feel the preset bias"
    # A later fresh run keeps the preset bias, not its two pending samples, and repeats the first run
    sim.random_bg.state = rng
    quiet(sim.run, fields())
    second = loadmat(checkpoint(sim, 2), squeeze_me=True)
    for key in ("w_A", "w_B", "wtmd_unscaled_u", "wtmd_unscaled_up", "wtmd_order_parameter_history"):
        np.testing.assert_array_equal(second[key], first[key], err_msg=key)


def bias_between_runs(root, platform):
    """A bias set between two fresh runs is the one the second run uses, together with the statistics
    of the first run, and the sample the first run left pending does not enter the hill of the second
    (started with start_langevin_step=1; preset_bias starts its second run with None)."""
    sim = quiet(LFTS, parameters(root / "between", platform, 4), random_seed=7)
    quiet(sim.run, fields())   # a hill at step 3, one pending sample from step 4
    assert len(sim.wtmd.order_parameter_history) == 1 and np.any(sim.wtmd.I0 != 0.0)
    sim.wtmd.up = np.full(sim.wtmd.bins, 40.0) / sim.wtmd.CV   # e.g. a bias read from a file
    new = same_start(sim, root / "between_new", platform, 4)
    quiet(sim.run, fields(), start_langevin_step=1)
    quiet(new.run, fields())
    assert_state(sim.wtmd, state(new.wtmd))
    assert sim.random_bg.state == new.random_bg.state
    for step in (1, 2, 3, 4):
        compare_checkpoints(sim, new, step)


def bias_at_intermediate_fields(root, platform):
    """The forces at the intermediate fields of a step (SaddlePointSearch.evaluate_force, which multistage
    integrators call) carry the WTMD bias of those fields."""
    sim = quiet(LFTS, parameters(root / "intermediate", platform), random_seed=3)
    random = np.random.default_rng(17)
    w_aux = random.normal(0.0, 5.0, (2, 64))
    trial = random.normal(0.0, 5.0, (1, 64))
    unbiased = quiet(sim.saddle.evaluate_force, trial, w_aux, 1)
    sim.wtmd.up = np.full(sim.wtmd.bins, 40.0) / sim.wtmd.CV
    biased = quiet(sim.saddle.evaluate_force, trial, w_aux, 1)
    fields = w_aux.copy()
    fields[sim.mpt.aux_fields_real_idx] = trial
    bias = np.zeros_like(unbiased)
    quiet(sim.wtmd.add_bias, fields, bias)
    assert np.max(np.abs(bias)) > 1e-3 * np.max(np.abs(unbiased)), (np.max(np.abs(bias)), np.max(np.abs(unbiased)))
    np.testing.assert_allclose(biased - unbiased, bias, rtol=0, atol=1e-9 * np.max(np.abs(bias)))


def samples_of_the_fields_before_the_step(root, platform):
    """Each WTMD sample holds Ψ and dH/dχN of the fields before its step, those at which Ψ and the bias of
    the step were computed. Three monomer types make dH/dχN depend on the concentrations too."""
    params = parameters(root / "samples", platform, 3)
    params.update(segment_lengths={"A": 1.0, "B": 1.0, "C": 1.0}, chi_n={"A,B": 15.0, "B,C": 15.0, "A,C": 30.0},
                  polymers=[{"volume_fraction": 1.0, "blocks": [
                      {"type": "A", "length": 0.3}, {"type": "B", "length": 0.4}, {"type": "C", "length": 0.3}]}])
    params["wtmd"]["update_freq"] = 10      # the run keeps all its samples
    accepted = []
    sim = quiet(LFTS, params, random_seed=19,
                callbacks={"accepted": lambda sample: accepted.append((np.array(sample["w_aux"]),
                                                                        dict(sample["phi"])))})
    random = np.random.default_rng(23)
    quiet(sim.run, {name: random.normal(0.0, 5.0, 64) for name in ("A", "B", "C")})
    assert len(sim.wtmd.order_parameter_history) == 3
    # Step n starts from the state that step n - 1 accepted
    for step in (2, 3):
        w_aux, phi = accepted[step - 2]
        assert sim.wtmd.order_parameter_history[step - 1] == sim.wtmd.order_parameter(w_aux), step
        dH = sim.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=sim.smearing.apply_to_dict(phi), wtmd=True)
        # The concentrations after the step would give other values
        after = sim.mpt.compute_h_deriv_chin(w_aux=w_aux, phi=sim.smearing.apply_to_dict(accepted[step - 1][1]),
                                             wtmd=True)
        assert any(abs(after[key] - dH[key]) > 1e-6 * abs(dH[key]) for key in sim.chi_n), (step, after, dH)
        for key in sim.chi_n:
            stored = sim.wtmd.dH_history[key][step - 1]
            assert abs(stored - dH[key]) <= 1e-12 * abs(dH[key]), (step, key, stored, dH[key])


def main():
    constructor_state()
    bias_updates()
    print("[PASS] WTMD constructor state, fresh start and the derivative of the bias")
    platform = python_platform()
    with tempfile.TemporaryDirectory(prefix="polymerfts_wtmd_lifecycle_") as directory:
        root = Path(directory)
        input_keys(root, platform)
        sim = repeated_runs(root, platform)
        manual_continuation(sim)
        checkpoint_restart(root, platform)
        preset_bias(root, platform)
        bias_between_runs(root, platform)
        bias_at_intermediate_fields(root, platform)
        samples_of_the_fields_before_the_step(root, platform)
        print(f"[PASS] {platform}: WTMD input keys, fresh runs, manual continuation, checkpoint restart, "
              "preset bias, a bias set between runs, the bias of intermediate forces and the fields of the samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
