#!/usr/bin/env python3
"""Gibbs-ensemble CL-FTS (gibbs_ensemble.py): two boxes exchange chains, point ions and volume.

Two homopolymers A and B (discrete chains, N = 20) on 4x4 grids:
1. Input: grand-canonical polymers, volume fractions that do not sum to 1, a cell of its own for a
   box, a missing or invalid cell, a phase without volume, a species absent from a box, a charged box
   that is not neutral, skip_steps >= cycle_steps, a zero sample interval, point ions of continuous
   chains without n_segments and a negative seed are rejected. A box without a volume takes that of
   the cell.
2. Mean-field limit (nbar 1e28) of the symmetric blend at chi N = 3: the boxes reach the
   Flory-Huggins binodal ln(phi/(1 - phi)) = chi N (2 phi - 1).
3. Mean-field limit with charged A chains (one charge per chain), Cl- counterions and Na+ Cl- salt
   as canonical point ions. The coexistence of the Flory-Huggins free energy with ideal ions,
   sum_b [sum_k n_k (ln(n_k/V_b) - 1) + V_b chi N phi_A phi_B], is solved independently for the
   neutral transfers (B, NaCl, A with its Cl). Started there, the exchange potentials of the
   boxes agree in every neutral combination and the amounts stay; started from distant boxes,
   the boxes stay neutral and approach it.
4. Mean-field limit with charged A chains and grand-canonical Na+ Cl- salt (no canonical ions):
   the Donnan potential of each box, sinh(psi) = rho_A/(2 e^mu), lets the boxes carry their net
   charge. The coexistence of the same free energy with these potentials is solved independently;
   started there, every exchange potential agrees and the amounts stay, and from distant boxes
   the amounts approach it.
5. Mean-field limit of a chain with a solvent of one segment (chi N = 18): from boxes with equal volumes,
   far from the lever rule, the boxes reach both branches of the Flory-Huggins binodal, solved
   independently, within 60 cycles. The volumes of the phases have to change by a factor of about
   two, which the mobility of the ideal mixture does at the same rate as the composition, while
   the cell of the boxes stays.
6. Fluctuating blend (nbar 1e4): three cycles continued by continue_run() reproduce six
   uninterrupted ones bit for bit (history and fields), and a second run() starts again from the
   boxes of the input; a continuation that lists the point ions in another order, or of a history
   with a cell for each box, is refused.
7. Mobility: the step of boxes with point ions and unequal chain lengths, with and without the
   projection on neutral moves, against the stated mobility, and the bounded step of boxes of
   equal composition, where the Hessians of the boxes are singular.
"""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq, fsolve
from polymerfts import CLFTS, GibbsEnsemble, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing

N, CHI = 20, 3.0


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(directory, cycles, nbar=1e28, fractions=((0.7, 0.3), (0.3, 0.7)), charge=0.0, ions=None,
               volumes=(1.0, 1.0)):
    params = {"nx": [4, 4], "lx": [1.0, 1.0], "ds": 1 / N, "chain_model": "discrete",
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": CHI}, "zeta_n": 50.0,
              "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 1.0}]},
                           {"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 1.0}]}],
              "platform": cpu_platform(), "langevin": {"nbar": nbar, "dt": 0.5, "max_step": 1},
              "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                            "statistics_save_interval": 1},
              "verbose_level": 0,
              "gibbs_ensemble": {"boxes": [{"volume": volume, "volume_fractions": list(phi)}
                                           for volume, phi in zip(volumes, fractions)],
                                 "cycles": cycles, "cycle_steps": 20, "skip_steps": 10, "exchange_rate": 1.0}}
    if charge:
        params["smearing"] = {"type": "gaussian", "a_int": 0.1}
        params["electrostatics"] = {"charges": {"A": charge}, "coupling": 100.0, "ions": ions}
    return params


def ideal_mobility(amounts, alphas):
    """(H^I + H^II + λ D)⁻¹ with the Hessian of the ideal mixture of each box, H = d²F/dn² for
    F = sum_i n_i ln(n_i/V), V = sum_i alpha_i n_i (docs/theory/ComplexLangevinSimulation.md, Section 8)."""
    hessian = np.diag(0.03 * (1 / amounts[0] + 1 / amounts[1]))
    for n in amounts:
        volume = alphas @ n
        rows = np.ones(len(n))[:, None] * alphas
        hessian = hessian + np.diag(1 / n) - (rows + rows.T) / volume + np.outer(alphas, alphas) * n.sum() / volume ** 2
    return np.linalg.inv(hessian)


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def expect_validation_error(make, message):
    try:
        quiet(make)
    except ValidationError as error:
        assert message in str(error), f"expected '{message}', got '{error}'"
        return
    raise AssertionError(f"expected a ValidationError with '{message}'")


def salt(concentration=0.01):
    return {"Na": {"valence": 1.0, "concentration": concentration, "smearing_length": 0.1},
            "Cl": {"valence": -1.0, "concentration": concentration, "smearing_length": 0.1}}


def inputs(root):
    params = parameters(root / "inputs", 2)
    params["polymers"][1] = {"chemical_potential": 0.0, "blocks": [{"type": "B", "length": 1.0}]}
    expect_validation_error(lambda: GibbsEnsemble(params), "grand-canonical")
    params = parameters(root / "inputs", 2, fractions=((0.7, 0.4), (0.3, 0.7)))
    expect_validation_error(lambda: GibbsEnsemble(params), "must sum to 1")
    params = parameters(root / "inputs", 2)
    params["gibbs_ensemble"]["skip_steps"] = 20
    expect_validation_error(lambda: GibbsEnsemble(params), "smaller than 'cycle_steps'")
    params = parameters(root / "inputs", 2)
    params["gibbs_ensemble"]["boxes"][0]["lx"] = [1.0, 1.0]
    expect_validation_error(lambda: GibbsEnsemble(params), "'lx' is the cell of both boxes")
    params = parameters(root / "inputs", 2)
    params.pop("lx")
    expect_validation_error(lambda: GibbsEnsemble(params), "Missing required parameter: 'lx'")
    for lx in (2.0, None, [1.0], ["a", 1.0], [-1.0, 1.0], [np.nan, 1.0]):
        params = parameters(root / "inputs", 2)
        params["lx"] = lx
        expect_validation_error(lambda: GibbsEnsemble(params), "lx")
    # Without a volume a box takes that of the cell, with its angles
    params = parameters(root / "inputs", 2)
    params.update(lx=[2.0, 1.5], angles=[60.0])
    for box in params["gibbs_ensemble"]["boxes"]:
        box.pop("volume")
    gibbs = quiet(lambda: GibbsEnsemble(params))
    np.testing.assert_allclose(gibbs.amounts @ gibbs.alphas, 3.0 * np.sin(np.pi / 3), rtol=1e-12, atol=0)
    params.update(lx=np.array([2.0, 1.5]), angles=np.array([90.0, 90.0, 60.0]))
    gibbs = quiet(lambda: GibbsEnsemble(params))
    np.testing.assert_allclose(gibbs.amounts @ gibbs.alphas, 3.0 * np.sin(np.pi / 3), rtol=1e-12, atol=0)
    params = parameters(root / "inputs", 2, volumes=(1.0, 0.0))
    expect_validation_error(lambda: GibbsEnsemble(params), "boxes[1].volume")
    params = parameters(root / "inputs", 2, charge=0.05, ions=salt())
    params["gibbs_ensemble"]["boxes"][0]["ion_concentrations"] = {"Na": 0.01, "Cl": 0.01}
    expect_validation_error(lambda: GibbsEnsemble(params), "not electroneutral")
    params = parameters(root / "inputs", 2, fractions=((1.0, 0.0), (0.3, 0.7)))
    expect_validation_error(lambda: GibbsEnsemble(params), "positive amount in both boxes")
    params = parameters(root / "inputs", 2)
    params["recording"]["sample_interval"] = 0
    expect_validation_error(lambda: GibbsEnsemble(params), "'recording.sample_interval' must be an integer")
    params = parameters(root / "inputs", 2, charge=0.05, ions=salt())
    params["chain_model"] = "continuous"
    expect_validation_error(lambda: GibbsEnsemble(params), "n_segments")
    expect_validation_error(lambda: GibbsEnsemble(parameters(root / "inputs", 2), random_seed=-5), "non-negative integer")


def mobility(root):
    """The step of the exchange against the stated mobility, without running a box."""
    def step(params, potentials):
        gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=1))
        move, _ = gibbs._exchange_step([np.asarray(potentials), np.zeros(len(potentials))])
        return gibbs, move

    # Point ions (no volume) and chains of unequal lengths: every species moves with screening
    # (Donnan), only neutral combinations without it
    difference = np.array([2e-3, -1e-3, 3e-3, 1.5e-3])
    for screened in (True, False):
        params = parameters(root / "mobility", 1, charge=0.05, ions=salt())
        params["polymers"][1]["blocks"][0]["length"] = 0.5
        if screened:
            params["electrostatics"]["screening_length"] = 0.5
        gibbs, move = step(params, difference)
        assert gibbs.neutral != screened
        np.testing.assert_allclose(gibbs.alphas, [1.0, 0.5], rtol=1e-12, atol=0)
        weights = gibbs.exchange_rate * ideal_mobility(gibbs.amounts, np.concatenate([gibbs.alphas, np.zeros(2)]))
        expected = -weights @ difference
        if not screened:
            z = gibbs.charges
            expected = -(weights - np.outer(weights @ z, z @ weights) / (z @ weights @ z)) @ difference
            np.testing.assert_allclose(z @ move, 0.0, rtol=0, atol=1e-14)
        assert np.max(np.abs(expected) / gibbs.amounts.min(axis=0)) < 0.5, "Unexpected step limiter in this case"
        np.testing.assert_allclose(move, expected, rtol=1e-10, atol=1e-14)

    # Boxes of equal composition: the Hessians alone are singular along the amounts
    gibbs, move = step(parameters(root / "mobility", 1, fractions=((0.5, 0.5), (0.5, 0.5))), [1e-3, 1e-3])
    np.testing.assert_allclose(move, -gibbs.exchange_rate * ideal_mobility(gibbs.amounts, gibbs.alphas) @ [1e-3, 1e-3],
                               rtol=1e-10, atol=0)
    assert np.max(np.abs(move)) < 0.01, f"step of equal boxes {move}"


def blend(root):
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "blend", 20), random_seed=1))
    quiet(gibbs.run)
    last = gibbs.history["cycles"][-1]["boxes"]
    binodal = brentq(lambda p: np.log(p / (1 - p)) - CHI * (2 * p - 1), 0.51, 0.999, xtol=1e-14)
    for b, expected in ((0, binodal), (1, 1 - binodal)):
        phi = last[b]["volume_fractions"][0]
        assert abs(phi - expected) < 2e-6, f"box {b}: phi_A {phi:.8f} against the binodal {expected:.8f}"
    assert abs(last[0]["volume"] - 1.0) < 1e-6 and abs(last[0]["volume"] + last[1]["volume"] - 2.0) < 1e-12, "volumes"


def tie_line(root):
    chi_n = 18.0

    def free_energy(phi):
        return phi * np.log(phi) + N * (1 - phi) * np.log(1 - phi) + chi_n * phi * (1 - phi)

    def derivative(phi):
        return np.log(phi) + 1 - N * np.log(1 - phi) - N + chi_n * (1 - 2 * phi)

    def tangent(x):
        a, b = np.exp(x) / (1 + np.exp(x))   # the logit keeps 0 < phi < 1
        return [derivative(a) - derivative(b), free_energy(a) - a * derivative(a) - free_energy(b) + b * derivative(b)]
    expected = fsolve(tangent, [np.log(0.02 / 0.98), 0.0], xtol=1e-13)
    expected = np.exp(expected) / (1 + np.exp(expected))
    params = parameters(root / "tie_line", 60, fractions=((0.1, 0.9), (0.5, 0.5)))
    params.update(segment_lengths={"A": 1.0, "S": 1.0}, chi_n={"A,S": chi_n})
    params["polymers"][1]["blocks"] = [{"type": "S", "length": 1 / N}]
    gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=1))
    quiet(gibbs.run)
    boxes = gibbs.history["cycles"][-1]["boxes"]
    for b in range(2):
        phi = boxes[b]["volume_fractions"][0]
        assert abs(phi - expected[b]) < 1e-5, f"box {b}: phi {phi:.7f} against the binodal {expected[b]:.7f}"
    assert abs(boxes[0]["volume"] / (2 * (expected[1] - 0.3) / (expected[1] - expected[0])) - 1) < 1e-4, "lever rule"
    # The volumes are those of the phases: the cell of both boxes stays
    for b in range(2):
        assert gibbs.box_params(b, gibbs.amounts[b], 60)["lx"] == params["lx"], f"box {b}: cell"


def coexistence():
    """Box I amounts (A, B, Na, Cl) per sqrt(Nbar) at the mean-field coexistence of 1 A, 1 B, 0.4 Na and
    1.4 Cl in total (A carries one charge), from the exchange potentials of the free energy."""
    total = np.array([1.0, 1.0, 0.4, 1.4])

    def potentials(n):
        volume = n[0] + n[1]
        phi_a, phi_b = n[0] / volume, n[1] / volume
        pressure = np.sum(n) / volume + CHI * phi_a * phi_b
        mu = np.log(n / volume)
        return mu + np.array([CHI * phi_b - pressure, CHI * phi_a - pressure, 0.0, 0.0])

    def neutral_differences(x):
        n = np.array([x[0], x[1], x[2], x[2] + x[0]])
        d = potentials(n) - potentials(total - n)
        return [d[1], d[2] + d[3], d[0] + d[3]]
    x = fsolve(neutral_differences, [0.82, 0.15, 0.12], xtol=1e-14)
    assert np.max(np.abs(neutral_differences(x))) < 1e-12
    box = np.array([x[0], x[1], x[2], x[2] + x[0]])
    return np.array([box, total - box])


def neutral_combinations(potentials):
    """Exchange potentials of the neutral transfers B, NaCl and A + Cl (A carries one charge)."""
    p = np.asarray(potentials)
    return np.array([p[1], p[2] + p[3], p[0] + p[3]])


def charged(root):
    """A chains carry one charge (z_A = 0.05 per segment, N = 20); amounts per sqrt(Nbar): n_k^b."""
    expected = coexistence()
    volumes = expected[:, 0] + expected[:, 1]
    fractions = [tuple(n[:2] / v) for n, v in zip(expected, volumes)]
    params = parameters(root / "charged_start", 3, charge=0.05, ions=salt(), fractions=fractions, volumes=volumes)
    for b in range(2):
        params["gibbs_ensemble"]["boxes"][b]["ion_concentrations"] = {
            "Na": expected[b, 2] / (N * volumes[b]), "Cl": expected[b, 3] / (N * volumes[b])}
    gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=1))
    quiet(gibbs.run)
    for cycle in gibbs.history["cycles"]:
        boxes = cycle["boxes"]
        difference = neutral_combinations(boxes[0]["exchange_potentials"]) - \
            neutral_combinations(boxes[1]["exchange_potentials"])
        assert np.max(np.abs(difference)) < 1e-6, f"cycle {cycle['cycle']}: neutral exchange differences {difference}"
    assert np.max(np.abs(gibbs.amounts - expected)) < 1e-6, "amounts moved away from the coexistence"

    # From distant boxes: neutral at every cycle, approaching the coexistence
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "charged", 30, charge=0.05, ions=salt()), random_seed=1))
    quiet(gibbs.run)
    charges = np.array([1.0, 0.0, 1.0, -1.0])
    distances = []
    for cycle in gibbs.history["cycles"]:
        amounts = np.array(cycle["amounts_after"])
        assert np.all(np.abs(amounts @ charges) < 1e-12), "boxes are not neutral"
        distances.append(np.max(np.abs(amounts - expected)))
    assert distances[-1] < 0.05 * distances[0] and all(b <= a * 1.001 for a, b in zip(distances, distances[1:])), \
        f"distance to the coexistence: {distances[0]:.2e} -> {distances[-1]:.2e}"


def donnan_coexistence(salt_density):
    """Box I amounts (A, B) per sqrt(Nbar) at the mean-field coexistence of 1 A and 1 B in total with
    grand-canonical salt of number density e^mu = salt_density in its reservoir (A carries one charge).
    The ions in a box follow its Donnan potential psi (per charge): densities e^(mu -+ psi), with
    sinh(psi) = rho_A/(2 e^mu) for neutrality; A gains psi in its chemical potential."""
    total = np.array([1.0, 1.0])

    def potentials(n):
        volume = n[0] + n[1]
        rho_a, rho_b = n / volume
        psi = np.arcsinh(rho_a / (2 * salt_density))
        pressure = rho_a + rho_b + 2 * salt_density * np.cosh(psi) + CHI * rho_a * rho_b
        return np.array([np.log(rho_a) + CHI * rho_b + psi - pressure, np.log(rho_b) + CHI * rho_a - pressure])
    x = fsolve(lambda x: potentials(x) - potentials(total - x), [0.85, 0.15], xtol=1e-13)
    assert np.max(np.abs(potentials(x) - potentials(total - x))) < 1e-12
    return np.array([x, total - x])


def donnan(root):
    """Charged A chains (one charge per chain) with grand-canonical Na+ Cl- of reservoir density 0.5."""
    ions = {"Na": {"valence": 1.0, "chemical_potential": np.log(0.5), "smearing_length": 0.1},
            "Cl": {"valence": -1.0, "chemical_potential": np.log(0.5), "smearing_length": 0.1}}
    expected = donnan_coexistence(0.5)
    volumes = expected.sum(axis=1)
    fractions = [tuple(n / v) for n, v in zip(expected, volumes)]
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "donnan_start", 3, charge=0.05, ions=ions,
                                                   fractions=fractions, volumes=volumes), random_seed=1))
    quiet(gibbs.run)
    for cycle in gibbs.history["cycles"]:
        boxes = cycle["boxes"]
        difference = np.subtract(boxes[0]["exchange_potentials"], boxes[1]["exchange_potentials"])
        assert np.max(np.abs(difference)) < 1e-6, f"cycle {cycle['cycle']}: exchange differences {difference}"
    assert np.max(np.abs(gibbs.amounts - expected)) < 1e-6, "amounts moved away from the Donnan coexistence"

    # From distant boxes
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "donnan", 20, charge=0.05, ions=ions), random_seed=1))
    quiet(gibbs.run)
    distances = [np.max(np.abs(np.array(cycle["amounts_after"]) - expected)) for cycle in gibbs.history["cycles"]]
    assert distances[-1] < 1e-3 * distances[0], f"distance to the coexistence: {distances[0]:.2e} -> {distances[-1]:.2e}"


def restart(root):
    def make(directory, cycles):
        return quiet(lambda: GibbsEnsemble(parameters(directory, cycles, nbar=1e4), random_seed=3))
    full = make(root / "full", 6)
    quiet(full.run)
    part = make(root / "part", 3)
    quiet(part.run)
    part = make(root / "part", 6)
    quiet(part.continue_run)
    a = json.loads((root / "full" / "gibbs_history.json").read_text())["cycles"]
    b = json.loads((root / "part" / "gibbs_history.json").read_text())["cycles"]
    assert a == b, "continued history differs from the uninterrupted one"
    for box in ("box0", "box1"):
        x = loadmat(root / "full" / box / "fields_000120.mat", squeeze_me=True)
        y = loadmat(root / "part" / box / "fields_000120.mat", squeeze_me=True)
        keys = [k for k in x if k.startswith("w_")]
        assert keys and all(np.array_equal(x[k], y[k]) for k in keys), f"{box}: continued fields differ"
    # A second run() starts again from the boxes of the input and replaces the history
    quiet(part.run)
    c = json.loads((root / "part" / "gibbs_history.json").read_text())["cycles"]
    assert c == a, "a second run() differs from the first"

    # A history with a cell for each box does not continue in the cell of both boxes
    path = root / "part" / "gibbs_history.json"
    old = json.loads(path.read_text())
    for box in old["settings"]["gibbs_ensemble"]["boxes"]:
        box["lx"] = [1.0, 1.0]
    path.write_text(json.dumps(old))
    expect_validation_error(lambda: make(root / "part", 6).continue_run(), "a cell of its own")

    # The amounts and chemical potentials follow the order of the point ions, which a continuation
    # must keep, grand-canonical ones included
    ions = dict(salt(), K={"valence": 1.0, "chemical_potential": np.log(N * 0.005), "smearing_length": 0.1})
    def charged(cycles, ions):
        params = parameters(root / "ions", cycles, charge=0.05, ions=ions)
        params["electrostatics"]["screening_length"] = 0.5
        return quiet(lambda: GibbsEnsemble(params, random_seed=3))
    quiet(charged(1, ions).run)
    for order in (("Cl", "Na", "K"), ("K", "Na", "Cl")):
        expect_validation_error(lambda: charged(2, {name: ions[name] for name in order}).continue_run(), "another order")
    expect_validation_error(lambda: charged(2, salt()).continue_run(), "input differs")


def snapped_parameters(directory, cycles, platform):
    """A native-accepted one-bead input offset that is significant relative to ds."""
    return {"nx": [4], "lx": [2.0], "chain_model": "discrete", "ds": 1e-6,
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 2.0}, "zeta_n": 100.0,
            "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 1.0009e-6}]},
                         {"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 2e-6}]}],
            "platform": platform, "verbose_level": 0,
            "langevin": {"nbar": 1e28, "dt": 0.01, "max_step": 1},
            "recording": {"dir": str(directory), "checkpoint_interval": 1,
                          "sample_interval": 1, "statistics_save_interval": 1},
            "gibbs_ensemble": {"boxes": [{"volume_fractions": [0.7, 0.3]}, {"volume_fractions": [0.3, 0.7]}],
                               "cycles": cycles, "cycle_steps": 1, "exchange_rate": 1e-8}}


def snapped_lengths(root, platform):
    """Gibbs amounts and pressure work use the same bead lengths as the native chains."""
    def make(directory, cycles):
        params = snapped_parameters(directory, cycles, platform)
        gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=31))
        assert params["polymers"][0]["blocks"][0]["length"] == 1.0009e-6
        assert gibbs.params["polymers"][0]["blocks"][0]["length"] == 1.0009e-6
        return gibbs

    full = make(root / "snapped_full", 2)
    native = quiet(lambda: CLFTS(full.box_params(0, full.amounts[0], 0), random_seed=31))
    native_alphas = np.array([native.prop_solver.get_polymer(p).get_alpha() for p in range(2)])
    np.testing.assert_array_equal(native_alphas, [1e-6, 2e-6])
    np.testing.assert_array_equal(full.alphas, native_alphas)
    quiet(full.run)
    for cycle in full.history["cycles"]:
        for b, box in enumerate(cycle["boxes"]):
            counts = np.asarray(box["volume_fractions"]) * box["volume"] / native_alphas
            np.testing.assert_allclose(cycle["amounts_before"][b], counts, rtol=2e-14, atol=0)
            expected_exchange = np.asarray(box["chemical_potentials"]) - native_alphas * box["pressure"]
            np.testing.assert_allclose(box["exchange_potentials"], expected_exchange, rtol=0, atol=2e-14)

    part = make(root / "snapped_part", 1)
    quiet(part.run)
    part = make(root / "snapped_part", 2)
    quiet(part.continue_run)
    assert part.history["cycles"] == full.history["cycles"], "snapped-length continuation differs"
    for box in ("box0", "box1"):
        a = loadmat(root / "snapped_full" / box / "fields_000002.mat", squeeze_me=True)
        b = loadmat(root / "snapped_part" / box / "fields_000002.mat", squeeze_me=True)
        keys = [key for key in a if key.startswith("w_")]
        assert keys and all(np.array_equal(a[key], b[key]) for key in keys), box
    # The input hash must still distinguish two supplied lengths that snap to the same bead count.
    changed = snapped_parameters(root / "snapped_part", 3, platform)
    changed["polymers"][0]["blocks"][0]["length"] = 1.0008e-6
    expect_validation_error(lambda: GibbsEnsemble(changed, random_seed=31).continue_run(), "input differs")

    # The physical meaning of amounts in a materially unsnapped old history cannot
    # be changed silently. New metadata must also agree with the current native model.
    path = root / "snapped_part" / "gibbs_history.json"
    stored = json.loads(path.read_text())
    assert stored["settings"]["effective_block_lengths"] == [[1e-6], [2e-6]]
    stored["settings"]["effective_block_lengths"] = [[1.0009e-6], [2e-6]]
    path.write_text(json.dumps(stored))
    expect_validation_error(lambda: make(root / "snapped_part", 3).continue_run(), "effective contour lengths differ")
    stored["settings"].pop("effective_block_lengths")
    path.write_text(json.dumps(stored))
    expect_validation_error(lambda: make(root / "snapped_part", 3).continue_run(), "effective contour lengths differ")


def snapped_charges(root, platform):
    """One A bead in an AB chain has one charge, independent of its accepted input offset."""
    params = snapped_parameters(root / "snapped_charged", 2, platform)
    params["polymers"][0]["blocks"].append({"type": "B", "length": 2e-6})
    params["smearing"] = {"type": "gaussian", "a_int": 0.1}
    params["electrostatics"] = {"charges": {"A": 1.0}, "coupling": 100.0, "ions": salt()}
    gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=31))
    expected_charges = np.array([1.0, 0.0, 1.0, -1.0])
    np.testing.assert_array_equal(gibbs.charges, expected_charges)
    assert gibbs.neutral
    for b, box in enumerate(params["gibbs_ensemble"]["boxes"]):
        box_params = gibbs.box_params(b, gibbs.amounts[b], 0)
        w, psi = gibbs._homogeneous_fields(box_params)
        fraction_a = box["volume_fractions"][0] / 3.0
        np.testing.assert_allclose(w, [2.0 * (1 - fraction_a), 2.0 * fraction_a], rtol=0, atol=2e-14)
        assert psi == 0.0
        np.testing.assert_allclose(box_params["electrostatics"]["ions"]["Cl"]["concentration"],
                                   0.01 + fraction_a, rtol=0, atol=2e-14)
    quiet(gibbs.run)
    for cycle in gibbs.history["cycles"]:
        for key in ("amounts_before", "amounts_after"):
            amounts = np.asarray(cycle[key])
            np.testing.assert_allclose(amounts @ expected_charges, 0.0, rtol=0, atol=2e-9)


def neutral_chain_exchange(root, platform):
    """Neutral polyampholytes exchange freely despite roundoff in their native lengths."""
    def parameters(directory, cycles, lengths=(0.2, 0.3, 0.5)):
        return {"nx": [4], "lx": [2.0], "chain_model": "discrete", "ds": 0.1,
                "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0},
                "chi_n": {"A,B": 1.0, "A,C": 2.0, "B,C": 2.0}, "zeta_n": 100.0,
                "polymers": [{"volume_fraction": 0.5,
                              "blocks": [{"type": monomer, "length": length}
                                         for monomer, length in zip(("A", "A", "B"), lengths)]},
                             {"volume_fraction": 0.5, "blocks": [{"type": "C", "length": 1.0}]}],
                "smearing": {"type": "gaussian", "a_int": 0.1},
                "electrostatics": {"charges": {"A": 1.0, "B": -1.0}, "coupling": 100.0},
                "platform": platform, "verbose_level": 0,
                "langevin": {"nbar": 1e28, "dt": 0.01, "max_step": 1},
                "recording": {"dir": str(directory), "checkpoint_interval": 1,
                              "sample_interval": 1, "statistics_save_interval": 1},
                "gibbs_ensemble": {"boxes": [{"volume_fractions": [0.7, 0.3]}, {"volume_fractions": [0.3, 0.7]}],
                                   "cycles": cycles, "cycle_steps": 1, "exchange_rate": 0.1}}

    def make(directory, cycles, lengths=(0.2, 0.3, 0.5)):
        return quiet(lambda: GibbsEnsemble(parameters(directory, cycles, lengths), random_seed=31))

    # Both have equal numbers of positive and negative beads. The residual depends
    # on Python's summation algorithm, so cover ordinary and compensated sums.
    for lengths in ((0.2, 0.3, 0.5), (0.2, 0.5, 0.7)):
        gibbs = make(root / "neutral_input", 1, lengths)
        np.testing.assert_array_equal(gibbs.charges, [0.0, 0.0])
        assert not gibbs.neutral, "Roundoff enabled a spurious constraint on neutral chains"

    # The zero-charge decision is relative to cancellation, not an absolute cutoff:
    # genuinely small charges of one sign must survive. Screening permits their net charge.
    small = parameters(root / "small_charge", 1)
    small["electrostatics"].update(charges={"A": 1e-16}, screening_length=1.0)
    gibbs = quiet(lambda: GibbsEnsemble(small, random_seed=31))
    np.testing.assert_allclose(gibbs.charges, [5e-16, 0.0], rtol=2e-14, atol=0)

    full = make(root / "neutral_full", 2)
    quiet(full.run)
    for cycle in full.history["cycles"]:
        before, after = (np.asarray(cycle[key]) for key in ("amounts_before", "amounts_after"))
        difference = np.subtract(*(box["exchange_potentials"] for box in cycle["boxes"]))
        # With neutral species the stated Gibbs mobility applies without the projection on neutral moves
        expected = -full.exchange_rate * ideal_mobility(before, full.alphas) @ difference
        assert np.max(np.abs(expected) / before.min(axis=0)) < 0.5, "Unexpected step limiter in this case"
        assert abs(expected[0]) > 1e-4, "This case must actually exchange polyampholytes"
        np.testing.assert_allclose(after[0] - before[0], expected, rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(after.sum(axis=0), before.sum(axis=0), rtol=0, atol=5e-16)

    part = make(root / "neutral_part", 1)
    quiet(part.run)
    part = make(root / "neutral_part", 2)
    quiet(part.continue_run)
    assert part.history["cycles"] == full.history["cycles"], "Neutral-chain continuation differs"
    for box in ("box0", "box1"):
        a = loadmat(root / "neutral_full" / box / "fields_000002.mat", squeeze_me=True)
        b = loadmat(root / "neutral_part" / box / "fields_000002.mat", squeeze_me=True)
        keys = [key for key in a if key.startswith(("w_", "psi_"))]
        keys += ["normal_noise_prev", "rng_state_json"]
        assert keys and all(np.array_equal(a[key], b[key]) for key in keys), box


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("inputs", inputs), ("mobility", mobility), ("blend binodal", blend), ("charged coexistence", charged),
                            ("Donnan coexistence", donnan), ("tie line", tie_line), ("restart", restart)):
            check(root)
            print(f"[PASS] {name}")
        for platform in PlatformSelector.avail_platforms():
            for name, check in (("snapped chain lengths", snapped_lengths), ("snapped chain charges", snapped_charges),
                                ("neutral chain exchange", neutral_chain_exchange)):
                check(root / platform, platform)
                print(f"[PASS] {platform}: {name}")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
