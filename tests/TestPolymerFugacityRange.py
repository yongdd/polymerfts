#!/usr/bin/env python3
"""Grand-canonical polymer densities need not have a representable fugacity.

The diagnostic partition and the propagators must still be representable. Exercise
exp(mu) overflow with subnormal Q, exp(mu) underflow with large Q, true Q = 0,
mixtures, nonzero stress, recorded operators and exact LM/Euler continuations.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS, PlatformSelector, ValidationError
from polymerfts.utils.checkpoint import validate_checkpoint
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def parameters(directory, platform, mu, *, model="discrete", reduced=False, length=1.0,
               method="lm", steps=4, mixed=False):
    polymers = [
        {"volume_fraction": 0.25, "blocks": [{"type": "A", "length": 1.0}]},
        {"chemical_potential": mu, "blocks": [{"type": "B", "length": length}]},
    ]
    if mixed:
        # The same propagators with a distinct chemical potential: each polymer
        # needs its own weight, including when exp(mu) cannot be materialized.
        polymers.append({"chemical_potential": mu - 1.3,
                         "blocks": [{"type": "B", "length": length}]})
    return {
        "nx": [4], "lx": [2.0], "ds": 1.0 if model == "discrete" else 0.25,
        "chain_model": model, "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 2.0}, "zeta_n": 10.0, "polymers": polymers,
        "platform": platform, "reduce_memory": reduced, "verbose_level": 0,
        "langevin": {"nbar": 1e4, "dt": 1e-7, "max_step": steps, "integrator": method},
        "recording": {"dir": str(directory), "checkpoint_interval": 1,
                      "sample_interval": 1, "statistics_save_interval": 3,
                      "pressure": True, "concentrations": True},
    }


def state(sim, fields):
    aux = sim.mpt.to_aux_fields(fields)
    phi, _ = quiet(sim.compute_concentrations, aux)
    partitions = [sim.prop_solver.get_partition_function(p)
                  for p in range(sim.prop_solver.get_n_polymer_types())]
    return aux, phi, partitions


def uniform_operators(root, platform, model, reduced, mu, offset):
    params = parameters(root, platform, mu, model=model, reduced=reduced, mixed=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sim = quiet(CLFTS, params, random_seed=7)
        potential = offset + 0.2j
        aux, phi, partitions = state(sim, {"A": np.zeros(4), "B": np.full(4, potential)})
        expected_numbers = np.array([0.25, np.exp(mu - potential), np.exp(mu - 1.3 - potential)])
        np.testing.assert_allclose(phi["A"], 0.25, rtol=0, atol=3e-13)
        np.testing.assert_allclose(phi["B"], sum(expected_numbers[1:]), rtol=3e-11, atol=0)
        np.testing.assert_allclose(sim.thermodynamics.number_densities(partitions), expected_numbers,
                                   rtol=3e-11, atol=0)
        np.testing.assert_allclose(sim.thermodynamics.grand_canonical_hamiltonian(partitions),
                                   -sum(expected_numbers[1:]), rtol=3e-11, atol=0)
        np.testing.assert_allclose(sim.thermodynamics.chemical_potentials(partitions),
                                   [np.log(0.25), mu, mu - 1.3], rtol=0, atol=3e-12)
        # Keep Q itself, with its true small/large value; no field or checkpoint gauge shift.
        for q in partitions[1:]:
            expected_q = np.exp(-potential)
            assert np.isfinite(q) and q != 0.0
            scale = max(abs(expected_q.real), abs(expected_q.imag))
            ratio = complex(q.real / scale, q.imag / scale) / complex(
                expected_q.real / scale, expected_q.imag / scale)
            np.testing.assert_allclose(ratio, 1.0, rtol=3e-11, atol=0)
        assert quiet(sim.prop_solver.check_total_partition)
        assert np.isfinite(sim._hamiltonian(aux, phi, partitions))
        for operator in ("pressure", "pressure_virial"):
            assert np.isfinite(getattr(sim.thermodynamics, operator)(aux, phi))
        assert np.all(np.isfinite(sim.evaluate_force(aux)))


def nonuniform_stress(root, platform, reduced, mu, offset):
    """A uniform shift of a two-bead potential scales Q and its cell derivative equally."""
    delta = np.array([0.7 + 0.15j, -0.9 - 0.2j, 0.4 + 0.1j, -0.2 - 0.05j])

    def make(chemical_potential, shift, lx=2.0):
        params = parameters(root, platform, chemical_potential, reduced=reduced, length=2.0)
        params["lx"] = [lx]
        sim = quiet(CLFTS, params, random_seed=11)
        fields = {"A": np.zeros(4), "B": shift / 2 + delta}
        aux, phi, partitions = state(sim, fields)
        return sim, aux, phi, partitions

    base, base_aux, base_phi, base_q = make(0.0, 0.0)
    sim, aux, phi, partitions = make(mu, offset)
    weight = np.exp(mu - offset)
    np.testing.assert_allclose(phi["B"], weight * base_phi["B"], rtol=3e-11, atol=0)
    np.testing.assert_allclose(sim.thermodynamics.number_densities(partitions)[1], weight * base_q[1],
                               rtol=3e-11, atol=0)
    for s in (base, sim):
        s.prop_solver.compute_stress()
    reference = np.asarray(base.prop_solver.get_stress_gce(base.fugacities))
    actual = np.asarray(sim.prop_solver.get_stress_gce_log(sim.log_fugacities))
    assert abs(reference[0]) > 1e-6, "The stress comparison needs a nonuniform chain"
    np.testing.assert_allclose(actual, weight * reference, rtol=3e-10, atol=1e-13 * weight)
    # Independent central difference of -Q in the ordinary reference system.
    epsilon = 1e-5
    plus = make(0.0, 0.0, 2.0 + epsilon)[3][1]
    minus = make(0.0, 0.0, 2.0 - epsilon)[3][1]
    np.testing.assert_allclose(reference[0], -(plus - minus) / (2 * epsilon), rtol=2e-7, atol=1e-9)
    assert quiet(sim.prop_solver.check_total_partition)
    for operator in ("pressure", "pressure_virial"):
        assert np.isfinite(getattr(sim.thermodynamics, operator)(aux, phi))


def zero_and_overflow(root, platform):
    """A true partition zero stays regular in GCE; a truly overflowing density stays nonfinite."""
    params = parameters(root / "zero", platform, 710.0)
    sim = quiet(CLFTS, params, random_seed=1)
    phase = np.array([0.0, 0.0, np.pi, -np.pi])
    aux, phi, partitions = state(sim, {"A": np.zeros(4), "B": 710.0 + 1j * phase})
    assert partitions[1] == 0.0
    np.testing.assert_allclose(phi["B"], np.exp(-1j * phase), rtol=3e-11, atol=3e-13)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert sim.thermodynamics.number_densities(partitions)[1] == 0.0
        assert sim.thermodynamics.grand_canonical_hamiltonian(partitions) == 0.0
        assert np.isfinite(sim.thermodynamics.pressure_virial(aux, phi))
    overflow = quiet(CLFTS, parameters(root / "overflow", platform, 710.0, steps=1), random_seed=1)
    overflow.draw_architectures = False
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            quiet(overflow.run, {"A": np.zeros(4), "B": np.zeros(4)})
        except ValidationError:
            pass
        else:
            raise AssertionError("A physically overflowing density must not complete a CLFTS run")


def compare_files(left, right, filename):
    a, b = (loadmat(path / filename, squeeze_me=True) for path in (left, right))
    prefixes = ("w_", "phi_", "normal_noise", "rng_state", "record_", "extension_state",
                "H_", "dH_", "drift_", "chemical_potential_", "number_density_", "pressure_",
                "structure_function_", "concentration_")
    keys = [key for key in a if key.startswith(prefixes)]
    assert keys, filename
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{filename}: {key}")


def restart(root, platform, method, mu, offset):
    paths = {name: root / name for name in ("full", "part", "continued")}
    sims = {}
    for name, steps, seed in (("full", 4, 17), ("part", 2, 17), ("continued", 4, 29)):
        sims[name] = quiet(CLFTS, parameters(paths[name], platform, mu, method=method, steps=steps),
                           random_seed=seed)
        sims[name].draw_architectures = False
    fields = {"A": np.array([0.1, -0.1, 0.2, -0.2]),
              "B": offset + np.array([0.2, -0.1, 0.1, -0.2])}
    for name in ("full", "part"):
        assert quiet(sims[name].run, fields)["status"] == "completed"
    checkpoint = paths["part"] / "fields_000002.mat"
    pending = loadmat(checkpoint, squeeze_me=True)
    validate_checkpoint(sims["continued"], pending)
    assert pending["record_sample_count"] == 2
    assert np.any(pending["normal_noise_prev"] != 0.0)
    assert quiet(sims["continued"].continue_run, checkpoint)["status"] == "completed"
    assert sims["full"].random_bg.state == sims["continued"].random_bg.state
    for filename in ("fields_000003.mat", "fields_000004.mat", "dH_000003.mat",
                     "structure_function_000003.mat"):
        compare_files(paths["full"], paths["continued"], filename)
    for step in (3, 4):
        data = loadmat(paths["continued"] / f"fields_{step:06d}.mat", squeeze_me=True)
        validate_checkpoint(sims["continued"], data)
    data = loadmat(paths["continued"] / "dH_000003.mat", squeeze_me=True)
    for key in ("H_history_real", "pressure_history_real", "number_density_history_real"):
        assert np.all(np.isfinite(data[key])), key
    assert np.all(np.asarray(data["number_density_history_real"])[:, 1] > 0), "Keep the tiny GCE population"
    # The saved fields retain the supplied large offset, and the actual small/large Q is unshifted.
    final = loadmat(paths["continued"] / "fields_000004.mat", squeeze_me=True)
    np.testing.assert_allclose(np.mean(final["w_B_real"]), offset, rtol=0, atol=2e-3)
    assert quiet(sims["continued"].prop_solver.check_total_partition)


def main():
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    with tempfile.TemporaryDirectory(prefix="polymerfts_polymer_fugacity_") as directory:
        root = Path(directory)
        for platform in platforms:
            for model in ("discrete", "continuous"):
                for reduced in (False, True):
                    for mu, offset in ((700.0, 700.0), (710.0, 710.0), (np.longdouble(710.0), 710.0), (720.0, 720.0),
                                       (750.0, 710.0), (-745.0, -700.0), (-750.0, -700.0)):
                        uniform_operators(root / platform / "operators", platform, model, reduced, mu, offset)
            for reduced in (False, True):
                for mu, offset in ((710.0, 710.0), (-750.0, -700.0)):
                    nonuniform_stress(root / platform / "stress", platform, reduced, mu, offset)
            zero_and_overflow(root / platform, platform)
            for method in ("lm", "euler"):
                for mu, offset in ((710.0, 710.0), (-750.0, -700.0)):
                    restart(root / platform / f"{method}_{mu}", platform, method, mu, offset)
            print(f"[PASS] {platform}: polymer fugacity range, stress, thermodynamics and exact restarts")
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
