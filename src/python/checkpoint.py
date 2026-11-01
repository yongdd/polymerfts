"""What a Langevin run saves and restores: its measurements, checkpoints and the state of extensions.

RunRecord holds what a run has measured since its last statistics file. checkpoint_data() writes the
settings and measurements of a checkpoint, and validate_checkpoint() checks a checkpoint before a
continuation changes any state. The extension functions capture and restore the state of an integrator
and of callbacks.
"""

import copy
import hashlib
import itertools
import json
import warnings

import numpy as np
from scipy.io import loadmat

from .utils import fields_hash, input_hash, json_ready, mat_ready, settings_json
from .validation import normalize_params, ValidationError


def checkpoint_settings(sim):
    """Effective settings that define a continuation; output paths/max_step may change."""
    solver = sim.prop_solver
    result = {
        "simulation": type(sim).__name__ if type(sim).__name__ in ("LFTS", "CLFTS") else
                      ("CLFTS" if solver.field_type == "complex" else "LFTS"),
        "nx": solver.nx, "lx": solver.get_lx(), "angles": solver.get_angles_degrees(),
        "bc": solver.bc, "ds": sim.ds, "chain_model": sim.chain_model,
        "numerical_method": solver.numerical_method, "segment_lengths": sim.segment_lengths,
        "polymers": sim.polymers, "chi_n": sim.chi_n, "zeta_n": sim.zeta_n,
        "dt": sim.langevin["dt"], "nbar": sim.langevin["nbar"], "dt_scaling": sim.dt_scaling,
        "smearing": {"type": sim.smearing.type, "params": sim.smearing.params},
        # None without electrostatics, which earlier checkpoints also give
        "electrostatics": None if getattr(sim, "electrostatics", None) is None else sim.electrostatics.settings(),
        "sample_interval": sim.recording["sample_interval"],
        "statistics_save_interval": sim.recording["statistics_save_interval"],
        "extensions": extension_settings(sim),
    }
    if hasattr(sim, "saddle"):
        result["saddle"] = {**sim.saddle.settings, "failure_policy": sim.saddle.settings.get("failure_policy", "retry"),
                            "max_retries": sim.saddle.settings.get("max_retries", 5)}
        result["compressor"] = sim.params["compressor"]
        wtmd = sim.wtmd
        result["wtmd"] = None if wtmd is None else {
            key: getattr(wtmd, key) for key in
            ("l", "kc", "dT", "sigma_psi", "psi_min", "psi_max", "dpsi", "bins", "update_freq")}
    else:
        result["dynamic_stabilization"] = sim.alpha_ds
    return json_ready(result)


def checkpoint_data(sim):
    """Version, compatibility settings and complete RNG/measurement state."""
    settings = checkpoint_settings(sim)
    data = {"checkpoint_format_version": 2, "checkpoint_settings_json": settings_json(settings),
            "checkpoint_settings_hash": input_hash(settings),
            "rng_state_json": json.dumps(sim.random_bg.state),
            "simulation_metadata_json": json.dumps(json_ready(sim.metadata)),
            "run_status_json": json.dumps(json_ready(sim.run_status))}
    data["extension_state_json"] = json.dumps(extension_state(sim), allow_nan=False)
    record = getattr(sim, "_recording_state", None)
    if record is None:
        raise ValidationError("Checkpointing requires the recording state of an accepted run step")
    data.update(record.checkpoint_arrays())
    return data


def system_data(sim):
    """The system, at the head of the structure function files."""
    chi_n = {}
    for key in sim.chi_n:
        monomer_pair = sorted(key.split(","))
        chi_n[monomer_pair[0] + "," + monomer_pair[1]] = sim.chi_n[key]
    return {"dim": sim.prop_solver.dim, "nx": sim.prop_solver.nx, "lx": sim.prop_solver.get_lx(), "chi_n": chi_n,
            "chain_model": sim.chain_model, "ds": sim.ds, "dt": sim.langevin["dt"], "nbar": sim.langevin["nbar"],
            "initial_params": mat_ready(sim.params)}


def fields_data(sim, w, langevin_step, normal_noise_prev, w_aux=None):
    """The system and the checkpoint of a fields file, without the fields of its model."""
    data = {"initial_params": mat_ready(sim.params),
            "dim": sim.prop_solver.dim, "nx": sim.prop_solver.nx, "lx": sim.prop_solver.get_lx(),
            "monomer_types": sim.monomer_types, "chi_n": sim.chi_n, "chain_model": sim.chain_model, "ds": sim.ds,
            "dt": sim.langevin["dt"], "nbar": sim.langevin["nbar"],
            "eigenvalues": sim.mpt.eigenvalues,
            "aux_fields_real": sim.mpt.aux_fields_real_idx, "aux_fields_imag": sim.mpt.aux_fields_imag_idx,
            "matrix_a": sim.mpt.matrix_a, "matrix_a_inverse": sim.mpt.matrix_a_inv,
            "langevin_step": langevin_step,
            "random_generator": sim.random_bg.state["bit_generator"],
            "random_state_state": str(sim.random_bg.state["state"]["state"]),
            "random_state_inc": str(sim.random_bg.state["state"]["inc"]),
            "normal_noise_prev": normal_noise_prev}
    data.update(checkpoint_data(sim))
    data["w_aux"] = sim.mpt.to_aux_fields(w) if w_aux is None else w_aux
    return data


def load_checkpoint(sim, file_name, allow_legacy=False):
    """Reads a checkpoint, checks it and restores the extensions, the metadata and the random numbers."""
    data = loadmat(file_name, squeeze_me=True)
    validate_checkpoint(sim, data, allow_legacy=allow_legacy)
    restore_extensions(sim, data)
    if "simulation_metadata_json" in data:
        previous_metadata = json.loads(str(data["simulation_metadata_json"]))
        sim.metadata["random_seed"] = previous_metadata.get("random_seed")
        sim.metadata["initial_fields_hash"] = previous_metadata.get("initial_fields_hash", "unknown")
        sim.metadata["parent_checkpoint"] = str(file_name)

    # Check if data["langevin_step"] is a multiple of statistics_save_interval
    interval = sim.recording["statistics_save_interval"]
    if "record_sample_count" not in data and data["langevin_step"] % interval != 0:
        print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'statistics_save_interval'.")
        print(f"The structure function will be correctly recorded after {(data['langevin_step'] // interval + 1) * interval}th langevin_step.")

    # Versioned checkpoints carry the complete PCG64 state; legacy scalar
    # fields are only needed for explicitly accepted unversioned files.
    if "rng_state_json" in data:
        sim.random_bg.state = json.loads(str(data["rng_state_json"]))
    else:
        sim.random_bg.state = {
            "bit_generator": "PCG64",
            "state": {"state": int(data["random_state_state"]), "inc": int(data["random_state_inc"])},
            "has_uint32": 0, "uinteger": 0}
    print("Restored Random Number Generator:", sim.random_bg.state)
    return data


def start_run_status(sim, start_langevin_step, initial_fields, restart_data, **counters):
    """The run status and metadata of a run that is about to start; `counters` carry over a restart."""
    first_step = 1 if start_langevin_step is None else start_langevin_step
    if isinstance(first_step, (bool, np.bool_)) or not isinstance(first_step, (int, np.integer)) or first_step < 1:
        raise ValidationError("start_langevin_step must be a positive integer")
    sim.run_status = {"status": "running", "last_completed_step": first_step - 1,
                      "requested_last_step": sim.langevin["max_step"], **counters}
    sim.metadata["initial_fields_hash" if restart_data is None else "restart_fields_hash"] = fields_hash(initial_fields)
    if restart_data is not None and "run_status_json" in restart_data:
        previous_status = json.loads(str(restart_data["run_status_json"]))
        for key in counters:
            sim.run_status[key] = previous_status.get(key, 0)
    sim.metadata["initial_rng_state"] = sim.random_bg.state


def _measurement_key(kind, pair):
    # Monomer names may contain underscores; joining names with '_' is ambiguous.
    return "record_" + kind + "_" + hashlib.sha256(pair.encode()).hexdigest()[:32]


def validate_checkpoint(sim, data, *, allow_legacy=False):
    version = data.get("checkpoint_format_version", 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        raise ValidationError("Checkpoint format version must be an integer")
    if version == 0:
        if not allow_legacy:
            raise ValidationError("Unversioned checkpoint: use allow_legacy=True to accept an unverified continuation")
        if sim.integrator.name != "lm" or sim.callbacks or getattr(sim.integrator, "settings", {}):
            raise ValidationError("Legacy checkpoints cannot restore custom integrators or callbacks")
        warnings.warn("Legacy checkpoint: settings are unverified and pending measurements cannot be recovered", RuntimeWarning)
        return
    if version not in (1, 2):
        raise ValidationError(f"Unsupported checkpoint format version {version}")
    try:
        stored = json.loads(str(data["checkpoint_settings_json"]))
        if input_hash(stored) != str(data["checkpoint_settings_hash"]):
            raise ValidationError("Checkpoint settings hash does not match its contents")
        # Verify the original settings hash before translating names. Earlier
        # checkpoints have legacy block keys and flat recording-period keys.
        stored = normalize_params(stored)
        for old, new in (("sf_computing_period", "sample_interval"),
                         ("sf_recording_period", "statistics_save_interval")):
            if old in stored:
                if new in stored:
                    raise ValidationError(f"Checkpoint specifies both '{old}' and '{new}'")
                stored[new] = stored.pop(old)
        current = checkpoint_settings(sim)
        # Version 1 predates extensions and always used LM without callbacks.
        if version == 1:
            stored.setdefault("extensions", {"integrator": {"name": "lm", "settings": {}}, "callbacks": {}})
        # JSON distinguishes 15 from 15.0; effective numerical settings do not.
        different = [key for key in current if current[key] != stored.get(key)]
        if different:
            raise ValidationError("Incompatible checkpoint settings: " + ", ".join(different))
        if version == 2:
            state = json.loads(str(data["extension_state_json"]))
            if not isinstance(state, dict) or set(state) != {"integrator", "callbacks"} or not isinstance(state["callbacks"], dict):
                raise ValidationError("Invalid extension checkpoint state")
            if set(state["callbacks"]) != set(sim.callbacks):
                raise ValidationError("Checkpoint callback state does not match registered callbacks")
            if not isinstance(state["integrator"], dict) or not all(isinstance(value, dict) for value in state["callbacks"].values()):
                raise ValidationError("Each extension checkpoint state must be a dictionary")
            json.dumps(state, allow_nan=False)
        stored_basis = np.asarray(data["matrix_a"])
        if stored_basis.size != sim.mpt.matrix_a.size or not np.allclose(
                stored_basis.reshape(sim.mpt.matrix_a.shape), sim.mpt.matrix_a, rtol=1e-12, atol=1e-12):
            raise ValidationError("Incompatible checkpoint auxiliary-field basis")
        step = data["langevin_step"]
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValidationError("Checkpoint step must be a nonnegative integer")
        if step > sim.langevin["max_step"]:
            raise ValidationError("Checkpoint step exceeds langevin.max_step or is negative")
        for key in ("simulation_metadata_json", "run_status_json"):
            if key in data:
                value = json.loads(str(data[key]))
                if not isinstance(value, dict):
                    raise ValidationError(f"Checkpoint '{key}' must describe a dictionary")
                if key == "run_status_json":
                    if value.get("last_completed_step") != step:
                        raise ValidationError("Checkpoint run status does not match its step")
                    for counter in ("saddle_failures", "noise_resamples"):
                        count = value.get(counter, 0)
                        if type(count) is not int or count < 0:
                            raise ValidationError(f"Invalid checkpoint run counter '{counter}'")
        n_fields = len(sim.monomer_types)
        for key in ("w_aux", "normal_noise_prev"):
            if key not in data or not np.all(np.isfinite(data[key])):
                raise ValidationError(f"Checkpoint has missing or non-finite '{key}'")
            if np.iscomplexobj(data[key]) and (key == "normal_noise_prev" or sim.prop_solver.field_type == "real"):
                raise ValidationError(f"Checkpoint '{key}' must be real")
        if np.size(data["w_aux"]) != n_fields * sim.prop_solver.n_grid:
            raise ValidationError("Checkpoint auxiliary field shape is incompatible")
        if "record_sample_count" not in data or "rng_state_json" not in data:
            raise ValidationError("Checkpoint is missing measurement or RNG state")
        n_noise = n_fields if sim.prop_solver.field_type == "complex" else len(sim.mpt.aux_fields_real_idx)
        if np.size(data["normal_noise_prev"]) != n_noise * sim.prop_solver.n_grid:
            raise ValidationError("Checkpoint noise shape is incompatible")
        rng = np.random.PCG64(0)
        rng.state = json.loads(str(data["rng_state_json"]))
        count = data["record_sample_count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValidationError("Invalid checkpoint sample count")
        for key in ["record_H_history", "record_drift_history"] + [_measurement_key("dH", key) for key in sim.chi_n]:
            if np.size(data[key]) != count or not np.all(np.isfinite(data[key])):
                raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        if np.iscomplexobj(data["record_drift_history"]) or np.any(data["record_drift_history"] < 0):
            raise ValidationError("Checkpoint drift magnitudes must be real and nonnegative")
        # The drifts of every step since the last statistics file, or since the first step of a run
        # continued by hand or from a checkpoint without the record (record_drift_step_start).
        # Checkpoints of earlier versions lack the record, whose continued runs record from the restart on
        pending_steps = step % sim.recording["statistics_save_interval"]
        if "record_drift_step_start" in data:
            first = data["record_drift_step_start"]
            if (isinstance(first, (bool, np.bool_)) or not isinstance(first, (int, np.integer))
                    or not step - pending_steps < first <= step + 1):
                raise ValidationError("Invalid checkpoint measurement 'record_drift_step_start'")
            pending_steps = step + 1 - int(first)
        for key, per_step in (("record_drift_step_history", n_fields), ("record_psi_drift_step_history", 1)):
            if key in data:
                values = np.atleast_1d(np.asarray(data[key]))
                if (np.size(values) != pending_steps * per_step or np.iscomplexobj(values)
                        or not np.all(np.isfinite(values)) or np.any(values < 0)):
                    raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        # Partition ratios of the same steps, in [0, 1]; NaN for the steps of a checkpoint without them
        if "record_partition_ratio_step_history" in data:
            values = np.atleast_1d(np.asarray(data["record_partition_ratio_step_history"]))
            n_ratios = len(sim.thermodynamics.partition_ratio_species) if hasattr(sim, "thermodynamics") else 0
            if (np.size(values) != pending_steps * n_ratios or np.iscomplexobj(values)
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValidationError("Invalid checkpoint measurement 'record_partition_ratio_step_history'")
        # Pending chemical potentials and pressures (absent in checkpoints of earlier versions or
        # of runs without them; continue_run then records NaN for those samples, which later
        # checkpoints keep). A species without chains has the chemical potential -inf
        n_columns = sim.thermodynamics.n_chemical_potentials if hasattr(sim, "thermodynamics") else 0
        for key, per_sample in (("record_chemical_potential_history", n_columns), ("record_pressure_history", 1),
                                ("record_number_density_history", n_columns)):
            if key in data:
                values = np.atleast_1d(np.asarray(data[key]))
                if np.size(values) != count * per_sample or np.any(np.isposinf(np.real(values))):
                    raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        # Pending sums of recording.concentrations and their number of steps, absent in checkpoints
        # of runs without them
        for name in getattr(sim, "concentration_names", []):
            key = _measurement_key("sf", "concentration_" + name)
            if key in data and (np.size(data[key]) != sim.prop_solver.n_grid or not np.all(np.isfinite(data[key]))):
                raise ValidationError(f"Invalid checkpoint concentration sum '{name}'")
        key = _measurement_key("sf", "concentration steps")
        if key in data and (np.size(data[key]) != 1 or not np.all(np.isfinite(data[key])) or np.any(np.real(data[key]) < 0)):
            raise ValidationError("Invalid checkpoint measurement 'concentration steps'")
        for monomer in sim.monomer_types:
            names = [kind + monomer + suffix for kind in ("w_", "phi_") for suffix in
                     (("",) if sim.prop_solver.field_type == "real" else ("_real", "_imag"))]
            for name in names:
                if np.size(data[name]) != sim.prop_solver.n_grid or not np.all(np.isfinite(data[name])) or np.iscomplexobj(data[name]):
                    raise ValidationError(f"Invalid checkpoint field '{name}'")
        sf_size = int(np.prod(sim.prop_solver.nx)) if sim.prop_solver.field_type == "complex" else int(
            np.prod(sim.prop_solver.nx[:-1])) * (sim.prop_solver.nx[-1] // 2 + 1)
        for pair in itertools.combinations_with_replacement(sim.monomer_types, 2):
            name = _measurement_key("sf", ",".join(pair))
            if np.size(data[name]) != sf_size or not np.all(np.isfinite(data[name])):
                raise ValidationError(f"Invalid checkpoint structure function '{name}'")
        electrostatics = getattr(sim, "electrostatics", None)
        if electrostatics is not None:
            # ψ, ψ(k), the previous noise and the stream of ψ (Electrostatics.checkpoint_data), the
            # pending drifts of ψ and, with Coulomb coupling, the pending charge structure sums
            missing = [key for key in electrostatics.CHECKPOINT_KEYS + ("record_psi_drift_history",) if key not in data]
            if missing:
                raise ValidationError(f"The checkpoint lacks electrostatic data: {', '.join(missing)}")
            for key in ("psi_real", "psi_imag", "psi_k_real", "psi_k_imag", "psi_noise_prev"):
                if (np.size(data[key]) != sim.prop_solver.n_grid or not np.all(np.isfinite(data[key]))
                        or np.iscomplexobj(data[key])):
                    raise ValidationError(f"Invalid checkpoint electrostatic array '{key}'")
            try:
                np.random.PCG64(0).state = json.loads(str(data["psi_rng_state_json"]))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValidationError("Invalid checkpoint random-number state of psi") from exc
            psi_drift = data["record_psi_drift_history"]
            if (np.size(psi_drift) != count or np.iscomplexobj(psi_drift) or not np.all(np.isfinite(psi_drift))
                    or np.any(psi_drift < 0)):
                raise ValidationError("Invalid checkpoint measurement 'record_psi_drift_history'")
            # The settings keep the point ions by name; the columns of the pending chemical
            # potentials and numbers follow their order (ion_names_json, absent in checkpoints
            # of earlier versions)
            if "ion_names_json" in data:
                stored_ions = json.loads(str(data["ion_names_json"]))
                if stored_ions != [ion["name"] for ion in electrostatics.ions]:
                    raise ValidationError(f"'electrostatics.ions' lists the point ions in another order than "
                                          f"the checkpoint ({', '.join(stored_ions)}); the recorded columns "
                                          "follow that order")
            # The charge structure function (with Coulomb coupling in a periodic box)
            if electrostatics.coupling > 0 and electrostatics.periodic:
                for key in electrostatics.CHARGE_SUMS:
                    name = _measurement_key("sf", key)
                    if (name not in data or np.size(data[name]) != int(np.prod(sim.prop_solver.nx))
                            or not np.all(np.isfinite(data[name]))):
                        raise ValidationError(f"Invalid checkpoint charge structure sum '{key}'")
        wtmd = getattr(sim, "wtmd", None)
        if wtmd is not None:
            def check_wtmd_array(key):
                if np.size(data[key]) != wtmd.bins or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key]):
                    raise ValidationError(f"Invalid checkpoint WTMD array '{key}'")
            def check_wtmd_copies(name):
                # Both copies are written from one array; the restore uses the unscaled
                # one, so a checkpoint whose scaled copy was edited is refused
                check_wtmd_array("wtmd_unscaled_" + name)
                if not np.allclose(np.ravel(data["wtmd_unscaled_" + name]) * wtmd.CV,
                                   np.ravel(data["wtmd_" + name]), rtol=1e-12, atol=0):
                    raise ValidationError(f"Checkpoint WTMD arrays 'wtmd_{name}' and 'wtmd_unscaled_{name}' disagree")
            # Newer checkpoints also hold the bias arrays without the factor CV ("wtmd_unscaled_*")
            unscaled = "wtmd_unscaled_u" in data
            for key in ("wtmd_u", "wtmd_up", "wtmd_I0"):
                check_wtmd_array(key)
            if unscaled:
                check_wtmd_copies("u")
                check_wtmd_copies("up")
            pending = np.size(data["wtmd_order_parameter_history"])
            if (pending >= wtmd.update_freq or not np.all(np.isfinite(data["wtmd_order_parameter_history"])) or
                    np.iscomplexobj(data["wtmd_order_parameter_history"])):
                raise ValidationError("Invalid checkpoint pending WTMD samples")
            for pair in sim.chi_n:
                suffix = "_".join(sorted(pair.split(",")))
                key = "wtmd_I1_" + suffix
                if np.any(data["wtmd_I0"]) or key in data:
                    check_wtmd_array(key)
                    if unscaled:
                        check_wtmd_copies("I1_" + suffix)
                key = "wtmd_dH_history_" + suffix
                if pending and (np.size(data[key]) != pending or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key])):
                    raise ValidationError(f"Invalid checkpoint WTMD history '{key}'")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid checkpoint: {exc}") from exc


class RunRecord:
    """What a Langevin run has measured since its last statistics file.

    `series` says what is recorded, {name: columns}. add(name, value) appends one number (columns 0)
    or one row of that many columns, per sample or per step as the run chooses. `sums` are the arrays
    that averages accumulate in place, {name: array}. To record something new, give it an entry in
    `series` and call add(): the statistics files, checkpoints and restarts follow from the table.

    The name "dH A,B" stands for dH/dχN of that pair. In CL-FTS the values are complex
    (complex_values), and complex series go into the statistics files as name_real and name_imag.
    """

    # Keys of the statistics files that differ from the name of the series
    FILE_KEYS = {"drift_history": "drift_max_history", "psi_drift_history": "psi_drift_max_history",
                 "drift_step_history": "drift_max_step_history", "psi_drift_step_history": "psi_drift_max_step_history"}
    # Series that stay real in CL-FTS
    REAL_SERIES = ("drift_history", "psi_drift_history", "drift_step_history", "psi_drift_step_history",
                   "partition_ratio_step_history")
    # Series that are complex even when a list is empty
    COMPLEX_SERIES = ("chemical_potential_history", "number_density_history", "pressure_history")

    def __init__(self, series, sums, complex_values=False):
        self.columns = series
        self.sums = sums
        self.complex_values = complex_values
        self.clear(1)

    def add(self, name, value):
        self.lists[name].append(value)

    def clear(self, next_step):
        """Starts the next statistics file, with the measurements of next_step."""
        self.lists = {name: [] for name in self.columns}
        for array in self.sums.values():
            array[:] = 0.0
        self.sample_count = 0
        self.first_step = next_step

    def array(self, name):
        """A series as an array: one row per entry."""
        values = np.asarray(self.lists[name], dtype=np.complex128 if name in self.COMPLEX_SERIES else None)
        return values if self.columns[name] == 0 else np.reshape(values, (-1, self.columns[name]))

    def statistics(self):
        """The arrays of a statistics file."""
        data = {"sample_count": self.sample_count}
        for name in self.columns:
            key = self.FILE_KEYS.get(name, name)
            if name.startswith("dH "):
                key = "dH_history_" + "_".join(sorted(name[3:].split(",")))
            values = self.array(name)
            if self.complex_values and name not in self.REAL_SERIES:
                data[key + "_real"], data[key + "_imag"] = np.real(values), np.imag(values)
            else:
                data[key] = values
        return data

    def _checkpoint_key(self, name):
        return _measurement_key("dH", name[3:]) if name.startswith("dH ") else "record_" + name

    def checkpoint_arrays(self):
        """The pending measurements, as a checkpoint stores them."""
        data = {self._checkpoint_key(name): self.array(name) for name in self.columns}
        data["record_sample_count"] = self.sample_count
        if "drift_step_history" in self.columns:
            data["record_drift_step_start"] = int(self.first_step)
        for name, array in self.sums.items():
            data[_measurement_key("sf", name)] = array
        return data

    def restore(self, data, first_step):
        """Puts back the pending measurements of a checkpoint that validate_checkpoint() accepted, in
        their own shapes (MATLAB squeezes them). Without any, the record starts empty at first_step."""
        self.first_step = first_step
        if data is None or "record_sample_count" not in data:
            return
        self.sample_count = int(data["record_sample_count"])
        for name, columns in self.columns.items():
            key, width = self._checkpoint_key(name), max(columns, 1)
            if key in data:
                values = np.asarray(data[key], dtype=np.complex128 if name in self.COMPLEX_SERIES else None)
            elif name in ("drift_step_history", "psi_drift_step_history"):
                continue    # these records start at the restart (first_step)
            else:
                # A series the checkpoint lacks keeps its rows in line with the others: NaN for the
                # pending samples, or for the steps of the drift record
                rows = len(self.lists.get("drift_step_history", [])) if name.endswith("_step_history") else self.sample_count
                values = np.full(rows * width, np.nan + 0j if name in self.COMPLEX_SERIES else np.nan)
            self.lists[name] = np.atleast_1d(values).tolist() if columns == 0 else list(np.reshape(values, (-1, columns)))
        for name, array in self.sums.items():
            key = _measurement_key("sf", name)
            if key in data:     # sums of an option that the checkpoint's run did not use start from zero
                array[:] = np.asarray(data[key]).reshape(array.shape)
        # A restored record ends at the step before this run's first step
        self.first_step = first_step - len(self.lists.get("drift_step_history", []))


def initialize_extensions(sim, integrator, callbacks, *, capture_initial_state=True):
    from .integrators import make_integrator
    sim.integrator = make_integrator(integrator)
    if callbacks is not None and not isinstance(callbacks, dict):
        raise ValidationError("callbacks must be a dictionary of stable names to callables")
    sim.callbacks = dict(callbacks or {})
    for name, callback in sim.callbacks.items():
        if not isinstance(name, str) or not name or not callable(callback):
            raise ValidationError("Each callback needs a nonempty string name and must be callable")
        interval = getattr(callback, "interval", 1)
        if type(interval) is not int or interval < 1:
            raise ValidationError("Callback interval must be a positive integer")
    for obj in [sim.integrator, *sim.callbacks.values()]:
        for method in ("state_dict", "load_state_dict"):
            if hasattr(obj, method) and not callable(getattr(obj, method)):
                raise ValidationError(f"Extension {method} must be callable")
        save = callable(getattr(obj, "state_dict", None))
        load = callable(getattr(obj, "load_state_dict", None))
        if save != load:
            raise ValidationError("Extensions must implement both state_dict and load_state_dict, or neither")
    # CLFTS captures after preparing any linear response. Other simulations
    # finish setup here and validate state before the first scheduled checkpoint.
    if capture_initial_state:
        sim._initial_extension_state = copy.deepcopy(extension_state(sim))
    sim.metadata["extensions"] = extension_settings(sim)


def extension_settings(sim):
    def settings(obj):
        value = getattr(obj, "settings", {})
        if not isinstance(value, dict):
            raise ValidationError("Extension settings must be a JSON dictionary")
        return value
    result = {"integrator": {"name": sim.integrator.name, "settings": settings(sim.integrator)},
              "callbacks": {name: {"callable": f"{getattr(cb, '__module__', type(cb).__module__)}."
                                              f"{getattr(cb, '__qualname__', type(cb).__qualname__)}",
                                   "interval": getattr(cb, "interval", 1),
                                   "settings": settings(cb)} for name, cb in sim.callbacks.items()}}
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Extension settings must contain finite JSON values") from exc


def extension_state(sim):
    return {"integrator": extension_object_state(sim.integrator),
            "callbacks": {name: extension_object_state(cb) for name, cb in sim.callbacks.items()}}


def extension_object_state(obj):
    value = obj.state_dict() if callable(getattr(obj, "state_dict", None)) else {}
    if not isinstance(value, dict):
        raise ValidationError("Extension state_dict must return a JSON dictionary")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Extension state must contain finite JSON values") from exc


def restore_extensions(sim, data=None):
    """Load before the RNG is changed; roll back extension objects on a bad state."""
    state = (json.loads(str(data["extension_state_json"]))
             if data is not None and "extension_state_json" in data else sim._initial_extension_state)
    previous = extension_state(sim)

    def load(values):
        objects = [(sim.integrator, values["integrator"])]
        objects.extend((obj, values["callbacks"][name]) for name, obj in sim.callbacks.items())
        for obj, value in objects:
            if callable(getattr(obj, "load_state_dict", None)):
                obj.load_state_dict(copy.deepcopy(value))
            elif value != {}:
                raise ValidationError("A stateless extension cannot load nonempty state")
    try:
        load(state)
    except Exception as exc:
        load(previous)
        raise ValidationError(f"Cannot restore extension state: {exc}") from exc
