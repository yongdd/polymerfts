"""MAT metadata and monomer-field readers, independent of the compiled backend.

Bundled analysis/conversion tools can load this module directly from the source
checkout without importing polymerfts or its native extension.
"""

import hashlib
import json

import numpy as np


_MAT_FIELD_KEYS = "polymerfts_mat_keys_json"


def _encode_mat_fields(value, max_length, *, top_level=False):
    """Escape only nested struct keys; MAT variable names keep their public API."""
    if isinstance(value, dict):
        names = [str(key) for key in value]
        if len(set(names)) != len(names):
            raise ValueError("MAT dictionary keys are not unique after conversion to strings")
        occupied = set(names)
        result, mapping = {}, {}
        for name, item in zip(names, value.values()):
            encoded = name
            if not top_level and (not name or name[0] == "_" or name[0].isdigit()
                                  or len(name) > max_length or not name.isascii()
                                  or name == _MAT_FIELD_KEYS):
                attempt = 0
                while True:
                    source = name if attempt == 0 else json.dumps([name, attempt])
                    encoded = "field_" + hashlib.sha256(source.encode()).hexdigest()[:24]
                    if encoded not in occupied:
                        break
                    attempt += 1
                occupied.add(encoded)
                mapping[encoded] = name
            result[encoded] = _encode_mat_fields(item, max_length)
        if mapping:
            result[_MAT_FIELD_KEYS] = json.dumps({"version": 1, "keys": mapping}, ensure_ascii=True)
        return result
    if isinstance(value, (list, tuple)):
        return [_encode_mat_fields(item, max_length) for item in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = _encode_mat_fields(value[index], max_length)
        return result
    return value


def decode_mat_fields(value):
    """Restore nested keys after ``scipy.io.loadmat(..., simplify_cells=True)``.

    Writers preserve ordinary struct keys and flat variable names. A struct with
    escaped keys carries ``polymerfts_mat_keys_json``, a versioned JSON mapping
    from stored keys to original names. Arrays are not converted to Python lists.
    A legacy numeric field with the marker's name remains an ordinary field.
    Malformed string mappings raise ValueError rather than overwrite user data.
    """
    if isinstance(value, dict):
        mapping = {}
        marker = value.get(_MAT_FIELD_KEYS)
        if isinstance(marker, str):
            try:
                record = json.loads(marker)
                if (not isinstance(record, dict) or set(record) != {"version", "keys"}
                        or type(record["version"]) is not int or record["version"] != 1
                        or not isinstance(record["keys"], dict) or not record["keys"]):
                    raise ValueError("Invalid mapping format or version")
                mapping = record["keys"]
                if any(not isinstance(key, str) or not isinstance(name, str)
                       or key == _MAT_FIELD_KEYS or key not in value
                       for key, name in mapping.items()):
                    raise ValueError("Invalid mapped field")
                names = [mapping.get(key, key) for key in value if key != _MAT_FIELD_KEYS]
                if len(set(names)) != len(names):
                    raise ValueError("Mapped fields have duplicate names")
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid MAT field-name mapping") from error
        return {mapping.get(key, key): decode_mat_fields(item)
                for key, item in value.items() if not mapping or key != _MAT_FIELD_KEYS}
    if isinstance(value, (list, tuple)):
        return [decode_mat_fields(item) for item in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = decode_mat_fields(value[index])
        return result
    return value


def read_monomer_fields(data, names=None):
    """Read flat monomer potentials using one representation for the whole file.

    SCFT/LFTS write w_<name>; CLFTS writes w_<name>_real and w_<name>_imag.
    Names such as A_real are valid, so a bare key cannot identify the format of
    one field. Prefer the writer's metadata; legacy files need exactly one
    complete representation across their monomer types. Bare complex arrays
    without simulation metadata are accepted as well.
    """
    if not isinstance(data, dict):
        raise ValueError("A fields file must contain a dictionary")

    def monomers(value):
        result = [str(name).strip() for name in np.atleast_1d(value).ravel()]
        if not result or any(not name for name in result) or len(set(result)) != len(result):
            raise ValueError("Monomer types must be distinct, nonempty names")
        return result

    stored_names = monomers(data["monomer_types"]) if "monomer_types" in data else None
    requested = monomers(names) if names is not None else stored_names
    if requested is None:
        raise ValueError("The file needs monomer_types, or the reader must supply the monomer names")
    all_names = stored_names if stored_names is not None else requested
    if any(name not in all_names for name in requested):
        missing = ["w_" + name for name in requested if name not in all_names]
        raise ValueError("The file holds no monomer field " + ", ".join(missing))

    formats = []
    for key in ("checkpoint_settings_json", "simulation_metadata_json"):
        if key not in data:
            continue
        try:
            metadata = json.loads(str(np.asarray(data[key]).item()))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON dictionary")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fields-file metadata '{key}'") from exc
        if key == "checkpoint_settings_json":
            simulation = metadata.get("simulation")
            if simulation in ("SCFT", "LFTS", "CLFTS"):
                formats.append("split" if simulation == "CLFTS" else "bare")
        else:
            precision = metadata.get("precision")
            if precision in ("float64", "complex128"):
                formats.append("split" if precision == "complex128" else "bare")
    if len(set(formats)) > 1:
        raise ValueError("Fields-file metadata disagrees about the monomer-field representation")

    def keys(kind):
        suffixes = ("",) if kind == "bare" else ("_real", "_imag")
        return ["w_" + name + suffix for name in all_names for suffix in suffixes]

    if formats:
        representation = formats[0]
    else:
        complete = [kind for kind in ("bare", "split") if all(key in data for key in keys(kind))]
        if len(complete) != 1:
            if complete:
                raise ValueError("Ambiguous monomer fields: both bare and split representations are complete; "
                                 "include the simulation metadata")
            missing = sorted({key for kind in ("bare", "split") for key in keys(kind) if key not in data})
            raise ValueError("No complete monomer-field representation; missing " + ", ".join(missing))
        representation = complete[0]
    missing = [key for key in keys(representation) if key not in data]
    if missing:
        raise ValueError("The file is missing " + ", ".join(missing))

    fields = {}
    for name in requested:
        key = "w_" + name
        if representation == "bare":
            field = np.asarray(data[key])
        else:
            real, imag = (np.asarray(data[key + suffix]) for suffix in ("_real", "_imag"))
            if real.shape != imag.shape or np.iscomplexobj(real) or np.iscomplexobj(imag):
                raise ValueError(f"{key}_real and {key}_imag must be real arrays of the same shape")
            if real.dtype.kind not in "fiub" or imag.dtype.kind not in "fiub":
                raise ValueError(f"{key} must contain numeric fields")
            if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
                raise ValueError(f"{key} must contain one finite value per grid point")
            field = real.astype(np.float64) + 1j * imag.astype(np.float64)
        if field.dtype.kind not in "fciub" or not np.all(np.isfinite(field)):
            raise ValueError(f"{key} must contain one finite value per grid point")
        fields[name] = np.ravel(field)
    return fields
