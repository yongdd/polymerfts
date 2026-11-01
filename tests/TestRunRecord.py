#!/usr/bin/env python3
"""RunRecord: the table of what a Langevin run records.

A quantity is one entry of `series` and one add(): its statistics file key, checkpoint
arrays and restart follow from the table. Checks the keys and shapes of the files, the
split of complex series, the restart of squeezed arrays (a MATLAB file drops the axes
of one row or one value), NaN for pending samples of a series that a checkpoint lacks,
and the sums, which restart from zero when the checkpoint's run did not use them.
"""

import numpy as np

from polymerfts.utils.checkpoint import RunRecord, _measurement_key


def checkpoint_of(record):
    """The arrays of a checkpoint as loadmat(..., squeeze_me=True) returns them."""
    return {key: np.squeeze(value) for key, value in record.checkpoint_arrays().items()}


def main():
    # A new quantity needs its entry and add(): the rest follows
    sums = {"A,B": np.zeros(4, np.complex128)}
    record = RunRecord({"H_history": 0, "drift_history": 0, "dH A,B": 0, "radius_history": 2}, sums)
    record.add("H_history", 1.5)
    record.add("drift_history", 0.25)
    record.add("dH A,B", 2.0)
    record.add("radius_history", [3.0, 4.0])
    record.sample_count += 1
    sums["A,B"] += 1.0 + 2.0j
    data = record.statistics()
    assert data["sample_count"] == 1
    assert data["H_history"].tolist() == [1.5] and data["drift_max_history"].tolist() == [0.25]
    assert data["dH_history_A_B"].tolist() == [2.0] and data["radius_history"].tolist() == [[3.0, 4.0]]

    # A checkpoint restores one sample in the shapes of the run, and the next add() goes on from it
    restored = RunRecord({"H_history": 0, "drift_history": 0, "dH A,B": 0, "radius_history": 2},
                         {"A,B": np.zeros(4, np.complex128)})
    restored.restore(checkpoint_of(record), 8)
    restored.add("radius_history", [5.0, 6.0])
    assert restored.array("radius_history").tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert restored.sample_count == 1 and restored.lists["H_history"] == [1.5]
    assert np.array_equal(restored.sums["A,B"], sums["A,B"])

    # clear() empties the lists and the sums in place, and starts at the next step
    record.clear(9)
    assert record.lists["H_history"] == [] and record.sample_count == 0 and record.first_step == 9
    assert np.all(sums["A,B"] == 0.0)

    # Complex series go into the statistics files as real and imaginary parts; the drifts stay real
    series = {"H_history": 0, "drift_history": 0, "drift_step_history": 2, "dH A,B": 0, "pressure_history": 0,
              "chemical_potential_history": 3, "partition_ratio_step_history": 2}
    record = RunRecord(series, {"concentration_A": np.zeros(2, np.complex128)}, complex_values=True)
    data = record.statistics()       # with nothing recorded the keys and shapes stay
    assert {"H_history_real", "H_history_imag", "dH_history_A_B_real", "dH_history_A_B_imag", "pressure_history_real",
            "drift_max_history", "drift_max_step_history", "partition_ratio_step_history"} <= set(data)
    assert "drift_max_history_real" not in data
    assert data["chemical_potential_history_real"].shape == (0, 3) and data["drift_max_step_history"].shape == (0, 2)

    # Two samples and three pending steps; the record of the steps started at step 4
    record.clear(4)
    for h in (1.0 + 1.0j, 2.0 + 2.0j):
        record.add("H_history", h)
        record.add("drift_history", 0.5)
        record.add("dH A,B", 1.0j)
        record.add("pressure_history", 3.0 + 0.0j)
        record.add("chemical_potential_history", np.array([1.0, 2.0, 3.0]))
        record.sample_count += 1
    for step in range(3):
        record.add("drift_step_history", [0.1, 0.2])
        record.add("partition_ratio_step_history", [0.5, 0.25])
    arrays = record.checkpoint_arrays()
    assert arrays["record_drift_step_start"] == 4 and arrays["record_drift_step_history"].shape == (3, 2)
    assert np.iscomplexobj(arrays["record_pressure_history"])
    assert np.array_equal(record.statistics()["H_history_imag"], [1.0, 2.0])

    # A checkpoint of an earlier version: NaN for the samples and steps it lacks
    data = checkpoint_of(record)
    for key in ("record_pressure_history", "record_chemical_potential_history", "record_partition_ratio_step_history",
                _measurement_key("sf", "concentration_A")):
        del data[key]
    restored = RunRecord(dict(series), {"concentration_A": np.zeros(2, np.complex128)}, complex_values=True)
    restored.restore(data, 7)
    assert restored.first_step == 7 - 3
    assert restored.array("pressure_history").shape == (2,) and np.all(np.isnan(restored.array("pressure_history")))
    assert restored.array("chemical_potential_history").shape == (2, 3)
    assert np.all(np.isnan(restored.array("chemical_potential_history")))
    assert restored.array("partition_ratio_step_history").shape == (3, 2)
    assert np.all(np.isnan(restored.array("partition_ratio_step_history")))
    # A quantity that is new to the run keeps its rows in line with the samples of the checkpoint
    new = RunRecord({**series, "radius_history": 0}, {"concentration_A": np.zeros(2, np.complex128)}, complex_values=True)
    new.restore(data, 7)
    new.add("radius_history", 4.0)
    assert new.array("radius_history").shape == (3,) and np.all(np.isnan(new.array("radius_history")[:2]))
    assert new.array("radius_history")[2] == 4.0
    # A sum the checkpoint lacks starts from zero, and restore() of nothing leaves an empty record
    assert np.all(restored.sums["concentration_A"] == 0.0)
    empty = RunRecord(dict(series), {}, complex_values=True)
    empty.restore(None, 5)
    assert empty.first_step == 5 and empty.sample_count == 0 and empty.lists["H_history"] == []
    print("RunRecord: OK")


if __name__ == "__main__":
    main()
