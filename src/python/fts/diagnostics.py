"""Diagnostics of a CL-FTS run, printed at its end: the distribution of the largest drift of every step and
the partition ratios of the single-site species.

The run records the values of every step (drift_max_step_history, psi_drift_max_step_history,
partition_ratio_step_history); RunDiagnostics summarizes them over the steps of one run.
"""

import json

import numpy as np


class RunDiagnostics:
    """The summaries of one CL-FTS run: add() the largest drifts and the partition ratios of a step, report()
    them at the end. `statistics_entries` names the species of the partition ratios in the statistics files."""

    def __init__(self, sim):
        # The largest drift of every step for each auxiliary field, named by its kind, and for ψ
        kinds = {i: "real" for i in sim.mpt.aux_fields_real_idx} | {i: "imaginary" for i in sim.mpt.aux_fields_imag_idx}
        labels = [f"auxiliary field {i} ({kinds.get(i, 'dropped')})" for i in range(len(sim.monomer_types))]
        self.drift_tail = DriftTailSummary(labels + (["psi"] if sim.electrostatics is not None else []))
        # |Q_s|/<|e^(-u_s)|> of the single-site species, which the statistics files name
        species = sim.thermodynamics.partition_ratio_species
        self.partition_ratios = PartitionRatioSummary(species)
        self.statistics_entries = {"partition_ratio_species_json": json.dumps(species)} if species else {}

    def add(self, field_drift_max, psi_drift_max=None, ratios=None):
        """One step: the largest drift of each auxiliary field, that of ψ (None without electrostatics) and
        the partition ratios (None without single-site species)."""
        self.drift_tail.add(np.r_[field_drift_max, [] if psi_drift_max is None else [psi_drift_max]])
        if ratios is not None:
            self.partition_ratios.add(ratios)

    def report(self):
        self.drift_tail.report()
        self.partition_ratios.report()


class DriftTailSummary:
    """Distribution of the largest drift of each step, summarized at the end of a CL-FTS run.

    Complex Langevin averages need the distribution of the drift magnitude to fall off
    exponentially or faster (Nagata, Nishimura & Shimasaki, Phys. Rev. D 94, 114515 (2016)).
    The counts stay in memory, twenty bins per decade (the printed median and maximum are
    bin edges); the step values themselves are recorded as drift_max_step_history and
    psi_drift_max_step_history. The summary gives
    numbers, not a verdict: a finite run cannot show an exponential tail, and the weight of
    a power-law tail sets the size of the bias.
    """
    EDGES = 10.0 ** (np.arange(-120, 241) / 20)

    def __init__(self, labels):
        self.labels = list(labels)
        self.counts = np.zeros((len(self.labels), self.EDGES.size + 1), dtype=np.int64)

    def add(self, values):
        self.counts[np.arange(len(self.labels)), np.searchsorted(self.EDGES, values)] += 1

    def report(self):
        steps = int(self.counts[0].sum()) if self.labels else 0
        if steps == 0:
            return
        print(f"Largest drift per step ({steps} steps of this run); S(u) = P(drift > u) at 10 and 100 "
              "times the median, and the log-log slope of S between them (scripts/analysis/drift_tail.py):")
        for label, counts in zip(self.labels, self.counts):
            beyond = steps - np.cumsum(counts)          # steps above EDGES[k]
            k_median = int(np.searchsorted(-beyond, -steps / 2))
            if k_median >= self.EDGES.size:
                continue
            median = self.EDGES[k_median]
            if median <= self.EDGES[0]:
                print(f"  {label}: no drift")
                continue
            k10 = min(np.searchsorted(self.EDGES, 10 * median), self.EDGES.size - 1)
            k100 = min(np.searchsorted(self.EDGES, 100 * median), self.EDGES.size - 1)
            s10, s100 = beyond[k10] / steps, beyond[k100] / steps
            slope = f"{np.log(s100 / s10) / np.log(self.EDGES[k100] / self.EDGES[k10]):.2f}" \
                if beyond[k100] >= 20 else "-"
            largest = self.EDGES[min(np.flatnonzero(counts).max(), self.EDGES.size - 1)]
            print(f"  {label}: median ~{median:.3g}, maximum < {largest:.3g}, "
                  f"S(10x) {s10:.3g}, S(100x) {s100:.3g}, slope {slope}")


class PartitionRatioSummary:
    """Smallest |Q_s|/<|e^(-u_s)|> of each canonical single-site species (ThermodynamicOperators.partition_ratios) over
    a run, and the fractions of steps below 0.1 and 0.01. The step values are recorded as
    partition_ratio_step_history. Like the drift summary it gives numbers, not a verdict: a ratio
    near 0 means Q_s came close to its zero, the pole of -n_s ln Q_s."""
    THRESHOLDS = (0.1, 0.01)

    def __init__(self, labels):
        self.labels = list(labels)
        self.steps = 0
        self.smallest = np.full(len(self.labels), np.inf)
        self.below = np.zeros((len(self.labels), len(self.THRESHOLDS)), dtype=np.int64)

    def add(self, ratios):
        self.steps += 1
        self.smallest = np.minimum(self.smallest, ratios)
        self.below += np.asarray(ratios)[:, None] < np.asarray(self.THRESHOLDS)[None, :]

    def report(self):
        if self.steps == 0 or not self.labels:
            return
        print(f"Partition ratio |Q_s|/<|e^(-u_s)|> of the single-site species ({self.steps} steps of this run; "
              "near 0: close to the pole of -n_s ln Q_s):")
        for label, smallest, below in zip(self.labels, self.smallest, self.below):
            print(f"  {label}: minimum {smallest:.3g}, steps below 0.1 {below[0] / self.steps:.3g}, "
                  f"below 0.01 {below[1] / self.steps:.3g}")
