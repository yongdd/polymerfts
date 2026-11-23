#!/usr/bin/env python3
"""
Block-resolved concentrations of a star with identical arms.

Identical blocks (the two equal A arms below) share one computation block that
holds both of them. get_block_concentration() must give each block its own
share, so the block concentrations add up to the total concentration. For each
available platform and both chain models (aggregation off, as the block API
requires) this test checks

  1. check_total_partition() passes,
  2. the sum of all block concentrations equals the sum of the total
     concentrations of A and B pointwise (max abs difference < 1e-12),
  3. material conservation of the totals, |mean(phi_A + phi_B) - 1| < 1e-12.

Fields are smooth random fields with standard deviation 5. Runs in seconds.
Exit code 0 on success, 1 on any failure.
"""

import sys

import numpy as np

from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
import checks
from checks import check, smooth_field, summary
from cuda_part import cuda_part_missing

NX = [12, 10, 8]
LX = [2.0, 1.8, 1.5]
DS = 0.05
TOL = 1e-12

checks.indent = "    "   # the check lines print indented


def main():
    rng = np.random.default_rng(7)
    w = {"A": smooth_field(rng, NX, LX), "B": smooth_field(rng, NX, LX)}
    # Star: two identical A arms and one B arm around node 0
    blocks = [["A", 0.3, 0, 1], ["A", 0.3, 0, 2], ["B", 0.4, 0, 3]]

    for platform in _core.PlatformSelector.avail_platforms():
        for chain_model in ["continuous", "discrete"]:
            tag = f"{platform}/{chain_model}"
            solver = PropagatorSolver(
                nx=NX, lx=LX, ds=DS, bond_lengths={"A": 1.0, "B": 1.0},
                bc=["periodic"] * 6, chain_model=chain_model,
                numerical_method="rqm4" if chain_model == "continuous" else None,
                platform=platform, aggregate_propagator_computation=False)
            solver.add_polymer(1.0, blocks)
            solver.compute_propagators({"A": w["A"].copy(), "B": w["B"].copy()})
            check(solver.check_total_partition(), f"{tag}: check_total_partition()")
            solver.compute_concentrations()
            phi_a = np.asarray(solver.get_concentration("A")).flatten()
            phi_b = np.asarray(solver.get_concentration("B")).flatten()
            phi_blocks = np.asarray(solver._propagator_computation.get_block_concentration(0))
            check(phi_blocks.shape[0] == len(blocks),
                  f"{tag}: one block concentration per block ({phi_blocks.shape[0]})")
            diff = np.max(np.abs(phi_blocks.sum(axis=0) - (phi_a + phi_b)))
            check(diff < TOL, f"{tag}: sum of block concentrations = phi_A + phi_B "
                              f"(max abs diff {diff:.3e} < {TOL})")
            cons = abs(np.mean(phi_a + phi_b) - 1.0)
            check(cons < TOL, f"{tag}: |mean(phi_A + phi_B) - 1| = {cons:.3e} < {TOL}")

    return summary() | cuda_part_missing()


if __name__ == "__main__":
    sys.exit(main())
