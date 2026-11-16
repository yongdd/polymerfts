"""Coexistence of a coacervate and its supernatant from a Gibbs-ensemble CL-FTS.

A polycation P and a polyanion M (N = 32, one charge per segment) in a solvent S of one segment,
chi N = 24 with the solvent. Two boxes exchange neutral combinations of chains, ions and solvent, and
with them volume, at fixed totals (polymerfts.GibbsEnsemble, docs/theory/ComplexLangevinSimulation.md,
Section 8). The table compares three systems from the same start. Flory-Huggins theory gives the polymer
fractions phi = 0.0291 and 0.3665. The fluctuations of neutral chains narrow the coexistence, and those
of the charges widen it. A salt of point ions, which have no volume and no chi, screens the charges.

Both boxes run in the same cell; their volumes in the table are those of the two phases, which only
count the material for the lever rule. The smearing length 0.2 R0 exceeds the grid spacing 0.1875 R0,
so the osmotic pressure in the exchange potentials does not depend on the lattice. The exchange potentials
of the two boxes agree in every neutral combination at coexistence; the table shows those of the polymer
pair, the solvent and the salt. The CL-FTS
output of the boxes goes to data_coacervate_<name>/box0 and box1, the history of the cycles to
gibbs_history.json there.
"""

import os

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np

from polymerfts import GibbsEnsemble

N = 32              # Segments of the chains, 1/ds
chi_n = 24.0        # chi N of the chains with the solvent
coupling = 1e4      # E = 4 pi l_B rho_0 N R0^2
a = 0.2             # Gaussian smearing length of the charges and fields, in R0

# Name, charged chains, concentration of each ion in ions per segment
systems = [("uncharged", False, 0.0), ("charged", True, 0.0), ("salt", True, 0.1)]


def parameters(name, charged, c_salt):
    params = {
        "nx": [16, 16, 16],             # Grid of each box
        "lx": [3.0, 3.0, 3.0],          # Cell of both boxes
        "chain_model": "discrete",
        "ds": 1 / N,
        "segment_lengths": {"P": 1.0, "M": 1.0, "S": 1.0},
        "chi_n": {"P,S": chi_n, "M,S": chi_n},
        "zeta_n": 100.0,

        # Polycation, polyanion and solvent; their volume fractions in each box come from gibbs_ensemble.boxes
        "polymers": [{"volume_fraction": 0.1, "blocks": [{"type": "P", "length": 1.0}]},
                     {"volume_fraction": 0.1, "blocks": [{"type": "M", "length": 1.0}]},
                     {"volume_fraction": 0.8, "blocks": [{"type": "S", "length": 1 / N}]}],

        "smearing": {"type": "gaussian", "a_int": a},

        "langevin": {"dt": 1.0, "nbar": 1.6e5, "max_step": 1, # max_step: set by the cycles
                     "integrator": "etd"},                    # Exact linear response of the homogeneous state

        "recording": {"dir": f"data_coacervate_{name}", "checkpoint_interval": 1, "sample_interval": 2,
                      "statistics_save_interval": 1},         # intervals: set by the cycles

        "gibbs_ensemble": {
            # Starting compositions; the totals stay fixed
            "boxes": [{"volume_fractions": [phi / 2, phi / 2, 1 - phi]} for phi in (0.025, 0.37)],
            "cycles": 150,              # Exchanges
            "cycle_steps": 500,         # Langevin steps of each box per exchange
            "skip_steps": 100,          # Steps at the start of each cycle left out of the averages
            "exchange_rate": 0.5,       # Rate of the exchange of chains and volume
        },
        "verbose_level": 0,
    }
    if charged:
        params["electrostatics"] = {"charges": {"P": 1.0, "M": -1.0}, "coupling": coupling}
    if c_salt > 0:
        ions = {"valence": 1.0, "concentration": c_salt, "smearing_length": a}
        params["electrostatics"]["ions"] = {"Na": ions, "Cl": dict(ions, valence=-1.0)}
    return params


rows = []
for name, charged, c_salt in systems:
    gibbs = GibbsEnsemble(parameters(name, charged, c_salt), random_seed=2010)
    summary = gibbs.run()
    last = gibbs.history["cycles"][-summary["averaged_cycles"]:]
    # Difference of the exchange potentials of the boxes in the neutral combinations P + M, S and Na + Cl
    potentials = np.array([[box["exchange_potentials"] for box in cycle["boxes"]] for cycle in last])
    differences = potentials[:, 0] - potentials[:, 1]
    pair, solvent = differences[:, 0] + differences[:, 1], differences[:, 2]
    salt = differences[:, 3] + differences[:, 4] if c_salt > 0 else None
    rows.append((name, summary["boxes"], pair, solvent, salt))

print()
print("  system      phi (box 0)   phi (box 1)   volumes        d(mu_P + mu_M)     d(mu_S) * 2 N      d(mu_Na + mu_Cl)")
for name, boxes, pair, solvent, salt in rows:
    phi = [sum(box["volume_fractions"][:2]) for box in boxes]
    print(f"  {name:10s}  {phi[0]:.4f}        {phi[1]:.4f}        {boxes[0]['volume']:5.2f} {boxes[1]['volume']:5.2f}"
          f"    {pair.mean():+.4f} +- {pair.std(ddof=1) / np.sqrt(len(pair)):.4f}"
          f"   {2 * N * solvent.mean():+.4f} +- {2 * N * solvent.std(ddof=1) / np.sqrt(len(solvent)):.4f}"
          + (f"   {salt.mean():+.1e} +- {salt.std(ddof=1) / np.sqrt(len(salt)):.1e}" if salt is not None else ""))
