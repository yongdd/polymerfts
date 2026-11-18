"""
Hexagonally Perforated Lamellar (HPL) Phase SCFT Simulation - Hexagonal Crystal System

This uses the hexagonal crystal system for the perforated lamellar phase.
The structure consists of lamellar layers with hexagonally arranged perforations,
where perforations in adjacent layers are staggered (ABAB stacking).

Crystal System: Hexagonal (a = b, alpha = beta = 90 deg, gamma = 120 deg)
Space Group: P6_3/mmc (No. 194, Hall number 488)

Axis ordering: [a, b, c] - standard crystallographic convention
- a, b axes: in-plane hexagonal directions (perforation arrangement)
- c axis: lamellar stacking direction

Initial fields loaded from PL.mat (converged PSCF solution).

Reference:
- Loo et al., Macromolecules 2005, 38, 4947

Results:
- Free energy: F = -0.2103856
- Box size: lx = [1.9607, 1.9607, 2.9518] (a = b, hexagonal, c/a ~ 1.505)
"""

import os
import sys

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import scft

# Initial fields of common phases (seeds.py in the folder above)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import seeds

# Major Simulation params
f = 0.4       # A-fraction of major BCP chain, f

params = {
    # HPL with Hexagonal crystal system and P6_3/mmc space group
    # Axis ordering: [a, b, c] - gamma=120 deg between a and b (axes 0,1)
    # Grid: nx[0] = nx[1]; the 6_3 screw of P6_3/mmc translates by c/2, so nz must be even, and
    # a multiple of 4 enables the faster z-mirror basis
    "nx": [48, 48, 72],             # Simulation grid numbers [a, b, c]
    "lx": [1.96, 1.96, 2.98],       # Box size [a, b, c] with a=b (near equilibrium)
    "angles": [90.0, 90.0, 120.0],  # Hexagonal: gamma=120 (between a,b)

    "reduce_memory": False,         # Reduce memory usage by storing only check points
    "relax_cell": True,        # Find box size that minimizes the free energy
    "stress_interval": 1,           # Compute stress every iteration
    "chain_model": "continuous",    # "discrete" or "continuous" chain model
    "ds": 1/100,                    # Contour step interval = 1/N_Ref

    "segment_lengths": {            # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 15},           # Flory-Huggins parameter * N_Ref

    "polymers": [{         # Polymer species
        "volume_fraction": 1.0,     # Volume fraction
        "blocks": [                 # AB diblock copolymer
            {"type": "A", "length": f},       # A-block
            {"type": "B", "length": 1-f},     # B-block
        ],
    }],

    "crystal_system": "Hexagonal",  # Relaxes a = b and c at gamma = 120 deg (the space group sets the same)

    "space_group": {
        "symbol": "P6_3/mmc",       # International symbol for HPL space group (No. 194)
        "hall_number": 488,              # Hall number
    },

    "optimizer": {
        "name": "am",               # Anderson Mixing
        "max_hist": 20,             # Maximum number of history
        "start_error": 1e-2,        # When switch to AM from simple mixing
        "mix_min": 0.1,             # Minimum mixing rate of simple mixing
        "mix_init": 0.1,            # Initial mixing rate of simple mixing
    },

    "max_iter": 2000,               # Maximum relaxation iterations
    "tolerance": 1e-8               # Convergence tolerance
}

# Initial fields: a converged PSCF solution at chi N = 50 (PL.mat), whose axes [c, a, b]
# become [a, b, c], interpolated to the grid and rescaled to chi N of params
seed = seeds.from_file(os.path.join(os.path.dirname(__file__), "PL.mat"), params["nx"], axes=(1, 2, 0))
chi_n_source = 50
chi_n_target = params["chi_n"]["A,B"]
w = {monomer: field * chi_n_target / chi_n_source for monomer, field in seed["w"].items()}

print(f"Initial field loaded from PL.mat (rescaled chi_n: {chi_n_source} -> {chi_n_target})")
for monomer, field in w.items():
    print(f"  w_{monomer}: min={field.min():.2f}, max={field.max():.2f}, mean={field.mean():.2f}")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=w)

# Save final results
calculation.save_results("PL_Hexagonal.json")

# Recording iteration results for debugging and refactoring
# Equilibrium: F = -0.2103856, lx = [1.9607, 1.9607, 2.9518], gamma = 120 deg
# Space group: P6_3/mmc (No. 194, Hall 488) - staggered perforations (ABAB)
# (with f=0.4, chi_n=15)
