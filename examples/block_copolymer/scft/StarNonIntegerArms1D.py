"""Lamellar phase of a symmetric A_n B_n star with a real arm number n (1D SCFT).

Each block is given once with a "multiplicity": n copies of the A arm and n copies
of the B arm meet at node 0. For a real n the junction takes the arm propagators to
the power n (n - 1 for the propagator that enters an arm), which continues the
integer arm number analytically. The symmetric A_n B_n melt has a tricritical
order-disorder transition at n_tc ≈ 5.4475: below it the lamellae vanish
continuously at (chiN)_s ≈ 10.495, above it the transition is first order and lies
below (chiN)_s (Kim, Kang, Cho, Yong, Kim, Macromolecules 59, 11903 (2026)).
The free energy is per reference chain, here one AB arm pair (alpha = n).
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import scft

n = 5.6         # number of A arms and of B arms (any real number >= 1)

params = {
    # "platform":"cpu-fftw",           # choose platform among [cuda, cpu-fftw, cpu-mkl]

    "nx":[64],          # Simulation grid numbers
    "lx":[1.45],        # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                        # where N_Ref is the length of one AB arm pair.

    "relax_cell":True,  # Relax the lamellar period
    "chain_model":"continuous",   # Continuous chains: real multiplicities need real fields
    "ds":1/400,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to a_Ref.
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B": 10.6},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{
        "volume_fraction":1.0,
        "blocks":[              # A_n B_n star: n copies of each arm attached at node 0
            {"type":"A", "length":0.5, "v":0, "u":1, "multiplicity":n},
            {"type":"B", "length":0.5, "v":0, "u":2, "multiplicity":n},
        ],},],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-10    # Terminate iteration if the self-consistency error is less than tolerance
}

# Lamellar initial fields
x = (np.arange(params["nx"][0]) + 0.5) / params["nx"][0]
w_A = 3.0 * np.cos(2 * np.pi * x)
w_B = -3.0 * np.cos(2 * np.pi * x)

calculation = scft.SCFT(params=params)
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Free energy relative to the disordered melt, per AB arm pair
print(f"A{n}B{n} star, chiN = {params['chi_n']['A,B']}: F = {calculation.free_energy:.8e}, "
      f"period = {calculation.prop_solver.get_lx()[0]:.6f}")
calculation.save_results("fields.mat")
