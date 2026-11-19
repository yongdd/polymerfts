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
f = 1.0/3.0     # A-fraction of major BCP chain, f

params = {
    "nx":[48,32],                  # Simulation grid numbers
    "lx":[2.7,1.6],                # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                   # where "a_Ref" is reference statistical segment length
                                   # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/90,                       # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 15},       # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields (seeds.py)
seed = seeds.cylinders(params["nx"], params["lx"])
print("w_A and w_B are initialized to cylindrical phase.")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("C2D.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -2.846E-15  [ 1.1072987E+00  ]     0.323240690   1.5324187E+00  lx=[  2.7000000, 1.6000000 ]
#        2   -8.092E-16  [ 1.0699399E+00  ]     0.137080934   1.0238845E+00  lx=[  2.7025811, 1.6048410 ]
#        3    1.581E-16  [ 1.0554717E+00  ]     0.068675649   7.2534463E-01  lx=[  2.7040263, 1.6076075 ]
#        4    6.243E-16  [ 1.0490457E+00  ]     0.039038563   5.5394703E-01  lx=[  2.7050075, 1.6095148 ]
#        5   -5.220E-16  [ 1.0459471E+00  ]     0.024399034   4.5908209E-01  lx=[  2.7057610, 1.6109921 ]
