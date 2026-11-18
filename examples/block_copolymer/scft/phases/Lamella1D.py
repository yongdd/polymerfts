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
f = 0.5         # A-fraction of major BCP chain, f

params = {
    "nx":[32],          # Simulation grid numbers
    "lx":[1.5],        # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                        # where "a_Ref" is reference statistical segment length
                        # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "stress_interval":1,        # Compute stress every iteration (for reproducibility)
    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 13.27},    # Interaction parameter, Flory-Huggins params * N_Ref

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

# Set initial fields: one cosine period (seeds.py)
seed = seeds.lamellae(params["nx"], params["lx"])
print("w_A and w_B are initialized to lamellar phase.")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("L1D.json")

# Recording first a few iteration results for debugging and refactoring
#        1    6.245E-16  [ 1.0476635E+00  ]    -0.010902630   1.9193450E-01  lx=[  1.5000000 ]
#        2    1.632E-16  [ 1.0499908E+00  ]    -0.011429389   1.9368282E-01  lx=[  1.4989939 ]
#        3    1.596E-16  [ 1.0524094E+00  ]    -0.011974402   1.9543028E-01  lx=[  1.4979574 ]
#        4    1.006E-16  [ 1.0549473E+00  ]    -0.012544007   1.9718508E-01  lx=[  1.4968796 ]
#        5    1.259E-15  [ 1.0576104E+00  ]    -0.013139495   1.9894012E-01  lx=[  1.4957592 ]
