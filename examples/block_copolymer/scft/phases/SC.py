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
f = 0.2       # A-fraction of major BCP chain, f

params = {
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[1.5,1.5,1.5],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 25},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
    
    "space_group" :{
        "symbol":"Pm-3m",   # IT symbol of the space group
        "hall_number": 517,      # (optional) Hall number of the space group
    },
    
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

# Set initial fields: smoothed wells of w_A at the A sphere positions (seeds.py)
sphere_positions = [[0, 0, 0]]
seed = seeds.wells(params["nx"], params["lx"], sphere_positions)
print("w_A and w_B are initialized to SC phase.")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("SC.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -6.644E-16  [ 1.2424863E+00  ]     1.701518256   4.0439346E+00  lx=[  1.5000000, 1.5000000, 1.5000000 ]
#        2   -2.007E-15  [ 1.1029599E+00  ]     0.140644692   1.1343688E+00  lx=[  1.5144880, 1.5144880, 1.5144880 ]
#        3    3.827E-16  [ 1.0973592E+00  ]     0.070045494   7.9569809E-01  lx=[  1.5187366, 1.5187366, 1.5187366 ]
#        4   -6.214E-16  [ 1.0963926E+00  ]     0.042043940   6.5553399E-01  lx=[  1.5219979, 1.5219979, 1.5219979 ]
#        5   -3.958E-15  [ 1.0966084E+00  ]     0.026548850   5.8345729E-01  lx=[  1.5247311, 1.5247311, 1.5247311 ]
