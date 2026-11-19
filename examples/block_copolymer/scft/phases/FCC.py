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
f = 24/90       # A-fraction of major BCP chain, f

params = {
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[1.91,1.91,1.91],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/90,                       # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 18.1},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
    
    "space_group" :{
        "symbol":"Fm-3m",   # IT symbol of the space group
        "hall_number": 523,      # (optional) Hall number of the space group
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
sphere_positions = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
seed = seeds.wells(params["nx"], params["lx"], sphere_positions)
print("w_A and w_B are initialized to FCC phase.")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("FCC.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -4.063E-16  [ 1.2029684E+00  ]     0.658794896   1.8147786E+00  lx=[  1.9100000, 1.9100000, 1.9100000 ]
#        2    1.932E-15  [ 1.1319192E+00  ]     0.254319806   1.1487383E+00  lx=[  1.9164471, 1.9164471, 1.9164471 ]
#        3    2.645E-16  [ 1.1089083E+00  ]     0.127488720   8.1038937E-01  lx=[  1.9200544, 1.9200544, 1.9200544 ]
#        4    1.477E-15  [ 1.0997613E+00  ]     0.072778894   6.2131193E-01  lx=[  1.9226493, 1.9226493, 1.9226493 ]
#        5   -7.002E-16  [ 1.0960894E+00  ]     0.044770669   5.1469335E-01  lx=[  1.9247578, 1.9247578, 1.9247578 ]
#        6    9.526E-16  [ 1.0949596E+00  ]     0.028583895   4.5473658E-01  lx=[  1.9265911, 1.9265911, 1.9265911 ]
