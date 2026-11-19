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

# Initial fields and cell: a converged double diamond (DD.mat), interpolated to the grid nx
nx = [48, 48, 48]
seed = seeds.from_file("DD.mat", nx)

params = {
    "nx":nx,                    # Simulation grid numbers
    "lx":seed["lx"],            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "stress_scale": 1.0,

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 15},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
    
    "space_group" :{
        "symbol":"Pn-3m",  # IT symbol of the space group
        "hall_number": 522,     # (optional) Hall number of the space group
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

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("DD.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -2.779E-15  [ 5.5192213E+06  ]     1.788033367   2.7878734E+00  lx=[  1.3521587, 1.3521587, 1.3521587 ]
#        2    4.527E-15  [ 1.7858209E+06  ]     1.336931553   2.1778370E+00  lx=[  1.4756274, 1.4756274, 1.4756274 ]
#        3   -5.008E-15  [ 5.6488767E+05  ]     1.028168554   1.8373544E+00  lx=[  1.5659123, 1.5659123, 1.5659123 ]
#        4   -2.681E-15  [ 1.9163812E+05  ]     0.814385428   1.6266268E+00  lx=[  1.6382999, 1.6382999, 1.6382999 ]
#        5    4.111E-15  [ 7.1587807E+04  ]     0.679200770   1.4908740E+00  lx=[  1.7001109, 1.7001109, 1.7001109 ]
