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

# Initial fields and cell: a converged single gyroid (SG.mat), interpolated to the grid nx
nx = [48, 48, 48]
seed = seeds.from_file("SG.mat", nx)

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
        "symbol":"I4_132",  # IT symbol of the space group
        "hall_number": 510,     # (optional) Hall number of the space group
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
calculation.save_results("SG.json")

# Recording first a few iteration results for debugging and refactoring
#        1    3.819E-16  [ 4.6413585E+00  ]    -0.202586228   1.4587485E-02  lx=[  2.1000000, 2.1000000, 2.1000000 ]
#        2    8.074E-16  [ 4.6515012E+00  ]    -0.202236384   1.3541228E-02  lx=[  2.0997814, 2.0997814, 2.0997814 ]
#        3    1.564E-15  [ 4.6610170E+00  ]    -0.201893476   1.2553215E-02  lx=[  2.0995788, 2.0995788, 2.0995788 ]
#        4    2.259E-15  [ 4.6699265E+00  ]    -0.201559236   1.1624874E-02  lx=[  2.0993920, 2.0993920, 2.0993920 ]
#        5   -3.654E-15  [ 4.6782516E+00  ]    -0.201235243   1.0756899E-02  lx=[  2.0992207, 2.0992207, 2.0992207 ]
