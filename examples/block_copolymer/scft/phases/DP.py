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
f = 0.40       # A-fraction of major BCP chain, f

# Initial fields and cell: a converged double primitive (DP.mat), interpolated to the grid nx
nx = [48, 48, 48]
seed = seeds.from_file("DP.mat", nx)

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
        "symbol":"Im-3m",  # IT symbol of the space group
        "hall_number": 529,     # (optional) Hall number of the space group
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
calculation.save_results("DP.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -2.050E-15  [ 3.5487081E+00  ]    -0.148662695   1.6862386E-02  lx=[  2.3750164, 2.3750164, 2.3750164 ]
#        2    4.851E-15  [ 3.5561440E+00  ]    -0.148472545   1.5883191E-02  lx=[  2.3747624, 2.3747624, 2.3747624 ]
#        3   -7.942E-15  [ 3.5632193E+00  ]    -0.148283298   1.4954160E-02  lx=[  2.3745224, 2.3745224, 2.3745224 ]
#        4   -2.574E-15  [ 3.5699390E+00  ]    -0.148096460   1.4074116E-02  lx=[  2.3742962, 2.3742962, 2.3742962 ]
#        5   -5.231E-15  [ 3.5763099E+00  ]    -0.147913214   1.3241973E-02  lx=[  2.3740835, 2.3740835, 2.3740835 ]
