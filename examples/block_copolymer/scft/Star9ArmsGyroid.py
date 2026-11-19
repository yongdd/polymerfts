import os

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import scft
import seeds     # initial fields of common phases (seeds.py)

# Major Simulation params
f = 0.36       # A-fraction of major BCP chain, f

params = {
    # "platform":"cpu-fftw",           # choose platform among [cuda, cpu-fftw, cpu-mkl]
    
    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[7.15,7.15,7.15],  # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/100,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 10.0},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # 9-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":0, "u":2},    # A-block
            {"type":"A", "length":f,   "v":0, "u":3},    # A-block
            {"type":"A", "length":f,   "v":0, "u":4},    # A-block
            {"type":"A", "length":f,   "v":0, "u":5},    # A-block
            {"type":"A", "length":f,   "v":0, "u":6},    # A-block
            {"type":"A", "length":f,   "v":0, "u":7},    # A-block
            {"type":"A", "length":f,   "v":0, "u":8},    # A-block
            {"type":"A", "length":f,   "v":0, "u":9},    # A-block
            {"type":"B", "length":1-f, "v":1, "u":10},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":11},    # B-block
            {"type":"B", "length":1-f, "v":3, "u":12},    # B-block
            {"type":"B", "length":1-f, "v":4, "u":13},    # B-block
            {"type":"B", "length":1-f, "v":5, "u":14},    # B-block
            {"type":"B", "length":1-f, "v":6, "u":15},    # B-block
            {"type":"B", "length":1-f, "v":7, "u":16},    # B-block
            {"type":"B", "length":1-f, "v":8, "u":17},    # B-block
            {"type":"B", "length":1-f, "v":9, "u":18},    # B-block
        ],},],

    "max_iter":2000,      # The maximum relaxation iterations
    "tolerance":1e-8,     # Terminate iteration if the self-consistency error is less than tolerance

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":60,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.02,         # Minimum mixing rate of simple mixing
        "mix_init":0.02,        # Initial mixing rate of simple mixing
    },
}

# Set initial fields: the level-set approximation of the double gyroid (seeds.py)
seed = seeds.gyroid(params["nx"], params["lx"], periods=2)
print("w_A and w_B are initialized to gyroid phase (two unit cells along each axis).")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #        1    6.750E-16  [ 1.1876171E+00  ]    -0.002372721   1.2529953E+00  lx=[  7.1500000, 7.1500000, 7.1500000 ]
    #        2   -3.710E-16  [ 1.1890070E+00  ]    -0.004786347   1.1394238E+00  lx=[  7.1499870, 7.1499870, 7.1499870 ]
    #        3    6.697E-16  [ 1.1913107E+00  ]    -0.006673674   1.0431815E+00  lx=[  7.1499733, 7.1499733, 7.1499733 ]
    #        4   -4.788E-15  [ 1.1943605E+00  ]    -0.008158554   9.6203651E-01  lx=[  7.1499591, 7.1499591, 7.1499591 ]
    #        5   -2.205E-15  [ 1.1980285E+00  ]    -0.009337239   8.9394663E-01  lx=[  7.1499446, 7.1499446, 7.1499446 ]
