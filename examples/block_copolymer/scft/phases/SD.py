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

# Initial fields and cell: a converged single diamond (SD.mat), interpolated to the grid nx
nx = [48, 48, 48]
seed = seeds.from_file("SD.mat", nx)

params = {
    "nx":nx,                    # Simulation grid numbers
    "lx":seed["lx"],            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,          # Relax cell lengths and allowed angles during SCFT iteration.
    "stress_interval":1,        # Compute stress every iteration (for reproducibility)
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
        "symbol":"Fd-3m",  # IT symbol of the space group
        "hall_number": 526,     # (optional) Hall number of the space group
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
calculation.save_results("SD.json")

# Recording first a few iteration results for debugging and refactoring
#        1    6.415E-15  [ 4.4570194E+00  ]    -0.200266460   2.0852423E-02  lx=[  2.5748822, 2.5748822, 2.5748822 ]
#        2   -2.582E-15  [ 4.4712041E+00  ]    -0.199743841   1.9356685E-02  lx=[  2.5746150, 2.5746150, 2.5746150 ]
#        3   -1.367E-14  [ 4.4845256E+00  ]    -0.199233780   1.7955758E-02  lx=[  2.5743657, 2.5743657, 2.5743657 ]
#        4   -2.025E-15  [ 4.4970116E+00  ]    -0.198738755   1.6650586E-02  lx=[  2.5741341, 2.5741341, 2.5741341 ]
#        5   -7.950E-15  [ 4.5086922E+00  ]    -0.198261007   1.5440848E-02  lx=[  2.5739201, 2.5739201, 2.5739201 ]
