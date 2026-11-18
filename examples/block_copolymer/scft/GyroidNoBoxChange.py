import os

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import scft
import seeds     # initial fields of common phases (seeds.py)

# Major Simulation params
f = 0.36        # A-fraction of major BCP chain, f

params = {
    # "platform":"cpu-fftw",           # choose platform among [cuda, cpu-fftw, cpu-mkl]
    
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[3.3,3.3,3.3],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":False,    # Keep the cell fixed during SCFT iteration.
    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 20},       # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "space_group" :{
        "symbol":"Ia-3d",  # IT symbol of the space group
        "hall_number": 530,     # (optional) Hall number of the space group
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

# Set initial fields: the level-set approximation of the double gyroid (seeds.py)
seed = seeds.gyroid(params["nx"], params["lx"])
print("w_A and w_B are initialized to gyroid phase.")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Run
calculation.run(initial_fields=seed["w"])

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #        1    4.709E-16  [ 1.0105497E+00  ]    -0.002963507   1.4442873E+00
    #        2    2.579E-16  [ 1.0113911E+00  ]    -0.007049824   1.0809692E+00
    #        3   -3.003E-17  [ 1.0128754E+00  ]    -0.009274646   8.8577386E-01
    #        4   -1.184E-15  [ 1.0147759E+00  ]    -0.010907833   7.8419605E-01
    #        5    2.635E-17  [ 1.0170333E+00  ]    -0.012485666   7.3162863E-01
