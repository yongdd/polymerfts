import time
import numpy as np
from polymerfts import lfts

f = 0.5         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[8.0, 8.0, 8.0],       # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/16,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 25},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "platform": "auto",       # Platform: "auto" (CUDA for 2D/3D if available), "cuda", "cpu-mkl" or "cpu-fftw"

    "polymers":[{      # Polymer species
        "volume_fraction":0.7,  # Volume fraction of polymer chain
        "blocks":[              # AB diblock copolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "volume_fraction":0.3,  
        "blocks":[              # Random Copolymer. Currently, Only single block random copolymer is supported.
            {"type":"R", "length":0.5, "monomer_fractions":{"A":0.5, "B":0.5},},
        ],}],

    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":8.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "checkpoint_interval":1000,        # Steps between full-state checkpoints
        "sample_interval":10,       # Steps between observable samples
        "statistics_save_interval":10000,    # Steps between statistics saves
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # Maximum the number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "compressor":{
        # "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        "name":"lram",              # Linear Response + Anderson Mixing
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345
np.random.seed(random_seed)

# Set initial fields
print("w_A and w_B are initialized to random Gaussian.")
w_A = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

# Initialize simulation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/params["langevin"]["max_step"]) )

# Recording first a few iteration results for debugging and refactoring
# (LRAM)
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#        6    3.719E-16  [ 1.0217331E+00  1.0004506E+00  ]     6.255662491   [3.7572438E-06 ]
# Langevin step:  1
#        4    2.634E-16  [ 1.1291045E+00  1.0007397E+00  ]     6.280765027   [6.3142449E-05 ]
# Langevin step:  2
#        6    1.691E-16  [ 1.6110832E+00  1.0102085E+00  ]     6.359458406   [6.4831667E-05 ]
# Langevin step:  3
#        7    1.307E-16  [ 2.1515404E+00  1.0240414E+00  ]     6.383830869   [5.0818546E-05 ]
# Langevin step:  4
#        7    9.496E-17  [ 2.7677549E+00  1.0393897E+00  ]     6.382316522   [7.0096484E-05 ]
# Langevin step:  5
#        7    6.836E-17  [ 3.4835118E+00  1.0555256E+00  ]     6.368531573   [6.2089066E-05 ]
