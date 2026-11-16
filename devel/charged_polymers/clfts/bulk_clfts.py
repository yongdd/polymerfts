"""Bulk CL-FTS of a 3:1 salt of smeared point ions in the units of brush_clfts.py (N = 100, rho0 = 1 nm^-3,
R0 = 10 nm, l_B = 0.7 nm, 0.5 nm grid, periodic 10 nm box), one configuration per run, long enough to check
convergence by intervals: the excess chemical potential mu_ex = Re(ln(N c) - ln Q) - ln(N c) = -Re ln Q of each ion
(docs/theory/ComplexLangevinSimulation.md, Section 7), recorded per 1e4 steps, against the one-loop theory on the
same grid.

usage: python bulk_clfts.py out.json valence molar cation_sigma_nm anion_sigma_nm [steps]"""
import glob
import json
import os
import sys

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.io import loadmat

from polymerfts import CLFTS

out_path = sys.argv[1]
valence, molar, s_cat, s_an = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 100000
N, lb, r0, n, dt = 100, 0.7, 10.0, 20, 0.001
interval = 10000
coupling = 4 * np.pi * lb * N * r0 ** 2
C = r0 ** 3 / N
a, a_x = s_cat / r0, s_an / r0
c = molar * 0.6022
d = os.path.splitext(out_path)[0] + "_data"
params = {
    "platform": "cpu-fftw", "nx": [n] * 3, "lx": [1.0] * 3, "chain_model": "discrete", "ds": 1.0 / N,
    "segment_lengths": {"S": 1.0}, "chi_n": {}, "zeta_n": 0.0,
    "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "S", "length": 1.0 / N}]}],
    "smearing": {"type": "gaussian", "a_int": a},
    "dynamic_stabilization": 0.05,
    "electrostatics": {"charges": {"S": 0.0}, "coupling": coupling, "n_segments": N, "psi_dt_scaling": 1000.0,
                       "ions": {"Y": {"valence": float(valence), "concentration": c, "smearing_length": a},
                                "X": {"valence": -1.0, "concentration": valence * c, "smearing_length": a_x}}},
    "langevin": {"max_step": steps, "dt": dt, "nbar": C ** 2},
    "recording": {"dir": d, "checkpoint_interval": steps, "sample_interval": 2, "statistics_save_interval": interval},
    "verbose_level": 0,
}
# One-loop excess chemical potential of each ion on the same grid
m = [np.fft.fftfreq(n, 1.0 / n)] * 3
k2 = sum(np.meshgrid(*[(2 * np.pi * mm) ** 2 for mm in m], indexing="ij"))
g_cat, g_an = np.exp(-a * a * k2), np.exp(-a_x * a_x * k2)
op = k2 / coupling + (valence ** 2 * c * g_cat + valence * c * g_an) / N
mask = k2 > 0
one_loop = [valence ** 2 * np.sum(g_cat[mask] / op[mask]) / (2 * N ** 2 * C), np.sum(g_an[mask] / op[mask]) / (2 * N ** 2 * C)]
row = dict(valence=valence, molar=molar, cation_sigma_nm=s_cat, anion_sigma_nm=s_an, cation_aW_nm=s_cat * np.sqrt(np.pi),
           steps=steps, dt=dt, one_loop=one_loop)
print(json.dumps(row), flush=True)
try:
    CLFTS(params=params, random_seed=7).run(initial_fields={"S": np.zeros(n ** 3)})
    row["status"] = "completed"
except Exception as exc:
    row["status"] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
files = sorted(glob.glob(os.path.join(d, "dH_*.mat")), key=lambda f: int(f.split("_")[-1].split(".")[0]))
intervals = []
for f in files:
    h = loadmat(f)
    mu = h["chemical_potential_history_real"][:, 1:3]
    intervals.append((mu - np.log(N * np.array([c, valence * c]))).mean(axis=0).tolist())
row["mu_ex_per_interval"] = intervals      # -Re ln Q of the cation and the anion, one row per 1e4 steps
row["interval_steps"] = interval
print(json.dumps(row), flush=True)
json.dump(row, open(out_path, "w"), indent=1)
print("RUN_DONE")
