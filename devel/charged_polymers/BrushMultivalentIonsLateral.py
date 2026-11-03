"""SCFT of a brush with trivalent ions in 2D or 3D: a planar layer against dense caps on the wall.

Fig. 4(a,b) of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101 (2025) shows a brush of 0.03 chains per
nm^2 in 1 mM trivalent salt that forms pinned micelles on a hexagon, about 24 nm apart. The planar layer
is stable against every small lateral wave, so the caps do not grow from noise: this script starts them
from a dense seed, lets the fields relax and compares the grand potential with the planar layer.

The grafts are fixed: every lateral column starts with the same number of chains, Eq. S30 of the paper
at each lateral position. dimensions = 3 places two caps in the rectangular cell of a hexagon of side
spacing, and dimensions = 2 makes ridges of period spacing.

The equations are those of BrushMultivalentIons1D.py, with the local self-energy of multivalent_brush.py,
and the Green function of the varying ionic strength is left out in the same way. How this compares with
the paper is in reports/MultivalentBrush.md. Lengths are in nm and energies in kT.

The Poisson-Boltzmann equation and the mixing of the pattern use CuPy when it is installed and a GPU is
present, and NumPy otherwise. The 3D run stores 2 * 250 propagators of 48 * 83 * 320 points, 5 GB.
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from multivalent_brush import Brush, array_module, n_segments, molar, to_numpy, volume

xp = array_module()
print(f"Poisson-Boltzmann and mixing of the pattern with {xp.__name__}")

dimensions = 3          # 3: caps on a hexagon, 2: ridges
spacing = 24.0          # Distance between the caps, nm
points_per_nm = 2.0     # Lateral grid

grafting = 0.03         # Grafted chains per nm^2
valence = 3             # Of the cation of the salt
salt = 1e-3 * molar     # Salt in the reservoir, nm^-3

length, nz = 80.0, 320  # Grid normal to the wall, nm
ds = 0.004              # Contour step. The grand potential per chain moves by 0.002 kT from ds = 0.001

mixing = 0.012          # Fraction of the residual in a step: the damped iteration is unstable from about 0.02
history = 20            # Past steps of the Anderson mixing
largest_step = 0.2      # Largest change of a field in a step, kT
tolerance = 1e-5        # Largest residual
max_steps = 4000

# The planar layer
planar = Brush(grafting, length, nz, ds)
planar.set_solution(valence, salt)
z = planar.z
thickness = 1.0 + grafting * n_segments * volume / 0.72    # The chains leave about 1 nm empty at the wall
inside = 0.5 * (1 - np.tanh((z - thickness) / 0.5)) * 0.5 * (1 + np.tanh((z - 0.7) / 0.3))
print("Planar layer")
x_flat, residual_flat = planar.relax(planar.pack(0.44 - 0.85 * inside, -0.85 * inside), mixing, history, largest_step,
                                     tolerance, max_steps, label="planar ")
omega_flat = planar.grand_potential(x_flat)
field_flat, u_flat = planar.unpack(x_flat)
rho_flat = planar.state["rho"]
print(f"  Height {planar.height():.3f} nm, largest phi {rho_flat.max() * volume:.3f}, residual {residual_flat:.1e}")

# Dense caps, as much of the segments as the layer holds in half-cylinders or half-spheres
if dimensions == 3:
    cells, size = [int(round(spacing * points_per_nm)), int(round(np.sqrt(3) * spacing * points_per_nm))], [spacing, np.sqrt(3) * spacing]
    sites = [(0.0, 0.0), (size[0] / 2, size[1] / 2)]
    radius = (3 * grafting * n_segments * volume * size[0] * size[1] / 2 / (2 * np.pi * 0.72)) ** (1 / 3)
else:
    cells, size = [int(round(spacing * points_per_nm))], [spacing]
    sites = [(size[0] / 2,)]
    radius = np.sqrt(2 * grafting * n_segments * volume * size[0] / (np.pi * 0.72))
brush = Brush(grafting, length, nz, ds, cells, size, xp=xp)
brush.set_solution(valence, salt)
points = np.meshgrid(*[(np.arange(n) + 0.5) * d for n, d in zip(cells, brush.dx)], indexing="ij")
nearest = np.full(cells, np.inf)
for site in sites:
    nearest = np.minimum(nearest, sum((np.mod(q - c + s / 2, s) - s / 2) ** 2 for q, c, s in zip(points, site, size)))
inside = 0.5 * (1 - np.tanh((np.sqrt(nearest[..., None] + z ** 2) - radius) / 0.5))
peak = np.argmax(rho_flat)
print(f"Caps {spacing} nm apart in {dimensions}D, grid {brush.shape}, radius of the seed {radius:.2f} nm")
seed = brush.pack(field_flat[-1] + (field_flat[peak] - field_flat[-1]) * inside, u_flat[peak] * inside)
x_caps, residual_caps = brush.relax(seed, mixing, history, largest_step, tolerance, max_steps, label="caps ")
omega_caps = brush.grand_potential(x_caps)
areal = to_numpy(brush.state["rho"]).sum(axis=-1) * brush.dz
print(f"  Lateral contrast {areal.std() / areal.mean():.3f}, residual {residual_caps:.1e}")
print(f"Grand potential of the caps against the planar layer: {(omega_caps - omega_flat) / grafting:.3f} kT per chain")
