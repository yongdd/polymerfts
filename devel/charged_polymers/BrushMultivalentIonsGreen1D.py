"""SCFT of a planar brush with the Green function solved in the varying ionic strength: Fig. 4(c,d).

BrushMultivalentIons1D.py takes the self-energy of an ion from the ionic strength around it. Here the
self-energy comes from the Green function of Eq. S25 of Duan, Agrawal and Wang, Phys. Rev. Lett. 134,
048101 (2025), without its nonlocal term I_ex: GreenFunction in multivalent_brush.py. A smaller cation
radius a_+ makes the uniform layer unstable at a period of 1 nm, which the local form of the self-energy
cannot select.

The script relaxes the brush of Fig. 4(c,d) at the two radii of the figure: 0.1 chains per nm^2 in 1 mM
trivalent salt. The oscillatory state grows from a small wave, so a plain damped iteration comes before
the Anderson mixing, which would also converge to the unstable flat layer. The oscillatory state settles
slowly and stops near a residual of 1e-3. Lengths are in nm and energies in kT.
reports/MultivalentBrush.md compares the result with the paper. The run takes about five minutes
on one CPU core.
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from multivalent_brush import Brush, born_energy, molar, n_segments, screening, volume

grafting = 0.1          # Grafted chains per nm^2
radii_plus = [0.25, 0.15]   # Cation radii of Fig. 4(c) and 4(d), nm
valence = 3             # Of the cation of the salt
salt = 1e-3 * molar     # Salt in the reservoir, nm^-3

length, nz = 40.0, 400  # Grid, nm
ds = 0.001              # Contour step

mixing = 0.008          # Fraction of the residual in a step
plain_steps = 6000      # Damped steps, in which the wave grows
history = 20            # Past steps of the Anderson mixing
anderson_steps = 3000
tolerance = 1e-6        # Largest residual

z = (np.arange(nz) + 0.5) * length / nz
thickness = grafting * n_segments * volume / 0.75
inside = 0.5 * (1 - np.tanh((z - thickness) / 0.5)) * 0.5 * (1 + np.tanh((z - 0.7) / 0.3))
for a_plus in radii_plus:
    print(f"Cation radius {a_plus * 10:.1f} A")
    brush = Brush(grafting, length, nz, ds, green_function=True, cation_radius=a_plus)
    brush.set_solution(valence, salt)
    # A check of the Green function in a uniform medium, then a layer of the right thickness with a small wave in it
    trial, bulk = 1.6, 0.5 * (valence ** 2 + valence) * salt
    for a, u in zip(brush.radii, brush.self_energy(np.full(nz, trial))):
        closed = born_energy(a, screening(trial)) - born_energy(a, screening(bulk))
        print(f"  Uniform I = {trial}, a = {a}: Green function {u[nz // 2]:.4f}, closed form {closed:.4f}")
    start = brush.pack(0.37 - 0.8 * inside + 1e-3 * np.sin(2 * np.pi * z) * inside, *[-0.85 * inside] * len(brush.radii))
    x, left = brush.relax(start, mixing, history, largest_step=0.1, tolerance=tolerance, max_steps=anderson_steps,
                          plain_steps=plain_steps, report=1000)
    phi = brush.state["rho"] * volume
    layer = (z > 2) & (z < 11)
    wave_part = phi[layer] - np.polyval(np.polyfit(z[layer], phi[layer], 3), z[layer])
    spectrum = np.abs(np.fft.rfft(wave_part * np.hanning(len(wave_part)), 16 * len(wave_part)))
    period = 1 / np.fft.rfftfreq(16 * len(wave_part), brush.dz)[1:][np.argmax(spectrum[1:])]
    print(f"  phi {phi[layer].min():.3f} to {phi[layer].max():.3f} in the layer, ripple {wave_part.std() * np.sqrt(2):.4f} at period {period:.2f} nm, "
          f"height {brush.height():.3f} nm, residual {left:.1e}")
    np.savetxt(f"brush_green_{a_plus * 10:.1f}.txt", np.column_stack([z, phi]), header="z_nm phi")
