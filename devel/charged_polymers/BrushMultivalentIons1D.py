"""SCFT of a polyelectrolyte brush with multivalent ions and ion correlations, in two solutions.

The script solves the same equations twice, and only the solution around the brush changes.

1. The condition of Fig. 2 of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101 (2025): monovalent
   counterions conserved in a box of length L and z:1 salt from a reservoir, for z = 1, 2, 3. The paper
   does not state L. One value, L = 6.6 um, brings the calculation to all 24 readings of the figure.
   This part checks the implementation.
2. The solution of the neutron reflectivity titration of Yu et al., Macromolecules 49, 5609 (2016):
   Na+, Y3+ and NO3- at a fixed ionic strength of 6 mM, without and with the self-energy. The added
   Y3+ is then shared between the brush and a finite depth of solution, as in a closed cell.

The self-energy is the local form of multivalent_brush.py, and each state relaxes by Anderson mixing from
the last one. How the two parts differ from the paper, and why, is in reports/MultivalentBrush.md.
The two parts take about four minutes on one CPU core.
"""

import os
import numpy as np
from scipy.optimize import brentq

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from multivalent_brush import Brush, box_length, molar, n_segments, segment

grafting = 0.1              # Grafted chains per nm^2
length, nz = 200.0, 800     # Grid, nm
ds = 0.001                  # Contour step. The field reaches 4 kT per segment, so ds = 1/N is far too coarse

mixing = 0.03               # Fraction of the residual in a step
history = 20                # Past steps of the Anderson mixing
largest_step = 0.5          # Largest change of a field in a step, kT
tolerance = 1e-7            # Largest residual
max_steps = 1500

# Part 1. Fig. 2 of the paper: rho_b in mol/L and log10(h/h0), read from the image to about +/-0.01
fig2 = {
    1: [(1e-06, 0.001), (1e-05, -0.001), (0.0001, -0.002), (0.001, -0.005),
        (0.01, -0.023), (0.1, -0.118), (1, -0.298), (10, -0.375)],
    2: [(1e-06, -0.013), (1e-05, -0.042), (0.0001, -0.164), (0.001, -0.379),
        (0.01, -0.446), (0.1, -0.451), (1, -0.392), (10, -0.378)],
    3: [(6.31e-07, -0.072), (1e-05, -0.276), (0.0001, -0.522), (0.001, -0.584),
        (0.01, -0.584), (0.1, -0.528), (1, -0.417), (10, -0.381)],
}

# Part 2. Y3+ in solution, mol/L; each state starts from the previous one
ionic_strength = 0.006  # sum of z^2 c / 2 of the solution, mol/L
y_series = [1e-14, 1e-10, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 7e-9, 1e-8, 1.5e-8, 2e-8, 3e-8, 5e-8, 1e-7, 3e-7, 1e-6,
            3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]
# Fig. 7 of Yu et al.: Y3+ added, mol/L, and log10(h/h6), read by eye to about +/-0.02
experiment = [(1e-6, -0.09), (1e-5, -0.145), (1e-4, -0.495), (5e-4, -0.545)]
brush_charge = 73.0     # Charges per nm^2 of the brush of the experiment
depth = 1.2e6           # Solution over the brush, nm. Not measured: fitted to the four symbols
if not experiment[-1][0] <= y_series[-1] <= ionic_strength / 6:
    raise ValueError("y_series must reach the last Y3+ added, and Y3+ alone gives at most ionic_strength / 6")

brush = Brush(grafting, length, nz, ds)
z, dz = brush.z, brush.dz


def fig2_solution(valence, rho_molar):
    """Part 1: conserved monovalent counterions, z:1 salt from a reservoir, no field at the far end."""
    brush.set_solution(valence, rho_molar * molar)


def experiment_solution(y_molar):
    """Part 2: Na+, Y3+ and NO3- at the fixed ionic strength, and the potential of the solution at the far end."""
    brush.set_solution(3, y_molar * molar, sodium=(ionic_strength - 6 * y_molar) * molar, counterions=False, grounded=True)


def solve(x, psi, plain_steps=0):
    """Self-consistent state from x and the potential psi, its height and the cation charge taken up per
    chain charge."""
    brush.psi = psi
    x, left = brush.relax(x, mixing, history, largest_step, tolerance, max_steps, plain_steps, report=0)
    if not left < tolerance:
        raise RuntimeError(f"No solution: residual {left:.1e}")
    cation = brush.state["cation"]
    uptake = brush.solution["valence"] * np.sum(cation - cation[-1]) * dz / (grafting * n_segments)
    return x, brush.psi, brush.height(), uptake


def walk(solution, values, x, psi):
    """Height and uptake along values, each state from the last. The onset of the dense layer is steep,
    so a step that fails is tried again from halfway on a log scale."""
    pending, done, states = list(values), [], {}
    while pending:
        value = pending[0]
        solution(value)
        try:
            x_new, psi_new, height, uptake = solve(x, psi)
        except (RuntimeError, ValueError):
            if not done or value < 1.001 * done[-1]:
                raise
            pending.insert(0, np.sqrt(done[-1] * value))
            continue
        x, psi = x_new, psi_new
        pending.pop(0)
        done.append(value)
        states[value] = (height, uptake)
    return states


def start():
    """A first state from a parabolic brush."""
    parabola = 3 * np.pi ** 2 / (8 * n_segments ** 2 * segment ** 2) * np.maximum(70.0 ** 2 - z ** 2, 0)
    return solve(brush.pack(0.5 + parabola, 0.0), np.zeros(nz), plain_steps=600)


print("Part 1: conserved counterions in a box of L = {:.1f} um, against Fig. 2 of the paper".format(box_length / 1000))
fig2_solution(1, 1e-10)     # z = 1 without salt: the one reference of all valences
x0, psi0, h0, _ = start()
print(f"  Salt-free brush: h0 = {h0:.3f} nm")
print("   z   rho_b [M]   Fig. 2    SCFT   difference")
differences = []
for valence, readings in fig2.items():
    states = walk(lambda rho: fig2_solution(valence, rho), [1e-10] + [r for r, _ in readings], x0, psi0)
    for rho, measured in readings:
        value = np.log10(states[rho][0] / h0)
        differences.append(value - measured)
        print(f"   {valence}  {rho:9.3g}  {measured:7.3f}  {value:7.3f}  {value - measured:7.3f}", flush=True)
rms = np.sqrt(np.mean(np.square(differences)))
print(f"  rms {rms:.4f}, largest {np.max(np.abs(differences)):.4f}, {len(differences)} readings")


def titration(correlations):
    """Y3+ in solution, log10(h/h6) and uptake along y_series, h6 being the height with hardly any Y3+."""
    brush.correlations = correlations
    experiment_solution(y_series[0])
    x, psi, h6, _ = start()
    states = walk(experiment_solution, y_series, x, psi)
    y = np.array(sorted(states))
    return y, np.log10([states[q][0] / h6 for q in y]), np.array([states[q][1] for q in y])


def closed_cell(added, y, uptake):
    """Y3+ left in solution, mol/L, when added mol/L is shared between the solution and the brush."""
    def balance(log_y):
        left = np.exp(log_y)
        return left + np.interp(log_y, np.log(y), uptake) * brush_charge / 3 / (depth * molar) - added
    return np.exp(brentq(balance, np.log(y[0]), np.log(added)))


print()
print("Part 2: Na+, Y3+ and NO3- at ionic strength 6 mM, log10(h/h6)")
without = titration(False)
with_u = titration(True)
print("  Y3+ in solution [M]   without the self-energy   with   uptake without   with")
for q in y_series:
    heights = [np.interp(np.log(q), np.log(y), h) for y, h, _ in (without, with_u)]
    uptakes = [np.interp(np.log(q), np.log(y), f) for y, _, f in (without, with_u)]
    print(f"  {q:9.1e}  {heights[0]:22.3f}  {heights[1]:16.3f}  {uptakes[0]:12.3f}  {uptakes[1]:7.3f}")

print()
print(f"  Closed cell {depth / 1e6:.1f} mm deep: added [M], experiment, without, with, Y3+ left [M]")
for added, measured in experiment:
    heights = [np.interp(np.log(closed_cell(added, y, f)), np.log(y), h) for y, h, f in (without, with_u)]
    left = closed_cell(added, with_u[0], with_u[2])
    print(f"  {added:9.1e}  {measured:10.3f}  {heights[0]:8.3f}  {heights[1]:8.3f}  {left:10.2e}")
