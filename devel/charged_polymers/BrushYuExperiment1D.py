"""The standard model of standard_brush.py against the neutron reflectivity of Yu et al., Macromolecules 49,
5609 (2016): a polystyrene sulfonate brush in solutions of NaNO3 and Y(NO3)3, heights read from their Fig. 7.

1. Y3+ titration at a fixed ionic strength of 6 mM, relative to the height in 6 mM NaNO3. The trough holds a
   finite pool of solution, and at 1 and 10 uM the brush can take up far more Y3+ than the pool holds, so what
   is added is shared between the two: a balance on the calculated uptake, with the depth of the pool as the
   one fitted number. The same titration without the self-energy shows that the collapse needs it.
2. Y(NO3)3 alone at 1.6, 30 and 160 mM, the symbols of ref. 33 of Yu et al., relative to 6 mM NaNO3.
3. NaNO3 up to 5 M, relative to D2O. The model needs a salt in the pool, so D2O is taken as NaNO3 with the
   charge of the brush spread over the pool, 0.09 mM. Yu et al. measure the same height up to 2 M and a smaller one above.

Each state relaxes by Anderson mixing from the last one. Why each element of the model is there, and how the
results compare with the experiment, is in reports/StandardBrush.md. The run takes about five minutes on
one CPU core.
"""

import os
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from standard_brush import Brush, molar, n_segments, segment

grafting = 0.1              # Grafted chains per nm^2
length, nz = 200.0, 1600    # Grid, nm: dz = 0.125 nm is within 0.003 of 0.0625 nm in every height
ds = 0.001                  # Contour step. The field reaches 4 kT per segment, so ds = 1/N is far too coarse

mixing = 0.03               # Fraction of the residual in a step
history = 20                # Past steps of the Anderson mixing
largest_step = 0.5          # Largest change of a field in a step, kT
tolerance = 1e-7            # Largest residual
max_steps = 1500

brush_charge = 73.0         # Charges per nm^2 of the brush of the experiment, from its mass of 25 mg/m^2
ionic_strength = 0.006      # sum of z^2 c / 2 of the solutions of part 1, mol/L
# Y3+ or NaNO3 in mol/L and log10 of the relative height, read by eye from Fig. 7 to about +/-0.02
titration = [(1e-6, -0.09), (1e-5, -0.145), (1e-4, -0.495), (5e-4, -0.545)]
yttrium_alone = [(1.6e-3, -0.53), (0.03, -0.53), (0.16, -0.53)]
nitrate = [1e-3, 0.01, 0.1, 1.0, 2.0, 3.0, 5.0]
y_series = [1e-14, 1e-10, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 7e-9, 1e-8, 1.5e-8, 2e-8, 3e-8, 5e-8, 1e-7, 3e-7, 1e-6,
            3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]
if not titration[-1][0] <= y_series[-1] <= ionic_strength / 6:
    raise ValueError("y_series must reach the last Y3+ added, and Y3+ alone gives at most ionic_strength / 6")


def at_6mM(y_molar):
    return 3, y_molar * molar, (ionic_strength - 6 * y_molar) * molar


def solve(brush, x, plain_steps=0):
    """Self-consistent state from x, its height and uptake."""
    x, left = brush.relax(x, mixing, history, largest_step, tolerance, max_steps, plain_steps)
    if not left < tolerance:
        raise RuntimeError(f"No solution: residual {left:.1e}")
    return x, brush.height(), brush.uptake()


def walk(brush, solution, values, x):
    """Height and uptake along values, each state from the last. The onset of the dense layer is steep, so a
    step that fails is tried again from halfway on a log scale."""
    pending, done, states = list(values), [], {}
    while pending:
        value = pending[0]
        brush.set_solution(*solution(value))
        psi = brush.psi
        try:
            x_new, height, uptake = solve(brush, x)
        except (RuntimeError, ValueError):
            if not done or value < 1.001 * done[-1]:
                raise
            brush.psi = psi
            pending.insert(0, np.sqrt(done[-1] * value))
            continue
        x = x_new
        pending.pop(0)
        done.append(value)
        states[value] = (height, uptake)
    return states, x


def start(brush, solution):
    """A first state from a parabolic brush."""
    brush.set_solution(*solution)
    brush.psi = np.zeros(brush.nz)
    parabola = 3 * np.pi ** 2 / (8 * n_segments ** 2 * segment ** 2) * np.maximum(70.0 ** 2 - brush.z ** 2, 0)
    return solve(brush, brush.pack(0.5 + parabola, 0.0), plain_steps=600)


def pooled(added, y, uptake, depth):
    """Y3+ left in solution, mol/L, when added mol/L is shared between a pool of depth nm and the brush."""
    def balance(log_y):
        return np.exp(log_y) + np.interp(log_y, np.log(y), uptake) * brush_charge / 3 / (depth * molar) - added
    return np.exp(brentq(balance, np.log(y[0]), np.log(added)))


def misfit(curve, depth=None):
    """rms against the titration, in a pool of depth nm, or in an unlimited solution without depth."""
    y, h, uptake = curve
    left = [q if depth is None else pooled(q, y, uptake, depth) for q, _ in titration]
    return np.sqrt(np.mean([(np.interp(np.log(q), np.log(y), h) - m) ** 2 for q, (_, m) in zip(left, titration)]))


brush = Brush(grafting, length, nz, ds)
print("Part 1: Y3+ titration at ionic strength 6 mM, log10(h/h6)")
curves = {}
for correlations in (True, False):
    brush.correlations = correlations
    x, h6, _ = start(brush, at_6mM(y_series[0]))
    states, x_end = walk(brush, at_6mM, y_series, x)
    y = np.array(sorted(states))
    curves[correlations] = (y, np.log10([states[q][0] / h6 for q in y]), np.array([states[q][1] for q in y]))
    if correlations:
        reference, collapsed = h6, x_end
print(f"  h6 = {reference:.3f} nm with the self-energy")
print("  Y3+ in solution [M]   with the self-energy   without   uptake with   without")
for q in y_series:
    heights = [np.interp(np.log(q), np.log(c[0]), c[1]) for c in curves.values()]
    uptakes = [np.interp(np.log(q), np.log(c[0]), c[2]) for c in curves.values()]
    print(f"  {q:9.1e}  {heights[0]:20.3f}  {heights[1]:9.3f}  {uptakes[0]:11.3f}  {uptakes[1]:9.3f}")

print()
log_depths = np.linspace(np.log(1e4), np.log(1e8), 401)       # 10 um to 10 cm
fits = {}
for correlations, curve in curves.items():
    scan = np.array([misfit(curve, np.exp(d)) for d in log_depths])
    i = int(np.argmin(scan))
    best = minimize_scalar(lambda d: misfit(curve, np.exp(d)), method="bounded",
                           bounds=(log_depths[max(i - 1, 0)], log_depths[min(i + 1, len(log_depths) - 1)]))
    inside = np.exp(log_depths[scan < best.fun + 0.01])
    fits[correlations] = np.exp(best.x)
    print(f"  {'With' if correlations else 'Without'} the self-energy: rms {misfit(curve):.3f} in an unlimited solution, "
          f"{best.fun:.3f} in a pool {np.exp(best.x) / 1e6:.2f} mm deep ({inside.min() / 1e6:.2f}-{inside.max() / 1e6:.2f} mm within 0.01)")
depth = fits[True]
print(f"  Pool {depth / 1e6:.2f} mm deep: added [M], experiment, with, without, Y3+ left with [M]")
for added, measured in titration:
    heights = [np.interp(np.log(pooled(added, c[0], c[2], depth)), np.log(c[0]), c[1]) for c in curves.values()]
    print(f"  {added:9.1e}  {measured:10.3f}  {heights[0]:7.3f}  {heights[1]:7.3f}  {pooled(added, *curves[True][::2], depth):10.2e}")

print()
print("Part 2: Y(NO3)3 alone, log10(h/h6), against ref. 33 of Yu et al.")
brush.correlations = True
states, _ = walk(brush, lambda y: (3, y * molar, 0.0), [q for q, _ in yttrium_alone], collapsed)
for q, measured in yttrium_alone:
    print(f"  {q:9.1e} M  experiment {measured:7.3f}  model {np.log10(states[q][0] / reference):7.3f}")

print()
d2o = brush_charge / (depth * molar)
print(f"Part 3: NaNO3, log10(h/h_D2O), with D2O as {d2o * 1e3:.2f} mM NaNO3, the charge of the brush spread over the pool")
x, h_d2o, _ = start(brush, (3, 0.0, d2o * molar))
states, _ = walk(brush, lambda c: (3, 0.0, c * molar), nitrate, x)
for c in nitrate:
    print(f"  {c:9.1e} M  model {np.log10(states[c][0] / h_d2o):7.3f}")
