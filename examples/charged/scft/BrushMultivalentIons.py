"""SCFT of a polyelectrolyte brush with multivalent ions and ion correlations, in two solutions.

The script solves the same equations twice, and only the solution around the brush changes.

1. The condition of Fig. 2 of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101 (2025): monovalent
   counterions conserved in a box of length L and z:1 salt from a reservoir, for z = 1, 2, 3. The paper
   does not state L. One value, L = 6.6 um, brings the calculation to all 24 readings of the figure.
   This part checks the implementation.
2. The solution of the neutron reflectivity titration of Yu et al., Macromolecules 49, 5609 (2016):
   Na+, Y3+ and NO3- at a fixed ionic strength of 6 mM, without and with the self-energy. The added
   Y3+ is then shared between the brush and a finite depth of solution, as in a closed cell.

The self-energy is not part of the polymerfts SCFT class, so this script writes the iteration itself:
PropagatorSolver gives the chain, and the Poisson-Boltzmann equation, the self-energy and the Newton
iteration are here. Lengths are in nm and energies in kT. How the two parts differ, and why, is in
docs/validation/MultivalentBrush.md. The two parts take about half an hour on one CPU core.
"""

import os
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import NoConvergence, brentq, newton_krylov
from scipy.special import erfcx

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import PropagatorSolver

n_segments = 100        # Segments N of a chain
segment = 1.0           # Statistical segment length b, nm
volume = 1.0            # Segment volume v, nm^3
grafting = 0.1          # Grafted chains per nm^2
chi = 0.5               # Segment-solvent interaction
bjerrum = 0.7           # Bjerrum length, nm
born = 0.25             # Ion radius of the self-energy, nm
molar = 0.6022          # nm^-3 per mol/L

length, nz = 200.0, 800     # Grid, nm
graft_width = 0.5           # Width of the graft layer at the wall, nm
ds = 0.001                  # Contour step. The field reaches 4 kT per segment, so ds = 1/N is far too coarse
floor = 6.0                 # Lowest field below its largest value, kT per segment

# Part 1. L is not in the paper: it is fitted to its Fig. 2
box_length = 6600.0     # Length that holds the counterions, nm
# Fig. 2 of the paper: rho_b in mol/L and log10(h/h0), read from the image to about +/-0.01
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

dz = length / nz
z = (np.arange(nz) + 0.5) * dz
weight = np.full(nz, dz)        # Beyond the grid the box holds only conserved ions, at the last potential
weight[-1] += box_length - length

solver = PropagatorSolver(nx=[nz], lx=[length / (segment * np.sqrt(n_segments))], ds=ds, bond_lengths={"A": 1.0},
                          bc=["absorbing", "reflecting"], chain_model="continuous", numerical_method="cn-adi2")
solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
q_graft = np.exp(-0.5 * (z / graft_width) ** 2)


def fig2_solution(valence, rho_molar):
    """Part 1: conserved monovalent C, z:1 salt from a reservoir, no field at the far end."""
    rho = rho_molar * molar
    return dict(valence=valence, conserved=grafting * n_segments, sodium=0.0, cation=rho, anion=valence * rho, far=1.0)


def experiment_solution(y_molar):
    """Part 2: Na+, Y3+ and NO3- at the fixed ionic strength, and psi = 0 in the solution at the far end."""
    y = y_molar * molar
    sodium = (ionic_strength - 6 * y_molar) * molar
    return dict(valence=3, conserved=0.0, sodium=sodium, cation=y, anion=sodium + 3 * y, far=-1.0)


def segment_density(potential):
    """Segments per nm^3 in the field per segment, normalized to grafting * n_segments per nm^2. Double
    precision holds about 700 kT over a chain, so the field keeps a floor where no chain reaches, far
    outside the brush, and a constant comes off it."""
    field = np.maximum(potential, potential.max() - floor) - (potential.max() - floor)
    solver.compute_propagators({"A": n_segments * field}, q_init={"G": q_graft})
    solver.compute_concentrations()
    phi = solver.get_concentration("A")
    return phi * grafting * n_segments / (phi.sum() * dz)


def born_energy(kappa):
    """Self-energy per z^2 of a Gaussian charge of radius born at inverse screening length kappa."""
    x = born * kappa
    return bjerrum / (2 * born) * (1 - x * erfcx(x / np.sqrt(np.pi)))


def ion_densities(psi, u, s):
    """Conserved ions, Na+, multivalent cations and anions at potential psi and self-energy u per z^2."""
    boltzmann = np.exp(-np.clip(psi + u, -300, 300))
    z_plus = s["valence"]
    return (s["conserved"] * boltzmann / np.sum(weight * boltzmann), s["sodium"] * boltzmann,
            s["cation"] * np.exp(-np.clip(z_plus * psi + z_plus ** 2 * u, -300, 300)),
            s["anion"] * np.exp(np.clip(psi - u, -300, 300)))


def poisson_boltzmann(rho, u, s, psi):
    """Newton iteration on Poisson-Boltzmann, with no field at the wall. Conserved ions couple all points
    through their amount, one dense row that the last line of the step takes out (Sherman-Morrison)."""
    alpha = 4 * np.pi * bjerrum
    z_plus = s["valence"]

    def error(p):
        c, sodium, cation, anion = ion_densities(p, u, s)
        laplacian = np.diff(np.r_[p[0], p, s["far"] * p[-1]], 2) / dz ** 2
        return laplacian + alpha * (c + sodium + z_plus * cation - anion - rho), c, sodium, cation, anion

    for _ in range(300):
        f, c, sodium, cation, anion = error(psi)
        if abs(f).max() < 1e-8:
            return psi, c, sodium, cation, anion
        bands = np.zeros((3, nz))
        bands[0, 1:] = bands[2, :-1] = 1 / dz ** 2
        bands[1] = -2 / dz ** 2 - alpha * (c + sodium + z_plus ** 2 * cation + anion)
        bands[1, 0] += 1 / dz ** 2
        bands[1, -1] += s["far"] / dz ** 2
        solved = solve_banded((1, 1), bands, np.column_stack([-f, alpha * c]))
        share = c * weight / max(s["conserved"], 1e-300)
        step = solved[:, 0] - solved[:, 1] * (share @ solved[:, 0]) / (1 + share @ solved[:, 1])
        t = 1.0
        while not np.linalg.norm(error(psi + t * step)[0]) < np.linalg.norm(f) and t > 1e-6:
            t /= 2
        psi = psi + t * step
    raise RuntimeError("Poisson-Boltzmann did not converge")


def solve(s, correlations, unknowns, psi, relax=0):
    """Self-consistent state in solution s; unknowns = field per segment and self-energy per z^2, at every
    grid point. relax damped steps come first, for a start far from the solution."""
    z_plus = s["valence"]
    kappa_bulk = np.sqrt(4 * np.pi * bjerrum * (s["sodium"] + z_plus ** 2 * s["cation"] + s["anion"]))
    latest = {"psi": psi}

    def residual(x):
        potential, u = x[:nz], x[nz:]
        rho = segment_density(potential)
        psi, c, sodium, cation, anion = poisson_boltzmann(rho, u, s, latest["psi"])
        latest.update(psi=psi, rho=rho, cation=cation)
        phi = np.clip(volume * rho, 0, 0.999)
        target = chi * (1 - 2 * phi) - np.log1p(-phi) - psi + u
        if not correlations:
            return np.r_[target - potential, -u]
        # The ionic strength, polymer included, smoothed over the ion radius
        ionic = gaussian_filter1d(0.5 * (rho + c + sodium + z_plus ** 2 * cation + anion), born / dz, mode="reflect")
        kappa = np.sqrt(8 * np.pi * bjerrum * ionic)
        return np.r_[target - potential, born_energy(kappa) - born_energy(kappa_bulk) - u]

    for _ in range(relax):
        unknowns = unknowns + 0.03 * residual(unknowns)
    x = newton_krylov(residual, unknowns, f_tol=1e-7, rdiff=1e-5, maxiter=80, inner_maxiter=50, line_search="armijo")
    residual(x)
    rho = latest["rho"]
    uptake = z_plus * np.sum(latest["cation"] - latest["cation"][-1]) * dz / (grafting * n_segments)
    return x, latest["psi"], 2 * np.sum(z * rho) / np.sum(rho), uptake


def walk(solution, values, correlations, x, psi):
    """Height and uptake along values, each state from the last. The onset of the dense layer is steep,
    so a step that fails is tried again from halfway on a log scale."""
    pending, done, states = list(values), [], {}
    while pending:
        value = pending[0]
        try:
            x, psi, height, uptake = solve(solution(value), correlations, x, psi)
        except (NoConvergence, RuntimeError):
            if not done or value < 1.001 * done[-1]:
                raise
            pending.insert(0, np.sqrt(done[-1] * value))
            continue
        pending.pop(0)
        done.append(value)
        states[value] = (height, uptake)
    return states


def start(s, correlations):
    """A first state from a parabolic brush."""
    parabola = 3 * np.pi ** 2 / (8 * n_segments ** 2 * segment ** 2) * np.maximum(70.0 ** 2 - z ** 2, 0)
    guess = np.r_[0.5 + parabola, np.zeros(nz)]
    return solve(s, correlations, guess, np.zeros(nz), relax=600)


print("Part 1: conserved counterions in a box of L = {:.1f} um, against Fig. 2 of the paper".format(box_length / 1000))
x0, psi0, h0, _ = start(fig2_solution(1, 1e-10), True)  # z = 1 without salt: the one reference of all valences
print(f"  Salt-free brush: h0 = {h0:.3f} nm")
print("   z   rho_b [M]   Fig. 2    SCFT   difference")
differences = []
for valence, readings in fig2.items():
    states = walk(lambda rho: fig2_solution(valence, rho), [1e-10] + [r for r, _ in readings], True, x0, psi0)
    for rho, measured in readings:
        value = np.log10(states[rho][0] / h0)
        differences.append(value - measured)
        print(f"   {valence}  {rho:9.3g}  {measured:7.3f}  {value:7.3f}  {value - measured:7.3f}", flush=True)
rms = np.sqrt(np.mean(np.square(differences)))
print(f"  rms {rms:.4f}, largest {np.max(np.abs(differences)):.4f}, {len(differences)} readings")


def titration(correlations):
    """Y3+ in solution, log10(h/h6) and uptake along y_series, h6 being the height with hardly any Y3+."""
    x, psi, h6, _ = start(experiment_solution(y_series[0]), correlations)
    states = walk(experiment_solution, y_series, correlations, x, psi)
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
