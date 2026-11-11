"""EC-SCFT of a polyelectrolyte brush that trivalent counterions collapse.

Electrostatic-correlation-augmented SCFT (EC-SCFT) of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101
(2025) adds to the mean field of a charged brush the self-energy of every charge: the free energy of the
correlation hole that the other charges form around it. A charge of valence z gains z^2 times more of it where
the ionic strength is high, so trivalent ions are drawn into the brush and collapse it further than
Poisson-Boltzmann theory does. docs/theory/ElectrostaticCorrelations.md derives the equations.

The brush is that of the paper: chains of N = 100 segments of b = 1 nm and v = 1 nm^3 grafted at 0.1 nm^-2,
chi = 0.5 with water, l_B = 0.7 nm and one negative charge per segment. The solution is that of the neutron
reflectivity experiment of Yu et al., Macromolecules 49, 5609 (2016): Na+, Y3+ and NO3- at a fixed ionic
strength of 6 mM, in contact with the far end of the grid. The script adds Y3+ step by step and prints the
height h = 2<z> relative to h6, the height in 6 mM NaNO3, and the Y3+ charge taken up by the brush per chain
charge.

- Chains: discrete Gaussian chains, as in the paper: each segment is a bead that carries one charge, joined to
  the next by a Gaussian bond of length b. They are grafted near the wall z = 0, propagated with rs (the bond
  summed on the grid) in a field per segment, and mixed with the solvent as an incompressible lattice gas.
- Ions: Boltzmann densities in the mean potential psi and their self-energy; psi from Poisson-Boltzmann. With
  spread screening each charge, segments included, feels psi averaged over its own Gaussian and is spread in
  the same way in the Poisson equation, as the Gaussian average of the variational theory gives.
- Self-energy: every charge is a Gaussian of Born radius a. In a uniform medium a charge of radius a_J has the
  self-energy per z^2
      u_J = (l_B/pi) int_0^inf dk k^2 h_J(k)^2 / (k^2 + sum_K kappa_K^2 h_K(k)^2),   h_K(k)^2 = exp(-a_K^2 k^2/pi),
  minus its value in the bulk solution, where kappa_K^2 = 8 pi l_B I_K is set by the partial ionic strength
  I_K of the charges of radius a_K. Every charge screens with its own spread, as the stationary point of the
  variational free energy requires. The paper screens with point charges (h_K = 1 in the denominator), the
  limit taken by Wang, Phys. Rev. E 81, 021501 (2010), and takes psi at the centre of each charge;
  screening = "point" selects these published equations.
- Local form: each species takes the uniform result at the partial ionic strengths averaged over a Gaussian of
  standard deviation a along z, and feels its self-energy averaged in the same way. The intrachain part of
  the polymer self-energy is left out, as in the brush calculations of the paper, so each segment is screened
  like a free ion. correlations = False drops the self-energy and gives the mean field (Poisson-Boltzmann, of spread
  charges unless screening = "point").

The Born radii are 1.7725 nm = sqrt(pi) b for the segment charges, 0.25 nm for Na+ and NO3- as in the paper, and
0.66 nm for Y3+, which fits the mean activity coefficients of Y(NO3)3 up to 0.1 mol/kg (Rard and Spedding, J. Chem.
Eng. Data 27, 454 (1982)) with point-charge screening and one radius for Y3+ and NO3-; the paper takes 0.25 nm
for every charge. The self-energy is not part of the polymerfts SCFT class, so the script writes the iteration
itself (damped steps and Anderson mixing) and takes the chains from PropagatorSolver. Lengths are in nm and
energies in kT. The run takes about one minute on one CPU core. docs/validation/BrushMultivalentIons.md compares
the model with Figs. 2 and 4 of the paper and with the experiment.
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import spsolve

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # One chain: nothing to run in parallel
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import PropagatorSolver

n_segments = 100        # Segments N of a chain
segment = 1.0           # Statistical segment length b, nm
volume = 1.0            # Segment volume v, nm^3
grafting = 0.1          # Chains per nm^2
chi = 0.5               # Segment-solvent interaction
bjerrum = 0.7           # Bjerrum length l_B, nm
molar = 0.6022          # nm^-3 per mol/L
radii = [1.7725, 0.25, 0.66]    # Born radius a of the segment charges, of Na+ and NO3-, and of Y3+, nm
screening = "spread"    # "spread": each charge screens with its own spread; "point": point charges, as in the paper
correlations = True     # False: mean field, no self-energy

ionic_strength = 0.006  # Of the solution, mol/L: NaNO3 and Y(NO3)3
yttrium = [1e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]   # Y3+, mol/L

length, nz = 200.0, 1600  # Grid of the brush and the solution next to it, nm
graft_width = 0.5       # Width of the graft layer at the wall, nm
floor = 6.0             # Lowest field below its largest value, kT per segment
mixing, history, largest_step, tolerance, max_steps = 0.03, 20, 0.5, 1e-7, 1500


def self_energy(a, kappa2, screen):
    """Self-energy per z^2 of a Gaussian charge of radius a in a uniform medium in which the charges of radius
    screen[K] give kappa2[K] (arrays of the grid or numbers), by the trapezoid rule in ln k."""
    k = np.geomspace(1e-7 / a, 15.0 / a, 800)
    weight = np.full(k.size, np.log(k[1] / k[0]))
    weight[[0, -1]] *= 0.5
    denominator = k ** 2 + sum(np.asarray(q)[..., None] * np.exp(-(b * k) ** 2 / np.pi) for q, b in zip(kappa2, screen))
    return bjerrum / np.pi * ((k ** 3 * np.exp(-(a * k) ** 2 / np.pi)) / denominator) @ weight


def partial_strengths(rho, sodium, cation, anion):
    """Ionic strengths sum z^2 c / 2 of the segment charges, of the monovalent ions and of Y3+."""
    return [0.5 * rho, 0.5 * (sodium + anion), 4.5 * cation]


class Brush:
    """A planar brush and its solution on nz points over length nm from the wall. The unknowns x of a state are
    the field per segment and the self-energies per z^2 of the three radii."""

    def __init__(self):
        self.dz = length / nz
        self.z = (np.arange(nz) + 0.5) * self.dz
        unit = segment * np.sqrt(n_segments)   # R0, the length unit of PropagatorSolver
        self.solver = PropagatorSolver(nx=[nz], lx=[length / unit], ds=1 / n_segments, bond_lengths={"A": 1.0},
                                       bc=["absorbing", "reflecting"], chain_model="discrete",
                                       numerical_method="rs")
        self.solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
        self.graft = np.exp(-0.5 * (self.z / graft_width) ** 2)
        self.screen = radii if screening == "spread" else [0.0] * len(radii)
        self.blurs = [lambda v, a=a: gaussian_filter(v, sigma=a / self.dz, mode="reflect") for a in radii]
        # The average over a charge of radius a, a Gaussian of standard deviation a / sqrt(pi) along z, as a
        # symmetric band matrix (reflected at both ends); the paper's equations take the centre of the charge
        spread = lambda a: gaussian_filter1d(np.eye(nz), a / np.sqrt(np.pi) / self.dz, axis=0, mode="reflect")
        self.spreads = [csc_matrix(spread(a)) if screening == "spread" else identity(nz, format="csc") for a in radii]
        off = 1 / self.dz ** 2   # d^2/dz^2 with no field at the wall and psi = 0 at the far end
        middle = np.r_[-off, np.full(nz - 2, -2 * off), -3 * off]
        self.laplacian = diags([np.full(nz - 1, off), middle, np.full(nz - 1, off)], [-1, 0, 1], format="csc")
        self.psi = np.zeros(nz)

    def set_solution(self, y):
        """The solution at the far end: y mol/L of Y(NO3)3 and NaNO3 to the ionic strength."""
        if 6 * y > ionic_strength:
            raise ValueError(f"Y(NO3)3 of {y} mol/L exceeds the ionic strength of {ionic_strength} mol/L")
        cation, sodium = y * molar, (ionic_strength - 6 * y) * molar
        self.solution = dict(cation=cation, sodium=sodium, anion=sodium + 3 * cation)
        strengths = partial_strengths(0.0, sodium, cation, self.solution["anion"])
        self.bulk = [float(self_energy(a, [8 * np.pi * bjerrum * i for i in strengths], self.screen)) for a in radii]

    def segment_density(self, field):
        """Segments per nm^3 in the field per segment. Double precision holds about 700 kT over a chain, so the
        field keeps a floor where hardly any chain reaches, and a constant comes off it."""
        top = field.max() - floor
        w = n_segments * (np.maximum(field, top) - top)
        self.solver.compute_propagators({"A": w}, q_init={"G": self.graft})
        self.solver.compute_concentrations()
        rho = self.solver.get_concentration("A")
        return rho * grafting * n_segments / (rho.sum() * self.dz)

    def ion_densities(self, psi, u_mono, u_tri):
        """Na+, Y3+ and NO3- at potential psi, averaged over each ion, and the self-energies per z^2 of the
        monovalent ions and of Y3+."""
        s, mono, tri = self.solution, self.spreads[1] @ psi, self.spreads[2] @ psi
        return (s["sodium"] * np.exp(-np.clip(mono + u_mono, -300, 300)),
                s["cation"] * np.exp(-np.clip(3 * tri + 9 * u_tri, -300, 300)),
                s["anion"] * np.exp(np.clip(mono - u_mono, -300, 300)))

    def poisson_boltzmann(self, rho, u_mono, u_tri):
        """Newton iteration from the last potential, with no field at the wall and the potential of the solution,
        zero, at the far end. With spread screening every charge, the segments too, is spread over its Gaussian."""
        alpha, (segments, mono, tri) = 4 * np.pi * bjerrum, self.spreads
        fixed = segments @ rho

        def error(psi):
            sodium, cation, anion = ions = self.ion_densities(psi, u_mono, u_tri)
            charge = mono @ (sodium - anion) + 3 * (tri @ cation) - fixed
            return self.laplacian @ psi + alpha * charge, ions

        psi = self.psi
        for _ in range(300):
            f, (sodium, cation, anion) = error(psi)
            if np.max(np.abs(f)) < 1e-8:
                self.psi = psi
                return psi, (sodium, cation, anion)
            response = mono @ diags(sodium + anion) @ mono + 9 * (tri @ diags(cation) @ tri)
            step = spsolve(csc_matrix(alpha * response - self.laplacian), f)
            t, size = 1.0, np.linalg.norm(f)
            while not np.linalg.norm(error(psi + t * step)[0]) < size and t > 1e-6:
                t /= 2
            psi = psi + t * step
        raise RuntimeError("Poisson-Boltzmann did not converge")

    def residual(self, x):
        """Field per segment and self-energies per z^2, against what they make: zero at the solution."""
        field, *u = x.reshape(1 + len(radii), nz)
        felt = [blur(v) for blur, v in zip(self.blurs, u)]
        rho = self.segment_density(field)
        psi, (sodium, cation, anion) = self.poisson_boltzmann(rho, felt[1], felt[2])
        phi = np.clip(volume * rho, 0, 0.999)
        target = chi * (1 - 2 * phi) - np.log1p(-phi) - self.spreads[0] @ psi + felt[0]
        self.state = dict(rho=rho, psi=psi, sodium=sodium, cation=cation, anion=anion)
        if correlations:
            strengths = partial_strengths(rho, sodium, cation, anion)
            made = [self_energy(a, [8 * np.pi * bjerrum * blur(i) for i in strengths], self.screen) - bulk
                    for a, blur, bulk in zip(radii, self.blurs, self.bulk)]
        else:
            made = [np.zeros(nz)] * len(radii)
        return np.concatenate([target - field] + [new - old for new, old in zip(made, u)])

    def relax(self, x, plain_steps=0):
        """Damped steps x + mixing * residual: plain_steps of them, then max_steps at most with Anderson mixing of
        the last history steps, each change of a field limited to largest_step kT. Returns the state and its
        largest residual."""
        past_x, past_g = [], []
        for step in range(plain_steps + max_steps):
            f = self.residual(x)
            largest = np.max(np.abs(f))
            if largest < tolerance:
                break
            g = mixing * f
            if step < plain_steps:
                x = x + g
                continue
            past_x.append(x)
            past_g.append(g)
            del past_x[:-history - 1], past_g[:-history - 1]
            new = x + g
            if len(past_x) > 1:
                dx, dg = np.diff(past_x, axis=0), np.diff(past_g, axis=0)
                product = dg @ dg.T
                weights = np.linalg.solve(product + 1e-10 * np.trace(product) * np.eye(len(dg)), dg @ g)
                new = new - weights @ (dx + dg)
            x = x + min(1.0, largest_step / np.max(np.abs(new - x))) * (new - x)
        field, rho = x[:nz], self.state["rho"]
        if rho[field < field.max() - floor].max(initial=0) > 1e-4 * rho.max():
            print("  Warning: chains reach the floor of the field. Compare this state with a higher floor.")
        return x, largest

    def height(self):
        rho = self.state["rho"]
        return 2 * np.sum(self.z * rho) / np.sum(rho)

    def uptake(self):
        """Y3+ charge in the brush, beyond that of the solution, per chain charge."""
        cation = self.state["cation"]
        return 3 * np.sum(cation - cation[-1]) * self.dz / (grafting * n_segments)


def solve(brush, x, plain_steps=0):
    """The state from x and the last potential; the potential is kept only when the state converges."""
    psi = brush.psi
    try:
        x, left = brush.relax(x, plain_steps)
        if not left < tolerance:
            raise RuntimeError(f"No solution: residual {left:.1e}")
    except (RuntimeError, ValueError):
        brush.psi = psi
        raise
    return x


brush = Brush()
print(f"Born radii {radii} nm, screening by {screening} charges" + ("" if correlations else ", mean field"))

# The swollen brush in NaNO3 alone, from a parabolic field
brush.set_solution(0.0)
parabola = 3 * np.pi ** 2 / (8 * n_segments ** 2 * segment ** 2) * np.maximum(70.0 ** 2 - brush.z ** 2, 0)
x = solve(brush, np.concatenate([0.5 + parabola] + [np.zeros(nz)] * len(radii)), plain_steps=600)
h6 = brush.height()
print(f"{ionic_strength * 1e3:g} mM NaNO3: h6 = {h6:.3f} nm")

# Y3+ added step by step, each state from the last one; a step that fails is halved (in log scale)
print(f"{'Y3+ (mol/L)':>12} {'log10(h/h6)':>12} {'uptake':>8}")
done, pending = [0.0], list(yttrium)
while pending:
    y = pending[0]
    brush.set_solution(y)
    try:
        x = solve(brush, x)
    except (RuntimeError, ValueError):
        if (done[-1] > 0 and y < 1.001 * done[-1]) or y < 1e-15:
            print(f"{y:12.3g}  no solution")
            break
        pending.insert(0, np.sqrt(done[-1] * y) if done[-1] > 0 else y / 10)
        continue
    pending.pop(0)
    done.append(y)
    print(f"{y:12.3g} {np.log10(brush.height() / h6):12.3f} {brush.uptake():8.3f}", flush=True)
