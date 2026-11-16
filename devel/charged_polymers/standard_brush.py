"""The standard model of this folder: SCFT of a polyelectrolyte brush in a solution of monovalent and multivalent
ions, with ion correlations.

It is the smallest model found that reproduces the neutron reflectivity titration of Yu et al.,
Macromolecules 49, 5609 (2016), and it keeps nothing else. reports/StandardBrush.md gives the reason and the
check of each element:

- Chains: Gaussian chains grafted at the wall z = 0, continuous with cn-adi2, in a field per segment, and
  Flory-Huggins mixing with the solvent. Every segment carries one negative charge.
- Ions: point charges in the Poisson-Boltzmann equation, in equilibrium with a neutral solution of a
  monovalent salt and a z:1 salt of multivalent cations at the far end of the grid.
- Correlations: every charge, segments included, feels z^2 u, the self-energy of a Gaussian charge of radius a
  in a uniform medium of the ionic strength around it, screened by charges of the same radius (N12 of
  reports/ElectrostaticCorrelations.md). The ionic strength is averaged over a Gaussian of width a, and the
  fields feel u averaged in the same way, so the equations are the stationary point of one free energy.

How much of the solution the brush can draw from, the finite pool of the experiment, is a balance on the uptake
and lives in the script. The self-energy is not part of the polymerfts SCFT class, so Brush writes the
iteration itself, damped steps and Anderson mixing, and takes only the chains from PropagatorSolver. Lengths
are in nm and energies in kT.
"""

from functools import lru_cache, partial

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

from polymerfts import PropagatorSolver

n_segments = 100        # Segments N of a chain. Heights scale with N; relative heights and uptake hardly change
segment = 1.0           # Statistical segment length b, nm
volume = 1.0            # Segment volume v, nm^3
chi = 0.5               # Segment-solvent interaction
bjerrum = 0.7           # Bjerrum length, nm
radius = 0.25           # Radius a of every charge, nm
molar = 0.6022          # nm^-3 per mol/L
graft_width = 0.5       # Width of the graft layer at the wall, nm
floor = 6.0             # Lowest field below its largest value, kT per segment


@lru_cache(maxsize=None)
def self_energy_table():
    """N12, u(x) = 2/pi int_0^inf t^2 e^(-t^2/pi) / (t^2 + x^2 e^(-t^2/pi)) dt in units of l_B / 2a, on a table
    of x = a kappa, by the trapezoid rule in ln t."""
    x = np.concatenate([[0.0], np.geomspace(1e-7, 1e3, 8001)])
    t = np.geomspace(1e-12, 20.0, 2400)
    weight = np.full(t.size, np.log(t[1] / t[0])) * t
    weight[[0, -1]] *= 0.5
    decay = np.exp(-t ** 2 / np.pi)
    u = np.array([2 / np.pi * (t ** 2 * decay / (t ** 2 + one ** 2 * decay)) @ weight for one in x])
    return x, u


def self_energy(kappa):
    """Self-energy per z^2 of a Gaussian charge of radius a in a uniform medium of inverse screening length kappa
    made by charges of the same radius."""
    x, u = self_energy_table()
    return bjerrum / (2 * radius) * np.interp(radius * kappa, x, u)


def screening(strength):
    """Inverse screening length at ionic strength sum z^2 c / 2."""
    return (8 * np.pi * bjerrum * strength) ** 0.5


class Brush:
    """The fields of a planar brush on a grid of nz points over length nm from the wall. The unknowns x of a state
    are the field per segment and the self-energy per z^2. correlations=False keeps u at zero."""

    def __init__(self, grafting, length, nz, ds, correlations=True):
        self.grafting, self.length, self.nz, self.correlations = grafting, length, nz, correlations
        self.dz = length / nz
        self.z = (np.arange(nz) + 0.5) * self.dz
        unit = segment * np.sqrt(n_segments)
        self.solver = PropagatorSolver(nx=[nz], lx=[length / unit], ds=ds, bond_lengths={"A": 1.0},
                                       bc=["absorbing", "reflecting"], chain_model="continuous", numerical_method="cn-adi2")
        self.solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
        self.n_steps = int(round(1 / ds))
        self.graft = np.exp(-0.5 * (self.z / graft_width) ** 2)
        self.blur = partial(gaussian_filter, sigma=radius / self.dz, mode="reflect")
        self.psi = np.zeros(nz)

    def set_solution(self, valence, cation, sodium):
        """A neutral solution at the far end: z:1 salt of multivalent cations at cation and a 1:1 salt of
        monovalent cations at sodium, nm^-3, with their monovalent anions."""
        anion = sodium + valence * cation
        self.solution = dict(valence=valence, cation=cation, sodium=sodium, anion=anion)
        self.bulk = self_energy(screening(0.5 * (sodium + valence ** 2 * cation + anion)))

    def pack(self, field, u):
        return np.concatenate([np.broadcast_to(field, self.nz), np.broadcast_to(u, self.nz)])

    def unpack(self, x):
        return x[:self.nz], x[self.nz:]

    def segment_density(self, field):
        """Segments per nm^3 in the field per segment. Double precision holds about 700 kT over a chain, so the
        field keeps a floor where hardly any chain reaches, and a constant comes off it."""
        top = field.max() - floor
        w = n_segments * (np.maximum(field, top) - top)
        self.solver.compute_propagators({"A": w}, q_init={"G": self.graft})
        self.solver.compute_concentrations()
        rho = self.solver.get_concentration("A")
        return rho * self.grafting * n_segments / (rho.sum() * self.dz)

    def ion_densities(self, psi, u):
        """Monovalent cations, multivalent cations and anions at potential psi and self-energy u per z^2."""
        s, z_plus = self.solution, self.solution["valence"]
        return (s["sodium"] * np.exp(-np.clip(psi + u, -300, 300)),
                s["cation"] * np.exp(-np.clip(z_plus * psi + z_plus ** 2 * u, -300, 300)),
                s["anion"] * np.exp(np.clip(psi - u, -300, 300)))

    def poisson_boltzmann(self, rho, u):
        """Newton iteration from the last potential, with no field at the wall and the potential of the solution,
        zero, half a cell past the far end."""
        alpha = 4 * np.pi * bjerrum
        z_plus = self.solution["valence"]
        off = 1 / self.dz ** 2

        def error(psi):
            sodium, cation, anion = ions = self.ion_densities(psi, u)
            ends = np.concatenate([psi[:1], psi, -psi[-1:]])
            return (ends[2:] - 2 * psi + ends[:-2]) * off + alpha * (sodium + z_plus * cation - anion - rho), ions

        psi = self.psi
        for _ in range(300):
            f, (sodium, cation, anion) = error(psi)
            if np.max(np.abs(f)) < 1e-8:
                self.psi = psi
                return psi, (sodium, cation, anion)
            bands = np.zeros((3, self.nz))
            bands[0, 1:] = bands[2, :-1] = -off
            bands[1] = 2 * off + alpha * (sodium + z_plus ** 2 * cation + anion)
            bands[1, 0] -= off
            bands[1, -1] += off
            step = solve_banded((1, 1), bands, f)
            t, size = 1.0, np.linalg.norm(f)
            while not np.linalg.norm(error(psi + t * step)[0]) < size and t > 1e-6:
                t /= 2
            psi = psi + t * step
        raise RuntimeError("Poisson-Boltzmann did not converge")

    def residual(self, x):
        """Field per segment and self-energy per z^2, against what they make: zero at the solution."""
        field, u = self.unpack(x)
        felt = self.blur(u)
        rho = self.segment_density(field)
        psi, (sodium, cation, anion) = self.poisson_boltzmann(rho, felt)
        phi = np.clip(volume * rho, 0, 0.999)
        target = chi * (1 - 2 * phi) - np.log1p(-phi) - psi + felt
        strength = 0.5 * (rho + sodium + self.solution["valence"] ** 2 * cation + anion)
        self.state = dict(rho=rho, psi=psi, sodium=sodium, cation=cation, anion=anion, strength=strength)
        made = self_energy(screening(self.blur(strength))) - self.bulk if self.correlations else np.zeros(self.nz)
        return np.concatenate([target - field, made - u])

    def height(self):
        rho = self.state["rho"]
        return 2 * np.sum(self.z * rho) / np.sum(rho)

    def uptake(self):
        """Excess multivalent charge in the brush per chain charge."""
        cation = self.state["cation"]
        return self.solution["valence"] * np.sum(cation - cation[-1]) * self.dz / (self.grafting * n_segments)

    def check_floor(self, x):
        field, rho = self.unpack(x)[0], self.state["rho"]
        if rho[field < field.max() - floor].max(initial=0) > 1e-4 * rho.max():
            print("  Warning: chains reach the floor of the field. Compare this state with a higher floor.")

    def relax(self, x, mixing, history=20, largest_step=0.2, tolerance=1e-5, max_steps=4000, plain_steps=0):
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
        self.check_floor(x)
        return x, largest
