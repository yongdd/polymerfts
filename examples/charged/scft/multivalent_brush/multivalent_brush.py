"""SCFT of a polyelectrolyte brush with multivalent ions and ion correlations, shared by the scripts of this folder.

The model is that of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101 (2025), with the corrections of
docs/theory/ElectrostaticCorrelations.md. Chains grafted at the wall z = 0 sit in a field per segment, and
the ions follow the Poisson-Boltzmann equation with the self-energy z^2 u of the Gaussian charge of an ion,
which the ionic strength around it lowers. Every segment carries one negative charge. Two forms of u:

- LocalSelfEnergy: the closed form of a uniform medium at the ionic strength averaged over the ion radius.
- GreenFunction: the Green function in the varying ionic strength of a planar brush.

The self-energy is not part of the polymerfts SCFT class, so Brush writes the iteration itself, damped
steps and Anderson mixing, and takes only the chains from PropagatorSolver. Lengths are in nm and energies
in kT.
"""

from functools import partial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import solve_banded
from scipy.linalg.lapack import dpttrf
from scipy.special import erfcx

from polymerfts import PropagatorSolver

n_segments = 100        # Segments N of a chain
segment = 1.0           # Statistical segment length b, nm
volume = 1.0            # Segment volume v, nm^3
chi = 0.5               # Segment-solvent interaction
bjerrum = 0.7           # Bjerrum length, nm
radius = 0.25           # Radius of the anions, the counterions and the charges of the chain, nm
molar = 0.6022          # nm^-3 per mol/L
graft_width = 0.5       # Width of the graft layer at the wall, nm
floor = 6.0             # Lowest field below its largest value, kT per segment
box_length = 6600.0     # Length that holds the counterions, fitted to Fig. 2 of the paper, nm

wave_numbers = 36       # Lateral k of the Green function
behind_wall = 5.0       # Grid of the Green function behind the wall, nm


def array_module():
    """CuPy when it is installed and finds a GPU, and NumPy otherwise."""
    try:
        import cupy
        cupy.zeros(1)
        return cupy
    except Exception:
        return np


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def born_energy(a, kappa, erfcx=erfcx):
    """Self-energy per z^2 of a Gaussian charge of radius a in a uniform medium of inverse screening length kappa."""
    x = a * kappa
    return bjerrum / (2 * a) * (1 - x * erfcx(x / np.sqrt(np.pi)))


def screening(strength):
    """Inverse screening length at ionic strength sum z^2 c / 2."""
    return (8 * np.pi * bjerrum * strength) ** 0.5


class LocalSelfEnergy:
    """u per z^2 from the closed form of a uniform medium, at the ionic strength averaged over a Gaussian of the
    ion radius. The fields of the ions and of the segments feel u averaged in the same way, which makes the
    equations the stationary point of Brush.grand_potential()."""

    def __init__(self, brush):
        if len(brush.radii) > 1:
            raise ValueError("The local self-energy takes one radius for all ions")
        self.xp = brush.xp
        if brush.xp is np:
            from scipy import ndimage, special
        else:
            from cupyx.scipy import ndimage, special
        self.erfcx = special.erfcx
        self.blur = partial(ndimage.gaussian_filter, sigma=[radius / d for d in brush.dx] + [radius / brush.dz],
                            mode=["wrap"] * len(brush.cells) + ["reflect"])

    def set_bulk(self, strength):
        self.bulk_strength = strength
        self.bulk = born_energy(radius, screening(strength))

    def felt(self, u):
        return self.blur(u)

    def __call__(self, strength):
        return [born_energy(radius, screening(self.blur(strength)), self.erfcx) - self.bulk]

    def free_energy(self, strength):
        """Density of the free energy of the correlations, kT/nm^3, with dw/dI = 2 (B(I) - B(bulk)) at the
        averaged ionic strength I, B being born_energy, and w = 0 in the bulk."""
        xp = self.xp
        averaged = self.blur(strength)
        table = np.geomspace(1e-9, 1.5 * max(30.0, self.bulk_strength, float(xp.max(averaged))), 20001)
        gain = 2 * (born_energy(radius, screening(table)) - self.bulk)
        energy = np.concatenate([[0.0], np.cumsum(0.5 * (gain[1:] + gain[:-1]) * np.diff(table))])
        energy = energy - np.interp(self.bulk_strength, table, energy)
        return xp.interp(averaged, xp.asarray(table), xp.asarray(energy))


def charge_bands(a, z, dz):
    """h[i + p, i] in row p + w, p = -w..w: the Gaussian charge along z with h(k) = exp(-k^2 a^2 / 2 pi),
    mirrored at both ends of the grid z, cut where it falls below 1e-16 of its peak."""
    width = a / np.sqrt(np.pi)
    w = int(np.ceil(8.6 * width / dz))
    start, end = z[0] - dz / 2, z[-1] + dz / 2
    bands = np.zeros((2 * w + 1, len(z)))
    for p in range(-w, w + 1):
        i = np.arange(max(0, -p), min(len(z), len(z) - p))
        for distance in (z[i + p] - z[i], z[i + p] + z[i] - 2 * start, 2 * end - z[i + p] - z[i]):
            bands[p + w, i] += np.exp(-0.5 * (distance / width) ** 2)
    return bands / bands.sum(axis=0).mean() / dz


class GreenFunction:
    """u per z^2 from the Green function of Eq. S25 in the varying ionic strength of a planar brush, without
    its nonlocal term I_ex. Each lateral wave number k has its own problem along z,

        [-eps d^2/dz^2 + eps k^2 + 2 I(z)] G_k(z, z') = delta(z - z'),
        u(z) = 1/2 int k dk / (2 pi) exp(-k^2 a^2 / pi) [h G_k h](z, z),

    with h the Gaussian charge along z, less the same integral in the bulk. The medium of the first cell
    continues behind the wall and that of the last cell past the far end, without end: the grid reaches
    behind_wall into both, and its end rows hold the pivot of the rest, so that neither end holds an image
    charge. G_k is the inverse of a tridiagonal
    matrix, and u needs only the few bands of it that h reaches: the pivots of the LDL^T factors from both
    ends give them in a time proportional to the grid."""

    def __init__(self, brush):
        if brush.cells or brush.xp is not np:
            raise ValueError("The Green function is written for a planar brush on NumPy")
        self.dz, self.nz = brush.dz, brush.nz
        self.pad = int(round(behind_wall / brush.dz))
        z = (np.arange(-self.pad, brush.nz + self.pad) + 0.5) * brush.dz
        self.wave = np.geomspace(0.03, np.sqrt(np.pi * 36) / min(brush.radii), wave_numbers)
        step = np.log(self.wave[1] / self.wave[0])
        self.charges = []       # The weight of each k in the integral, and the bands of h, at each radius
        for a in brush.radii:
            factor = 0.5 * step * self.wave ** 2 / (2 * np.pi) * np.exp(-self.wave ** 2 * a ** 2 / np.pi)
            factor[[0, -1]] *= 0.5
            self.charges.append((factor, charge_bands(a, z, brush.dz)))
        self.bulk = [0.0] * len(brush.radii)

    def set_bulk(self, strength):
        self.bulk = [0.0] * len(self.bulk)
        self.bulk = self(np.full(self.nz, strength))

    def felt(self, u):
        return u

    def __call__(self, strength):
        """u per z^2 at each radius, less its bulk value, at the ionic strength on the grid of the brush."""
        n = 2 * self.pad + self.nz
        eps = 1 / (4 * np.pi * bjerrum)
        off = -eps / self.dz ** 2
        extended = np.r_[np.full(self.pad, strength[0]), strength, np.full(self.pad, strength[-1])]
        diagonal = -2 * off + eps * self.wave[:, None] ** 2 + 2 * extended
        diagonal[:, [0, -1]] = 0.5 * (diagonal[:, [0, -1]] + np.sqrt(diagonal[:, [0, -1]] ** 2 - 4 * off ** 2))
        # G_k[i, i] from the pivots of both directions, then G_k[i, i + m] = -off / forward[i] * G_k[i + 1, i + m]
        forward = np.array([dpttrf(d, np.full(n - 1, off))[0] for d in diagonal])
        backward = np.array([dpttrf(d[::-1], np.full(n - 1, off))[0][::-1] for d in diagonal])
        band = 1 / (forward + backward - diagonal)
        ratio = -off / forward[:, :-1]
        sums = [[factor @ band] for factor, _ in self.charges]      # Bands of sum_k factor_k G_k
        for m in range(1, max(len(h) for _, h in self.charges)):
            band = ratio[:, :n - m] * band[:, 1:]
            for s, (factor, _) in zip(sums, self.charges):
                s.append(factor @ band)
        u = []
        for s, (_, h), bulk in zip(sums, self.charges, self.bulk):
            w = len(h) // 2
            total = np.zeros(n)
            for m in range(2 * w + 1):      # [h G h](i, i) = sum over p, q of h[i + p, i] G[i + p, i + q] h[i + q, i]
                padded = np.zeros(n + 2 * w)
                padded[w:w + n - m] = s[m]
                total += (2 if m else 1) * np.sum(h[:2 * w + 1 - m] * h[m:] * sliding_window_view(padded, n)[:2 * w + 1 - m], axis=0)
            u.append(self.dz * total[self.pad:self.pad + self.nz] - bulk)
        return u


class Brush:
    """The fields of a brush on a grid of nz points over length nm from the wall and, for a lateral pattern,
    of cells points over size nm along the wall, periodic. Without cells the brush is planar. Each lateral
    column holds the same chains, Eq. S30 of the paper at each lateral position.

    The unknowns x of a state are the field per segment and the self-energy per z^2 at each ion radius:
    radius, and cation_radius for the multivalent cations when it differs. Arrays are of the module xp."""

    def __init__(self, grafting, length, nz, ds, cells=(), size=(), green_function=False, cation_radius=radius, xp=np):
        self.grafting, self.length, self.nz, self.xp = grafting, length, nz, xp
        self.dz = length / nz
        self.z = (np.arange(nz) + 0.5) * self.dz
        self.cells, self.size = tuple(cells), list(size)
        self.shape = self.cells + (nz,)
        self.axes = tuple(range(len(cells)))
        self.columns = int(np.prod(cells))
        self.dx = [s / n for s, n in zip(size, cells)]
        unit = segment * np.sqrt(n_segments)
        self.solver = PropagatorSolver(nx=list(self.shape), lx=[s / unit for s in size] + [length / unit], ds=ds,
                                       bond_lengths={"A": 1.0}, bc=["periodic", "periodic"] * len(cells) + ["absorbing", "reflecting"],
                                       chain_model="continuous", numerical_method="cn-adi2")
        self.solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
        self.n_steps = int(round(1 / ds))
        self.graft = np.broadcast_to(np.exp(-0.5 * (self.z / graft_width) ** 2), self.shape).copy()
        weight = np.full(self.shape, self.dz)    # Beyond the grid the box holds only conserved ions, at the last potential
        weight[..., -1] += box_length - length
        self.weight = xp.asarray(weight)
        self.psi = xp.zeros(self.shape)
        self.radii = [radius] if cation_radius == radius else [radius, cation_radius]
        self.self_energy = GreenFunction(self) if green_function else LocalSelfEnergy(self)
        self.correlations = True    # False keeps u at zero
        if cells:                   # Lateral eigenvalues of the finite differences, for the preconditioner
            wave2 = np.zeros(cells)
            for i, n in enumerate(cells):
                one = (2 - 2 * np.cos(2 * np.pi * np.arange(n) / n)) / self.dx[i] ** 2
                wave2 = wave2 + one.reshape([-1 if j == i else 1 for j in self.axes])
            self.half = tuple(n // 2 + 1 if i == len(cells) - 1 else n for i, n in enumerate(cells))
            self.wave2 = xp.asarray(wave2[tuple(slice(0, n) for n in self.half)].reshape(-1, 1))

    def set_solution(self, valence, cation, sodium=0.0, counterions=True, grounded=False):
        """Ions in nm^-3 from a reservoir: z:1 salt of multivalent cations at cation, and sodium ions, with their
        anions. counterions keeps the monovalent counterions of the chains in the box of box_length. grounded
        holds the potential of the reservoir at the far end; otherwise no field crosses it."""
        anion = sodium + valence * cation
        self.solution = dict(valence=valence, cation=cation, sodium=sodium, anion=anion, far=-1.0 if grounded else 1.0,
                             conserved=self.grafting * n_segments if counterions else 0.0)
        self.self_energy.set_bulk(0.5 * (sodium + valence ** 2 * cation + anion))

    def pack(self, field, *u):
        """The unknowns of a state from the field and the self-energies, each broadcast to the grid."""
        xp = self.xp
        return xp.concatenate([xp.broadcast_to(xp.asarray(v), self.shape).ravel() for v in (field,) + u])

    def unpack(self, x):
        return [v.reshape(self.shape) for v in self.xp.split(x, 1 + len(self.radii))]

    def segment_density(self, field):
        """Segments per nm^3 in the field per segment. Double precision holds about 700 kT over a chain, so the
        field keeps a floor where hardly any chain reaches, and a constant comes off it. In a lateral pattern
        the forward start is 1 / q+ of each column."""
        field = to_numpy(field)
        top = field.max() - floor
        w = n_segments * (np.maximum(field, top) - top)
        self.solver.compute_propagators({"A": w.ravel()}, q_init={"G": self.graft.ravel()})
        back = self.solver.get_propagator(0, 1, 0, self.n_steps).reshape(self.shape)
        partition = np.sum(self.graft * back, axis=-1, keepdims=True)
        if self.columns > 1:
            self.solver.compute_propagators({"A": w.ravel()}, q_init={"G": (self.graft * partition.mean() / partition).ravel()})
        self.solver.compute_concentrations()
        rho = self.solver.get_concentration("A").reshape(self.shape)
        self.chain_energy = self.grafting * np.mean(n_segments * top - np.log(partition))
        return self.xp.asarray(rho * self.grafting * n_segments / (rho.sum() * self.dz / self.columns))

    def ion_densities(self, psi, u):
        """Conserved counterions, sodium, multivalent cations and anions at potential psi and the self-energies u."""
        xp, s = self.xp, self.solution
        boltzmann = xp.exp(-xp.clip(psi + u[0], -300, 300))
        z_plus = s["valence"]
        return (s["conserved"] * boltzmann / (xp.sum(self.weight * boltzmann) / self.columns), s["sodium"] * boltzmann,
                s["cation"] * xp.exp(-xp.clip(z_plus * psi + z_plus ** 2 * u[-1], -300, 300)),
                s["anion"] * xp.exp(xp.clip(psi - u[0], -300, 300)))

    def laplacian(self, v):
        xp = self.xp
        ends = xp.concatenate([v[..., :1], v, self.solution["far"] * v[..., -1:]], axis=-1)
        out = (ends[..., 2:] - 2 * v + ends[..., :-2]) / self.dz ** 2
        for i in self.axes:
            out = out + (xp.roll(v, 1, i) + xp.roll(v, -1, i) - 2 * v) / self.dx[i] ** 2
        return out

    def preconditioner(self, mean_d):
        """The inverse of laplacian - mean_d(z), one tridiagonal system along z per lateral mode, by parallel
        cyclic reduction: each level acts on whole arrays, with no loop along z."""
        xp, nz = self.xp, self.nz
        off = 1 / self.dz ** 2
        b = -2 * off - self.wave2 - mean_d.reshape(1, -1)
        b = xp.concatenate([b[:, :1] + off, b[:, 1:-1], b[:, -1:] + self.solution["far"] * off], axis=1)
        a_ = xp.full_like(b, off)
        c_ = xp.full_like(b, off)
        a_[:, 0] = 0.0
        c_[:, -1] = 0.0
        levels = []
        stride = 1
        while stride < nz:
            below = lambda f, fill: xp.concatenate([xp.full_like(f[:, :stride], fill), f[:, :-stride]], axis=1)
            above = lambda f, fill: xp.concatenate([f[:, stride:], xp.full_like(f[:, :stride], fill)], axis=1)
            alpha = -a_ / below(b, 1.0)
            gamma = -c_ / above(b, 1.0)
            b, a_, c_ = b + alpha * below(c_, 0.0) + gamma * above(a_, 0.0), alpha * below(a_, 0.0), gamma * above(c_, 0.0)
            levels.append((stride, alpha, gamma))
            stride *= 2

        def apply(f):
            fh = xp.fft.rfftn(f, axes=self.axes).reshape(-1, nz)
            for stride, alpha, gamma in levels:
                zero = xp.zeros_like(fh[:, :stride])
                fh = fh + alpha * xp.concatenate([zero, fh[:, :-stride]], axis=1) + gamma * xp.concatenate([fh[:, stride:], zero], axis=1)
            return xp.fft.irfftn((fh / b).reshape(self.half + (nz,)), s=self.cells, axes=self.axes)
        return apply

    def conjugate_gradient(self, d, apply, f):
        """Solves (d - laplacian) x = f, with the preconditioner from the mean of d."""
        xp = self.xp
        x = xp.zeros_like(f)
        stop = 1e-6 * float(xp.linalg.norm(f))
        if stop == 0:
            return x
        r = f.copy()
        z_ = -apply(r)
        p = z_.copy()
        product = float(xp.sum(r * z_))
        for _ in range(1200):
            q = d * p - self.laplacian(p)
            step = product / float(xp.sum(p * q))
            x = x + step * p
            r = r - step * q
            if float(xp.linalg.norm(r)) < stop:
                break
            z_ = -apply(r)
            next_product = float(xp.sum(r * z_))
            p = z_ + next_product / product * p
            product = next_product
        return x

    def solve_linear(self, d, rights):
        """Solves (d - laplacian) x = f for each f of rights: by bands in a planar brush, else by conjugate gradients."""
        if not self.cells:
            bands = np.zeros((3, self.nz))
            bands[0, 1:] = bands[2, :-1] = -1 / self.dz ** 2
            bands[1] = 2 / self.dz ** 2 + d
            bands[1, 0] -= 1 / self.dz ** 2
            bands[1, -1] -= self.solution["far"] / self.dz ** 2
            return list(solve_banded((1, 1), bands, np.column_stack(rights)).T)
        apply = self.preconditioner(self.xp.mean(d.reshape(-1, self.nz), axis=0))
        return [self.conjugate_gradient(d, apply, f) for f in rights]

    def poisson_boltzmann(self, rho, u):
        """Newton iteration from the last potential, with no field at the wall. Conserved ions couple all points
        through their amount, one dense row that the step takes out (Sherman-Morrison)."""
        xp, s = self.xp, self.solution
        alpha = 4 * np.pi * bjerrum
        z_plus = s["valence"]

        def error(psi):
            c, sodium, cation, anion = ions = self.ion_densities(psi, u)
            return self.laplacian(psi) + alpha * (c + sodium + z_plus * cation - anion - rho), ions

        psi = self.psi
        for _ in range(300):
            f, ions = error(psi)
            if float(xp.max(xp.abs(f))) < 1e-8:
                self.psi = psi
                return psi, ions
            c, sodium, cation, anion = ions
            first, second = self.solve_linear(alpha * (c + sodium + z_plus ** 2 * cation + anion), [f, -alpha * c])
            share = c * self.weight / (max(s["conserved"], 1e-300) * self.columns)
            step = first - second * xp.sum(share * first) / (1 + xp.sum(share * second))
            t, size = 1.0, float(xp.linalg.norm(f))
            while not float(xp.linalg.norm(error(psi + t * step)[0])) < size and t > 1e-6:
                t /= 2
            psi = psi + t * step
        raise RuntimeError("Poisson-Boltzmann did not converge")

    def residual(self, x):
        """Field per segment and self-energies per z^2, against what they make: zero at the solution."""
        xp = self.xp
        field, *u = self.unpack(x)
        felt = [self.self_energy.felt(v) for v in u]
        rho = self.segment_density(field)
        psi, (c, sodium, cation, anion) = self.poisson_boltzmann(rho, felt)
        phi = xp.clip(volume * rho, 0, 0.999)
        target = chi * (1 - 2 * phi) - xp.log1p(-phi) - psi + felt[0]
        strength = 0.5 * (rho + c + sodium + self.solution["valence"] ** 2 * cation + anion)
        self.state = dict(rho=rho, psi=psi, c=c, sodium=sodium, cation=cation, anion=anion, strength=strength)
        made = self.self_energy(strength) if self.correlations else [xp.zeros(self.shape)] * len(u)
        return xp.concatenate([(target - field).ravel()] + [(new - old).ravel() for new, old in zip(made, u)])

    def height(self):
        rho = to_numpy(self.state["rho"]).reshape(-1, self.nz).mean(axis=0)
        return 2 * np.sum(self.z * rho) / np.sum(rho)

    def describe(self):
        rho = self.state["rho"]
        text = f"height {self.height():.3f} nm, largest phi {float(rho.max()) * volume:.3f}"
        if self.cells:
            areal = to_numpy(rho).reshape(-1, self.nz).sum(axis=1)
            text += f", lateral contrast {areal.std() / areal.mean():.3f}"
        return text

    def check_floor(self, x):
        field, rho = to_numpy(self.unpack(x)[0]), to_numpy(self.state["rho"])
        if rho[field < field.max() - floor].max(initial=0) > 1e-4 * rho.max():
            print("  Warning: chains reach the floor of the field. Compare this state with a higher floor.")

    def relax(self, x, mixing, history=20, largest_step=0.2, tolerance=1e-5, max_steps=4000, plain_steps=0, report=50, label=""):
        """Damped steps x + mixing * residual: plain_steps of them, then max_steps at most with Anderson mixing of
        the last history steps, each change of a field limited to largest_step kT. Prints every report steps,
        unless report is 0. Returns the state and its largest residual."""
        xp = self.xp
        past_x, past_g = [], []
        for step in range(plain_steps + max_steps):
            f = self.residual(x)
            largest = float(xp.max(xp.abs(f)))
            if report and step % report == 0:
                print(f"  {label}step {step}: residual {largest:.2e}, {self.describe()}", flush=True)
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
                dx, dg = xp.stack(past_x[1:]) - xp.stack(past_x[:-1]), xp.stack(past_g[1:]) - xp.stack(past_g[:-1])
                product = to_numpy(dg @ dg.T)
                weights = np.linalg.solve(product + 1e-10 * np.trace(product) * np.eye(len(dg)), to_numpy(dg @ g))
                new = new - xp.asarray(weights) @ (dx + dg)
            x = x + min(1.0, largest_step / float(xp.max(xp.abs(new - x)))) * (new - x)
        self.check_floor(x)
        return x, largest

    def grand_potential(self, x):
        """Per lateral area, kT/nm^2, up to a constant, with the local self-energy and no field at the far end."""
        if not isinstance(self.self_energy, LocalSelfEnergy) or self.solution["far"] != 1:
            raise ValueError("The grand potential is written for the local self-energy and no field at the far end")
        xp, s = self.xp, self.solution
        self.residual(x)
        field, state = self.unpack(x)[0], self.state
        chains = self.chain_energy - float(xp.sum(field * state["rho"])) * self.dz / self.columns
        phi = xp.clip(volume * state["rho"], 0, 0.999)
        mixture = float(xp.sum((1 - phi) * xp.log1p(-phi) + chi * phi * (1 - phi))) / volume * self.dz / self.columns
        ideal = xp.zeros(self.shape)
        for n, bulk in ((state["cation"], s["cation"]), (state["anion"], s["anion"]), (state["sodium"], s["sodium"])):
            if bulk > 0:
                ideal = ideal + n * (xp.log(n / bulk) - 1) + bulk
        if s["conserved"] > 0:
            ideal = ideal + state["c"] * (xp.log(state["c"]) - 1)
        ideal = float(xp.sum(self.weight * ideal)) / self.columns
        coulomb = float(xp.sum(xp.diff(state["psi"], axis=-1) ** 2)) / self.dz ** 2
        for i in self.axes:
            coulomb = coulomb + float(xp.sum((xp.roll(state["psi"], -1, i) - state["psi"]) ** 2)) / self.dx[i] ** 2
        coulomb = coulomb * self.dz / self.columns / (8 * np.pi * bjerrum)
        correlations = 0.0
        if self.correlations:
            correlations = float(xp.sum(self.weight * self.self_energy.free_energy(state["strength"]))) / self.columns
        return chains + mixture + ideal + coulomb + correlations
