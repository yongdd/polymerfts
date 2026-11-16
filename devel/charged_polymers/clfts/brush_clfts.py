"""3D CL-FTS of the polyelectrolyte brush of Fig. 2 of Duan, Agrawal and Wang, PRL 134, 048101 (2025), with
discrete chains and the real-space bond "rs", started from the 1D SCFT (mean field) of the same model.

Model (nm, kT): N = 100 segments of b = 1 nm and v = 1 nm^3 grafted at 0.1 chains/nm^2 near an absorbing wall;
every segment carries -1; a solvent of one segment with chi = 0.5; point ions from a reservoir: C+ (the
monovalent counterions), the z-valent cation Y and the anion X; every charge is a Gaussian of a_W = 0.25 nm (code
smearing a_W/sqrt(pi)); l_B = 0.7 nm. Code units: R0 = b sqrt(N) = 10 nm, concentrations per segment of
rho0 = 1 nm^-3, nbar = (rho0 R0^3 / N)^2 = 100.

CL-FTS rejects grafted chains (only SCFT passes q_init), so this script patches both in. The lateral axes are
reflecting walls (the default; their images deplete the ions next to them) or periodic (--lateral-bc periodic).
The reservoir: c_s per ion; CL-FTS takes beta mu_s = ln(N c_s) + mu_ex,s with mu_ex the one-loop self-energy of the
reservoir on this grid, so that the bulk holds c_s; SCFT and a mean-field run (--nbar 0) take ln(N c_s).

usage: python brush_clfts.py out.json --valence 3 --rho 1e-3 --c-mono 2.5e-3 [options]
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import polymerfts.fts.clfts as clfts_module
from polymerfts import CLFTS, SCFT
from polymerfts.theory import chains
from polymerfts.theory.electrostatics import homogeneous_potential

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--valence", type=int, default=1)
parser.add_argument("--rho", type=float, default=1e-6, help="Y in the reservoir, mol/L")
parser.add_argument("--c-mono", type=float, default=2.5e-3, help="C+ in the reservoir, mol/L")
parser.add_argument("--far", default=None, help="C,Y,X in the reservoir, mol/L (the far field of far_field.json); overrides --c-mono and the Y, X of --rho")
parser.add_argument("--dx", type=float, default=0.5, help="grid spacing, nm")
parser.add_argument("--lateral", type=float, default=10.0, help="box along the wall, nm")
parser.add_argument("--height", type=float, default=100.0, help="box normal to the wall, nm")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--equilibration", type=int, default=0, help="steps before the profiles are averaged")
parser.add_argument("--interval", type=int, default=10, help="steps between measurements")
parser.add_argument("--alpha", type=float, default=0.0, help="dynamic_stabilization")
parser.add_argument("--psi-scale", type=float, default=1.0, help="psi_dt_scaling")
parser.add_argument("--psi-relax", type=float, default=0.0, help="if > 0: precondition psi by the ion response D0 and set psi_dt_scaling so that one step relaxes the neutrality mode by this fraction (s dt D0)")
parser.add_argument("--zeta", type=float, default=100.0, help="zeta_n")
parser.add_argument("--freeze-psi", action="store_true", help="diagnostic: psi keeps its SCFT value (no electrostatic fluctuations; the ions take the chemical potentials of mean field)")
parser.add_argument("--mobility", default=None, help="mobilities of the auxiliary fields W-,W+ (dt_scaling), default 1,1")
parser.add_argument("--uniform-relax", type=float, default=0.0, help="if > 0: the uniform mode of psi (the Donnan potential) takes its own mobility s0 = value / (dt d0), so that one step removes about this fraction of the net charge")
parser.add_argument("--ion-smear", type=float, default=None, help="Gaussian smearing of the ions, nm; default a_W/sqrt(pi)")
parser.add_argument("--cation-smear", type=float, default=None, help="Gaussian smearing of the z-valent cation Y, nm; default --ion-smear")
parser.add_argument("--a-int", type=float, default=None, help="smearing of the monomers (fields and segment charges), nm; default the ion smearing a_W/sqrt(pi)")
parser.add_argument("--nbar", type=float, default=None, help="0: mean field (no noise); default (rho0 R0^3 / N)^2 with rho0 = 1/v")
parser.add_argument("--volume", type=float, default=1.0, help="volume v of a charged segment (and of a solvent site), nm^3")
parser.add_argument("--edge", type=float, default=0.033, help="volume fraction at which the outer edge of the film is read (SLD 6.2e-6 of Yu et al.)")
parser.add_argument("--integrator", default="lm")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--platform", default="cuda")
parser.add_argument("--print-every", type=int, default=1, help="measurements between progress lines")
parser.add_argument("--psi-boost", type=float, default=0.0, help="extra mobility of psi in the solution above the brush, zero below --boost-from and full above --boost-to (nm): a fixed, position-dependent mobility, which keeps the stationary distribution, so that the dilute solution relaxes")
parser.add_argument("--boost-from", type=float, default=60.0)
parser.add_argument("--boost-to", type=float, default=70.0)
parser.add_argument("--restart", default=None, help="continue from this checkpoint (fields_<step>.mat) up to --steps")
parser.add_argument("--lateral-bc", default="reflecting", choices=["reflecting", "periodic"], help="boundary of the lateral axes")
parser.add_argument("--start-from", default=None, help="new run (step 0, own seed) from the fields and psi of this checkpoint, "
                    "mirror-tiled laterally up to --lateral")
parser.add_argument("--save-interval", type=int, default=None, help="checkpoint and statistics interval; default --steps (a restart keeps that of its first run)")
args = parser.parse_args()

N, b, v, sigma, chi, lb, a_w = 100, 1.0, args.volume, 0.1, 0.5, 0.7, 0.25
if args.nbar is None:
    args.nbar = ((b * np.sqrt(N)) ** 3 / (v * N)) ** 2      # (rho0 R0^3 / N)^2 with rho0 = 1/v
r0 = b * np.sqrt(N)
molar = 0.6022 * v                  # segments of rho0 per mol/L
smear = (a_w / np.sqrt(np.pi) if args.ion_smear is None else args.ion_smear) / r0   # code smearing of the ions, R0
smears = {"C": smear, "X": smear, "Y": smear if args.cation_smear is None else args.cation_smear / r0}
coupling = 4 * np.pi * lb / v * N * r0 ** 2
z = args.valence
reservoir = {"C": (1.0, args.c_mono * molar), "Y": (float(z), args.rho * molar),
             "X": (-1.0, (args.c_mono + z * args.rho) * molar)}
if args.far:
    c_c, c_y, c_x = (float(t) for t in args.far.split(","))
    reservoir = {"C": (1.0, c_c * molar), "Y": (float(z), c_y * molar), "X": (-1.0, c_x * molar)}
nz, nl = int(round(args.height / args.dx)), int(round(args.lateral / args.dx))
z_grid = (np.arange(nz) + 0.5) * args.dx
phi_brush = sigma * N * v / args.height
graft = np.exp(-0.5 * (z_grid / 0.5) ** 2)


def mu_excess(dx, n=96):
    """One-loop mu_ex per z^2 of each reservoir ion on the grid (spectral k up to pi/dx per axis), code units:
    (1/2N^2)(1/C) int d^3k/(2pi)^3 Gamma_s^2 / (k^2/E + sum_t z_t^2 c_t Gamma_t^2 / N), lengths in R0. With periodic
    lateral axes the lateral integral is the sum over the modes of the box, (1/L^2) sum_k, so that the solution in the
    box holds the reservoir concentrations at one loop."""
    c_ = (v ** -1 * r0 ** 3 / N)
    k = (np.arange(n) + 0.5) / n * np.pi / (dx / r0)
    w = np.pi / (dx / r0) / n
    if args.lateral_bc == "periodic":
        lateral = 2 * np.pi * np.fft.fftfreq(nl, d=dx / r0)
        kx, ky, kz = np.meshgrid(lateral, lateral, k, indexing="ij", sparse=True)
        weight = (w / np.pi) / (args.lateral / r0) ** 2
    else:
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij", sparse=True)
        weight = 8 * w ** 3 / (2 * np.pi) ** 3
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    screening = sum(q ** 2 * c * np.exp(-smears[name] ** 2 * k2) for name, (q, c) in reservoir.items()) / N
    return {name: float(weight * np.sum(np.exp(-smears[name] ** 2 * k2) / (k2 / coupling + screening))
                        / (2 * N ** 2 * c_)) for name in reservoir}


mu_ex = mu_excess(args.dx)
mean_field = args.nbar <= 0


def electrostatics(with_excess):
    """The potential is measured from the Donnan potential psi0 of the homogeneous box, not from the reservoir: at low
    salt the brush sits hundreds of kT per chain from the reservoir and the propagators would overflow. The gauge
    psi -> psi - psi0, mu_s -> mu_s - z_s psi0/N changes nothing else."""
    mu = {name: float(np.log(N * c) + (q ** 2 * mu_ex[name] if with_excess else 0.0)) for name, (q, c) in reservoir.items() if c > 0}
    # The same psi0 for SCFT and CL-FTS (that of the reservoir without the self-energy), so that the SCFT potential is
    # the start of CL-FTS in its own gauge
    ideal = [{"valence": q, "concentration": None, "chemical_potential": float(np.log(N * c))} for q, c in reservoir.values() if c > 0]
    psi0 = homogeneous_potential(-phi_brush, ideal, coupling, 0.0, N)
    ions = {name: {"valence": reservoir[name][0], "smearing_length": smears[name],
                   "chemical_potential": m - reservoir[name][0] * psi0 / N} for name, m in mu.items()}
    return {"charges": {"A": -1.0}, "coupling": coupling, "n_segments": N, "ions": ions}


common = {
    "chain_model": "discrete", "ds": 1.0 / N, "numerical_method": "rs",
    "segment_lengths": {"A": 1.0, "S": 1.0}, "chi_n": {"A,S": chi * N},
    "polymers": [{"volume_fraction": phi_brush, "blocks": [{"type": "A", "length": 1.0, "v": 0, "u": 1}],
                  "initial_conditions": {0: "G"}},
                 {"volume_fraction": 1.0 - phi_brush, "blocks": [{"type": "S", "length": 1.0 / N}]}],
    "smearing": {"type": "gaussian", "a_int": smear if args.a_int is None else args.a_int / r0},
}

# 1. SCFT of the planar brush along z (incompressible, mean-field electrostatics)
start = time.time()
scft_params = dict(copy.deepcopy(common), platform="cpu-fftw", nx=[nz], lx=[args.height / r0], bc=["absorbing", "reflecting"],
                   relax_cell=False, electrostatics=electrostatics(False),
                   optimizer={"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.02, "mix_init": 0.02},
                   max_iter=20000, tolerance=1e-8)
parabola = 3 * np.pi ** 2 / (8 * N ** 2 * b ** 2) * np.maximum(60.0 ** 2 - z_grid ** 2, 0) * N
scft = SCFT(params=scft_params)
result = scft.run(initial_fields={"A": parabola, "S": np.zeros(nz)}, q_init={"G": graft}, return_result=True)
phi_a = np.asarray(result.phi["A"]).real
h_scft = 2 * np.sum(z_grid * phi_a) / np.sum(phi_a)
out = dict(args=vars(args), reservoir={k: c / molar for k, (q, c) in reservoir.items()}, mu_ex=mu_ex,
           z=z_grid.tolist(), scft=dict(converged=bool(result.converged), iterations=int(result.iterations),
                                        height=h_scft, phi_a=phi_a.tolist(), seconds=time.time() - start))
print(f"SCFT: converged {result.converged} after {result.iterations} iterations, h = {h_scft:.3f} nm, "
      f"{time.time() - start:.1f} s", flush=True)
json.dump(out, open(args.out, "w"))

# 2. CL-FTS in 3D, from the SCFT fields. Grafted chains: pass validation without initial_conditions, give q_init
_validate = clfts_module.validate_lfts_params


def validate(params, complex_fields):
    stripped = copy.deepcopy(params)
    for polymer in stripped["polymers"]:
        polymer.pop("initial_conditions", None)
    return _validate(stripped, complex_fields=complex_fields)


q_graft = np.broadcast_to(graft, (nl, nl, nz)).ravel().astype(np.complex128)
clfts_module.validate_lfts_params = validate
clfts_module.solve_chains = lambda sim, w_input, q_init=None: chains.solve_chains(sim, w_input, q_init={"G": q_graft})

params = dict(copy.deepcopy(common), platform=args.platform, nx=[nl, nl, nz],
              lx=[args.lateral / r0, args.lateral / r0, args.height / r0],
              bc=[args.lateral_bc] * 4 + ["absorbing", "reflecting"],
              zeta_n=args.zeta, dynamic_stabilization=args.alpha, verbose_level=1,
              electrostatics=dict(electrostatics(not (mean_field or args.freeze_psi)),
                                  psi_dt_scaling=1e-12 if args.freeze_psi else args.psi_scale),
              langevin={"max_step": args.steps, "dt": args.dt, "nbar": 1e30 if mean_field else args.nbar,
                        "integrator": args.integrator},
              recording={"dir": os.path.splitext(args.out)[0] + "_data", "checkpoint_interval": args.save_interval or args.steps,
                         "sample_interval": args.interval, "statistics_save_interval": args.save_interval or args.steps})


class Profiles:
    """h = 2<z> of each measurement and the lateral means of phi_A and of the ions after the equilibration."""
    def __init__(self):
        self.interval = args.interval
        self.steps, self.heights, self.hamiltonian, self.charge, self.count = [], [], [], [], 0
        self.edges = []        # outer edge of the film, where the lateral mean of phi_A falls to args.edge
        self.far = []          # net charge per nm^2 at z > 70 nm and the mean C, X densities there
        self.sums = {}

    def __call__(self, sample):
        phi = np.asarray(sample["phi"]["A"]).reshape(nl, nl, nz).mean(axis=(0, 1))
        self.steps.append(int(sample["step"]))
        self.heights.append(float(2 * np.sum(z_grid * phi.real) / np.sum(phi.real)))
        inside = np.nonzero(phi.real >= args.edge)[0]
        if len(inside) and inside[-1] + 1 < nz:
            i = inside[-1]
            self.edges.append(float(z_grid[i] + (phi.real[i] - args.edge) / (phi.real[i] - phi.real[i + 1]) * args.dx))
        else:
            self.edges.append(float(z_grid[inside[-1]]) if len(inside) else 0.0)
        self.hamiltonian.append([float(np.real(sample["hamiltonian"])), float(np.imag(sample["hamiltonian"]))])
        if sample["step"] > args.equilibration:
            densities, _ = sim.electrostatics.ion_densities()
            profiles = {"A": phi}
            for ion, c in zip(sim.electrostatics.ions, densities):
                profiles[ion["name"]] = np.asarray(c).reshape(nl, nl, nz).mean(axis=(0, 1))
            for key, value in profiles.items():
                self.sums[key] = self.sums.get(key, 0) + value
            self.count += 1
        densities, _ = sim.electrostatics.ion_densities()
        charge = sum(ion["valence"] * np.mean(c) for ion, c in zip(sim.electrostatics.ions, densities)) - phi_brush
        self.charge.append(float(np.real(charge)) * args.height)      # net charge per nm^2 of wall
        lateral = {ion["name"]: np.real(np.asarray(c).reshape(nl, nl, nz).mean(axis=(0, 1))) for ion, c in zip(sim.electrostatics.ions, densities)}
        net = sum(ion["valence"] * lateral[ion["name"]] for ion in sim.electrostatics.ions) - phi.real
        far = z_grid > 70.0
        self.far.append([float(np.sum(net[far]) * args.dx)] + [float(lateral[k][far].mean()) for k in ("C", "X") if k in lateral])
        if len(self.steps) % args.print_every == 0:
            a_abs = np.abs(np.asarray(sample["phi"]["A"])).reshape(nl, nl, nz)
            s_abs = np.abs(np.asarray(sample["phi"]["S"])).reshape(nl, nl, nz)
            ions_abs = max(float(np.max(np.abs(c))) for c in densities)
            w_abs = np.max(np.abs(np.asarray(sample["w_aux"])), axis=1)
            print(f"    max |phi_A| {a_abs.max():.3g} at z = {z_grid[np.unravel_index(a_abs.argmax(), a_abs.shape)[2]]:.1f} nm, "
                  f"max |phi_S| {s_abs.max():.3g} at z = {z_grid[np.unravel_index(s_abs.argmax(), s_abs.shape)[2]]:.1f} nm, "
                  f"max |c_ion| {ions_abs:.3g}, max |W-| {w_abs[0]:.3g}, max |W+| {w_abs[1]:.3g}, max |psi| {np.max(np.abs(sim.electrostatics.psi)):.3g}", flush=True)
            print(f"  step {sample['step']}: h = {self.heights[-1]:.3f} nm, H = {self.hamiltonian[-1][0]:.4f}"
                  f"{self.hamiltonian[-1][1]:+.2e}i, net charge {self.charge[-1]:+.3f}/nm2, above 70 nm {self.far[-1][0]:+.3f}/nm2, drift {sample['drift_max']:.3g}, {sample['step_seconds']:.3f} s/step", flush=True)


# The ions screen psi with the response D = sum_s z_s^2 c_s / N (code units), so the neutrality mode of psi relaxes
# at the rate s D: with E = 4 pi l_B rho0 N R0^2 ~ 9e4 that takes ~1e6 steps at s ~ 1. With --psi-relax, the exact
# (Ornstein-Uhlenbeck) step of psi takes the rate s (k^2/E + D0) with a uniform D0, and the step adds D0 psi to the
# charge density, so the linear part of the ion response is integrated exactly and only the rest stays explicit.
# Any positive mobility keeps the stationary distribution.
d0 = (phi_brush + sum(q ** 2 * c for q, c in reservoir.values())) / N
if args.psi_relax > 0:
    params["electrostatics"]["psi_dt_scaling"] = args.psi_relax / (args.dt * d0)
    print(f"psi preconditioned: D0 = {d0:.3e}, psi_dt_scaling = {params['electrostatics']['psi_dt_scaling']:.4g}", flush=True)
out["psi"] = dict(d0=d0, psi_dt_scaling=params["electrostatics"]["psi_dt_scaling"], preconditioned=args.psi_relax > 0)
class PsiBoostNoise:
    """The extra mobility dM(z) of psi in the solution (attached to the step of psi below) and its own noise. It is
    registered as a callback that does nothing, so that the checkpoint keeps its random state: the state of the
    generator before the last draw, from which load_state_dict draws that noise again. This restores both the
    previous noise of the averaged (Leimkuhler-Matthews) step and the generator."""
    def __init__(self):
        ramp = np.clip((z_grid - args.boost_from) / (args.boost_to - args.boost_from), 0.0, 1.0)
        self.extra = np.broadcast_to(args.psi_boost * ramp, (nl, nl, nz)).ravel().copy()
        self.interval = args.interval      # due with the profiles, so it adds no snapshot of the fields
        self.settings = dict(mobility=args.psi_boost, start=args.boost_from, full=args.boost_to, seed=args.seed + 2000)
        self.sigma = None                  # standard deviation of the noise of psi, set once psi exists
        self.load_state_dict({"before_last_draw": None})

    def __call__(self, sample):
        pass

    def state_dict(self):
        return {"before_last_draw": self.before_last_draw}

    def load_state_dict(self, state):
        self.random = np.random.default_rng(self.settings["seed"])
        self.before_last_draw, self.previous = state["before_last_draw"], np.zeros(self.extra.size)
        if self.before_last_draw is not None:
            self.random.bit_generator.state = self.before_last_draw
            self.previous = self.random.normal(0.0, self.sigma, self.extra.size)

    def draw(self):
        """Noise of this step, the sum of this draw and the previous one."""
        self.before_last_draw = self.random.bit_generator.state
        xi = self.random.normal(0.0, self.sigma, self.extra.size)
        noise, self.previous = xi + self.previous, xi
        return noise


def checkpoint_callbacks(path):
    from scipy.io import loadmat
    data = loadmat(path, variable_names=["checkpoint_settings_json"], squeeze_me=True)
    return json.loads(str(data["checkpoint_settings_json"]))["extensions"]["callbacks"]


profiles = Profiles()
callbacks = {"profiles": profiles}
boost = None            # the boost noise kept in the checkpoints
if args.psi_boost > 0 and not (mean_field or args.freeze_psi):
    if args.restart and "psi_boost" not in checkpoint_callbacks(args.restart):
        print("the checkpoint does not keep the state of the psi boost noise: it starts again from its seed", flush=True)
    else:
        boost = PsiBoostNoise()
        callbacks["psi_boost"] = boost
sim = CLFTS(params=params, random_seed=args.seed, callbacks=callbacks)
if args.psi_relax > 0:
    es = sim.electrostatics
    scale, dt = es.psi_dt_scaling, args.dt
    rate = es.screened_k_sq / es.coupling + d0
    es._decay = np.exp(-rate * dt * scale)
    es._response = -np.expm1(-rate * dt * scale) / rate
    es._noise = np.sqrt(-np.expm1(-2 * rate * dt * scale) / (2 * rate * dt))
    es._averaged_noise = np.sqrt(-np.expm1(-rate * dt * scale) / (4 * rate * dt))
    _step = es.step
    es.step = lambda c: _step(c + d0 * es.psi)
if args.mobility:
    # A mobility per field changes the dynamics, not the stationary distribution
    sim.dt_scaling = np.array([float(m) for m in args.mobility.split(",")])
    print("mobilities of the auxiliary fields:", sim.dt_scaling, flush=True)
if args.uniform_relax > 0 and not args.freeze_psi:
    # The uniform mode of psi is the Donnan potential: its force is the net charge of the box, and the chains do not
    # respond to it (their charge is fixed), only the ions, with the response d0. A mobility of its own keeps the
    # stationary distribution; the exact step at k = 0 (rate 0) is psi_0 += s0 dt c_0 + noise of amplitude sqrt(s0).
    es = sim.electrostatics
    s0 = args.uniform_relax / (args.dt * d0)
    zero = (0,) * es._response.ndim
    es._response[zero] = args.dt * s0
    es._noise[zero] = np.sqrt(s0)
    es._averaged_noise[zero] = np.sqrt(s0) / 2
    out["psi"]["uniform_mobility"] = s0
    print(f"uniform mode of psi: mobility {s0:.4g} (others {es.psi_dt_scaling:.4g})", flush=True)
if args.psi_boost > 0 and not (mean_field or args.freeze_psi):
    # In the dilute solution the ions respond to psi with D = sum z^2 c / N ~ 7e-5 (6 mM), so psi relaxes there at
    # the rate s dt D per step: ~5e6 steps at s = 3. An extra mobility dM(z), zero in the brush, is added as an explicit
    # real-space step with its own averaged (Leimkuhler-Matthews) noise from the force of the same state. With the
    # k-space step it is one LM step of mobility K = K_k + dt dM, exact for the Gaussian part like the k-space step.
    es = sim.electrostatics
    noise_source = boost or PsiBoostNoise()
    noise_source.sigma = es.sigma
    extra = noise_source.extra
    _plain_step = es.step

    def boosted_step(c):
        force = es.force(c)
        _plain_step(c)
        noise = noise_source.draw()
        es._set_psi_k(es._fourier.forward(es.psi + args.dt * extra * force + 1j * np.sqrt(extra) / 2 * noise))
    es.step = boosted_step
    out["psi"]["boost"] = dict(mobility=args.psi_boost, start=args.boost_from, full=args.boost_to,
                               checkpointed=boost is not None)
    print(f"psi mobility boosted by {args.psi_boost:g} above z = {args.boost_from:g}-{args.boost_to:g} nm", flush=True)
w0 = {name: np.broadcast_to(np.asarray(result.w[name]).real, (nl, nl, nz)).ravel().copy() for name in ("A", "S")}
psi0 = np.broadcast_to(np.asarray(result.psi).real, (nl, nl, nz)).ravel().astype(np.complex128)
if args.start_from:
    # Fields and psi of a checkpoint; a lateral axis shorter than this box is repeated with its mirror image, which is
    # how a reflecting box continues (and a periodic box of twice its size)
    from scipy.io import loadmat
    data = loadmat(args.start_from, squeeze_me=True)
    shape = tuple(int(n) for n in data["nx"])

    def tiled(values):
        grid = np.reshape(values, shape)
        for axis in (0, 1):
            while grid.shape[axis] < nl:
                grid = np.concatenate([grid, np.flip(grid, axis=axis)], axis=axis)
        assert grid.shape == (nl, nl, nz), f"cannot tile {shape} to {(nl, nl, nz)}"
        return grid.ravel().copy()
    w0 = {name: tiled(data[f"w_{name}_real"] + 1j * data[f"w_{name}_imag"]) for name in ("A", "S")}
    psi0 = tiled(data["psi_real"] + 1j * data["psi_imag"])
    out["start_from"] = args.start_from
    print(f"start from {args.start_from} ({shape} -> {(nl, nl, nz)})", flush=True)
elif not (mean_field or args.freeze_psi):
    # Start psi with the fluctuations of its linear response, so that the ions feel their self-energy from the first
    # step: their chemical potentials hold +z^2 mu_ex, which a smooth psi would not compensate (Y3+ would gain e^10).
    # The exact step of psi keeps the variance noise^2 / (2 dt (k^2/E + D)) per mode; D = d0 for the ions.
    es = sim.electrostatics
    rate = es.screened_k_sq / es.coupling + d0
    xi_k = es._fourier.forward(np.random.default_rng(args.seed + 1000).normal(0.0, es.sigma, psi0.size))
    psi0 = psi0 + 1j * es._fourier.backward(np.sqrt(1.0 / (2 * args.dt * rate)) * xi_k)
start = time.time()
status = None
try:
    status = sim.continue_run(args.restart) if args.restart else sim.run(initial_fields=w0, initial_psi=psi0)
except Exception as exc:
    status = {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
    print("CL-FTS failed:", status["reason"], flush=True)
mean = {k: (v / profiles.count) for k, v in profiles.sums.items()} if profiles.count else {}
out["clfts"] = dict(status=status.get("status"), reason=status.get("reason"), seconds=time.time() - start,
                    steps=profiles.steps, heights=profiles.heights, edges=profiles.edges, charge=profiles.charge, far=profiles.far, hamiltonian=profiles.hamiltonian,
                    samples=profiles.count, profiles={k: [np.real(v).tolist(), np.imag(v).tolist()] for k, v in mean.items()},
                    height=(float(2 * np.sum(z_grid * mean["A"].real) / np.sum(mean["A"].real)) if mean else None))
json.dump(out, open(args.out, "w"))
print("RUN_DONE", out["clfts"]["status"], out["clfts"]["height"], flush=True)
