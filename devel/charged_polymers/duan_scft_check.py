"""Reproduce the numbers and figures of reports/DUAN_SCFT_CHECK.md.

Run from this folder. Each line is one configuration; anything over a minute belongs in a job.

  python duan_scft_check.py analytic                       # S33, Eq. 7, S39, S40, f_ion, Gamma = 0, Eq. 6
  python duan_scft_check.py correlation                    # u^ex, S1(k), two-particle factor 2, RPA pressure
  python duan_scft_check.py layer 3 2000 corr [--out DIR]  # one z, L, closure (corr or mf) of the uniform layer
  python duan_scft_check.py tables [--layer-dir DIR]       # Figs. 2-3 against the equations, box-length tables
  python duan_scft_check.py figures [--layer-dir DIR]      # the eight figures of the report
  python duan_scft_check.py saved --scan-dir DIR           # independent check of the archived SCFT solutions
  python duan_scft_check.py code --scan-dir DIR            # chain floor and nonlocal quadrature of models/duan_scft.py
  python duan_scft_check.py scft 2 0.001 2000 WARM OUT_DIR # one extended-box SCFT point, WARM = a profiles.npz
  python duan_scft_check.py scft 2 0.001 200 WARM OUT_DIR --grounded  # real box of length L with psi(L) = 0
  python duan_scft_check.py walk 2 0.001 WARM OUT_DIR --values 200,300,450  # box lengths in steps, each from the last solution
  python duan_scft_check.py walk 2 0.001 WARM OUT_DIR --values 0.03,0.02 --background  # fixed 1:1 background salt in M, no C
  python duan_scft_check.py walk 3 1e-06 WARM OUT_DIR --values 0.006 --background --chain 730 --stretch 7.3 --box 800 --dz 0.25

reports/duan_scft_check_data.json holds what cannot be recomputed from this folder alone: pixel readings of Figs. 2-3, the
archived L = 200 nm scan and the extended-box SCFT results, plus the stored layer scans used by default. --layer-dir
replaces those with the files written by `layer --out`. --scan-dir is the archived scan with one profiles.npz per run.

Units: nm, kT. Polymer charge -1 per monomer. Paper: Duan, Agrawal & Wang, Phys. Rev. Lett. 134, 048101 (2025).
"""
import argparse
import json
from pathlib import Path
import signal
import time
from types import MethodType
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, fsolve
from scipy.special import erfcx, logsumexp

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "reports/duan_scft_check_data.json"
FIGURE_DIR = HERE / "reports/figures"
lB, a, b, N, sigma, chi, MOLAR = 0.7, 0.25, 1.0, 100, 0.1, 0.5, 0.6022
LS = [200, 2000, 20000, 200000]
SCAN_H0 = 50.91  # 2<z> of the salt-free brush in the archived scan, nm
# Critical concentrations marked on Fig. 2, in M, read from the image
MARK = {1: dict(s=1.35e-2, sss=3.5), 2: dict(s=6.9e-6, ss=8.9e-4, sss=1.26), 3: dict(s=1.8e-7, ss=2.75e-5, sss=1.04)}


def u(x):
    return 1 - x * erfcx(x / np.sqrt(np.pi))


def du(x):
    e = erfcx(x / np.sqrt(np.pi))
    return -(1 + 2 * x * x / np.pi) * e + 2 * x / np.pi


def w(I):
    """Self-energy per z^2 at I = sum z^2 rho (S33 with eps = 1/(4 pi lB))."""
    return lB / (2 * a) * u(a * np.sqrt(4 * np.pi * lB * I))


def f_corr(I):
    """Charging integral: int_0^I w dI'; substitution I' = t^2 removes the sqrt singularity at 0."""
    if I <= 0:
        return 0.0
    return quad(lambda t: 2 * t * w(t * t), 0, np.sqrt(I), epsabs=1e-12, epsrel=1e-12, limit=200)[0]


def gamma_eq7(z, rb):
    k = np.sqrt(4 * np.pi * lB * z * (z + 1) * rb)
    R = -z * (z + 1) * k * lB / 4 * du(a * k)
    return (1 - R) / (z + 1 - R)


class Slab:
    """Uniform layer of fixed polymer density x in Donnan equilibrium with a z:1 reservoir, not linearised,
    so the SI expansions can be tested against it. No canonical counterions."""

    def __init__(self, z, rho_b, corr=True):
        self.z, self.rb, self.corr = z, rho_b, corr
        self.Ib = z * z * rho_b + z * rho_b
        self.wb = w(self.Ib) if corr else 0.0

    def densities(self, psi, wval):
        dw = wval - self.wb if self.corr else 0.0
        z = self.z
        return self.rb * np.exp(-z * psi - z * z * dw), z * self.rb * np.exp(psi - dw)

    def solve(self, x):
        """Neutrality + self-consistent ionic strength, unknowns (psi, ln I)."""
        z = self.z

        def eq(y):
            psi, lnI = y
            I = np.exp(lnI)
            rp, rm = self.densities(psi, w(I) if self.corr else 0.0)
            return [(z * rp - rm - x) / max(x + self.Ib, 1e-30), (z * z * rp + rm + x) / I - 1.0]

        starts = []
        for psi0 in [0.0, -0.5, -1, -2, -4, -8, 0.5, 1, 2]:
            starts.append((psi0, np.log(self.Ib + x + 1e-30)))
            starts.append((psi0, np.log(self.Ib + (1 + z) * x + 1e-30)))
        for s in starts:
            y, info, ok, _ = fsolve(eq, s, full_output=True, xtol=1e-12)
            if ok == 1 and max(abs(np.array(eq(y)))) < 1e-9:
                return y
        raise RuntimeError(f"no solution z={z} rb={self.rb} x={x}")

    def pressure(self, x):
        """S39 for the bulk-referenced ion pressure; P = sum rho + sum rho_K u_K - f_c."""
        psi, lnI = self.solve(x)
        I = np.exp(lnI)
        wv = w(I) if self.corr else 0.0
        rp, rm = self.densities(psi, wv)
        P = rp + rm
        if self.corr:
            P += I * wv - f_corr(I)
        Pb = self.rb + self.z * self.rb
        if self.corr:
            Pb += self.Ib * self.wb - f_corr(self.Ib)
        return P - Pb, rp, rm, psi, I

    def grand(self, x):
        """Grand potential density g(x) = f - mu_+ rho_+ - mu_- rho_-; independent of S39."""
        psi, lnI = self.solve(x)
        I = np.exp(lnI)
        wv = w(I) if self.corr else 0.0
        rp, rm = self.densities(psi, wv)
        z, wb = self.z, self.wb
        mup = np.log(self.rb) + z * z * wb
        mum = np.log(z * self.rb) + wb
        f = rp * (np.log(rp) - 1) + rm * (np.log(rm) - 1) - mup * rp - mum * rm
        if self.corr:
            f += f_corr(I)
        return f


def analytic():
    print("== T1 S33: u(x) against the Fourier integral, Gaussian charge h(k)^2 = exp(-k^2 a^2/pi)")
    for x in [0.01, 0.3, 1.0, 3.0]:
        kap = x / a
        val = quad(lambda k: np.exp(-(a * k) ** 2 / np.pi) * k * k / (k * k + kap * kap), 0, np.inf)[0]
        print(f"  x={x:5.2f} fourier={2 * a * val / np.pi:.10f} u(x)={u(x):.10f}")

    print("\n== T2 S38/Eq.7: Gamma from exact Donnan with x << rho_b")
    for z, rm_ in [(1, 0.3), (2, 0.1), (3, 0.03), (3, 0.3)]:
        rb = rm_ * MOLAR
        x = 1e-4 * rb
        rp = Slab(z, rb).pressure(x)[1]
        print(f"  z={z} rho_b={rm_:5.2f} M  exact Gamma={1 - z * (rp - rb) / x:+.6f}  Eq.7={gamma_eq7(z, rb):+.6f}")

    print("\n== T3 S39/S40: f_ion. Thermodynamic pressure x g'(x) - g(x) vs S39 vs S40")
    for z, rm_ in [(1, 0.3), (3, 0.3), (2, 0.01), (3, 0.01)]:
        rb = rm_ * MOLAR
        for corr in [False, True]:
            s = Slab(z, rb, corr=corr)
            x = 0.02 * rb
            d = 1e-3 * x
            g0 = s.grand(0.0 + 1e-12 * rb)
            dg = (s.grand(x + d) - s.grand(x - d)) / (2 * d)
            thermo = x * dg - s.grand(x) + g0
            P = s.pressure(x)[0]
            G = (1 / (z + 1)) if not corr else gamma_eq7(z, rb)
            S40 = x * x * G / (z * rb)
            print(f"  z={z} rho_b={rm_:5.2f} M corr={corr!s:5} S39={P:+.6e} thermo={thermo:+.6e} "
                  f"S40={S40:+.6e} exact/S40={P / S40:.4f}")

    print("\n== T4 main text: dilute-limit correlation pressure, f_ion = sum drho - c kappa^3")
    for I in [1e-6, 1e-5]:
        k = np.sqrt(4 * np.pi * lB * I)
        P = I * (w(I) - w(0)) - (f_corr(I) - I * w(0))
        print(f"  kappa={k:.4e} -P_corr/kappa^3={-P / k ** 3:.5f}   1/(24 pi)={1 / (24 * np.pi):.5f}   pi/24={np.pi / 24:.5f}")

    print("\n== T5 Gamma = 0 (paper: 'rho_b** is determined at Gamma=0 through Eq. (7)'), Gamma minimum, charge reversal")
    grid = np.geomspace(1e-7, 50, 4000) * MOLAR
    for z in [2, 3]:
        G = np.array([gamma_eq7(z, r) for r in grid])
        zero = brentq(lambda r: gamma_eq7(z, r), grid[0], grid[np.argmin(G)]) / MOLAR
        print(f"  z={z}: Eq.7 Gamma=0 at {zero:.3e} M; Fig.2 rho** reads {MARK[z]['ss']:.2e} M; ratio {zero / MARK[z]['ss']:.0f}; "
              f"min Gamma={G.min():+.3f} at {grid[np.argmin(G)] / MOLAR:.3f} M")
        # Eq. 7 value at the concentration that Fig. 2 calls rho**
        print(f"       Gamma(Fig.2 rho**) = {gamma_eq7(z, MARK[z]['ss'] * MOLAR):+.3f}")

    print("\n== T6 Eq.6: onset ratios against Fig. 2")
    for z in [2, 3]:
        print(f"  z=1/z={z}: Fig.2 {MARK[1]['s'] / MARK[z]['s']:.3g}  Eq.6 corr {(z * (z + 1) / 2) ** 3:.3g}  Eq.6 MF {z * (z + 1) / 2:.3g}")

    print("\n== T7 S34 rho_b >> rho_P and |drho| << rho_b at the regime boundaries")
    print("  polymer density in the collapsed brush: 0.75/v = %.3f nm^-3 = %.2f M" % (0.75, 0.75 / MOLAR))
    for lab, z, r in [("rho*  z=3", 3, 1.8e-7), ("rho** z=3", 3, 2.75e-5), ("valley z=3", 3, 3e-3), ("rho** z=2", 2, 8.9e-4)]:
        print(f"  {lab}: rho_b/rho_P = {r * MOLAR / 0.75:.2e}")


def form_factor(k):
    """S1(k) = (1/N) sum_ij exp(-k^2 |i-j| b^2/6), Gaussian chain."""
    q = np.exp(-(k * b) ** 2 / 6)
    return 1 + 2 * np.sum((1 - np.arange(1, N) / N) * q ** np.arange(1, N))


def gaussian_self(I_ion, rho_P, S=True):
    """(1/2) int d3k/(2pi)^3 h^2(k) G(k), with eps = 1/(4 pi lB); Born constant included."""
    def f(lk):
        k = np.exp(lk)
        s = form_factor(k) if S else 1.0
        return k * k * k * np.exp(-(k * a) ** 2 / np.pi) / (k * k / (4 * np.pi * lB) + I_ion + rho_P * s)
    return 0.5 / (2 * np.pi ** 2) * quad(f, np.log(1e-4), np.log(60.0 / a), limit=400)[0]


def uex_mid(kappa):
    """Eq. 3b for the central monomer: (1/2) sum_{j != i} <G(r_ij)>, unperturbed Gaussian chain."""
    i = N // 2
    tot = 0.0
    for j in range(N):
        if j != i:
            m = abs(i - j)
            tot += quad(lambda k: k * k * 4 * np.pi * lB / (k * k + kappa ** 2) * np.exp(-k * k * m * b * b / 6) / (2 * np.pi ** 2), 0, np.inf)[0]
    return 0.5 * tot


def corr_f(Iion, rp, S):
    def g(lk):
        k = np.exp(lk)
        s = form_factor(k) if S else 1.0
        h2 = np.exp(-(k * a) ** 2 / np.pi)
        t = 4 * np.pi * lB * (Iion + rp * s) * h2 / (k * k)
        return k ** 3 * (np.log1p(t) - t)
    return 0.5 / (2 * np.pi ** 2) * quad(g, np.log(1e-4), np.log(60.0 / a), limit=600)[0]


def rpa_pressure(Iion, rp, S):
    """Pressure at fixed composition ratios: P = sum_K rho_K d f/d rho_K - f; derivatives by central difference."""
    e = 1e-4
    f0 = corr_f(Iion, rp, S)
    dI = (corr_f(Iion * (1 + e), rp, S) - corr_f(Iion * (1 - e), rp, S)) / (2 * e)
    dp = (corr_f(Iion, rp * (1 + e), S) - corr_f(Iion, rp * (1 - e), S)) / (2 * e)
    return dI + dp - f0, dp / rp


def correlation():
    print("== (a) u_P^ex (central monomer) vs u_P^0 change relative to kappa_b -> 0")
    print(" kappa[1/nm]  1/kappa  u^0-u^0(0)   u^ex(Eq.3b)  ratio |u^ex/du^0|")
    for kap in [0.3, 0.5, 1.0, 2.0, 4.0]:
        d0 = lB / (2 * a) * (u(a * kap) - 1)
        ue = uex_mid(kap)
        print(f"  {kap:8.2f} {1 / kap:8.2f} {d0:10.3f} {ue:12.3f} {abs(ue / d0):10.2f}")

    print("\n== (b) S1(k) at the Debye wave number, and I_ex effect on the monomer self-energy")
    for k in [0.3, 1.0, 2.0, 4.0]:
        print(f"  k={k:4.1f}  S1(k)={form_factor(k):7.2f}")
    print("  state: uniform, polymer rho_P, neutralising z-valent ions plus salt rho_b")
    print("  z rho_P  rho_b   kappa_local  u_P local(S=1)  u_P with S1(k)  diff   ion u_+ local  ion u_+ with S1  diff")
    for z, rp, rb in [(1, 0.2, 0.18), (2, 0.5, 0.02), (3, 0.7, 0.002), (3, 0.3, 0.2)]:
        Iion = z * z * (rb + rp / z) + z * rb       # cations neutralise the chain, anions z rho_b
        k = np.sqrt(4 * np.pi * lB * (Iion + rp))
        v0 = gaussian_self(Iion, rp, False)
        v1 = gaussian_self(Iion, rp, True)
        print(f"  {z} {rp:4.2f} {rb:6.3f} {k:9.2f} {v0:13.3f} {v1:13.3f} {v1 - v0:8.3f} {z * z * v0:12.3f} {z * z * v1:12.3f} {z * z * (v1 - v0):8.3f}")

    print("\n== (c) density response to a weak pair potential, two independent particles on a 1D ring")
    n = 400
    x = np.arange(n) * 2 * np.pi / n
    ext1 = np.exp(np.cos(x)); ext1 /= ext1.sum()
    ext2 = np.exp(np.cos(x - 1.0)); ext2 /= ext2.sum()
    eps = 1e-4
    U = eps * np.exp(np.cos(x[:, None] - x[None, :]) * 2)
    # exact: marginal of particle 1 under p1 p2 exp(-U)
    joint = ext1[:, None] * ext2[None, :] * np.exp(-U)
    exact = joint.sum(1) / joint.sum() - ext1
    # paper: one-body u_1(x) = (1/2) <U(x, R_2)>
    u1 = 0.5 * (U * ext2[None, :]).sum(1)
    paper = ext1 * np.exp(-u1); paper = paper / paper.sum() - ext1
    print(f"  max|exact drho|={abs(exact).max():.4e}  max|paper drho|={abs(paper).max():.4e}  ratio exact/paper={np.dot(exact, paper) / np.dot(paper, paper):.4f}")

    print("\n== (e) correlation pressure of the layer: local form of the paper against the RPA with the Gaussian form factor")
    print("   f = (1/2) int d3k/(2 pi)^3 [ln(1 + 4 pi lB D h2/k^2) - 4 pi lB D h2/k^2]; D = I_ion + rho_P S1(k), or S1=1")
    print("  z rho_P rho_b   P_corr(S=1)  P_corr(S1)   mu_P(S=1)  mu_P(S1)")
    for z, rp, rb in [(1, 0.2, 0.18), (2, 0.5, 0.02), (3, 0.35, 0.002), (3, 0.7, 0.002), (3, 0.7, 0.06)]:
        Iion = z * z * (rb + rp / z) + z * rb
        p0, m0 = rpa_pressure(Iion, rp, False)
        p1, m1 = rpa_pressure(Iion, rp, True)
        print(f"  {z} {rp:4.2f} {rb:6.3f} {p0:12.4f} {p1:11.4f} {m0:10.3f} {m1:10.3f}")


lnI = np.linspace(np.log(1e-18), np.log(120.0), 2400)
I_tab = np.exp(lnI)
f_tab = cumulative_trapezoid(w(I_tab) * I_tab, lnI, initial=0.0) + I_tab[0] * w(I_tab[0])


def fc(I):
    return np.interp(np.log(I), lnI, f_tab)


class Sys:
    """Uniform-layer force balance with the paper's self-energy in the SI semicanonical ensemble.

    Regions: reservoir (z:1 salt rho_b only), far part of the box (salt + released counterions), brush layer of
    thickness h. n_C = sigma N is fixed in a box of length L. Balance: mixing + ion pressure difference = elastic.
    h comes from a uniform layer, so it is a model of the trend; positions of onsets are what is compared.
    """

    def __init__(self, z, rho_b, corr=True, L=200.0):
        self.z, self.rb, self.corr, self.L = z, rho_b, corr, L
        self.Ib = z * (z + 1) * rho_b
        self.wb = w(self.Ib) if corr and rho_b > 0 else (w(1e-12) if corr else 0.0)

    def psi_of(self, I, x, lamC):
        """Vector bisection of neutrality at given ionic strengths; charge is monotonic in psi."""
        z, rb = self.z, self.rb
        dw = (w(I) - self.wb) if self.corr else np.zeros_like(I)
        lo, hi = np.full_like(I, -80.0), np.full_like(I, 80.0)
        for _ in range(48):
            m = 0.5 * (lo + hi)
            Q = z * rb * np.exp(np.clip(-z * m - z * z * dw, -600, 600)) + lamC * np.exp(np.clip(-m - dw, -600, 600)) \
                - z * rb * np.exp(np.clip(m - dw, -600, 600)) - x
            lo = np.where(Q > 0, m, lo)
            hi = np.where(Q > 0, hi, m)
        return 0.5 * (lo + hi), dw

    def region(self, x, lamC):
        """All self-consistent roots (psi, I); lowest grand potential wins."""
        z, rb = self.z, self.rb
        grid = np.geomspace(1e-17, 100.0, 500)
        psi, dw = self.psi_of(grid, x, lamC)
        rp = rb * np.exp(-z * psi - z * z * dw)
        rm = z * rb * np.exp(psi - dw)
        rc = lamC * np.exp(-psi - dw)
        G = np.log((z * z * rp + rm + rc + x) / grid)
        idx = np.flatnonzero(G[:-1] * G[1:] <= 0)
        best = None
        for i in idx:
            t = G[i] / (G[i] - G[i + 1])
            I = np.exp(np.log(grid[i]) + t * np.log(grid[i + 1] / grid[i]))
            for _ in range(3):
                p, d = self.psi_of(np.array([I]), x, lamC)
                rp1, rm1, rc1 = rb * np.exp(-z * p[0] - z * z * d[0]), z * rb * np.exp(p[0] - d[0]), lamC * np.exp(-p[0] - d[0])
                I = z * z * rp1 + rm1 + rc1 + x
            wv = w(I) if self.corr else 0.0
            om = -(rp1 + rm1 + rc1) - (I * wv - fc(I) if self.corr else 0.0)
            if best is None or om < best[0]:
                best = (om, p[0], I, rp1, rm1, rc1, wv)
        if best is None:
            raise RuntimeError("no root")
        return best

    def pressure(self, x, lamC):
        om, p, I, rp, rm, rc, wv = self.region(x, lamC)
        return -om, p, I, rp, rm, rc

    def far(self, c):
        """Far part of the box at given counterion density c; returns the C fugacity and pressure."""
        z, rb = self.z, self.rb
        grid = np.geomspace(1e-17, 100.0, 500)
        dw = (w(grid) - self.wb) if self.corr else np.zeros_like(grid)
        lo, hi = np.full_like(grid, -80.0), np.full_like(grid, 80.0)
        for _ in range(48):
            m = 0.5 * (lo + hi)
            Q = z * rb * np.exp(-z * m - z * z * dw) + c - z * rb * np.exp(m - dw)
            lo = np.where(Q > 0, m, lo)
            hi = np.where(Q > 0, hi, m)
        psi = 0.5 * (lo + hi)
        rp, rm = rb * np.exp(-z * psi - z * z * dw), z * rb * np.exp(psi - dw)
        G = np.log((z * z * rp + rm + c) / grid)
        i = np.flatnonzero(G[:-1] * G[1:] <= 0)[0]        # far region is dilute: unique root at low I
        t = G[i] / (G[i] - G[i + 1])
        I = np.exp(np.log(grid[i]) + t * np.log(grid[i + 1] / grid[i]))
        for _ in range(4):
            d = (w(np.array([I]))[0] - self.wb) if self.corr else 0.0
            l_, h_ = -80.0, 80.0
            for _ in range(60):
                m = 0.5 * (l_ + h_)
                if z * rb * np.exp(-z * m - z * z * d) + c - z * rb * np.exp(m - d) > 0:
                    l_ = m
                else:
                    h_ = m
            rp1, rm1 = rb * np.exp(-z * m - z * z * d), z * rb * np.exp(m - d)
            I = z * z * rp1 + rm1 + c
        wv = w(I) if self.corr else 0.0
        P = rp1 + rm1 + c + (I * wv - fc(I) if self.corr else 0.0)
        return c * np.exp(m + d), P, m, I

    def lnc_for(self, hval, x):
        nC = sigma * N

        def excess(lc):
            c = np.exp(lc)
            lam = self.far(c)[0]
            return self.region(x, lam)[5] * hval + c * (self.L - hval) - nC
        hi = np.log(nC / (self.L - hval))
        lo = hi - 150.0
        c0 = getattr(self, "_lc", None)
        if c0 is not None:
            for d in (0.3, 2.0, 12.0):
                a_, b_ = c0 - d, min(c0 + d, hi)
                if a_ < b_ and excess(a_) < 0 < excess(b_):
                    lo, hi = a_, b_
                    break
        self._lc = brentq(excess, lo, hi, xtol=1e-7, rtol=1e-9, maxiter=200)
        return self._lc

    def balance(self, phi):
        hval = sigma * N / phi
        lc = self.lnc_for(hval, phi)
        lam, Pfar, psif, If = self.far(np.exp(lc))
        Pb = self.pressure(phi, lam)
        mix = -np.log1p(-phi) - phi - chi * phi * phi
        return mix + Pb[0] - Pfar - 3 * sigma * hval / (N * b * b), Pb, np.exp(lc), lam, hval


def equilibrium(z, rho_b, corr, L):
    s = Sys(z, rho_b, corr, L)
    phis = np.geomspace(max(0.04, sigma * N / (0.8 * L)), 0.97, 44)
    vals = [s.balance(p)[0] for p in phis]
    # first sign change from the swollen side: pressure-elastic > 0 inside means swell, so root from dilute side
    roots = []
    for i in range(len(phis) - 1):
        if vals[i] * vals[i + 1] < 0 and vals[i + 1] > 0:      # stable branch: pressure rises under compression
            roots.append(brentq(lambda p: s.balance(p)[0], phis[i], phis[i + 1], xtol=1e-5))
    if not roots:
        return None
    # lowest total free energy among stable roots: integrate the force from each root
    if len(roots) > 1:
        def F(p0, p1):
            ps = np.linspace(p0, p1, 60)
            hs = sigma * N / ps
            return np.trapezoid([s.balance(p)[0] for p in ps], hs)
        # F(h1) - F(h0) = - int f dh, with f the outward pressure
        G = [0.0]
        for i in range(len(roots) - 1):
            G.append(G[-1] - F(roots[i], roots[i + 1]))
        phi = roots[int(np.argmin(G))]
    else:
        phi = roots[0]
    e, Pb, cfar, lam, hval = s.balance(phi)
    om, p, I, rp, rm, rc = Pb
    return dict(phi=float(phi), h=float(hval), gamma=float(1 - z * (rp - rho_b) / phi), rc_frac=float(rc / phi), rp=float(rp),
                far_C=float(cfar), kappa=float(np.sqrt(4 * np.pi * lB * I)), nroots=len(roots))


def layer(z, L, closure, out):
    row = []
    for r in 10.0 ** np.arange(-9, 1.51, 0.5):
        try:
            q = equilibrium(z, r * MOLAR, closure == "corr", L)
        except Exception as e:
            q = dict(err=str(e)[:80])
        row.append(dict(rho=float(r), **(q or {})))
        print(r, row[-1], flush=True)
    name = f"z{z}_L{L:g}_{closure}"
    stored = json.load(open(DATA_FILE))["layer"].get(name)
    if stored:
        d = [abs(p["h"] - q["h"]) for p, q in zip(row, stored) if "h" in p and "h" in q]
        print(f"largest difference from the stored scan: {max(d):.3e} nm over {len(d)} points")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        json.dump(row, open(Path(out) / f"{name}.json", "w"), indent=0)


def layer_reader(data, layer_dir):
    """Returns rows(z, L, kind) and curve(z, L, key, kind) over the stored or recomputed layer scans."""
    def rows(z, L, kind="corr"):
        name = f"z{z}_L{L:g}_{kind}"
        return json.load(open(Path(layer_dir) / f"{name}.json")) if layer_dir else data["layer"][name]

    def curve(z, L, key, kind="corr"):
        r = [q for q in rows(z, L, kind) if key in q]
        return np.array([q["rho"] for q in r]), np.array([q[key] for q in r])
    return rows, curve


def ionic_curves(data):
    """Y uptake f and heights against rho at ionic strength 6 mM for N = 100 and N = 730, relative to 6 mM 1:1 salt. N = 730 is
    direct where the walk reached, a 1/N fit of the chain ladders from N = 198 up to 1e-6 M, and above that the N = 100 series
    moved by its offset at 1e-6 M."""
    rows = data["ionic_series"]
    edges = ["0.03", "0.05", "0.09"]
    ref = {r["N"]: r for r in rows if r["series"] == "reference"}
    short = sorted((r for r in rows if r["series"] == "n100"), key=lambda r: r["rho"])
    h6 = ref[730]["h_moment"] * ref[100]["h_moment"] / short[0]["h_moment"]  # takes out the Y trace of the reference at 1e-10 M

    def per_chain(r):
        return np.array([3 * r["plus_excess"] / (sigma * r["N"]), r["h_moment"] / r["N"], *[r["edge"][t] / r["N"] for t in edges]])
    ladders = {}
    for r in rows:
        if r["series"] == "ladder" and r["N"] >= 198:
            ladders.setdefault(r["rho"], []).append(r)
    points = {r["rho"]: per_chain(r) for r in rows if r["series"] == "walk730"}
    for rho, some in ladders.items():
        if rho not in points:
            n = np.array([r["N"] for r in some], dtype=float)
            v = np.array([per_chain(r) for r in some])
            points[rho] = np.array([np.polyval(np.polyfit(1 / n, v[:, k], 1), 1 / 730) for k in range(v.shape[1])])
    base = {r["rho"]: per_chain(r) for r in short}
    for rho in [1e-5, 1e-4, 5e-4]:
        points[rho] = np.r_[base[rho][0] + points[1e-6][0] - base[1e-6][0], base[rho][1:] * points[1e-6][1] / base[1e-6][1]]

    def curve(rho, v, n, h_ref, e_ref):
        return dict(rho=rho, f=v[:, 0], h=np.log10(v[:, 1] * n / h_ref), **{"edge " + t: np.log10(np.maximum(v[:, k + 2] * n, 1e-9) / e_ref[t]) for k, t in enumerate(edges)})
    rho = np.array(sorted(points))
    return (curve(np.array([r["rho"] for r in short]), np.array([per_chain(r) for r in short]), 100, ref[100]["h_moment"], ref[100]["edge"]),
            curve(rho, np.array([points[q] for q in rho]), 730, h6, ref[730]["edge"]))


def closed_cell(curve, height, nominal, D, fresh=False):
    """Bulk Y in M and height from rho_nominal = rho_b + Gamma(rho_b) / D, with Gamma for 73 charges per nm^2 and D in nm.
    fresh = True replaces the solution at every reading, with the brush keeping what it holds."""
    def at(values, r):
        return np.interp(np.log(r), np.log(curve["rho"]), values) if r >= curve["rho"][0] else values[0] * r / curve["rho"][0]
    held, out = 0.0, []
    for c in nominal:
        balance = lambda lr: np.exp(lr) * MOLAR + (at(curve["f"], np.exp(lr)) * 73 / 3 - held) / D - c * MOLAR
        r = 1e-16 if balance(np.log(1e-16)) > 0 else np.exp(brentq(balance, np.log(1e-16), np.log(c)))
        if fresh:
            held = max(held, at(curve["f"], r) * 73 / 3)
        out.append((r, at(height, r)))
    return out


def best_depth(curve, height, nominal, measured, fresh=False):
    """Depth D in nm with the lowest rms against the readings, its rms, and the range within rms + 0.01."""
    Ds = np.geomspace(1e4, 1e8, 241)
    e = np.array([np.sqrt(np.mean([(q[1] - y) ** 2 for q, y in zip(closed_cell(curve, height, nominal, D, fresh), measured)])) for D in Ds])
    ok = Ds[e < e.min() + 0.01]
    return Ds[e.argmin()], e.min(), ok[0], ok[-1]


def reading_depth(curve, nominal, measured):
    """For each reading alone: added Y, bulk rho in M, uptake f and the depth D in mm that puts the height on it."""
    out = []
    for c, y in zip(nominal, measured):
        r = np.exp(brentq(lambda lr: np.interp(lr, np.log(curve["rho"]), curve["h"]) - y, np.log(curve["rho"][0]), np.log(curve["rho"][-1])))
        g = np.interp(np.log(r), np.log(curve["rho"]), curve["f"])
        out.append((c, r, g, g * 73 / 3 / ((c - r) * MOLAR) / 1e6))
    return out


def rinse(curve, D, changes=100):
    """Replace the solution over the brush collapsed at 500 uM by Y-free solution of depth D, nm: bulk rho in M, f and height."""
    held = np.interp(np.log(closed_cell(curve, curve["h"], [5e-4], D)[0][0]), np.log(curve["rho"]), curve["f"]) * 73 / 3
    out = []
    for k in range(1, changes + 1):
        r = np.exp(brentq(lambda lr: np.interp(lr, np.log(curve["rho"]), curve["f"]) * 73 / 3 + np.exp(lr) * MOLAR * D - held, np.log(curve["rho"][0]), np.log(1e-2)))
        f = np.interp(np.log(r), np.log(curve["rho"]), curve["f"])
        held = f * 73 / 3
        out.append((k, r, f, np.interp(np.log(r), np.log(curve["rho"]), curve["h"])))
    return out


def tables(layer_dir):
    data = json.load(open(DATA_FILE))
    dig, scan = data["digitized"], data["scan"]
    rows, linear = layer_reader(data, layer_dir)

    def curve(z, L, key, kind="corr"):
        rho, y = linear(z, L, key, kind)
        return np.log10(rho), y

    def h0_of(L, kind="corr"):
        """Salt-free reference: the z=1 plateau; a brush without salt holds only monovalent C at every valence."""
        return curve(1, L, "h", kind)[1][0]

    print("== A. Fig. 3 readings against Eq. 7 and Eq. 9")
    print("  Eq9[M] uses the Gamma read from Fig. 3 and rho_b as the number in mol/L; Eq9[nm^-3] uses Eq. 7 and rho_b in nm^-3")
    print("  z rho[M]    Gamma(fig) Gamma(Eq7)  psi(fig)  Eq9[M]  Eq9[nm^-3]  fig/Eq9[nm^-3]")
    gerr, ratio = [], []
    for q in dig["fig3"]:
        if q["observable"] == "gamma":
            gerr.append(abs(q["value"] - gamma_eq7(q["zplus"], q["rho_molar"] * MOLAR)))
    for q in dig["fig3"]:
        if q["observable"] != "potential" or abs(q["value"]) < 0.04:
            continue
        z, c = q["zplus"], q["rho_molar"]
        G = gamma_eq7(z, c * MOLAR)
        gf = [p["value"] for p in dig["fig3"] if p["observable"] == "gamma" and p["zplus"] == z and p["rho_molar"] == c][0]
        e_M = -((z * z - 1) * gf + 1) / (z * z * (z + 1) * c)
        e_n = -((z * z - 1) * G + 1) / (z * z * (z + 1) * c * MOLAR)
        ratio.append(q["value"] / e_n)
        print(f"  {z} {c:7.3f} {gf:9.3f} {G:9.3f} {q['value']:9.3f} {e_M:8.3f} {e_n:10.3f} {q['value'] / e_n:13.3f}")
    print(f"  max |Gamma(fig) - Eq.7| over {len(gerr)} points = {max(gerr):.3f}; fig/Eq9 = {np.mean(ratio):.3f} +- {np.std(ratio):.3f}, median {np.median(ratio):.3f}")

    print("\n== B. Eq. 7 against Fig. 2: Gamma = 0, Gamma minimum, and the minimum of the collapsed branch h ~ rho_b/(-Gamma)")
    grid = np.geomspace(1e-5, 30, 20000) * MOLAR
    for z in [2, 3]:
        G = np.array([gamma_eq7(z, r) for r in grid])
        i = np.argmin(G)
        zero = brentq(lambda r: gamma_eq7(z, r), grid[0], grid[i])
        neg = G < 0
        hb = np.where(neg, grid / np.maximum(-G, 1e-300), np.inf)
        j = np.argmin(hb)
        fig2 = [(q["rho_molar"], q["log10_h_over_h0"]) for q in dig["fig2"] if q["zplus"] == z]
        low = min(fig2, key=lambda p: p[1])
        print(f"  z={z}: Gamma=0 at {zero / MOLAR:.4g} M, min Gamma {G[i]:+.3f} at {grid[i] / MOLAR:.3f} M, Eq.8 collapsed branch minimum at {grid[j] / MOLAR:.3f} M (Gamma={G[j]:+.3f}); "
              f"Fig.2 rho** {MARK[z]['ss']:.3g} M (Gamma={gamma_eq7(z, MARK[z]['ss'] * MOLAR):+.3f}, ratio {zero / MOLAR / MARK[z]['ss']:.0f}), lowest digitized point {low}")
    print("  rho_b*** and kappa_b b:")
    for z in [1, 2, 3]:
        print(f"  z={z}: Fig.2 rho*** {MARK[z]['sss']} M, kappa_b b = {np.sqrt(4 * np.pi * lB * z * (z + 1) * MARK[z]['sss'] * MOLAR):.2f}; kappa_b b = 1 at {1 / (4 * np.pi * lB * z * (z + 1)) / MOLAR:.3f} M")
    print("  Eq. 6 onset ratios: " + "; ".join(f"z=1/z={z}: Fig.2 {MARK[1]['s'] / MARK[z]['s']:.3g}, Eq.6 {(z * (z + 1) / 2) ** 3:.3g}" for z in [2, 3]))

    print("\n== C. Layer model against the 24 digitized Fig. 2 points")
    print("  L[nm]  h0[nm]  rms(all)  rms z=1  rms z=2  rms z=3  max|d|")
    for L in LS:
        h0 = h0_of(L)
        d = {1: [], 2: [], 3: []}
        for q in dig["fig2"]:
            x, h = curve(q["zplus"], L, "h")
            d[q["zplus"]].append(np.interp(np.log10(q["rho_molar"]), x, np.log10(h / h0)) - q["log10_h_over_h0"])
        e = np.array(sum(d.values(), []))
        print(f"  {L:7d} {h0:6.2f} {np.sqrt(np.mean(e ** 2)):8.3f} " + " ".join(f"{np.sqrt(np.mean(np.array(d[z]) ** 2)):8.3f}" for z in [1, 2, 3]) + f" {abs(e).max():7.3f}")
    d = []
    for q in dig["fig2"]:
        s = sorted((r["rho_molar"], r["height_moment_nm"]) for r in scan if r["zplus"] == q["zplus"] and r["direction"] == "up")
        d.append(np.interp(np.log10(q["rho_molar"]), np.log10([p for p, h in s]), np.log10([h / SCAN_H0 for p, h in s])) - q["log10_h_over_h0"])
    print(f"  archived SCFT scan, L=200: rms(all) {np.sqrt(np.mean(np.array(d) ** 2)):.3f}, max|d| {max(map(abs, d)):.3f}")
    h0 = h0_of(20000)
    for z in [1, 2, 3]:
        pts = [q for q in dig["fig2"] if q["zplus"] == z and q["rho_molar"] >= 0.01]
        x, h = curve(z, 20000, "h")
        y = np.log10(h / h0)
        d = [np.interp(np.log10(q["rho_molar"]), x, y) - q["log10_h_over_h0"] for q in pts]
        print(f"  z={z}, rho_b >= 0.01 M, L=20000: max|d| = {max(map(abs, d)):.3f} over {len(pts)} points; layer minimum {y.min():+.3f}; lowest Fig. 2 point {min(q['log10_h_over_h0'] for q in pts):+.3f}")

    print("\n== D. Onset against box length, log10(h/h0) = -0.04 on the swollen side; slope d log rho*/d log L")
    print("  predicted slope -2z/(z+1): z=2 -1.33, z=3 -1.50")
    for z in [2, 3]:
        on = []
        for L in LS:
            x, h = curve(z, L, "h")
            y = np.log10(h / h0_of(L)) + 0.04
            i = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))[0]
            on.append(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
        s = np.diff(on)
        print(f"  z={z}: rho* [M] = " + ", ".join(f"L={L}: {10 ** o:.3g}" for L, o in zip(LS, on)) + "; slopes " + ", ".join(f"{v:.2f}" for v in s) + f"; Fig.2 {MARK[z]['s']:.3g}")
        print(f"       box length that puts the layer-model onset at the Fig. 2 mark: {10 ** np.interp(-np.log10(MARK[z]['s']), -np.array(on), np.log10(LS)):.3g} nm")

    print("\n== E. Layer model at the neutral crossing and at the height minimum")
    for z in [2, 3]:
        for L in LS[:3]:
            x, h = curve(z, L, "h")
            g = curve(z, L, "gamma")[1]
            y = np.log10(h / h0_of(L))
            neutral = y[-1]
            i = np.flatnonzero((y[:-1] > neutral) & (y[1:] <= neutral))
            k = np.argmin(y)
            t = (y[i[0]] - neutral) / (y[i[0]] - y[i[0] + 1]) if len(i) else np.nan
            gc = g[i[0]] + t * (g[i[0] + 1] - g[i[0]]) if len(i) else np.nan
            print(f"  z={z} L={L}: high-salt level {neutral:+.3f}; crosses it at 10^{x[i[0]] + t * (x[i[0] + 1] - x[i[0]]):.2f} M with Gamma={gc:+.3f}; minimum {y[k]:+.3f} at 10^{x[k]:.1f} M, Gamma={g[k]:+.3f}")

    print("\n== F. Box-length SCFT, one row per run; h_m = 2<z>; log10(h_m/50.91)")
    for r in data["extended_box"]:
        if r["status"] == "completed":
            res = max(r["w_res"], r["poisson_res"], r["self_res"])
            x, h = curve(r["z"], r["Leff"], "h")
            lay = np.interp(np.log10(r["rho"]), x, np.log10(h / h0_of(r["Leff"])))
            print(f"  z={r['z']} L={r['Leff']:7.0f} rho={r['rho']:7.0e} h_m={r['h_moment']:6.2f} log={np.log10(r['h_moment'] / SCAN_H0):+.3f} layer={lay:+.3f} Gamma={r['gamma_core']:+.3f} c_far={r['c_far']:.5f} max residual={res:.1e} chain_err={r['chain_err']:.1e}")
        else:
            print(f"  z={r['z']} L={r['Leff']:7.0f} rho={r['rho']:7.0e} {r['status']}: {r['error'][:60]}")

    print("\n== G. Archived L = 200 nm scan: sign of psi_wall - psi_far, Gamma in the core, height, far counterions")
    for z in [2, 3]:
        print(f"  z={z}: rho[M]  log10(h_m/50.91)  Gamma_core  psi_wall-psi_far  psi_wall  psi_far   Eq.7 Gamma")
        pick = sorted([r for r in scan if r["zplus"] == z and r["direction"] == "up"], key=lambda r: r["rho_molar"])
        for r in pick[::2]:
            print(f"     {r['rho_molar']:9.3g} {np.log10(r['height_moment_nm'] / SCAN_H0):+7.3f} {r['gamma_core_0.5']:+7.3f} {r['psi_wall_relative']:+9.3f} {r['psi_wall']:+8.3f} {r['psi_far']:+8.3f} {r['gamma_eq7']:+7.3f}")

    print("\n== H. Size of the collapsed state, layer model, z=3, rho_b=1 mM")
    for L in [2000, 20000]:
        q = [q for q in rows(3, L) if abs(q["rho"] - 1e-3) < 1e-9][0]
        k = q["kappa"]
        print(f"  L={L}: phi={q['phi']:.3f} kappa={k:.2f}/nm 1/kappa={1 / k:.3f} nm  kappa a={k * 0.25:.2f}  z^2 lB kappa={9 * lB * k:.1f}  C fraction={q['rc_frac']:.3f}  Gamma={q['gamma']:+.3f}")
    print(f"  far-box C at L=200 if all sigma N is released: {10 / 200:.3f} nm^-3 = {10 / 200 / MOLAR * 1000:.0f} mM")

    print("\n== I. Eq. 7 against the exact layer without counterions: fixed polymer density x, salt reservoir only")
    print("  rho_b/x is the small parameter of S34; Gamma = 1 - z (rho_+ - rho_b)/x")
    print("  z rho_b[M]  rho_b/x(0.5)  Eq.7     exact x=0.05  exact x=0.5")
    for z in [2, 3]:
        for c in [1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0]:
            rb = c * MOLAR
            s = Sys(z, rb, True, 1e9)
            g = []
            for x in [0.05, 0.5]:
                rp = s.region(x, 0.0)[3]
                g.append(1 - z * (rp - rb) / x)
            print(f"  {z} {c:8.4f} {rb / 0.5:11.2e} {gamma_eq7(z, rb):+8.3f} {g[0]:+11.3f} {g[1]:+11.3f}")

    print("\n== J. Layer model: potential of the brush relative to the far box, against height")
    print("  z L rho[M] log10(h/h0) Gamma psi_brush-psi_far psi_far")
    for z in [2, 3]:
        for L in [2000, 20000]:
            h0 = h0_of(L)
            last = None
            for q in rows(z, L):
                if "phi" not in q or q["rho"] < 1e-6:
                    continue
                s = Sys(z, q["rho"] * MOLAR, True, L)
                e, Pb, cfar, lam, hv = s.balance(q["phi"])
                pf = s.far(cfar)[2]
                d = Pb[1] - pf
                flag = " <- sign change" if last is not None and last * d < 0 else ""
                last = d
                print(f"  {z} {L:6d} {q['rho']:9.3g} {np.log10(q['h'] / h0):+7.3f} {q['gamma']:+7.3f} {d:+8.3f} {pf:+8.3f}{flag}")

    print("\n== K. Width of the low-salt fall: decades of rho_b from log10(h/h0) = -0.1 to -0.3")

    def width(x, y):
        p, q = [np.interp(-v, -y, x) for v in (-0.1, -0.3)]
        return q - p

    for z in [2, 3]:
        pts = sorted((np.log10(q["rho_molar"]), q["log10_h_over_h0"]) for q in dig["fig2"] if q["zplus"] == z)[:4]
        print(f"  z={z}: Fig. 2 points {width(*map(np.array, zip(*pts))):.2f}", end="")
        for L in LS[:3]:
            x, h = curve(z, L, "h")
            y = np.log10(h / h0_of(L))
            k = np.argmin(y) + 1
            print(f"; layer L={L} {width(x[:k], y[:k]):.2f}", end="")
        s = sorted((r["rho_molar"], r["height_moment_nm"]) for r in scan if r["zplus"] == z and r["direction"] == "up")
        x = np.log10([p for p, h in s])
        y = np.log10([h / SCAN_H0 for p, h in s])
        k = np.argmin(y) + 1
        print(f"; SCFT L=200 {width(x[:k], y[:k]):.2f}")

    print("\n== L. Mean-field layer: no self-energy, h0 from z=1, low-salt plateau after the exchange against -(1/2) log10 z")
    for z in [2, 3]:
        x, h = curve(z, 20000, "h", "mf")
        y = np.log10(h / h0_of(20000, "mf"))
        print(f"  z={z} L=20000: at 1e-4 M {np.interp(-4, x, y):+.3f}, at 1e-3 M {np.interp(-3, x, y):+.3f}, at 1e-2 M {np.interp(-2, x, y):+.3f}; -(1/2) log10 z = {-0.5 * np.log10(z):+.3f}; high-salt level {y[-1]:+.3f}, minimum {y.min():+.3f}")

    print("\n== M. Half exchange: layer model against rho_+ = rho_b (z rho_b rho_C/c^2)^z exp(-z(z-1) du_B - 2z du_f)")
    print("  z L     rho_b(layer) rho_b(formula)  ratio   c[nm^-3]   du_B    du_f")
    for z in [2, 3]:
        for L in LS[:3]:
            r = [q for q in rows(z, L) if "phi" in q]
            x = np.log10([q["rho"] for q in r])
            f = np.array([q["rc_frac"] for q in r])
            i = np.flatnonzero((f[:-1] > 0.5) & (f[1:] <= 0.5))[0]
            t = (f[i] - 0.5) / (f[i] - f[i + 1])
            log_half = x[i] + t * (x[i + 1] - x[i])
            phi, kap, c = [r[i][k] * (r[i + 1][k] / r[i][k]) ** t for k in ("phi", "kappa", "far_C")]
            rb = 10 ** log_half * MOLAR
            Ib = z * (z + 1) * rb
            du_B = w(kap * kap / (4 * np.pi * lB)) - w(Ib)
            du_f = w(2 * c + Ib) - w(Ib)
            pred = ((phi / (2 * z)) * (2 * c * c / (z * phi)) ** z * np.exp(z * (z - 1) * du_B + 2 * z * du_f)) ** (1 / (z + 1))
            print(f"  {z} {L:6d} {rb / MOLAR:11.3e} {pred / MOLAR:12.3e} {pred / rb:8.2f} {c:10.5f} {du_B:+7.3f} {du_f:+7.3f}")

    print("\n== N. SCFT with psi(L) = 0 at the far boundary: log10(h_m/50.91) against dpsi/dz = 0 and Fig. 2")
    print("  z rho[M]   dpsi/dz=0,L=200  psi=0,L=200  psi=0,L=100  Fig.2")
    for z, c in sorted({(r["z"], r["rho"]) for r in data["grounded_box"]}):
        base = [r for r in scan if r["zplus"] == z and r["direction"] == "up" and abs(r["rho_molar"] / c - 1) < 1e-6][0]
        g = {r["L"]: r for r in data["grounded_box"] if r["z"] == z and r["rho"] == c}
        fig = [q["log10_h_over_h0"] for q in dig["fig2"] if q["zplus"] == z and abs(q["rho_molar"] / c - 1) < 1e-6]
        print(f"  {z} {c:7.0e} {np.log10(base['height_moment_nm'] / SCAN_H0):+13.3f} {g[200.0]['log_h']:+12.3f} {g[100.0]['log_h']:+12.3f} " + (f"{fig[0]:+8.3f}" if fig else "       -"))

    print("\n== O. One box length for all valences: SCFT stepped in L against the 24 readings of Fig. 2")
    common = {(r["z"], f"{r['rho']:.3e}", r["graft"]): r for r in data["common_box"]}

    def walked(z, c, rows, key):
        r = sorted((q for q in rows if q["z"] == z and abs(q["rho"] / c - 1) < 1e-6), key=lambda q: q[key])
        return np.log([q[key] for q in r]), np.log([q["h_moment"] for q in r]), r
    up = [r for r in data["box_scan"] if r["direction"] == "up"]
    readings = [(q["zplus"], q["rho_molar"], q["log10_h_over_h0"]) for q in dig["fig2"]]

    def misfit(L, chosen):
        out = []
        for z, c, y in chosen:
            if c < 1:
                x, lh, _ = walked(z, c, up, "L")
                out.append(np.interp(np.log(L), x, lh) / np.log(10) - np.log10(SCAN_H0) - y)
            else:
                out.append(np.log10(common[z, f"{c:.3e}", 0.5]["h_moment"] / SCAN_H0) - y)
        return np.array(out)
    Ls = np.geomspace(3000, 12000, 121)
    groups = dict(all=readings, z1=[q for q in readings if q[0] == 1], z2=[q for q in readings if q[0] == 2], z3=[q for q in readings if q[0] == 3],
                  fall=[q for q in readings if q[0] > 1 and q[1] <= 1e-3])
    print("  readings        best L[um]  rms     L with rms < best + 0.002")
    for name, chosen in groups.items():
        e = np.array([np.sqrt(np.mean(misfit(L, chosen) ** 2)) for L in Ls])
        ok = Ls[e < e.min() + 0.002]
        print(f"  {name:6s} n={len(chosen):2d} {Ls[e.argmin()] / 1000:10.2f} {e.min():7.4f}   {ok[0] / 1000:.2f} to {ok[-1] / 1000:.2f}")
    print("  L needed at each reading:  z rho[M]  Fig.2  L[um]  c_far[mM]")
    for z, c, y in groups["fall"]:
        x, lh, r = walked(z, c, up, "L")
        v = lh / np.log(10) - np.log10(SCAN_H0)
        i = np.flatnonzero((v[:-1] - y) * (v[1:] - y) <= 0)
        if len(i):
            t = (y - v[i[0]]) / (v[i[0] + 1] - v[i[0]])
            print(f"  {z} {c:8.2e} {y:+.3f} {np.exp(x[i[0]] + t * (x[i[0] + 1] - x[i[0]])) / 1000:6.2f} {np.interp(t, [0, 1], [r[i[0]]['c_far'], r[i[0] + 1]['c_far']]) / MOLAR * 1e3:7.2f}")
        else:
            print(f"  {z} {c:8.2e} {y:+.3f}  no crossing, range {v.min():+.3f} to {v.max():+.3f}")
    print("  Direct solutions at L = 6.5 um: z rho[M]  Fig.2  SCFT  difference  narrow graft  Gamma_core  c_far[mM]")
    e, n = [], []
    h0n = common[1, f"{1e-10:.3e}", 0.125]["h_moment"]
    for z, c, y in readings:
        p, q = common[z, f"{c:.3e}", 0.5], common[z, f"{c:.3e}", 0.125]
        e.append(np.log10(p["h_moment"] / SCAN_H0) - y)
        n.append(np.log10(q["h_moment"] / h0n) - y)
        print(f"  {z} {c:8.2e} {y:+.3f} {y + e[-1]:+.3f} {e[-1]:+.3f} {n[-1]:+.3f} {p['gamma_core']:+.3f} {p['c_far'] / MOLAR * 1e3:6.2f}")
    e, n = np.array(e), np.array(n)
    print(f"  all 24: rms {np.sqrt(np.mean(e ** 2)):.4f}, largest {abs(e).max():.4f}, within 0.01 {(abs(e) <= 0.01).sum()}; narrow graft rms {np.sqrt(np.mean(n ** 2)):.4f}")
    for z in [1, 2, 3]:
        print(f"  z={z}: rms {np.sqrt(np.mean(e[[q[0] == z for q in readings]] ** 2)):.4f}")
    print("  Width from log10(h/h0) = -0.1 to -0.3 in decades, by linear interpolation between the readings")
    for z in [2, 3]:
        pts = [(np.log10(c), y) for w_, c, y in readings if w_ == z and c <= 1e-3]
        x, y0 = zip(*pts)
        y1 = [np.log10(common[z, f"{10 ** q:.3e}", 0.5]["h_moment"] / SCAN_H0) for q in x]
        print(f"  z={z}: Fig. 2 {np.interp(0.3, -np.array(y0), x) - np.interp(0.1, -np.array(y0), x):.2f}, SCFT at 6.5 um {np.interp(0.3, -np.array(y1), x) - np.interp(0.1, -np.array(y1), x):.2f}")
    down = [r for r in data["box_scan"] if r["direction"] == "down"]
    diff = [abs(r["h_moment"] - q["h_moment"]) for r in down for q in up if q["z"] == r["z"] and q["rho"] == r["rho"] and q["L"] == r["L"]]
    print(f"  Stepping L down from 64 um: {len(diff)} points, largest height difference from stepping up {max(diff):.1e} nm")

    print("\n== P. Fixed 1:1 background, no canonical C, against the eight readings of the fall (rho_b <= 1 mM, z = 2, 3)")
    back = data["background_scan"]
    ref = walked(1, 1e-10, back, "c1")

    def misfit_c(c1, chosen, relative):
        h0 = np.exp(np.interp(np.log(c1), ref[0], ref[1])) if relative else SCAN_H0
        return np.array([np.interp(np.log(c1), *walked(z, c, back, "c1")[:2]) / np.log(10) - np.log10(h0) - y for z, c, y in chosen])
    cs = np.geomspace(1e-4, 0.3, 121)
    for relative in [False, True]:
        for name in ["fall", "z2", "z3"]:
            chosen = [q for q in groups["fall"] if name == "fall" or q[0] == int(name[1])]
            e = np.array([np.sqrt(np.mean(misfit_c(c1, chosen, relative) ** 2)) for c1 in cs])
            print(f"  h0 {'of the same background' if relative else 'salt-free 50.91 nm    '} {name:5s} best c1 {cs[e.argmin()] * 1e3:6.1f} mM, rms {e.min():.4f}")

    print("\n== Q. Y3+ data of Yu et al. 2016 at their own solution: NaNO3 + Y(NO3)3 at ionic strength 6 mM")
    ref6 = [r for r in back if r["z"] == 1 and r["c1"] == 0.006][0]["h_moment"]
    print(f"  Height in 6 mM 1:1 salt, their reference: {ref6:.2f} nm")
    print("  rho[M]   Na[mM]  Fig.7 of Yu  SCFT   Gamma_core  C share in brush  Y share by Donnan without correlation at rho_P = 0.2")
    for r in data["experiment_solution"]:
        na, y = r["c1"] * MOLAR, r["rho"] * MOLAR
        x = brentq(lambda x: na * x + 3 * y * x ** 3 - (na + 3 * y) / x - 0.2, 1.0, 1e3)
        print(f"  {r['rho']:7.0e} {r['c1'] * 1e3:6.2f} {r['figure7']:+9.3f} {np.log10(r['h_moment'] / ref6):+8.3f} {r['gamma_core']:+8.3f} {r['c_core']:12.3f} {3 * y * x ** 3 / 0.2:14.2f}")

    print("\n== R. Box length fitted to the experimental symbols, with the calculation of section O (canonical C, no Na background)")
    symbols = data["figure7"]
    on_grid = {z: sorted({r["rho"] for r in up if r["z"] == z and r["rho"] > 1e-9}) for z in [2, 3]}

    def line(z, c, L):
        return np.interp(np.log10(c), np.log10(on_grid[z]), [np.interp(np.log(L), *walked(z, q, up, "L")[:2]) / np.log(10) - np.log10(SCAN_H0) for q in on_grid[z]])
    Ls = np.geomspace(500, 64000, 121)
    print("  symbols              best L[um]  rms    L with rms < best + 0.01  rms at 6.6 um")
    chosen = {"Y fall, 4": [q for q in symbols if q["ion"] == "Y" and q["rho"] < 1e-3], "Y all, 7": [q for q in symbols if q["ion"] == "Y"],
              "Ba, 2": [q for q in symbols if q["ion"] == "Ba"], "Y and Ba, 9": symbols}
    for name, some in chosen.items():
        e = np.array([np.sqrt(np.mean([(line(q["z"], q["rho"], L) - q["log10_h_over_h0"]) ** 2 for q in some])) for L in Ls])
        ok = Ls[e < e.min() + 0.01]
        print(f"  {name:20s} {Ls[e.argmin()] / 1000:9.2f} {e.min():7.3f}   {ok[0] / 1000:.1f} to {ok[-1] / 1000:.1f} {np.interp(6600, Ls, e):16.3f}")

    print("\n== S. Chain length at ionic strength 6 mM, z = 3: Y uptake f and height per monomer; 730 fit = 1/N from N = 198")
    ionic = data["ionic_series"]
    rungs = {}
    for r in ionic:
        if r["series"] == "ladder":
            rungs.setdefault(r["rho"], {})[r["N"]] = (3 * r["plus_excess"] / (sigma * r["N"]), r["h_moment"] / r["N"])
    walked_730 = {r["rho"]: (3 * r["plus_excess"] / 73, r["h_moment"] / 730, r["phi_max"]) for r in ionic if r["series"] == "walk730"}
    print("  rho[M]   f at N = 100  198   339  730 fit   h/N at 100  198   339  730 fit   direct N = 730: f, h/N")
    for rho, some in sorted(rungs.items()):
        n = np.array([q for q in sorted(some) if q >= 198], dtype=float)
        fit = [np.polyval(np.polyfit(1 / n, [some[int(q)][k] for q in n], 1), 1 / 730) for k in [0, 1]]
        print(f"  {rho:7.1e}      " + " ".join(f"{some[q][0]:.3f}" for q in [100, 198, 339]) + f"  {fit[0]:.4f}      " + " ".join(f"{some[q][1]:.3f}" for q in [100, 198, 339]) + f"  {fit[1]:.4f}", *(f"  {walked_730[rho][0]:.4f} {walked_730[rho][1]:.4f}",) if rho in walked_730 else ())
    print("  Direct N = 730 from the swollen side: rho[M], f, h/N, phi_max")
    for rho, q in sorted(walked_730.items()):
        print(f"  {rho:9.3e} {q[0]:.4f} {q[1]:.4f} {q[2]:.3f}")
    print("  Collapse is by a dense wall layer, N = 100: rho[M], phi_max, thickness with phi > 0.4 [nm], h [nm]")
    for r in sorted((r for r in ionic if r["series"] == "n100" and r["rho"] >= 3e-9), key=lambda r: r["rho"]):
        print(f"  {r['rho']:7.1e} {r['phi_max']:.3f} {r['dense']:5.1f} {r['h_moment']:6.2f}")

    print("\n== T. Onset of the dense layer: rho up from 3e-9 and down from 5e-9 M in 2 % steps at fixed N")
    print("  N   points on both  largest height gap[nm]  largest phi_max step, between rho[M]  last rho reached going up")
    for n in [100, 198, 339]:
        rising = {r["rho"]: r for r in ionic if r["series"] == "onset_up" and r["N"] == n}
        falling = {r["rho"]: r for r in ionic if r["series"] == "onset_down" and r["N"] == n}
        common = sorted(set(rising) & set(falling))
        x = sorted(falling)
        step = max((falling[q]["phi_max"] - falling[p]["phi_max"], p, q) for p, q in zip(x, x[1:]))
        print(f"  {n:3d} {len(common):6d} {max(abs(rising[q]['h_moment'] - falling[q]['h_moment']) for q in common):18.1e} {step[0]:14.3f}, {step[1]:.3g} to {step[2]:.3g} {max(rising):14.3g}")

    print("\n== U. Closed cell: rho_nominal = rho_b + Gamma(rho_b) / D with 73 charges per nm^2, against the Y symbols of Fig. 7 of Yu et al.")
    short, long = ionic_curves(data)
    nominal = [r["rho"] for r in data["experiment_solution"]]
    measured = [r["figure7"] for r in data["experiment_solution"]]
    print("  Readings " + " ".join(f"{q:.1e}" for q in nominal) + "  Fig. 7 " + " ".join(f"{y:+.3f}" for y in measured))
    print("  curve, height              no depth limit: rms | best D[mm] (within rms + 0.01), rms, model, bulk rho[M]")
    for name, curve, key, fresh in [("N = 100, 2<z>", short, "h", False), ("N = 100, 2<z>, fresh", short, "h", True), ("N = 730, 2<z>", long, "h", False), ("N = 730, 2<z>, fresh", long, "h", True),
                                    ("N = 730, edge 0.03", long, "edge 0.03", False), ("N = 730, edge 0.05", long, "edge 0.05", False), ("N = 730, edge 0.09", long, "edge 0.09", False)]:
        D, e, lo, hi = best_depth(curve, curve[key], nominal, measured, fresh)
        rows = closed_cell(curve, curve[key], nominal, D, fresh)
        free = [np.interp(np.log(c), np.log(curve["rho"]), curve[key]) for c in nominal]
        print(f"  {name:26s} {np.sqrt(np.mean((np.array(free) - measured) ** 2)):.3f} [" + " ".join(f"{q:+.3f}" for q in free) + f"] | {D / 1e6:.3g} ({lo / 1e6:.3g} to {hi / 1e6:.3g}) {e:.3f} [" + " ".join(f"{q[1]:+.3f}" for q in rows) + "] " + " ".join(f"{q[0]:.2e}" for q in rows))
    print("  Depth each reading needs alone: reading[M], bulk rho[M], f, D[mm]; for N = 100 then N = 730")
    for curve in [short, long]:
        for c, r, g, need in reading_depth(curve, nominal, measured):
            print(f"  {c:.1e} {r:.2e} {g:.3f} {need:.3g}")
    D = best_depth(long, long["h"], nominal, measured)[0]
    print(f"  Rinse of the collapsed brush with Y-free solution, N = 730, D = {D / 1e6:.3g} mm: changes, bulk rho[M], f, log10(h/h6)")
    for k, r, f, y in rinse(long, D):
        if k in [1, 3, 10, 30, 100]:
            print(f"  {k:4d} {r:.2e} {f:.3f} {y:+.3f}")


def figures(layer_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open(DATA_FILE))
    curve = layer_reader(data, layer_dir)[1]
    ext = [r for r in data["extended_box"] if r["status"] == "completed"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "lines.linewidth": 1.8})
    box = {200: "#86b6ef", 2000: "#3987e5", 20000: "#1c5cab", 200000: "#0d366b"}
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#9a9892"

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True, constrained_layout=True)
    for z, ax in zip([1, 2, 3], axes):
        ax.axhline(-0.38, color=grid, lw=1, ls=(0, (4, 3)))
        h0 = {L: curve(1, L, "h")[1][0] for L in box}
        for L, col in box.items():
            x, h = curve(z, L, "h")
            ax.plot(x, np.log10(h / h0[L]), color=col, label=f"Layer model, L = {L / 1000:g} µm")
            pts = [r for r in ext if r["z"] == z and r["Leff"] == L]
            if pts:
                ax.plot([r["rho"] for r in pts], [np.log10(r["h_moment"] / SCAN_H0) for r in pts], ls="", marker="s", ms=6, mfc="white", mec=col, mew=1.4)
        pts = [q for q in data["digitized"]["fig2"] if q["zplus"] == z]
        ax.plot([q["rho_molar"] for q in pts], [q["log10_h_over_h0"] for q in pts], ls="", marker="o", ms=5, color=ink, label="Paper Fig. 2", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(1e-8, 30)
        ax.set_ylim(-0.7, 0.06)
        ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel(r"$\rho_b$ [M]")
        ax.set_title(f"$z_+ = {z}$", fontsize=10)
    axes[0].set_ylabel(r"$\log_{10}(h/h_0)$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="s", ms=6, mfc="white", mec=ink, mew=1.4))
    labels.append("SCFT in the extended box, colour = L")
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(FIGURE_DIR / "height_vs_box_length.png", dpi=200)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True, constrained_layout=True)
    for z, ax in zip([1, 2, 3], axes):
        ax.axhline(-0.38, color=grid, lw=1, ls=(0, (4, 3)))
        pts = sorted((r for r in data["scan"] if r["zplus"] == z and r["direction"] == "up"), key=lambda r: r["rho_molar"])
        ax.plot([r["rho_molar"] for r in pts], [np.log10(r["height_moment_nm"] / SCAN_H0) for r in pts], color=box[200], label="SCFT, L = 0.2 µm")
        pts = sorted((r for r in data["common_box"] if r["z"] == z and r["graft"] == 0.5 and r["rho"] > 1e-9), key=lambda r: r["rho"])
        ax.plot([r["rho"] for r in pts], [np.log10(r["h_moment"] / SCAN_H0) for r in pts], color=box[20000], marker="s", ms=5, mfc="white", mec=box[20000], mew=1.2, label="SCFT, L = 6.5 µm")
        pts = [q for q in data["digitized"]["fig2"] if q["zplus"] == z]
        ax.plot([q["rho_molar"] for q in pts], [q["log10_h_over_h0"] for q in pts], ls="", marker="o", ms=4, color=ink, label="Paper Fig. 2", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(1e-7, 30)
        ax.set_ylim(-0.7, 0.06)
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel(r"$\rho_b$ [M]")
        ax.set_title(f"$z_+ = {z}$", fontsize=10)
    axes[0].set_ylabel(r"$\log_{10}(h/h_0)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(FIGURE_DIR / "fig2_one_box.png", dpi=200)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True, constrained_layout=True)
    for z, ax in zip([1, 2, 3], axes):
        ax.axvspan(0.01, 30, color="#efeee9", lw=0, label="10 mM and above")
        ax.axhline(-0.38, color=grid, lw=1, ls=(0, (4, 3)), label="Neutral brush")
        for kind, col, name in [("corr", box[20000], "With self-energy"), ("mf", orange, "No self-energy")]:
            x, h = curve(z, 20000, "h", kind)
            ax.plot(x[x > 0], np.log10(h[x > 0] / curve(1, 20000, "h")[1][0]), color=col, label=name)
        pts = [q for q in data["digitized"]["fig2"] if q["zplus"] == z]
        ax.plot([q["rho_molar"] for q in pts], [q["log10_h_over_h0"] for q in pts], ls="", marker="o", ms=4, color=ink, label="Paper Fig. 2", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 30)
        ax.set_ylim(-0.7, 0.12)
        ax.set_xticks([1e-4, 1e-2, 1])
        ax.set_xlabel(r"$\rho_b$ [M]")
        ax.set_title(f"$z_+ = {z}$", fontsize=10)
    axes[0].set_ylabel(r"$\log_{10}(h/h_0)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.savefig(FIGURE_DIR / "self_energy_on_off.png", dpi=200)

    short, long = ionic_curves(data)
    nominal = [r["rho"] for r in data["experiment_solution"]]
    measured = [r["figure7"] for r in data["experiment_solution"]]
    depth = best_depth(long, long["h"], nominal, measured)[0]
    def crossing(x, y, t=-0.3):
        return np.exp(brentq(lambda lx: np.interp(lx, np.log(x), y) - t, np.log(x[0]), np.log(x[-1])))
    bulk = np.geomspace(1e-9, 1e-3, 121)
    added = np.geomspace(1e-7, 1e-3, 241)
    unlimited, cell = np.interp(np.log(bulk), np.log(long["rho"]), long["h"]), np.array([q[1] for q in closed_cell(long, long["h"], added, depth)])
    pts = sorted((q for q in data["digitized"]["fig2"] if q["zplus"] == 3 and q["rho_molar"] <= 1e-3), key=lambda q: q["rho_molar"])
    x0, x1 = crossing(bulk, unlimited), crossing(np.array(nominal), measured)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(bulk, unlimited, color=box[2000], label="SCFT at 6 mM, unlimited solution")
    ax.plot(added, cell, color=orange, label=f"SCFT at 6 mM, closed cell {depth / 1e6:.1f} mm deep")
    ax.plot([q["rho_molar"] for q in pts], [q["log10_h_over_h0"] for q in pts], color=grid, label="Line of the paper, no NaNO$_3$")
    ax.plot(nominal, measured, ls="", marker="o", ms=6, color=ink, label="Experiment at 6 mM, Yu et al. 2016", zorder=5)
    ax.annotate("", xy=(x1, -0.3), xytext=(x0, -0.3), arrowprops=dict(arrowstyle="<->", color=ink, lw=1))
    ax.text(np.sqrt(x0 * x1), -0.285, f"about {float(f'{x1 / x0:.1g}'):g} times", ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1e-3)
    ax.set_ylim(-0.7, 0.06)
    ax.set_xlabel(r"${\rm Y}^{3+}$ [M]")
    ax.set_ylabel(r"$\log_{10}(h/h_0)$")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.savefig(FIGURE_DIR / "collapse_at_6mM.png", dpi=200)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    ax = axes[0]
    rows = data["ionic_profiles"]["rows"]
    for r, col in zip([q for q in rows if q["N"] == 339], [box[200], box[2000], box[20000], box[200000]]):
        ax.plot(data["ionic_profiles"]["z_over_N"], r["phi"], color=col, label=f"{r['rho'] * 1e9:g} nM")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$z/N$ [nm]")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("Dense layer under a swollen layer", fontsize=10)
    ax.legend()
    ax = axes[1]
    ladder = sorted((r for r in data["ionic_series"] if r["series"] == "ladder" and r["N"] == 339), key=lambda r: r["rho"])
    ax.plot(short["rho"], short["f"], color=box[200], label="N = 100")
    ax.plot([r["rho"] for r in ladder], [3 * r["plus_excess"] / (sigma * 339) for r in ladder], color=box[2000], label="N = 339")
    ax.plot(long["rho"], long["f"], color=orange, ls=(0, (4, 3)), label="N = 730")
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1e-6)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"${\rm Y}^{3+}$ in solution [M]")
    ax.set_ylabel(r"${\rm Y}^{3+}$ share of the brush charge")
    ax.set_title("Chain length hardly matters", fontsize=10)
    ax.legend()
    ax = axes[2]
    for n, col in zip([100, 198, 339], [box[200], box[2000], box[200000]]):
        rising = sorted((r for r in data["ionic_series"] if r["series"] == "onset_up" and r["N"] == n), key=lambda r: r["rho"])
        falling = sorted((r for r in data["ionic_series"] if r["series"] == "onset_down" and r["N"] == n), key=lambda r: r["rho"])
        ax.plot([r["rho"] * 1e9 for r in falling], [r["phi_max"] for r in falling], color=col, label=f"N = {n}")
        ax.plot([r["rho"] * 1e9 for r in rising], [r["phi_max"] for r in rising], ls="", marker="o", ms=5, mfc="white", mec=col, mew=1.2)
    ax.plot([], [], ls="", marker="o", ms=5, mfc="white", mec=ink, mew=1.2, label="Going up")
    ax.plot([], [], color=ink, label="Going down")
    ax.set_xlim(3, 5)
    ax.set_ylim(0.3, 0.6)
    ax.set_xlabel(r"${\rm Y}^{3+}$ in solution [nM]")
    ax.set_ylabel(r"Largest $\phi$")
    ax.set_title("The dense layer appears without hysteresis", fontsize=10)
    ax.legend(loc="lower right")
    fig.savefig(FIGURE_DIR / "layers_and_chain_length.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)
    grid_z = np.array(data["ionic_profiles"]["z_over_N"])
    same = sorted((q for q in rows if q["rho"] == 3e-8), key=lambda q: q["N"])
    for r, col in zip(same, [box[200], box[2000], box[200000]]):
        axes[0].plot(grid_z * r["N"], r["phi"], color=col, label=f"N = {r['N']}")
        axes[1].plot(grid_z, r["phi"], color=col, label=f"N = {r['N']}")
    axes[0].set_xlim(0, 200)
    axes[1].set_xlim(0, 0.7)
    axes[0].set_xlabel("z [nm]")
    axes[1].set_xlabel("z/N [nm]")
    axes[0].set_title("N sets the length", fontsize=10)
    axes[1].set_title("The shape does not change", fontsize=10)
    for ax in axes:
        ax.set_ylim(0, 0.8)
        ax.set_ylabel(r"$\phi$")
    axes[0].legend()
    fig.savefig(FIGURE_DIR / "chain_length_scaling.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)
    ax = axes[0]
    _, _, low, high = best_depth(long, long["h"], nominal, measured)
    need = reading_depth(long, nominal, measured)
    ax.axhspan(low / 1e6, high / 1e6, color="#efeee9", lw=0, label="Fit to all four")
    ax.axhline(depth / 1e6, color=orange, label=f"Best, {depth / 1e6:.1f} mm")
    ax.plot([q[0] for q in need], [q[3] for q in need], ls="", marker="o", ms=6, color=ink, label="Each reading alone", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-7, 1e-3)
    ax.set_ylim(0.03, 20)
    ax.set_xlabel(r"${\rm Y}^{3+}$ added [M]")
    ax.set_ylabel("Solution depth D [mm]")
    ax.set_title("One depth does not fit every reading", fontsize=10)
    ax.legend(loc="upper right")
    ax = axes[1]
    steps = rinse(long, depth)
    ax.axhline(0, color=grid, lw=1, ls=(0, (4, 3)))
    ax.axhline(cell[-1], color=grid, lw=1, ls=(0, (4, 3)))
    ax.text(1.15, -0.03, "Swollen", color=ink, va="top")
    ax.text(1.15, cell[-1] - 0.02, "Collapsed, before the rinse", color=ink, va="top")
    ax.plot([q[0] for q in steps], [q[3] for q in steps], color=orange)
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(-0.7, 0.06)
    ax.set_xlabel("Changes of ${\\rm Y}^{3+}$-free solution")
    ax.set_ylabel(r"$\log_{10}(h/h_6)$")
    ax.set_title("A rinse leaves the brush collapsed", fontsize=10)
    fig.savefig(FIGURE_DIR / "closed_cell.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), sharey=True, constrained_layout=True)
    marks = {2: (8.9e-4, (1e-2, 1e-1)), 3: (2.75e-5, (1e-3, 1e-2))}
    rho = np.geomspace(1e-7, 30, 800)
    for z, ax in zip([2, 3], axes):
        ss, band = marks[z]
        ax.axhline(0, color=grid, lw=1)
        ax.axvspan(*band, color="#e4e2dc", lw=0, label="Height minimum in Fig. 2")
        ax.axvline(ss, color=ink, lw=1, ls=(0, (4, 3)), label=r"$\rho_b^{**}$ marked in Fig. 2")
        ax.plot(rho, [gamma_eq7(z, r * MOLAR) for r in rho], color=blue, label="Eq. 7")
        x, g = curve(z, 2000, "gamma")
        ax.plot(x, g, color=orange, label=r"Layer model, L = 2 µm")
        pts = [r for r in ext if r["z"] == z and r["Leff"] == 2000]
        ax.plot([r["rho"] for r in pts], [r["gamma_core"] for r in pts], ls="", marker="s", ms=6, mfc="white", mec=orange, mew=1.4, label="SCFT, L = 2 µm")
        ax.set_xscale("log")
        ax.set_xlim(1e-7, 30)
        ax.set_ylim(-1.45, 1.1)
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel(r"$\rho_b$ [M]")
        ax.set_title(f"$z_+ = {z}$", fontsize=10)
    axes[0].set_ylabel(r"$\Gamma$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(FIGURE_DIR / "gamma_against_eq7.png", dpi=200)


def log_chain(V, dz):
    """Beads 1..N, Gaussian bonds, zero ghost cells at z<0, mirror at the far wall; all in log space.
    Written independently of models/duan_scft.py."""
    half = int(np.ceil(8 * b / dz))
    off = np.arange(-half, half + 1) * dz
    lk = -1.5 * (off / b) ** 2 + np.log(dz) + 0.5 * np.log(3 / (2 * np.pi * b * b))
    nz = len(V)

    def bond(lq):
        ext = np.r_[np.full(half, -np.inf), lq, lq[-1:-half - 1:-1]]
        # log convolution; windows of the kernel on the extended array
        windows = np.lib.stride_tricks.sliding_window_view(ext, 2 * half + 1) + lk
        return logsumexp(windows, axis=1)

    z = (np.arange(nz) + 0.5) * dz
    lsrc = -0.5 * (z / (0.5 * b)) ** 2
    lb_ = [-V]
    for n in range(N - 1):
        lb_.append(-V + bond(lb_[-1]))
    lb_ = lb_[::-1]               # lb_[n] = backward propagator at bead n, including bead n's Boltzmann factor
    lq = lsrc
    rho = np.zeros(nz)
    lQ = None
    for n in range(N):
        s = lq + lb_[n]           # forward has no bead-n factor yet
        Q = logsumexp(s)
        lQ = Q if lQ is None else lQ
        assert abs(Q - lQ) < 1e-9, Q - lQ
        rho += np.exp(s - Q) / dz
        if n < N - 1:
            lq = bond(lq - V)
    return rho * sigma, lQ


def saved(scan_dir):
    """Archived brush solutions against the paper's equations, with code independent of models/duan_scft.py:
    log-domain discrete chain, residuals of Poisson and of the self-energy, mass, Gamma."""
    v = 1.0
    print(f"{'run':24s} {'z':>2s} {'rho_b':>9s} {'|drho|/rho':>10s} {'pois':>9s} {'selfE':>9s} {'W':>9s} {'mass':>9s} {'dh_m':>9s} {'dG':>9s}")
    worst = dict(chain=0, pois=0, selfE=0, W=0)
    for r in json.load(open(DATA_FILE))["scan"]:
        p = Path(scan_dir) / r["run"] / "profiles.npz"
        if not p.exists():
            print("missing", r["run"]); continue
        d = np.load(p)
        zv, rhoP, rhoC, rhop, rhom, W, psi, uP, upl, um = [d[k] for k in "z rhoP rhoC rhop rhom W psi uP up um".split()]
        z_, rb = float(r["zplus"]), r["rho_molar"] * MOLAR
        dz = zv[1] - zv[0]
        V = W / N
        rho, _ = log_chain(V, dz)
        chain = np.abs(rho - rhoP).max() / rhoP.max()
        phi = v * rhoP
        target = chi * (1 - 2 * phi) - np.log1p(-phi) - psi + uP          # z_P = -1
        Wres = np.abs(target - V).max()
        lap = np.diff(np.pad(psi, 1, mode="edge"), 2) / dz ** 2
        pois = np.abs(lap + 4 * np.pi * lB * (-rhoP + rhoC + z_ * rhop - rhom)).max()
        I0 = 0.5 * (rhoP + z_ ** 2 * rhop + rhoC + rhom)
        I0s = gaussian_filter1d(I0, sigma=a / dz, mode="reflect")
        Ib = 0.5 * z_ * (z_ + 1) * rb
        kap, kb = np.sqrt(8 * np.pi * lB * I0s), np.sqrt(8 * np.pi * lB * Ib)
        U = lB / (2 * a) * (u(a * kap) - u(a * kb))
        selfE = max(np.abs(U - um).max(), np.abs(z_ ** 2 * U - upl).max(), np.abs(U - uP).max())
        bp = rb * np.exp(-z_ * psi - upl)
        bm = z_ * rb * np.exp(psi - um)
        ions = max(np.abs(bp - rhop).max() / max(rhop.max(), 1e-300), np.abs(bm - rhom).max() / max(rhom.max(), 1e-300))
        mass = abs(rhoP.sum() * dz - 10) + abs(rhoC.sum() * dz - 10)
        hm = 2 * (zv * rhoP).sum() / rhoP.sum()
        dh = hm - r["height_moment_nm"]
        mask = rhoP > 0.5 * rhoP.max()
        G = np.sum(rhoP[mask] * (1 - z_ * (rhop[mask] - rb) / rhoP[mask])) / rhoP[mask].sum()
        dG = G - r["gamma_core_0.5"]
        for k, val in dict(chain=chain, pois=pois, selfE=selfE, W=Wres).items():
            worst[k] = max(worst[k], val)
        print(f"{r['run']:24s} {int(z_):2d} {r['rho_molar']:9.3g} {chain:10.2e} {pois:9.1e} {selfE:9.1e} {Wres:9.1e} {mass:9.1e} {dh:9.1e} {dG:9.1e}  ions {ions:.1e}")
    print("worst", worst)


def make_scft(zp, rb, nz=1600, L=200.0, graft=0.5):
    """models/duan_scft.py with the paper's parameters; its constructor is bypassed, so polymerfts is not needed."""
    from models.duan_scft import DuanBrushSCFT
    s = DuanBrushSCFT.__new__(DuanBrushSCFT)
    s.N, s.b, s.v, s.zP, s.zp, s.sigma = N, b, 1.0, -1.0, float(zp), sigma
    s.aP = s.aion = s.aplus = a
    s.lB, s.chi, s.with_counterions, s.c1, s.L, s.Nz = lB, chi, True, 0.0, L, nz
    s.dz = L / nz
    s.z = (np.arange(nz) + 0.5) * s.dz
    s.corr = True
    s.set_bulk(rb)
    src = np.exp(-0.5 * (s.z / graft) ** 2)
    s.R0 = 10.0
    s.q_init = {"G": src / (src.sum() * s.dz / s.R0)}
    return s


def code(scan_dir):
    """Run the components of models/duan_scft.py directly.
    (1) default real-space chain on the archived fields against the log-domain chain;
    (2) nonlocal self-energy against S33 in a uniform state, with quadrature refinement.
    """
    from models.duan_scft import DuanBrushSCFT
    print("== (1) default realspace chain vs log-domain chain on the archived fields")
    bad = []
    for r in json.load(open(DATA_FILE))["scan"]:
        d = np.load(Path(scan_dir) / r["run"] / "profiles.npz")
        s = make_scft(r["zplus"], r["rho_molar"] * MOLAR)
        V = d["W"] / 100
        rho = DuanBrushSCFT._chain_density(s, V)
        ref, _ = log_chain(V, s.dz)
        bad.append((np.abs(rho - ref).max(), r["zplus"], str(r["rho_molar"])[:9], float(np.ptp(V)) * 100))
    bad.sort(reverse=True)
    for e, z, rm, c in bad[:8]:
        print(f"  max|rho_rs-rho_log|={e:.3e}  z={z} rho={rm} M  field contrast over the chain={c:.0f} kT")
    print(f"  points with error>1e-6: {sum(q[0] > 1e-6 for q in bad)} of {len(bad)}")

    print("\n== (2) uniform state: U(I0) relative to bulk, against S33")
    for Ib, I0 in [(0.18, 1.2), (0.01, 1.2), (0.01, 0.1)]:
        exact = 0.7 / (2 * 0.25) * (u(0.25 * np.sqrt(8 * np.pi * 0.7 * I0)) - u(0.25 * np.sqrt(8 * np.pi * 0.7 * Ib)))
        for nq in [64, 256]:
            for nz, L in [(400, 50.0), (800, 50.0)]:
                s = make_scft(1, 2 * Ib / 2, nz=nz, L=L)
                s.kappa_b = np.sqrt(8 * np.pi * s.lB * Ib)
                s._nonlocal_setup(nq=nq)
                U = s._nonlocal_U(np.full(nz, I0), Ib)[0.25]
                print(f"  Ib={Ib:5.2f} I0={I0:4.2f} nq={nq:3d} dz={L / nz:.4f}: nonlocal mid={U[nz // 2]:+.5f} S33={exact:+.5f} rel.err={U[nz // 2] / exact - 1:+.4f}  near-wall excess={U[1] - U[nz // 2]:+.3f}")


def coupled_newton(s):
    """Solve the equal-radius equations for V and one base self-energy; PB is solved inside every residual.
    Copied from studies/prl_runs.py at tag v0.10.0."""
    from scipy.optimize import newton_krylov
    assert s.aP == s.aplus == s.aion

    def residual(values):
        V, U = values[:s.Nz], values[s.Nz:]
        s.W = s.N * V.copy()
        s.uP, s.up, s.um = s.zP ** 2 * U, s.zp ** 2 * U, U.copy()
        rho = s._chain_density(V)
        c, p, m = s._solve_pb(rho)
        phi = np.clip(s.v * rho, 0, .999)
        target = s.chi * (1 - 2 * phi) - np.log1p(-phi) + s.zP * s.psi + s.uP
        energies = s._self_energies(rho, c, p, m)
        return np.r_[target - V, energies[2] - U]

    def callback(values, error):
        print(json.dumps(dict(newton_max_residual=float(abs(error).max()))), flush=True)
    values = newton_krylov(residual, np.r_[s.W / s.N, s.um], f_tol=3e-8, rdiff=1e-5, maxiter=80, inner_maxiter=50, callback=callback, line_search="armijo")
    residual(values)


def extended_chain(s, potential):
    """Scaled positive chains with an extended exponent range and one Q, for the fields where the default chain of
    models/duan_scft.py hits its 1e-300 floor. Copied from studies/prl_runs.py at tag v0.10.0."""
    ld = np.longdouble
    potential = np.asarray(potential, dtype=ld)
    half = int(np.ceil(8 * s.b / s.dz))
    offsets = np.arange(-half, half + 1, dtype=ld) * ld(s.dz)
    kernel = ld(s.dz) * np.sqrt(ld(3) / (2 * ld(np.pi) * ld(s.b) ** 2)) * np.exp(-ld(1.5) * (offsets / ld(s.b)) ** 2)

    def bond(q):
        return np.convolve(np.r_[np.zeros(half, dtype=ld), q, q[-1:-half - 1:-1]], kernel, mode="valid")

    shift = potential.min()
    ew = np.exp(-(potential - shift))
    back = np.empty((s.N, s.Nz), dtype=ld)
    scales = np.empty(s.N, dtype=ld)
    back[-1] = ew / ew.max()
    scales[-1] = np.log(ew.max())
    for n in range(s.N - 2, -1, -1):
        q = ew * bond(back[n + 1])
        scale = q.max()
        if scale <= 0 or not np.isfinite(scale):
            raise FloatingPointError("Invalid extended propagator")
        back[n] = q / scale
        scales[n] = scales[n + 1] + np.log(scale)
    src = np.asarray(s.q_init["G"], dtype=ld)
    denominator = np.mean(src * back[0])
    if denominator <= 0:
        raise FloatingPointError("Extended partition vanished")
    log_q = np.log(denominator) + scales[0] - s.N * shift
    q_a = src / src.max()
    la = np.log(src.max())
    density = np.zeros(s.Nz, dtype=ld)
    error = 0.
    for n in range(s.N):
        sl = q_a * back[n] * np.exp(la + scales[n] - scales[0]) / denominator
        error = max(error, float(abs(sl.mean() - 1)))
        density += sl / s.N
        if n < s.N - 1:
            q_a = bond(ew * q_a)
            scale = q_a.max()
            q_a /= scale
            la += np.log(scale)
    if not np.isfinite(density).all() or error > 1e-10:
        raise FloatingPointError(f"Extended contour consistency failed: {error}")
    s.log_partition = float(log_q)
    s.extended_partition_error = error
    return np.asarray(density * (s.sigma * s.N / s.L), dtype=float)


def pb_ext(s, rhoP, nC_total=None, n_newton=300):
    """Poisson-Boltzmann with the box extended by a flat far region (same psi and u as the last cell).

    Only the canonical counterion normalisation changes: mass = sum_i w_i rho_C,i with w_last = dz + V_res. This is
    the exact long-box limit while the far end of the 200 nm box is flat. All other equations are unchanged.
    """
    from scipy.linalg import solve_banded
    alpha, amount, invdz2 = 4 * np.pi * s.lB, s.sigma * s.N, 1 / s.dz ** 2
    weight = np.full(s.Nz, s.dz)
    weight[-1] += s.Vres
    psi = s.psi.copy()
    far = -1.0 if s.grounded else 1.0       # ghost cell at the far face: psi = 0 or dpsi/dz = 0

    def evaluate(p):
        bc = np.exp(-np.clip(p + s.um, -300, 300))
        c = amount * bc / (weight * bc).sum()
        plus = s.rho_b * np.exp(-np.clip(s.zp * p + s.up, -300, 300))
        minus = s.zp * s.rho_b * np.exp(-np.clip(-p + s.um, -300, 300))
        lap = np.diff(np.r_[p[0], p, far * p[-1]], 2) * invdz2
        return lap + alpha * (s.zP * rhoP + c + s.zp * plus - minus), c, plus, minus

    tol = min(1e-9 * max(1.0, alpha * (abs(rhoP).max() + s.zp * s.rho_b)), 2e-8)
    for _ in range(n_newton):
        f, c, plus, minus = evaluate(psi)
        if abs(f).max() < tol:
            s.psi = psi
            return c, plus, minus
        ab = np.zeros((3, s.Nz))
        ab[0, 1:] = invdz2
        ab[2, :-1] = invdz2
        ab[1, :] = -2 * invdz2 - alpha * (c + s.zp ** 2 * plus + minus)
        ab[1, 0] += invdz2
        ab[1, -1] += far * invdz2
        yz = solve_banded((1, 1), ab, np.column_stack([-f, alpha * c]))
        y, q = yz[:, 0], yz[:, 1]
        vec = c * weight / amount
        step = y - q * (vec @ y) / (1 + vec @ q)
        norm = np.linalg.norm(f)
        t = 1.0
        for _ in range(40):
            if np.linalg.norm(evaluate(psi + t * step)[0]) < norm:
                psi = psi + t * step
                break
            t *= 0.5
        else:
            raise RuntimeError(f"PB line search failed {abs(f).max():.3e}")
    raise RuntimeError("PB did not converge")


def pb_background(s, rhoP, nC_total=None, n_newton=300):
    """Poisson-Boltzmann with a fixed grand-canonical 1:1 background and no canonical counterions; psi(L) = 0."""
    from scipy.linalg import solve_banded
    alpha, invdz2 = 4 * np.pi * s.lB, 1 / s.dz ** 2
    psi = s.psi.copy()

    def evaluate(p):
        c = s.c1 * np.exp(-np.clip(p + s.um, -300, 300))
        plus = s.rho_b * np.exp(-np.clip(s.zp * p + s.up, -300, 300))
        minus = (s.zp * s.rho_b + s.c1) * np.exp(-np.clip(-p + s.um, -300, 300))
        lap = np.diff(np.r_[p[0], p, -p[-1]], 2) * invdz2
        return lap + alpha * (s.zP * rhoP + c + s.zp * plus - minus), c, plus, minus

    tol = min(1e-9 * max(1.0, alpha * (abs(rhoP).max() + s.zp * s.rho_b)), 2e-8)
    for _ in range(n_newton):
        f, c, plus, minus = evaluate(psi)
        if abs(f).max() < tol:
            s.psi = psi
            return c, plus, minus
        ab = np.zeros((3, s.Nz))
        ab[0, 1:] = invdz2
        ab[2, :-1] = invdz2
        ab[1, :] = -2 * invdz2 - alpha * (c + s.zp ** 2 * plus + minus)
        ab[1, 0] += invdz2
        ab[1, -1] -= invdz2
        step = solve_banded((1, 1), ab, -f)
        norm = np.linalg.norm(f)
        t = 1.0
        for _ in range(40):
            if np.linalg.norm(evaluate(psi + t * step)[0]) < norm:
                psi = psi + t * step
                break
            t *= 0.5
        else:
            raise RuntimeError(f"PB line search failed {abs(f).max():.3e}")
    raise RuntimeError("PB did not converge")


def scft(zp, rho, Leff, warm, out, grounded=False, background=None, graft=0.5, dz=0.125, stretch=1.0):
    from scipy.optimize import NoConvergence
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    s = make_scft(zp, rho * MOLAR, nz=int(round(Leff / dz)), L=Leff, graft=graft) if grounded else make_scft(zp, rho * MOLAR, graft=graft)
    s.Vres = Leff - s.L
    s.grounded = grounded
    s._solve_pb = MethodType(pb_ext, s)
    if background is not None:
        s.with_counterions = False
        s.set_c1(background)
        s._solve_pb = MethodType(pb_background, s)
    s._chain_density = MethodType(extended_chain, s)
    d = np.load(warm)
    for k in ["W", "psi", "uP", "up", "um"]:     # stretch > 1 starts a longer chain from a shorter one at the same density
        setattr(s, k, np.interp(s.z, d["z"] * stretch, d[k] * (stretch if k == "W" else 1.0)))
    if grounded:
        s.psi = s.psi - d["psi"][-1]
    res = dict(z=zp, rho=rho, Leff=Leff, grounded=grounded, warm=str(warm), status="running")
    try:
        for cycle in range(6):
            try:
                coupled_newton(s)
                break
            except NoConvergence as e:
                x = np.asarray(e.args[0])
                s.W = s.N * x[:s.Nz].copy()
                s.um = x[s.Nz:].copy()
                s.uP, s.up = s.zP ** 2 * s.um, s.zp ** 2 * s.um
                if cycle == 5:
                    raise
        rho_P = s._chain_density(s.W / s.N)
        c, plus, minus = s._solve_pb(rho_P)
        # independent residuals, chain from the log-domain code
        ref = log_chain(s.W / s.N, s.dz)[0]
        phi = s.v * rho_P
        target = s.chi * (1 - 2 * phi) - np.log1p(-phi) + s.zP * s.psi + s.uP
        lap = np.diff(np.r_[s.psi[0], s.psi, (-1.0 if grounded else 1.0) * s.psi[-1]], 2) / s.dz ** 2
        en = s._self_energies(rho_P, c, plus, minus)
        res.update(chain_err=float(abs(ref - rho_P).max()), w_res=float(abs(target - s.W / s.N).max()),
                   poisson_res=float(abs(lap + 4 * np.pi * s.lB * (s.zP * rho_P + c + s.zp * plus - minus)).max()),
                   self_res=float(max(abs(p - q).max() for p, q in zip(en, (s.uP, s.up, s.um)))),
                   mass=float(rho_P.sum() * s.dz), c_mass=float(c.sum() * s.dz + c[-1] * s.Vres))
        mask = rho_P > 0.5 * rho_P.max()
        res.update(h_moment=float(2 * (s.z * rho_P).sum() / rho_P.sum()), h_equimolar=float(rho_P.sum() * s.dz / rho_P.max()),
                   gamma_core=float(np.sum(rho_P[mask] * (1 - s.zp * (plus[mask] - s.rho_b) / rho_P[mask])) / rho_P[mask].sum()),
                   c_core=float(np.sum(c[mask]) / rho_P[mask].sum()), c_far=float(c[-1]), psi_far=float(s.psi[-1]), phi_max=float(phi.max()), plus_excess=float((plus - plus[-1]).sum() * s.dz),
                   status="completed" if max(res.get("w_res", 1), res.get("poisson_res", 1), res.get("self_res", 1)) < 1e-6 else "unconverged")
        np.savez_compressed(out / "profiles.npz", z=s.z, rhoP=rho_P, rhoC=c, rhop=plus, rhom=minus, W=s.W, psi=s.psi, uP=s.uP, up=s.up, um=s.um)
    except Exception as e:
        res.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
    res["seconds"] = time.perf_counter() - t0
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    return res


def walk(zp, rho, warm, out, values, background=False, graft=0.5, step_seconds=1500, box=200.0, dz=0.125, stretch=1.0):
    """Take one (z, rho) through box lengths, or background densities in M. Each solve starts from the last converged one;
    a failed step is retried from a point halfway on a log scale, up to three times."""
    def stop(*_):
        raise TimeoutError("step limit")
    signal.signal(signal.SIGALRM, stop)
    previous = None
    for value in values:
        tries = [value]
        while tries:
            v = tries[0]
            run = Path(out) / f"v{v:g}"
            signal.alarm(step_seconds)
            res = scft(zp, rho, box if background else v, warm, run, grounded=background, background=v if background else None, graft=graft, dz=dz, stretch=stretch)
            stretch = 1.0 if res["status"] == "completed" else stretch
            signal.alarm(0)
            if res["status"] == "completed":
                warm, previous = run / "profiles.npz", v
                tries.pop(0)
            elif previous is not None and len(tries) < 4:
                tries.insert(0, float(np.sqrt(previous * v)))
            else:
                tries.pop(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("analytic")
    sub.add_parser("correlation")
    q = sub.add_parser("layer")
    q.add_argument("z", type=int, choices=[1, 2, 3])
    q.add_argument("L", type=float)
    q.add_argument("closure", choices=["corr", "mf"])
    q.add_argument("--out")
    for name in ["tables", "figures"]:
        sub.add_parser(name).add_argument("--layer-dir")
    for name in ["saved", "code"]:
        sub.add_parser(name).add_argument("--scan-dir", required=True)
    q = sub.add_parser("scft")
    q.add_argument("z", type=int)
    q.add_argument("rho_molar", type=float)
    q.add_argument("L", type=float)
    q.add_argument("warm")
    q.add_argument("out")
    q.add_argument("--grounded", action="store_true")
    q = sub.add_parser("walk")
    q.add_argument("z", type=int)
    q.add_argument("rho_molar", type=float)
    q.add_argument("warm")
    q.add_argument("out")
    q.add_argument("--values", required=True, help="comma list of box lengths in nm, or of background densities in M")
    q.add_argument("--background", action="store_true")
    q.add_argument("--graft", type=float, default=0.5)
    q.add_argument("--chain", type=int, default=N, help="chain length; --stretch then scales the starting profile")
    q.add_argument("--stretch", type=float, default=1.0)
    q.add_argument("--box", type=float, default=200.0, help="box length with --background, nm")
    q.add_argument("--dz", type=float, default=0.125)
    q.add_argument("--step-seconds", type=int, default=1500)
    args = p.parse_args()
    if args.command == "analytic":
        analytic()
    elif args.command == "correlation":
        correlation()
    elif args.command == "layer":
        layer(args.z, args.L, args.closure, args.out)
    elif args.command == "tables":
        tables(args.layer_dir)
    elif args.command == "figures":
        figures(args.layer_dir)
    elif args.command == "saved":
        saved(args.scan_dir)
    elif args.command == "code":
        code(args.scan_dir)
    elif args.command == "walk":
        globals()["N"] = args.chain
        walk(args.z, args.rho_molar, args.warm, args.out, [float(v) for v in args.values.split(",")], args.background, args.graft,
             args.step_seconds, args.box, args.dz, args.stretch)
    else:
        scft(args.z, args.rho_molar, args.L, args.warm, args.out, args.grounded)


if __name__ == "__main__":
    main()
