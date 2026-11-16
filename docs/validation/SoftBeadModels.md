# Validation of the soft-bead models

What the validation runs check for [SoftBeadDiblockMelt.py](../../examples/block_copolymer/clfts/SoftBeadDiblockMelt.py), against Lequieu, *Macromolecules* **2024**, 57, 10870. The other systems of Lequieu's paper (a homopolymer solution, a polyampholyte solution and polyelectrolyte gels) are in the same comparison. The labels are those of [Validation of the Examples](ExampleValidation.md).

The paper shows that molecular dynamics and CL-FTS of the same soft-bead models give the same pressure and chemical potentials. Its systems map onto the compressible model of polymerfts without missing features: discrete chains of N beads and N − 1 bonds, Gaussian-smeared excluded volume and Coulomb interactions (the paper's pair kernels are Γ² of the code), explicit solvent and ions as molecules of one segment. The paper's excluded volume differs from the compressibility penalty by terms linear in the fixed numbers and the volume, which shift P by BC²/2 and the chemical potentials by BCα. Its plotted μ_ex − μ_s is −⟨ln Q⟩ including the self-interaction of the beads.

- Checked (validation runs, ETD integrator): CL-FTS against the paper's CL-FTS, read from the vector data of its figures:

| System (figure) | Quantity | polymerfts | Lequieu 2024 | Mean field |
|-----------------|----------|-----------:|-------------:|-----------:|
| Diblock melt χN = 2, N = 50 (Fig. 2) | βP R_g³ / β(μ_ex − μ_s) | 254.37 ± 0.11 / 102.106 ± 0.001 | 254.254 / 102.109 | 257.5 / 101.0 |
| Diblock melt χN = 10, N = 100 (Fig. 2) | βP R_g³ / β(μ_ex − μ_s) | 264.20 ± 0.27 / 105.778 ± 0.003 | 263.897 / 105.773 | 267.5 / 105.0 |
| Homopolymer solution C = 0.7, N = 50 (Fig. 1) | βP R_g³ / β(μ_ex − μ_s) | 2.993 ± 0.037 / 8.494 ± 0.001 | 2.953 / 8.4875 | 3.150 / 7.0 |
| Polyampholyte C = 3, N = 50, E = 10³ (Fig. 4) | βP_ex R_g³ / β(μ_ex − μ_s) | 1.313 ± 0.021 / 4.394 ± 0.002 | 1.388 / 4.390 | 4.5 / 3.0 |
| Polyelectrolyte gel with salt, E = 10³ (Fig. 5) | βP_ex / μ_ex,+ + μ_ex,− / μ_ex,1 + 20 μ_ex,− | 12.699 ± 0.021 / 0.1555 ± 0.0003 / 19.369 ± 0.026 | 12.712 / 0.156 / 19.386 | 12.788 / 0.200 / 17.779 |
| Polyelectrolyte gel with salt, E = 10⁴ (Fig. 5) | same | 12.285 ± 0.045 / −0.513 ± 0.002 / 8.181 ± 0.047 | 12.273 / −0.515 / 8.201 | 12.788 / 0.200 / 17.779 |

Pressures are extrapolated to δτ → 0 from δτN = 0.5 and 1 where both were run. The chemical potentials agree within 0.003–0.02 kT, for fluctuation corrections of 0.04–9.6 kT, and the pressures within about one standard error, except the polyampholyte, whose excess pressure lies 0.075 (2 % of its fluctuation correction) below the paper and changes with the time step (1.388 ± 0.018 at δτN = 1, 1.332 ± 0.015 at 0.5). The paper gives no errors for its CL-FTS values. In a Gaussian-core fluid (one-bead molecules, βu0 = 0.5, a = 0.2, ρ = 30) CL-FTS agrees with an exact Monte Carlo simulation of the particles: P = 251.240 ± 0.017 against 251.2285 ± 0.0052, and μ_ex = 14.4910 against 14.4900 ± 0.0021 from Widom insertion (after the shift (1/2V) dμ/dρ between the insertion of one particle and the derivative dA/dn).
- The example `SoftBeadDiblockMelt.py` (χN = 2, 40000 steps at δτN = 0.5, 36 min on an A10 GPU) gives βP R_g³ = 254.35 ± 0.08 and β(μ_ex − μ_s) = 102.107 ± 0.001.
- Not checked: the lamellar diblock points, the two-phase polyampholyte points, the dilute points (C ≤ 0.05, 1–40 chains per box), the gels at higher polymer fraction and at E = 10⁵.
