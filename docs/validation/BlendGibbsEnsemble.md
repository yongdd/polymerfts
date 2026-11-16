# Validation of the blend Gibbs ensemble

What the tests and validation runs check for [GibbsEnsemble.py](../../examples/blend/clfts/GibbsEnsemble.py). The labels are those of [Validation of the Examples](ExampleValidation.md).

## Checks

- Checked (`TestGibbsEnsemble`, small boxes of a symmetric blend at χN = 3 with N = 20): in the mean-field limit N̄ = 10²⁸, the Flory-Huggins binodal within 2×10⁻⁶ and the coexistence of charged chains with canonical ions and with grand-canonical salt (Donnan potentials) against independent solutions within 10⁻⁶; at N̄ = 10⁴, bitwise continuation of a run.
- Checked (validation runs with the settings of the example): in the mean-field limit the boxes reach the Flory-Huggins binodal. At N̄ = 10⁴, where fluctuations shift the binodal by −0.014, the Gibbs ensemble agrees with an independent route: canonical CL-FTS runs of one box at fixed compositions, whose exchange chemical potential βμ_A − βμ_B vanishes at the binodal of this symmetric blend.
- Checked (validation runs): the fluctuating binodal of a symmetric blend of continuous chains in Fredrickson and Delaney 2023, Fig. 7.2, at C = 4 and 10. Eleven of the thirteen points agree within about one published error, including all at C = 10.
- Not reproduced: two points of that figure at C = 4. At χN = 2.25, two runs give 0.3231 and 0.3219 against 0.308 ± 0.006, 2.3 errors away. At χN = 2.15, published as 0.460 ± 0.030 close to the critical point, the run has no stable coexistence: after about 200 cycles one box empties (0.22 of a total volume of 15) and the other holds the overall composition.
- Not checked for this blend at N̄ = 10⁴: the dependence on `exchange_rate` and `cycle_steps`, and finite-size effects (both routes use the same cell). [CoacervateGibbs.py](CoacervateGibbs.md) checks both for a coacervate.
- The method follows Riggleman and Fredrickson (*J. Chem. Phys.* **2010**, 132, 024104) in the intensive form of Mester, Lynd and Fredrickson (*Soft Matter* **2013**, 9, 11288): both boxes stay in one cell, a convergence parameter, and each is held at the reference density, so their volumes only count the material and their pressures need not agree ([CL-FTS theory, Section 8](../theory/ComplexLangevinSimulation.md#8-gibbs-ensemble)).

## The example

![Compositions of the Gibbs boxes and the exchange chemical potential of canonical CL-FTS](../figures/gibbs_ensemble_blend.png)

| Volume fraction of A in the A-rich phase | Value |
|------------------------------------------|-------|
| Flory-Huggins binodal | 0.92928 |
| `GibbsEnsemble`, N̄ = 10²⁸ (after 40 cycles) | 0.92928 |
| `GibbsEnsemble`, N̄ = 10⁴, 400 cycles (second half) | 0.91498 ± 0.00009 |
| `GibbsEnsemble`, N̄ = 10⁴, the example (40 cycles) | 0.9148 |
| Canonical CL-FTS, N̄ = 10⁴: zero of βμ_A − βμ_B | 0.91497 ± 0.00013 |

Settings of the example: 16 × 16 grid in a cell of 2 R₀ × 2 R₀, discrete chains with N = 32, χN = 3, ζN = 100, δτN = 0.5. The canonical runs take 200,000 steps at each of eight compositions φ = 0.86–0.96, recorded in 20 blocks of which the first is left out. Their exchange chemical potential is fitted with ln(φ/(1 − φ)) plus a quadratic in φ (largest residual 9×10⁻⁶). All compositions use the same random seed, so their noise is correlated: the zero is fitted block by block, and its error is the spread of the 19 zeros. The error of the Gibbs ensemble comes from blocks of 20 cycles.

## Fig. 7.2 of Fredrickson and Delaney

The dilute branch of Fig. 7.2 of Fredrickson and Delaney, *Field-Theoretic Simulations in Soft Matter and Quantum Fluids* (Oxford, 2023): the volume fraction of A in the B-rich phase, from the second half of the cycles, averaged over both phases, with block errors of 0.0001–0.0012.

| χN | C = 4, published | C = 4, `GibbsEnsemble` | C = 10, published | C = 10, `GibbsEnsemble` |
|----|-------------|-----------------------|--------------|------------------------|
| 3.0 | 0.092 ± 0.003 | 0.0917 | 0.078 ± 0.001 | 0.0786 |
| 2.75 | 0.128 ± 0.004 | 0.1301 | 0.110 ± 0.002 | 0.1104 |
| 2.5 | 0.194 ± 0.004 | 0.1932 | 0.162 ± 0.002 | 0.1622 |
| 2.25 | 0.308 ± 0.006 | 0.3219 | 0.258 ± 0.005 | 0.2565 |
| 2.2 | 0.368 ± 0.010 | 0.3721 | 0.284 ± 0.004 | 0.2881 |
| 2.15 | 0.460 ± 0.030 | one phase | 0.328 ± 0.005 | 0.3281 |
| 2.1 | | | 0.376 ± 0.006 | 0.3833 |

Settings of these runs, as published: continuous chains (`rqm4`, ds = 0.01), ζN = 100, a = 0.2 R_g, 24³ grid points in a cell of 4.8 R_g, N̄ = (6^(3/2) C)², with C = (n_A + n_B) R_g³/V. They use `etd` at δτN = 1, 500 steps per cycle and `exchange_rate` 0.5 for 120 cycles from the Flory-Huggins binodal, and 400 cycles near the critical point (C = 4, χN ≤ 2.25; C = 10, χN ≤ 2.2). δτN = 0.5 at χN = 2.5 gives 0.1926 and 0.1623. The published values are read from the error bars of the figure.
