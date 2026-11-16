# Validation of the coacervate Gibbs ensemble

What the validation runs check for [CoacervateGibbs.py](../../examples/charged/clfts/CoacervateGibbs.py). The labels are those of [Validation of the Examples](ExampleValidation.md).

## Checks and limits

- Checked (validation runs with the settings of the example): polycation and polyanion (N = 32, z = ±1, discrete chains) in a solvent of one segment, χN = 24 with the solvent, E = 10⁴, N̄ = 1.6×10⁵, ζN = 100, 16³ grid points in a cell of 3 R₀, Gaussian smearing a = 0.2 R₀, `etd` at δτN = 1. All three systems agree with one-loop theory, which adds the Gaussian fluctuations of both homogeneous phases on the same lattice and solves for equal exchange potentials. The salt divides as one-loop theory says: 0.09999 and 0.10001 in both.
- Checked (validation runs): the settings of the exchange do not matter. Seven runs of the charged system with other `exchange_rate`, `cycle_steps` and starting volumes give φ = 0.02468 ± 0.00005 and 0.37090 ± 0.00025 (mean and spread) against 0.02470 and 0.37085 from one-loop theory. The spread exceeds the block error of most single runs, so it is the better estimate of the error.
- Checked (validation runs): canonical CL-FTS. Ten compositions on the two branches (200,000 steps each), solved with the two conditions of the Gibbs ensemble, give φ = 0.02469 ± 0.00004 and 0.37090 ± 0.00014 on 16³ points.
- Checked (validation runs): the grid. On 24³ points in the same cell (a = 1.6 Δx), three Gibbs runs give a dense branch of 0.37012, 0.37060 and 0.37087, with a mean of 0.37053 ± 0.00022, and canonical runs of 110,000 steps give 0.02469 ± 0.00005 and 0.37076 ± 0.00019. Together they lie 0.0002 ± 0.0002 below 16³, where one-loop theory shows no shift. The spread of these runs is several times the block error of one run (0.00005–0.00012). The exchange potentials at δτN = 0.5 agree with those at δτN = 1 within 0.004, the error on these runs, on both grids.
- Checked (validation runs): a smearing shorter than the grid spacing, on 16³ points. At a = 0.1 R₀ (0.53 Δx) a Gibbs run gives 0.02099 and 0.36959, and canonical runs on the two branches 0.02100 ± 0.00004 and 0.36996 ± 0.00014. At a = 0.05 R₀ (0.27 Δx) five Gibbs runs, three from above and two from below, give 0.01836 ± 0.00004 and 0.3511 ± 0.0003, and canonical runs 0.01836 ± 0.00003 and 0.3506 ± 0.0002. Both routes solve the same conditions, so this checks the Gibbs ensemble and not the bias through the pressure described below. `a_int = 0` is rejected for charged chains, which it would leave as point charges.
- Checked (validation runs): finite size. At one grid spacing the cells of 2.25, 3 and 4.5 R₀ differ by 0.0005 in the dense branch, within 0.0004 of one-loop theory in each cell.
- Checked (validation runs): the volumes of the phases do not enter even on a coarse lattice. At a = 0.15 R₀ on 12³ points (a = 0.6 Δx, N̄ = 1.6×10⁵), starting volumes of 37 : 17 and 17 : 37 give 0.37118 ± 0.00015 and 0.37098 ± 0.00007, against 0.37109 from one-loop theory.
- Checked (validation runs): a coexistence that exists only through charge fluctuations. At χN = 22 and N̄ = 10⁴, Flory-Huggins theory gives one phase (critical value 22.16), and so does one-loop theory without charges. Canonical CL-FTS of one box at twelve compositions φ = 0.05–0.30 (60,000 steps each) gives exchange potentials that satisfy the Gibbs-Duhem relation in all eleven intervals (residual 0.0002, error 0.001), and their Maxwell construction agrees with one-loop theory within two errors. In the cell of 4.5 R₀ on 24³ points (twelve compositions, 60,000 steps each) it gives 0.0734 ± 0.0009 and 0.2137 ± 0.0011 against 0.0730 and 0.2117. The dense branch falls with the cell in both routes.
- Limits found: near the critical point one Gibbs run determines neither branch. The volumes and both compositions drift together along a soft mode: the runs in the table lie up to 9 % off the Maxwell construction in φ₀ and up to 3 % in φ₁. Runs from above and from below enclose the Maxwell construction but stay 0.009 apart after 400 cycles, each with exchange potentials equal within noise, and the run in the cell of 4.5 R₀ is still falling after 200 cycles: φ₁ is 0.2176 in the last sixth, 0.004 above its Maxwell construction, with d(μ_P + μ_M) = 0.007 ± 0.006 there against 0.018 ± 0.004 over the second half. The averages of the second half and their block errors understate this uncertainty, so the table gives no errors for them; the Maxwell construction is the sharper route there.
- Limits found: a run can rest off coexistence with exchange potentials that agree within noise, also far from the critical point. At a = 0.05 R₀ one of the five runs stayed at 0.35236 for 100 cycles, 0.0015 above the other four, with block errors of 0.0001. Canonical runs at that composition show differences of 0.007 ± 0.002 in μ_P + μ_M and −0.005 ± 0.001 in 2Nμ_S, where the run itself shows none. Compare independent runs from both sides.
- Limits found: a smearing shorter than the grid spacing leaves a bias through the pressure ([CL-FTS theory, Section 8](../theory/ComplexLangevinSimulation.md#8-gibbs-ensemble)). At a = 0.15 R₀ on 12³ points and χN = 22, the Maxwell construction gives 0.2304 and one-loop theory with the exchange potentials of the boxes 0.2145; the Gibbs run moves toward the latter and reaches 0.2202 after 300 cycles.
- Not checked: published Gibbs-ensemble binodals of a coacervate; the Maxwell construction at χN = 24, where the compositions between the branches phase separate.

## The example

The example (150 cycles, second half; errors from six blocks are about 0.00005 and 0.00014):

| System | φ, Gibbs ensemble | φ, one-loop theory |
|---|---|---|
| Flory-Huggins theory | | 0.0291, 0.3665 |
| Neutral chains | 0.02949, 0.36561 | 0.02950, 0.36607 |
| Charged chains | 0.02469, 0.37033 | 0.02470, 0.37085 |
| Charged chains, salt c = 0.1 of each ion | 0.02614, 0.36956 | 0.02613, 0.37007 |

## One setting changed

The charged system with one setting changed (second half of 200 cycles at `exchange_rate` 0.5 and 500 steps per cycle unless stated):

| Changed setting | φ₀ | φ₁ | φ, one-loop theory |
|---|---|---|---|
| None | 0.02461 | 0.37114 ± 0.00015 | 0.02470, 0.37085 |
| `exchange_rate` 0.2, 300 cycles | 0.02469 | 0.37056 ± 0.00007 | |
| `exchange_rate` 1.0 | 0.02470 | 0.37066 ± 0.00049 | |
| `cycle_steps` 200, 500 cycles | 0.02476 | 0.37108 ± 0.00014 | |
| `cycle_steps` 1000, 100 cycles | 0.02462 | 0.37120 ± 0.00016 | |
| Starting volumes 37 : 17 | 0.02471 | 0.37078 ± 0.00023 | |
| Starting volumes 17 : 37 | 0.02468 | 0.37091 ± 0.00015 | |
| Cell 2.25 R₀, 12³ | 0.02471 | 0.37071 ± 0.00021 | 0.02477, 0.37105 |
| Cell 4.5 R₀, 24³, 120 cycles | 0.02458 | 0.37043 ± 0.00004 | 0.02467, 0.37069 |
| Cell 3 R₀, 24³, 120 cycles, three runs | 0.02471 | 0.37053 ± 0.00022 | 0.02470, 0.37085 |

## Near the critical point

Near the critical point (χN = 22, N̄ = 10⁴, a = 0.2 R₀; Gibbs runs at `exchange_rate` 0.5 unless stated, second half):

| Cell, grid | Route | φ₀ | φ₁ |
|---|---|---|---|
| 3 R₀, 16³ | One-loop theory | 0.0731 | 0.2163 |
| 3 R₀, 16³ | Maxwell construction, degree 5 (degree 6) | 0.0712 ± 0.0011 (0.0717 ± 0.0013) | 0.2172 ± 0.0015 (0.2166 ± 0.0018) |
| 3 R₀, 16³ | Gibbs from 0.06 and 0.24, 400 cycles | 0.0753 | 0.2215 |
| 3 R₀, 16³ | Gibbs from 0.085 and 0.20, 400 cycles | 0.0708 | 0.2124 |
| 3 R₀, 16³ | Gibbs from 0.06 and 0.24, `exchange_rate` 1.0, 300 cycles | 0.0736 | 0.2183 |
| 2.25 R₀, 12³ | One-loop theory | 0.0732 | 0.2199 |
| 2.25 R₀, 12³ | Maxwell construction, degree 5 | 0.0699 ± 0.0022 | 0.2197 ± 0.0030 |
| 2.25 R₀, 12³ | Gibbs from 0.06 and 0.24, 400 cycles | 0.0762 | 0.2215 |
| 4.5 R₀, 24³ | One-loop theory | 0.0730 | 0.2117 |
| 4.5 R₀, 24³ | Maxwell construction, degree 5 (degree 6) | 0.0734 ± 0.0009 (0.0733 ± 0.0011) | 0.2137 ± 0.0011 (0.2138 ± 0.0013) |
| 4.5 R₀, 24³ | Gibbs from 0.06 and 0.24, 200 cycles (not converged) | 0.0766 | 0.2206 |
