# Phase Examples

Initial configurations of network phases are obtained from the Kevin Dorfman's publicly available data.

## Available Phases

| Phase | File | Space Group | Hall # | Relative CE Free Energy (βΔF/n_0) | PSCF |
|-------|------|-------------|--------|---------------------|------|
| Lamellar | Lamella1D.py | - | - | -0.1429224 | ✓ |
| Cylinder | Cylinder2D.py | - | - | -0.0891607 | ✓ |
| A15 | A15.py | Pm-3n | 520 | -0.8685846 | ✓ |
| BCC | BCC.py | Im-3m | 529 | -0.0899204 | ✓ |
| FCC | FCC.py | Fm-3m | 523 | -0.0889892 | ✓ |
| HCP | HCP_Hexagonal.py | P6_3/mmc | 488 | -0.1345346 | ✓* |
| Sigma | Sigma.py | P4_2/mnm | 419 | -0.4695218 | ✓* |
| SC | SC.py | Pm-3m | 517 | -0.1218046 | ✓ |
| SD | SD.py | Fd-3m | 526 | -0.1927212 | ✓ |
| SG | SG.py | I4_132 | 510 | -0.1973548 | ✓ |
| SP | SP.py | Pm-3m | 517 | -0.1735695 | ✓ |
| DD | DD.py | Pn-3m | 522 | -0.1997210 | ✓ |
| DG | DG.py | Ia-3d | 530 | -0.2131829 | ✓ |
| DP | DP.py | Im-3m | 529 | -0.1445778 | ✓ |
| Fddd | Fddd.py | Fddd | 336 | -0.1606698 | ✓* |
| PL | PL_Hexagonal.py | P6_3/mmc | 488 | -0.2103856 | ✓ |

✓: checked against [PSCF](https://github.com/dmorse/pscfpp) (pscfpp v1.4.0); density and fields agree within machine precision; free energies are compared after converting the reference-chain units and species standard states.

\*: with half the example's ds (the example's ds gives an odd number of contour steps per block, which PSCF rounds to an even number).

---

## Initial Field Generation

The scripts build their initial fields with [`../seeds.py`](../seeds.py), shared with
`../phase_diagram/` and `../crystal_systems/`:

- **Sphere and cylinder phases** (A15, BCC, FCC, SC, Sigma, HCP, Cylinder2D): smoothed wells of
  w_A at the sphere positions (`seeds.wells`); HCP and Sigma give their **Wyckoff positions**.
- **Network phases and Fddd** (DD, DG, DP, SD, SG, SP, Fddd): a converged solution in a .mat file,
  interpolated to the grid of the script (`seeds.from_file`). PL loads a converged PSCF solution
  (PL.mat). The interpolation (`seeds.resample`) treats the fields as periodic on cell-centred grids.
- **Lamella1D**: one cosine period (`seeds.lamellae`).

Example (HCP with P6_3/mmc):
```python
# Wyckoff 2c positions for P6_3/mmc
sphere_positions = [
    [1/3, 2/3, 1/4],
    [2/3, 1/3, 3/4],
]
seed = seeds.wells(params["nx"], params["lx"], sphere_positions)
```

---

## Standard Axis Orderings

| Crystal System | lx Convention | Constraint |
|----------------|---------------|------------|
| Cubic | [a, a, a] | lx[0] = lx[1] = lx[2] |
| Tetragonal | [a, a, c] | lx[0] = lx[1] |
| Hexagonal | [a, a, c] | lx[0] = lx[1] |
| Orthorhombic | [a, b, c] | all independent |

---

## Mesh Reduction

The space group reduces the work in two ways (see [Space Group Theory](../../../../docs/theory/SpaceGroup.md#physical-bases-for-crysfft)):

- **Fields**: SCFT iterates the fields on the irreducible mesh, the grid points that no symmetry
  operation maps onto each other, roughly the number of operations of the group fewer than the grid.
- **Propagators**: the propagators run on a physical basis, a sub-grid on which the transforms work
  directly: one eighth of the grid (m3 basis) for the cubic, tetragonal and orthorhombic examples,
  half the grid (z-mirror basis) for P6_3/mmc, and the irreducible mesh where no physical basis
  applies (SG). The cost of an iteration follows this sub-grid, not the irreducible mesh.

| Phase | Space Group | Grid | Irreducible mesh (grid / mesh) | Propagator basis |
|-------|-------------|------|--------------------------------|------------------|
| A15 | Pm-3n | 64³ | 5472 (48x) | m3, 1/8 grid |
| BCC | Im-3m | 32³ | 408 (80x) | m3, 1/8 grid |
| FCC | Fm-3m | 32³ | 240 (137x) | m3, 1/8 grid |
| SC | Pm-3m | 32³ | 816 (40x) | m3, 1/8 grid |
| SD | Fd-3m | 48³ | 728 (152x) | m3, 1/8 grid |
| SG | I4_132 | 48³ | 2312 (48x) | irreducible mesh |
| SP | Pm-3m | 32³ | 816 (40x) | m3, 1/8 grid |
| DD | Pn-3m | 48³ | 2600 (43x) | m3, 1/8 grid |
| DG | Ia-3d | 32³ | 344 (95x) | m3, 1/8 grid |
| DP | Im-3m | 48³ | 1300 (85x) | m3, 1/8 grid |
| HCP | P6_3/mmc | 48×48×48 | 4908 (23x) | z-mirror, 1/2 grid |
| PL | P6_3/mmc | 48×48×72 | 7362 (23x) | z-mirror, 1/2 grid |
| Sigma | P4_2/mnm | 64×64×32 | 8448 (16x) | m3, 1/8 grid |
| Fddd | Fddd | 84×48×24 | 3024 (32x) | m3, 1/8 grid |

---

Last updated: 2026-11-18
