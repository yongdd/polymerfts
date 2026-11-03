# Brush with Multivalent Ions

Two models of a polyelectrolyte brush with multivalent ions and the ion self-energy of Duan, Agrawal and
Wang, Phys. Rev. Lett. 134, 048101 (2025). Both take the chains from `PropagatorSolver` with `cn-adi2` and
write the Poisson-Boltzmann equation, the self-energy and the iteration themselves.

- `standard_brush.py` is the standard model of this folder: the self-energy in the form that follows from
  [the theory](reports/ElectrostaticCorrelations.md), in the solution of an experiment. It is the smallest
  model found that reproduces the Y3+ titration of the neutron reflectivity experiment of Yu et al.,
  Macromolecules 49, 5609 (2016).
- `multivalent_brush.py` solves the equations of the paper, with the corrections of the theory, and
  reproduces Fig. 2 of the paper.

| File | Content |
|------|---------|
| `standard_brush.py` | The standard model: `Brush`, with the chains, the ions, the self-energy N12, the residual and damped steps with Anderson mixing |
| `BrushYuExperiment1D.py` | The standard model in the solutions of Yu et al.: the Y3+ titration at ionic strength 6 mM in a pool of fitted depth, Y(NO3)3 alone, and NaNO3 |
| `multivalent_brush.py` | The model of the paper: `Brush` (chains, ions, residual, damped steps with Anderson mixing, grand potential), and two forms of the self-energy, `LocalSelfEnergy`, screened by point charges as in the paper or with `spread=True` by charges of the ion radius, and `GreenFunction` |
| `BrushMultivalentIons1D.py` | Planar brush with the local self-energy of the paper, in the box of Fig. 2 of the paper and in the solution of the experiment of Yu et al. with a closed cell; the argument `spread` screens by charges of the ion radius |
| `duan_scft_check.py`, `models/duan_scft.py` | Independent discrete-chain check of the paper, written up in [reports/DUAN_SCFT_CHECK.md](reports/DUAN_SCFT_CHECK.md) |

`BrushMultivalentIonsGreen1D.py` and `BrushMultivalentIonsLateral.py`, which take `multivalent_brush.py`
to Fig. 4 of the paper, are in git at commit `2796421c`.

Run a script from this folder, each on one CPU core, BrushYuExperiment1D.py in about five minutes and
BrushMultivalentIons1D.py in about four:

```bash
python BrushYuExperiment1D.py
python BrushMultivalentIons1D.py
python BrushMultivalentIons1D.py spread
```

[The standard model](reports/StandardBrush.md) gives the reason and the check of each element and compares
the results with the experiment. [Validation of the brush with multivalent ions](reports/MultivalentBrush.md)
compares `multivalent_brush.py` with Figs. 2–4 of the paper.
