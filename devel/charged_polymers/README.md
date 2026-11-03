# Brush with Multivalent Ions

`multivalent_brush.py` holds the SCFT of a polyelectrolyte brush with multivalent ions and the ion
self-energy of Duan, Agrawal and Wang, Phys. Rev. Lett. 134, 048101 (2025), with the corrections of
[the theory](reports/ElectrostaticCorrelations.md). The chains come from
`PropagatorSolver` with `cn-adi2`; the module writes the Poisson-Boltzmann equation, the self-energy and
the iteration. The three `BrushMultivalentIons*` scripts use it.

| File | Content |
|------|---------|
| `multivalent_brush.py` | The model: `Brush` (chains, ions, residual, damped steps with Anderson mixing, grand potential), and two forms of the self-energy, `LocalSelfEnergy` and `GreenFunction` |
| `BrushMultivalentIons1D.py` | Planar brush with the local self-energy, in the box of Fig. 2 of the paper and in the solution of the experiment of Yu et al. (2016) with a closed cell |
| `BrushMultivalentIonsGreen1D.py` | Planar brush with the Green function solved in the varying ionic strength: the flat and the oscillatory layer of Fig. 4(c,d) |
| `BrushMultivalentIonsLateral.py` | Brush in 2D or 3D: dense caps on a hexagon or ridges against the planar layer, by grand potential, Fig. 4(a,b) |
| `duan_scft_check.py`, `models/duan_scft.py` | Independent discrete-chain check of the paper, written up in [reports/DUAN_SCFT_CHECK.md](reports/DUAN_SCFT_CHECK.md) |

Run a script from this folder:

```bash
python BrushMultivalentIons1D.py
```

The planar scripts run on one CPU core, BrushMultivalentIons1D.py in about four minutes and
BrushMultivalentIonsGreen1D.py in about five. BrushMultivalentIonsLateral.py computes its chains on
the GPU, and its Poisson-Boltzmann equation as well when CuPy is installed; on an A10, ridges 20 nm
apart in 2D take about five minutes and the caps of the default 3D cell about half an hour.
[Validation of the brush with multivalent ions](reports/MultivalentBrush.md)
compares the results with the paper.
