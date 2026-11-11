# Documentation

## Getting Started

| Document | Description |
|----------|-------------|
| [Installation.md](getting-started/Installation.md) | Installing the library |
| [QuickStart.md](getting-started/QuickStart.md) | Running your first simulation |
| [SimulationGUI.md](getting-started/SimulationGUI.md) | Browser editor for preparing simulation scripts over SSH (beta) |
| [PscfConversion.md](getting-started/PscfConversion.md) | Export periodic PSCF input, import fields and generate polymerfts SCFT scripts |
| [Features.md](getting-started/Features.md) | Overview of library capabilities |

## Theory

| Document | Description |
|----------|-------------|
| [ChainPropagators.md](theory/ChainPropagators.md) | Chain propagators, partition functions, and concentrations |
| [PolymerFieldTheory.md](theory/PolymerFieldTheory.md) | Field theory formulations (monomer, auxiliary fields), smeared and electrostatic interactions |
| [SelfConsistentFieldTheory.md](theory/SelfConsistentFieldTheory.md) | SCFT saddle-point equations and iterative methods |
| [ElectrostaticCorrelations.md](theory/ElectrostaticCorrelations.md) | EC-SCFT: self-energy of spread charges, its Green function, the uniform state and the local form for a planar brush |
| [LangevinFieldTheoreticSimulation.md](theory/LangevinFieldTheoreticSimulation.md) | What L-FTS and CL-FTS share: Langevin dynamics, observables and output files, structure functions, renormalization and calibration |
| [PartialSaddlePointApproximation.md](theory/PartialSaddlePointApproximation.md) | L-FTS: partial saddle-point approximation, saddle-point iteration and the `LFTS` class |
| [ComplexLangevinSimulation.md](theory/ComplexLangevinSimulation.md) | CL-FTS: complex Langevin dynamics, correct convergence, stabilization, charged systems, thermodynamic operators and the Gibbs ensemble |
| [StructureFunctionEstimators.md](theory/StructureFunctionEstimators.md) | Derivation of the structure function estimators, their completion with charges, and a comparison with BioFTS |
| [NumericalMethods.md](theory/NumericalMethods.md) | Pseudo-spectral and real-space solvers |
| [StressTensor.md](theory/StressTensor.md) | Stress tensor calculation |
| [SpaceGroup.md](theory/SpaceGroup.md) | Space group symmetry |
| [WellTemperedMetadynamics.md](theory/WellTemperedMetadynamics.md) | Well-Tempered Metadynamics for enhanced sampling |

## Internals

| Document | Description |
|----------|-------------|
| [DeveloperGuide.md](internals/DeveloperGuide.md) | Layers, class hierarchies, ownership, one SCFT iteration, where to change what, development practice |
| [FFTImplementation.md](internals/FFTImplementation.md) | FFT, DCT, DST, CrysFFT implementation |
| [PropagatorSystem.md](internals/PropagatorSystem.md) | Propagator identification and scheduling |
| [MemoryManagement.md](internals/MemoryManagement.md) | Memory optimization and checkpointing |
| [CudaMPS.md](internals/CudaMPS.md) | CUDA Multi-Process Service (MPS) for running multiple simulations on one GPU |

## Validation

| Document | Description |
|----------|-------------|
| [NumericalValidation.md](validation/NumericalValidation.md) | Regression coverage, comparisons with published results and other codes, differences and limits |
| [ExampleValidation.md](validation/ExampleValidation.md) | What is validated for each example, with the compared numbers and figures |
| [BlendGibbsEnsemble.md](validation/BlendGibbsEnsemble.md) | Gibbs ensemble of a blend against Flory-Huggins theory, canonical CL-FTS and Fredrickson and Delaney (2023), Fig. 7.2 |
| [CoacervateGibbs.md](validation/CoacervateGibbs.md) | Gibbs ensemble of a coacervate against one-loop theory and canonical CL-FTS |
| [ComplexCoacervate.md](validation/ComplexCoacervate.md) | Salt-free coacervate against Lee, Popov and Fredrickson (2008) and Danielsen et al. (2019) |
| [Electrolytes.md](validation/Electrolytes.md) | Point ions against Netz (2001), one-loop theory and exact Monte Carlo |
| [BrushMultivalentIons.md](validation/BrushMultivalentIons.md) | EC-SCFT of a brush with trivalent counterions against Duan et al. (2025), Yu et al. (2016) and bulk activities |
| [SoftBeadModels.md](validation/SoftBeadModels.md) | Soft-bead models against Lequieu (2024) |
| [StarNonIntegerArms.md](validation/StarNonIntegerArms.md) | A_nB_n stars with integer and real arm numbers against Kim et al. (2026) |

## References

| Document | Description |
|----------|-------------|
| [Parameters.md](references/Parameters.md) | Complete parameter reference |
| [Tools.md](references/Tools.md) | Utility scripts for visualization and data conversion |
| [Bibliography.md](references/Bibliography.md) | Publication references |
| [Errata.md](references/Errata.md) | Corrections to the publications on the algorithms of this library |
| [RelatedSoftware.md](references/RelatedSoftware.md) | Other polymer field theory software |
