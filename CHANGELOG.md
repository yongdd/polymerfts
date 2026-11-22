# Changelog

Notable changes to PolymerFTS. The project follows a loose semantic versioning scheme: the
minor version changes when results or the Python interface change, the patch version for
fixes that leave converged results unchanged.

Commit hashes refer to the `main`/`devel` history on GitHub.

## [Unreleased]

### Installation and platforms

- CI builds the package with MKL alone on Linux, checks that `cpu-mkl` is its only platform and runs
  the whole test suite (2026-11-18, the whole suite since 2026-11-19, so that the numerical tests
  run on cpu-mkl too). Pushes run CI on `devel` only (2026-11-21): `main` receives only merge
  commits that `devel` already holds, and a second push of the same commit left a cancelled run on
  it. The script tests without the package run inside the Linux CPU job instead of a job of their
  own.
- The package classifiers list Python 3.13 with 3.11-3.15 (2026-11-21).
- `import polymerfts` sets `OMP_NUM_THREADS=1` and `OMP_STACKSIZE=1G` unless they are set, before
  NumPy and the native module load (2026-11-18). Since 0.10.0 `scft`, `lfts` and `clfts` set them
  after libgomp had read its environment, so they had no effect: the CPU Fourier transforms used every
  core, and NumPy's OpenBLAS ran 64 threads on a 64-core node. Installation describes the threads,
  including `OPENBLAS_NUM_THREADS` for callbacks that call BLAS (docs/getting-started/Installation.md).
- Clang builds with `POLYMERFTS_WERROR=ON` (2026-11-15; Clang 19 as the C++ compiler, with and
  without CUDA): a missing `override` is added and an unused member of the MKL trigonometric
  transform is removed; the unused variables of eight C++ tests are removed or marked
  `[[maybe_unused]]`.
- Outside MSVC the host C++ code is compiled with `-ffp-contract=off` (2026-11-09). With FMA in the
  target (`-march=x86-64-v3`, `native`, or a compiler whose default target has it, as on the CI
  runners) GCC fused products into additions, and full- and reduced-memory computations, which
  group the same additions differently, differed in the last bit: the stress of
  `TestBlockMultiplicity` with MKL in CI. MKL, FFTW and the `.cu` files are unaffected. A
  floating-point model given in the flags (`-ffp-model`, `-fp-model`, `-ffp-contract`,
  `-ffast-math`, `-Ofast`) is kept, with a configuration message (docs/getting-started/Installation.md).
- The GitHub repository is renamed from `langevin-fts` to `polymerfts`, the name of the
  package and its import (2026-10-27); GitHub redirects the old URLs.
- The Conda environment and CI use conda-forge (Miniforge) instead of Anaconda's
  `defaults` channel, on Linux and Windows (2026-10-24). OpenBLAS is selected with
  `libblas=*=*openblas` and supplies LAPACKE through `openblas`; spglib comes from
  conda-forge with its C library; the MKL backend installs `mkl-devel`.
- The supplied environment uses Python 3.15 and pins every package to an exact version
  (2026-10-26), so a new environment gets the libraries tested on Linux and Windows:
  among them NumPy 2.5.4, SciPy 1.18.1, OpenBLAS 0.3.34, spglib 2.8.0, CMake 4.4.4 and
  pybind11 3.1.0. The installation commands and CI pin MKL 2026.1.0 and FFTW 3.3.11.
  Python 3.15 runs in UTF-8 mode, so output redirected to a file on Windows keeps the
  Greek symbols of the console output. The conda recipe builds for Python 3.11-3.15 and
  no longer pins NumPy or the pybind11 ABI, and the CI lint job uses Python 3.15.
  The environment no longer installs `git`: on Windows, conda-forge's git brings MSYS
  tools such as `link.exe` that shadow the MSVC linker when conda's paths come first,
  and a fresh CUDA configuration then failed with a misleading message about nvcc.
- Windows builds with MSVC (2026-10-24): CMake finds conda's libraries under
  `<prefix>/Library`, uses MSVC warning flags, the conforming preprocessor that the CUDA
  headers need and MSVC's LLVM OpenMP runtime (CMake 3.30+; configuration stops with an
  error on older versions), names the module
  `_core.pyd` and defines `FFTW_DLL`, without which `fftw_version` came from MKL's FFTW
  wrappers. LAPACKE uses `std::complex` under MSVC, the package adds the spglib DLL
  directory before importing `_core`, and the tests no longer use variable-length arrays
  or POSIX-only calls. The out-of-memory check of `TestCudaDeviceBuffer` runs as its own
  ctest entry, `TestCudaDeviceBufferOutOfMemory`, which is skipped (exit code 77) when the
  driver backs device allocations with system memory once the device memory is full, the
  default sysmem fallback of WDDM drivers; the construction then cannot run out
  (2026-10-26: before, the skip covered the whole test).
- CI builds and runs the basic tests on Windows with MSVC and Ninja (MKL + FFTW) besides
  Linux (2026-10-26). The Windows job also runs `tests/scripts`, whose symbolic-link tests are
  skipped where links cannot be created (Windows without Developer Mode).
- The server subfolder of the simulation script editor and the output paths of the PSCF converter,
  which its `simulation.py` writes, refuse any path with a root or a drive, and `:` in folders and
  filenames, on every platform (2026-10-26, 2026-11-17): Windows does not count a path such as
  `/tmp/x` as absolute but roots it on the current drive, and `C:x` is relative to the current
  folder of drive C (tests/scripts/TestSimulationWeb.py, TestPscfScript.py).
- spglib is a dependency of the Python package, not only of its `full` extra (2026-10-27): `_core`
  links the spglib library of the pip package, so a wheel installed without that package could not
  import `_core`.
- `CudaCommon.h` includes `<cuda_runtime.h>` itself (2026-10-27). Host-compiled sources reached the
  CUDA runtime API only through Thrust, whose CUDA 12.9 headers no longer include it outside nvcc, so
  the Python module failed to compile with CUDA 12.9.

### Added

- CUDA runs boxes with walls of any size, and complex fields with walls for continuous chains
  (2026-11-22): odd sizes along absorbing walls, more than 6142 points along an absorbing wall, and
  `rqm4`/`rk2` with complex fields and walls, which ran on the CPU before (`platform="auto"`) or were
  refused. `platform="auto"` runs every 2D and 3D box on CUDA when it is available. Results agree
  with the CPU to 4e-14 (relative).
- Electrostatics in boxes that mix periodic axes and walls (2026-11-11): the platform Fourier
  transform gives such a box the DFT along the periodic axes and the cosine or sine series along the
  walls, computed with one FFT of the grid whose wall axes are reordered and combined as in Makhoul's
  cosine transform: `MakhoulTransform` with periodic axes on the CPU and `CudaMakhoulTransform` (one
  cuFFT of the grid between a reordering and a combining kernel) on CUDA (both 2026-11-16;
  docs/internals/FFTImplementation.md, Section 2.5).
  The coefficients are complex, the full FFT spectrum along the periodic axes and the cosine or sine
  series along the walls, also for real fields, which take the real part of `backward`.
  `PropagatorSolver.fourier` takes the cosine series along every wall, absorbing ones included, so
  for electrostatics and smearing every wall is a wall of vanishing dielectric constant. CL-FTS and
  SCFT accept `electrostatics` in such boxes (SCFT with `cn-adi2` or `rs`, CL-FTS, whose fields are
  complex, with `rs`), for example a brush with periodic lateral axes, and smearing there uses the
  platform transform instead of SciPy. In a CL-FTS box a few Debye lengths wide the images in the
  lateral walls deplete the ions across the whole box. `TestFourierTransform` checks the mixed transforms against NumPy/SciPy on every platform, and
  `TestConfinedElectrostatics` the Poisson-Boltzmann solution and the CL-FTS forces in a box
  periodic along one axis.
- `examples/charged/ecscft/BrushMultivalentIons.py` (2026-11-11): a polyelectrolyte brush that
  trivalent counterions collapse, in EC-SCFT (Duan, Agrawal and Wang 2025): every charge carries
  its self-energy in the ionic strength around it, screened by charges of their own spread or, as
  in the paper, by point charges. The chains are discrete, one charged bead per segment as in the
  paper (`rs`), and with the spread screening each charge feels the mean potential averaged over
  itself. The script writes the iteration around `PropagatorSolver` in the Na⁺/Y³⁺/NO₃⁻ solution
  of the experiment of Yu et al. (2016).
  docs/theory/ElectrostaticCorrelations.md derives the equations and lists where they differ from
  the published ones; docs/validation/BrushMultivalentIons.md compares the model with Figs. 2–4 of
  the paper, the experiment, bulk activity coefficients and CL-FTS of the same brush.
- MAT files keep names that a MAT structure cannot hold as fields (2026-11-14), such as monomer
  names that start with an underscore or exceed the field-name limit, and non-ASCII names: they are
  stored under reversible identifiers listed in `polymerfts_mat_keys_json`, and
  `polymerfts.utils.mat_fields.decode_mat_fields` restores them after `scipy.io.loadmat`. SciPy
  dropped such fields with a warning or refused the file. Ordinary ASCII names are unchanged (docs/references/Parameters.md; tests/TestMatFieldNames.py).
- Simulation metadata also hash the source file of the simulation class when it lies outside the
  package (`simulation_source`, `simulation_source_sha256`; 2026-11-16), such as a modified copy of
  `scft.py` as QuickStart describes or a subclass in a script; `python_source_sha256` covers the
  package only (tests/TestPackageHashes.py).
- Blocks take a `multiplicity` (2026-11-08): a block (v, u) with multiplicity m stands for m copies
  of itself and of everything on its u side, attached at v, so a star with n identical arms is given
  by one arm. m may be any real number >= 1: the junction takes the propagator of that branch to the
  power m (m - 1 for the propagator entering a copy), and the copies count m times in the
  concentrations, the stress and the chain length alpha. This continues the arm number analytically,
  e.g. the symmetric A_n B_n stars around their tricritical arm number n_tc ≈ 5.4475 (Kim et al.,
  Macromolecules 59, 11903 (2026)). Integer multiplicities reproduce the explicit arms: the same
  partition functions, concentrations and stress equal to round-off, and, without propagator
  aggregation, the same keys and junction products. Multiplicities lie between 1 and 10⁶
  (2026-11-16), which keeps the binary exponents of the junction products exact; SCFT and L-FTS take
  real ones, CL-FTS and complex fields integers only: a propagator computation for complex fields
  refuses, when it is created, a junction power that is not an integer of at most 10⁶, from a
  non-integer multiplicity or from identical branches whose powers sum beyond 10⁶. A junction keeps
  the range of its whole product: in a star with n arms in a strong field q_A^(n-1) can overflow
  while the product stays finite (tests/TestJunctionRange.py). A real power of a negative numerical
  propagator (Crank-Nicolson next to absorbing walls, a pseudo-spectral method in a strong field)
  stops the computation with an error. Polymers with multiplicities are computed without propagator
  aggregation; `get_block_concentration` gives all copies of a block. The drawings
  `polymer_<id>.png` mark a block with copies by ×m, and the PSCF export refuses multiplicities
  (docs/theory/ChainPropagators.md, Section 4.3; docs/validation/StarNonIntegerArms.md;
  examples/block_copolymer/scft/StarNonIntegerArms1D.py; tests/TestBlockMultiplicity.py).
- Discrete chains take `numerical_method: "rs"` (2026-11-05): the bond is summed on the grid,
  one stencil per axis, instead of through FFTs, on the CPU and CUDA backends for real and complex
  fields. Every concentration stays positive where fields change by hundreds of kT per chain
  (grafted and charged brushes), where the FFT round-off of the default `"ps"` turns them negative.
  Like `cn-adi2` it mixes periodic and non-periodic axes and takes a different wall at each end of an
  axis, and it needs an orthogonal cell; with a space group its propagators stay in the reduced
  basis. Its stress is the derivative of its own free energy, so the box lengths relax and the
  CL-FTS pressure works as with `"ps"`; angles stay at 90 degrees. A narrow absorbing axis, where
  the images of the walls would cancel, multiplies by the dense n × n matrix of its folded kernel,
  summed once per box from a compact sine/cosine expansion: n multiply-adds per point and n² stored
  values per kernel along that axis (docs/theory/NumericalMethods.md, Section 4.3).
- `compute_stress()` takes complex fields with reflecting and absorbing walls (2026-11-06): the
  pseudo-spectral methods multiply the cosine and sine coefficients of the real and imaginary
  parts of the propagators as complex numbers, on the CPU and for `"ps"` on CUDA, and `"rs"` sums
  on the grid. Before, the stress of complex fields needed a periodic box
  (tests/TestStressComplexWalls.py).
- `save_vtk` takes `space_group` (2026-11-06): the group of the fields, `calc.sg`. A
  hexagonal-axes setting samples the lattice axes a and b at the nodes, i/n, and its points are
  written there, as `PropagatorSolver.get_grid_points` places them; without the argument these
  fields are drawn half a grid cell off along a and b.
- The simulation GUI exposes Tetragonal and Monoclinic 3D cells, and Hexagonal
  and Oblique 2D cells, with their length/angle constraints and the corresponding
  SCFT `crystal_system` in exported scripts (2026-10-30).
- The density notebooks of `scripts/analysis` draw every monomer type: `plot_2d_density.ipynb` all
  types of the file, where it drew A and B (2026-11-06), and `plot_3d_isodensity.ipynb` the type set
  in `monomer_type`, the first type of the file by default, where it read type A only (2026-11-17;
  tests/scripts/TestDensityPlot2D.py, TestIsodensityPlot.py).
- CL-FTS records the partition ratio |Q_s|/<|e^(-u_s)|> of every step for each canonical
  single-site species, discrete chains of one segment and point ions
  (`partition_ratio_step_history`, columns named by `partition_ratio_species_json`; 2026-10-29).
  It is 1 for real fields and approaches 0 near a zero of Q_s, the pole of -n_s ln Q_s. The end
  of a run prints its smallest value and the fractions of steps below 0.1 and 0.01,
  `scripts/analysis/drift_tail.py` summarizes the records, and checkpoints keep the pending
  steps (NaN for the steps of a checkpoint without the record).
- SCFT runs space groups of every crystal system (2026-10-29); release 0.10.0 refused monoclinic and
  triclinic groups. With `relax_cell` a monoclinic group frees the angle between the two axes
  perpendicular to its unique axis (β for the unique axis b, γ for c, α for a) and a triclinic group
  all three angles. The simulation script editor lists all 530 Hall settings, and the PSCF export
  writes a monoclinic group with the unique axis b as PSCF's monoclinic lattice and one with the
  unique axis a or c as a triclinic lattice with a fixed cell, which it refuses with box relaxation
  (docs/theory/SpaceGroup.md; tests/TestScftObliqueSpaceGroups.py).
- `recording.concentrations` (CL-FTS, 2026-11-01) records the mean concentration of every monomer
  type and point ion on the grid in the `structure_function_NNNNNN.mat` files
  (`concentration_<name>_real`/`_imag`,
  `concentration_step_count`), over every step of each interval at no extra cost. Checkpoints keep
  the pending sums; a continued run reproduces them bit for bit, and a run continued from a
  checkpoint without them averages from the restart on.
  `examples/charged/clfts/CounterionsBetweenPlates.py` uses it instead of a callback.
- `examples/charged/clfts/CoacervateGibbs.py` (2026-11-01): the coexistence of a coacervate and
  its supernatant from `GibbsEnsemble`, with neutral chains, with charges and with a salt of point
  ions; the charges widen a coexistence that the fluctuations of neutral chains narrow.
- `examples/charged/scft/SaltDopedLamella.py` (2026-10-29): a salt that the B block of a lamella
  solvates, with the ions as molecules of one segment that carry their own chi, and the
  Coulomb interaction dragging the weakly selective anion after the cation.
  docs/references/Parameters.md and the theory pages describe how ions get chi this way
  (point ions have none). docs/validation/ExampleValidation.md records the check against the
  locally neutral limit: as E grows, the lamella approaches that of one neutral salt molecule
  with the mean chi, linearly in 1/E.
- Reproductions of published results as examples and validation records (2026-10-28):
  `examples/charged/scft/CounterionsBetweenPlates.py` (Netz 2001, Poisson-Boltzmann between plates),
  `examples/charged/clfts/CounterionsBetweenPlates.py` (Netz's loop expansion at weak coupling),
  `SmearedSaltSelfEnergy.py` (Born self-energy and screening; Wang 2010, Agrawal and Wang 2022),
  `PolyampholyteSolution.py` (McCarty et al. 2019), `ComplexCoacervate.py` (Lee, Popov and
  Fredrickson 2008), `examples/block_copolymer/clfts/SoftBeadDiblockMelt.py` (Lequieu 2024) and
  `examples/blend/clfts/HomopolymerSolutionBook.py` (Fredrickson and Delaney 2023).
  docs/validation/NumericalValidation.md lists the papers and codes (PSCF, BioFTS, polycomp)
  that polymerfts reproduces, and docs/validation/ExampleValidation.md gives the compared numbers, including
  exact Monte Carlo checks of the CL-FTS of point ions and the results that are not reproduced
  (the phase diagram of Lee et al.; the converged dense branch of Danielsen et al., 3.4–5 %
  below the paper, a difference that the paper's contour step Δs = 0.01 mostly explains).
- Documentation: a closed box is the boundary to a medium of vanishing dielectric constant, so
  ions are repelled by images of the same sign; the checked range of charged CL-FTS (counterions
  between plates up to Ξ ≈ 0.6, the O(δτ) bias of the ψ step); the recorded pressure dilates the
  box at a fixed number of grid points, which differs from the pressure at fixed grid spacing when
  the smearing is not resolved; comparisons with BioFTS (with the corrections of its issue 1, which
  removes the time-step dependence the structure-function comparison reported) and polycomp.
- Documentation: docs/theory/WellTemperedMetadynamics.md, Section 3.5, gives Ψ and its bias as the
  code computes them, from scaled modes, with no bias at Ψ = 0 and at zero modes (2026-11-17); it
  showed the formula of deep-langevin-fts, which gives NaN at Ψ = 0.
- Documentation: the validation documents moved to `docs/validation/`, and the five longest entries
  of `ExampleValidation.md` have their own pages there (2026-11-01).
- Documentation: `examples/README.md` is a short index of the examples, and
  `docs/validation/ExampleValidation.md` holds what the tests and validation runs check for each
  example, with the compared numbers and figures (2026-10-29).
- Documentation: a system of point ions alone takes a neutral solvent of one segment as the
  reference chain, with `zeta_n: 0` so that it has no field (2026-10-29). With ζN > 0 the
  field of the solvent made CL-FTS diverge below about 0.01 solvent molecules per grid cell;
  `clfts/CounterionsBetweenPlates.py` and `clfts/SmearedSaltSelfEnergy.py` use ζN = 0, which
  leaves ψ and the ions bit for bit unchanged.
- Examples of charged systems (2026-10-28): `examples/charged/scft/ChargedDiblockLamella.py` relaxes
  the period of a charged diblock lamella with point counterions over a range of added salt,
  `examples/charged/scft/PolyelectrolyteBrush.py` solves a charged brush grafted to a charged wall of a
  closed box with discrete chains and the real-space `rs` method, and `examples/charged/clfts/SaltReservoir.py` runs a
  charged diblock melt in contact with a salt reservoir (grand-canonical point ions) and compares
  the uptake of ions with the Donnan equilibrium. `examples/block_copolymer/scft/crystal_systems/`
  relaxes cylinders in hexagonal, oblique and triclinic cells and a lamella in a monoclinic cell, and
  scans the free energy against the angle of fixed cells.
- Electrostatics in closed boxes and fixed charges (2026-10-28): `electrostatics` accepts
  boxes with walls on every side, where ψ lives in the cosine series (no normal derivative at
  the walls), in CL-FTS and SCFT, and `electrostatics.fixed_charge`, a fixed charge density on
  the grid that adds <ψ c_f> to H, c_f to the charge density and its mean to the
  electroneutrality and the counterions. With walls the charge structure function is not
  recorded, the structure functions lack the ψ completion, and the SCFT stress uses the cosine
  modes of ψ and of the ions. SCFT takes a fixed charge in the axis order of `nx` with
  `axis_ordering`, like the initial fields. `TestConfinedElectrostatics` checks the
  Poisson-Boltzmann limit between charged layers against an independent solution, the forces,
  restarts, the stress and pressure with fixed charges in a periodic box, the SCFT stress in a
  closed box against dF/dL and charged chains grafted to a wall.
- Gibbs-ensemble CL-FTS (`GibbsEnsemble`, 2026-10-28): two CL-FTS boxes exchange chains,
  canonical point ions and volume at fixed totals until the exchange potentials
  βμ_p - α_p βΠR₀³/C (βμ_s for ions) agree; with charges, no screening and only canonical ions
  only neutral combinations move (a chain whose charges cancel to the relative precision of the
  charge check counts as neutral), while charged grand-canonical ions give each box its Donnan
  potential and every species moves. Every species needs a positive amount in both boxes.
  Both boxes run in the fixed cell `lx`, the intensive Gibbs ensemble of Mester, Lynd and
  Fredrickson (2013) (2026-11-01; docs/theory/ComplexLangevinSimulation.md, Section 8): the amounts
  and volumes count the material for the lever rule, the top-level `lx` is required as for `CLFTS`,
  and the optional `boxes[].volume` sets the volume of a phase at the start. A cell that followed
  the volume at one grid would give the boxes different grid spacings, and the part of the chemical
  potentials that depends on the spacing would drive them apart. The ensemble agrees with 11 of the
  13 points of Fredrickson and Delaney (2023), Fig. 7.2, within about one error. The amounts move
  with the inverse Hessian of the ideal mixture at the reference density (2026-11-01), so that the
  exchange of volume along a tie line keeps pace with the composition also when a solvent is much
  shorter than the chains. Each cycle continues the boxes from their fields files (fields, ψ and
  the previous noises) with the uniform modes moved to the homogeneous saddle point of the new
  composition; `run()`
  starts from the input boxes and `continue_run()` continues an interrupted run from
  `gibbs_history.json`; the history hashes, and the boxes take, only the keys of the input that
  CL-FTS reads, so the options of other methods may be dropped before `continue_run()`
  (2026-11-17). CLFTS gains `draw_architectures` and `run(psi_noise_prev=...)`.
  `TestGibbsEnsemble` checks, in the mean-field limit, the Flory-Huggins binodal of a blend,
  the coexistence of charged chains with canonical ions (neutral combinations) and with
  grand-canonical salt (Donnan potentials) against independent solutions, the input checks and
  bitwise restarts; `examples/blend/clfts/GibbsEnsemble.py` runs a blend. The PSCF export refuses
  a `gibbs_ensemble` section, whose two boxes one PSCF input cannot describe.
- Mean-field electrostatics in SCFT (2026-10-28): SCFT accepts `electrostatics` (with
  `coupling`), point ions and grand-canonical ions, and iterates the electrostatic potential ψ
  with the fields. The residual of ψ is the Newton step of the linearized Poisson equation with
  the charge response of the homogeneous state, which keeps the iteration stable for large E.
  The free energy adds -(1/2E)<ψ(k² + κ²)ψ> and the ions, the homogeneous reference the
  electrostatic energy of the homogeneous state (Donnan potential or screened net charge), and
  the stress the Coulomb kernel and the ion smearing. `SCFT.cell.stress(w, phi)` gives the
  lattice derivatives at fixed fields, `run(initial_psi=...)` an initial ψ (real and finite, else
  `ValidationError`; tests/TestScftElectrostatics.py), and the results
  `psi` and `ion_chemical_potentials`; canonical ions enter `grand_potential` with
  -N c_s βμ_s, and `MeanFieldElectrostatics.ion_densities(psi)` gives the densities and ln Q of
  the ions as in CL-FTS. The coupling must be positive, `space_group` is refused
  (`ValidationError`), and `coexistence_quantities()` rejects results with electrostatics.
  `TestScftElectrostatics` checks the neutral reduction, the
  strong-screening limit (χ_eff), the stress against dF/dL, the homogeneous states, the
  equivalence of canonical and grand-canonical ions, a fixed charge with `axis_ordering` and
  the platforms.
- CL-FTS grand-canonical species (2026-10-28): a polymer with `chemical_potential` in place of
  `volume_fraction`, or a point ion with `chemical_potential` in place of `concentration`, has a
  fixed chemical potential βμ (in the convention of the recorded chemical potentials) instead
  of a fixed amount. Its term of H is -e^(βμ) Q per reference chain, its concentration
  e^(βμ) times the unnormalized density of the chains or ions, and the volume fractions of the
  canonical polymers sum to at most 1. The chemical potential of a polymer must lie in
  [-700, 700], where the fugacity e^(βμ) is a normal double, or `ValidationError` is raised. A
  charged grand-canonical species makes the uniform mode of ψ evolve as the Donnan potential, so
  its input need not be electroneutral (a polyampholyte without net charge does not). The
  osmotic pressure includes their ideal gas e^(βμ) Q and conformations, and
  `number_density_history` records the numbers of chains and ions of all species. The partition
  check of a grand-canonical species is normalized by the integral of the absolute integrand, so
  that cancellation near Q = 0 does not amplify round-off (`TestPartitionIntegrand`).
  `TestClftsGrandCanonical` checks the agreement of the ensembles on the same fields, the
  forces, the Donnan and Flory-Huggins mean-field limit with LM and etd, the pressure against
  -dH/dV at fixed chemical potentials and restarts.
- CL-FTS point ions (`electrostatics.ions`, 2026-10-28): ions with a valence, a Gaussian smearing
  length and a concentration per segment of the reference density, without volume and without
  χ. An ion feels z_s (Γ_s * ψ)/N like a segment, H gains -N c_s ln Q_s and the charge density
  the smeared ion charge. The counterions that neutralize the polymers are added to one ion
  species (`counterion`, automatic when unambiguous). The ions add their chemical potentials
  (after the polymers), their ideal gas and smearing dilation to the osmotic pressure, and their
  charge to the charge structure function. `electrostatics.n_segments` gives the number of
  segments of the reference chain, which continuous chains need for point ions and for
  `bjerrum_length`. `Electrostatics.ion_densities()` gives the densities and ln Q of each ion
  from one weight e^(-(u - u_min)) per ion, so that ln Q stays finite in a potential beyond
  ±709 kT. `TestClftsPointIons` checks the forces, the linear response, the
  pressure against -dH/dV, restarts and the Debye-Hückel statistics of ψ.
- Fourier transforms of whole fields, `PropagatorSolver.fourier` (2026-10-27): the same
  interface on CPU (FFTW, MKL) and CUDA (cuFFT) for `forward`/`backward` transforms and for
  Fourier multipliers, `apply([(M, fields), ...])` = backward(Σ M(k) · forward(fields)) with
  matrix or diagonal multipliers stored once by `set_multiplier` (in GPU memory with
  CUDA). `k_squared()` gives |k|² in the coefficient layout: the `rfftn` half spectrum for
  real fields, the full spectrum for complex fields, the cosine series along walls. For real
  fields a multiplier acts as a real operator, made Hermitian where the half spectrum holds
  both k and -k, so every backend gives the same result.
  `AbstractFactory::create_fourier_transform()`; `TestFourierTransform`; explained step
  by step in `examples/tools/FourierTransform.py`.
- `SpaceGroup.fixed_angles()` gives the angles a space group fixes (None where an angle is free),
  which the cell check, SCFT box relaxation and SCFT's choice of the physical basis use,
  `FourierTransform.alias_moments()` the tied-alias moments of the box of a solver, and
  `minimal_alias_k_squared(nx, lx, angles)` and `minimal_alias_moments(nx, lx, angles)` of the
  native module |k|² and these moments for every mode of a periodic grid on a given cell, with the
  alias search of the solvers, which the smearing and WTMD use (2026-11-17). The smearing takes the
  angles the box was given, so that its |k|² equals that of the solver bit for bit also after each
  step of a box relaxation.
- CUDA runs boxes with reflecting walls on odd grid sizes, like the CPU backends (2026-10-26).
  Release 0.10.0 refused every odd non-periodic size, although only the DST-2 of absorbing axes
  needs an even length, which the error message now names. `TestCudaOddReflecting` compares CUDA
  with the CPU in 1D, 2D and 3D boxes, and `TestCudaFFTMixedBC` checks odd lengths against direct
  DCT and DST sums.
- CL-FTS integrator `"etd"` (`langevin.integrator`, 2026-10-26): Leimkuhler-Matthews steps
  whose linear part is integrated exactly for each wavevector. Before the run `CLFTS`
  measures the Jacobian of the forces at the homogeneous state with complex-step
  derivatives, one computation of the concentrations per field for all wavevectors, so it
  holds the response of any chain architecture, the smearing and, with electrostatics, ψ,
  which this integrator advances together with the auxiliary fields. For a linear force
  the step is the exact flow and its stationary covariance is exact for any time step; the
  noise is averaged over two steps as in LM. Eigen-directions that are unstable at the
  homogeneous state (near the order-disorder transition and in ordered phases) take LM
  steps and limit the time step there. In the disordered diblock of
  `TestStructureFunctionRPA`, LM diverges at δτN = 4, while `etd` stays within 1 % of the
  RPA up to δτN = 64; the test runs it at δτN = 16. Integrators declare the hook with
  `linear_response` and `set_linear_response()`; each simulation keeps its own copy of such
  an integrator (a stateful one captures its initial state after the response is prepared),
  and L-FTS rejects them. `TestClftsEtd`; `tests/validation/validate_etd_order.py`
  measures the order of the time-step error.
- CL-FTS chemical potentials and osmotic pressure (2026-10-26): every sample records
  βμ_p = ln(φ_p/α_p) - ln Q_p of all species (`chemical_potential_history`), and with
  `recording.pressure` the osmotic pressure ∂ln Z/∂V (`pressure_history`) in its virial form
  (`sim.thermodynamics.pressure_virial`, 2026-10-29). It adds to the dilation operator, which
  `sim.thermodynamics.pressure` returns (the chain stress, the box dependence of the smearing
  functions, the field terms of H, the normalizations of the Gaussian field integrals and the terms
  of ψ), the Schwinger-Dyson terms
  (1/2)[mean(Ω dH/dΩ) - n_g/(C V/R₀³)] of every auxiliary field and the same, mode by mode, for ψ.
  Their exact average is zero; they remove the Gaussian fluctuation of the field terms and
  normalizations, of order sqrt(n_g/2)/(C V/R₀³) per field and sample, so the standard deviation
  per sample is 0.078 instead of the 0.41 of the dilation operator in the diblock of
  `TestClftsThermodynamics`, and 0.013 instead of 10.3 in the homopolymer solution of
  `HomopolymerSolutionBook.py` at L = 1.6 R_g with 16³ grid points. Checkpoints keep the
  pending samples; one without them gives NaN for those samples. With the pressure and
  `numerical_method: "rk2"`, `CLFTS` warns when the contour step b sqrt(ds) R₀ of a monomer type
  exceeds its smearing length, or with no smearing (2026-10-29): the O(ds²) contour error of the
  `rk2` pressure is then large, 0.05 to 0.07 per C/R₀³ in the coacervate of Danielsen et al. at
  ds = 0.01.
  `TestClftsThermodynamics` checks the pressure against -dH/dV at fixed grid values, both
  operators against Flory-Huggins at the homogeneous saddle point and when the warning appears.
- Screened electrostatics (`electrostatics.screening_length`, 2026-10-26): a background
  electrolyte that is not simulated explicitly turns the Coulomb kernel into E/(k² + κ²),
  with the screening length 1/κ, the Debye length of the background alone (explicit ions
  screen in addition). The uniform mode of ψ then fluctuates and the explicit species may
  carry a net charge; unscreened runs keep their checkpoint settings.
- CL-FTS records the largest drift over the grid of every Langevin step, for each auxiliary
  field (`drift_max_step_history`) and for ψ (`psi_drift_max_step_history`), in the `dH`
  files (2026-10-26). Checkpoints keep the steps since the last statistics file, and the
  end of a run prints a summary of the distribution. The tail of this distribution decides
  whether complex Langevin averages converge correctly (Nagata, Nishimura & Shimasaki,
  PRD 94, 114515 (2016)); `drift_max_history` holds only the sampled steps.
  `scripts/analysis/drift_tail.py` reads the records of a run and prints the tail and its
  log-log slopes. `TestDriftStepHistory` checks the records, restarts and checkpoints.
- CL-FTS of charged systems (2026-10-24):
  - The optional `electrostatics` input gives valences per segment (`charges`) and the
    Bjerrum length (discrete chains, N = 1/ds) or the coupling E; unknown keys raise
    `ValidationError`.
  - The charges interact through the smeared charge density with the Coulomb kernel E/k².
    A fluctuating electrostatic potential ψ carries this interaction. Its noise comes from
    the jumped stream of the auxiliary fields (`PCG64.jumped()`), so runs with neighboring
    seeds share no stream. ψ is kept in Fourier space, where its stiff linear part is
    integrated exactly for each wavevector. Like the Leimkuhler-Matthews integrator of the
    auxiliary fields, ψ takes the mean of the current and the previous noise
    (`psi_noise_prev` in checkpoints) when the integrator declares `averaged_noise = True`,
    as `lm` does; `euler` declares fresh noise, and a custom integrator used with
    electrostatics must declare one or the other. Its Gaussian statistics are then exact for
    any time step, although the charge density of each step is explicit. With fresh noise
    alone, as before and still with the Euler integrator, the variance of ψ is biased at
    first order in the time step: in `TestChargedStructureFunctionRPA` the charge structure
    functions from ψ lay 2 % above the RPA, and the field-field form up to 30 % below it.
  - Charged block copolymers and counterions are ordinary polymers and one-segment
    molecules in `polymers`; the input must be electroneutral.
  - Smearing is required with charges, with a positive Gaussian length for each charged
    type, and Gaussian smearing in CL-FTS accepts one length per monomer type.
  - Checkpoints store ψ, its Fourier transform (for bitwise continuation), the previous
    noise and the stream of ψ, with the pending drifts of ψ and charge structure sums.
    `continue_run()` checks all of them, and a checkpoint that lacks one or holds an invalid
    one raises `ValidationError`; `Electrostatics` saves, checks and restores this state
    (`checkpoint_data`, `validate_checkpoint`, `restore`). The run metadata records the
    initial state of the ψ stream (`initial_psi_rng_state`). A run continued by hand takes ψ
    through `run(..., initial_psi=...)` and warns without it, since ψ then restarts at zero;
    without Coulomb coupling ψ stays zero and a nonzero `initial_psi` is rejected.
  - `psi_drift_max_history` records the largest drift of ψ apart from `drift_max_history`,
    whose maximum the exactly integrated linear part of ψ would dominate.
  - Some composition modes have a zero eigenvalue and therefore no auxiliary field, for
    example two types that differ only in charge. The structure functions along these
    modes come from ψ; without this, S_ij would miss the charge correlations of such
    systems.
  - The structure function files also hold `charge_structure_function`, the structure
    function of the smeared charge density. It comes from two estimators that use ψ:
    - the mixed form (k²/E)⟨ψc⟩;
    - the field-field form (k²/E)²[V N̄^(1/2) ⟨ψψ⟩ + E/k²].

    Their difference has mean zero, so at each wavevector the two are combined with the
    weight of least variance, stored as `charge_structure_function_weight`. The
    field-field form had 20-600 times less variance where screening is strong, and the
    mixed form is better where it is weak. Its pending sums continue after a restart.
    The two forms are also stored alone (`charge_structure_function_mixed`,
    `charge_structure_function_field_field`): in the Gaussian regime the combination
    follows from the linear response of the chains and does not show how well ψ is
    sampled.
  - Limitations: no random copolymers.

  `TestClftsElectrostatics` checks the reductions to neutral CL-FTS and to one common
  smearing length (bit for bit), the forces against finite differences of H, the ψ step
  against the exact flow of its force, the stationary variance of ψ without charges and
  with a linear charge response, and bitwise restarts. `TestChargedStructureFunctionRPA`
  compares two systems with the RPA: a charged diblock with counterions, and a
  polyampholyte melt whose charge mode has no auxiliary field. It estimates standard
  errors from block means and requires every tolerance to exceed 2 of them.
  Example: `examples/charged/clfts/ChargedDiblock.py`.

### Interface changes

- Python interface changes of 2026-11-22:
  SCFT keeps the axis order of the input, the space group and the bases of the fields in one object,
  `SCFT.grid` (scft/grid.py); `SCFT.sg` and `SCFT.axis_ordering` stay. An `axis_ordering` that does
  not name each axis of `nx` once raises `ValidationError` (was `ValueError`; a string such as "ba"
  is refused). `MeanFieldElectrostatics` takes the simulation and its `start()` sets the initial ψ.
  SCFT builds the one Anderson vector of the fields, ψ and the relaxing lattice parameters:
  `CellRelaxation.parameters()`, `residual()` and `apply()` replace `update()`, and `n_parameters`
  replaces `n_variables`. `validate_clfts_params` replaces `validate_lfts_params(...,
  complex_fields=True)`. `LFTS.forces()` is the one L-FTS force with the WTMD bias (the loop and the
  saddle-point search use it); `chains.partition_functions` and `chains.chain_hamiltonian`;
  `checkpoint.save_statistics` writes the statistics files of L-FTS and CL-FTS;
  `force_jacobian(sim, joint_psi)` and `symmetric_jacobian(sim, J, noise_direction, mobility)` take
  what they read as arguments. A misspelled key of a point ion or of a Gibbs box names the closest key
  (`validation.ION_KEYS`, `GIBBS_BOX_KEYS`). The defaults of the options are written once, where the
  option is read, and the validators check an option when it is given; the WTMD range (`psi_min`,
  `psi_max`, `dpsi`) and the smearing type keep a second copy in the validator, which checks the
  bins and the smearing before the simulation is built. The L-FTS and CL-FTS constructors
  build their parts in the same order, so some console lines come in another order.
  `Polymer.get_propagator_key` is replaced by `PropagatorComputationOptimizer.get_propagator_key(p,
  v, u)`; the key parser (`PropagatorCode.get_deps_from_key`) and the unused bindings
  `ComputationEdge.deps`, `.height`, `.junction_ends`, `ComputationBlock.n_segment_right` and
  `Polymer.display_architecture` are gone. `create_computation_box` given a bare number for the
  angles or a bare string for the boundary conditions raises `TypeError` (was `RuntimeError`); size
  messages name `n_basis`; other C++ exceptions keep pybind11's type (`MemoryError` for an
  allocation failure, was `RuntimeError`).
- C++ interface changes of 2026-11-22: `CudaRealTransform` and `CudaFFT` are removed: CUDA wall
  boxes, the Pmmm CrysFFT and `CudaFourierTransform` use `CudaMakhoulTransform`, which takes boxes of
  walls only and a factor per coefficient in its backward transform, and the CUDA wall coefficients
  have the CPU scale (half of REDFT10/RODFT10 per axis). `FFT<T>::backward_with_factors` is pure;
  `FftwFFT<T>`, `MklFFT<T>` and `createFFT<T>` take the grid as a vector (no dimension template). The
  CPU ObliqueZ CrysFFT is one class, `CrysFFTObliqueZ`, for both libraries
  (`CrysFFTObliqueZBase`, `FftwCrysFFTObliqueZ` and `MklCrysFFTObliqueZ` are removed). The CPU solvers
  share their context and one `compute_exp_dw` in `CpuSolver`, as on CUDA. Propagator keys are
  written once by the optimizer after the contour lengths are mapped; `ComputationEdge` records the
  dependencies, ds index, start (`PropagatorStart`) and graft label where its key is written, and no
  code reads them back from the key (`get_deps_from_key`, the key extractors and the second set of
  keys in `Polymer` are gone). `ComputationCommon` owns the backend memory in one `BackendMemory`;
  `SpaceGroup` maps every operation through one exact integer map of the grid indices. `CudaStream`
  owns the CUDA streams. One ordered plan, `plan_crysfft()`, chooses the CrysFFT engine on both
  platforms, and one enum, `CrysFFTEngine`, names it (`SpaceGroup::physical_basis()` returns it;
  `CrysFFTChoice`, `CudaCrysFFTMode`, `select_crysfft_mode()` and `crysfft_mapping_usable()` are
  removed). `SpaceGroup.enable_best_physical_basis()` (C++ and Python) holds the physical-basis choice
  of SCFT, which scft.py made by trying one basis after another. `enable_m3_physical_basis()` refuses a
  grid that the recursive 3m engine does not take (nz/2 < 2); the basis ran on the full grid. The recursive 3m tables
  (`Recursive3mTables`), the Pmmm tables and the stress-multiplier value are built once for both
  platforms (CrysFFTTables.h), and the CPU and CUDA pseudo-spectral bases share their CrysFFT and
  boundary helpers (PseudoSolverHelpers.h).
- Python interface changes of 2026-11-21:
  A key that an entry of `polymers` or of its `blocks` does not take raises `ValidationError` with
  the message of the section keys (`validation.POLYMER_KEYS` and `BLOCK_KEYS`): a misspelled
  `multiplicity` built a plain block and a misspelled `initial_conditions` a free chain, without a
  word. `fts_consistent_energy` is ignored with one warning by every method, as `stress_interval`;
  SCFT printed a line about it and L-FTS and CL-FTS listed it as an SCFT option.
  `polymerfts.__version__` is the version of the native module that was imported
  (`_core.build_info`), not that of whichever polymerfts distribution is installed.
  `LFTS/CLFTS.save_simulation_data(path, w_aux, phi, langevin_step, normal_noise_prev, record)`
  computes the monomer fields from `w_aux`; its `w` argument is gone. `GibbsEnsemble` requires
  `recording.sample_interval` in its own input check (the message is unchanged); `WTMD.Mk` is
  removed. Gibbs histories written between 3824309f and 7e8996a5 (development versions, never
  released) no longer continue; histories of released versions continue as before.
- C++ interface changes of 2026-11-21: `PlatformSelector::create_factory<T>(platform,
  reduce_memory)` holds the platform dispatch once and replaces `create_factory_real` and
  `create_factory_complex`; `ComputationBackend::copy` and
  `SpaceGroup::get_z_mirror_shift()` (`z_mirror_half_grid_shift()` stays) are removed; the
  `Scheduler` belongs to the full-storage computations, and the reduced-memory computations compute
  each propagator whole, one at a time in height order. `CudaSolver<T>` holds the context every CUDA
  solver shares (box, molecules, streams, space group and its maps, the full grids of a
  reduced-basis run with the expansion, reduction and mask around the steps); `compute_exp_dw()`
  takes no mapping arguments and `CudaSolverPseudoBase::update_dw_impl()` is gone. The CPU solvers
  take the space group in their constructors, as the CUDA solvers do, and build their tables once
  (`CpuSolver::set_space_group()` is removed); one recursive 3m CrysFFT class runs over the real
  transform of each library; `FFT<T>` holds no state (`MklFFT` has its own `backward_with_factors`).
- Python interface changes of 2026-11-19:
  A key that an option section does not take (`langevin`, `recording`, `saddle`, `compressor`,
  `wtmd`, `optimizer`, `space_group`, `smearing`, `electrostatics`, `gibbs_ensemble`) raises
  `ValidationError` naming it, the closest valid key and the keys of the section, as a misspelled
  top-level key does (`validation.SECTION_KEYS`); before, a misspelled key in most sections left its
  option at the default without a word (for example `langevin.integrater` ran `lm`, and the
  deep-langevin-fts names `wtmd.l` and `wtmd.dT` ran with ℓ = 4 and ΔT = 5). Sections of another
  method are still ignored as a whole.
  The SCFT option `stress_interval` is removed: the cell relaxation computes the stress in every
  iteration, together with the concentrations (`PropagatorSolver.compute_concentrations_and_stress()`).
  Every method ignores a given `stress_interval` with one warning; a later release will refuse it.
  Results of the default (1) do not change. Fewer stress computations did not pay: the gyroid with 4
  was as fast on cpu-mkl and 13 % slower on CUDA, a 1D lamella needed 3.6 times the iterations.
  `SCFT.stress_interval` is gone, `SCFT.cell.error_at()` and `current_error()` are replaced by
  `SCFT.cell.stress_error(w, phi)`, and `SCFT.compute_concentrations(w, q_init, stress=False)`
  computes the stress with the concentrations when `stress` is set; a subclass that overrides
  `compute_concentrations` must accept `stress` (the run loop passes it).
  A numerical method of the other chain model raises an error also in `_core.resolve_numerical_method`
  and the native factories (discrete chains took `ps` for `rqm4`, `rk2` or `cn-adi2`).
  A Gibbs-ensemble continuation may change `verbose_level`, `platform`, `reduce_memory` and
  `aggregate_propagator_computation`, as a CL-FTS continuation may; histories written by earlier
  versions continue with the input that wrote them (those of 3824309f..7e8996a5 only until
  2026-11-21). An L-FTS or CL-FTS checkpoint whose input held keys that no release read in `saddle`
  or `compressor` continues without them; a Gibbs history of such an input, which stores only a
  hash of it, cannot continue.
  `LFTS/CLFTS.save_simulation_data` takes `w_aux` and the `RunRecord` of the run as required
  arguments. Parameters.md states that `sim.wtmd.u`, `up` and `I1` hold U, U′ and I₁ divided by CV,
  while the WTMD statistics files and checkpoints store them times CV.
- `scripts/analysis/plot_wtmd_chib_AB.ipynb` reads the chain model, N, ζN, smearing, grid, box and N̄
  that z_inf needs from a `fields_*.mat` file of the run (`renormalization.read_simulation`), which
  refuses oblique boxes, unequal segment lengths and files of another run; its parameters
  `chain_model`, `n_segments`, `zeta_n`, `smearing` and `z_inf` are gone (2026-11-19). A continuous,
  compressible or smeared run left at the former defaults got a wrong z_inf without a message.
  `seeds.from_file` of the SCFT examples reads JSON results too.
- Python interface changes of 2026-11-18:
  `MeanFieldElectrostatics` keeps ψ as its `psi` property, transformed once when it is set, with the
  point-ion densities computed once per ψ; its methods take no ψ argument and it takes the smearing,
  whose |k|² it shares. `SCFT.psi` is gone (`SCFT.electrostatics.psi`; `SCFTResult.psi` and the saved
  `psi` are unchanged). `Smearing.k_mag` is removed; `Smearing.set_box` with the transform of real
  fields in a periodic box takes |k|² of oblique cells from the transform, so the solver box must be
  set first, and `Smearing.stress` takes optional lattice derivatives. `WTMD` is built as
  `WTMD(params, nx, lx, nbar, eigenvalues, real_fields_idx, angles=None, fourier=None,
  verbose_level=1)` from the `wtmd` input, whose defaults live in wtmd.py; its unused arguments `u`,
  `up`, `I0`, `I1` are removed, and its line of Ψ prints from `verbose_level` 1 on.
  `draw_architectures` is a class attribute of `LFTS` and `CLFTS`. `GibbsEnsemble` leaves the keys
  its cycles set (`langevin.max_step`, `recording.dir`, the recording intervals other than
  `sample_interval`, `recording.pressure`, the volume fractions of `polymers`) out of the hash of its
  history, so a continuation may change or drop them; a history written before continues with the
  input that wrote it. `SpaceGroup.symmetrize_reduced_basis` is removed
  (`sg.irreducible_to_reduced(sg.reduced_to_irreducible(x))`), and so are bindings no code used:
  `ComputationBox.get_dim` and `get_angles` (`get_angles_degrees` stays),
  `get_force_off_diagonal_stress`, `ComputationBlock.n_segment_left` and `v_u`, `Polymer.get_n_segment`,
  `get_block_index_from_edge`, `get_adjacent_nodes` and `get_block_indexes`,
  `Molecules.get_global_ds` and `get_bond_lengths`, and the optimizer's `get_computation_block`,
  `display_propagators`, `display_blocks` and `get_monomer_type_from_key`.
- C++ interface changes of 2026-11-19: one `CpuFactory<T>(FFTBackend)` replaces `FftwFactory` and
  `MklFactory`; `FFT<T>` keeps only its `double*` forward and backward (with
  `backward_with_factors`); `PropagatorComputation` gains `compute_concentrations_and_stress()` and
  loses `set_space_group()` and `get_force_off_diagonal_stress()`; `Polymer::get_block_indexes()`,
  `Scheduler::display()`, `SpaceGroup::enable_physical_basis()`, the z-mirror translation getter and
  the ';' form of propagator keys are removed; one `CudaAndersonMixing` keeps its history in device
  or pinned host memory (`CudaAndersonMixingReduceMemory` and `PinnedCircularBuffer` are removed).
- An error of the native layer names its place once (2026-11-18): a catch block that rethrows passes
  a polymerfts error through unchanged instead of prepending its own file and function, so a wrong
  monomer type reports one location instead of three. Python exception types are unchanged.
- The phase examples share `examples/block_copolymer/scft/seeds.py` (moved from `phase_diagram/`),
  with the gyroid level set, sphere wells, cylinders and a periodic `resample` of fields read from
  MAT files (2026-11-18).
- SCFT, L-FTS and CL-FTS check the top-level keys of their input against the keys each method reads
  (`polymerfts.utils.validation.METHOD_KEYS` and `check_method_keys`; 2026-11-17, `mask` since
  2026-10-28). Release 0.10.0 ignored the keys it did not read without a word. The options of another
  method, such as `relax_cell`, `crystal_system`, `optimizer` or `space_group` in L-FTS and CL-FTS,
  `langevin`, `recording`, `saddle`, `compressor` or `wtmd` in SCFT, `saddle`, `compressor` and `wtmd`
  in CL-FTS and `dynamic_stabilization` in L-FTS, are ignored with one warning that lists them, at
  the line that builds the simulation (also through `from_file` and `GibbsEnsemble`, which warns
  once), so one input serves SCFT and the Langevin methods at its top level; options nested in a
  section, such as `recording.pressure` in L-FTS, are checked as before. L-FTS still refuses
  `electrostatics`, which defines a charged system that it cannot simulate, and L-FTS and CL-FTS
  refuse `axis_ordering`, which reorders the shared `nx`, `lx`, `bc` and `angles` of SCFT. A key
  that no method reads, such as a misspelled key or `mask` (a `PropagatorSolver` option), raises
  `ValidationError` naming the key and the closest valid one. Legacy names are translated first
  (tests/TestParams.py, TestConfigFile.py, TestParameterNames.py, TestGibbsEnsemble.py).
- A top-level key given as None (`null` or `~` in YAML) is the same as an absent key for every
  option of SCFT, L-FTS, CL-FTS and the Gibbs ensemble (2026-11-17): the method uses its default or
  reports a required key as missing, and a `from_file` override given as None removes the key of the
  file, as in 0.10.0. Before, some options refused None and others kept it as their value; `relax_cell: None`
  meant false (now missing) and `aggregate_propagator_computation: None` turned aggregation off (now
  the default, on). Inside a section None keeps its documented meaning, such as
  `electrostatics.counterion: None` (tests/TestParams.py, TestConfigFile.py).
- L-FTS and CL-FTS check `angles` like SCFT (2026-11-16): the angle checks moved from `SCFT` to
  `validate_system_params`, so all three classes, and the Gibbs ensemble before it computes the
  volumes of its boxes, raise `ValidationError` with the same message when a 3D box does not give
  three angles (an empty list included, which the native box took as an orthogonal cell), when a 1D
  or 2D box gives angles of its missing axes other than 90 degrees, or when `angles` is not a flat
  list; L-FTS and CL-FTS raised `RuntimeError` from the native box before, or `TypeError` for a list
  that is not flat (tests/TestParams.py). The angles of the missing axes must lie within 1e-10 rad of
  90 degrees, the tolerance of the computation box, where 1e-10 degrees was required (2026-11-17).
- A box refuses a nearly flat cell, V < 0.1 abc (one oblique angle outside 5.74 to 174.26 degrees),
  when it is created or changed, and points to the reduced (Niggli) cell (2026-10-28): the exact
  alias search (see Fixed) spans about abc/V periods per axis.
- The volume element of a box is the product of the grid spacings in an orthogonal cell and V/M in
  an oblique one, wherever it is used (2026-11-17). A box created with angles [90, 90, 90] or changed
  by `set_lx()` or `set_lattice_parameters()` took V/M, which can differ in the last bit from the
  product that a box created without angles takes, and the integrals of a space-group basis and
  `PropagatorSolver.get_dv()` took V/M in every cell; the box stores its volume element once
  (`ComputationBox.get_dv_base()`). Results of these boxes change at round-off: at most 7e-16
  relative in Q, φ and the stress of one evaluation in a space-group cell, about 1e-13 in a
  converged SCFT free energy (`TestBoxVolumeElement`).
- The simulation classes share more of their setup (2026-11-16). `WTMD` saves, checks and restores
  its own checkpoint state (`checkpoint_data`, `validate_checkpoint`, `restore` and `settings`).
  `SymmetricPolymerTheory.dt_scaling()` gives the mobilities of the auxiliary fields,
  `PropagatorSolver.bc_lower` gives the boundary conditions in lowercase (`bc` keeps the spelling of
  the input), and one context manager, `utils.checkpoint.recording_run_status`, writes
  `run_status.json` around the runs of LFTS and CLFTS (2026-11-17), in place of
  `polymerfts.utils.write_run_status`. Results do not change.
- Every method refuses a monomer name containing `,`, the separator of the two names of a `chi_n`
  pair, with `ValidationError` (2026-11-16); such a name stopped SCFT, L-FTS and CL-FTS with a
  `ValueError` from the field theory (tests/TestPairRecords.py).
- `CLFTS` no longer has the attributes `solver`, `molecules` and `propagator_computation_optimizer`
  (2026-11-16), which nothing used; `prop_solver` gives the same objects.
- An angle that the crystal system of a space group fixes (90 or 120 degrees, or the right angles at
  the monoclinic unique axis) must lie within 1e-10 rad (5.7e-9 degrees) of its value, the
  orthogonality test of the computation box (2026-11-16). SCFT (`ValidationError`) and the
  computation box of a `PropagatorSolver`, when it is created and when it changes, refuse cells
  whose fixed angles are 5.7e-9 to 9e-5 degrees off, which release 0.10.0 accepted within its
  relative 1e-6 (see Fixed; `TestCrysFFTCellChange`).
- `Polymer.get_n_segment_total()` counts every copy of a block (2026-11-08): it returns the
  segments of one whole chain as a float, which a real block multiplicity makes non-integer.
- `ComputationBlock.n_repeated` and the repeat count in each `ComputationEdge.deps` entry are
  floats for every polymer (2026-11-08), since a real block multiplicity makes them non-integer;
  `Block.n_copies` is a new read-only attribute.
- The MATLAB scripts of `scripts/analysis` are removed (2026-11-06); notebooks and Python scripts
  do their work: `plot_2d_density.m`, `plot_3d_isodensity.m`,
  `plot_structure_function.m` and `plot_wtmd_statistics.m` by the notebooks of the same name,
  `plot_2d_slice.m` by the new `plot_2d_slice.ipynb`, which draws every monomer type
  (tests/scripts/TestSlicePlot.py), and
  `convert_into_auxiliary_fields.m` by the new `convert_into_auxiliary_fields.py`, which takes the
  file on the command line, also reads CL-FTS fields and writes the lattice angles
  (tests/scripts/TestConvertAuxiliaryFields.py).
- `numerical_method` names the method of either chain model (2026-11-05): the FFT solver of discrete
  chains, formerly `"discrete"` and chosen whatever `numerical_method` said, is `"ps"`, the default, and
  `"rs"` is the new real-space one. A discrete chain with `"rqm4"`, `"rk2"` or `"cn-adi2"` raises an
  error instead of a note. A checkpoint of discrete chains written before records `"discrete"` and
  continues with `"ps"`, the same solver (2026-11-16).
- `LR(jk, fourier)` replaces `LR(nx, lx, jk)` (2026-11-02, 2026-11-17): it updates the fields
  through `fourier`, a `PropagatorSolver.fourier`, instead of NumPy's real FFT, and `nx` and `lx`,
  which it does not need, are gone (see Fixed for the walls).
- `scft.py`, `lfts.py` and `clfts.py` hold the flow of each method (2026-11-02), and the details moved
  to objects with one topic each. `LFTS.saddle` (`saddle.py`) holds the compressors and the search:
  `sim.find_saddle_point(w)` is now `sim.saddle.solve(w)`, which also returns the smeared
  concentrations, `(phi, phi_smeared, hamiltonian, iterations, error_level)` (2026-11-17),
  `sim.compressor` is `sim.saddle.compressor` and the input `saddle` is `sim.saddle.settings`; `LR`
  and `LRAM` are still exported by the package.
  `SCFT.cell` (`cell_relaxation.py`) holds the lattice parameters that vary and the stress,
  `sim.cell.stress(w, phi)`; `crystal_system`, `stress_scale`,
  `lx_reduced_indices`, `lx_full_indices`, `angles_reduced_indices` and `angles_full_indices` moved
  there too. Both Langevin
  classes have `structure_functions` (`structure_functions.py`), and `SCFTResult` and `IterationInfo`
  live in `scft_results.py`.
  `checkpoint.RunRecord` replaces `restore_recording_state` and the lists of the loops: a new recorded
  quantity needs one entry in its series and one `add()`, and a checkpoint without it gives NaN for
  its pending samples. Results do not change. The console header is shared too: SCFT prints
  `Platform:` without the space before the colon and its `Pressure scale` line after the chains,
  and the lines of the conformational asymmetry and of χN lose their trailing spaces.
- `chains.py` holds the polymer chains (2026-11-02): reading them, their drawings, the
  `PropagatorSolver` of SCFT, LFTS and CLFTS with the polymers added (`chain_solver`), and through
  `chain_fields` and `solve_chains` what the three `compute_concentrations` share, so that the
  methods keep only what is their own. `file_formats.py` holds `load_config` and `save_vtk`. Both
  moved out of `polymerfts.utils`, and with them `process_polymer_blocks`, `process_random_copolymer`,
  `read_chains`, `mean_concentrations` and the drawing functions; the package still exports
  `load_config`, `ConfigError`, `save_vtk` and `draw_polymer_architectures`. The timings that
  `SCFT.compute_concentrations` returns now have the keys of the Langevin classes: `solver` for the
  propagators instead of `pseudo`, and `phi` for all of the concentration integration.
- `src/python` has four folders (2026-11-02): `scft`, `fts`, `theory` and `utils`. User scripts keep
  `from polymerfts import scft, lfts, clfts` and the names the package exports. Submodules are
  imported with their folder: `polymerfts.utils.common` (formerly `polymerfts.utils`), `.checkpoint`,
  `.validation` and `.file_formats`; `polymerfts.theory.propagator_solver`, `.symmetric_formulation`
  (formerly `polymer_field_theory`), `.smearing`, `.electrostatics` and `.chains`; `polymerfts.fts.lfts`, `.clfts`, `.integrators`, `.wtmd`, `.gibbs_ensemble`,
  `.saddle`, `.structure_functions` and `.thermodynamics`; `polymerfts.scft.scft`, `.scft_results`,
  `.cell_relaxation` and `.phase_equilibrium`. `python_source_sha256` in the metadata covers all folders.
- What a run saves and restores moved from `polymerfts.utils` to `polymerfts.utils.checkpoint`
  (2026-11-01): `validate_checkpoint`, `checkpoint_data` and the functions of the extension state.
- The PSCF import refuses a PSCF result of one monomer type (`nMonomer 1`), whose script SCFT
  refuses, before it writes anything; the import of the fields alone still takes it (2026-10-30;
  tests/scripts/TestPscfScript.py). Release 0.10.0 wrote the bundle, and its script stopped when it
  constructed SCFT.
- `PropagatorSolver.integral`, `mean`, `inner_product`, `multi_inner_product` and `zero_mean` take
  fields in the active basis of the solver, or with `basis="irreducible"` in the irreducible basis of
  the group, instead of choosing the basis by the size of the array (2026-10-30;
  tests/TestPhysicalBasisOperators.py).
- SCFT refuses `relax_cell` with a method that computes no stress (`cn-adi2`) with `ValidationError`
  in its input check, before it builds a solver (2026-10-29, 2026-11-17; tests/TestParams.py); such a
  run stopped at its first iteration.
- The PSCF export reads space groups as SCFT does (2026-10-29, 2026-11-17): a symbol with several
  Hall settings needs `hall_number`, `hall_number` must be a setting of the symbol, and the symbol
  must equal spglib's short symbol after spaces and tabs at its ends are trimmed (`P6_3/mmc`; SCFT
  refuses `P63/mmc` and `I a -3 d`, and so does the export). Release 0.10.0 took the first setting
  of the symbol, the crystal system of a given `hall_number` whatever the symbol, and symbols
  without their spaces and underscores (tests/scripts/TestPscfConverter.py).
- CUDA size limits (`TestCudaOddReflecting`, `TestCudaDCT`): an absorbing axis of more than 6142
  points in a 2D or 3D box is refused with a message that names the axis, where it failed at a
  kernel launch with a generic CUDA error (2026-10-28), and 1D boxes with reflecting walls take up to
  6142 points instead of 4096, for which the cosine transforms reserved shared memory in the DCT-I
  layout (2026-10-29).
- The examples are grouped by physical system and then by simulation method (2026-10-28):
  `examples/block_copolymer/`, `examples/blend/` and `examples/charged/`, each with `scft/`,
  `lfts/` and `clfts/` as available, and `examples/tools/` for the Fourier transform and the
  extension examples. `examples/README.md` lists every example in a table of systems and
  methods. The scripts are unchanged apart from the paths in their comments.
- The C++ factory functions return `std::unique_ptr` (2026-10-26):
  `PlatformSelector::create_factory_real`/`create_factory_complex` and every `create_*`
  function of `AbstractFactory`. C++ callers no longer delete the objects. An object
  created from others still borrows raw pointers to them, so they must outlive it; local
  `unique_ptr`s declared in creation order are destroyed in the right order. The C++
  tests that use the factory hold the objects in `unique_ptr`s; 19 of them deleted the box
  before the computation, which was harmless only because none used `cn-adi2`, whose
  computation releases its hold on the box when it is destroyed. The Python interface is
  unchanged.
- Constructing `SCFT`, `LFTS` or `CLFTS` no longer writes `polymer_<id>.png` into the
  current directory, where it could overwrite an existing file of that name (2026-10-26).
  `LFTS` and `CLFTS` draw the chain architectures into `recording.dir` when a run starts;
  `SCFT` draws them only into the new `polymer_figure_dir`, when given.
  `draw_polymer_architectures()` takes a `directory`, and its colors beyond the sixth
  monomer type come from a fixed local stream instead of the global NumPy random state.
  The scripts of the PSCF conversion draw no figures and no longer change the working
  directory.
- `scripts/analysis/renormalization.py` replaces `renormalization_discrete.m` and
  `renormalization_rpa.m` (2026-10-24):
  - It reads a fields file or takes options and needs only NumPy and SciPy.
  - It evaluates z∞ and the bare segment length a_b/a of CL-FTS from Eqs. (29) and (26) of
    Matsen et al., *J. Chem. Phys.* **164**, 014905 (2026), with compressibility and smearing.
  - It prints the χ_bN for a target z∞χ_bN.
  - The 1/k² singularity of both integrands is integrated exactly, so z∞ agrees with the
    bond sum to 10⁻¹³ with 64 Gauss-Legendre points per axis; integrating it numerically
    left errors of 10⁻⁷–10⁻⁶ with 128 points.
  - Smearing lengths must be numbers; the per-type `a_int` of charged CL-FTS is rejected,
    since the formulas have one Γ(k) for all monomer types.

  Discrete chains without compressibility and smearing give the values of the former
  `renormalization_discrete.m`, as `tests/scripts/TestRenormalization.py` checks. Continuous
  chains now use the N → ∞ limit of the same formulas instead of the athermal RPA of
  `renormalization_rpa.m`, so their z∞ changes. For example, a 32³ grid with L = 4R₀ and
  N̄ = 10⁴ gives 0.813 instead of 0.858: the RPA integral also counted a finite term from
  k ≲ 1/R₀.
- `SymmetricPolymerTheory.compute_h_deriv_chin(w_aux, phi=None, wtmd=False)` no longer takes
  `chi_n` as its first argument and uses the χN of the object (2026-10-24, 2026-11-17); the call of
  0.10.0, `compute_h_deriv_chin(chi_n, w_aux, phi)`, raises a `TypeError` that names the keywords.
- The table of crystal systems, `CRYSTAL_SYSTEMS` of `scft.py`, is in `polymerfts.utils.validation`
  with the dimensions of each system; the input check of `crystal_system` and the box relaxation
  read it (2026-11-17).
- `SymmetricPolymerTheory` exposes its χN matrix as `matrix_chi`; `SCFT.matrix_chi` is the
  same array, so the matrix is built in one place, and `compute_eigen_system(matrix_chi,
  zeta_n)` takes the matrix instead of the pair dictionary (2026-10-24).
- `parse_chi_n` moved from `polymerfts.utils` to `polymerfts.utils.validation`, next to the other
  input checks (2026-10-24); `polymerfts.utils` no longer re-exports it (2026-11-16).
- With a space group, the internal `advance_propagator_single_segment()` binding takes and
  returns fields in the reduced basis, like `compute_propagators()`. Before, it expected
  full-grid arrays but read and wrote only their first `n_basis` entries. The C++ method
  `PropagatorComputation::compute_propagators_reduced()` is removed;
  `compute_propagators()` already accepts reduced-basis fields (2026-10-25).
- C++ platform factories implement only `create_backend()` (2026-10-25).
  `AbstractFactory::create_propagator_computation()` and `create_molecules_information()`
  are now written once, in `src/common/AbstractFactory.cpp` and `AbstractFactory.h`,
  instead of three identical copies. The numerical method is resolved and validated there,
  once. The CPU and CUDA backend constructors take the resolved `NumericalMethodInfo`
  instead of the method names.
- `check_total_partition()` no longer prints the error of each polymer, evaluates every
  polymer instead of stopping at the first failure and takes an optional `tolerance`
  (2026-10-26). The errors come from the new `PropagatorSolver.partition_errors()`
  (native `compute_partition_errors()`); the simulation loops print them.
- Checks move to the layer that owns the state (2026-10-26):
  - `PropagatorComputation` returns only results of the current fields and box. Before
    `compute_propagators()`, after a failed computation and after
    `update_laplacian_operator()` its getters raise "Propagators not computed"; the
    concentration getters also need `compute_concentrations()`, `get_stress()` needs
    `compute_stress()` (see Fixed). `PropagatorSolver` relies on these checks instead of its own
    flag.
  - The native factory rejects mixed periodic and non-periodic boundaries for the
    pseudo-spectral methods (`supports_mixed_boundaries`), not only `PropagatorSolver`;
    an unknown method lists the valid ones, and `PropagatorSolver` resolves the method
    with the new `resolve_numerical_method()` of the native registry.
  - While a `cn-adi2` computation uses a box, `set_lattice_parameters()` rejects
    non-orthogonal angles before changing the box. `PropagatorSolver` checked this in
    Python, and its constructor used 1e-10 degrees instead of the 1e-10 radians of the
    box. `PropagatorSolver` creates the box in its constructor and uses its orthogonality.
  - `SCFT` checks the cell against the space group with the new `SpaceGroup.check_cell()`,
    the check the computation box applies, instead of a Python copy with other angle
    tolerances (`ValidationError` as before). The rule against box relaxation in
    rhombohedral-axes settings stays in Python.
  - The binding of `create_anderson_mixing()` takes a 64-bit size and rejects too many
    variables with the native message (`RuntimeError`); `SCFT` and `LFTS` no longer
    repeat the check. The native binding also checks the sizes of `w` and `q_init`, so
    `PropagatorSolver.compute_propagators()` raises `RuntimeError` instead of `ValueError`
    for a wrong size.
- `SpaceGroup` prints nothing while it is constructed or a physical basis is enabled;
  `display_info()` prints the summary, which `SCFT` calls, and `PropagatorSolver` reports
  the active basis with its size (2026-10-26).
- The CPU z-mirror crystallographic FFT no longer reads the undocumented
  `FTS_FFTW_EXHAUSTIVE`, `FTS_FFTW_PATIENT`, `FTS_FFTW_ESTIMATE`, `FTS_FFTW_WISDOM_ONLY`,
  `FTS_FFTW_WISDOM_FILE` and `FTS_CRYSFFT_HEX_DCT_FFT` (2026-10-26). It plans with
  `FFTW_MEASURE` and the native DCT, the previous default, and the FFT-based DCT path
  that only the last variable enabled is removed, as are the unused `print_array`
  helpers of Anderson mixing.
- `scripts/pscf/pscf_converter.py export` no longer reads polymerfts Python scripts (2026-11-17).
  Release 0.10.0 read the module-level `params` of a script, which needed a fields file made by
  running the script anyway, and could export parameters its simulations did not receive (such as
  those of a script that runs `SCFT(dict(params, ds=0.01))`). Export the result the run
  saves (`save_results`), or `params` written with `json.dump` or as YAML with `--fields`; a `.py`
  path fails with this advice.
- `run(normal_noise_prev=...)` of L-FTS and CL-FTS checks the previous noise before the run starts
  (2026-11-17): real, finite numbers, one per real (L-FTS) or auxiliary (CL-FTS) field and grid
  point, else `ValidationError` (`utils.validation.previous_noise_values`, which CL-FTS also applies
  to `psi_noise_prev`). Complex values lost their imaginary part, and a wrong size or strings stopped
  the run after its status had become "running" (tests/TestFtsReliability.py).
- L-FTS and CL-FTS check their `initial_fields` (complex in CL-FTS, also `initial_psi` and
  `psi_noise_prev`) like SCFT before a run starts and raise `ValidationError` (2026-11-17): complex
  fields stopped L-FTS with a solver `ValueError`, strings with `UFuncTypeError`, ragged lists with
  `ValueError` and a missing type with `KeyError` (tests/TestFtsReliability.py).
- A computation that fails during a run raises `NumericalError` (a `RuntimeError`,
  `polymerfts.NumericalError`), and input errors stay `ValidationError` (2026-11-17). `SCFT.run`
  raises it when a residual or the error level is not finite, and SCFT and L-FTS when a partition
  function with real fields is not positive and finite at an accepted state (see Fixed; L-FTS raised
  `ValidationError` for the non-finite Hamiltonian of such a state). SCFT, L-FTS and CL-FTS also raise
  it, where they raised `ValidationError`, for a failed partition consistency check, lost material
  conservation, and a non-finite Hamiltonian, force, field, concentration or electrostatic
  potential (also the fields an integrator returns).
  `SaddlePointError` is now a subclass of `NumericalError`; L-FTS still records it as
  `aborted_saddle_failure`. `NumericalError.fields` holds the monomer fields `{type: array}` of the
  failed state where the check has them: on the full grid as `SCFTResult.w` in SCFT, as
  `initial_fields` in L-FTS and CL-FTS (tests/TestGceIonRange.py, TestPartitionGuard.py,
  TestFtsReliability.py).
- L-FTS and CL-FTS fields files save `angles` at the top level (2026-11-17). Each density notebook
  reads them in its first cell, and older files through their checkpoint settings or input angles;
  the grid points of an SCFT result with a space group follow from its Hall number: the settings
  430-488 except the seven R settings on rhombohedral axes use hexagonal axes, the rule of
  `SpaceGroup.uses_hexagonal_axes()`. The notebooks need no polymerfts (tests/scripts).
- SCFT chooses the physical basis of a space group from the angles the group fixes
  (`SpaceGroup.fixed_angles()`), after the cell check, instead of the angles of the cell
  (2026-11-17): when the group fixes all three at 90 degrees m3 (where `m3_crysfft_feasible()`), else
  Pmmm, else the irreducible basis; when it fixes α = β = 90 degrees and leaves γ free or fixes it at
  120 degrees, z-mirror, also for a monoclinic group with the unique axis c in a cell with γ = 90
  degrees (tests/TestSpaceGroupBasisSelection.py).
- Smearing, WTMD, `save_vtk`, the classification of 2D cells by γ and the angles that a given
  `crystal_system` fixes use the right-angle test of the computation box, 1e-10 rad
  (`utils.validation.angle_fixed_at` and `FIXED_ANGLE_TOLERANCE`; 2026-11-17), in place of |cos| < 1e-12
  (smearing, WTMD) and 1e-6 degrees (crystal systems): a cell within 1e-10 rad of right angles takes
  smearing with walls, which raised `ValueError`, and a 2D cell 1e-8 degrees off 120 is oblique
  (tests/TestParams.py).
- SCFT refuses `smearing`, and so `electrostatics`, which needs it, together with `space_group` in
  its input check (`ValidationError`, 2026-11-17), before it builds a solver; it raised
  `NotImplementedError` after the solver was built (tests/TestParams.py).
- The input check of SCFT, before it builds a solver, requires `max_iter` to be an integer of at least
  1 and a `crystal_system` given without a space group to be one of the systems of the dimension of
  the box, else `ValidationError` (2026-11-17): a `max_iter` of 10.5 or `"5"` stopped with a
  `TypeError`, and a crystal system the dimension lacks raised `ValueError`. It also refuses there
  `relax_cell` with `electrostatics` in a box with walls (tests/TestParams.py).
- SCFT writes its results atomically in every format: a new file next to the target, flushed to the
  disk, replaces it, so a failed write leaves an existing file as it was (MAT 2026-10-27, JSON and YAML
  2026-11-17; tests/TestScftResult.py). The `run_status.json` of L-FTS and CL-FTS is flushed to the
  disk too. JSON results end with a newline, and `.yaml` results are converted as the JSON ones (keys
  as strings, complex numbers as `{real, imag}`) but keep non-finite numbers as YAML `.inf` and `.nan`
  (2026-11-17).
- WTMD: `psi = sim.wtmd.add_bias(w_aux, forces)` computes Ψ of the auxiliary fields, adds the bias
  to the forces of the real fields and returns Ψ, and `sim.wtmd.order_parameter(w_aux)` gives Ψ alone
  (2026-11-17). They replace `compute_order_parameter(langevin_step, w_aux)` and
  `add_bias_to_langevin(psi, langevin)`, which read the modes that the first left in `w_aux_k`.
  Results do not change.
- The saddle-point searches of L-FTS share one retry routine, `SaddlePointSearch.solve_with_retries`
  (2026-11-17): the initial, accepted and intermediate searches fail alike, with `NumericalError` for
  a non-finite Hamiltonian (at an intermediate saddle point it now also names a partition function
  that is not positive) and `SaddlePointError` after the retries, which no longer start from fields
  that are not finite. `LFTS.evaluate_force` is now `SaddlePointSearch.evaluate_force(fields, w_aux,
  langevin_step)`, whose compressor iterations count in `sim.saddle.iterations`, and
  `theory.chains.prepare_run` gives both run loops their first fields. Results do not change.
- The PSCF export refuses a `chi_n` key of two names joined without a separator (`AB`), which SCFT
  refuses and release 0.10.0 took (2026-11-17; see Fixed; tests/scripts/TestPscfConverter.py).
- `PropagatorComputation.advance_propagator_single_segment()` raises unless `compute_propagators()` ran
  for the current fields and cell (2026-11-17): the step uses the fields and tables of that call, and
  before any such call it stepped with fields that were never set (tests/TestInputChecks.py).
- Space-group symmetry (with its CrysFFT paths) and CL-FTS are no longer marked beta in the
  documentation (2026-11-17).
- The 2D plots of `scripts/analysis` fill the grid cell around each point with its value, with the
  second lattice vector pointing up; their images were transposed, with the first axis pointing up
  (2026-11-06).
- The simulation script editor refuses, before it writes a script, discrete-chain block lengths
  that are not multiples of `ds`, with the closest valid length (at least one segment, rounded as the
  library rounds: a ratio within 1e-9 of a half is that half, so 0.15 with ds = 0.1 suggests 0.2),
  `cn-adi2` with box relaxation, and the compressors `lr` and `lram` when χN gives an imaginary
  auxiliary field besides the pressure field; the library refused these when the script ran
  (2026-10-28, 2026-10-29, 2026-11-17; tests/scripts/TestSimulationConstraints.py,
  TestSimulationModel.py).

### Fixed

- `PropagatorSolver.angles` follows `set_lattice_parameters` (2026-11-21); after an SCFT angle
  relaxation it gave the starting angles.
- The console line of WTMD prints a non-integer `ell` as given (`l=2.5`, was `l=2`; 2026-11-21).
- The `Lamellar` tutorial labels the axes of its initial-field plot, the plane x = 0, as z and y
  (2026-11-21); they read x and y. It draws a Matplotlib slice of φ_A instead of a Plotly
  isosurface and points to the analysis notebooks, so no tutorial needs Plotly; the tutorial
  README names the requirements (`pip install "polymerfts[jupyter]"`). QuickStart replaces its
  stale listing of the converted imports by one example and a `grep` check.
- QuickStart and Parameters.md give the `reduce_memory` figures of the current code (2026-11-21,
  measured again on an A10): Gyroid.py at 64³ takes 1002 MiB of GPU memory with full storage and
  390 MiB with `reduce_memory` (1064 and 420 MiB were written for the code of 2026-10-21).
- `plot_wtmd_chib_AB.ipynb` prints phase areas that sum P(Ψ) times the bin width (2026-11-19); they
  added up to 1/dψ (the ordered fraction was right).
- `load_config` reads YAML and JSON files as UTF-8 under every locale (2026-11-18): under a Windows
  cp949 or cp1252 locale a file with Greek letters raised `UnicodeDecodeError`.
- The tutorials run on every installation (2026-11-18): `00_QuickStart` named `cpu-fftw`, which
  MKL-only and CUDA-only installations lack, the `Diblock` appendix `cpu-mkl` and `Lamellar` `cuda`.
  Their outputs are stripped, invalid escapes that Python 3.14 warns about, private `_molecules` and
  repeated OpenMP lines are fixed, and the private `tutorials/TODO.md` is removed.
- `scripts/analysis/plot_wtmd_chib_AB.ipynb` computes `z_inf` from the grid, box and N̄ of the
  statistics file with `renormalization.py`, or takes the user's value (2026-11-18). It used
  0.7691138 for every system, which is 0.49-0.83 on common grids, so the reweighted F(Ψ) could be
  off by a factor of 1.6; the monomer pair is a parameter (tests/scripts/TestWtmdReweighting.py).
  `plot_structure_function.ipynb` writes CSV, since Excel output needed openpyxl.
- Examples (2026-11-18): fields read from MAT files are resampled as periodic, cell-centred fields
  (`zoom(..., grid_mode=True, mode="grid-wrap")`), so DD, DP, SD, SG and PL converge in 190, 60, 58,
  53 and 150 instead of 195, 71, 66, 63 and 151 iterations to the same free energies;
  `clfts/Lamella.py` and the editor's CL-FTS preset use N̄ = 10⁵, where αds = 0.01 is reliable without
  smearing; stale and mislabelled iteration records (PL_Hexagonal, MixtureBlockRandom, BottleBrushLamella3D,
  the extra AM blocks), dead timers, unused imports and names, debug prints and wrong comments are
  fixed or removed.
- On CUDA, dividing a discrete block concentration by its bead factor no longer turns an infinite
  density into NaN (2026-11-18).
- `SCFT.run` took complex `initial_fields` and `q_init` as their real parts, with only NumPy's
  `ComplexWarning`, accepted NaN and infinite values, and stopped with a bare `ValueError` or
  `OverflowError` for ragged lists and integers beyond the float range (`10**400`). Both are checked
  before the optimizer is reset (2026-11-16): each must hold booleans, integers or real numbers that
  are finite at float64 precision (Python integers too, within the double range), or `ValidationError`
  names the entry; strings, also strings of numbers, are refused too (tests/TestScftInitialFields.py,
  TestScftInitialPropagators.py).
- With a space group, `SCFT.run` took `q_init` only in the reduced basis of the solver and refused
  profiles on the full grid, where the initial fields are given, with a size error (with
  `axis_ordering`, reduced ones as well). Full-grid profiles in the axis order of `nx` are converted,
  and profiles that break the symmetry of the group raise `ValidationError` (2026-11-13;
  tests/TestScftInitialPropagators.py).
- Graft maps (`initial_conditions`) read from JSON files, whose object keys are strings, were refused
  by the native `add_polymer` with a `TypeError`. Node keys written as decimal strings are converted
  to integers when the input is read; other and colliding keys raise `ValidationError` (2026-11-14;
  tests/TestParameterNames.py).
- SCFT with `relax_cell` and no space group relaxed another cell than the one given when the lengths
  that a `crystal_system` ties (Tetragonal, Cubic, Hexagonal) differed, also in a 2D cell with
  γ = 120 degrees, which is classified as Hexagonal2D: the first step set b, and for Cubic also c,
  to a without a message. Such cells, and cells whose angles differ from those the given crystal
  system fixes, raise `ValidationError` (2026-10-28; tests/TestParams.py).
- SCFT with `relax_cell`, a `crystal_system` that ties lengths (a = b, a = b = c) and no space group
  never converged when the tied stresses differ: the convergence test required each stress to
  vanish, while the optimizer moves the tied lengths together. Tied lengths count as the sum of their
  stresses divided by √n, and a converged run warns when they do not vanish separately (2026-10-28;
  tests/TestScftOptions.py).
- `SCFT.run` went on iterating with NaN up to `max_iter` when a residual or the error level was not
  finite, as when the iteration diverges, and returned these fields as an unconverged result; it
  stops with `NumericalError` (2026-11-14; tests/TestGceIonRange.py checks it with grand-canonical
  ions whose density overflows).
- `SCFTResult.angles` and the `angles` that `on_iteration` receives were `None` for fixed hexagonal
  and oblique cells; they hold the angles of every 2D and 3D cell (2026-11-13;
  tests/TestScftResultGeometry.py).
- `SCFT.run` stops when `on_iteration` returns the NumPy False of a comparison, as for
  `return info.error_level > 1e-3` (2026-11-02). It stopped only for the Python `False`, and such a
  callback let the run go on without a message. `None` still continues.
- SCFT reported no error when a partition function with real fields was not positive and finite
  at its final fields; it raises `NumericalError` (2026-10-26), as L-FTS does at a saddle point
  whose Hamiltonian is then not finite (see Interface changes). A discrete chain grafted next to an
  absorbing wall in a strong field (TODO.md, item 29) leaves its propagator below the round-off
  level of the spectral transforms, and Q came out of order -1e-19 against +4.67e-28, its value and
  sign varying with the platform and the FFTW plan. Intermediate iterates are not checked, since
  `rqm4` and `cn-adi2` can pass through Q <= 0 during saddle iterations. A polymer of volume fraction
  exactly zero is not checked: it contributes no log Q, and an absent grafted species with a
  zero graft has Q = 0 (`TestScftAbsentPolymer`). The check cannot detect a positive Q made of
  round-off noise; `check_total_partition()` reports that inconsistency. `TestPartitionGuard`
  covers the checks.
- `SCFT.compute_chemical_potentials(w)` failed with a size error for the full-grid fields of
  `SCFTResult.w` of a run with a space group; it takes them (2026-10-28;
  tests/TestPhaseEquilibrium.py).
- `coexistence_quantities()` (2026-10-27) and the PSCF export (2026-11-14) refused boundary names
  that SCFT accepts in another case, such as `"Periodic"`; they compare them in any case
  (tests/TestPhaseEquilibrium.py, tests/scripts/TestPscfConverter.py).
- L-FTS with the `lr` or `lram` compressor diverged in boxes with walls: release 0.10.0 measured and
  applied J(k) with NumPy's `rfftn` whatever the boundaries, and a reflecting 16×12×10 box gave NaN
  where the saddle point is now found in 5 iterations. J(k) is measured and applied in the cosine
  series along reflecting walls; absorbing walls refuse both compressors (2026-10-28; see Interface
  changes; tests/TestLftsDetails.py).
- `LFTS` with the `lr` and `lram` compressors takes J(0) = 1 without `zeta_n` (2026-11-02).
  A uniform shift of the pressure field changes no concentration, so J(0) is zero; the measured
  value was rounding, and exactly zero on some small 1-D grids, where the update divided by zero.
  With `zeta_n`, J(0) stays measured.
- The Hamiltonian of L-FTS and CL-FTS left out its field coefficients below 1e-8 in magnitude, which
  `np.isclose(c, 0)` took for zero: the compressibility term for `zeta_n` above about 5e7 and the
  small linear terms of a nearly symmetric ternary χ (such as -5.5e-9), and L-FTS did not set the
  uniform mode of such a compressibility field to its saddle value. Every nonzero coefficient counts
  (2026-11-13; tests/TestSmallFieldCoefficients.py).
- WTMD (tests/TestLftsDetails.py, TestWtmdRange.py, TestWtmdZeroModes.py, TestWtmdLifecycle.py,
  TestCheckpointValidation.py, TestParams.py):
  - The statistics divided the histogram by the samples inside the Ψ range only (NaN when none
    lay inside, I1/I0 too small when some lay outside) and summed the hills by a circular
    convolution, so a hill near `psi_max` wrapped around to `psi_min`. They divide by all samples
    and sum the hills by a linear convolution over bins extended by 8σ, so samples just outside the
    range deposit their tails (2026-10-28). A run continued from a 0.10.0 checkpoint deposits its
    further hills this way.
  - The order parameter gave the last plane of the half spectrum weight 1 also for odd nz, where
    its conjugate partner is not stored, so Ψ missed part of the full-spectrum sum (2026-10-27); it
    gave NaN forces at Ψ = 0, a zero exchange field, which now gets no bias (2026-10-28), and for
    `ell` < 2 at a single zero Fourier coefficient, which now gets no force (2026-10-30; an `ell`
    below 1, for which Ψ is no norm, raises `ValidationError`); its powers
    overflowed at large `ell` or large fields, where Ψ and its gradient now stay finite
    (2026-11-14).
  - Checkpoints also store the bias arrays without the factor CV, so a continued run reproduces the
    uninterrupted one bit for bit; dividing the stored arrays by CV changed their last bits
    (2026-10-28). L-FTS refuses, when it is constructed, a `dpsi` that does not divide
    `psi_max - psi_min`, which gave bins of another width than the kernels and integrals use
    (`ValidationError`; 2026-10-28, 2026-11-17); a run that release 0.10.0 started with such a bin
    width cannot be continued.
  - A fresh L-FTS `run()` with WTMD (from step 1) drops the samples an earlier run of the same
    object left pending (2026-11-16); they entered the first hill of the new run. The bias and
    statistics that `sim.wtmd` holds stay, also a bias set on it between runs.
- L-FTS saved the field of a monomer type named `aux` under `w_aux`, the key of the auxiliary fields
  in its checkpoints, so `continue_run()` refused them. The statistics, structure-function and WTMD
  records of L-FTS and CL-FTS name a pair by its two names in sorted order joined with `_`, so two
  pairs that give the same name (A, B_C with A_B, C) overwrote each other. L-FTS and CL-FTS refuse
  such monomer names, and L-FTS the name `aux`, with `ValidationError` (2026-11-16;
  tests/TestPairRecords.py).
- Result files (tests/TestScftResult.py, TestLftsDetails.py, TestLftsRun.py,
  TestVtkFieldIntegrity.py, TestMatFieldNames.py, TestNumpyScalarSerialization.py):
  - The checkpoints and `run_status.json` of L-FTS and CL-FTS were created with mode 0600 by
    `mkstemp`; they get the permissions of other new files (2026-10-28).
  - SCFT JSON results wrote the chemical potential -inf of a species of volume fraction 0 as the
    token `-Infinity`, which strict JSON parsers reject; it is stored as
    `{"nonfinite_float": "-inf"}` (2026-10-28).
  - Options given as `None` or as paths stopped L-FTS and CL-FTS when they saved `initial_params`
    to a MAT file, and SCFT when it saved a result (`None` in MAT files, paths in every format;
    2026-10-27).
  - `save_vtk` kept one array of field names that collide once spaces become underscores (`A B`,
    `A_B`); it refuses them before it opens the file (2026-11-14).
  - The `initial_params` of MAT files lost the entries of a monomer name that starts with an
    underscore, which SciPy's MAT writer drops with a warning, and a name of more than 63
    characters stopped the writer; such names are stored under reversible identifiers (2026-11-14;
    see Added).
  - Parameters given as long-double NumPy scalars (`np.longdouble`) stopped L-FTS and CL-FTS when
    they were constructed, and SCFT when it saved a result, with a `RecursionError` from the hash
    of the settings; they are recorded at double precision (2026-11-14).
- `save_vtk` drew the fields of a hexagonal-axes space group half a grid cell off along a and b: such
  a setting samples a and b at the nodes, and `save_vtk` wrote every axis at the cell centres. Given
  the group (`space_group`, see Added) it writes the points at the nodes (2026-11-06;
  tests/TestSaveVtk.py).
- Console output no longer raises `UnicodeEncodeError` when stdout is redirected to a file
  or pipe whose encoding lacks the Greek letters and subscripts it prints (χN, κ,
  βF/(ρ₀V/N)), such as that of a legacy Linux locale (`ko_KR.EUC-KR`, `en_US.ISO-8859-15`) or
  cp949 and cp1252 on Windows: these streams write such characters as backslash escapes
  (2026-10-25). Under EUC-KR or cp949, which lack the subscript, SCFT stopped at the end of every
  run; under encodings without Greek letters, SCFT, L-FTS and CL-FTS stopped at their start.
  `TestConsoleEncoding` runs a short SCFT with an ASCII stdout.
- `SymmetricPolymerTheory.to_aux_fields()` dropped the imaginary parts of complex fields given as a
  dictionary {type: array}; it keeps them (2026-10-30; tests/TestPolymerFieldTheory.py).
- Native API calls that ran with wrong results or memory errors (tests/TestComputationSafety.py,
  TestInputChecks.py, TestComplexMask.py, TestAndersonMixing.py):
  - `get_stress_gce` read past a fugacity list shorter than the polymer types (2026-10-27), and
    `get_total_concentration` and the grand-canonical getter returned a zero field for a name that
    is not a monomer type (2026-10-28). Both raise errors.
  - A propagator computation built with an optimizer from another `Molecules` object overflowed the
    heap; it is refused (2026-10-28).
  - Native objects of real fields took complex arrays as their real parts, with only NumPy's
    `ComplexWarning`. Every real input of the binding refuses them with `ValueError`: the solvers
    (2026-10-28), and the computation box, the fields and `q_init` of the native propagator
    computation, Anderson mixing, the `SpaceGroup` conversions and masks (2026-11-17). The message
    says that the input takes real values and, for the fields of a solver, that complex fields need
    a solver created with `field_type="complex"`.
  - `zero_mean` with a mask divided the masked integral by the whole volume, so the mean over the
    accessible volume did not vanish; it subtracts that mean, also in the reduced and m3 bases
    (2026-10-29).
  - Anderson mixing of complex fields took bilinear products of the residuals where its Gram matrix
    needs Hermitian ones, on the CPU and in both CUDA memory modes (2026-11-13).
- Discrete chains in a masked box gave NaN concentrations at masked sites where w ds exceeds about
  745, since exp(-w ds) underflows to zero there; masked sites are skipped before the division by
  the Boltzmann factor, on the CPU and CUDA (2026-11-13; tests/TestMaskedDiscreteWeights.py).
- Native state and input checks (2026-10-26):
  - `PropagatorSolver` returned the partition functions, propagators, concentrations and stress
    of the previous box after `set_lx()` or `set_lattice_parameters()`, since its flag of
    computed fields was never cleared; like the native `PropagatorComputation`, it also returned
    concentration arrays that `compute_concentrations()` had not written and the stress of an
    earlier `compute_stress()`, and the native getters returned Q = 0 before a computation. The
    getters raise an error that names the missing computation unless their results belong to
    the current fields and box (tests/TestComputationSafety.py).
  - `Molecules.add_solvent()` after the propagator optimizer was built made the computation
    read past its solvent arrays; it raises once the optimizer has fixed the contour lengths,
    as `add_polymer()` did.
  - `PropagatorSolver.get_dv()` and `get_grid_points()` ignored the lattice angles and
    later box changes, and `gaussian_initial_condition()`, which uses these points, centered
    its Gaussian at lx/2; they use the volume and the lattice vectors of the box (new
    `get_lattice_vectors()`), the Gaussian lies at the center of the cell (2026-10-27), and the
    grid points of hexagonal-axes space groups follow their node-centered in-plane convention.
  - `PropagatorSolver.n_grid` gave the full grid before the first computation even with
    a space group, so fields sized from it were rejected; it gives the reduced basis.
  - A solvent took the field of its monomer type from an earlier `compute_propagators()` call when
    the fields of a later call left that type out; the solvent concentrations raise an error that
    names the missing type (2026-10-27; tests/TestComputationSafety.py).
- Native input checks (2026-10-25; `TestMoleculesValidation`, `TestInputChecks`):
  - A monomer name that ends with an underscore, which the name check allows, no longer
    stops the propagator key parser with "invalid stoi argument".
  - `Molecules.get_solvent()` and `Polymer.get_n_segment()` raise an error for an index
    out of range instead of reading past the end of their arrays.
  - A NaN or infinite contour step `ds` is rejected (2026-10-26). The contour length
    mapping compared with `<= 0`, which NaN passes: `PropagatorSolver` then ran with one
    segment per block and returned a plausible partition function. SCFT, L-FTS and CL-FTS
    checked `ds` before.
- Long multiblock chains no longer exhaust the process stack (2026-11-14): the propagator keys
  were generated by a recursion over the blocks, and a linear chain of 24,000 blocks overflowed an
  8 MiB stack. The architecture diagram that `SCFT`, `LFTS` and `CLFTS` print when they are
  constructed recursed over the blocks as well (2026-11-16); both now keep their own stack. Above
  100 blocks the diagram prints one line per block, since its indentation grows with the depth and
  the diagram of a 2048-block chain took 23 MB (`TestPropagatorCodeStack`). The propagator keys
  still grow with the square of the number of blocks (TODO.md, item 37).
- The propagator statistics printed `reduction = -nan %` when the chains need no propagator
  steps, as with discrete chains of one segment only; they print 0 % (2026-10-29).
- `PropagatorSolver.set_lattice_parameters()` with a space group accepted an angle that the group
  fixes moved by up to a relative 1e-6 (9e-5 degrees). A solver chooses its CrysFFT engine once,
  for the angles of its box, and kept an engine that assumes exact angles while the box treated the
  changed cell as oblique: the ObliqueZ engine went on folding with Q up to 1e-5 (relative) off the
  full grid, and the Pmmm DCT and recursive 3m engines differed from a fresh solver by up to 3e-7.
  The box refuses such a change before it changes the cell (2026-11-02 for the ObliqueZ engine,
  2026-11-16 for every engine; see Interface changes; `TestCrysFFTCellChange`).
- Oblique cells with strongly anisotropic grid spacing, for example 4×32 points on a
  120-degree cell with equal lengths, used the wrong |k|² for some modes (2026-10-26).
  The search for the alias of least reciprocal-metric form looked one period around the
  centered index and missed the minimum (28 of the 128 in-plane modes of that example),
  in the Boltzmann factors and the stress of the full-grid path, in the ObliqueZ CrysFFT and
  in the smearing kernel, which came out asymmetric, so that the CL-FTS forces did not match
  the Hamiltonian, and in the WTMD filter. The search is now exact
  (`src/common/MinimalFormAlias.h`), and smearing and WTMD use it too, the smearing also for the
  tied-alias derivatives of its stress (2026-10-30); on grids with comparable spacing it picks the
  same aliases as before, so results there are unchanged (see Interface changes for the flat cells
  it refuses). `TestMinimalFormAlias` checks the search against a brute-force one on 44,000 modes
  of random oblique and triclinic cells; `TestCrysFFTObliqueZCuda` checks two such grids and
  `TestSmearingAliases` the smearing.
- An m3 physical basis enabled on a grid with nz/2 < 8, such as Im-3m 8³ or 12³, made
  `PropagatorSolver` fail with "recursive 3m CrysFFT is unavailable" on every platform
  (2026-10-26). The CrysFFT selector applied the nz/2 >= 8 size rule, meant for choosing
  a basis, to a basis already enabled. The rule now only chooses
  (`SpaceGroup.m3_crysfft_feasible()`, which SCFT uses), and the recursive 3m engines,
  exact for nz/2 >= 2, run an enabled m3 basis on any grid it maps.
  `TestCrysFFTM3SmallGrid` compares these runs with full-grid ones on every platform.
- The git commit and source state that a build records (`build_info` of the native module, in the
  run metadata) were those of the last CMake configuration: a rebuild after a commit or an edit
  kept the old ones, and untracked files did not count as changes. Every build records them
  (2026-10-27).
- CUDA CN-ADI with reflecting or absorbing walls stopped with `cudaErrorInvalidValue` on an axis
  longer than about 4000 points, whose tridiagonal coefficients it copied into shared memory
  (2026-10-25). Lines too long for the shared memory are solved in global memory and agree with
  the CPU solver (`TestCnAdiLongLines`).
- `platform="auto"` chose CUDA for 2D and 3D inputs that CUDA refuses: absorbing axes of odd length
  or of more than 6142 points, and complex fields of continuous chains with walls; it keeps them on a
  CPU backend (2026-10-27, 2026-10-28; `TestCudaOddReflecting`).
- The PSCF export (`scripts/pscf/pscf_converter.py`) exported CL-FTS parameters given with
  `--fields` (Langevin parameters without the `saddle` and `compressor` that L-FTS requires) as an
  incompressible model, which CL-FTS does not run (its `zeta_n` is 100 by default). It refuses them
  with a message that names CL-FTS (2026-11-17; tests/scripts/TestPscfConverter.py).
- The PSCF export refused `chi_n` keys that SCFT accepts, `A_B`, `A B` and `A/B` ("Ambiguous
  interaction key"); it reads them with `parse_chi_n`, as the simulations do (2026-11-14;
  tests/scripts/TestPscfConverter.py).
- Security: the API of the simulation script editor, `/api/save` included, answered every process
  that reaches the loopback interface, so any account of a shared machine could write scripts
  through it. The server prints a random per-run token in the editor URL, and every `/api/` request
  must carry it (2026-10-27; tests/scripts/TestSimulationWeb.py).
- The simulation script editor named the 27th monomer type `M27`, a name with digits that the
  library refuses, so its scripts with more than 26 types failed. It names the types after Z AA, AB
  and so on (2026-10-28; tests/scripts/check_simulation_cells.cjs, which
  TestSimulationConstraints.py runs where Node is installed).
- The PSCF import named the 27th field of a PSCF file without monomer names `M27`, which the library
  and the script import refuse, so files of more than 26 monomer types needed `--monomers`. It names
  them A to Z, then AA, AB and so on, as the simulation editor does (2026-11-17;
  tests/scripts/TestPscfConverter.py).
- Saving a script on the server from the Simulation Script Builder failed on Windows, which has none
  of the directory descriptors (`os.O_DIRECTORY`, `dir_fd`) that keep a POSIX save inside its base
  directory (2026-10-26; tests/scripts/TestSimulationWeb.py). Windows creates the folders one at a
  time, refuses symbolic links and junctions by their resolved paths, and writes the script with LF
  line endings. A folder that another process replaces by a link during a save is not caught there,
  unlike on POSIX.
- `plot_structure_function.ipynb` gave every mode of the real-FFT half spectrum the same weight:
  the planes kz = 0 and, for an even nz, kz = nz/2, which stand for one mode of the full spectrum,
  counted as much as the others, which stand for two, so a shell average depended on the
  orientation of an anisotropic structure. It also averaged the files with equal weights whatever
  their numbers of samples. It weights the modes by the full-spectrum modes they stand for and the
  files by their `sample_count` (2026-10-30). It reads the structure functions of 1D and 2D L-FTS
  runs (2026-11-16); it assumed a 3D grid (`nx[2]`) (tests/scripts/TestStructureFunctionPlot.py).
- The plots of `scripts/analysis` draw the fields at the grid points of the solvers, in the cell of
  the file's lattice vectors (2026-11-06): `plot_2d_density.ipynb` drew every cell rectangular.
  `plot_3d_isodensity.ipynb` drew in grid indices; it now uses the cell centres, or the nodes along
  a and b for a hexagonal-axes space
  group, draws the outline of the cell with true proportions and interpolates a non-orthogonal cell
  linearly onto a Cartesian grid, which Plotly's isosurface needs (the periodic field fills the box
  around the cell). The notebooks take the angles saved in the file, or for older files those of
  its checkpoint settings or input, and read CL-FTS files (the real parts) and the `phi`
  structure of older files, which failed (tests/scripts/TestDensityPlot2D.py, TestIsodensityPlot.py).
- `plot_structure_function.ipynb` groups the modes of a cell that is not orthogonal by their true
  |k| (2026-11-06): the reciprocal metric of the cell, with the index alias of the smallest |k| that
  the solvers use. It took |k|^2 = (2 pi)^2 sum (n_i/L_i)^2, which holds for orthogonal cells
  only, so modes of different |k| shared a shell. Shells group values equal to a relative 1e-9,
  so that round-off of the metric does not split one; the rounding to 7 decimals could. The
  structure function files of L-FTS and CL-FTS save `angles`; for older files the notebook reads
  those of the input in `initial_params` (tests/scripts/TestStructureFunctionPlot.py and
  tests/TestStructureFunctionRecords.py).

### Performance

- Performance work of 2026-11-22 (structural review):
  - CUDA wall boxes and the Pmmm CrysFFT run through one cuFFT of the reordered grid
    (`CudaMakhoulTransform`, as on the CPU) instead of per-axis cosine and sine kernels. A10, 4
    streams, time of a propagator computation: walls 0.88-0.93x on 64³ and 0.67-0.69x on 128³,
    complex fields 0.56-0.72x, the Pmmm CrysFFT 0.86-0.93x; a 128³ wall box needs 620-770 MB less
    device memory. These results change at round-off (largest relative difference 6e-15 against the
    previous CUDA results).
  - The CPU ObliqueZ CrysFFT runs as on CUDA: one real FFT of the physical grid after Makhoul's z
    permutation and the pair update of kz and Nz/2 - kz. One diffusion on 64³ (one thread, AMD EPYC
    7543) takes 1.08 ms with FFTW (was 1.39) and 1.52 ms with MKL (was 2.00); 5-39 % faster from
    24x24x30 on, a few microseconds slower on grids of 6-16 points per axis. The complete solver on
    64³ P6/mmm gains 1.6-1.8x over the full grid (was 1.2-1.4x). CPU ObliqueZ results change at
    round-off (3e-14).
  - Orthogonal grids with an odd `nx` or `ny` and a z mirror (32 Hall numbers, for example Pmmm,
    P4/mmm and monoclinic groups at γ = 90°) take the ObliqueZ CrysFFT instead of the full-grid FFT:
    Pmmm 63x64x64 takes 54 ms on CUDA (was 87, A10) and 578 ms on cpu-mkl (was 924); results agree
    with the run without a space group to 1e-14.
  - Everything else is bitwise unchanged on cpu-mkl and CUDA: the CUDA CN-ADI solver (one matrix
    record per axis, line offsets computed in the kernels), the CPU solver context, the FFT classes
    without a dimension template, one stress-dyad loop, the propagator keys and the computation
    bookkeeping.
- Performance work of 2026-11-21 (review 10). Results are bitwise unchanged:
  - Reduced memory with discrete chains keeps no checkpoint at the segment where a junction branch
    ends; a junction reads only the half-bond states. A discrete bottlebrush (64³, `reduce_memory`,
    aggregation) needs 16 % less host memory on cpu-mkl and 20 % less pinned memory on CUDA, where
    `compute_propagators` is 7 % faster (A10).
  - The CPU solvers build the stress basis of their Pseudo tables at the first stress, as CUDA does:
    runs without a stress (L-FTS without pressure, CL-FTS, SCFT without `relax_cell`) need 53 MB
    (real) and 106 MB (complex) less peak memory at 128³ (cpu-mkl, `reduce_memory`).
  - The SCFT smearing stress of an orthogonal cell without free angles no longer computes the angle
    derivatives and zeroes them, and mean-field electrostatics transforms the ion densities once per
    field.
  - CUDA: the solvers keep no per-stream work buffers that hold copies or are never written (the
    RQM4 full step stays in its batch buffer, the RK2 step runs in its output, the two CN-ADI 3D
    sweeps share one intermediate, a space group needs one full grid per stream for the
    pseudo-spectral and CN-ADI steps), and the stress workspace (for `rk2` and `ps` also the second
    spectrum, which only the stress reads) exists only from the first stress on. At 128³ with 4
    streams (A10) runs with a stress need 64 MiB less device memory with `rqm4` and `rk2` (128 MiB
    with a P-1 group; `ps` 64 MiB with P-1 and unchanged without a group, as are the `rqm4` wall
    path, the m3 CrysFFT and `rs`), and runs without a stress 64-384 MiB less (`rqm4` 104, `rk2`
    168, `ps` 104, complex `ps` 384, `cn-adi2` 64-128 MiB).
  - CPU: FftwFFT multiplies the bond factor into the coefficients a propagator step gives up and
    runs the complex-to-real transform on them instead of on a thread-local copy (cpu-fftw `rk2`
    steps 6-8 % faster on one thread and 10-13 % on four at 96³-128³), and the stress of a
    non-orthogonal cell is accumulated in one pass (hexagonal 64³ 5 %, 2D oblique 256² 10 % faster
    on cpu-mkl with four threads).
- Performance work of 2026-11-19 (review 9). Results are bitwise unchanged:
  - With `reduce_memory`, the concentrations and the stress come from one recomputation of the
    propagators instead of two: an SCFT iteration with cell relaxation is 26-31 % faster (64³, rqm4
    and ps, cpu-fftw and CUDA A10).
  - Cells with an axis orthogonal to the other two (hexagonal, monoclinic) search the least and tied
    aliases once per mode of each component instead of once per mode of the grid: the stress basis of
    a hexagonal 64³ cell takes 2.2 instead of 11.6 ms, a CUDA cell change of the HCP relaxation 6.6
    instead of 13.9 ms (run 9 % faster), a hexagonal 64³ cell change on cpu-mkl 8.0 instead of 18.4 ms.
    Pseudo keeps no zero stress tables in 1D and 2D, and a CPU solver whose stress takes the CrysFFT
    fast path builds no Boltzmann tables.
  - CUDA: full-memory runs allocate no junction staging and accumulator per stream (256 MiB at 128³
    with 4 streams), and the pseudo-spectral solvers share one spectral scratch per stream between the
    steps and the stress, with no periodic spectra on walls: 13-23 % less device memory at 128³
    (periodic rqm4 2372 → 2044 MiB, walls rqm4 3408 → 2640 MiB, 4 streams). One Anderson mixing class
    keeps its history in device or pinned host memory and adds the history terms in one pass per 16
    terms (26 % less time per call at 128³ with 20 histories).
  - CPU: the wall transforms and the Pmmm CrysFFT apply the bond and Boltzmann factors inside their
    loops (1-3 % per step with walls); every per-thread scratch is an `AlignedScratch`, and FftwFFT
    takes scratch only for the arrays it copies.
- Performance work of 2026-11-18 (review 8). Results are bitwise unchanged unless stated:
  - `compute_propagators` resolves the jobs of the schedule when the computation is built and looks
    up no propagator keys: a linear chain of 1000 blocks on a 16-point grid takes 3.4 instead of 52 ms
    (CPU), linear in the number of blocks.
  - Reduced-memory concentrations and stress recompute each chunk from the checkpoint where it
    starts: 13-20 % and 6-18 % faster at 64³. A CPU junction skips `pow` for a branch power of one
    (discrete star propagators 6-8 % faster).
  - The CPU `cn-adi2` solver keeps the Thomas factors of each axis and its neighbour tables and solves
    the lines of a plane together: 8-9x faster per step at 64³, 4.6-5.7x at 256², 1.4-1.9x in 1D (one
    thread). Its work arrays and those of `rs` belong to the solver, per thread, and are freed with it.
  - The CPU pseudo-spectral steps apply the bond factor, the 1/N and the Boltzmann factor in the
    loops around the backward transform (`FFT<T>::backward_with_factors`): rqm4 8 %, rk2 3.5 % and
    ps 5 % faster at 64³ with FFTW. The CPU stress reuses its per-thread arrays, the CrysFFT engines
    run on the caller's fields, and `PropagatorSolver.fourier` copies real periodic coefficients once
    (6-8 % with FFTW).
  - A cell change of an oblique cell skips the alias search for every mode within half the shortest
    period (an exact early exit, about 80 % of the modes of a hexagonal grid) and computes the least
    forms once for all contour steps: a hexagonal 64³ cell change takes 19 instead of 44 ms on
    cpu-mkl and 8 instead of 44 ms on CUDA, and the HCP relaxation 8.8 instead of 12.0 s (CUDA A10,
    64³, 100 iterations). The CPU solvers rebuild their full-grid tables only while a full-grid path
    uses them (Ia-3d m3 basis: 5.3 instead of 13.3 ms per cell change); CUDA keeps no full-grid
    Boltzmann tables for the ObliqueZ steps of continuous chains and builds the stress basis at the
    first stress (Langevin runs save about 51 MB at 128³).
  - On CUDA every field is uploaded once per `compute_propagators`; a reduced basis is expanded on
    the device. The segment stress reads the two propagators where they are instead of copying each
    pair on a second stream with events: `compute_stress` takes 20-27 % less time with a space group
    at 64³ and 3-7 % less on periodic grids (A10). The cuFFT plans of a solver share one work area
    per stream and each stream creates only the plans it runs: at 128³ with four streams a periodic
    solver takes 346 MB (ps), 410 MB (rk2) and 282 MB (rqm4) less device memory, and
    `PropagatorSolver.fourier` 18-32 MB less. With CrysFFT steps and a full-grid stress (ObliqueZ,
    or the off-diagonal flag) the solvers allocate only the buffers of the stress.
  - SCFT electrostatics transform ψ once per iteration and compute the point ions once per ψ, and
    the SCFT smearing of real fields in periodic boxes works on the half spectrum of
    `PropagatorSolver.fourier`, on the GPU with CUDA: an electrostatic relaxation iteration (48³,
    two ions) takes 152 instead of 175 ms on cpu-mkl and 35 instead of 58 ms on CUDA (A10), a
    hexagonal relaxation with smearing (48×48×40, CUDA) 82 instead of 103 ms. Relaxations with smearing change at round-off (stress up
    to 5e-15 relative; with electrostatics F up to 1e-14, since the electrostatics take the
    separable |k|² of the smearing in orthogonal cells).
  - The fields are mapped between monomer and auxiliary fields with row sums instead of `np.matmul`,
    whose OpenBLAS threads kept spinning after each mapping and took the CPUs of the OpenMP threads:
    a CL-FTS step of a charged diblock takes 100 instead of 264 ms (32³, cpu-mkl, 4 CPUs) and on
    CUDA 18 instead of 78 ms (A10), an L-FTS step with WTMD (64³, CUDA) 242 instead of 733 ms.
    Results with two monomer types are bitwise unchanged; with three or more, L-FTS, CL-FTS and
    SCFT results change at round-off (1e-13 relative after 10 L-FTS steps).
  - CL-FTS reuses dH/dw and the charge density of the step for the pressure, caches the ion charge
    per ψ, and S(k) samples keep their list of fluctuating fields. `scipy.signal` is imported when the
    WTMD bias is updated (`import polymerfts` 0.60 instead of 0.76 s).
- The periodic transforms of `FftwFFT` run their FFTW plans on the caller's arrays through FFTW's
  new-array execute functions when these have the alignment of the planning arrays (2026-11-16).
  Only the input of the complex-to-real transform, which FFTW overwrites, arrays 8 bytes off (packed
  fields after an odd number of grid points) and overlapping input and output go through per-thread
  scratch of that alignment; before, every transform copied its input, those of real fields also
  their output, and each object kept its planning arrays. With the same plans a round trip on one thread of an AMD EPYC 7543 takes 10-26 % less time
  on 32³-128³ (complex fields 12-25 %), and `compute_propagators` (rqm4, 64³, one thread) takes 1.4 s
  instead of 1.55 s there and 1.3 s instead of 1.5 s on an Intel Xeon 6737P. Results are bitwise
  unchanged. The FFTW and MKL Pmmm CrysFFT engines are one engine over the real FFT of the selected
  library.
- On CUDA a junction takes one kernel launch over up to 32 device-resident sources and their
  powers, with the product in registers, instead of one launch per factor (2026-11-08, 2026-11-16);
  the reduced-memory mode stages two checkpoints per launch. An integer power of at most 4 (up to
  four identical branches) is still taken by successive multiplications, and such junctions are
  bitwise unchanged. A larger one, a repeated branch such as `A10:5` or an integer block
  multiplicity up to 10⁶, takes one `pow` per point and changes at round-off (2.7e-15 relative for a
  41-arm star); a 64³ star with multiplicity 10⁶ takes 17 ms per `compute_propagators` on an A10, as
  a non-integer multiplicity does.
- `cn-adi2` on CUDA factorizes the tridiagonal matrix of each axis once, with the Sherman-Morrison
  vector of periodic axes, instead of in every solve of every line, and stages the lines along the
  contiguous axis in shared memory, whose reads were not coalesced (2026-11-05). One step on an A10
  takes 1.2 ms instead of 2.1 ms with reflecting walls and 1.3 ms instead of 4.2 ms periodic on
  128³; the gyroid benchmark of docs/theory/NumericalMethods.md, Section 6, runs in 30.5 s instead
  of 49.2 s at Ns = 1000. The per-stream Thomas workspaces of the size of the grid are gone.
- The DCT and DST of the CPU backends run as one reordered real FFT of the whole grid
  (`MakhoulTransform`, 2026-11-02), one class for FFTW and MKL: `createFFT()` returns `MakhoulFFT`
  for a box of walls, and `FftwFFT` and `MklFFT` are periodic only (2026-11-17). Both libraries took
  each line in turn with a copy per line and per axis. The real FFT of the grid runs directly on the scratch
  arrays of `MakhoulTransform`, which also splits complex fields (2026-11-16). On 64³, one thread,
  double precision, a round trip with walls takes 3.2 ms instead of 16 ms with FFTW and 3.9 ms
  instead of 12 ms with MKL on an AMD EPYC 7543, and 2.6 ms and 2.4 ms on an Intel Xeon 6737P;
  `compute_propagators` gains 4-6x on the AMD and 3-4x on the Intel processor. The 1D FFTW plans
  came out twice as slow in some processes; 32 repeated runs show no such run now. The new path is
  also 1.6-2.1x faster on AMD, 1.2-1.7x on Intel, than one `fftw_plan_r2r` of the grid. The Pmmm
  CrysFFT runs on the same transform and gains 6.3-6.5x with FFTW instead of 3.8-4.0x and 5.6-5.9x
  with MKL instead of 2.2x, and its stress on a cell of three different lengths gains 1.6-2.1x,
  where MKL lost time before. Results change at round-off: Q agrees with the previous code to 7e-14
  on grids up to 96³, and all space groups agree with the full grid to 1.2e-13 on both libraries.
- The MKL ObliqueZ CrysFFT takes its z DCT through a real FFT of the same length instead of a
  mirrored sequence of twice that length, and its xy FFT on contiguous z planes, where this step on
  64³ ran 1.8 times faster, transpositions included (2026-11-02). With a P6/mmm group on 64³ the
  solver was slower than without a group (0.9x) and now gains 1.2-1.3x (double precision, one thread,
  AMD EPYC 7543), FFTW 1.5-1.6x. All space groups agree with the full grid on MKL as before, to 1.2e-13.
- The transforms of every Langevin step in the Python layer use `PropagatorSolver.fourier`,
  the platform FFT, on the GPU with CUDA, instead of NumPy/SciPy on the host (2026-10-27):
  the `etd` step (its matrices are Fourier multipliers, applied in one call), smearing,
  the transforms of ψ, the LR compressor
  of L-FTS and WTMD; at each sample, the structure functions of L-FTS and CL-FTS (all
  densities in one call, all fields in another; boxes with walls keep numpy.fft, whose
  coefficients they record) and the dilation of the smearing for the osmotic pressure.
  Results change at round-off: over 8 steps of the charged diblock of
  `TestClftsEtd` the Hamiltonians and ψ agree with the previous code to 8e-16 (LM and
  `etd`, CPU and CUDA), an L-FTS run with LRAM, smearing and WTMD to 7e-10, within its
  saddle tolerance, and an SCFT run with smearing and cell relaxation to 3e-11, within its
  tolerance. Lamella with Gaussian smearing on an A10, ms per step: CL-FTS 32³ LM 20.8 → 17.1,
  `etd` 23.7 → 17.3; 64³ LM 172 → 99, `etd` 231 → 99; L-FTS (LRAM) 32³ 136 → 94, 64³
  808 → 291. `set_linear_response()` of an integrator receives `fourier`;
  `Smearing` and `WTMD` take the transform as an optional argument, and `LR` requires it.
- L-FTS smears the concentrations of an accepted step and of the last saddle iteration once (they
  were smeared again for the next forces), and CL-FTS computes δH/δw once per step, reusing that of
  the progress line for the next forces (2026-11-17). Results are bitwise unchanged.
- The CUDA ObliqueZ CrysFFT runs the DCT along z and the FFT in the plane as one real 3D
  FFT of the half grid after Makhoul's permutation along z; one kernel applies the
  Boltzmann factors of each pair kz, Nz/2 − kz between the two transforms (2026-10-25).
  A diffusion step takes three kernels and two cuFFT calls instead of eight kernels, four
  cuFFT calls and a device copy. The propagators of an AB diblock with P6/mmm and the
  z-mirror basis (continuous chains, RQM4, double precision, RTX 5060 Ti) take 1.2-1.6
  times less time from 24×24×30 to 96³. Relative to the full-grid FFT this path now takes
  1.66 times as long at 24×24×30, 0.89 at 48×48×60 and 0.51 at 96³ (before: 2.26, 1.09
  and 0.69). Results change at round-off (about 1e-15 relative). `TestCrysFFTObliqueZCuda`
  compares the CUDA and CPU engines on random fields. The spectral step updates the half spectrum
  in place, in mirror pairs, so each stream holds one spectrum (2026-10-26; bitwise identical).
- CUDA, results unchanged bit for bit (2026-11-17):
  - With the Pmmm DCT or recursive 3m CrysFFT and its stress fast path, the pseudo-spectral solvers
    hold nothing of the full grid: the full-grid cuFFT plans, the per-stream work buffers and the
    full-grid k-space tables (bond factors, Fourier basis) exist only for a path whose propagator steps
    or segment stress run on the full grid (the solvers take the space group in their constructors
    and allocate the work of its path there; the stress allocates the full-grid work when the
    off-diagonal stress flag moves it to the full grid), and a cell change no longer recomputes
    full-grid tables that no step uses; exp(-w ds) and the RQM4 step buffers hold the reduced
    basis. At 128³ with four streams a computation with the Pmmm or m3 basis holds 1.0-1.2 GiB less
    device memory (968 instead of 2188 MiB with Pmmm and `rqm4`). The full-grid tables are created
    when a full-grid path first needs them and computed once per cell, where every pseudo-spectral
    solver computed them when it was created (about 85 MB of host memory at 128³). On the CrysFFT
    path of discrete chains the computation uses the reduced-basis exp(-w ds) of the solver directly
    instead of a synchronous device copy per monomer type at every field update.
  - The recursive 3m CrysFFT transforms its input directly instead of copying it to the output
    first (each step and stress multiplier); it stages only an input that cuFFT cannot use directly
    (not 16-byte aligned, such as the second propagator of a stress pair of an odd basis size, e.g.
    an m3 basis on 18³; `TestCrysFFTCellChange`), and an output must be aligned. The periodic
    backward transform of `CudaFFT` with a complex spectrum no longer copies the spectrum, which is
    scratch afterwards.
  - The unused CUDA DCT and DST types 1 and 4 (about 2000 lines with the kernels they left unused),
    the undocumented `FTS_PROFILE_CRYSFFT_PMMM`, `_PMMM_DETAIL` and `_HEX_DETAIL` timers of CUDA and
    `FTS_PROFILE_CRYSFFT_HEX_CPU_DETAIL` of the CPU, and unused internal kernels, methods and
    constructor parameters of the CUDA solvers are removed.
  - cuFFT plans are held by an owning handle, `CufftPlan` (`CudaCommon.h`), in the solvers, the
    Fourier and DCT/DST transforms and the CrysFFT engines, which drops their flags, release
    functions and cleanup catches; the solvers and `Pseudo` share one rule for the size of a
    spectrum.
- CUDA CrysFFT, results unchanged bit for bit (2026-10-26):
  - The CrysFFT steps transform in place, which saves the device copy of an out-of-place
    Pmmm or recursive 3m diffusion.
  - The z-major DCT routines that ObliqueZ no longer uses are removed.
- CUDA, results unchanged bit for bit (2026-10-25):
  - The pseudo-spectral solver owns the CrysFFT k-space tables, one per key for all streams (the
    Boltzmann factors of each step coefficient and the stress multipliers of each axis group
    and bond coefficient; 2026-11-17), instead of one set per stream. It builds them before the
    propagator and stress loops and rebuilds them in place when the cell changes; the CrysFFT
    objects of the streams keep their plans and work buffers and take the table as an argument,
    and the recursive 3m and ObliqueZ ones share their twiddle factors. A recursive 3m table is one
    allocation instead of 16.
  - The CrysFFT Boltzmann tables are built when the computation is created and when the cell
    changes, for the monomer types and contour steps of the blocks (2026-10-26), instead of with
    allocations and synchronous uploads at the first step of each kind inside the propagator loop
    (`TestCudaCrysTablesAllocation`).
  - The Anderson mixers no longer make redundant device copies.
  - With a space group and a mask, the mask multiplies the propagator in the reduced basis
    after the reduction; it was expanded to the full grid at every step.
  - `update_dw` uploads or expands each field once per monomer instead of once per contour
    step size, and reads fields already on the device without a copy.
  - The discrete-chain CrysFFT steps no longer copy the propagator through work buffers.
  - The reduced-basis expansion and reduction and the CrysFFT steps of the RQM4, RK2 and
    discrete solvers are implemented once for the three solvers; unused device buffers are
    removed.
- Cells with α = β = 90° and γ ≠ 90° use the ObliqueZ CrysFFT with the irreducible basis
  too, when the space group has a z-mirror, for example a hexagonal or monoclinic
  `SpaceGroup` passed to `PropagatorSolver` without a physical basis (2026-10-25). They
  used the full-grid FFT. The transform is the one of the z-mirror basis, which SCFT
  enables on these cells, so SCFT is unchanged; on small CUDA grids it remains slower
  than the full-grid FFT (see above). `TestSpaceGroupPlatformEquivalence` adds a
  monoclinic P2/m cell (γ = 100°, odd n_x). `TestCrysFFTSelector` checks the CrysFFT
  each space group, grid and cell takes: the equivalence tests pass whatever is chosen.
- The CPU CrysFFT engines (Pmmm, recursive 3m, ObliqueZ) no longer build a copy of their
  k-space tables per OpenMP thread (2026-10-26): for a 128³ recursive 3m run with four contour
  coefficients and eight threads this saves about 0.9 GB and seven of eight table builds after
  every cell change. As on CUDA, the pseudo-spectral solver owns the tables, one per key for all
  threads (2026-11-17): the Boltzmann factors of each step coefficient, built for the monomer
  types and contour steps of the blocks when the CrysFFT path is set up, and the stress
  multipliers of each axis group and bond coefficient, built by `prepare_stress()` before the
  stress loop; a cell change rebuilds them in place. The engines take the table as an argument, so
  the propagator and stress loops take no lock and build no table, and the Pmmm stress multipliers
  are built once per cell instead of at every segment. A recursive 3m table is
  one allocation instead of 16, its arrays a cache line apart beyond their length: the Gyroid
  example with Ia-3d on 32³ takes 5-7 % less time per iteration (AMD EPYC 7543, four threads). The
  FFTW ObliqueZ engine frees its planning arrays after planning. Results are unchanged bit for bit
  (3072 arrays compared with the previous code: both libraries, `rqm4`, `rk2` and `ps`, both memory
  modes, local ds, multiplicities, cell changes, one and four threads); `TestCpuCrysFFTTables` checks
  concurrent calls against serial ones and tables rebuilt in place against a fresh engine.
- Code shared between backends, results unchanged (2026-10-26): the CPU and CUDA
  recursive 3m engines share the fold, twiddle and multiplier helpers
  (`src/common/Recursive3mHelpers.h`), and the CPU and CUDA ObliqueZ engines the
  reciprocal metric (`src/common/ReciprocalMetric.h`), whose cell geometry the computation box
  uses too (2026-11-17); both solvers choose their CrysFFT engine through `plan_crysfft()` in
  `CrysFFTSelector.h`.
- `plot_structure_function.ipynb` groups the modes into shells with `np.bincount` instead of a
  Python loop over the grid (2026-11-16).

### Tests

- Tests changed 2026-11-22 (structural review): the 21 near-copy C++ regression tests are merged into
  table-driven files that keep every stored value, grid, box, seed and tolerance: `TestPseudoLinear3D`,
  `TestPseudoSolvent3D`, `TestPseudoBranched3D` and `TestPseudoBranchedMixture3D` (from the 12
  `TestPseudo*{Continuous,Discrete}3D{,Complex}`), `TestAggregation` (3D, Dendritic, SideChains),
  `TestCrankNicolson` (1D/2D/3D) and `TestScft` (1D and 2D; the 3D box runs as the entry `TestScft3D`).
  Each counts its runs; the complex linear runs compare the complex values (formerly the real parts),
  the real solvent runs check the total partition function of both blends, and the branched runs
  compare Q relative to its stored value to 1e-9 (an absolute 1e-7 was 0.15 % of a Q of 7e-5). The C++ propagator tests
  take their computations from `make_platform_runs()` (TestComputationFactory.h): the 19 local
  device checks that ended a test before its CPU parts are gone, so a CUDA build without a visible GPU
  runs the CPU parts and then fails, as `TestParameterNames` now does too; `TestAbsorbingVsMask` runs
  on MKL-only builds. `TestChargedStructureFunctionRPA` runs its two systems as two entries
  (`TestChargedStructureFunctionRPAPolyampholyte`). New: `TestSpaceGroupOrbits` (the orbits and
  physical bases of all 530 Hall numbers against the spglib operations), the CUDA complex walls of
  continuous chains in `TestStressComplexWalls` and `TestComplexNonPeriodicMirror`, the lengths CUDA
  refused before in `TestCudaDCTDST`, `TestCudaFFTMixedBC` and `TestCudaOddReflecting`, strided lines
  in `TestCudaTridiagonal`, the stream handle in `TestCudaDeviceBuffer`. `TestCompareBranchKey` checks
  the order of `ComparePropagatorKey`; `TestPropagatorCodeReduction` checks its third key; the
  parser tests check the recorded edges (`TestPropagatorParserSafety` is gone, its graft-label check
  is in `TestMoleculesValidation`). The never-compiled Richardson branch of
  `TestRealSpaceLinearContinuous3D` is removed.
- Tests changed 2026-11-21 (review 10): `TestClftsElectrostatics` checks the screened S_ij and
  charge structure functions in a second short run with canonical point ions and an asymmetric fixed
  charge (sign and conjugate errors of their terms passed before); `TestComputationSafety` checks the
  grand-canonical concentrations of a star with three identical arms, so a fugacity weight that
  misses the copies of a block fails; `TestGibbsEnsemble` checks which samples enter the averages
  with `sample_interval` 2 and `skip_steps`, and the ion columns with a grand-canonical ion listed
  first; `TestWtmdLifecycle` checks that the recorded L-FTS drift includes the WTMD bias and the
  printed ℓ; `TestMinimalFormAlias` checks four two-component cells against the brute-force alias
  search, and the per-mode and grid alias loops share one tie-combination routine;
  `TestOpenMPDefaults` checks the stack size that libgomp took (OMP_DISPLAY_ENV), not only the
  environment; `TestParams` checks unknown polymer and block keys and the retired
  `fts_consistent_energy`; `TestScftOptions` checks `PropagatorSolver.angles` after an angle
  relaxation and no longer runs two relaxations for the `stress_interval` warning. `TestTutorials`
  calls the frame function of every animation but draws no frame (62 → 32 s on an A10).
  `TestConcentrationsAndStress` and `TestCrankNicolson1D/2D/3D` carry the `numerical` label too.
  The test alias `python_platform` is gone (`cpu_platform`). `TestCudaDCTDST` replaces `TestCudaDCT`
  and `TestCudaDST` (the same cases in one parameterized test, 381 instead of 1411 lines; the round
  trip runs on CUDA too), and `TestCudaDeviceBuffer` and `TestCudaCrysTablesAllocation` use the
  `ScopedStreams` of `TestCudaPart.h`. `tests/scripts/TestVersionNumbers` checks that
  pyproject.toml, CMakeLists.txt (whose number `polymerfts.__version__` reports), the conda recipe
  and CITATION.cff give the same version. `TestCpuFFT` replaces the six `TestCpuFFT{1D,2D,3D}` and
  `Complex` tests with their stored values and adds an O(N²) DFT on even and odd grids;
  `TestDCTDSTPropagator` replaces `TestDCT2Propagator` and `TestDST2Propagator`; the MKL ObliqueZ
  work-array check runs in `TestCpuCrysFFTTables`, so CPU-only builds run it; `TestCrankNicolson1D/2D/3D`
  compare every solver with their stored references only, not also with the solver checked before it,
  which the references imply.

Tests new since 0.10.0:
- Tests added 2026-11-19: `TestConcentrationsAndStress` (the combined call against the two calls,
  bit for bit, in both memory modes on every platform), `TestOpenMPDefaults` (the OpenMP defaults of
  the package reach the runtime; Linux), `TestTutorials` (the code cells of every tutorial notebook,
  Langevin runs capped at 5 steps); new checks in `TestScftElectrostatics` (a relaxation with point
  ions against finite differences of F), `TestWtmdLifecycle` (every WTMD key at a non-default value),
  `TestGibbsEnsemble` (continuations that change the platform and the memory mode),
  `TestParams` (unknown section keys), `TestSmearingLattice` (length derivatives on the half
  spectrum), `TestAndersonMixing` (20 histories in both memory modes), `TestMinimalFormAlias`
  (component-wise grid loops against the per-mode search).
- `TestAbsentPointIons`, `TestAbsentSolvent`: point ions and solvents of zero amount, whose Q is
  zero, have zero density and finite observables; such ions run and continue bit for bit, and
  occupied solvents keep their normalization.
- `TestAndersonMixing`: Anderson mixing of complex fields with the Hermitian Gram matrix, the
  wrap-around and reset of the history against a least-squares solve, non-finite histories.
- `TestBlockMultiplicity`: integer multiplicities against explicit arms (Q, φ, stress); real ones:
  conservation, full against reduced memory bit for bit, stress against -d ln Q/dL, convexity in the
  arm number, continuity at integers; input errors, SCFT and L-FTS.
- `TestBoxVolumeElement`: boxes created with or without right angles, or changed to an orthogonal
  cell, with and without a mask, against a box created without angles (dv, volumes, Q and φ bit for
  bit; cpu-fftw to 1e-13), and oblique cells, on every platform; every box, created or moved, has
  its own volume element (the product of the spacings or V/M, times the mask), which `get_dv()`
  gives, and the integrals of a Pmmm reduced basis use it (2026-11-17).
- `TestCanonicalIonRange`, `TestGceIonRange`: densities and ln Q of canonical and grand-canonical
  ions stay finite in large uniform potentials and Donnan offsets, with short runs.
- `TestChains`: `chain_fields` averages the monomer fields of a random copolymer in double
  precision for integer, float32, float64 and complex input.
- `TestChargedStructureFunctionRPA`: the charged CL-FTS structure functions of a charged diblock
  with counterions and of a polyampholyte against the Debye-Hückel RPA, with tolerances above two
  standard errors of block means.
- `TestCheckpointCompatibility`: L-FTS, CL-FTS and discrete-chain checkpoints (format 2) written by
  main at 895d7e75 (whose version string reads 0.10.0, though it does not descend from the release)
  and a CL-FTS checkpoint written by release 0.10.0 continue to the end of an uninterrupted run of
  this version: those of main bit for bit on the writer's machine (the release's to 2e-16 in the
  structure functions), else within 1e-10 (CL-FTS) or 100 times the saddle tolerance (L-FTS)
  (tests/data/old_checkpoints). Continued from the release, the chemical
  potentials of its pending samples are NaN and the drift of every step is recorded from the
  restart on.
- `TestClftsElectrostatics`, `TestClftsEtd`, `TestClftsGrandCanonical`, `TestClftsPointIons`,
  `TestClftsThermodynamics`, `TestConfinedElectrostatics`, `TestGibbsEnsemble`,
  `TestScftElectrostatics`: the charged systems, the `etd` integrator, grand-canonical species,
  point ions, the chemical potentials and pressure, walls and fixed charges, the Gibbs ensemble and
  mean-field electrostatics, as the entries above describe; also the refusal of a wrong
  `psi_noise_prev` before a run starts, and a Gibbs ensemble that warns once about the options of
  other methods and continues without them (2026-11-17).
- `TestCnAdiLongLines`: CN-ADI lines of 4000 to 13,000 points in 1D, 2D and 3D, periodic,
  reflecting and absorbing, also a contiguous line behind a periodic axis, CUDA against the CPU.
- `TestConsoleEncoding`: SCFT with an ASCII stdout writes its Greek letters as escapes.
- `TestContourPrecision`, `TestDiscreteComposition`: contour lengths keep their values across
  rounding and aggregation (φ equals the block fractions to 2e-13); discrete chains use their
  lengths N ds in compositions, the uniform saddle point and the chemical potentials.
- `TestCpuCrysFFTFullGrid`: the CPU CrysFFT paths (Pmmm DCT, recursive 3m, ObliqueZ; Pmmm, m3,
  z-mirror and irreducible bases) against the full grid with unequal bond lengths, blocks that are
  not multiples of ds and block multiplicities, for `rqm4`, `rk2` and `ps` in both memory modes and
  after a cell change (Q, φ, stress; 2026-11-17).
- `TestCpuFFTAddressIndependence`: the periodic FFTW transforms give the same bits at every 8-byte
  offset within a 64-byte line (MKL at offsets of 0, 16, 32 and 48 bytes), on even and odd grids,
  for real and complex fields, in place (FFTW also with overlapping arrays), and keep their inputs;
  also the recursive 3m and ObliqueZ CrysFFT engines and `backward_with_factors` (2026-11-18).
- `TestCpuMakhoulTransform`: the DCT and DST of `MakhoulFFT` over FFTW and MKL on grids of one to
  three axes, and the transforms of boxes that mix periodic axes and walls on 154 grids of two and
  three axes per library (even, odd and unit lengths, every mix of axes), against their O(N^2)
  definitions; real and complex fields of standard deviation 5, in-place and concurrent calls.
- `TestCpuCrysFFTTables`: one CPU CrysFFT engine (Pmmm, recursive 3m, ObliqueZ, each library)
  called by eight threads with two tables against serial calls, tables and multipliers rebuilt in
  place on a cell change against a fresh engine, tables of another size refused.
- `TestCrysFFTCellChange`: Pmmm DCT, recursive 3m and m3 CrysFFT solvers after a cell change, one
  axis alone included, against fresh solvers (Q, φ, stress) on every platform; fixed angles moved
  within or beyond 1e-10 rad, monoclinic P2/m included; the off-diagonal stress flag turned on after
  a computation on the stress fast path and a cell change, and on again after it was off over
  another cell change, against a solver created with the flag (2026-11-17).
- `TestCrysFFTM3SmallGrid`: an m3 basis on Im-3m grids of 8³ and 12³ against the full grid.
- `TestCrysFFTObliqueZCuda`, `TestCrysFFTSelector`, `TestCudaCrysTablesAllocation`: the CUDA
  ObliqueZ engine against the CPU one; the engine that each space group, grid and cell takes; no
  device allocation in the propagator and stress loops or across cell changes, the tables of a
  coefficient taking the same device memory with one and four streams, and the stress after a cell
  change against a fresh computation; on the stress fast path the CUDA solvers hold less than two
  full-grid fields of work per stream, and the off-diagonal stress flag adds at least two
  (2026-11-17).
- `TestCudaCrysFFTStreams`: four CUDA streams against one, bit for bit, across two cell changes
  (Q, φ, stress; Pmmm DCT and recursive 3m, continuous and discrete chains).
- `TestCudaMakhoulTransform`: the CUDA transforms of boxes that mix periodic axes and walls against
  the O(N^2) definitions (to 1.9e-14) and `MakhoulTransform` (to 7.3e-16), for real and complex
  fields, with the inputs kept; on two grids of more than two million points every grid-stride loop
  of the kernels takes more than one stride. Both Makhoul tests take the definitions, grids and
  mixes of axes from `tests/TestMakhoulDefinitions.h`.
- `TestCudaOddReflecting`: CUDA with odd reflecting axes in 1D, 2D and 3D against each CPU backend.
- `TestCudaRealSpaceRecovery`: a failed upload of the CUDA `rs` operators leaves the solver unusable
  until a complete refresh, which matches a fresh solver bit for bit.
- `TestDriftStepHistory`: the drift maxima and partition ratios of every CL-FTS step, their
  restarts and checkpoint checks.
- `TestExamples`: every script under `examples/` builds its simulation, in a copy of its directory,
  with platforms the build lacks replaced by a CPU platform; warnings about the input are errors, and
  with a CUDA device the first simulation runs briefly (6 SCFT iterations, 5 Langevin steps) and the
  script continues to its second run (2026-11-18).
- `TestFourierTransform`: `PropagatorSolver.fourier` against NumPy and SciPy in periodic boxes,
  boxes with walls and mixed boxes, oblique cells, odd sizes and extreme magnitudes, with
  multipliers, CUDA against the CPU; the Boltzmann factor of the solver mode by mode on 4×32 points
  at 120 degrees, and the discrete-chain stress against central differences on 6×40 points at 125
  degrees, which the one-period alias search of 0.10.0 fails (2026-11-17).
- `TestGceZeroPartition`: a zero grand-canonical Q is regular, a zero canonical Q is accepted only
  for a species of zero amount, with real and complex fields on every platform and in both memory
  modes.
- `TestInputChecks`, `TestMoleculesValidation`, `TestPropagatorParserSafety`,
  `TestRandomCopolymerFractions`: input checks of the bindings (complex arrays refused by every
  real input, with a message that names `field_type` only for the fields of a solver) and of
  molecules and polymers, malformed propagator keys, fractions of random blocks; reduced-basis single
  steps of discrete chains and with `reduce_memory` on fields of standard deviation 5; the
  propagator statistics of a chain that needs no steps print 0 %.
- `TestJunctionProduct`, `TestJunctionRange`: junction products keep the range of the whole product
  (overflowing partial products, powers up to 10⁶, complex factors); stars in fields of ±500 keep
  finite propagators, with full and reduced memory and one and four streams equal bit for bit.
- `TestLftsIdealLimit`: L-FTS without interactions against the analytic densities and H, with LM and
  Euler steps and checkpoints; no compressor is built in the ideal limit, so it runs once
  (2026-11-18).
- `TestMaskedDiscreteWeights`: a field changed only inside a binary mask leaves Q, the propagators
  and φ unchanged.
- `TestMatFieldNames`: MAT files keep monomer names that start with an underscore, long names and
  Unicode ion names, and runs with them restart exactly.
- `TestMinimalFormAlias`: the minimal-form alias search against a brute-force one on 44,000 modes of
  random oblique and triclinic cells, and near ties and exact ties of oblique cells from 35 to 145
  degrees that catch a wrong early-exit bound (2026-11-18).
- `TestNumpyScalarSerialization`: long-double NumPy scalars in metadata, SCFT output and
  checkpoints; L-FTS and CL-FTS restarts on every platform.
- `TestObliqueZStress`: the six stress components and lattice changes of ObliqueZ cells (P6/mmm,
  monoclinic P2/m, an odd z half-length) with and without the space group, CUDA against the CPU.
- `TestPackageHashes`: the source hash of the metadata from a copy of `utils/common.py` and of a
  simulation class outside the package.
- `TestPairRecords`: monomer names whose pair keys collide, `aux` in L-FTS and names with `,` are
  refused; records, WTMD keys and restarts keep the pairs apart.
- `TestPartitionGuard`: the solver reports a non-positive or non-finite Q, and SCFT and L-FTS
  reject it at accepted states.
- `TestPartitionIntegrand`, `TestPhysicalBasisOperators`: the partition diagnostic of
  grand-canonical species, and masked means, integrals and inner products in the physical and the
  irreducible basis.
- `TestPeriodicBoundaryCase`: boundary names in any case for the pressure, `etd` and Gibbs boxes.
- `TestPropagatorCodeStack`: chains of 2048 blocks under a 256 KiB stack, and the size of their
  architecture print.
- `TestRealSpaceAbsorbingRange`, `TestRealSpaceDiscrete`: `rs` against `ps` (Q, φ, stress; 1D to
  3D, walls, masks, complex fields, space groups), absorbing walls, a brush in a strong field, the
  stress against -d ln Q/dL, CUDA against the CPU.
- `TestRunRecord`: the `RunRecord` table of the measurements of a run.
- `TestScftAbsentPolymer`: SCFT accepts a grafted polymer of zero amount, whose Q is zero.
- `TestScftInitialFields`, `TestScftInitialPropagators`: SCFT checks `initial_fields` and `q_init`,
  also on reduced grids and with permuted axes, before it uses the solver.
- `TestScftObliqueSpaceGroups`, `TestScftResultGeometry`: monoclinic and triclinic SCFT (relaxed
  angles, the six stress components with the space group against the full grid); results and
  callbacks keep fixed and relaxed angles.
- `TestSimulationRelease`: SCFT, L-FTS and CL-FTS are freed by reference counting alone.
- `TestSmallFieldCoefficients`: small finite coefficients of H (large `zeta_n`, a nearly symmetric
  ternary χ) in H, its gradient and the uniform saddle point of L-FTS.
- `TestSmearingAliases`: exact minimal aliases of smearing and WTMD on anisotropic oblique grids;
  the CL-FTS gradient against central differences; the |k|² and alias moments of the smearing equal
  those of the solver bit for bit on every platform, and a cell within 1e-10 rad of right angles
  takes smearing with walls, the orthogonal WTMD and the orthogonal VTK grid (2026-11-17).
- `TestStressComplexWalls`: the stress of complex fields with reflecting and absorbing walls against
  mirrored periodic boxes and -d ln Q/dL, `ps` against `rs`, CUDA against the CPU.
- `TestStructureFunctionRecords`: recorded S_ij against a recomputation from the saved fields.
- `TestThinObliqueCell`: a short z axis leaves the in-plane diffusion of a hexagonal cell unchanged.
- `TestVtkFieldIntegrity`: VTK export refuses colliding field names without touching existing files.
- `TestWtmdLifecycle`, `TestWtmdRange`, `TestWtmdZeroModes`: the WTMD bias and statistics across
  runs, restarts and a bias set by hand; finite norms and gradients at large ℓ and extreme fields;
  zero Fourier coefficients; the forces at intermediate saddle-point fields carry the bias, and each
  sample holds Ψ and dH/dχN of the fields before its step (three monomer types; 2026-11-17); U′ equals
  dU/dΨ after bias updates (2026-11-18).
- `tests/scripts`: `TestConvertAuxiliaryFields`, `TestDensityPlot2D`, `TestIsodensityPlot`,
  `TestSlicePlot`, `TestStructureFunctionPlot` and `TestWtmdPlot` check the scripts and notebooks of
  `scripts/analysis`, among them the axes the density notebooks take from the Hall number of an
  SCFT result, against `SpaceGroup.uses_hexagonal_axes()` for all 530 settings and, without
  polymerfts, at the ends of the trigonal and hexagonal ranges (2026-11-17);
  `TestDriftTail` runs `drift_tail.py` on the one-pole model of Aarts, Seiler,
  Sexty and Stamatescu (2017), whose power-law tail makes Re<z²> wrong by more than 20 standard
  errors while the Schwinger-Dyson identities vanish; `TestMatReaders`, `TestRenormalization` and
  `TestValidationStandalone` check the MAT readers of the bundled tools, `renormalization.py` and
  the input validation without the compiled module.
- `tests/validation/validate_etd_order.py`, a script outside the test suite, measures the order of
  the time-step error of the integrators.

Tests that check more or differently than in 0.10.0:
- `TestCheckpointValidation`: a WTMD checkpoint that lacks one of the unscaled copies of the bias
  while it holds the others is refused.
- `TestComplexMask`: `zero_mean` with masks, read-only arrays and the lifetime of every object a
  binding keeps alive.
- `TestComputationSafety`, `TestComputationState`: the messages of the method checks, invalid
  inputs, flat cells, the getters before, between and after computations, the mixed-boundary rule
  of the factory, the box of a `cn-adi2` computation, which rejects non-orthogonal angles itself,
  oblique grid geometry and `n_grid` with a space group, on every platform and in both memory modes.
- `TestContourLengthMapping`: segment counts near a rounding boundary, local contour steps and the
  discrete lengths of molecule construction.
- `TestSimulationBox` (formerly `TestCpuSimulationBox` and `TestCudaSimulationBox`), `TestGridLimits`:
  grids whose number of points overflows 64 bits, on every platform; the angles of the axes a 1D or
  2D box lacks within 1e-10 rad of 90 degrees; the CUDA launch configuration (2048 threads per block,
  `LFTS_GPU_NUM_BLOCKS` and `LFTS_GPU_NUM_THREADS` values that are not integers).
- `TestCudaDCT`, `TestCudaDST`, `TestCudaFFTMixedBC`, `TestCudaTridiagonal`: the transform types 2
  and 3 only (2026-11-17), 1D DCT lengths at the limit of shared memory; odd reflecting axes against
  direct DCT and DST sums; a 6000-point line from global memory against shared memory.
- `TestCudaDeviceBuffer`: the owning cuFFT plan handle (creation, replacement, moves, a failed
  creation, reset, destruction) by the live plan count (2026-11-17).
- `TestExtensions`, `TestFtsReliability`, `TestLftsRun`: restarts through `exact_restart`;
  `TestLftsRun` requires the continuation bit for bit, where it allowed 1e-8; a stateful custom
  integrator with a linear response; the architecture figure in the recording directory; options
  given as None. `TestFtsReliability` and `TestClftsElectrostatics` check that the numerical
  failures raise `NumericalError` (partition consistency, the Hamiltonian and force at an
  intermediate L-FTS saddle point, intermediate CL-FTS fields and forces with and without ψ, the
  concentrations and ψ after a CL-FTS step) and refuse a malformed previous noise of a continued
  run and an `initial_psi` of strings or a ragged list (2026-11-17). `TestFtsReliability`: a saddle
  search that ends with non-finite fields stops the run as `aborted_saddle_failure`, with no retry
  from those fields, and every run of an object checks the partition consistency; `TestLftsRun`:
  the CL-FTS forces carry the dynamical stabilization (2026-11-17).
- `TestLftsDetails`: the restored WTMD bias, the order parameter on the half spectrum and the hills;
  `saddle_failures`; LR and LRAM with reflecting walls, J(k) of CUDA against the CPU and J(0); the
  permissions of result files.
- `TestConfigFile`, `TestParameterNames`, `TestParams`: graft maps through JSON, YAML and MAT files;
  the angle checks of L-FTS and CL-FTS, SCFT options that cannot converge, malformed space groups,
  `relax_cell` with `cn-adi2` and WTMD inputs; `dpsi`, smearing with a space group, `max_iter` and
  `crystal_system` refused by name, the 1e-10 rad limits of the angles a crystal system fixes and of
  the 2D classification, the top-level keys of another method ignored with one warning at the
  caller's line, without checking their sections, and those of no method (a misspelled key) refused,
  also given through `from_file` or a legacy name, `axis_ordering` refused by L-FTS and CL-FTS, and
  top-level keys given as None, also in a shared YAML file and as a `from_file` override
  (2026-11-17).
- `TestPartitionFunctionCheck`, `TestPhaseEquilibrium`, `TestPolymerFieldTheory`: the partition
  error against the absolute integrand; boundary names in any case and the chemical potentials of
  full-grid fields with a space group; `compute_h_deriv_chin` without `chi_n`, its call of 0.10.0,
  and dictionary fields of `to_aux_fields`.
- `TestPropagatorCode`, `TestPropagatorCodeReduction`, `TestScheduler`: real repeat counts in the
  keys; the schedules of one to four streams.
- `TestSaveVtk`, `TestScftCallback`, `TestScftOptions`, `TestScftResult`: `save_vtk` with
  `space_group`; a NumPy False from `on_iteration`; `polymer_figure_dir` and tetragonal cells; None
  options, paths and strict JSON in results; a failed write of a MAT, JSON or YAML result leaves the
  existing file as it was, and the fields of an SCFT `NumericalError` with a space group lie on the
  full grid, from the partition check of an iteration and at the end of the run, a non-finite
  residual and a non-finite error level (2026-11-17).
- `TestSmearingLattice`: smearing through `PropagatorSolver.fourier` against NumPy; the smearing
  |k|² equals that of the solver in every step of an SCFT box relaxation (oblique, hexagonal and a
  fixed angle of 105 degrees; 2026-11-17).
- `TestSpaceGroupPlatformEquivalence`: a monoclinic P2/m cell (γ = 100°, odd n_x), masks with RK2
  and CN-ADI and the irreducible basis of oblique cells.
- `TestStructureFunctionRPA`: CL-FTS with `etd` at δτN = 16.
- `tests/scripts`: `TestPscfScript`, `TestSimulationConstraints`, `TestSimulationModel`,
  `TestSimulationSpaceGroup` and `TestSimulationWeb` check the new crystal systems of the editor,
  its export checks and token, Windows paths and links, and that the PSCF script tool leaves the
  working directory alone; the suggested length of a discrete block at half a segment, and through
  `check_simulation_cells.cjs` the names A to Z, AA, AB and so on of added monomer types
  (2026-11-17). The editor and PSCF script tests fail on the warning about the options of another
  method, and validate SCFT exports and the inputs of imported scripts with `validate_scft_params`
  also without a backend (2026-11-17).
- `TestSpaceGroupCell`: `SpaceGroup.fixed_angles()` of each crystal system; the cell check of a
  monoclinic group with the unique axis c and of a triclinic group (2026-11-17).
- `TestSpaceGroupBasisSelection`: the physical basis that SCFT chooses for Im-3m, Pmmm (also on a
  grid with an odd nx), P4/m, P6/mmm, P2/m with the unique axes b and c and P-1, and one SCFT
  iteration with the m3, Pmmm, irreducible and z-mirror bases against the full grid from random
  fields of standard deviation 5 (2026-11-17).
- `tests/scripts/TestPscfConverter`: the PSCF export refuses a `.py` path, CL-FTS fields files and
  parameters, a `gibbs_ensemble` section and an explicit `zeta_n`, and still exports SCFT results,
  L-FTS fields files and L-FTS parameters; the import names a 27th field AA (2026-11-17).
- `TestCpuFFTMixedBC` is removed: `TestCpuMakhoulTransform` covers its round trips and DCT and DST
  checks.
- `TestStress1D`, `TestStress1DComplex`, `TestStressLinear2D`, `TestStressLinear2DComplex`,
  `TestStressLinear3D` and `TestStressDiscrete1D` and their ten input files (1.4 MB) are replaced by
  `TestStressFiniteDifference` (2026-11-18), with the user's approval of the changed conditions: each
  derivative of the free energy by a length or angle against its central difference at fixed smooth
  fields, without an SCFT loop, in 1D, 2D and 3D (orthogonal, oblique and triclinic cells), for real
  and complex fields, `rqm4`, `rk2` (3D `rk2` was not checked before) and `ps`, and branched chains
  in every aggregation and memory mode, on every platform (several old tests ran on cpu-fftw only);
  14 s instead of 187 s.
- Every test carries a CTest label, `basic`, `numerical` or `examples`, and configuration stops for a
  test without one (2026-11-18): `ctest -L numerical` now includes the stress, space-group phase,
  SCFT, smearing and phase-equilibrium tests that had no label.
- `TestSpaceGroupPhases` requires the reduced-memory free energies to equal the full-memory ones bit
  for bit (2026-11-18).
- Changed 2026-11-19, with the user's approval of the conditions: `TestStressFiniteDifference` has
  a row of two different diblocks, and each solver is built at another cell and moved to the
  checked one, so per-polymer bugs and stale stress tables fail it; `TestAggregationDiscreteCpu`
  of 0.10.0, which ran at zero fields where Q is 1 in every mode, is removed
  (`TestPseudoBranchedDiscrete3D` covers the same modes with fields); the cn-adi2 case of
  `TestSpaceGroupPlatformEquivalence` uses blocks 0.35/0.65 (two contour steps); `TestScftOptions`
  runs the 3D `axis_ordering` case for 10 iterations (17.6 → 6.7 s) and checks that a given
  `stress_interval` is ignored with one warning. `TestCrankNicolson1D/2D/3D` check stored
  reference values and the reflecting/absorbing case of `TestCompareRealSpacePseudoSpectral` the
  mirrored box, so a build without a GPU tests the CPU CN-ADI solver; `TestScft1D` runs its CUDA
  part on four streams. `TestExamples` runs the examples on the platform they select (it ran the
  "auto" Langevin examples on a CPU platform), and `TestWtmdReweighting` runs every cell of the
  notebook. One `cpu_platform()` (tests/cuda_part.py) replaces 22 copies. `TestCudaFFT`,
  `TestCudaCubReduceSum`, `TestCudaCubReduceSumClass` and `TestCudaPinnedMemory`, which tested
  cuFFT, CUB and cudaMemcpy only, are removed.
- `TestClftsElectrostatics` checks the fluctuation of Im ψ (2026-11-18): Re ψ of the decoupled
  neutral case is round-off, exactly zero with MKL, so the test failed in a build with MKL alone.
- Tests added 2026-11-18: `TestException` and `TestErrorMessages` (an error passed through the
  wrappers keeps one location and its type); `TestScftElectrostatics` (2D oblique electrostatic
  stress against finite differences), `TestScftOptions` (the 3D `axis_ordering` angle
  permutation), `TestSmearingLattice` (the angle derivative at the Nyquist planes on the half
  spectrum), `TestConfigFile` (UTF-8 under an ASCII locale), `TestGibbsEnsemble` (continuations
  that change the keys of the cycles, histories of the old hash) and `TestCudaDeviceBuffer` (plans
  that share a work area). The RPA tests run at `verbose_level` 0 and the test helpers draw no
  architectures (about 60 s less in the suite).
- `TestCrysFFTBench`, a benchmark outside ctest, times its full-FFT baseline as the solvers run it;
  docs/internals/FFTImplementation.md gives the gains measured this way (CUDA recursive 3m 2.3x,
  Pmmm 1.5x, where the former baseline gave 3.1x and 2.0x).

Test helpers: `tests/fts_checks.py` holds what the Python tests of simulation runs share: `quiet`
and `silenced`, which discard printed output; `cpu_platform` (of tests/cuda_part.py), the one
platform that runs a test of a fix made in Python (the first CPU platform, else CUDA);
`compare_files`, which compares the key sets of two MAT files both ways and their values bit for bit
or to a relative tolerance; `exact_restart`, which checks a run continued from the checkpoint of a
shorter one against the uninterrupted run (random state, partition check and recorded files); and
the inputs that several tests run. With `tests/cuda_part.py` and `tests/TestCudaPart.h`, a test that
runs on every platform fails in a CUDA build without a visible device instead of passing without its
CUDA part. CTest runs the Python tests with `PYTHONDONTWRITEBYTECODE=1`, so that they write no
`tests/__pycache__`. CI runs `tests/scripts` a second time after each build, with the built package,
so that the script tests that need polymerfts run too (2026-11-17); on Linux it sets
`POLYMERFTS_SCRIPT_TEST_PLATFORM=cpu-mkl`, so that the imported PSCF scripts of
`TestPscfScriptIntegration`, which were skipped there, run as well.

## [0.10.0] - 2026-10-23

The first release since v0.9.1 (May 2025). The
[release notes](.github/release-notes/v0.10.0.md) summarize changes relative to that
release. The dated development history below also records features introduced,
corrected or removed between releases; those entries do not all describe v0.9.1.

### Release highlights

- CPU-MKL, CPU-FFTW and CUDA backends share the propagator computation layer;
  continuous and discrete chains support space-group reduced bases and CrysFFT paths.
- Canonical SCFT exposes full and relative free energies, molecular chemical potentials,
  grand potentials and reference-species phase-coexistence calculations.
- L-FTS and CL-FTS support custom integrators and named observable callbacks, with
  checkpointed extension state. CL-FTS and the browser script builder remain beta.
- Numerical corrections cover structure functions, compressible SCFT, non-orthogonal
  stress and smearing, propagator aggregation, memory ownership and checkpoint restart.
- User tools include a browser simulation script builder and bidirectional PSCF
  conversion; see [Simulation GUI](docs/getting-started/SimulationGUI.md) and
  [PSCF conversion](docs/getting-started/PscfConversion.md).

### Installation and input migration

- Build and install Python wheels with `python -m pip wheel` and `python -m pip install`;
  remove them with `python -m pip uninstall polymerfts`. Direct CMake installation is no
  longer the documented Python installation path. See
  [Installation](docs/getting-started/Installation.md).
- The user scripts moved from `tools/` to `scripts/`: the MATLAB scripts and notebooks of the
  analysis to `scripts/analysis/` (`tools/README.md` became `POSTPROCESSING_REFERENCES.md` there),
  and the PSCF sample fields to `tests/data/pscf/`. The exporter
  `tools/conversion_to_pscf/to_pscf.py --file_name <script>`, which read a polymerfts input script,
  and the importer `tools/read_pscf_output/to_matlab.py` are replaced by `export` and `import` of
  `scripts/pscf/pscf_converter.py`: `export` reads a saved SCFT result, or a parameter file with
  `--fields`, and `import-script` also translates supported PSCF calculations.
- Unvalidated Docker image definitions have been removed.
- The supplied environment uses Anaconda `defaults` and Python 3.14. MKL and FFTW are
  installed separately and enabled explicitly; CUDA detection defaults to `AUTO`.
  At least one backend is required. Python 3.11+, NumPy 2+, C++20, spglib and LAPACKE
  are required; CUDA builds require toolkit 12.2+ and CMake 3.25.2+.
- Public parameter names are now `polymers`, `relax_cell`, `stress_scale`,
  `blocks[].monomer_fractions`, `space_group.hall_number` and
  `recording.checkpoint_interval`, `sample_interval`, `statistics_save_interval`.
  The previous names remain accepted at the input boundary; supplying both names for
  one option is an error. See [Parameters](docs/references/Parameters.md).
- Replace v0.9.1's `reduce_gpu_memory_usage` with `reduce_memory` explicitly; it is
  not an alias. Memory saving now uses contour checkpoints and recomputation.
- `numerical_method` selects the propagator method;
  `langevin.integrator` selects `lm` (the default) or `euler`. Custom methods and
  callbacks use the contracts in [Extensions](docs/internals/Extensions.md).
- Simulation metadata records source/build information, backend, numerical method,
  input hashes and execution status. Version-2 checkpoints retain custom integrator
  and callback state. v0.9.1 checkpoints are unversioned and require
  `continue_run(path, allow_legacy=True)`; settings and pending measurements cannot
  be verified or fully restored. Version-1 checkpoints from development builds
  support default LM without extensions. v0.9.1 does not restore the complete state
  in version-2 checkpoints; resume them with v0.10.0.

### Detailed numerical and development history

- Stress for non-orthogonal crystal systems (2026-01-18, 9c93e0c1, 68293424), and the
  CrysFFT stress path units and axis components (2026-07-30, a6baa345).
- Angle update sign in SCFT box relaxation was a gradient ascent (2026-07-30, 0f242575);
  runs started at exactly 90-degree angles no longer sit in a dead zone (2026-07-31, d51db6e9).
- Incompressible SCFT mean normalization pins the P-projected mean to the saddle point
  (2026-02-23, 7218692e); the normalization became optional (`fts_consistent_energy`,
  2026-03-22, ce2f3cdf). Compressible L-FTS: jk baseline subtraction and mean(W+) pinning
  (2026-02-14, 926fed2b).
- Anderson mixing solves the history system with SVD (LAPACKE) instead of Gaussian
  elimination (2026-02-12, 68995d48, dfdcde2d). LAPACKE is now a build requirement.
- `dt_scaling` excludes the pressure-field eigenvalue (2026-02-11, 604dfad1).
- Concentration normalization uses `n_segment_total` (2026-01-19, e4b09a3f); non-monotonic
  convergence and the reduce-memory stress computation were fixed (d09d3635).
- Space groups: coordinate-system bug and reduced-basis SCFT (2026-01-21, ad6b94c7),
  tetragonal groups (3ed69773), physical-basis handling with the cell-centered symmetry
  convention (2026-01-31, b467d607), hexagonal/trigonal groups via a hybrid orbit convention
  and a group-invariant k representative (2026-08-01, 6006bb8e), Recursive3m fold for
  odd glide translations (f047b583), discrete chains with a space group read `exp_dw` out of
  bounds (2026-07-30, 5493d009), MKL Pmmm CrysFFT DCT normalization (2026-07-31, d118514c).
- Non-periodic boundary conditions: MKL DST-III dropped the highest mode (2026-07-01,
  0225dd3c), CUDA discrete-chain DCT/DST propagator (cc3e48a4), CUDA glue kernels for
  grids above 65536 points (619028e2), wrong per-axis tables for mixed non-periodic BCs on
  CUDA (2026-08-01, a0d02557), complex-field propagators with non-periodic BCs on all
  platforms (2026-08-18, c0ab6e0c).
- Phase test fixtures (`tests/data/*.mat`) were re-converged under the current cell-centered
  symmetry convention and the expected free energies replaced by converged values
  (2026-10-19, 9e69f9e2). The example `DG.mat`/`SG.mat` were re-converged library outputs
  since 2026-01-21.
- CUDA RK2 stress (2026-10-19): complex fields gathered q(-k) in place through the k -> -k
  map (a data race on grids larger than one kernel launch), and non-periodic 1D/2D boxes
  read the wrong basis tables (V11 = 0). RQM4 and the discrete chain were correct.
- CUDA DCT-II of odd length (2026-10-20): the pre-op kernels applied the Nyquist branch to
  the slot N/2 of odd sequences, so the CUDA CrysFFT (Pmmm and z-mirror bases) was wrong
  whenever a logical grid size is 2 mod 4 (the physical grid is half the logical one), e.g.
  Pmmm 12x12x10 or P6/mmm 24x24x30. Sizes divisible by 4, the CPU platforms and the
  irreducible basis were correct. The 1D DCT-II also packed its spectrum in place into the
  caller's N-double buffer, writing two doubles past its end (1D CUDA runs with reflecting
  boundaries); it now packs into its own scratch buffer. The 1D DCT-III wrote its N/2 + 1
  complex outputs into an N-double work buffer, one or two doubles past its end; the
  buffer now holds them.
- SCFT runs with an mmm orthorhombic group and the m3 basis (Pmmm, Pbca, Pnma, ... whenever
  `m3_crysfft_feasible()` holds: even nx, nz >= 16 and a compatible translation parity,
  e.g. Pmmm 10x10x18 or Pnma 12x12x16) saved scrambled full-grid fields: the
  irreducible-to-reduced conversion was skipped because the two bases have the same size
  there, although their orderings differ (2026-10-20). The iteration itself was correct.
- CPU solvers with a space group but without a CrysFFT physical basis (for example a
  hexagonal cell with the irreducible basis) multiplied the reduced-basis mask over the
  full grid, reading past its end (NaN); the mask is now applied in the basis of the
  propagator after the reduction, as on CUDA (2026-10-20).
- Continuing an L-FTS run from a checkpoint restored the Langevin noise array with a
  squeezed shape when there is a single real field (every AB system), so the continued
  trajectory differed from an uninterrupted run; the continuation also no longer
  re-solves the saddle point of the loaded fields (2026-10-19). CL-FTS, whose noise array
  has one row per field, reshapes it the same way for safety.
- SCFT box relaxation with a space group reduces the stress of lengths tied by the crystal
  system (Cubic, Tetragonal, Hexagonal, Trigonal) by averaging the tied components, as runs without
  a space group already did; the components agree to round-off for symmetrized fields, so
  converged results are unchanged but such runs are not byte-identical (2026-10-19).
- `fts_consistent_energy`: the mean concentrations now include random-copolymer blocks and
  the chain length (f_p / alpha_p); before, random copolymers raised and blends of
  unequal chain lengths pinned the wrong mean (2026-10-19).
- Continuous-chain stress with full propagator storage dropped the chain-end segments of
  every block on the assumption that q(r, 0) = 1 carries no stress. That holds only for
  free ends with periodic or reflecting boundaries; with absorbing walls, a mask or a
  grafting point the stress lost an O(ds) term (absorbing walls: -2.7e-2 relative error
  against a finite difference of ln Q at ds = 0.05, -6.7e-3 at ds = 0.0125; now 9.6e-5 and
  6.3e-7). The reduce-memory mode always kept these terms; both modes now agree
  (2026-10-20).
- A failed construction freed nothing it had allocated, at any level: propagator
  computations, solvers, `Pseudo`, the FFT, DCT/DST and CrysFFT transforms, and the CUDA
  computation box. When a full-storage CUDA run did not fit in GPU memory, the process kept
  every buffer allocated up to the failure (22.5 GB on an A10 for a 128^3 grid), so a retry
  with `reduce_memory` in the same process failed as well. Every constructor now releases
  what it holds when it throws; a sweep that runs out of GPU memory at 210 different points
  (box, transforms, solvers, backend, propagators) leaves at most 4 MiB behind, against up
  to 1.1 GB before. Errors thrown inside the parallel loops of the full-storage mode (out
  of memory while building a CrysFFT cache, a grafted polymer without its `q_init`)
  terminated the process; they now reach Python as exceptions (2026-10-20). The 3D DCT-1
  and DST-1 transforms on CUDA did not check that their cuFFT plan was created. The CUDA
  Anderson-mixing constructors and their history buffers now release their memory too
  (they kept up to 1.4 GB after a failed construction for a 2x128^3 field).
- Reduce-memory mode now gives results identical to full storage (it used separately
  written code): the CPU continuous solvent partition function was computed from
  exp(-w ds/2)^2 with uniform cell weights, ignoring a mask; the CUDA discrete variant
  normalized each block with the polymer found by a key search that ignored the polymer
  index, so two polymer types sharing a propagator pair could be normalized with the
  wrong volume fraction (2026-10-20).

- A masked computation box given a second space group, or none after one, kept the mask in
  the first group's reduced basis, so the next computation read past it (NaN or wrong
  partition functions); changing the box size of a masked box with a space group did the
  same, and the accessible volume was not updated when the box changed (2026-10-21).
  `set_space_group` now reduces the full-grid mask each time and commits only after every
  step succeeded. A computation created without a space group on a box that still held one
  wrote past its buffers (the backend took the reduced size, the solver worked on the full
  grid); a computation now sets the box's group, also to none, and refuses to run if a
  later computation gave the box another one. A group built for another grid is rejected.
- Smearing and non-orthogonal cells (2026-10-21). The kernel Γ(k) took
  k² = kx² + ky² + kz² from the box lengths alone, ignoring the cell angles (in a hexagonal
  cell the (1,-1) mode got twice the k² of (1,0)); it now uses the cell metric with the
  minimal alias, as the propagator solvers do. The L-FTS and CL-FTS noise variance used the
  product of the box lengths instead of the cell volume, so runs in hexagonal, oblique,
  monoclinic or triclinic cells sampled an effective N̄ larger by 1/(volume factor)²
  (1.33 for γ = 120°; σ was 0.93 times too small). Orthogonal cells are unchanged.
- SCFT box relaxation with smearing (2026-10-21): the kernel was built once for the initial
  box, so the relaxed model had the smearing length a_int L/L0, and the stress lacked the
  term from Γ(k(L)). The relaxed period depended on the starting box (a_int = 0.05, χN = 20:
  1.6235 from L0 = 1.7, 1.6084 from L0 = 1.3); the kernel now follows the box and the
  stress includes Σ_i <φ_i (∂Γ/∂θ * w_i)>, and both starts relax to the minimum of F(L),
  1.7138.
- **L-FTS and CL-FTS structure functions were too small by the number of monomer types M**
  (one half for AB, one third for ABC) from 2024-09-01 (042cca91; releases 0.9.0 and 0.9.1)
  to 2026-10-21. The estimator μ_i = Σ_k A⁻¹_ki Ω_k/λ_k was written for the earlier
  auxiliary-field matrix (orthonormal eigenvectors), where it was right; with the symmetric
  theory A = √M O and A⁻¹ = Oᵀ/M, and the estimator needs A_ik. For a disordered AB diblock
  (χN = 6, N̄ = 10⁶) the recorded S(k) was 0.50 of RPA and of the estimator
  n<|W₋(k)|²>/(χN)² - 1/(2χN) used by other codes (lfts_cpu, clfts_gpu); now it agrees with
  RPA (tests/TestStructureFunctionRPA.py). Recorded L-FTS files can be corrected by the
  factor M (with smearing they also carry Γ(k), see below). CL-FTS had three more errors,
  so its files cannot: it multiplied μ(k) by the conjugate of φ(k) instead of φ(-k), which
  differ for complex fields; it paired the
  concentrations and partition functions of one step with the fields of the next in H,
  dH/dχN, S(k) and the saved φ; and with a compressible model it left out the fluctuating
  field of the largest eigenvalue. Compressible AB (ζN = 100, dt = 1): S_AA was 0.43 of RPA,
  now 1.006. The CL-FTS fields are unchanged (to round-off, about 10⁻¹⁴ after 300 steps);
  the recorded H now pairs each field with its own partition function.
- **Compressible SCFT (`zeta_n`) solved the incompressible equations** (2026-10-21): the
  residual χφ - Pw was used for both models, and ζ entered nowhere, so every compressible
  run returned the incompressible structure (Σφ = 1 everywhere) and its free energy was the
  compressible Hamiltonian away from its saddle point (1D AB lamella, χN = 15: residual
  9.68, F = -0.364 for ζN = 5 and -0.322 for ζN = 100; with `fts_consistent_energy`
  residual 2.18, F = 1.886 and -0.060). The residual is now the compressible one,
  E = χφ + ζ(Σφ - 1) - w, with its component along (1, ..., 1), where the stiff ζ term lies,
  scaled by κ/(κ + ζ) (see the field residual entry below); the field means, which have
  no gauge freedom in the compressible model, are pinned to the saddle point. ζN = 5 now
  gives 0.90 <= Σφ <= 1.07 for that lamella, the free energy equals the mean-field value
  computed from the fields, and F and φ approach the incompressible solution as 1/ζ
  (tests/TestScftCompressible.py). L-FTS and CL-FTS were not affected.
- Multi-monomer Hamiltonian for three or more monomer types (2026-10-21): the vector S of
  the linear coefficients S/λ and of the constant was computed from the eigenvectors before
  their sign normalization and √M scaling, so for unequal χ the field part of H was off by a
  term linear in the composition (ABC, χN = 9.5/16.625/9.5, φ = (0.5, 0.3, 0.2): 3.624
  instead of 3.658; compressible ABC: off by about 200). Structures, φ and phase boundaries
  were unchanged (only uniform field modes shift), but absolute H, chemical potentials and
  `fts_consistent_energy` free energies were wrong. AB systems were correct.
- dH/dχN for three or more monomer types (2026-10-21): it differentiated the coefficients at
  fixed auxiliary fields but missed the change of the polymer part through A(χ) (the
  eigenvectors rotate with χ), and blew up when an eigenvector changed sign between χ ± ε
  (χN = 10/30/20: ±4·10⁴ instead of about 0.1), and it left out fields dropped at a zero
  eigenvalue (AB at χN = 0, f = 0.3: 0.25 instead of f(1 - f) = 0.21; B and C chemically
  identical: 10-20 %). It is now the derivative at fixed auxiliary fields Ω including the
  rotation term <φ · (∂A/∂χ) Ω>, computed analytically from the first-order perturbation of
  the eigenvectors in a frame that moves only between real, imaginary and dropped fields
  (docs/theory/PolymerFieldTheory.md, Section 4.5), so nearly equal eigenvalues need no
  care; this is the exact dF/dχN of the L-FTS ensemble, where the imaginary fields sit at
  their saddle point (the derivative at fixed monomer fields would differ from it by
  O(1/C)). Dropped fields add their mean-field term. Checked by dH/dχ_ab = <φ_a φ_b> at the
  field saddle for 2, 3 and 4 monomer types (with nearly equal eigenvalues and dropped
  fields) and by a zero-dimensional three-component model of the L-FTS approximation
  (tests/TestPolymerFieldTheory.py). AB systems at χN ≠ 0 are unchanged.
- Multi-monomer field theory with a composition eigenvalue next to the pressure's zero
  (2026-10-22): for 10⁻¹² < |λ| < about 10⁻⁸ (e.g. χ_AB N = χ_AC N = 15, χ_BC N = 10⁻⁸)
  `SymmetricPolymerTheory` failed with "Invalid inverse of matrix A". eigh mixed a part of
  the uniform pressure vector into that eigenvector (about ε|χN|/|λ|, 1.8·10⁻⁷ there), and
  the zero test of the eigenvalues used 10⁻¹² while the roles of the fields used 10⁻⁸. The
  eigenvectors are now made orthogonal to the pressure vector when they hold such a part
  (matrices without such a part are unchanged bit for bit), and one tolerance, |λ| <= 10⁻⁸,
  decides which fields
  are dropped (tests/TestPolymerFieldTheory.py).
- Smearing with reflecting or absorbing walls (2026-10-21) convolved with a periodic FFT,
  coupling the two walls (a ramp in a reflecting box: 0.89 in the first cell instead of
  0.08). Wall axes are now smeared in the DCT-II basis, i.e. with the even image of the
  field across the wall; across an absorbing wall the propagators are odd, so fields and
  concentrations are even there too. An axis with different walls on its two sides is
  accepted (CN-ADI).
- Stress at alias ties (2026-10-21): the stress table v⊗v took the first of the alias
  representatives that tie for the smallest |k|², on the Brillouin-zone boundary of
  oblique cells and on the Nyquist planes. For a hexagonal field dH/da and dH/db differed
  (1.7·10⁻⁶ relative for a random P6₃/mmc field on 18×18×16), and the forced angle derivatives of an
  orthogonal cell missed the central difference. The table is now averaged over the tied
  representatives (tests/TestStressAliasTies.py). SCFT box relaxation was not affected (it
  averages tied lengths, and the a = b derivative was exact); orthogonal cells without
  angle derivatives are bit-identical.
- Structure functions with smearing (2026-10-21): the fields couple to the smeared densities,
  so the estimator gives Γ(k)·S(k); in a periodic box the recorded S(k) is now divided by
  Γ(k) and is that of the densities (tests/TestStructureFunctionRPA.py compares smeared
  L-FTS and CL-FTS runs with the smeared RPA). With walls Γ is defined in the cosine basis,
  and Γ·S is recorded as it is.
- A solvent in a masked box (2026-10-21) had a nonzero concentration inside the obstacles
  (0.23 in tests/TestPropagatorKeys.py); it is now zero there and its volume average is its
  volume fraction. With a space group the reduced-basis integrals (volume averages, inner
  products, the solvent partition function) had no mask weights (Q_solvent of a masked
  Im-3m box: 0.98946 with the space group, 0.96279 without); they now weight each orbit by
  its count times the mask, and agree with the full grid to 1e-14.
- WTMD (2026-10-21): the volume and |k| of the order parameter ignored the cell angles; a
  continued run lost I1 and the samples since the last update (dH/dψ = I1/I0 was
  underestimated after a restart; checkpoints now carry them); and ψ, computed from the
  fields before a Langevin step, was paired with dH/dχN of the fields after it.
- L-FTS steps whose saddle point failed were skipped (2026-10-21): the step counter moved
  on, so a recording step lost its files and the next recording period held the samples
  of two. The step is now repeated from the same fields with new noise (up to five times);
  runs without failures are unchanged.
- Compressible L-FTS (2026-10-21): the uniform mode of the pressure-like field was pinned to
  the value the first saddle-point solve left (the convergence test, a standard deviation,
  ignores this mode), which shifted H and dH/dχN. Every imaginary field except the
  incompressible pressure field now has its uniform mode at the analytic saddle value
  -(c1 + Σ_j A_jk <φ_j>)/(2 c2), set in every saddle iteration so that the recorded H
  belongs to the returned fields.
- SCFT box relaxation with free angles (2026-10-21): the angles entered the optimizer in
  degrees while their residual is per radian, making the angle steps 57 times smaller
  (Oblique2D hexagonal cylinders from γ = 100°: 1493 iterations with cpu-mkl and 1544 with
  cpu-fftw, now 241; the converged cell is unchanged).
- SCFT `axis_ordering` (2026-10-21) reordered nx, lx and angles but not the initial fields
  and the boundary conditions, which were then read in the wrong order (dF = 9.4·10⁻² for
  the fields and 9.0·10⁻² for walls on the wrong axis in tests/TestScftOptions.py). The
  initial fields, `q_init` and `bc` are given in the order of nx and reordered; the results
  (fields, concentrations, saved files) are in the standard order [a, b, c].
- SCFT field residual (2026-10-21): it projected with P = I - e eᵀ(χN)⁻¹/c,
  c = 1ᵀ(χN)⁻¹1. Its pressure part, (1, ..., 1)Σφ/c, pushed the pressure along the excess
  density for c < 0 (AB with χN < 0; ABC with χ_AB N = χ_BC N = 10, χ_AC N = 60, or with
  χ_AB N = χ_AC N = 10, χ_BC N = 50), grew without bound near c = 0 for either sign (the
  latter with χ_BC N = 39.9 and 40.1, c = +5·10⁻⁴ and -5·10⁻⁴), and in compressible runs
  it was amplified at small ζN (AB f = 0.3, χN = 20, ζN = 0.5); these runs ended in NaN
  or did not converge. The residual is now P⊥(χφ - w) + κ e(Σφ - 1) with the orthogonal
  projector P⊥ = I - eeᵀ/M and κ = max(max|λ|, 1)/2 (λ: the eigenvalues of P⊥ χN P⊥), and
  for compressible runs P⊥E + κ/(κ + ζN) (I - P⊥)E (docs/theory/SelfConsistentFieldTheory.md,
  Section 2.2). It needs no (χN)⁻¹ and the fixed points are unchanged. For incompressible
  AB runs with χN >= 1 it differs from the old residual by a constant that the subtracted
  means remove (κ = χN/2 = 1/c), so the iteration is the same up to round-off. In a sweep
  of 70 runs (1D and 2D, AB and ABC, incompressible and ζN = 0.5 to 100) all converge,
  against 18 before; tests/TestScftOptions.py checks 18 of them. A negative `zeta_n` is
  rejected.
- Aggregation with blocks of different local ds (2026-10-21) failed with `map::at`: the
  keys that contain an aggregated propagator took the ds index of the outer block for
  every dependency. Grafting labels with digits ("G1") failed the same way (the digits were
  read as segment numbers); labels may now contain letters, digits and underscores only.
- Aggregation counted some blocks twice (2026-10-22, ef957694). When identical side chains
  of two or more blocks are attached at different positions of a polymer (for example on
  backbone blocks of different lengths), the right key rewritten for the blocks next to the
  aggregated side chains can itself be merged at its own left key; it then still rewrote the
  blocks beyond it, and so did the merged key, so these blocks stayed under two right keys
  and their concentrations were counted twice (Σφ ≠ 1, `check_total_partition()` false; Σφ =
  1.12 for a bottlebrush with B2-A1 side chains on backbone blocks of 1, 2, 3 and 1 segments
  in a random field). Continuous chains were affected since 59ca6db2 (2024-06, so v0.9.1
  too) and, from 0dea94bf (2026-10-21) on, failed with "Contour length 0.000000 not found"
  instead; discrete chains since b9e0d487 (2025-12). Runs with
  `aggregate_propagator_computation` set to False were correct.
- `set_lattice_parameters` with angles that form no cell (2026-10-21) raised the error after
  it had stored the new lengths and angles, so a caught error left the box inconsistent;
  the cell is now checked before anything changes.
- The CUDA box's device `zero_mean` added the sum to real fields instead of subtracting the
  mean, and the device multi-component inner product and zero mean ignored a space group
  (2026-10-21; no simulation path calls them).
- A masked CUDA box given a space group built for another grid read the mask out of bounds
  before the grid mismatch was reported (2026-10-21): the reduced mask weights were built
  before the group was checked. The checks now run first.

### Interface changes

- Reflecting or absorbing boundaries in a non-orthogonal cell, and a space group with
  reflecting or absorbing boundaries, are rejected (2026-10-21). Both were accepted and
  gave silently wrong results: the wall transforms have no metric cross terms (the
  pseudo-spectral path computed an orthogonal box, CN-ADI already refused), and walls do
  not commute with the group's operations (Ia-3d, 16³, reflecting walls: Q off by less
  than 1 % to tens of percent depending on the field;
  groups whose mirror planes lie on the walls, such as Pm-3m, happened to be exact).
  `Smearing` takes the cell `angles` and has `set_box()` and `stress()`.
- `SymmetricPolymerTheory.compute_h_deriv_chin(chi_n, w_aux, phi)` takes the concentrations
  (2026-10-21): the rotation term needs them whenever the eigenvectors depend on χ (three or
  more monomer types; a call without `phi` is then rejected), and `chi_n` must be the one
  the object was built with. `h_deriv_chin_from_moments()` takes the averages instead. The
  coefficient derivatives `h_const_deriv_chin`, `h_coef_mu1_deriv_chin` and
  `h_coef_mu2_deriv_chin` are removed.
- `SymmetricPolymerTheory.compute_hamiltonian()` takes the computation box (2026-01-26,
  65309e1f, for the reduced-basis integrals of a space group); the tutorials that called it
  without the box are updated (2026-10-21).
- Grafting labels (the values of `initial_conditions` and `grafting_points`, the keys of
  `q_init`) may contain
  letters, digits and underscores only (2026-10-21); other characters are rejected.
- A mask without the symmetry of the space group is rejected (2026-10-21): the reduced basis
  keeps one value per orbit, so such a mask was silently replaced (an off-center obstacle
  in an Im-3m box disappeared, Q off by 4.5 %).
- SCFT accepts a singular χN matrix (2026-10-22): the field residual no longer uses
  (χN)⁻¹, so SCFT no longer rejects every singular χN, which included any zero pair in a
  three-monomer system (det χN = 2 χ_AB χ_AC χ_BC) and chemically identical monomers, nor
  1ᵀ(χN)⁻¹1 = 0. There the auxiliary field of a zero eigenvalue of the projected χN has a
  linear term and acts as a uniform field: free energy and box are the limit of the
  neighbors (χ_AB N = χ_AC N = 10, χ_BC N = 40: the mean of χ_BC N = 40 -+ 10⁻⁴ to 10⁻¹¹).
  A single monomer type is still rejected in the
  incompressible model. An ABC triblock whose B and C are identical gives the free energy of
  the AB diblock to 10⁻¹³, and a zero pair (χ_AC N = 0) the limit χ_AC N → 0
  (tests/TestScftOptions.py).
- SCFT exposes the full canonical Helmholtz free energy as `free_energy_absolute` (2026-10-22),
  F = Σ_p (φ_p/α_p)[ln(φ_p/(α_p Q_p)) - 1] + <½ φᵀ χN φ - w·φ>
  (+ the compressibility penalty). `free_energy_homogeneous` is the full homogeneous
  Flory-Huggins value including ideal mixing entropy. `free_energy`, iteration logs
  and progress callbacks use F minus this homogeneous value; the final summary prints
  both relative and full values. A disordered AB melt with alpha = 1, f = 0.3 and chi N = 8
  has relative F = 0 and full F = 0.68. Full F minus the ideal mixing term equals the L-FTS/CL-FTS Hamiltonian
  including its constant at the saddle point; those Hamiltonians are unchanged.
- CE SCFT results include molecular chemical potentials and grand potentials
  (2026-10-22), with saddle-point field means restored analytically. Reference-species
  exchange chemical potentials and reduced grand potentials are available through
  `SCFTResult.coexistence_quantities()`. `find_binary_coexistence()` locates a common
  tangent using only canonical phase evaluations; `homogeneous_thermodynamics()`
  evaluates homogeneous polymer mixtures analytically. Bulk coexistence requires
  converged incompressible periodic states with free chains. See
  docs/theory/PhaseEquilibrium.md and examples/scft/CanonicalCoexistence.py.
- `fts_consistent_energy` is removed (2026-10-22). It pinned the field means of an
  incompressible run to the saddle point so that the free energy matched the field-theoretic
  Hamiltonian; the free energy no longer depends on the field means, and incompressible
  fields have zero means. A parameter file that sets it gets a warning.

- **CUDA 11 is no longer supported** (2026-10-21): the GPU backend needs CUDA 12.2 or
  newer, because the device code is C++20 (nvcc 11.x has no `-std=c++20`; nvcc 12.0/12.1
  cannot compile `std::source_location`). It also needs CMake 3.25.2+ (which passes
  `-std=c++20` to nvcc), a host compiler with `std::source_location` (GCC 11+, Clang 15+)
  and an NVIDIA driver for the toolkit (R550+ for CUDA 12.4). `POLYMERFTS_USE_CUDA` is now
  `AUTO`, `ON` or `OFF`, `AUTO` by default: the GPU backend is built when that toolchain is
  found, otherwise the configure step warns, with the reason and the remedy, and builds
  without it; `ON` makes a missing piece an error, `OFF` skips the check. A build directory
  of an older version keeps the former BOOL switch, whose `ON` becomes `AUTO`. All checks
  look at the nvcc that compiles the code (`CMAKE_CUDA_COMPILER`, `CUDACXX` or PATH); a
  missing or too old one is not kept in the cache, and `-DCMAKE_CUDA_HOST_COMPILER` or
  `CUDAHOSTCXX` selects its host compiler. The cuFFT library is the one of that toolkit,
  also when a conda environment on `CMAKE_PREFIX_PATH` carries CUDA libraries of another
  version, and the default compute capabilities follow the toolkit (60-90 with CUDA 12.2,
  plus 100 and 120 from 12.8, 75-120 with CUDA 13). pip builds (`pyproject.toml`) and the
  conda recipe require CMake 3.25.2+; the conda recipe now declares spglib and LAPACKE and
  builds the MKL backend (FFTW would make the Apache-2.0 package GPL-bound).
  `POLYMERFTS_INSTALL_HEADERS` now also
  installs the static libraries and exports `polymerfts::polymerfts_backends`, so a
  downstream project can `find_package(polymerfts)`.
- CUDA error checks follow the CUDA C++ Best Practices Guide (2026-10-21): every kernel launch
  is followed by a `cudaGetLastError()` check, and the cuFFT, CUB and runtime calls whose
  return codes were ignored (about 210) are checked. Before, about half the launches were
  checked only by a later `cudaPeekAtLastError()`, so an error was reported at an unrelated
  place and, since the peek does not clear it, again by the next call after the exception
  (for example on a retry from Python). No synchronization was added. The CN-ADI solver
  requests at most the device's opt-in shared-memory limit instead of an unchecked 96000 bytes.
  The check macros are named as is conventional for macros (C++ Core Guidelines ES.32):
  `CUDA_CHECK(...)`, `CUFFT_CHECK(...)` and `CUDA_CHECK_LAUNCH()` after a kernel launch
  (formerly `gpu_error_check`, `cufft_error_check`). Release calls (`cudaFree`,
  `cudaFreeHost`, `*Destroy`) go through `ignore_release_error()`, so a failed release no
  longer leaves its error for the next, unrelated launch check. A failed allocation names
  its size and the calling file, line and function.
- C++ ownership (2026-10-21): device memory is held in `CudaDeviceBuffer<T>` /
  `CudaPinnedBuffer<T>` (`CudaCommon.h`), host arrays in `std::vector` and owned objects in
  `std::unique_ptr`, replacing hand-written allocation and cleanup code (the FFTW/MKL plans
  and the `fftw_malloc` buffers of the CPU FFT classes keep their `release()` functions).
  `createFFT()` returns a `std::unique_ptr<FFT<T>>`, and the computation classes take a
  backend factory returning `std::unique_ptr<ComputationBackend<T>>`. The factory functions
  still return raw pointers that the caller owns. Results are unchanged. The MKL ObliqueZ
  CrysFFT sizes each of its per-thread z buffers from its own requirement (two grids with the
  same Nx*Ny*Nz but a different Nz overflowed the spectrum buffer on one thread). Every owned
  CUDA allocation, including the backend's propagator and checkpoint arrays, is counted, so
  the tests check that every object returns all of it, also after an injected or a real
  allocation failure; GCC and Clang reject `delete` on a buffer. The CPU CN-ADI tridiagonal
  tables hold one entry per grid position along their axis (they were sized to the whole
  grid). The per-monomer tables in the propagator loops are read with `at()`, so a missing
  entry throws instead of being inserted from several threads at once, and the CUDA
  `set_space_group` builds its device table before it changes the box. cuFFT plans, CUDA
  streams and events are counted too (`count_*`/`release_*` helpers), so the tests check
  that every object releases them; a failed `cufftDestroy` no longer clears an unrelated
  error (cuFFT has its own runtime). The CUDA CrysFFT solver step no longer copies the
  bond-length map, and the FFTW ObliqueZ step reads its profiling variable once. The CPU
  CN-ADI solver keeps its work arrays in reused per-thread buffers instead of grid-sized
  variable-length arrays on the stack (a GCC extension; 2 MB each at 64^3, several per
  step), and the duplicate copies of the CN-ADI step functions, unused on both
  platforms, were removed (about 500 lines).
- The Python layer is organised for researchers: each simulation class reads its options
  straight from the `params` dictionary, and the package has fewer files (`compressor.py`
  merged into `lfts.py`; `config.py` and `io.py` into `utils.py`; `result.py` into
  `scft.py`). `LR`, `LRAM`, `load_config`, `save_vtk`, `SCFTResult` and `IterationInfo`
  stay importable from `polymerfts`. The `SystemSpec`/`*Options` layer tried during the
  2026-10 cleanup was removed again before release.
- Python helpers (2026-10-20): `SCFTResult.phi` and `SCFTResult.w` are on the full grid
  also with a space group, and `w` is a dictionary like `initial_fields`, so a result can
  restart `SCFT.run`. `load_config` evaluates numpy expressions such as
  `"4.0*np.sqrt(3)/2"` with a restricted evaluator instead of `eval`; anything outside
  numbers, arithmetic, `np.pi`, `np.e` and a list of elementwise functions, with a
  complex or non-finite value, or nested deeply enough to exhaust the parser, stays a
  string.
  `LFTS.from_file` and `CLFTS.from_file` join `SCFT.from_file`. `save_vtk` writes
  several fields per file, 1D/2D grids, flat arrays with `nx`, and non-orthogonal cells
  (`angles`, also `[gamma]` for a 2D cell as in the params dictionary), places the points
  at the cell centres, and writes vectorized; its second argument is now named `fields`.
  `LR` and `LRAM` return arrays in the shape of their input; `LR` failed for flat fields.
  `LR` and `WTMD` pass `axes` to `np.fft.irfftn` together with `s`; omitting it is
  deprecated since NumPy 2.0 (results unchanged).
- Grid-size limits (2026-10-20, `src/common/GridLimits.h`): grid arrays are indexed with
  32-bit integers, so a computation box or space group of more than 1,073,217,535 points (half of
  INT_MAX - 2^20, because the largest arrays hold two fields), Anderson mixing of more
  than 2,146,435,071 variables, and a CUDA launch of more than 2^20 threads
  (`LFTS_GPU_NUM_BLOCKS` x `LFTS_GPU_NUM_THREADS`) or of more than 1024 threads per block
  are rejected with a clear error.
  Before, the grid-point product could overflow silently and SCFT/L-FTS failed with a
  pybind11 type error. The grid-point product of the Python box factory, the
  multi-component inner products (CUDA kernel and host loop) and the space-group
  offsets are formed in 64 bits, and the 2D DCT-4/DST-4 and DST-3 kernels no longer
  need one thread per row element (rows longer than 1024 points failed to launch).
  The 1D CUDA DCT-II/III (reflecting) launched one block with a thread per point rounded
  up to a power of two, and the DST-II/III (absorbing) a thread per pair of points, so 1D
  reflecting grids longer than 1024 points and absorbing grids longer than 2046 points
  could not run on CUDA; the kernels now loop, and a 1D line that does not fit in the
  shared memory of a block (beyond 4096 points for the DCT-II with 48 KB per block) is
  rejected with a message pointing to the CPU platforms, the default for 1D. Both limits are importable as `_core.MAX_GRID_POINTS` and `_core.MAX_GRID_ARRAY`.
- A space group checks the cell against its crystal system when it is attached to a
  computation box and on every box change (`set_lx`, `set_lattice_parameters`), in C++
  (2026-10-20): cubic needs a = b = c and right angles, tetragonal a = b and right angles,
  orthorhombic right angles, hexagonal (and trigonal in hexagonal axes) a = b with angles
  [90, 90, 120], rhombohedral-axes trigonal a = b = c and alpha = beta = gamma, and
  monoclinic and triclinic groups a cell metric that every rotation preserves (the
  monoclinic unique axis perpendicular to the other two). `scft.py` checked its input
  already; a `PropagatorSolver` with a space group ran on any cell, where the symmetry
  operations are not isometries of the box.
- Removed small features that carried disproportionate code: the ADAM optimizer
  (`optimizer.name` must be `am`), the `validation_runtime` option of L-FTS (SCFT, L-FTS
  and CL-FTS check the partition consistency once per run and raise `ValidationError`
  on failure, as `validation_runtime` did by default) and `configure_logging` (the
  package prints).
  `polymerfts.draw_polymer_architectures` redraws the architecture figures that the
  constructors write. Docstrings were cut to the essentials; the parameter documentation
  moved to docs/references/Parameters.md.
- Complex Langevin FTS (`polymerfts.CLFTS`, `examples/clfts/`) promoted from the private
  development folder; dynamical stabilization follows Willis and Matsen.
- Default `stress_interval` changed from 3 to 1 (2026-03-30, c05e3108).
- Mixed periodic/non-periodic boundary conditions raise instead of silently switching the
  solver (2026-07-30, 7e966860).
- Removed numerical methods ETDRK4, CN-ADI4-LR, CN-ADI4-GR and SDC (2026-01-17, f5ab51c4;
  2026-01-28, 05bd9fc0); `numerical_method` accepts `rqm4`, `rk2` and `cn-adi2`.
- Removed the cell-averaged bond-function feature (2026-01-17, ed66e3c8) and the analytical
  sinc correction (0e626ecd).
- Removed deprecated helpers: the `Array` device-array class, `validate_scft_params`
  (use `validate_system_params`), `PropagatorSolver.set_space_group` (pass `space_group`
  to the constructor), the SCFT context manager, `LFTSResult`,
  `save_config`/`create_template_config` (2026-10-19).
- `SpaceGroup.physical_basis_name()` reports the active physical basis
  (`irreducible`, `pmmm`, `m3`, `z-mirror`). `SpaceGroup.m3_crysfft_feasible()` tells
  whether the recursive 3m CrysFFT works on the grid (even nx, nz >= 16, translation
  parity); SCFT enables the m3 basis only then and otherwise uses the Pmmm basis (or the
  irreducible basis for groups without xyz mirrors), instead of failing in the solver or
  silently running the m3 basis through the full-grid FFT on small grids.
- SCFT checks that the box matches the crystal system of the space group (cubic: equal
  lengths and 90-degree angles; tetragonal: a = b and 90-degree angles; orthorhombic:
  90-degree angles; hexagonal/trigonal on hexagonal axes:
  a = b and gamma = 120; rhombohedral axes: a = b = c and alpha = beta = gamma) and rejects
  box relaxation for rhombohedral-axes settings (2026-10-20). A failed
  `enable_m3_physical_basis()` leaves the `SpaceGroup` unchanged instead of half-updated
  (which corrupted memory in the next `symmetrize`). The physical basis is frozen once a
  computation box or solver holds the group (their reduced-basis maps are snapshots), the
  CrysFFT selector rejects a Pmmm/m3 basis on a non-orthogonal cell and a z-mirror basis
  with alpha or beta != 90 (reachable only when a physical basis is enabled by hand on a
  `SpaceGroup` passed to `PropagatorSolver` or through the C++ API; `SCFT` checks the cell
  before enabling one), and
  `symmetrize_reduced_basis` requires a C-contiguous float64 array like `zero_mean`
  because it works in place (both bindings now refuse dtype conversions, which would have
  modified a copy). `SpaceGroup.uses_hexagonal_axes()` tells the axes setting of a
  trigonal group, so the cell check requires the hexagonal cell for hexagonal-axes
  settings and the rhombohedral cell for rhombohedral-axes settings; `angles` with other
  than three entries is rejected for a 3D box.
- The `SpaceGroup` field conversions (`to_reduced_basis`, `from_reduced_basis`,
  `reduced_to_irreducible`, `irreducible_to_reduced`) keep the layout of their input: a
  1-D input gives a 1-D output, an `(n_fields, n_in)` input an `(n_fields, n_out)` output
  (before, two of them squeezed a single field and the other two always returned 2-D).
  Grid compatibility with the translation parts of a space group is checked per axis: a
  1/2 translation along z alone (P6_3/mmc) no longer forces even nx and ny (2026-10-20).
- CUDA real transforms (`CudaRealTransform`): the lengths each type supports were
  verified against FFTW (DCT-2/3/4, DST-1/4 and the 2D/3D DCT-1 accept any length; the
  1D DCT-1 needs an odd number of points; DST-2 and the 2D/3D DST-3 need even lengths)
  and the constructors now reject unsupported lengths instead of returning wrong
  values; nothing in the library passes such lengths (2026-10-20).
- The CPU stress of an orthogonal 2D box no longer computes the shear term (index 2) that
  the CUDA side and the 3D path skip unless `set_force_off_diagonal_stress(True)` is set
  (SCFT sets it when angles are optimized); the platforms now return identical vectors.
- Parameter checks (2026-10-19): crystal systems are checked against the box dimension,
  `max_iter` must be positive; L-FTS/CL-FTS accept
  `verbose_level` 0 and reject other values than 0-2, reject unknown compressor names and
  WTMD on non-3D grids; `"platform": "auto"` may be given explicitly.
  `PlatformSelector.avail_platforms()` lists `cuda` only when a GPU is visible, so CUDA
  builds fall back to the CPU on nodes without a device.
- Python bindings (2026-10-19): input arrays are made C-contiguous and cast to the expected
  dtype (a strided view was read as if contiguous; `zero_mean`, which works in place,
  rejects arrays that would have to be copied), the complex-field factory reads a real
  `mask`, objects created by a factory keep the objects they point to alive,
  `PlatformSelector.avail_platforms` and the `PropagatorCode` helpers are static methods,
  `get_computation_propagators` lost its misspelled name.
- CL-FTS uses the same idioms as L-FTS (propagator solver accessors, architecture figures,
  material-conservation and partition checks); `calculate_sigma` lives in `utils`.
- CUDA: `compute_single_segment_stress` lives once in `CudaSolverPseudoBase` (as on the CPU
  side); RQM4, RK2 and the discrete solver supply the bond factor and their scratch
  buffers. RK2 now transforms the propagator pair in one batched cuFFT call like the
  other two (2026-10-20).
- The propagator computations exist once, in `src/common/`: `ComputationContinuous`,
  `ComputationDiscrete` and their reduce-memory variants decide what is computed and in
  which order, and every operation on grid data goes through `ComputationBackend`
  (`CpuComputationBackend`, `CudaComputationBackend`). The eight platform classes
  (`CpuComputationContinuous`, `CudaComputationReduceMemoryDiscrete`, ...) were removed:
  the factories create the computations with their backend, and the C++ tests obtain
  them from the factories (`tests/TestComputationFactory.h`). About 10,000 lines of
  duplicated CPU/CUDA code were removed (2026-10-20).
  CUDA stress sums the segment stresses of a block on the device and copies them once per
  block instead of once per segment with a device-wide synchronization. CUDA reduce-memory
  keeps the 2√N recomputation buffers on the device instead of round-tripping each
  recomputed segment through pinned host memory; checkpoints stay in pinned host memory.
  Results are unchanged except for the two fixes above and round-off in complex-field
  normalizations and in discrete-chain solvent concentrations (exp(-w ds) is recomputed
  from w with the global ds instead of taken from the solver).
- CUDA non-periodic boundaries: the DCT/DST transforms and the kernels around them are
  issued on the propagator's stream (`CudaFFTBase::forward_stream`), like the periodic
  cuFFT path; they ran on the legacy default stream, which serialized the streams
  (2026-10-20). Results are unchanged.
- `TestConvergenceOrderLite` uses a Gaussian-smoothed field (white noise on a 0.03 grid is
  outside the asymptotic regime at ds = 0.04-0.01) and now passes: RQM4 order 3.6, RK2 2.0.
- CPU CrysFFT (2026-10-20): the Recursive3m k-cache generator and the ObliqueZ DCT pipelines
  exist once (about 300 lines removed); the ObliqueZ DCT scale factors are the analytic
  constants (FFTW: 2 forward, REDFT01 backward; MKL: none) instead of a least-squares
  calibration run at construction, which changes results only at round-off; the
  per-thread Boltzmann/k caches are owned by the CrysFFT object (a map keyed by thread
  id; each thread remembers its slot, so the steady state takes no lock) instead of
  thread-local maps keyed by the object's address, so they are released when the object
  is destroyed (long sessions that create many solvers no longer accumulate stale
  tables in every worker thread).
- Build: C++20, CMake 3.18+ (3.25.2+ for the GPU backend), spglib and LAPACKE are required at configure time; the CPU
  backends share a `cpu-common` object library; `POLYMERFTS_WERROR` turns warnings into
  errors (on in CI); the LAPACKE header may be `mkl_lapacke.h` (conda-forge's MKL build of
  liblapacke ships no `lapacke.h`); `POLYMERFTS_CUDA_ARCHITECTURES` selects the compute capabilities;
  the installed `_core.so` carries an RPATH to the libraries it was linked against; the
  staged package in `build/python` drops modules that no longer exist in the sources. The generated build no longer depends on `LIBRARY_PATH` or
  `CPLUS_INCLUDE_PATH` of the configuring shell: libraries are linked by absolute path and
  the CUDA toolkit directories are always passed explicitly. Packaging with pip
  (`pyproject.toml`) and conda (`conda-recipe/`) was added (2026-01-09, fe3a54df).

### Performance

- CUDA, 2026-10-20, A10, 64^3, results unchanged except where noted. CN-ADI line solvers
  were launched with 256 blocks for 4096 lines (16 busy blocks): 1.7x faster iterations.
  CrysFFT stress applies one multiplier per group of equivalent axes (one pass instead of
  three for cubic groups, two for tetragonal; stress changes at the 1e-14 level and the
  CPU path does the same): BCC iterations 1.4x (continuous) and 1.8x (discrete) faster.
  Box relaxation: the CrysFFT tables reuse their device buffers and are uploaded on
  their own stream (BCC with box relaxation 80 -> 54 ms per iteration, plain RQM4
  113 -> 107 ms). The segment stress sums all components in one pass with a
  deterministic two-stage reduction (stress changes at the 1e-16 level; iterations
  1.14x faster at 32^3, 1.06-1.09x at 64^3). The last RQM4 half-step factor is applied
  inside the Richardson combination, real-field DCT/DST transforms run in the output
  buffer, and the default launch is 1024 blocks. Block concentrations add up to 32
  segment products per kernel launch instead of one (concentration time -35 to -40%,
  results unchanged). The Gyroid example (32^3, box relaxation) runs in 4.5 s instead of
  5.05 s.
- CPU: the CrysFFT path of the RQM4, RK2 and discrete solvers no longer expands the
  propagator to the full grid every step, and their work arrays are kept per thread
  by the solver and released with it (1.2-1.6x faster with Im-3m at 64^3).
- Every CUDA kernel now loops over its index range, so results do not depend on the
  launch configuration.

### Added

- Space-group reduced basis with CrysFFT fast paths (Pmmm DCT, Recursive 3m, ObliqueZ) and
  physical bases (m3, pmmm, z-mirror) for 3D periodic real fields.
- Crystal systems with angle optimization (monoclinic, triclinic, 2D oblique/hexagonal).
- `TestSpaceGroupPhases` with 14 converged phase fixtures.
- Well-tempered metadynamics (`WTMD`) for L-FTS, `Smearing`, `SCFTResult`/`IterationInfo`,
  YAML/JSON parameter files (`load_config`, `SCFT.from_file`).
- Wheel installation records let pip manage package upgrades and removal.
- `TestStressNonPeriodic` (2026-10-20): the absorbing-wall tolerance of the continuous
  chain is 1e-5 instead of 1.5e-2 (the measured finite-difference mismatch is 3e-7 since
  the chain-end stress fix; it was 5e-3), and the 3D mixed reflecting/absorbing case also
  runs for continuous chains (tolerance 1e-4, measured 1.2e-5, was 4.8e-3).
- Tests (2026-10-20): `TestConfigFile`, `TestScftCallback`, `TestScftResult`,
  `TestSaveVtk` and `TestCompressors` cover the Python helpers (those that run a solver
  skip on builds without a CPU platform); `TestGridLimits` covers the grid-size limits and
  `TestSpaceGroupCell` the cell check of space groups.
- Tests (2026-10-19): `TestStressComplexPlatforms` (complex stress on every platform, 64^3
  grid), `TestStressNonPeriodicRK2`, `TestComplexMask` (array handling and object
  lifetimes of the bindings), `TestLftsRun` (L-FTS/CL-FTS run loops and checkpoint
  continuation); `TestConvergenceOrderLite` had no `__main__` block and never ran under
  ctest; `TestParams` regained the solver-option assertions.

## [0.9.1] - 2025-05-02

- Complex-valued fields: stress computation and `CudaPseudo` for complex type.
- Pure-Python library split out of the example scripts (`src/python/`).
- Fixes: wrong dH_ref/dχN, `np.complex64` → `np.complex128`.
- `SymmetricPolymerTheory`: eigenvalues and eigenvectors reordered.

## [0.9.0] - 2024-10-02

- `SymmetricPolymerTheory` applied throughout the examples; variable renamed `eigen` →
  `auxiliary`.
- Fix: wrong sign of dH/dW.
- SCFT fields can be saved in YAML; `reduce_gpu_memory_usage` added to all SCFT examples.

## [0.8.4] - 2024-08-03

- Grand canonical ensemble functions (`get_total_concentration_gce`, under development).
- Discrete chains: segment indices start from 1; `propagator_junction_start` removed.
- Fix: standard deviation of the stress in `TestPseudo`.

## [0.8.3.1] - 2024-06-22

- Propagator aggregation reworked (variable names follow the paper); fixes for complex side
  chains; Simpson-rule dependence removed from aggregation.

## [0.8.3] - 2024-05-25

- Free energy printed per chain (H → H/CV).
- Async operations in parallel propagator jobs; PSCF conversion tools; propagator code
  order redefined; `check_total_partition()` simplified.

## [0.8.2] - 2024-03-22

- Fix: dendrimer handling in `PropagatorAnalyzer::add_polymer`; key decomposition
  simplified.

## [0.8.1] - 2024-01-15

- Crank-Nicolson real-space solver up to 3D (CPU and CUDA).
- Branched-polymer architecture drawing; multi-GPU fix for the discrete chain model;
  OpenMP replaces batched cuFFT.

## [0.8.0] - 2023-07-11

- First tagged release of the C++/CUDA core with Python bindings: SCFT and L-FTS for
  arbitrary acyclic branched polymers, continuous and discrete chain models.
