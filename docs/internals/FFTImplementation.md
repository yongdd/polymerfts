# FFT Implementation

This document describes the FFT implementations used in the polymer field theory library, including DCT/DST for non-periodic boundaries and crystallographic FFT (CrysFFT) for space-group symmetry acceleration.

## Table of Contents

1. [Overview](#1-overview)
2. [Discrete Cosine Transform (DCT)](#2-discrete-cosine-transform-dct)
3. [Discrete Sine Transform (DST)](#3-discrete-sine-transform-dst)
4. [Multi-dimensional Transforms](#4-multi-dimensional-transforms)
5. [Normalization](#5-normalization)
6. [Crystallographic FFT (CrysFFT)](#6-crystallographic-fft-crysfft)
7. [Implementation Notes](#7-implementation-notes)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [References](#9-references)

---

## 1. Overview

The library provides spectral transforms for solving the modified diffusion equation:

| Transform | Boundary Condition | Physical Meaning |
|-----------|-------------------|------------------|
| FFT | Periodic | Cyclic boundary |
| DCT | Reflecting | Neumann (zero flux) |
| DST | Absorbing | Dirichlet (zero value) |

**Implementations:**
- **CPU**: MKL or FFTW library for FFT, `MakhoulFFT` (`MakhoulTransform` over the real FFT of that library) for DCT/DST
- **CUDA**: cuFFT for FFT, custom `CudaRealTransform` for DCT/DST, `CudaMakhoulTransform` for boxes that mix periodic axes and walls (Section 2.5)

---

## 2. Discrete Cosine Transform (DCT)

### 2.1 Types and Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DCT-1 | REDFT00 | $`X_k = x_0 + (-1)^k x_{N} + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi k n}{N}\right)`$ | N+1 |
| DCT-2 | REDFT10 | $`X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k (2n+1)}{2N}\right)`$ | N |
| DCT-3 | REDFT01 | $`X_k = x_0 + 2\sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi n (2k+1)}{2N}\right)`$ | N |
| DCT-4 | REDFT11 | $`X_k = 2\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)`$ | N |

### 2.2 Boundary Conditions and Symmetry

A "point" is a grid index and a "half-point" lies halfway between two indices (a cell face of the cell-centered grid).

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DCT-1 | Even at point $`n=0`$ | Even at point $`n=N`$ | Neumann-Neumann (node-centered) |
| DCT-2 | Even at half-point $`n=-1/2`$ | Even at half-point $`n=N-1/2`$ | Neumann-Neumann (cell-centered; reflecting BC) |
| DCT-3 | Even at point $`n=0`$ | Odd at point $`n=N`$ | Neumann-Dirichlet |
| DCT-4 | Even at half-point $`n=-1/2`$ | Odd at half-point $`n=N-1/2`$ | Neumann-Dirichlet (half-sample) |

**Inverse relationships:**
- DCT-1 is self-inverse: DCT-1(DCT-1(x)) = 2N · x (for an N+1 point array)
- DCT-2 and DCT-3 are inverses: DCT-3(DCT-2(x)) = 2N · x
- DCT-4 is self-inverse: DCT-4(DCT-4(x)) = 2N · x

### 2.3 FCT Algorithm (CUDA)

The CUDA implementation uses the Fast Cosine Transform (FCT) method [1], which uses N-point FFT instead of the traditional 2N-point odd extension:

| DCT Type | FFT Size | FFT Type | Speedup |
|----------|----------|----------|---------|
| DCT-1 | N | Z2D | ~2x |
| DCT-2 | N | Z2D | ~2x |
| DCT-3 | N | D2Z | ~2x |
| DCT-4 | 2N | Z2Z | Phase rotation |

In 2D and 3D, DCT-1 and DST-1 transform the whole mirror-extended array
($`2N_x \times 2N_y \times 2N_z`$ points, $`2(N_x+1) \times 2(N_y+1) \times 2(N_z+1)`$ for DST-1) with one FFT,
so the N-point saving of the type-1 rows applies to 1D transforms only.

### 2.4 DCT and DST of a Whole Grid (CPU)

`MakhoulTransform` runs the DCT-2/3 and DST-2/3 of a grid of one to three axes, with any mix of cosine and sine axes, through one periodic real FFT of the grid. `createFFT()` (`FFTFactory.h`) returns it for a box of reflecting and absorbing axes as `MakhoulFFT`, one class for both CPU libraries over the real FFT of the selected one (`FftwRealFFT`, `MklRealFFT`), the transform they run fastest; `FftwFFT` and `MklFFT` are periodic only. `CrysFFTPmmm` uses `MakhoulTransform` as well, over the real FFT of the selected library.

Makhoul's reordering [1] stores the even samples of every axis forward and the odd samples backward. With $`V`$ the real FFT of the reordered grid and $`W_N^k = e^{-i\pi k/(2N)}`$ on an axis of $`N`$ points, one axis gives

```math
X_k = 2\,\mathrm{Re}\left(W_N^k V_k\right).
```

The real FFT of the last axis leaves complex lines on the other axes, and on a complex line the cosine transform is $`W_N^k V_k + W_N^{-k} V_{-k}`$. Modes $`k`$ and $`N-k`$ of these axes therefore combine in pairs, one whole plane or row at a time,

```math
Y_k = W_N^k V_k + W_N^{-k} V_{-k}, \qquad Y_{N-k} = i\left(W_N^k V_k - W_N^{-k} V_{-k}\right),
```

where mode 0 gains 2 and mode $`N/2`$ gains $`\sqrt{2}`$. The last axis, which holds only the half spectrum, then gives real coefficients as in one dimension, and the DCT-3 runs these steps backward. A sine axis reduces to a cosine axis through alternating signs and reversed coefficients,

```math
\mathrm{DST2}[x]_k = \mathrm{DCT2}\left[(-1)^j x_j\right]_{N-1-k},
```

so it changes only the tables that place the samples and the coefficients. The real FFT runs unnormalized on per-thread scratch of `MakhoulTransform`, which starts on a 64-byte boundary and which it may overwrite, so the transform adds no copy and no 1/N pass. Complex fields transform their real and imaginary parts in turn, read from and written to every second value.

### 2.5 Boxes that Mix Periodic Axes and Walls

The Fourier transform of a box that mixes periodic axes and walls (`FourierTransform.h`) has complex coefficients: the DFT along the periodic axes and the cosine or sine series along the walls; `PropagatorSolver.fourier` takes the cosine series along every wall. On the CPU, `MakhoulTransform` runs this transform as well. A periodic axis is neither reordered nor combined, so the same real FFT of the grid leaves the DFT on it. The coefficients $`Y`$ of a real field are conjugate under $`\mathbf{k}_p \to -\mathbf{k}_p`$ on the periodic axes, with the wall coefficients unchanged, so the half of the last axis that the real FFT does not hold is $`Y(\mathbf{k}_p, -k) = \overline{Y(-\mathbf{k}_p, k)}`$. A wall as the last axis then gives the cosine transform of a complex line,

```math
X_k = a + b, \qquad X_{N-k} = i\,(a - b), \qquad a = W_N^k\, Y(\mathbf{k}_p, k), \qquad b = W_N^{-k}\, \overline{Y(-\mathbf{k}_p, k)},
```

and a periodic last axis takes $`\overline{Y(-\mathbf{k}_p, N-k)}`$ for its upper half. The backward transform reads the Hermitian part $`\left(C(\mathbf{k}) + \overline{C(-\mathbf{k})}\right)/2`$ of the coefficients, which holds the coefficients of the real part of the field, and runs these steps backward. A complex field has $`C = C[\mathrm{Re}\, f] + i\, C[\mathrm{Im}\, f]`$ and is restored from the Hermitian parts of $`C`$ and of $`-iC`$.

On CUDA, `CudaMakhoulTransform` runs the same scheme with one cuFFT of the grid. A kernel reorders the samples along the wall axes; D2Z transforms a real field, and Z2Z transforms a complex field in place, without the split into its real and imaginary parts that the CPU makes. A second kernel forms each coefficient from its $`2^w`$ modes, $`w`$ the number of wall axes, by the cosine transform of a complex line, $`X_k = W_N^k V_k + W_N^{-k} V_{-k}`$, along every wall axis at once, a sine axis taking its coefficients in reverse order; it reads the half of a real field's spectrum that D2Z does not hold as $`V(-\mathbf{k}) = \overline{V(\mathbf{k})}`$. The backward transform forms each mode from its coefficients, from their Hermitian part for a real field, with $`V_k = W_N^{-k}\left(X_k - i X_{N-k}\right)`$ ($`X_N = 0`$) per wall axis, twice the inverse of the forward step as REDFT01, then runs Z2D or the inverse Z2Z and restores the samples. Each output is gathered from its inputs, so one pass covers all wall axes and no pair of values is updated in place. The scratch holds about two grids of doubles per thread on the CPU and two to three, with the work area of cuFFT, on CUDA (Section 8.6).

---

## 3. Discrete Sine Transform (DST)

### 3.1 Types and Definitions

| Type | FFTW Name | Formula | Grid Points |
|------|-----------|---------|-------------|
| DST-1 | RODFT00 | $`Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (n+1)(k+1)}{N+1}\right)`$ | N |
| DST-2 | RODFT10 | $`Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(k+1)}{2N}\right)`$ | N |
| DST-3 | RODFT01 | $`Y_k = (-1)^k x_{N-1} + 2\sum_{n=0}^{N-2} x_n \sin\left(\frac{\pi (n+1)(2k+1)}{2N}\right)`$ | N |
| DST-4 | RODFT11 | $`Y_k = 2\sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (2n+1)(2k+1)}{4N}\right)`$ | N |

### 3.2 Boundary Conditions and Symmetry

| Type | Left Boundary | Right Boundary | Physical Meaning |
|------|---------------|----------------|------------------|
| DST-1 | Odd at point $`n=-1`$ | Odd at point $`n=N`$ | Dirichlet-Dirichlet (node-centered) |
| DST-2 | Odd at half-point $`n=-1/2`$ | Odd at half-point $`n=N-1/2`$ | Dirichlet-Dirichlet (cell-centered; absorbing BC) |
| DST-3 | Odd at point $`n=-1`$ | Even at point $`n=N-1`$ | Dirichlet-Neumann |
| DST-4 | Odd at half-point $`n=-1/2`$ | Even at half-point $`n=N-1/2`$ | Dirichlet-Neumann (half-sample) |

**Inverse relationships:**
- DST-1 is self-inverse: DST-1(DST-1(x)) = 2(N+1) · x
- DST-2 and DST-3 are inverses: DST-3(DST-2(x)) = 2N · x
- DST-4 is self-inverse: DST-4(DST-4(x)) = 2N · x

### 3.3 FST Algorithm (CUDA)

| DST Type | FFT Size | FFT Type | Traditional Size | Speedup |
|----------|----------|----------|-----------------|---------|
| DST-1 | N+1 | Z2D | 2(N+1) | ~2x |
| DST-2 | N | Z2D | 2N | ~2x |
| DST-3 | N | D2Z | 2N | ~2x |
| DST-4 | 2N | Z2Z | 2N | Phase rotation |

---

## 4. Multi-dimensional Transforms

### Supported Configurations (CudaRealTransform)

| Dimension | Supported Types | Mixed Types |
|-----------|-----------------|-------------|
| 1D | DCT/DST-1,2,3,4 | N/A |
| 2D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 |
| 3D | DCT/DST-1,2,3,4 | DCT/DST-2,3,4 |

**Note**: `CudaRealTransform` cannot mix type 1 with other types, because its type-1 transform of a 2D or 3D grid is one FFT of the whole extended grid. `MklRealTransform` transforms each axis on its own and accepts any mix.

### 3D Transform Algorithm (CudaRealTransform)

On the CPU, the wall transforms (`MakhoulFFT`) and the Pmmm CrysFFT run DCT/DST-2/3 as one real FFT of the whole grid instead (Section 2.4). `MklRealTransform` runs every type, type 1 included, as 1D transforms along z, y and x, one line at a time (Section 8.2).

**Types 2,3,4 (Separable):**
- Batched 1D transforms along Z dimension (innermost)
- Batched 1D transforms along Y dimension
- Batched 1D transforms along X dimension (outermost)

**Type-1 (Non-separable):**
- Mirror/odd extension in all three dimensions
- Single 3D D2Z FFT
- Extract coefficients from symmetric/antisymmetric positions

### Memory Layout

| Transform Type | Buffer Size per Element |
|----------------|------------------------|
| DCT-1 | N+1 |
| DCT-2/3 | N |
| DCT-4 | 2N (complex) |
| DST-1 | N+3 (FCT/FST padded) |
| DST-2 | N/2+1 complex |
| DST-3 | N+2 (padded) |
| DST-4 | 2N (complex) |

---

## 5. Normalization

Both DCT and DST follow FFTW's unnormalized convention.

### DCT Normalization Factors

| Transform | Round-trip Scale |
|-----------|------------------|
| 1D DCT-1 | 2N for N+1 points |
| 1D DCT-2/3/4 | 2N |
| 2D DCT-1 | 4 N_x N_y for (N_x+1)(N_y+1) points |
| 2D DCT-2/3/4 | 4 N_x N_y |
| 3D DCT-1 | 8 N_x N_y N_z for (N_x+1)(N_y+1)(N_z+1) points |
| 3D DCT-2/3/4 | 8 N_x N_y N_z |

Multi-dimensional normalization is the product of the per-dimension factors (DCT-1 contributes 2N per dimension of N+1 points; types 2/3/4 contribute 2N per dimension of N points).

### DST Normalization Factors

| Transform | Round-trip Scale |
|-----------|------------------|
| 1D DST-1 | 2(N+1) |
| 1D DST-2/3/4 | 2N |
| 2D DST-1 | 4(N_x+1)(N_y+1) |
| 2D DST-2/3/4 | 4 N_x N_y |
| 3D DST-1 | 8(N_x+1)(N_y+1)(N_z+1) |
| 3D DST-2/3/4 | 8 N_x N_y N_z |

---

## 6. Crystallographic FFT (CrysFFT)

CrysFFT replaces the standard 3D FFT with symmetry-reduced transforms when a compatible space group is enabled.

### 6.1 Terminology

- **Logical grid**: Full simulation grid, size `Nx x Ny x Nz`
- **Physical grid**: CrysFFT working grid (reduced by symmetry)
- **Irreducible grid**: Smallest symmetry-unique grid (one point per orbit)

### 6.2 Algorithms

#### Pmmm DCT (Mirror Symmetry)

For space groups with three perpendicular mirror planes:

```
q_out = DCT-III[ exp(-k^2 * coeff) * DCT-II[q_in] ]
```

- Physical grid: `(Nx/2) x (Ny/2) x (Nz/2)`
- Wavenumbers: k = 2π * i / L (the physical DCT grid spans L/2, so the mode spacing is π/(L/2) = 2π/L)

#### 3m Recursive (2x2x2) Algorithm

For space groups with 3m translations (mirror + translational offsets):

- Same physical grid size `(Nx/2) x (Ny/2) x (Nz/2)`
- Uses eight sub-transforms with precomputed twiddle factors
- Faster than Pmmm DCT with CUDA; on the CPU the Pmmm DCT (`MakhoulTransform`) is faster (Sections 8.3 and 8.5)

#### ObliqueZ (z-mirror)

For space groups with z-mirror and α=β=90° (γ arbitrary):

```
q_out = DCT-III_z[ IFFT_xy[ exp(-k^2 * coeff) * FFT_xy[ DCT-II_z[q_in] ] ] ]
```

- Physical grid: `Nx x Ny x (Nz/2)`
- Supports oblique in-plane angles (e.g., hexagonal γ=120°)
- For glide mirror (t_z=1/2): requires `Nz % 4 == 0`

### 6.3 Activation Criteria

CrysFFT is enabled automatically when:
- `space_group` is set
- Dimension is 3D
- Boundary conditions are periodic
- Fields are real-valued
- Grid parity matches the algorithm: `Nx`, `Ny`, `Nz` all even for the Pmmm/3m variants; only `Nz` even for ObliqueZ (odd `Nx`/`Ny` are allowed with the z-mirror physical basis and on cells with γ≠90°)

**Selection order:** an enabled physical basis (Section 6.4) fixes the variant:
3m recursive for the m3 basis (on any grid it maps, since the engines are exact
for `Nz/2 >= 2`; the `Nz/2 >= 8` size rule below only chooses), Pmmm DCT for the
Pmmm basis and ObliqueZ for the z-mirror basis. With the irreducible basis, a cell
with α=β=90° and γ≠90° (e.g. hexagonal) uses ObliqueZ when the space group has a z-mirror, and the standard FFT
otherwise. An orthogonal cell needs `Nx`, `Ny`, `Nz` all even, and the first
applicable variant is used:

1. **3m recursive** if the space group provides 3m translations, the
   translation-parity fold permutation is valid (every generator translation
   must be an integral number of grid cells with a bijective parity map — see
   `Recursive3mFoldParity.h`; e.g. for Fddd's quarter-cell glides at most one
   of `Nx/4`, `Ny/4`, `Nz/4` may be odd), and `Nz/2 >= 8` (a performance rule:
   smaller grids gain nothing from the fold; `SpaceGroup.m3_crysfft_feasible()`,
   which SCFT uses to choose the m3 basis, applies it too)
2. **Pmmm DCT** if space group has mirror planes in x/y/z
3. **ObliqueZ** if space group has a z-mirror
4. Fall back to standard FFT

The 3m fold pairs each twiddle factor with the Boltzmann sub-array selected by
the translation-parity permutation; glide translations by an odd number of
grid cells permute this pairing away from the identity.
If the shortest generator translations give a permutation that is not a
bijection, the selector retries with each generator composed with a centering
translation, so on even grids with `Nz/2 >= 8` the parity test accepts exactly
the grids on which the m3 physical basis exists (the argument is in the comment
of `src/common/CrysFFTSelector.h`). A rejected grid continues with steps 2-4;
Fddd, which has no pure mirror, ends on the standard FFT.

### 6.4 Physical Basis Choices

| Basis | Grid Size | Mapping Overhead |
|-------|-----------|-----------------|
| Irreducible | Minimal (one per orbit) | Gather/scatter required |
| Pmmm physical | (Nx/2)(Ny/2)(Nz/2) | Identity (none) |
| 3m physical | (Nx/2)(Ny/2)(Nz/2) | Identity (none) |
| Z-mirror physical | Nx * Ny * (Nz/2) | Identity (none) |

The solver uses the basis the `SpaceGroup` holds. SCFT (`scft.py`) enables one
automatically: for an orthogonal cell the m3 basis when the recursive 3m CrysFFT is
feasible, otherwise the Pmmm basis; for a cell with α=β=90° and γ≠90° the z-mirror
basis; the irreducible basis stays when none applies. With `PropagatorSolver`, enable a
basis on the `SpaceGroup` before passing it (`enable_m3_physical_basis()`,
`enable_pmmm_physical_basis()`, `enable_z_mirror_physical_basis()`).

### 6.5 Phase Support Table

| Phase | Space group | 3m recursive | Pmmm DCT | ObliqueZ |
|-------|-------------|:------------:|:--------:|:--------:|
| BCC | Im-3m (529) | ✓ | ✓ | |
| FCC | Fm-3m (523) | ✓ | ✓ | |
| SC | Pm-3m (517) | ✓ | ✓ | |
| A15 | Pm-3n (520) | ✓ | ✓ | |
| DG (Gyroid) | Ia-3d (530) | ✓ | ✗ | |
| DD | Pn-3m (522) | ✓ | ✗ | |
| SG | I4_132 (510) | ✗ | ✗ | |
| HCP | P6_3/mmc (488) | ✗ | ✗ | ✓ |
| PL | P6_3/mmc (488) | ✗ | ✗ | ✓ |

---

## 7. Implementation Notes

### 7.1 CPU

**FFTW backend** (`POLYMERFTS_USE_FFTW`):

- `FftwFFT<T, DIM>`: Periodic FFT using FFTW3
- `FftwRealFFT`: the real FFT of the grid under `MakhoulTransform` (Section 2.4)
- `FftwCrysFFTRecursive3m`: Recursive 3m algorithm
- `FftwCrysFFTObliqueZ`: DCT-z + FFT-xy

**MKL backend** (`POLYMERFTS_USE_MKL`):

- `MklFFT<T, DIM>`: Periodic FFT using MKL DFTI
- `MklRealFFT`: the real FFT of the grid under `MakhoulTransform` (Section 2.4)
- `MklRealTransform1D/2D/3D`: DCT/DST of all types (FFT-based algorithms + MKL TT interface, see Section 8.2)
- `MklCrysFFTRecursive3m`: Recursive 3m algorithm
- `MklCrysFFTObliqueZ`: DCT-z through a real FFT of the same length (Makhoul) + FFT-xy on contiguous z planes

**Both backends**:

- `MakhoulFFT<T>`: DCT/DST of boxes of reflecting and absorbing axes through `MakhoulTransform` over the real FFT of the selected library (Section 2.4); `createFFT()` in `FFTFactory.h` returns it for such boxes and `FftwFFT` or `MklFFT` for periodic ones
- `CrysFFTPmmm`: Pmmm DCT through `MakhoulTransform` over the real FFT of the selected library (`createRealFFT()` in `FFTFactory.h`)
- `CpuFourierTransform`: the transforms of `PropagatorSolver.fourier`; boxes that mix periodic axes and walls run `MakhoulTransform` with periodic axes (Section 2.5)

`CpuSolverPseudoBase` holds one CrysFFT engine (`crysfft_`, behind the interface `CpuCrysFFT`): `CrysFFTPmmm`, or the recursive 3m or ObliqueZ engine of the library of the selected platform (`cpu-mkl` or `cpu-fftw`). The OpenMP threads of the propagator and stress loops call this one engine at once, each on its own thread-local scratch. The k-space tables belong to the solver, one per key for all threads: the Boltzmann factors of each step coefficient (`crysfft_boltz_`), built for the monomer type and contour step of every block (`crysfft_block_coeffs()`) when the CrysFFT path is set up, and the stress multipliers of each axis group and bond coefficient (`crysfft_multipliers_`; Pmmm and recursive 3m), built by `prepare_stress()` before the first stress loop. A cell change (`update_laplacian_operator()`) rebuilds every table in place. The engines take the table as an argument (`diffusion(in, out, table)`, `apply_multiplier(in, out, table)`) and hold none themselves, so the loops only read the tables and never build one; a missing key is an error. A recursive 3m table holds the real and imaginary parts of the eight folded octants, 16 physical-grid arrays, in one allocation, each array padded by one 64-byte line: the multiply loop reads all 16 at the same index, and on a physical grid of a power-of-two size the unpadded arrays would fall into one cache set (twice the time per diffusion on 16³).

**Threading**: FFTW/MKL multithreading is disabled; concurrency is handled at the solver level (OpenMP over propagators). Thread-local buffers are used to avoid race conditions.

**Arrays of the periodic FFTW transforms**: `FftwFFT` plans on `fftw_malloc` arrays, which it frees after planning, and runs its plans on the caller's arrays through FFTW's new-array execute functions when these have the alignment of the planning arrays (`fftw_alignment_of()` is 0) and the input and output do not overlap. An array 8 bytes off, such as a packed field after an odd number of grid points, goes through thread-local scratch that starts on a 64-byte boundary, and so does the input of the complex-to-real transform, which FFTW overwrites. The same plan thus runs on one alignment class, and the result does not depend on where the caller's arrays lie (`TestCpuFFTAddressIndependence`).

### 7.2 CUDA

- `CudaFFT<T, DIM>`: Standard FFT using cuFFT
- `CudaRealTransform1D/2D/3D`: DCT/DST using FCT/FST algorithms
- `CudaCrysFFT`: Pmmm DCT via CudaRealTransform
- `CudaMakhoulTransform`: the DFT along periodic axes and the cosine or sine series along walls through one cuFFT of the grid, with reordering and combining kernels (Section 2.5); `CudaFourierTransform` uses it for boxes that mix periodic axes and walls
- `CudaCrysFFTRecursive3m`: Recursive 3m on GPU with stream-aware diffusion
- `CudaCrysFFTObliqueZ`: DCT-z + FFT-xy as one real 3D FFT of the physical grid
  after Makhoul's permutation along z (even samples forward, odd samples backward).
  One kernel forms the DCT coefficients of `kz` and `Nz/2 - kz` from the half
  spectrum, applies both Boltzmann factors and returns to the half spectrum, so a
  diffusion step is three kernels and two cuFFT calls

`CudaSolverPseudoBase` keeps one CrysFFT object per stream, with its own cuFFT plans
and work buffers (`clone()` shares the read-only twiddle factors). The k-space tables
belong to the solver, one per key for all streams: the Boltzmann factors of each step
coefficient (`crysfft_boltz_`), built when the CrysFFT path is set up, and the stress
multipliers of each axis group and bond coefficient (`crysfft_multipliers_`), built by
`prepare_stress()` before the first segment loop. A cell change
(`update_laplacian_operator()`) waits for the device and rebuilds every table in place.
The engines take the table as an argument (`diffusion(in, out, table, stream)`,
`apply_multiplier(in, out, table, stream)`), so the propagator and stress loops only look
tables up and never build or allocate one; a missing key is an error.

**Backward Compatibility:**

```cpp
// Legacy class names (still supported)
typedef CudaRealTransform1D CudaDCT;
typedef CudaRealTransform1D CudaDST;
typedef CudaRealTransform2D CudaDCT2D;
typedef CudaRealTransform3D CudaDCT3D;
```

### 7.3 API Reference

```cpp
// Transform type enumeration
enum CudaTransformType {
    CUDA_DCT_1 = 0, CUDA_DCT_2 = 1, CUDA_DCT_3 = 2, CUDA_DCT_4 = 3,
    CUDA_DST_1 = 4, CUDA_DST_2 = 5, CUDA_DST_3 = 6, CUDA_DST_4 = 7
};

// 3D transform
class CudaRealTransform3D {
public:
    // Same type for all dimensions
    CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type);

    // Different types per dimension
    CudaRealTransform3D(int Nx, int Ny, int Nz,
                        CudaTransformType type_x,
                        CudaTransformType type_y,
                        CudaTransformType type_z);

    void execute(double* d_data);
    double get_normalization() const;
};
```

**Usage Example:**

```cpp
int Nx = 32, Ny = 32, Nz = 32;
double* d_data;
cudaMalloc(&d_data, sizeof(double) * Nx * Ny * Nz);

// DCT-2 in X/Y, DST-2 in Z (Neumann in XY, Dirichlet in Z)
CudaRealTransform3D transform(Nx, Ny, Nz, CUDA_DCT_2, CUDA_DCT_2, CUDA_DST_2);
transform.execute(d_data);
```

---

## 8. Performance Benchmarks

### 8.1 DCT/DST Numerical Accuracy

Relative error against FFTW (maximum absolute difference divided by the largest FFTW value; 3D transforms of uniform random input on 32³ and 64³ grids): the CUDA transforms (all types) and the MKL DCT/DST types 1-3 agree to machine precision (< 7×10⁻¹⁶); the MKL DCT-4 and DST-4 (TT interface) agree to 4×10⁻¹⁴ on 32³ and 7×10⁻¹³ on 64³. `MakhoulFFT` over either library agrees with the O(N²) definitions to 6×10⁻¹⁵, relative to the largest value, on grids of one to three axes with even, odd and unit lengths, every mix of cosine and sine axes, and real and complex fields (`TestCpuMakhoulTransform`). With periodic axes among the walls (Section 2.5), `MakhoulTransform` over either library agrees with the DFT and the series to 3×10⁻¹⁵ on 154 grids of two and three axes with every mix of periodic, cosine and sine axes, real fields taking the real part of the inverse of coefficients without symmetry. `CudaMakhoulTransform` agrees with them to 2×10⁻¹⁴ on these grids and 40 larger ones, and to 3×10⁻¹⁵ at the corners and 16 random points of two grids of more than two million points, where every grid-stride loop of its kernels takes more than one stride; it agrees with `MakhoulTransform` to 8×10⁻¹⁶ on all of them (`TestCudaMakhoulTransform`).

### 8.2 FFTW vs MKL Benchmark (CPU)

**Setup**: forward and backward transform of one real field through `PropagatorSolver.fourier` in double precision, cubic grids, one thread, best of ten runs of ten round trips; FFTW 3.3.11 and MKL 2025.0. Reflecting and absorbing boundaries take the same time (`MakhoulTransform`, Section 2.4).

| Boundary | Processor, library | 32³ | 64³ | 96³ |
|----------|--------------------|-----|-----|-----|
| Periodic | AMD EPYC 7543, FFTW | 0.19 ms | 2.0 ms | 8.5 ms |
| Periodic | AMD EPYC 7543, MKL | 0.31 ms | 2.9 ms | 11.4 ms |
| Periodic | Intel Xeon 6737P, FFTW | 0.16 ms | 1.7 ms | 7.9 ms |
| Periodic | Intel Xeon 6737P, MKL | 0.14 ms | 1.4 ms | 5.6 ms |
| Walls | AMD EPYC 7543, FFTW | 0.33 ms | 3.2 ms | 12.1 ms |
| Walls | AMD EPYC 7543, MKL | 0.47 ms | 3.9 ms | 15.1 ms |
| Walls | Intel Xeon 6737P, FFTW | 0.28 ms | 2.6 ms | 12.4 ms |
| Walls | Intel Xeon 6737P, MKL | 0.26 ms | 2.4 ms | 10.8 ms |

The processor decides which library leads. On the Intel Xeon 6737P MKL loads its AVX-512 kernels (`libmkl_avx512`). On the AMD EPYC 7543 it reports processor type 0 and loads its generic kernels (`libmkl_def`), whose DFT kernels use 128-bit vectors (a sample of the disassembly); `MKL_ENABLE_INSTRUCTIONS` and `MKL_CBWR` do not change this. One call of `compute_propagators` (AB diblock, `rqm4`, ds = 0.01, 64³, no space group, one thread) takes 1.4 s with FFTW and 1.8 s with MKL on the AMD processor and 1.3 s and 1.1 s on the Intel processor; with reflecting walls 2.2 s and 2.7 s on the AMD processor and 1.9 s and 1.8 s on the Intel processor. Among the CPU backends, `platform="auto"` takes `cpu-fftw` when a build has both libraries.

`MakhoulTransform` beats the r2r transforms of FFTW on the whole grid (one plan with `fftw_plan_r2r`, `FFTW_MEASURE`). The same transforms in a standalone program, round trip without the Python layer:

| Grid | AMD EPYC 7543: r2r plan | Reordered real FFT | Intel Xeon 6737P: r2r plan | Reordered real FFT |
|------|-------------------------|--------------------|----------------------------|--------------------|
| 32³ | 0.58 ms | 0.32 ms | 0.46 ms | 0.28 ms |
| 64³ | 5.1 ms | 3.2 ms | 4.2 ms | 2.6 ms |
| 96³ | 21.5 ms | 11.4 ms | 20.1 ms | 12.3 ms |
| 128³ | 60.6 ms | 29.1 ms | 49.2 ms | 30.8 ms |
| 192³ | 232 ms | 134 ms | 211 ms | 172 ms |

`MklRealTransform`, which the solvers do not use, provides DCT and DST of types 1 to 4 for internal use. `tests/BenchmarkMklFftw` compares it with FFTW plans made with `FFTW_ESTIMATE` (the solvers time their plans with `FFTW_MEASURE` or `FFTW_PATIENT`) on 3D transforms, single-threaded, on the AMD EPYC 7543:
- **DCT-IV, DST-IV**: MKL TT interface, O(N log N)
- **DCT-II, DCT-III, DST-II, DST-III**: FFT-based N-point algorithm with precomputed twiddle factors (Makhoul), O(N log N)
- **DCT-I, DST-I**: FFT-based via symmetric/antisymmetric extension (for N points: 2(N-1)- and 2(N+1)-point FFTs), O(N log N)

| Transform | 32³ | 48³ | 64³ |
|-----------|-----|-----|-----|
| DFT | 0.54× | 1.09× | 0.77× |
| DCT-2 | 0.80× | 1.23× | 1.04× |
| DCT-3 | 0.78× | 1.17× | 1.08× |
| DST-2 | 0.78× | 1.28× | 1.02× |
| DST-3 | 0.74× | 1.12× | 0.95× |

Speedup = FFTW time / MKL time (>1× means MKL is faster)

### 8.3 CrysFFT Benchmark (64³, Im-3m BCC)

**Setup**: one diffusion step of a real field in double precision (transform, multiplier, inverse transform), nx=[64,64,64], coeff=0.01, median of five runs of 200 iterations, NVIDIA A10 GPU; CPU: FFTW, one thread, AMD EPYC 7543 (`tests/TestCrysFFTBench 64 64 64 200 --cpu --obliquez`). The standard FFT runs as in the solvers, with a stored multiplier and `FFTW_MEASURE` plans. This measures the transforms alone; Section 8.5 gives the complete solver.

| Platform | Standard FFT | 3m physical | Pmmm physical |
|----------|-------------|-------------|---------------|
| CUDA | 0.131 ms | 0.056 ms (2.3x) | 0.090 ms (1.5x) |
| CPU | 1.86 ms | 0.43 ms (4.3x) | 0.33 ms (5.6x) |

### 8.4 ObliqueZ Benchmark (64³, Hexagonal P6/mmm)

**Setup**: as in Section 8.3, with γ=120° (same benchmark)

| Platform | Standard FFT | ObliqueZ |
|----------|-------------|----------|
| CUDA | 0.132 ms | 0.104 ms (1.3x) |
| CPU | 1.85 ms | 1.43 ms (1.3x) |

### 8.5 CrysFFT in the Complete Solver

**Setup**: time of `compute_propagators`, `compute_concentrations` and `get_concentration` in double precision with a space group, against the same solver without one on the same symmetric field: AB diblock with f = 0.3, ds = 0.01, nx=[64,64,64], median of repeated calls. NVIDIA A10 GPU; CPU: one thread, AMD EPYC 7543. Each range covers `rqm4`, `rk2` and the discrete chain model.

| Engine (space group, physical basis) | CUDA, 4 streams | CUDA, 1 stream | CPU-FFTW | CPU-MKL |
|--------------------------------------|-----------------|----------------|----------|---------|
| Recursive 3m (Im-3m and Ia-3d, m3) | 3.4-4.0x | 2.4-2.5x | 4.8-5.2x | 4.9-5.0x |
| Pmmm DCT (Im-3m and Pmmm, Pmmm) | 1.9-2.2x | 1.6x | 5.9-6.3x | 5.6-5.9x |
| ObliqueZ (P6/mmm, z-mirror) | 1.6-1.8x | 1.3x | 1.3-1.5x | 1.3-1.4x |

The gain depends on the grid. With `rqm4`, CUDA with 4 streams:

| Engine | 24³ | 32³ | 48³ | 64³ | 96³ | 128³ |
|--------|-----|-----|-----|-----|-----|------|
| Recursive 3m (Im-3m, m3) | 0.9x | 1.1x | 3.4x | 3.5x | 6.2x | 7.3x |
| ObliqueZ (P6/mmm, z-mirror) | 0.7x | 0.8x | 1.4x | 1.7x | 2.1x | 2.0x |

A space group therefore saves no time with CUDA below about 48³: the Gyroid example on 32³ takes the same time per SCFT iteration with and without it. The CPU backends gain at every size in this table (4 threads: 3.5-9x with the recursive 3m engine and 1.2-2.0x with ObliqueZ).

Stress: on a cubic cell the group gains 1.8-2.1x with CUDA (4 streams) and 5x on the CPU on the m3 basis, and 1.3x with CUDA and 4-6x on the CPU on the Pmmm basis. On a cell of three different lengths with the Pmmm basis the stress takes longer than on the full grid with CUDA (0.4-0.5x), and the CPU gains 1.6-2.1x. With ObliqueZ it takes about the same time (0.9x).

Whole SCFT iterations of the Gyroid example on 64³ with stress at every iteration, Ia-3d on the m3 basis against no group, 4 streams or threads: 2.6x with CUDA, 4.7x with MKL, 4.6x with FFTW.

### 8.6 Boxes that Mix Periodic Axes and Walls

**Setup (CPU)**: forward and backward transform of one real field through the Fourier transform of a box with periodic x and y and a reflecting z wall (P/P/R), or periodic x, a reflecting y wall and an absorbing z wall (P/R/A), in double precision, cubic grids, one thread, best of five runs of five round trips, through `MakhoulTransform` with periodic axes (Section 2.5).

| Box | Processor, library | 32³ | 64³ | 96³ |
|-----|--------------------|-----|-----|-----|
| P/P/R | AMD EPYC 7543, FFTW | 0.34 ms | 3.1 ms | 13 ms |
| P/P/R | AMD EPYC 7543, MKL | 0.47 ms | 4.0 ms | 17 ms |
| P/P/R | Intel Xeon 6737P, FFTW | 0.27 ms | 2.7 ms | 16 ms |
| P/P/R | Intel Xeon 6737P, MKL | 0.26 ms | 2.5 ms | 13 ms |
| P/R/A | AMD EPYC 7543, FFTW | 0.38 ms | 3.4 ms | 15 ms |
| P/R/A | AMD EPYC 7543, MKL | 0.51 ms | 4.4 ms | 18 ms |
| P/R/A | Intel Xeon 6737P, FFTW | 0.30 ms | 2.8 ms | 16 ms |
| P/R/A | Intel Xeon 6737P, MKL | 0.28 ms | 2.8 ms | 14 ms |

A complex field takes two real FFTs, one for its real and one for its imaginary part. The per-thread scratch holds about two grids of doubles.

**Setup (CUDA)**: forward and backward transform of one field in device memory, in the same boxes, through `CudaMakhoulTransform` (Section 2.5); CUDA 12.4.1, double precision, best of five runs of twenty calls timed with CUDA events. The scratch is that of the transform: its device buffers and the work area of its cuFFT plans.

| Box, field | GPU | 32³ | 64³ | 96³ | Scratch, 96³ |
|------------|-----|-----|-----|-----|--------------|
| P/P/R, real | A10 | 0.069 ms | 0.22 ms | 0.87 ms | 21 MiB |
| P/P/R, complex | A10 | 0.076 ms | 0.32 ms | 1.2 ms | 14 MiB |
| P/R/A, real | A10 | 0.078 ms | 0.28 ms | 1.1 ms | 21 MiB |
| P/R/A, complex | A10 | 0.086 ms | 0.40 ms | 1.4 ms | 14 MiB |
| P/P/R, real | A40 | 0.068 ms | 0.20 ms | 0.73 ms | 21 MiB |
| P/P/R, complex | A40 | 0.076 ms | 0.29 ms | 1.0 ms | 14 MiB |
| P/R/A, real | A40 | 0.077 ms | 0.25 ms | 0.90 ms | 21 MiB |
| P/R/A, complex | A40 | 0.084 ms | 0.35 ms | 1.2 ms | 14 MiB |

From 64³ on, a complex field takes 1.3-1.5x the time of a real field, since its Z2Z transforms twice the data of D2Z. On 32³ the kernel launches take most of the time. The scratch holds the samples and the half spectrum of a real field or the samples of a complex field, transformed in place, and one cuFFT work area for both directions.

---

## 9. References

1. Makhoul, J. "A Fast Cosine Transform in One and Two Dimensions." *IEEE Trans. Acoust., Speech, Signal Process.*, **28**(1), 27-34 (1980).

2. Martucci, S.A. "Symmetric convolution and the discrete sine and cosine transforms." *IEEE Trans. Signal Process.*, **42**(5), 1038-1051 (1994).

3. Britanak, V., Yip, P., & Rao, K.R. *Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations.* Academic Press (2007).

4. Qiang, Y. & Li, W. "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform." *Macromolecules* **53**, 9943-9952 (2020).

5. Ten Eyck, L. F. "Crystallographic Fast Fourier Transforms." *Acta Cryst.* **A29**, 183-191 (1973).

6. FFTW Manual - Real-even/odd DFTs: http://www.fftw.org/fftw3_doc/
