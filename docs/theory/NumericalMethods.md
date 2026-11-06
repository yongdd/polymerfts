# Numerical Methods for Polymer Field Theory

This document provides comprehensive documentation of the numerical methods available for solving the modified diffusion equation in polymer self-consistent field theory (SCFT), with emphasis on supporting both orthogonal and non-orthogonal crystal systems.

## Table of Contents

1. [Overview](#1-overview)
2. [Modified Diffusion Equation](#2-modified-diffusion-equation)
3. [Pseudo-Spectral Methods](#3-pseudo-spectral-methods)
4. [Real-Space Methods](#4-real-space-methods)
5. [Material Conservation](#5-material-conservation)
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Method Selection Guide](#7-method-selection-guide)
8. [Crystal System Constraints](#8-crystal-system-constraints)
9. [References](#9-references)

---

## 1. Overview

### Quick Reference

| Method | Order | Solver Type | Speed (vs CN-ADI2, Section 6) | Material Conservation |
|--------|-------|-------------|-------------------|----------------------|
| **RQM4** | 4th | Pseudo-spectral | 2.3x slower | Exact (10⁻¹⁶–10⁻¹⁴, round-off) |
| **RK2** | 2nd | Pseudo-spectral | 1.1x slower | Exact (10⁻¹⁶–10⁻¹⁴, round-off) |
| **CN-ADI2** | 2nd | Real-space | baseline | Exact (10⁻¹⁶–10⁻¹⁴, round-off) |

Discrete chains take **PS** (bond convolution through FFTs, the default, Section 3.4) or **RS** (the same bond summed on the grid, Section 4.3).

### Decision Guide

| Use Case | Recommended Method |
|----------|-------------------|
| Standard SCFT/FTS (periodic BC) | **RQM4** (default) |
| Fast iterations, prototyping | **RK2** |
| Discrete chains in strong fields (grafted or charged brushes) | **RS** |

### Runtime Selection

```python
from polymerfts import SCFT

params = {
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],
    "ds": 0.01,
    "chain_model": "continuous",
    "numerical_method": "rqm4",  # or "rk2", "cn-adi2"; discrete chains: "ps" or "rs"
    # ... other parameters
}

calculation = SCFT(params=params)
```

---

## 2. Modified Diffusion Equation

In polymer field theory, the chain propagator $`q(\mathbf{r}, s)`$ satisfies the modified diffusion equation:

```math
\frac{\partial q(\mathbf{r}, s)}{\partial s} = \frac{b^2}{6} \nabla^2 q(\mathbf{r}, s) - w(\mathbf{r}) q(\mathbf{r}, s)
```

where:
- $`q(\mathbf{r}, s)`$ is the chain propagator at position $`\mathbf{r}`$ and contour variable $`s`$
- $`b`$ is the statistical segment length
- $`w(\mathbf{r})`$ is the potential field
- $`s \in [0, 1]`$ parameterizes position along the chain contour

The initial condition is typically $`q(\mathbf{r}, 0) = 1`$ for a free chain end.

### Boundary Conditions

Both pseudo-spectral and real-space methods support the following boundary conditions:

| Boundary Condition | Description | Mathematical Form | Transform |
|-------------------|-------------|-------------------|-----------|
| **Periodic** | Cyclic boundary | $`q(0) = q(L)`$ | FFT |
| **Reflecting** | Neumann (zero flux) | $`\partial q / \partial n = 0`$ | DCT |
| **Absorbing** | Dirichlet (zero value) | $`q = 0`$ at boundary | DST |

> **Note**: Pseudo-spectral methods (RQM4, RK2, PS) cannot mix **periodic** with **non-periodic** boundary conditions in different directions (e.g., a thin film periodic in $`x, y`$ and reflecting in $`z`$); such configurations raise an error on all platforms and require a real-space method, `cn-adi2` for continuous or `rs` for discrete chains. Non-periodic conditions may be mixed with each other (e.g., reflecting in $`x`$ and absorbing in $`y`$), and uniform all-periodic, all-reflecting, or all-absorbing configurations are fully supported.

---

## 3. Pseudo-Spectral Methods

Pseudo-spectral methods use spectral transforms to solve the diffusion operator efficiently. The transform type depends on boundary conditions:
- **Periodic BC**: Fast Fourier Transform (FFT)
- **Reflecting BC**: Discrete Cosine Transform (DCT-II/III)
- **Absorbing BC**: Discrete Sine Transform (DST-II/III)

All pseudo-spectral methods use **cell-centered grids** where grid points are at $`x_i = (i + 0.5) \cdot dx`$ for $`i = 0, 1, ..., N-1`$, with boundaries at cell faces (not grid points). The fields of a hexagonal-axes space group sample the lattice axes a and b at the nodes instead ([Space Group](SpaceGroup.md#how-hexagonaltrigonal-groups-are-handled)); in a periodic box this is a translation, which the propagators do not see.

### 3.1 Lattice Vectors and Metric Tensors

#### Real-Space Lattice Vectors

For a general crystal system, the simulation box is defined by three lattice vectors $`\mathbf{a}`$, $`\mathbf{b}`$, $`\mathbf{c}`$ with lengths $`L_a`$, $`L_b`$, $`L_c`$ and angles:
- $`\alpha`$: angle between $`\mathbf{b}`$ and $`\mathbf{c}`$
- $`\beta`$: angle between $`\mathbf{a}`$ and $`\mathbf{c}`$
- $`\gamma`$: angle between $`\mathbf{a}`$ and $`\mathbf{b}`$

Using the standard crystallographic convention:
- $`\mathbf{a}`$ along the x-axis
- $`\mathbf{b}`$ in the xy-plane
- $`\mathbf{c}`$ in general position

The metric tensor $`G_{ij}`$ relates the dot product of lattice vectors:

```math
G = \begin{pmatrix} L_a^2 & L_a L_b \cos\gamma & L_a L_c \cos\beta \\ L_a L_b \cos\gamma & L_b^2 & L_b L_c \cos\alpha \\ L_a L_c \cos\beta & L_b L_c \cos\alpha & L_c^2 \end{pmatrix}
```

#### Reciprocal Lattice

The reciprocal metric tensor $`G^* = G^{-1}`$ gives the squared wavevector magnitude:

```math
|\mathbf{k}|^2 = (2\pi)^2 \sum_{i,j} G_{ij}^* n_i n_j = (2\pi)^2 \left( G_{11}^* n_1^2 + G_{22}^* n_2^2 + G_{33}^* n_3^2 + 2G_{12}^* n_1 n_2 + 2G_{13}^* n_1 n_3 + 2G_{23}^* n_2 n_3 \right)
```

For **orthogonal systems** ($`\alpha = \beta = \gamma = 90°`$), the cross-terms vanish:

```math
|\mathbf{k}|^2 = \left(\frac{2\pi n_1}{L_a}\right)^2 + \left(\frac{2\pi n_2}{L_b}\right)^2 + \left(\frac{2\pi n_3}{L_c}\right)^2
```

### 3.2 RK2 (Rasmussen-Kalosakas)

The **Rasmussen-Kalosakas (RK) algorithm** [2] uses symmetric operator splitting with 2nd-order accuracy.

> **Note**: RK here refers to Rasmussen-Kalosakas, not Runge-Kutta.

For the modified diffusion equation with Hamiltonian operator $`\hat{H} = -\frac{b^2}{6}\nabla^2 + w(\mathbf{r})`$, the RK algorithm approximates the propagator step using symmetric Strang splitting:

```math
e^{-\hat{H}\Delta s} \approx e^{-\frac{1}{2}w\Delta s} \cdot e^{\frac{b^2}{6}\nabla^2 \Delta s} \cdot e^{-\frac{1}{2}w\Delta s}
```

**Algorithm:**

1. **Potential half-step** (real space):

```math
q^* = \exp\left(-\frac{w \Delta s}{2}\right) q^n
```

2. **Diffusion step** (Fourier space):

```math
\hat{q}^{**} = \exp\left(-\frac{b^2 k^2 \Delta s}{6}\right) \hat{q}^*
```

3. **Potential half-step** (real space):

```math
q^{n+1} = \exp\left(-\frac{w \Delta s}{2}\right) q^{**}
```

**Properties:**
- **2nd-order accuracy** in $`\Delta s`$ (convergence order ≈ 2.0)
- Time-reversible (due to symmetric splitting)
- 2 FFTs per step
- Fastest pseudo-spectral method (2.0x less total run time than RQM4 at Ns = 1000 in the benchmark of Section 6)

> **Note**: RK2 for continuous chains is mathematically equivalent to the **N-bond model** for discrete chains described in Park et al. (2019) [5]. Both use the same Boltzmann factor $`e^{-b^2 k^2 \Delta s / 6}`$ in Fourier space.

### 3.3 RQM4 (Richardson Extrapolation)

**RQM4** (Ranjan-Qin-Morse 4th-order) [6] extends the RK algorithm to 4th-order accuracy using Richardson extrapolation.

**Algorithm:**

For one full step from $`s_n`$ to $`s_{n+1} = s_n + \Delta s`$:

1. **Full step**: Apply RK once with step size $`\Delta s`$:

```math
q_{full} = q_{RK}(s_{n+1}; \Delta s)
```

2. **Half steps**: Apply RK twice with step size $`\Delta s/2`$:

```math
q_{half} = q_{RK}(s_{n+1}; \Delta s/2)
```

3. **Richardson extrapolation**:

```math
q^{n+1} = \frac{4 q_{half} - q_{full}}{3}
```

**Properties:**
- **4th-order accuracy** in $`\Delta s`$ (convergence order ≈ 4.0)
- 6 FFTs per step (3 RK applications)
- Default method for highest accuracy

### 3.4 Discrete Chain Model

For discrete chains, the propagator is computed using **recursive integral equations** (Chapman-Kolmogorov equations) rather than solving a differential equation. This section describes PS (`numerical_method="ps"`, the default); RS (Section 4.3) sums the same bond on the grid.

**Units and Conventions:**
- **Unit length**: $`R_0 = bN^{1/2}`$, where $`b`$ is the statistical segment length and $`N`$ is the polymerization index
- **Contour step size**: $`\Delta s = 1/N`$
- A polymer chain has **N segments** (monomers) connected by **N-1 bonds**

The propagator evolution from segment $`i`$ to segment $`i+1`$ follows:

1. **Bond convolution** (Fourier space):

```math
\hat{q}^*(\mathbf{k}) = \hat{g}(\mathbf{k}) \cdot \hat{q}_i(\mathbf{k})
```

2. **Full-segment Boltzmann weight** (real space):

```math
q_{i+1}(\mathbf{r}) = \exp(-w(\mathbf{r}) \Delta s) \cdot q^*(\mathbf{r})
```

where $`\hat{g}(\mathbf{k}) = \exp\left(-\frac{b^2 |\mathbf{k}|^2 \Delta s}{6}\right)`$ is the Fourier transform of the Gaussian bond function.

### 3.5 Boltzmann Weight Calculation

For each wavevector index $`(n_1, n_2, n_3)`$:

```
k_sq = (2*pi)^2 * ( G_star_11 * n1^2 + G_star_22 * n2^2 + G_star_33 * n3^2
                  + 2 * G_star_12 * n1 * n2
                  + 2 * G_star_13 * n1 * n3
                  + 2 * G_star_23 * n2 * n3 )

boltz_bond[idx] = exp(-b^2 * k_sq * ds / 6)
```

For non-orthogonal cells, each index triple is first replaced by its alias $`(n_1 + a_1 N_1, n_2 + a_2 N_2, n_3 + a_3 N_3)`$ with the smallest $`|\mathbf{k}|^2`$, so that the discrete operator keeps the point-group symmetry of the cell. The search is exact: an alias whose $`|\mathbf{k}|^2`$ does not exceed that of the centered index satisfies $`|n_d + a_d N_d| \le \sqrt{f\,(G^{*-1})_{dd}}`$ with $`f`$ the metric form of the centered index, which bounds the $`a_d`$ to examine. On nearly isotropic grids this is $`a_d \in \lbrace -1, 0, 1 \rbrace`$; with strongly anisotropic grid spacing the minimum can lie further away.

---

## 4. Real-Space Methods

Real-space methods work on the grid instead of through spectral transforms: CN-ADI2 with finite differences for continuous chains, RS with a direct bond sum for discrete chains. They use **cell-centered grids** (same as pseudo-spectral methods) for consistent boundary condition handling.

### 4.1 Grid Discretization

Grid points are located at cell centers:

```math
x_i = \left(i + \frac{1}{2}\right) \Delta x, \quad i = 0, 1, \ldots, N-1
```

**Boundary condition implementation using ghost cells:**

| BC Type | Ghost Cell Value | Effect on Laplacian |
|---------|------------------|---------------------|
| **Periodic** | $`q_{-1} = q_{N-1}`$ | Cyclic tridiagonal system |
| **Reflecting** | $`q_{-1} = q_0`$ (symmetric) | Diagonal += off-diagonal |
| **Absorbing** | $`q_{-1} = -q_0`$ (antisymmetric) | Diagonal -= off-diagonal |

### 4.2 CN-ADI2 (Crank-Nicolson ADI)

The diffusion part of each step uses the Crank-Nicolson method, a semi-implicit scheme that is unconditionally stable:

```math
(I - \frac{\Delta s}{2} L) q^{n+1} = (I + \frac{\Delta s}{2} L) q^n
```

where $`L = \frac{b^2}{6}\nabla_h^2`$ is the discrete Laplacian operator. The field enters by symmetric splitting: the propagator is multiplied by $`e^{-w \Delta s/2}`$ before this diffusion step and again after it.

**ADI Splitting (3D):**

For 3D problems, the operator is split into three sequential 1D solves:

1. **X-direction sweep**: Solve tridiagonal system for $`q^*`$
2. **Y-direction sweep**: Solve tridiagonal system for $`q^{**}`$
3. **Z-direction sweep**: Solve tridiagonal system for $`q^{n+1}`$

Each direction requires solving a tridiagonal (or cyclic tridiagonal for periodic BC) system using:
- **Thomas algorithm**: For non-periodic boundaries ($`O(N)`$)
- **Sherman-Morrison formula**: For periodic boundaries ($`O(N)`$ with constant overhead)

**Properties:**
- **2nd-order accuracy** in both space and time
- Supports periodic, reflecting, and absorbing boundaries
- CUDA parallelizes across systems, solving thousands of tridiagonal systems simultaneously

The matrices do not change from step to step, because the field is split off, so the CUDA solver computes the Thomas factors of each axis once (with the Sherman-Morrison vector of periodic axes), and a solve is two sweeps over them. Each thread solves one line; the lines along the contiguous axis are first copied, a warp of them per block, into shared memory, so that their reads and writes coalesce like those of the other axes.

### 4.3 RS (Real-Space Bond Convolution)

`numerical_method="rs"` computes the bond of the discrete chain model (Section 3.4) as a sum on the grid:

```math
q_{i+1}(\mathbf{r}) = e^{-w(\mathbf{r})\Delta s} \sum_{\mathbf{r}'} g(\mathbf{r}-\mathbf{r}')\, q_i(\mathbf{r}')
```

In an orthogonal box the Gaussian bond is a product of one kernel per axis, so a bond is three one-dimensional stencils,

```math
q^{*}_j = g_0\, q_j + \sum_{d=1}^{H} g_d \left( q_{j-d} + q_{j+d} \right).
```

The kernel samples a Gaussian of variance $`\sigma^2 = b^2 \Delta s / 3`$ along each axis ($`b^2 \Delta s / 6`$ for the half bonds at chain ends and junctions), is summed to one and ends at $`H \approx 13.6\,\sigma / \Delta x`$, where it falls below $`10^{-40}`$ of its peak. Its width is set so that the sampled kernel has exactly the variance $`\sigma^2`$: on grids that resolve the bond this changes nothing, and on coarse grids it keeps the size of the chain (the plain samples are 3.6 % short at $`\Delta x = b`$). Points beyond the box come from the ghost cells of Section 4.1: periodic axes wrap, reflecting walls mirror the grid about the cell face, $`q_{-1-j} = q_j`$, and absorbing walls do so with the opposite sign, the symmetries of the DCT-II and DST-II. Each end of an axis has its own condition, and a kernel longer than the box is folded repeatedly.

**Why sum on the grid.** For real fields every term of the sum is a positive weight times $`q`$, so every value of $`q`$ keeps its own relative precision. An FFT returns every value with an error of about $`10^{-16}`$ of the largest value of $`q`$: a value $`10^{-50}`$ of the maximum is the sum of Fourier modes of order one that cancel, which double precision cannot carry. When the field changes by more than about 30 kT per chain between regions, the propagators span more than sixteen decades where the chains are. In a grafted polyelectrolyte brush the propagator from the free ends is $`10^{-60}`$ to $`10^{-150}`$ of its maximum inside the brush, so PS returns concentrations as negative as their maximum there, while RS keeps them positive and gives the same partition function from every segment. Cutting small values of $`q`$ or of $`\hat g(\mathbf{k})`$ does not help: the forward transform has already lost them.

**Stress.** The stress of a bond is $`\sum_{\mathbf r} q_1\, (\partial G / \partial L_a)\, q_2`$, where $`\partial G / \partial L_a`$ is the bond with its kernel along axis $`a`$ differentiated with respect to the box length at fixed grid offsets. The derivative includes the change of the width that keeps the variance on the grid, so the stress is the exact derivative of the RS free energy; where RS and PS agree, so do their stresses.

**Space groups.** The Gaussian of an orthogonal cell commutes with the operations of its space group, so with a space group RS keeps the propagators in the reduced basis and expands them to the full grid for each bond.

**Relation to PS.** The transform of the sampled kernel differs from $`\hat g(k) = e^{-\sigma^2 k^2/2}`$ by about $`e^{-\pi^2 \sigma^2 / (2 \Delta x^2)}`$ near the largest wavenumber of the grid, with $`\sigma`$ the standard deviation of a half bond. When $`\sigma`$ spans about three grid spacings this is $`10^{-18}`$, and RS and PS agree to round-off in random fields (`tests/TestRealSpaceDiscrete.py`); on coarser grids they are two discretizations of the same chain model.

**Cost.** The sum costs $`2H+1`$ multiply-adds per point and axis, with $`H`$ growing as $`\sigma/\Delta x`$, while an FFT costs about $`5\log_2 M`$ operations per point. Time of one propagator step of a discrete chain (N = 20, $`\sigma = 0.58\,b`$), single thread on an AMD EPYC 7543, double precision:

| Grid | $`\Delta x`$ | Kernel points per axis | Boundaries | PS, cpu-mkl | PS, cpu-fftw | RS |
|------|------|------|------|------|------|------|
| 64³ | $`b`$ | 17 | periodic | 2.7 ms | 2.3 ms | 3.2 ms |
| 64³ | $`b`$ | 17 | reflecting | 4.2 ms | 3.6 ms | 3.3 ms |
| 64³ | $`b/2`$ | 33 | periodic | 2.8 ms | 2.4 ms | 5.7 ms |
| 64³ | $`b/2`$ | 33 | reflecting | 4.2 ms | 3.6 ms | 5.7 ms |
| 128³ | $`b`$ | 17 | periodic | 28 ms | 26 ms | 29 ms |
| 128³ | $`b`$ | 17 | reflecting | 41 ms | 38 ms | 29 ms |

With walls, where the cosine and sine transforms cost more than a periodic FFT, RS is as fast as PS on grids of spacing $`b`$.

On CUDA a block copies a tile of the grid with its halos into shared memory once, and the pass along the first axis also multiplies by $`e^{-w\Delta s}`$ and the mask; a kernel of more than about 3400 points per axis for real fields, or 2000 for complex fields ($`\sigma \gtrsim 125\,\Delta x`$ or $`75\,\Delta x`$), does not fit: the solver refuses such a grid when it is built or the box changes. Time of one step of real fields on an NVIDIA A10, double precision (N = 20; N = 100 on the 1D grid):

| Grid | $`\Delta x`$ | Boundaries | PS | RS |
|------|------|------|------|------|
| 1600 | $`b/8`$ | reflecting | 0.075 ms | 0.014 ms |
| 512² | $`b`$ | reflecting | 0.28 ms | 0.078 ms |
| 64³ | $`b`$ | periodic | 0.15 ms | 0.11 ms |
| 64³ | $`b`$ | reflecting | 0.30 ms | 0.11 ms |
| 64³ | $`b/2`$ | periodic | 0.15 ms | 0.17 ms |
| 128³ | $`b`$ | periodic | 1.5 ms | 0.74 ms |
| 128³ | $`b`$ | reflecting | 3.5 ms | 0.74 ms |
| 256³ | $`b`$ | periodic | 11 ms | 5.7 ms |
| 256³ | $`b`$ | reflecting | 36 ms | 5.7 ms |

### 4.4 Limitations

Real-space methods have some limitations compared to pseudo-spectral methods:

1. **Stress computation**: Not implemented (CN-ADI2)
2. **Non-orthogonal cells**: Only orthogonal unit cells supported
3. **Spatial accuracy**: $`O(\Delta x^2)`$ vs spectral (exponential) accuracy (CN-ADI2)

---

## 5. Material Conservation

Material conservation requires that forward and backward partition functions are equal ($`Q_{forward} = Q_{backward}`$). This is ensured when the evolution operator $`U`$ satisfies the Hermiticity condition $`(VU)^\dagger = VU`$ where $`V`$ is the volume matrix.

| Method | Conservation | Mechanism |
|--------|--------------|-----------|
| **RQM4** | Exact | Symmetric splitting: $`e^{-w \cdot ds/2} \cdot e^{L \cdot ds} \cdot e^{-w \cdot ds/2}`$ |
| **RK2** | Exact | Same symmetric splitting as RQM4 |
| **CN-ADI2** | Exact | Symmetric Crank-Nicolson time stepping |
| **PS** | Exact | Symmetric bond (real, even $`\hat g(\mathbf k)`$) with $`e^{-w \cdot ds}`$ at each segment |
| **RS** | Exact | Symmetric bond (even kernel, wrapped or mirrored at the walls) with $`e^{-w \cdot ds}`$ at each segment |

The remaining mismatch is round-off and does not depend on the method. For an AB diblock ($`f = 0.4`$) on a 32³ grid with $`ds = 0.01`$ and random fields of standard deviation 5, the largest relative deviation of $`Q`$ along the chain (returned by `partition_errors()`) is $`1\times10^{-16}`$–$`3\times10^{-15}`$ on CUDA and about $`2\times10^{-14}`$ on cpu-mkl, for all five methods (discrete chains for PS and RS) and for periodic, reflecting and absorbing boundaries; the mean of the total concentration differs from 1 by at most $`1.1\times10^{-15}`$ (CUDA) and $`8\times10^{-15}`$ (cpu-mkl).

**Reference**: Yong & Kim, *Phys. Rev. E* **2017**, 96, 063312

---

## 6. Performance Benchmarks

The benchmarks below use the system of Fig. 1 from Song et al., *Chinese J. Polym. Sci.* **2018**, 36, 488-496 (gyroid of a continuous-chain AB diblock with f = 0.375 at χN = 18 on a 32³ grid), but in a fixed cubic box of L = 3.65 $`bN^{1/2}`$ instead of the stress-free cell used there (about 3.81 $`bN^{1/2}`$).

### Benchmark Configuration

- **Platform**: NVIDIA A10 GPU (CUDA), `OMP_MAX_ACTIVE_LEVELS=0` and `OMP_NUM_THREADS=4`, so the two propagators of each iteration are computed one after another
- **System**: AB diblock copolymer, continuous chains, $`b_A = b_B`$, Gyroid phase
- **Conditions**: f = 0.375, χN = 18, M = 32³, L = 3.65 (box fixed), Ns = 1/ds
- **Initial field**: gyroid level-set field as in `examples/block_copolymer/scft/Gyroid.py`, with amplitudes 0.3 and 0.1
- **Optimizer**: Anderson mixing, `max_hist` = 20, `start_error` = 10⁻², `mix_init` = `mix_min` = 0.02
- **SCFT convergence**: tolerance = 10⁻⁹ (reached after 240–300 iterations)
- **Max iterations**: 2000

### Free Energy vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | -0.477010930978363 | -0.476977370031310 | -0.476974360495475 | -0.476974130350877 | -0.476974114332803 | -0.476974113395157 |
| **RK2** | -0.475796762870488 | -0.476698716000712 | -0.476906701588300 | -0.476957357088977 | -0.476969930359763 | -0.476972400411139 |
| **CN-ADI2** | -0.477733629652859 | -0.478950812741258 | -0.479251268938022 | -0.479326309667460 | -0.479345088848207 | -0.479348787593391 |

### Error vs Contour Steps

**Pseudo-spectral methods** (reference value $`F_{\rm ref}`$ = -0.476974113087715):

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | 3.6818e-05 | 3.2569e-06 | 2.4741e-07 | 1.7263e-08 | 1.2451e-09 | 3.0744e-10 |
| **RK2** | 1.1774e-03 | 2.7540e-04 | 6.7411e-05 | 1.6756e-05 | 4.1827e-06 | 1.7127e-06 |

**Real-space method** (reference value $`F_{\rm ref}`$ = -0.479351354862727, which differs from the pseudo-spectral value because of the finite-difference spatial discretization):

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **CN-ADI2** | 1.6177e-03 | 4.0054e-04 | 1.0009e-04 | 2.5045e-05 | 6.2662e-06 | 2.5675e-06 |

### Convergence Order Analysis

Orders computed from the errors above, $`\log_2(\text{error}_{N_s} / \text{error}_{2N_s})`$:

| Ns transition | RQM4 order | RK2 order | CN-ADI2 order |
|---------------|------------|-----------|---------------|
| 40 → 80 | 3.50 | 2.10 | 2.01 |
| 80 → 160 | 3.72 | 2.03 | 2.00 |
| 160 → 320 | 3.84 | 2.01 | 2.00 |
| 320 → 640 | 3.79 | 2.00 | 2.00 |

### Execution Time Comparison

Total time of the SCFT setup and iterations:

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | 3.4 s | 6.0 s | 11.5 s | 22.6 s | 44.8 s | 69.4 s |
| **RK2** | 1.6 s | 2.9 s | 5.7 s | 11.4 s | 22.4 s | 34.9 s |
| **CN-ADI2** | 1.6 s | 2.8 s | 5.2 s | 10.0 s | 19.7 s | 30.5 s |

### Speedup Summary (at Ns=1000)

| Method | Speedup (vs CN-ADI2) |
|--------|----------------------|
| **RK2** | 1.1x slower |
| **RQM4** | 2.3x slower |
| **CN-ADI2** | baseline |

The ratios depend on the grid. One propagator computation of the same diblock (ds = 0.01, one stream, A10) takes 24, 88 and 869 ms with RQM4, 12, 30 and 314 ms with RK2, and 12, 30 and 261 ms with CN-ADI2 on 32³, 64³ and 128³ grids.

### Key Findings

1. **RK2 and CN-ADI2 are the fastest methods** - 2.0x and 2.3x less total run time than RQM4 at Ns = 1000, but only 2nd-order accurate
2. **RQM4 achieves 4th-order convergence** - converges to F = -0.47697411
3. **RK2 achieves 2nd-order convergence** - converges to F = -0.47697240 (differs by ~1.7e-6 from RQM4)
4. **CN-ADI2** converges to a different free energy (F = -0.47935) due to finite-difference spatial discretization error
5. **CN-ADI2** needs 2.1-2.3x less run time than RQM4 at every Ns of this benchmark (32³ grid), and 3.3x less per propagator computation at 128³

### Pseudo-Spectral vs Real-Space Discrepancy

The CN-ADI2 method converges to a free energy (F = -0.47935) that differs from pseudo-spectral methods (F = -0.47697) by approximately 0.5%. This arises from different spatial discretization approaches:

- **Pseudo-spectral**: Use spectral (Fourier) representation, achieving exponential convergence for smooth periodic fields
- **Real-space**: Use finite-difference approximations, introducing $`O(\Delta x^2)`$ spatial error

For the 32³ grid used in these benchmarks, the finite-difference spatial error dominates the contour discretization error.

**Implication**: When comparing free energies between pseudo-spectral and real-space solvers, ensure sufficient spatial resolution for the real-space method. For periodic boundary conditions, pseudo-spectral methods are more efficient and accurate.

---

## 7. Method Selection Guide

### Detailed Recommendations

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Standard SCFT/FTS (periodic BC) | **RQM4** | Fastest 4th-order, recommended default |
| Fast iterations, prototyping | **RK2** | Fastest pseudo-spectral method, lower accuracy |
| Brush with grafted delta-function | **CN-ADI2** | Handles sharp initial conditions smoothly |
| Discrete chains in strong fields (grafted or charged brushes) | **RS** | Positive concentrations where FFT round-off is amplified (Section 4.3) |
| Stress calculations | **RQM4**, **RK2**, **PS** or **RS** | All BCs supported (real fields); CN-ADI2 has no stress |
| Comparison with finite difference | **CN-ADI2** | Same discretization approach |

### Feature Comparison

| Feature | RQM4 | RK2 | CN-ADI2 | PS | RS |
|---------|------|-----|---------|----|----|
| Chain model | Continuous | Continuous | Continuous | Discrete | Discrete |
| Periodic BC | Yes (FFT) | Yes (FFT) | Yes | Yes (FFT) | Yes |
| Reflecting BC | Yes (DCT) | Yes (DCT) | Yes | Yes (DCT) | Yes |
| Absorbing BC | Yes (DST) | Yes (DST) | Yes | Yes (DST) | Yes |
| Mixed periodic/non-periodic BC | No² | No² | Yes | No² | Yes |
| Stress calculation | Yes³ | Yes³ | No | Yes³ | Yes³ |
| Complex fields | Yes | Yes | No | Yes | Yes |
| Non-orthogonal cells | Yes | Yes | No | Yes | No |
| Accuracy order | 4th | 2nd | 2nd | –¹ | –¹ |

¹ PS and RS compute the discrete chain model itself (bond convolution via the Chapman-Kolmogorov equation), not a contour discretization of the continuous chain.

² Mixing periodic with reflecting/absorbing boundary conditions in different directions raises an error for pseudo-spectral methods on all platforms; such periodic/non-periodic mixes require a real-space method (`cn-adi2` or `rs`). Purely non-periodic mixes (reflecting with absorbing) and uniform all-reflecting or all-absorbing configurations work with RQM4/RK2/PS.

³ `compute_stress()` supports periodic, reflecting, and absorbing boundary conditions (per-mode DCT/DST Parseval weights are folded into the spectral stress tables; for complex fields the coefficients of the real and imaginary parts are multiplied as complex numbers). Box optimization (`relax_cell=True`) works with non-periodic boundaries for the box lengths (angles stay at 90°).

---

## 8. Crystal System Constraints

Different crystal systems impose constraints on the lattice parameters:

### 8.1 Orthorhombic/Tetragonal/Cubic

```math
\alpha = \beta = \gamma = 90°
```

Cross-terms in $`|\mathbf{k}|^2`$ vanish. Off-diagonal stress components are zero at equilibrium.

**Constraints:**
- Orthorhombic: $`L_a \neq L_b \neq L_c`$ (3 independent lengths)
- Tetragonal: $`L_a = L_b \neq L_c`$ (2 independent lengths)
- Cubic: $`L_a = L_b = L_c`$ (1 independent length)

### 8.2 Hexagonal/Trigonal

```math
\alpha = \beta = 90°, \quad \gamma = 120°
```

Cross-term $`G_{12}^*`$ is non-zero.

**Constraints:**
- $`L_a = L_b \neq L_c`$ (2 independent lengths)
- $`\gamma`$ fixed at 120°

### 8.3 Monoclinic

```math
\alpha = \gamma = 90°, \quad \beta \neq 90°
```

Cross-term $`G_{13}^*`$ is non-zero.

**Constraints:**
- $`L_a`$, $`L_b`$, $`L_c`$ all independent (3 lengths)
- $`\beta`$ is a free parameter (1 angle)
- The angle derivative $`\partial H/\partial\beta`$ (stress array index 4) drives $`\beta`$ optimization

> **Note on the stress convention**: `get_stress()` returns lattice-parameter derivatives directly, ordered as $`[\partial H/\partial L_1, \partial H/\partial L_2, \partial H/\partial L_3, \partial H/\partial\gamma, \partial H/\partial\beta, \partial H/\partial\alpha]`$ in 3D (in 2D, $`\partial H/\partial\gamma`$ is stored at index 2). Box relaxation steps each optimized parameter along the negative of its derivative. The "off-diagonal stress components" discussed in this section are these angle derivatives, not Cartesian stress-tensor components.

### 8.4 Triclinic

```math
\alpha, \beta, \gamma \text{ arbitrary}
```

All cross-terms may be non-zero.

**Constraints:**
- All 6 parameters ($`L_a`$, $`L_b`$, $`L_c`$, $`\alpha`$, $`\beta`$, $`\gamma`$) are independent
- All off-diagonal stress components may be non-zero

---

## 9. References

1. Matsen, M. W. "The standard Gaussian model for block copolymer melts." *J. Phys.: Condens. Matter* **14**, R21 (2002).

2. Rasmussen, K. O. & Kalosakas, G. "Improved numerical algorithm for exploring block copolymer mesophases." *J. Polym. Sci. B: Polym. Phys.* **40**, 1777 (2002).

3. Tzeremes, G., Rasmussen, K. O., Lookman, T. & Saxena, A. "Efficient computation of the structural phase behavior of block copolymers." *Phys. Rev. E* **65**, 041806 (2002).

4. Arora, A., Morse, D. C., Bates, F. S. & Dorfman, K. D. "Accelerating self-consistent field theory of block polymers in a variable unit cell." *J. Chem. Phys.* **146**, 244902 (2017).

5. Park, S. J., Yong, D., Kim, Y. & Kim, J. U. "Numerical implementation of pseudo-spectral method in self-consistent mean field theory for discrete polymer chains." *J. Chem. Phys.* **150**, 234901 (2019).

6. Ranjan, A., Qin, J. & Morse, D. C. "Linear response and stability of ordered phases of block copolymer melts." *Macromolecules* **41**, 942-954 (2008).

7. Stasiak, P. & Matsen, M. W. "Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases." *Eur. Phys. J. E* **34**, 110 (2011).

8. Song, J. Q., Liu, Y. X. & Zhang, H. D. "An efficient algorithm for self-consistent field theory calculations of complex self-assembled structures of block copolymer melts." *Chinese J. Polym. Sci.* **36**, 488-496 (2018).

9. Yong, D. & Kim, J. U. "Finite volume method for self-consistent field theory of polymers: Material conservation and application." *Phys. Rev. E* **96**, 063312 (2017).

10. Crank, J. & Nicolson, P. "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Proc. Cambridge Phil. Soc.* **43**, 50-67 (1947).

11. Peaceman, D. W. & Rachford, H. H. "The numerical solution of parabolic and elliptic differential equations." *J. Soc. Indust. Appl. Math.* **3**, 28-41 (1955).
