# Self-Consistent Field Theory with Electrostatic Correlations

Mean-field theory gives each ion the energy $`z\psi`$ of its charge in the mean potential. Electrostatic-correlation-augmented SCFT (EC-SCFT; Duan, Agrawal and Wang, Reference 1) adds the self-energy of each charge: the free energy of the correlation hole that the other charges form around it. The self-energy grows as $`z^2`$ and with the local ionic strength, so multivalent ions are drawn into dense charged regions. In a polyelectrolyte brush this collapses the brush further than Poisson-Boltzmann theory does. The theory follows the Gaussian variational treatment of electrolytes of Z.-G. Wang (Reference 2). [BrushMultivalentIons.py](../../examples/charged/ecscft/BrushMultivalentIons.py) solves the equations of Section 5.

## Table of Contents

1. [Model](#1-model)
2. [Variational Free Energy](#2-variational-free-energy)
3. [Uniform State](#3-uniform-state)
4. [Intrachain Correlations](#4-intrachain-correlations)
5. [Planar Brush](#5-planar-brush)
6. [Relation to the Published Equations](#6-relation-to-the-published-equations)
7. [Range of Validity](#7-range-of-validity)
8. [References](#8-references)

## 1. Model

Chains of $`N`$ segments (P) mix with a solvent (S) of one site each; segments and solvent sites have the volume $`v`$ and fill the space. Mobile ions $`K`$ come from a reservoir and take no volume. Species $`K`$ carries the valence $`z_K`$, and every charge is spread over a Gaussian of Born radius $`a_K`$,

```math
h_K(k)=e^{-k^2a_K^2/(2\pi)},\qquad h_K(r)=\frac{1}{2\sqrt2\,a_K^3}\,e^{-\pi r^2/(2a_K^2)} .
\qquad\text{(1)}
```

Alone in a medium of permittivity $`\epsilon=1/(4\pi\ell_B)`$ (energies in $`k_BT`$), such a charge has the Born energy $`z_K^2\ell_B/(2a_K)`$. In the Gaussian smearing $`\Gamma(k)=e^{-\sigma^2k^2/2}`$ of the [electrostatic interactions of polymerfts](PolymerFieldTheory.md), $`\sigma=a_K/\sqrt\pi`$. The charge density and the electrostatic energy are

```math
\hat\rho_e=\sum_Kz_K\,h_K*\hat\rho_K,\qquad
H_e=\frac12\iint d\mathbf r\,d\mathbf r'\,\hat\rho_e(\mathbf r)\,C(\mathbf r,\mathbf r')\,\hat\rho_e(\mathbf r'),\qquad
-\nabla\cdot\left[\epsilon\nabla C(\mathbf r,\mathbf r')\right]=\delta(\mathbf r-\mathbf r'),
\qquad\text{(2)}
```

where the sum runs over the segments and the ions, and the other interaction is the Flory-Huggins contact $`\chi v\int\hat\rho_P\hat\rho_S`$.

## 2. Variational Free Energy

A Hubbard-Stratonovich transformation turns $`H_e`$ into a fluctuating potential $`\Psi`$, and the contact interaction and the incompressibility into fields $`W_P`$, $`W_S`$ and $`\Xi`$. EC-SCFT keeps the latter at their saddle points ($`W_K\to-i\omega_K`$, $`\Xi\to i\xi`$) and replaces the action of $`\Psi`$ by a Gaussian reference with mean $`-i\psi`$ and covariance $`G`$:

```math
L_{\rm ref}[\Psi]=\frac12\iint d\mathbf r\,d\mathbf r'\,\left[\Psi(\mathbf r)+i\psi(\mathbf r)\right]G^{-1}(\mathbf r,\mathbf r')\left[\Psi(\mathbf r')+i\psi(\mathbf r')\right] .
\qquad\text{(3)}
```

The average of the coupling $`e^{-iz_Kh_K*\Psi}`$ over this reference gives every charge the Boltzmann factor $`e^{-z_Kh_K*\psi-u_K}`$, in the mean potential averaged over the charge, with the self-energy

```math
u_K(\mathbf r)=\frac{z_K^2}{2}\iint d\mathbf r'd\mathbf r''\,h_K(\mathbf r-\mathbf r')\,G(\mathbf r',\mathbf r'')\,h_K(\mathbf r''-\mathbf r) .
\qquad\text{(4)}
```

For grafted chains, a solvent in equilibrium with pure solvent and ions from a reservoir, the variational free energy is

```math
\begin{aligned}
F_{\rm var}={}&-\ln\tilde Q_P-\lambda_S\int d\mathbf r\,e^{-\omega_S}-\sum_K\lambda_K\int d\mathbf r\,e^{-z_K\textcolor{red}{h_K*}\psi-u_K}\\
&+\int d\mathbf r\left[\chi v\rho_P\rho_S-\omega_P\rho_P-\omega_S\rho_S\textcolor{red}{+\xi}\left(\rho_P+\rho_S-\frac1v\right)+\frac12\psi\nabla\cdot(\epsilon\nabla\psi)\right]\\
&+\frac12\iint d\mathbf r\,d\mathbf r'\left[C^{-1}-G^{-1}\right](\mathbf r,\mathbf r')\,G(\mathbf r',\mathbf r)-\frac12\ln\frac{\det G}{\det C} .
\end{aligned}
\qquad\text{(5)}
```

$`\tilde Q_P`$ is the partition function of the grafted chains in the field $`V_P=\omega_P+z_Ph_P*\psi+u_P`$ per segment (Section 5), $`\lambda_S=1/v`$, and $`\lambda_K`$ is the activity of ion $`K`$. A chain in $`V_P`$ treats its segments as independent charges; Section 4 gives the intrachain part this leaves out. The sign of $`\xi`$ is that of the main text of Reference 1, opposite to its Supplemental Material (Section 6).

The stationary point of $`F_{\rm var}`$ with respect to $`\psi`$ gives the Poisson equation with Boltzmann ions, using $`\delta(-\ln\tilde Q_P)/\delta V_P=\rho_P`$ and the symmetry of $`h_K`$:

```math
-\nabla\cdot(\epsilon\nabla\psi)=z_P\textcolor{red}{h_P*}\rho_P+\sum_Kz_K\textcolor{red}{h_K*}\rho_K,\qquad \rho_K=\lambda_K\,e^{-z_K\textcolor{red}{h_K*}\psi-u_K} .
\qquad\text{(6)}
```

The stationary points with respect to $`\rho_P`$, $`\rho_S`$, $`\xi`$ and $`\omega_S`$ give

```math
\omega_P=\chi v\rho_S\textcolor{red}{+\xi},\qquad \omega_S=\chi v\rho_P\textcolor{red}{+\xi},\qquad \rho_P+\rho_S=\frac1v,\qquad \rho_S=\lambda_Se^{-\omega_S} .
\qquad\text{(7)}
```

With $`\phi=v\rho_P`$ the solvent gives $`\omega_S=-\ln(1-\phi)`$, so $`\xi=-\ln(1-\phi)-\chi\phi`$ and

```math
\omega_P=\chi(1-2\phi)-\ln(1-\phi) .
\qquad\text{(8)}
```

The stationary point with respect to $`G`$ collects $`\frac12(C^{-1}-G^{-1})`$ from the last line of Eq. (5) and $`\frac12z_K^2\,h_K\rho_Kh_K`$ from each $`u_K`$ of Eq. (4), segments included:

```math
-\nabla\cdot\left[\epsilon\nabla G(\mathbf r,\mathbf r')\right]+\sum_Kz_K^2\iint d\mathbf r_1d\mathbf r_2\,\textcolor{red}{h_K(\mathbf r-\mathbf r_1)}\,\rho_K(\mathbf r_1)\,\textcolor{red}{h_K(\mathbf r_1-\mathbf r_2)}\,G(\mathbf r_2,\mathbf r')=\delta(\mathbf r-\mathbf r') .
\qquad\text{(9)}
```

Every charge screens with its own spread; the red factors are what Eq. S25 of Reference 1 replaces by delta functions (Section 6). Wang (Reference 2, Appendix A) derives Eq. (9) and then replaces $`h_K`$ by a delta function in the screening term, because that term stays finite for point charges:

```math
-\nabla\cdot\left[\epsilon\nabla G(\mathbf r,\mathbf r')\right]+2I(\mathbf r)\,G(\mathbf r,\mathbf r')=\delta(\mathbf r-\mathbf r'),\qquad I=\frac12\sum_Kz_K^2\rho_K .
\qquad\text{(10)}
```

Eq. (10) is the Green function of the later work of the field, Reference 3 and Eq. S25 of Reference 1. The self-energy of Eq. (4) keeps the spread, which removes the divergence of a point charge. Eq. (10) is not the stationary point of the free energy whose self-energy is Eq. (4): the screening cloud is made of point charges around a spread charge. The difference is small when the screening length is much longer than the radius, $`\kappa a\ll1`$, and grows with $`\kappa a`$ (Section 3).

The action is complex, so the stationary point is not a bound on the free energy, and the theory gives no estimate of its own error.

## 3. Uniform State

In a uniform medium, Eq. (9) gives $`G(k)=4\pi\ell_B/\left(k^2+\sum_K\kappa_K^2h_K(k)^2\right)`$, where $`\kappa_K^2=4\pi\ell_B\sum z^2\rho`$ sums over the species of radius $`a_K`$. The self-energy of a charge of radius $`a_J`$ is

```math
u_J=z_J^2\,\frac{\ell_B}{\pi}\int_0^\infty dk\,\frac{k^2h_J(k)^2}{k^2+\sum_K\kappa_K^2\textcolor{red}{h_K(k)^2}} .
\qquad\text{(11)}
```

At $`\kappa_K=0`$ it is the Born energy. The $`u_J`$ are the derivatives $`\partial f/\partial\rho_J`$ of the free energy density

```math
f=\frac12\int\frac{d^3k}{(2\pi)^3}\,\ln\left(1+\frac{\sum_K\kappa_K^2h_K(k)^2}{k^2}\right),
\qquad\text{(12)}
```

the Gaussian (one-loop) free energy of the electrostatic fluctuations of charges with these spreads, which CL-FTS of the same model approaches at weak coupling ([Electrolytes](../validation/Electrolytes.md)). A function $`f`$ with these derivatives exists because the mixed derivatives are symmetric in the species $`J`$ and $`K`$:

```math
\frac{\partial u_J}{\partial\rho_K}=-4\ell_B^2z_J^2z_K^2\int_0^\infty dk\,\frac{k^2h_J(k)^2h_K(k)^2}{\left(k^2+\sum_L\kappa_L^2h_L(k)^2\right)^2}=\frac{\partial u_K}{\partial\rho_J} .
```

With the point-charge screening of Eq. (10), $`h_K^2`$ is missing from this numerator, and the mixed derivatives differ when $`a_J\ne a_K`$: for charges of different radii no free energy has those self-energies as its derivatives. With one radius $`a`$ for all charges and $`\kappa^2=\sum_K\kappa_K^2`$, Eq. (11) becomes

```math
u_J=z_J^2\,\frac{\ell_B}{2a}\,u(a\kappa),\qquad u(x)=\frac{2}{\pi}\int_0^\infty dt\,\frac{t^2e^{-t^2/\pi}}{t^2+x^2e^{-t^2/\pi}} .
\qquad\text{(13)}
```

The screening of Eq. (10) sets the red factor $`h_K=1`$ in the denominator of Eq. (11) and gives the closed form of Reference 3 (Eq. 6) and of Eq. S33 of Reference 1:

```math
u(x)=1-x\,e^{x^2/\pi}\,\mathrm{erfc}\frac{x}{\sqrt\pi} .
\qquad\text{(14)}
```

Point charges screen also at wavelengths shorter than the radius, so Eq. (14) gives the larger gain $`1-u`$. The gain of Eq. (13) is smaller by 2 % at $`a\kappa=0.1`$ and by 11 % at $`a\kappa=1.3`$, the value in a brush collapsed by trivalent ions of radius 0.25 nm.

## 4. Intrachain Correlations

The segments of one chain fluctuate together, which Eq. (5) leaves out. Their electrostatic correlation through $`G`$ is the intrachain energy

```math
H^{\rm ex}=\frac12\sum_i\sum_{j\ne i}U(\mathbf R_i,\mathbf R_j),\qquad
U(\mathbf r,\mathbf r')=z_P^2\iint d\mathbf r_1d\mathbf r_2\,h_P(\mathbf r-\mathbf r_1)\,G(\mathbf r_1,\mathbf r_2)\,h_P(\mathbf r_2-\mathbf r'),
\qquad\text{(15)}
```

summed over the pairs of segments $`i`$, $`j`$ of each chain. Reference 1 takes it at first order (first cumulant) about the chain in $`V_P`$, whose partition function is $`\tilde Q_P^0`$:

```math
\ln\tilde Q_P\approx\ln\tilde Q_P^0-\left\langle H^{\rm ex}\right\rangle_0 .
\qquad\text{(16)}
```

Through $`U`$, this term adds to the stationary point with respect to $`G`$ the intrachain pair density of the segment charges; the left side of Eq. (9) gains $`\int d\mathbf r''K_{\rm ex}(\mathbf r,\mathbf r'')\,G(\mathbf r'',\mathbf r')`$ with

```math
K_{\rm ex}(\mathbf r,\mathbf r')=z_P^2\sum_{\rm chains}\sum_i\sum_{j\ne i}\left\langle h_P(\mathbf r-\mathbf R_i)\,h_P(\mathbf R_j-\mathbf r')\right\rangle_0 .
\qquad\text{(17)}
```

For point charges $`K_{\rm ex}=2I_{\rm ex}`$ of Eq. S26 of Reference 1, whose published form carries an extra factor $`G(\mathbf r,\mathbf r')`$; the rescaling $`z\to z/s`$, $`\epsilon\to\epsilon/s^2`$, $`G\to s^2G`$, which leaves the physics and every other term of the equation unchanged, shows that factor to be spurious. The segment density that follows from Eq. (16) involves correlations of three segments of a chain, so no field per segment reproduces it; Eq. 3b of Reference 1, the energy of a segment in the intrachain correlations, counts each pair with a factor ½ and gives half the density response when it is used as a field (Section 6).

Reference 1 drops $`\langle H^{\rm ex}\rangle_0`$ in its brush calculations, and with it $`K_{\rm ex}`$, its derivative with respect to $`G`$. Each segment is then screened like a free ion, with the self-energy $`u_P`$ of Eq. (4) in $`V_P`$. Section 7 gives the size of what this leaves out.

## 5. Planar Brush

Chains are grafted at $`\sigma_g`$ chains per area with the grafting weight $`g(z)`$ ($`\int g\,dz=1`$) near a wall at $`z=0`$. As in Reference 1, a chain is discrete: $`N`$ beads, each a charge $`z_P=-1`$, joined by Gaussian bonds of length $`b`$. The propagators of the beads $`i=1,\dots,N`$ in the field $`w`$ per bead are

```math
q_1(z)=g(z)\,e^{-w(z)},\quad q_{i+1}(z)=e^{-w(z)}\int dz'\,\Phi(z-z')\,q_i(z');\qquad
q_N^\dagger(z)=e^{-w(z)},\quad q_{i-1}^\dagger(z)=e^{-w(z)}\int dz'\,\Phi(z-z')\,q_i^\dagger(z'),
\qquad\text{(18)}
```

with the bond $`\Phi(z)=\sqrt{3/(2\pi b^2)}\,e^{-3z^2/(2b^2)}`$ along $`z`$, the wall absorbing and the far end reflecting. The bead density and the partition function of the grafted chains of a wall area $`A`$ are

```math
\rho_P(z)=\frac{\sigma_g}{Q}\sum_{i=1}^Nq_i(z)\,q_i^\dagger(z)\,e^{w(z)},\qquad Q=\int dz\,g(z)\,q_1^\dagger(z),\qquad \ln\tilde Q_P=\sigma_g A\ln Q,
\qquad\text{(19)}
```

so that $`\int\rho_Pdz=\sigma_g N`$ ([Chain Propagators](ChainPropagators.md)). The first beads then lie in $`g(z)\,q_1^\dagger(z)/Q`$: the grafting points move within the width of $`g`$. With fixed grafting points, as in Eq. S30 of Reference 1, the forward propagator would start from $`g(z)\,e^{-w(z)}/q_1^\dagger(z)`$ and $`\ln\tilde Q_P=\sigma_g A\int dz\,g(z)\ln q_1^\dagger(z)`$. The ions come from a solution of concentrations $`c_K^b`$ at the far end, where $`\psi=0`$; their activities are $`\lambda_K=c_K^b\,e^{u_K^b}`$, with $`u_K^b`$ the self-energy in the solution. Eqs. (6)-(8) become, for the field per bead, the ions and the potential,

```math
w(z)=\chi(1-2\phi)-\ln(1-\phi)+z_P\textcolor{red}{h_P*}\psi+z_P^2\bar U_P,\qquad
\rho_K(z)=c_K^b\,e^{-z_K\textcolor{red}{h_K*}\psi-z_K^2\bar U_K},
\qquad\text{(20)}
```

```math
\frac{d^2\psi}{dz^2}=-4\pi\ell_B\left(\sum_Kz_K\textcolor{red}{h_K*}\rho_K+z_P\textcolor{red}{h_P*}\rho_P\right),\qquad \frac{d\psi}{dz}(0)=0,\qquad \psi=0\ \text{in the solution},
\qquad\text{(21)}
```

where $`\bar U_K`$ is the self-energy per $`z_K^2`$ relative to the solution and $`h_K*`$ averages along $`z`$ over the charge, a Gaussian of standard deviation $`a_K/\sqrt\pi`$ reflected at the wall and at the far end. The brush height is $`h=2\langle z\rangle`$ of $`\rho_P`$. Without self-energy, $`\bar U_K=0`$, these are Poisson-Boltzmann theory of the smeared charges with the self-consistent field of the chains.

Reference 1 solves Eq. (10) for $`G`$, split into a local part and a long-range part that carries the variation of the ionic strength. The local form used here takes for $`G`$ at each point that of a uniform medium (Eq. (11), with the screening of Eq. (9)), at the partial ionic strengths around the charge:

```math
U_J(z)=\frac{u_J\left(\{\langle I_K\rangle_J(z)\}\right)-u_J^b}{z_J^2},\qquad \bar U_J=\langle U_J\rangle_J,
\qquad\text{(22)}
```

where $`I_K=\frac12\sum z^2\rho`$ sums over the charges of radius $`a_K`$, segments included, $`\langle\cdot\rangle_J`$ averages over a Gaussian of standard deviation $`a_J`$ along $`z`$, and $`u_J^b`$ is the value in the solution. Each species feels its self-energy averaged as its ionic strengths are.

With one radius $`a`$ for all charges, $`f`$ of Eq. (12) depends on the ionic strength alone, and the local form replaces the last line of Eq. (5) by the correlation free energy

```math
F_c=A\int dz\,f\big(\langle I\rangle(z)\big),\qquad
\frac{1}{A}\frac{\delta F_c}{\delta\rho_J(z)}=\frac{z_J^2}{2}\left\langle f'\big(\langle I\rangle\big)\right\rangle(z)=\left\langle u_J\big(\langle I\rangle\big)\right\rangle(z),
\qquad\text{(23)}
```

because the Gaussian average is symmetric. The self-energy that every species feels is then the derivative of one correlation free energy, and Eqs. (20)-(22) are the stationary point of the mean-field grand potential (Eq. (5) without $`u_K`$ and without its last line) plus $`F_c`$. With several radii, the same holds when all species are averaged with one kernel, $`F_c=A\int dz\,f\left(\{\langle I_K\rangle(z)\}\right)`$. The example averages each species over its own radius, as the validation runs did; Eq. (22) is then a closure of the same accuracy rather than such a stationary point.

The example takes $`g(z)\propto e^{-z^2/(2w_g^2)}`$ with $`w_g=0.5`$ nm and the solution of the experiment of Reference 4 (Na⁺, Y³⁺ and NO₃⁻ at an ionic strength of 6 mM), iterates $`w`$ and the three $`U_J`$ (segments, monovalent ions, trivalent ions) with damped steps and Anderson mixing, solves Eq. (21) by Newton's method in each step, and Eq. (18) by `rs`, which sums the bond on the grid.

## 6. Relation to the Published Equations

The equations of this page differ from those printed in Reference 1 and its Supplemental Material (SI) where the table says; in the displayed equations the terms that differ are red. The last column gives what the difference does to [BrushMultivalentIons.py](../../examples/charged/ecscft/BrushMultivalentIons.py).

| Here | Reference 1 | Difference | Effect on the example |
|---|---|---|---|
| Eq. (5) | SI Eq. S8 | S8 is the exponent of the partition function, not the action: as an action the ion term has the opposite sign | None |
| Eqs. (5), (6), (20), (21) | SI Eq. S11 and the equations that follow from it | The SI takes the mean potential at the centre of each charge, $`e^{-z_K\psi-u_K}`$, with point charges in the Poisson equation; the Gaussian average of $`e^{-iz_Kh_K*\Psi}`$ gives $`h_K*\psi`$, and the Poisson equation then has the charges spread | `"point"` takes the published form; at the radii of the example at most 0.001 in $`\log_{10}(h/h_6)`$ |
| Eqs. (5), (7) | SI Eqs. S11, S27a, S27b, S28, S32 | The SI writes $`-\xi`$, opposite to the main text and to its own ion weights; here $`+\xi`$ | None: $`\xi`$ is eliminated in Eq. (8) |
| Eq. (7) | SI Eqs. S27a, S27b | The fluctuation terms lack the derivative of $`\epsilon`$ with respect to the densities | None: $`\epsilon`$ is uniform |
| Eq. (9) | SI Eq. S25, main text Eq. 5 | S25 screens with point charges ($`h_K\to\delta`$, Eq. (10)) without saying so, which is not the stationary point of S11; Eq. (9) is | The default `screening = "spread"`; `"point"` gives the published form |
| Eqs. (11), (13) | SI Eq. S33 | S33 is the uniform state of point-charge screening, Eq. (14) | At the radii of the example, 0.007–0.013 in $`\log_{10}(h/h_6)`$ |
| Eq. (17) | SI Eq. S26 | The extra factor $`G(\mathbf r,\mathbf r')`$ is removed | None: the term is dropped, as in Reference 1 |
| Section 4 | Main text Eq. 3b, SI Eq. S27c | Eq. 3b is an energy per segment with a factor ½ for pairs; used as a field it gives half the density response, and no field per segment is exact | None: the term is dropped |
| Section 4 | SI Eqs. S21–S22 | The intrachain propagator $`g(\mathbf r,\mathbf r';i,j)`$ needs the start $`g(\mathbf r,\mathbf r';i,i)=e^{-V_P(\mathbf r)}\delta(\mathbf r-\mathbf r')`$, which the SI does not state | None: the term is dropped |
| Section 4 | SI Eqs. S29–S32 | The intrachain term of a grafted chain needs $`\langle\ln Q\rangle`$ over the grafting points, where $`\ln\langle Q\rangle`$ is used | None: the term is dropped |
| Eq. (19) | SI Eq. S30 | The start of the forward propagator at a grafting point lacks the factor $`e^{-V_P}`$ of the first bead | None: $`q_1=g\,e^{-w}`$ carries it, and the grafting points move within $`g(z)`$ |

The main text writes Eqs. 3a and 5 with the wrong integration variables; SI Eqs. S12 and S25 are right. The scaling analysis of Reference 1 (main text Eqs. 6–9, SI Eqs. S34–S42) does not enter these equations. In it, S40 is twice too large (with S41b, S41c and lines 2–3 of Eq. 8), the dilute correlation free energy has $`\pi/24`$ for $`1/(24\pi)`$, line 4 of Eq. 8 vanishes at the $`\chi=1/2`$ used for the figures, the balance stated for S41a does not give its result, S42 lacks a factor $`\rho_b`$, and Fig. 3 puts $`\rho_b`$ in mol/L into Eq. 9.

## 7. Range of Validity

- **Gaussian closure.** Only the electrostatic potential fluctuates, at second order. The coupling of a trivalent ion, $`z^2\ell_B\kappa`$, reaches about 30 in a brush collapsed by trivalent ions of radius 0.25 nm, where $`1/\kappa`$ is shorter than the radius; there the closure is not controlled.
- **Intrachain correlations.** Section 4 drops $`\langle H^{\rm ex}\rangle_0`$. For an unperturbed Gaussian chain the intrachain self-energy of a segment is comparable to the free-segment self-energy at $`\kappa\approx1`$ nm⁻¹ and smaller at the high ionic strength of a collapsed brush. A random-phase estimate with the chain form factor makes the correlation pressure 12–48 % more negative. The first-cumulant closure of Eq. (16) itself needs a small $`H^{\rm ex}`$, which fails at low ionic strength.
- **Density fluctuations.** The segments and ions screen as if they could move freely. In a dense incompressible layer ($`\phi\approx0.7`$) this overestimates $`\kappa`$ by about 8 % and the correlation gain by about 4 %.
- **Local form.** Eq. (22) answers a change of ionic strength at the same point only and selects no length. The layers that Reference 1 finds in a collapsed brush at small cation radii, and lateral structure, need the Green function of Eq. (9) or (10); with Eq. (9) the layers are weaker and appear only below a smaller radius ([validation](../validation/BrushMultivalentIons.md#fig-4-of-duan-et-al)).

- **Against CL-FTS.** At the radii of the example (Y³⁺ 0.66 nm) CL-FTS of the brush, which keeps the fluctuations that these approximations close or leave out, collapses it 0.012–0.041 further in $`\log_{10}(h/h_6)`$ ([validation](../validation/BrushMultivalentIons.md#complex-langevin-simulations)).

## 8. References

1. C. Duan, N. R. Agrawal and R. Wang, Electrostatic Correlation Augmented Self-Consistent Field Theory and Its Application to Polyelectrolyte Brushes, *Phys. Rev. Lett.* **2025**, 134, 048101, and its Supplemental Material.
2. Z.-G. Wang, Fluctuation in electrolyte solutions: The self energy, *Phys. Rev. E* **2010**, 81, 021501.
3. N. R. Agrawal and R. Wang, Self-Consistent Description of Vapor-Liquid Interface in Ionic Fluids, *Phys. Rev. Lett.* **2022**, 129, 228001.
4. J. Yu, J. Mao, G. Yuan, S. Satija, Z. Jiang, W. Chen and M. Tirrell, Structure of Polyelectrolyte Brushes in the Presence of Multivalent Counterions, *Macromolecules* **2016**, 49, 5609.
