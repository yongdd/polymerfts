# Self-Consistent Field Theory with Electrostatic Correlations

This document derives the theory of Duan, Agrawal and Wang (Reference 1), which adds the fluctuations of the electrostatic field to SCFT through a Gaussian variational reference. Each charged species then carries a self-energy set by its surroundings. The published equations are given here with corrections, listed in [Section 12](#12-corrections-to-the-published-equations).

[BrushMultivalentIons1D.py](../BrushMultivalentIons1D.py) solves the planar working equations of [Section 6.1](#61-working-equations); `BrushMultivalentIonsLateral.py`, in git at commit `2796421c`, solves them in 2D and 3D, and `BrushMultivalentIonsGreen1D.py` solves the Green function of S25 in the planar brush. [Validation of the brush with multivalent ions](MultivalentBrush.md) compares them with the paper and with experiment. The [standard model](StandardBrush.md) solves the planar equations with N12 and the ions of a neutral solution, and compares them with experiment.

Numbered equations carry the number of the original: `7`, `S26`, where S marks the Supplemental Material. An asterisk marks an equation that differs from the paper. `N` marks an equation that is not in the paper. Inside an equation, red marks what was changed or added; a removed term is gone, so a red note says what it was. The last column of the table in Section 12 gives which. N equations are new as a whole and stay black, apart from the coefficient in N7.

## Table of Contents

1. [Model](#1-model)
2. [Action and Variational Free Energy](#2-action-and-variational-free-energy)
3. [Green Function and Self-Energy](#3-green-function-and-self-energy)
4. [Chain](#4-chain)
5. [Mean-Field Equations](#5-mean-field-equations)
6. [Grafted Brush](#6-grafted-brush)
7. [Uniform State](#7-uniform-state)
8. [High-Salt Expansion](#8-high-salt-expansion)
9. [Low-Salt Ion Exchange](#9-low-salt-ion-exchange)
10. [Height Scaling](#10-height-scaling)
11. [Range of Validity](#11-range-of-validity)
12. [Corrections to the Published Equations](#12-corrections-to-the-published-equations)
13. [References](#13-references)

## 1. Model

$`n_P`$ chains of $`N_P`$ monomers with valency $`z_P`$, solvent S, $`n_C=-z_Pn_PN_P`$ monovalent counterions C, and salt ions $`\pm`$ with valency $`z_\pm`$. Polymers and C are canonical. Solvent and salt are in equilibrium with a reservoir at salt concentration $`\rho_b`$. For a $`z_+{:}1`$ salt the reservoir has $`\rho_+^b=\rho_b`$ and $`\rho_-^b=z_+\rho_b`$.

Charge is spread over a Gaussian of Born radius $`a_K`$. The paper states only "Gaussian"; this width is the one that gives its $`u(x)`$:

```math
h_K(k)=e^{-k^2a_K^2/(2\pi)},\qquad h_K(r)=\frac{1}{2\sqrt2\,a_K^3}\,e^{-\pi r^2/(2a_K^2)} .
\tag{N1}
```

The Hamiltonian, charge density and Coulomb operator are

```math
H=\chi v\int d\mathbf r\,\hat\rho_P\hat\rho_S+\frac12\iint d\mathbf r\,d\mathbf r'\,\hat\rho_e(\mathbf r)C(\mathbf r,\mathbf r')\hat\rho_e(\mathbf r'),
\qquad \hat\rho_e=\sum_{K=P,C,\pm}z_Kh_K*\hat\rho_K,
\tag{S2}
```

```math
-\nabla\cdot\left[\epsilon(\mathbf r)\nabla C(\mathbf r,\mathbf r')\right]=\delta(\mathbf r-\mathbf r') .
\tag{S3}
```

$`\epsilon`$ is scaled by $`\beta e^2`$. For uniform permittivity $`\epsilon=1/(4\pi\ell_B)`$. Incompressibility is $`\sum_Kv_K\hat\rho_K=1`$ with $`v_P=v_S=v`$.

## 2. Action and Variational Free Energy

After the Hubbard-Stratonovich transformation, $`Z=\int e^{-L}`$ with

```math
\begin{aligned}
L={}&-\lambda_PQ_P-\lambda_SQ_S-\ln N_C\\
&+\int d\mathbf r\left[\chi v\rho_P\rho_S-iW_P\rho_P-iW_S\rho_S-i\Xi\left(\rho_P+\rho_S-v^{-1}\right)-\frac12\Psi\nabla\cdot(\epsilon\nabla\Psi)\right]\\
&-\sum_{K=C,\pm}\lambda_K\int d\mathbf r\,e^{-iz_Kh_K*\Psi\textcolor{red}{+iv_K\Xi/v}}\qquad\textcolor{red}{\text{overall sign reversed}} .
\end{aligned}
\tag{S8*}
```

The electrostatic field gets the Gaussian reference action

```math
L_{\rm ref}[\Psi]=\frac12\iint d\mathbf r\,d\mathbf r'\,[\Psi(\mathbf r)+i\psi(\mathbf r)]\,G^{-1}(\mathbf r,\mathbf r')\,[\Psi(\mathbf r')+i\psi(\mathbf r')],
\tag{S10}
```

and the other fields take saddle points $`W_K\to-i\omega_K`$, $`\Xi\to i\xi`$. This sign of $`\xi`$ differs from the SI. It makes every species feel $`+v_K\xi/v`$, as in the main text and in the SI's ion weight. The smearing of the mean potential is neglected, $`h_K*\psi\to\psi`$, as in the SI.

```math
\begin{aligned}
F_{\rm var}={}&-\lambda_PQ_P-\lambda_SQ_S\\
&+\int d\mathbf r\left[\chi v\rho_P\rho_S-\omega_P\rho_P-\omega_S\rho_S\textcolor{red}{+\xi}\left(\rho_P+\rho_S-v^{-1}\right)+\frac12\psi\nabla\cdot(\epsilon\nabla\psi)\right]\\
&-\sum_{K=C,\pm}\lambda_K\int d\mathbf r\,e^{-z_K\psi-u_K^0-v_K\xi/v}
+\frac12\iint d\mathbf r\,d\mathbf r'\left[C^{-1}-G^{-1}\right]G-\frac12\ln\frac{\det G}{\det C} .
\end{aligned}
\tag{S11*}
```

## 3. Green Function and Self-Energy

The self-energy of one charge from its own spread is

```math
u_K^0(\mathbf r)=\frac{z_K^2}{2}\iint d\mathbf r'd\mathbf r''\,h_K(\mathbf r-\mathbf r')\,G(\mathbf r',\mathbf r'')\,h_K(\mathbf r''-\mathbf r) .
\tag{S12}
```

$`\delta F_{\rm var}/\delta G=0`$ gives

```math
-\nabla_{\mathbf r}\cdot\left[\epsilon\nabla_{\mathbf r}G(\mathbf r,\mathbf r')\right]+2I_0(\mathbf r)G(\mathbf r,\mathbf r')+\int d\mathbf r''\,2I_{ex}(\mathbf r,\mathbf r'')G(\mathbf r'',\mathbf r')=\delta(\mathbf r-\mathbf r'),
\tag{S25}
```

```math
I_0(\mathbf r)=\frac12\sum_{K=P,C,\pm}z_K^2\rho_K(\mathbf r),
```

S25 takes $`h_K\to\delta`$ in the screening, as the paper does without saying so. The stationary point of S11 keeps the spread, because $`u_K^0`$ of S12 contains it, and the local term of S25 is then

```math
2I_0(\mathbf r)\,\delta(\mathbf r-\mathbf r')\;\to\;\sum_{K=P,C,\pm}z_K^2\int d\mathbf r''\,h_K(\mathbf r-\mathbf r'')\,\rho_K(\mathbf r'')\,h_K(\mathbf r''-\mathbf r') .
\tag{N11}
```

The equations below keep the form of the paper, with screening by point charges. The nonlocal part is the intrachain pair density, with no Green function inside it:

```math
I_{ex}(\mathbf r,\mathbf r')=\frac{z_P^2}{2}\sum_{l}\sum_i\sum_{j\ne i}\left\langle\delta(\mathbf r-\mathbf R_{l,i})\,\delta(\mathbf r'-\mathbf R_{l,j})\right\rangle\qquad\textcolor{red}{\text{no factor }G(\mathbf r,\mathbf r')} .
\tag{S26*}
```

The published S26 has an extra factor $`G(\mathbf r,\mathbf r')`$. Dimensions cannot show the error, because $`G`$ is dimensionless in these units, but a rescaling can. The physics does not change under $`z\to z/s`$, $`\epsilon\to\epsilon/s^2`$, $`G\to s^2G`$. Every term of S25 is invariant, while a factor $`G`$ in $`I_{ex}`$ would pick up $`s^2`$.

In propagators,

```math
\begin{aligned}
I_{ex}(\mathbf r,\mathbf r')=\frac{z_P^2}{2}\lambda_P\sum_ie^{V_P(\mathbf r;i)}\Bigg[&\sum_{j=i+1}^{N_P}q(\mathbf r;i)\,g(\mathbf r,\mathbf r';i,j)\,e^{V_P(\mathbf r';j)}q^\dagger(\mathbf r';j)\\
+&\sum_{j=1}^{i-1}q(\mathbf r';j)\,e^{V_P(\mathbf r';j)}g^\dagger(\mathbf r',\mathbf r;j,i)\,q^\dagger(\mathbf r;i)\Bigg]\qquad\textcolor{red}{\text{no factor }G(\mathbf r,\mathbf r')} .
\end{aligned}
```

## 4. Chain

The electrostatic self-energy of a chain has a one-body part $`u_P^0`$ and a two-body part

```math
H^{\rm ex}=\sum_i\sum_{j\ne i}\frac{z_P^2}{2}\iint d\mathbf r'd\mathbf r''\,h_P(\mathbf R_i-\mathbf r')G(\mathbf r',\mathbf r'')h_P(\mathbf r''-\mathbf R_j)
\equiv\frac12\sum_i\sum_{j\ne i}U(\mathbf R_i,\mathbf R_j).
\tag{S14}
```

The reference is the one-body Hamiltonian

```math
H^0=-\sum_i\ln\Phi(\mathbf R_{i+1}-\mathbf R_i)+\sum_iV_P^0(\mathbf R_i),\qquad V_P^0=\omega_P+z_P\psi+u_P^0 .
\tag{S15}
```

The first-cumulant closure of S16–S17 is a statement about the partition function:

```math
\ln Q_P\approx\ln Q^0-\left\langle H^{\rm ex}\right\rangle_0 .
\tag{N2}
```

The reference propagators are

```math
q_0(\mathbf r;i)=e^{-V_P^0(\mathbf r)}\int d\mathbf r'\,\Phi(\mathbf r-\mathbf r')\,q_0(\mathbf r';i-1),\qquad q_0(\mathbf r;1)=e^{-V_P^0(\mathbf r)},
\tag{S21a}
```

```math
q_0^\dagger(\mathbf r;i)=e^{-V_P^0(\mathbf r)}\int d\mathbf r'\,\Phi(\mathbf r-\mathbf r')\,q_0^\dagger(\mathbf r';i+1),\qquad q_0^\dagger(\mathbf r;N_P)=e^{-V_P^0(\mathbf r)} .
\tag{S21b}
```

The intrachain function $`g_0`$ follows S22 with the initial condition, which the SI does not state and which S20 and S26 require,

```math
g_0(\mathbf r,\mathbf r';i,i)=e^{-V_P^0(\mathbf r)}\,\delta(\mathbf r-\mathbf r').
```

With it, the extra energy per monomer is S20 and Eq. 3b:

```math
u_P^{\rm ex}(\mathbf r;i)=\frac{z_P^2}{2}\left[\sum_{j=1}^{i-1}\frac{\int d\mathbf r'\,q_0(\mathbf r';j)e^{V_P^0(\mathbf r')}G(\mathbf r',\mathbf r)\,g_0^\dagger(\mathbf r',\mathbf r;j,i)}{q_0(\mathbf r;i)}+\sum_{j=i+1}^{N_P}\frac{\int d\mathbf r'\,g_0(\mathbf r,\mathbf r';i,j)G(\mathbf r,\mathbf r')e^{V_P^0(\mathbf r')}q_0^\dagger(\mathbf r';j)}{q_0^\dagger(\mathbf r;i)}\right].
\tag{S20}
```

This is an energy: $`\sum_iu_P^{\rm ex}`$ gives $`\langle H^{\rm ex}\rangle_0`$. The factor ½ counts each pair once, so S20 must not be used as the potential in the propagators. A one-body potential has no such factor, as in Hartree theory.

The density is the derivative of N2. Writing $`\hat\delta_k(\mathbf r)=\delta(\mathbf r-\mathbf R_k)`$,

```math
\rho_P(\mathbf r)=\rho_P^0(\mathbf r)-\frac{n_P}{2}\sum_k\sum_i\sum_{j\ne i}\iint d\mathbf r_1d\mathbf r_2\,U(\mathbf r_1,\mathbf r_2)
\left[\left\langle\hat\delta_k(\mathbf r)\hat\delta_i(\mathbf r_1)\hat\delta_j(\mathbf r_2)\right\rangle_0-\left\langle\hat\delta_k(\mathbf r)\right\rangle_0\left\langle\hat\delta_i(\mathbf r_1)\hat\delta_j(\mathbf r_2)\right\rangle_0\right].
\tag{N3}
```

For independent particles N3 reduces to a one-body potential $`2u^{\rm ex}`$. For a connected chain it involves three-monomer correlations, and no one-body potential reproduces it exactly.

## 5. Mean-Field Equations

Fields:

```math
\omega_P=\chi v\rho_S\textcolor{red}{+\xi}-\frac12\frac{\partial\epsilon}{\partial\rho_P}(\nabla\psi)^2,\qquad
\omega_S=\chi v\rho_P\textcolor{red}{+\xi}-\frac12\frac{\partial\epsilon}{\partial\rho_S}(\nabla\psi)^2 .
\tag{S27*}
```

Solvent and incompressibility:

```math
\rho_S=\lambda_Se^{-\omega_S},\qquad \sum_{K=P,S,C,\pm}v_K\rho_K=1 .
\tag{S27d-e}
```

Poisson equation:

```math
-\nabla\cdot(\epsilon\nabla\psi)=\sum_{K=P,C,\pm}z_K\rho_K .
\tag{S23}
```

Mobile ions:

```math
\rho_C=\frac{n_C}{Q_C}\,e^{-z_C\psi-u_C^0-v_C\xi/v},\qquad Q_C=\int d\mathbf r\,e^{-z_C\psi-u_C^0-v_C\xi/v},\qquad
\rho_\pm=\lambda_\pm e^{-z_\pm\psi-u_\pm^0-v_\pm\xi/v} .
\tag{S24}
```

## 6. Grafted Brush

Chains are tethered at $`\mathbf r_l`$ with the first monomer fixed. The density and the forward initial condition are

```math
\rho_P(\mathbf r)=\sum_i\tilde q(\mathbf r;i)\,e^{V_P(\mathbf r;i)}\,q^\dagger(\mathbf r;i),\qquad
\tilde q(\mathbf r;1)=\sum_l\frac{\delta(\mathbf r-\mathbf r_l)\,\textcolor{red}{e^{-V_P(\mathbf r;1)}}}{q^\dagger(\mathbf r_l;1)} .
\tag{S30*}
```

Each bead then integrates to $`n_P`$. The free energy is

```math
\begin{aligned}
F_{\rm var}={}&-\ln\tilde Q_P-n_C\ln Q_C-\lambda_SQ_S\\
&+\int d\mathbf r\left[\chi v\rho_P\rho_S-\omega_P\rho_P-\omega_S\rho_S\textcolor{red}{+\xi}\left(\rho_P+\rho_S-v^{-1}\right)+\frac12\psi\nabla\cdot(\epsilon\nabla\psi)\right]\\
&-\sum_{K=\pm}\lambda_K\int d\mathbf r\,e^{-z_K\psi-u_K^0-v_K\xi/v}
+\frac12\iint d\mathbf r\,d\mathbf r'\left[C^{-1}-G^{-1}\right]G-\frac12\ln\frac{\det G}{\det C},
\end{aligned}
\tag{S32*}
```

```math
\ln\tilde Q_P=\sum_l\ln q^\dagger(\mathbf r_l;1).
\tag{S31}
```

### 6.1 Working Equations

Figs. 1–3 of Reference 1 use a planar brush with $`v_C=v_\pm=0`$, $`\chi=1/2`$, uniform $`\epsilon`$, equal radii $`a`$ and $`u_P^{\rm ex}=0`$. The paper does not say whether they keep $`I_{ex}`$. Dropping $`u_P^{\rm ex}`$ consistently also drops $`I_{ex}`$, because $`I_{ex}`$ is its derivative with respect to $`G`$, and Reference 2 writes $`G`$ with the local ionic strength alone. The equations below do the same. Fig. 4 of the paper adds lateral structure and a smaller cation, which these planar equations with equal radii do not cover.

Incompressibility with $`\phi=v\rho_P`$ gives, with $`\lambda_S=1/v`$,

```math
\xi=-\ln(1-\phi)-\chi\phi,\qquad
\omega_P=\chi(1-2\phi)-\ln(1-\phi),\qquad
V_P=\omega_P+z_P\psi+z_P^2U .
\tag{N4}
```

The ions and the potential are

```math
\rho_+=\rho_b\,e^{-z_+\psi-z_+^2U},\qquad
\rho_-=z_+\rho_b\,e^{\psi-U},\qquad
\rho_C=\frac{\sigma N}{\int_0^Ldz\,e^{-\psi-U}}\,e^{-\psi-U},
```

```math
\frac{d^2\psi}{dz^2}=-4\pi\ell_B\left(z_P\rho_P+\rho_C+z_+\rho_+-\rho_-\right).
```

$`U(z)`$ is the self-energy per unit $`z^2`$ relative to the reservoir. The paper solves S25 for it, with $`G=G_s+G_l`$: a local part, and a long-range part that carries the varying ionic strength. The local-density closure uses the uniform result of [Section 7](#7-uniform-state) at each $`z`$, which leaves $`G_l`$ out:

```math
U(z)=\frac{\ell_B}{2a}\left[u\big(a\kappa(z)\big)-u(a\kappa_b)\right],\qquad
\kappa^2=4\pi\ell_B\sum_Kz_K^2\rho_K,\qquad
\kappa_b^2=4\pi\ell_Bz_+(z_++1)\rho_b .
\tag{N5}
```

[BrushMultivalentIons1D.py](../BrushMultivalentIons1D.py) solves these equations with continuous chains, and smooths the ionic strength over $`a`$ before N5, in place of the long-range part of $`G`$. The field of each ion takes $`u`$ smoothed in the same way, so that the equations remain the stationary point of a grand potential. [Validation of the brush with multivalent ions](MultivalentBrush.md) lists what it assumes beyond them.

The conservation of C means that the box length $`L`$ is part of the model, see [Section 9](#9-low-salt-ion-exchange).

## 7. Uniform State

For a uniform state S25 without $`I_{ex}`$ gives $`G(k)=1/(\epsilon k^2+2I_0)`$, so

```math
u_K=\frac{z_K^2}{8\pi\epsilon a_K}\,u(a_K\kappa),\qquad u(x)=1-xe^{x^2/\pi}\operatorname{erfc}\frac{x}{\sqrt\pi},
\tag{S33}
```

```math
u'(x)=\frac{2x}{\pi}-\left(1+\frac{2x^2}{\pi}\right)e^{x^2/\pi}\operatorname{erfc}\frac{x}{\sqrt\pi},\qquad \kappa^2=\frac{2I_0}{\epsilon}.
```

With N11 the uniform state has $`G(k)=1/[\epsilon(k^2+\kappa^2h^2(k))]`$, and with the Gaussian spread of N1 the same prefactor holds with

```math
u(x)=\frac{2}{\pi}\int_0^\infty dt\,\frac{t^2e^{-t^2/\pi}}{t^2+x^2e^{-t^2/\pi}} .
\tag{N12}
```

The gain $`1-u`$ is then 2 % smaller at $`x=0.1`$ and 11 % smaller at $`x=1.3`$, the collapsed state of a brush with $`z_+=3`$.

With equal radii define $`w=u(a\kappa)/(8\pi\epsilon a)`$ and $`\mathcal I=2I_0=\sum_Kz_K^2\rho_K`$. The charging integral is the correlation free energy density:

```math
f_c(\mathcal I)=\int_0^{\mathcal I}d\mathcal I'\,w(\mathcal I').
```

The osmotic pressure of the ions, with polymer counted in $`\mathcal I`$, is

```math
P_{\rm ion}=\sum_{K=+,-,C}\rho_K+\mathcal Iw(\mathcal I)-f_c(\mathcal I).
\tag{N6}
```

Relative to the reservoir this is S39:

```math
f_{\rm ion}=\sum_{K=\pm}\left(\rho_K-\rho_K^b\right)-\sum_{K=P,\pm}\int_0^1d\tau\,\rho_K\left[u_K(\tau)-u_K\right]+\sum_{K=\pm}\int_0^1d\tau\,\rho_K^b\left[u_K^b(\tau)-u_K^b\right].
\tag{S39}
```

In the dilute limit,

```math
f_{\rm ion}=\sum_{K=\pm}\left(\rho_K-\rho_K^b\right)-\frac{\kappa^3-\kappa_b^3}{\textcolor{red}{24\pi}}.
\tag{N7}
```

## 8. High-Salt Expansion

This section holds only for

```math
\rho_b\gg\alpha\rho_P,
\tag{S34}
```

where $`\alpha`$ is the charged fraction, $`\alpha=1`$ in the paper, and counterions C are neglected. Define the effective relative charge

```math
\Gamma=1-\frac{z_+(\rho_+-\rho_b)}{\alpha\rho_P},\qquad \Delta\rho=\frac{\Gamma\alpha\rho_P}{z_+},
\tag{S37}
```

```math
\rho_+=\rho_b+\frac{\alpha\rho_P}{z_+}-\Delta\rho,\qquad\rho_-=z_+(\rho_b-\Delta\rho).
\tag{S36}
```

Eliminating $`\psi`$ from S24 gives

```math
\frac{\rho_+}{\rho_b}\left(\frac{\rho_-}{z_+\rho_b}\right)^{z_+}=\exp\left[-(u_+-u_+^b)-z_+(u_--u_-^b)\right],
\tag{S35}
```

and linearising, with polymer included in the ionic strength,

```math
\frac{(z_++1)\Gamma-1}{\Gamma-1}=-\frac{z_+(z_++1)\kappa_b}{16\pi\epsilon}\,u'(a\kappa_b)\equiv R .
\tag{7}
```

With $`\Delta\rho=\Gamma\alpha\rho_P/z_+`$ the left side of S35 gives $`(\alpha\rho_P/z_+)[1-(z_++1)\Gamma]/\rho_b`$. The ionic strength counts the polymer as $`I_0`$ does, so $`\delta(2I)=(z_++1)\alpha\rho_P(1-\Gamma)`$, and with $`\epsilon\kappa_b^2=z_+(z_++1)\rho_b`$ this gives $`\delta\kappa=\alpha\rho_P(1-\Gamma)\kappa_b/(2z_+\rho_b)`$. The right side is $`-z_+(z_++1)(8\pi\epsilon)^{-1}u'(a\kappa_b)\,\delta\kappa`$. Dividing out the common factor gives Eq. 7.

Solved for $`\Gamma`$,

```math
\Gamma=\frac{1-R}{z_++1-R},
```

with the mean-field limit $`\Gamma=1/(z_++1)`$ at $`R=0`$.

Let $`\Delta u=(u_--u_-^b)`$ be the self-energy shift per unit $`z^2`$. The linearised S24 gives $`\Delta u=\psi+\Delta\rho/\rho_b`$ and

```math
\psi=-\frac{\alpha\rho_P}{z_+^2(z_++1)\textcolor{red}{\rho_b}}\left[(z_+^2-1)\Gamma+1\right].
\tag{S42*}
```

The Fig. 3 axis in these units is

```math
\frac{\psi_S}{\sigma Nv/h}=-\frac{(z_+^2-1)\Gamma+1}{z_+^2(z_++1)\,\rho_bv},\qquad \textcolor{red}{\rho_b\text{ in nm}^{-3}=0.6022\,\rho_b[\mathrm M]}.
\tag{9*}
```

The potential reverses at $`\Gamma=-1/(z_+^2-1)`$.

For the pressure, $`f_{\rm ion}=xg'(x)-g(x)+g(0)`$ with $`x=\alpha\rho_P`$ and $`g'(x)=-\psi+\Delta u=\Delta\rho/\rho_b=\Gamma x/(z_+\rho_b)`$. To second order,

```math
f_{\rm ion}=\frac{\alpha^2\rho_P^2\,\Gamma}{\textcolor{red}{2}z_+\rho_b}.
\tag{S40*}
```

The mean-field limit shows the factor 2: $`\Gamma=1/(z_++1)`$ and $`\sum_K\delta\rho_K=\tfrac12z_+(z_++1)\rho_b\psi^2`$ give $`\alpha^2\rho_P^2/[2z_+(z_++1)\rho_b]`$. For $`z_+=1`$ this is the standard $`\rho_P^2/(4\rho_b)`$.

## 9. Low-Salt Ion Exchange

S34 fails here, so $`\Gamma`$ of Eq. 7 does not apply. The brush holds C at the lowest salt and multivalent cations after them. Nothing in this section is linearised in $`\rho_P/\rho_b`$.

Take three uniform regions: reservoir, far box with counterion density $`c`$, brush with self-energy shifts $`\Delta u_B`$ and $`\Delta u_f`$ relative to the reservoir, per unit $`z^2`$. For $`c\gg z_+\rho_b`$ anions neutralise the far box:

```math
e^{\psi_f}=\frac{c}{z_+\rho_b}\,e^{\Delta u_f}.
```

In the brush $`\rho_+=\rho_be^{-z_+\psi_B-z_+^2\Delta u_B}`$ and $`\rho_C=c\,e^{-(\psi_B-\psi_f)-(\Delta u_B-\Delta u_f)}`$. Eliminating $`\psi_B`$,

```math
\rho_+=\rho_b\left(\frac{z_+\rho_b\,\rho_C}{c^2}\right)^{z_+}\exp\left[-z_+(z_+-1)\Delta u_B-2z_+\Delta u_f\right].
\tag{N8}
```

Half of C is exchanged at $`z_+\rho_+=\rho_C=\alpha\rho_P/2`$, with $`c\approx\sigma N/[2(L-h)]`$:

```math
\rho_{b,1/2}^{\,z_++1}=\frac{\alpha\rho_P}{2z_+}\left(\frac{2c^2}{z_+\alpha\rho_P}\right)^{z_+}\exp\left[z_+(z_+-1)\Delta u_B+2z_+\Delta u_f\right].
```

Therefore

```math
\rho_b^*\propto \textcolor{red}{c^{\,2z_+/(z_++1)}\propto L^{-2z_+/(z_++1)}} .
\tag{6*}
```

Correlation enters only through the exponential. This replaces Eq. 6, which the SI does not derive. The form of Eq. 6, $`1/[z_+(z_++1)]^3`$, follows from $`\kappa_b^3\approx\rho_b`$, which puts bulk values in place of $`\kappa`$ and $`\rho_+`$ in the brush. At the onset the brush has $`\rho_b\ll\rho_P`$, so that step does not hold.

The neutral crossing and the height minimum have no closed form. They follow from the force balance of the uniform layer, with N6 in each region and $`\phi=\sigma Nv/h`$:

```math
-\ln(1-\phi)-\phi-\chi\phi^2+P_{\rm ion}^{\rm brush}-P_{\rm ion}^{\rm far}=\frac{3\sigma h}{Nb^2}.
```

## 10. Height Scaling

The force balance is $`f_{\rm ela}+f_{\rm int}+f_{\rm ion}=0`$ with

```math
f_{\rm ela}\approx-\frac{\sigma h}{Nb^2},\qquad f_{\rm int}\approx w_2\rho_P^2+w_3\rho_P^3,\qquad\rho_P=\frac{\sigma N}{h}.
```

For the mixing free energy here, $`w_2=v^2(1/2-\chi)`$ and $`w_3=v^3/3`$.

| Regime | Balance | Height |
|---|---|---|
| Osmotic, brush held by C | $`f_{\rm ion}\approx\textcolor{red}{\rho_C\approx\alpha\rho_P}`$ | $`h\approx Nb\,\alpha^{1/2}`$ (S41a\*) |
| After exchange, no correlation | $`f_{\rm ion}\approx\rho_+\approx\alpha\rho_P/z_+`$ | $`h\approx Nb\,(\alpha/z_+)^{1/2}`$ (N9) |
| Salted, requires S34 | S40\* with $`\Gamma>0`$ | $`h\approx Nb\,\alpha^{2/3}\left(\dfrac{\sigma\Gamma}{\textcolor{red}{2}z_+b\rho_b}\right)^{1/3}`$ (S41b\*) |
| Collapsed, requires S34 | $`w_3\rho_P^3+f_{\rm ion}\approx0`$ with $`\Gamma\lt0`$ | $`h\approx-Nb\,\alpha^{-2}\,\dfrac{\textcolor{red}{2}\sigma w_3z_+}{\Gamma b}\,\rho_b`$ (S41c\*) |
| Neutral, $`\chi=1/2`$ | $`w_3\rho_P^3\approx\sigma h/(Nb^2)`$ | $`h\approx Nb\,(w_3\sigma^2/b^2)^{1/4}`$ (N10) |
| Neutral, $`\chi\lt1/2`$ | $`w_2\rho_P^2\approx\sigma h/(Nb^2)`$ | $`h\approx Nb\,(\sigma w_2/b)^{1/3}`$ (S41d) |

S41c\* has its minimum where $`d\ln|\Gamma|/d\ln\rho_b=1`$, not at $`\partial\Gamma/\partial\rho_b=0`$.

## 11. Range of Validity

| Result | Range |
|---|---|
| S33, N5, N6 | Gaussian variational closure. Uniform state; N5 in an inhomogeneous brush is a local-density approximation, without a shortest length ([Section 11.1](#111-response-of-the-self-energy-at-finite-wave-number)) |
| S25, S33 | Screening by point charges, not the stationary point of S11. N11 and N12 keep the charge spread |
| N2, N3 | First order in intrachain $`U`$ |
| Working equations | $`u_P^{\rm ex}`$ and $`I_{ex}`$ dropped. RPA with a Gaussian form factor makes the correlation pressure 12–48 % more negative, as an upper estimate |
| Eq. 7, S40\*, S42\*, Eq. 9\* | $`\rho_b\gg\alpha\rho_P`$, reached only at several molar for $`\rho_P\approx1`$ M |
| N7 | $`a\kappa\ll1`$ |
| N8, Eq. 6\* | $`c\gg z_+\rho_b`$, uniform regions, finite box with conserved C |
| All of it, in the collapsed $`z_+=3`$ state | $`\kappa a\approx1.3`$ and $`z_+^2\ell_B\kappa\approx33`$: the coupling is strong for a Gaussian closure |

### 11.1 Response of the Self-Energy at Finite Wave Number

N5 takes $`u`$ at a point from $`I`$ at that point. S25, without $`I_{ex}`$, answers a small wave of ionic strength about a uniform medium at first order, with $`G(k)=4\pi\ell_B/(k^2+\kappa^2)`$ and $`h_K(k)=e^{-k^2a_K^2/2\pi}`$:

```math
\delta u_K(\mathbf q)=-z_K^2\,R_K(q)\,\delta I(\mathbf q),\qquad
R_K(q)=\int\frac{d^3k}{(2\pi)^3}\,h_K(k)\,h_K(\vert\mathbf k+\mathbf q\vert)\,G(k)\,G(\vert\mathbf k+\mathbf q\vert)
\tag{N13}
```

$`R_K(0)`$ is minus the derivative of the closed form S33 with respect to $`I`$, and $`R_K(q)`$ falls at $`q\gtrsim\kappa,1/a_K`$. The stability of a uniform mixture of chain segments and ions against a wave along $`\mathbf q`$ follows from

```math
M_{JK}(q)=\frac{\delta_{JK}}{S_J(q)}+\delta_{JP}\delta_{KP}\left(\frac{1}{1-\phi}-2\chi\right)+\frac{4\pi\ell_B z_Jz_K}{q^2}-\frac12 z_J^2z_K^2R_J(q)
\tag{N14}
```

with $`S_P=\rho_PNg_D(q)`$, $`g_D`$ the Debye function, and $`S_\pm=\rho_\pm`$. The layer is unstable where an eigenvalue of $`M`$ is negative. In the local form $`R_J(q)=R_J(0)`$ and short waves meet nothing but $`1/S_\pm`$: the instability has no shortest period. In N13 the charge spread and the screening cut it off, so that an instability appears at one period. For the collapsed layer of Fig. 4 of Reference 1, $`\kappa=5.3`$ nm⁻¹, the smallest eigenvalue is +2.2 at a period of 1 nm for $`a_+=2.5`$ Å and −1.4 at a period of 1.02 nm for $`a_+=1.5`$ Å.

## 12. Corrections to the Published Equations

| Paper | Problem | Here | Red shows |
|---|---|---|---|
| S8 | Written as the exponent of $`Z`$, not as $`L`$ | S8\* | Changed: sign of the ion term in the exponent; overall sign in a red note |
| Eq. 3a, Eq. 5 | Integration variables | S12, S25 | Nothing: these two are right in the SI |
| S11, S27a, S27b, S28, S32 | $`\xi`$ has the opposite sign from the main text and from the ion weight | S11\*, S27\*, S32\* | Changed: $`-\xi`$ to $`+\xi`$ |
| S21–S22 | Initial condition of $`g`$ not stated | [Section 4](#4-chain) | Nothing: added equation |
| S25, S33 | $`h_K\to\delta`$ in the screening is not stated, and S25 is not the stationary point of S11 | N11, N12 | Nothing: new equations. The other sections keep the form of the paper |
| S26 | Spurious $`G(\mathbf r,\mathbf r')`$ | S26\* | Removed: $`G`$, in a red note |
| Eq. 3b as a potential, S27c | Density taken from a one-body potential with a factor ½ | N3 | Nothing: new equation |
| S30 | Initial condition lacks $`e^{-V_P}`$ | S30\* | Added: $`e^{-V_P}`$ |
| Main text, dilute $`f_{\rm ion}`$ | $`\pi/24`$ | N7 | Changed: $`\pi/24`$ to $`1/(24\pi)`$ |
| S40 | Twice too large | S40\* | Added: the factor 2 |
| S41a | Stated balance does not give the printed result | S41a\*, N9 | Changed: $`\rho_+\approx\alpha\rho_P/z_+`$ to $`\rho_C\approx\alpha\rho_P`$ |
| S41b, S41c, Eq. 8 lines 2–3 | Inherit the factor 2 of S40 | S41b\*, S41c\* | Added: the factor 2 |
| S41d, Eq. 8 line 4 | Vanishes at $`\chi=1/2`$ | N10 | Nothing: new equation |
| S42 | $`\rho_b`$ missing | S42\* | Added: $`\rho_b`$ |
| Fig. 3 axis | $`\rho_b`$ in mol/L inside Eq. 9 | Eq. 9\* | Added: the unit conversion |
| Main text, lowest height at $`\partial\Gamma/\partial\rho_b=0`$ | S41c\* has its minimum at $`d\ln\vert\Gamma\vert/d\ln\rho_b=1`$ | [Section 10](#10-height-scaling) | Nothing: a statement below the table there |
| Eq. 6, regime boundaries from $`\Gamma`$ | Eq. 6 has no derivation, and the boundaries use Eq. 7 outside S34 | [Section 9](#9-low-salt-ion-exchange) | Changed: all of the right side of 6\* |

## 13. References

1. Duan, C., Agrawal, N. R. & Wang, R. "Electrostatic Correlation Augmented Self-Consistent Field Theory and Its Application to Polyelectrolyte Brushes." *Phys. Rev. Lett.* **134**, 048101 (2025), with its Supplemental Material.

2. Duan, C. & Wang, R. "Ion Correlation-Driven Hysteretic Adhesion and Repulsion between Opposing Polyelectrolyte Brushes." *ACS Macro Lett.* **13**, 1127 (2024).
