#!/usr/bin/env python3
"""A short orthogonal axis must not change diffusion in an oblique plane.

The alias gate once compared G12 with the largest diagonal of the whole
reciprocal metric. A short z axis then disabled xy alias minimization, giving
different full-grid and ObliqueZ results for identical z-independent fields.
Merely fixing that gate made the three-dimensional alias search enormous:
the z contribution inflated both the xy search bounds and its tie tolerance.

Strong hexagonal fields on 8x8x4 have the same Q, concentrations and stress
at c=2 and c=1e-7. Check both chain models, every backend, irreducible and
z-mirror bases, and full/reduced-memory agreement. Discrete-chain length and
angle stresses are also checked against finite differences of -ln(Q).
"""

import contextlib
import io

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver, SpaceGroup
from cuda_part import cuda_part_missing


NX = [8, 8, 4]
ANGLES = [90.0, 90.0, 120.0]
THIN = 1e-7


def fields():
    sg = SpaceGroup(NX, "P6/mmm")
    rng = np.random.default_rng(531)
    result = {}
    for monomer in ("A", "B"):
        xy = rng.normal(size=(NX[0], NX[1], 1))
        field = sg.symmetrize(np.broadcast_to(xy, NX).reshape(-1).copy())
        result[monomer] = (field - np.mean(field)) * (5.0 / np.std(field))
        assert np.max(np.ptp(result[monomer].reshape(NX), axis=2)) < 1e-14
    return result


def make_solver(platform, chain, c, basis="none", reduce_memory=False):
    sg = None if basis == "none" else SpaceGroup(NX, "P6/mmm")
    if basis == "z-mirror":
        sg.enable_z_mirror_physical_basis()
    solver = PropagatorSolver(
        nx=NX, lx=[2.0, 2.0, c], angles=ANGLES, ds=0.05,
        bond_lengths={"A": 1.0, "B": 1.0}, bc=["periodic"] * 6,
        chain_model=chain, platform=platform, space_group=sg,
        reduce_memory=reduce_memory)
    solver.add_polymer(1.0, [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]])
    solver.set_force_off_diagonal_stress(True)
    return solver, sg


def evaluate(solver, sg, full_fields):
    w = {m: sg.to_reduced_basis(f) if sg is not None else f for m, f in full_fields.items()}
    solver.compute_propagators(w)
    assert solver.check_total_partition()
    solver.compute_concentrations()
    solver.compute_stress()
    phi = []
    for monomer in ("A", "B"):
        concentration = np.asarray(solver.get_concentration(monomer))
        phi.append(sg.from_reduced_basis(concentration) if sg is not None else concentration)
    phi = np.asarray(phi)
    assert abs(np.mean(phi.sum(axis=0)) - 1.0) < 1e-11
    return solver.get_partition_function(0), phi, np.asarray(solver.get_stress())


def compare(result, reference, label):
    q_error = abs(result[0] / reference[0] - 1.0)
    phi_error = np.max(np.abs(result[1] - reference[1]))
    stress_scale = max(np.max(np.abs(reference[2])), 1e-20)
    stress_error = np.max(np.abs(result[2] - reference[2])) / stress_scale
    assert q_error < 1e-10, (label, "Q", q_error)
    assert phi_error < 1e-10, (label, "concentrations", phi_error)
    assert stress_error < 1e-8, (label, "stress", stress_error)
    print(f"PASS {label}: Q {q_error:.2e}, phi {phi_error:.2e}, stress {stress_error:.2e}")


def fourier_plane(platform):
    solver, _ = make_solver(platform, "discrete", 2.0)
    transform = solver.fourier
    reference = transform.k_squared()
    solver.set_lx([2.0, 2.0, THIN])
    thin = transform.k_squared()
    np.testing.assert_allclose(thin[:, :, 0], reference[:, :, 0], rtol=1e-13, atol=1e-12)
    # For a=b=2, gamma=120, G11=G22=1/3 and G12=1/6. The
    # (3,3,0) mode's least-form alias is (-5,3,0) or (3,-5,0).
    expected = 4.0 * np.pi ** 2 * 19.0 / 3.0
    np.testing.assert_allclose(thin[3, 3, 0], expected, rtol=1e-13, atol=0.0)


def stress_derivatives(platform, full_fields):
    solver, sg = make_solver(platform, "discrete", THIN)
    result = evaluate(solver, sg, full_fields)
    expected = (result[2][0] + result[2][1], result[2][3])
    for vary_angle, derivative in enumerate(expected):
        estimates = []
        for h in (2e-3, 1e-3, 5e-4):
            energies = []
            for sign in (1, -1):
                lx, angles = [2.0, 2.0, THIN], list(ANGLES)
                if vary_angle:
                    angles[2] += np.rad2deg(sign * h)
                else:
                    lx[0] += sign * h
                    lx[1] += sign * h
                solver.set_lattice_parameters(lx, angles)
                solver.compute_propagators(full_fields)
                energies.append(-np.log(solver.get_partition_function(0)))
            estimates.append((energies[0] - energies[1]) / (2 * h))
        # Alias ties can introduce an O(h) term; remove it and the O(h^2)
        # term, as in TestStressAliasTies.
        difference = (8 * estimates[2] - 6 * estimates[1] + estimates[0]) / 3
        assert abs(difference - derivative) < 1e-7 * max(abs(derivative), 1e-12), (
            platform, "angle" if vary_angle else "lengths", derivative, difference)
        solver.set_lattice_parameters([2.0, 2.0, THIN], ANGLES)


def main():
    full_fields = fields()
    platforms = PlatformSelector.avail_platforms()
    for platform in platforms:
        with contextlib.redirect_stdout(io.StringIO()):
            fourier_plane(platform)
        print(f"PASS {platform}: public Fourier k_squared on the unchanged xy plane")
    backend_reference = {}
    for chain in ("continuous", "discrete"):
        for platform in platforms:
            with contextlib.redirect_stdout(io.StringIO()):
                solver, sg = make_solver(platform, chain, 2.0)
                reference = evaluate(solver, sg, full_fields)
            if chain in backend_reference:
                compare(reference, backend_reference[chain], f"{chain}/{platform} vs first backend")
            else:
                backend_reference[chain] = reference
            for basis in ("none", "irreducible", "z-mirror"):
                memory_reference = None
                for reduce_memory in (False, True):
                    with contextlib.redirect_stdout(io.StringIO()):
                        solver, sg = make_solver(platform, chain, THIN, basis, reduce_memory)
                        result = evaluate(solver, sg, full_fields)
                    label = f"{chain}/{platform}/{basis}/reduce_memory={reduce_memory}"
                    compare(result, reference, label)
                    if memory_reference is not None:
                        assert result[0] == memory_reference[0], (label, "memory-mode Q")
                        assert np.array_equal(result[1], memory_reference[1]), (label, "memory-mode concentrations")
                    memory_reference = result
            if chain == "discrete":
                with contextlib.redirect_stdout(io.StringIO()):
                    stress_derivatives(platform, full_fields)
                print(f"PASS {platform}: thin-cell stress derivatives")
    if cuda_part_missing():
        return 1
    print("Thin oblique cells: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
