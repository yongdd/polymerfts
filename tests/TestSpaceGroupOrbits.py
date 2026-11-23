#!/usr/bin/env python3
"""The orbits and physical bases of SpaceGroup against the symmetry operations, for all
530 Hall numbers.

SpaceGroup samples grid point i of an axis with N points at (i + 1/2)/N, the cell centers;
hexagonal-axes settings sample x and y at the nodes i/N. Every operation (R, t) of the
spglib database then maps every grid point onto a grid point. Checked here in floating
point, independently of the integer grid maps of SpaceGroup.cpp, on grids with even and
odd, equal and unequal sizes:

- every image lies on the grid and in the orbit of its point;
- an orbit holds as many points as a point of it has distinct images;
- each orbit is represented by its first point in flat order, and the orbits are numbered
  in the order of their representatives (the layout of reduced-basis fields);
- a rhombohedral-axes setting refuses a grid with Nz != Nx;
- every physical basis a group admits lists the points of its definition (the Pmmm octant,
  the even-index octant of m3, the z half of the z-mirror basis), maps every grid point to
  a basis point of its orbit (the Pmmm and z-mirror bases to its mirror image), counts the
  grid points of each basis point, and maps each basis point to its orbit.
"""

import sys

import numpy as np
import spglib

from polymerfts import SpaceGroup

GRIDS = [[12, 12, 12], [6, 6, 6], [9, 9, 9], [12, 12, 9], [24, 24, 16], [20, 16, 24]]


def operation_images(sg, hall, nx):
    """The flat index of the image of every grid point under every operation, shape
    (n_ops, n_points), and the largest distance of an image from a grid point (in grid spacings)."""
    offset = np.array([0.0, 0.0, 0.5] if sg.uses_hexagonal_axes() else [0.5, 0.5, 0.5])[:, None]
    n = np.array(nx)[:, None]
    x = (np.indices(nx).reshape(3, -1) + offset) / n
    ops = spglib.get_symmetry_from_database(hall)
    g = (np.einsum("oij,jp->oip", ops["rotations"], x) + ops["translations"][:, :, None]) * n - offset
    i = np.rint(g)
    worst = float(np.max(np.abs(g - i)))
    i = np.mod(i.astype(np.int64), n)
    return (i[:, 0] * nx[1] + i[:, 1]) * nx[2] + i[:, 2], worst


def check_orbits(sg, hall, nx):
    images, worst = operation_images(sg, hall, nx)
    orbit_of = np.array(sg.get_full_to_reduced_map())
    sizes = np.array(sg.get_orbit_counts())
    representatives = np.array(sg.get_reduced_basis_indices())
    n_orbits = sg.get_n_reduced_basis_full()
    problems = []
    if worst > 1e-9:
        problems.append(f"an image lies {worst:.1e} grid spacings off the grid")
    if not np.array_equal(orbit_of[images], np.broadcast_to(orbit_of, images.shape)):
        problems.append("an image lies in another orbit")
    distinct = 1 + np.count_nonzero(np.diff(np.sort(images, axis=0), axis=0), axis=0)
    if sizes.size != n_orbits or not np.array_equal(distinct, sizes[orbit_of]):
        problems.append("an orbit size differs from the number of distinct images of its points")
    first = np.full(n_orbits, orbit_of.size)
    np.minimum.at(first, orbit_of, np.arange(orbit_of.size))
    if not np.array_equal(representatives, first) or np.any(np.diff(representatives) <= 0):
        problems.append("the representatives are not the first points of the orbits in flat order")
    return problems


def z_mirror_shift(hall, nz):
    """The z offset of the z-mirror half grid: 0 for the mirror (x, y, -z), nz/4 for (x, y, -z + 1/2)."""
    ops = spglib.get_symmetry_from_database(hall)
    best = None
    for R, t in zip(ops["rotations"], ops["translations"]):
        lattice_xy = np.all(np.abs(t[:2] - np.rint(t[:2])) < 1e-10)
        if np.array_equal(R, np.diag([1, 1, -1])) and lattice_xy:
            tz = abs(t[2] - np.floor(t[2] + 0.5))
            best = tz if best is None or tz + 1e-10 < best else best
    return 0 if best < 1e-8 else nz // 4


def expected_points(name, hall, nx):
    """Basis points of each physical basis, and the mirror fold of the Pmmm and z-mirror bases."""
    Nx, Ny, Nz = nx
    ix, iy, iz = np.indices(nx).reshape(3, -1)
    if name in ("pmmm", "m3"):
        step = 2 if name == "m3" else 1
        o = np.indices([Nx // 2, Ny // 2, Nz // 2]).reshape(3, -1) * step
        points = (o[0] * Ny + o[1]) * Nz + o[2]
        if name == "m3":
            return points, None
        fold = [np.where(i < N // 2, i, N - 1 - i) for i, N in ((ix, Nx), (iy, Ny), (iz, Nz))]
        return points, (fold[0] * (Ny // 2) + fold[1]) * (Nz // 2) + fold[2]
    shift = z_mirror_shift(hall, Nz)
    h = np.indices([Nx, Ny, Nz // 2]).reshape(3, -1)
    points = (h[0] * Ny + h[1]) * Nz + (h[2] + shift) % Nz
    zs = (iz - shift) % Nz
    fold = np.where(zs < Nz // 2, zs, Nz - 1 - zs)
    return points, (ix * Ny + iy) * (Nz // 2) + fold


def check_physical_basis(sg, name, hall, nx, orbit_of, n_orbits):
    points = np.array(sg.get_reduced_basis_indices())
    basis_of = np.array(sg.get_full_to_reduced_map())
    counts = np.array(sg.get_orbit_counts())
    expected, expected_basis_of = expected_points(name, hall, nx)
    problems = []
    if not np.array_equal(points, expected):
        problems.append("the basis points differ from the definition")
    elif not np.array_equal(basis_of[points], np.arange(points.size)):
        problems.append("a basis point does not map to itself")
    elif not np.array_equal(orbit_of[points[basis_of]], orbit_of):
        problems.append("a grid point maps to a basis point of another orbit")
    if expected_basis_of is not None and not np.array_equal(basis_of, expected_basis_of):
        problems.append("a grid point does not map to its mirror image")
    if not np.array_equal(counts, np.bincount(basis_of, minlength=points.size)):
        problems.append("the counts differ from the grid points of each basis point")
    to_orbit = sg.irreducible_to_reduced(np.arange(n_orbits, dtype=float))
    if sg.get_n_reduced_basis_full() != n_orbits or not np.array_equal(to_orbit, orbit_of[points]):
        problems.append("the basis-to-orbit map differs from the orbits of the basis points")
    return problems


def main():
    failed = []
    n_grids = n_hexagonal = n_odd = n_refused = 0
    n_bases = {"pmmm": 0, "m3": 0, "z-mirror": 0}
    enable = {"pmmm": "enable_pmmm_physical_basis", "m3": "enable_m3_physical_basis",
              "z-mirror": "enable_z_mirror_physical_basis"}
    for hall in range(1, 531):
        checked = 0
        for nx in GRIDS:
            try:
                sg = SpaceGroup(nx, hall)
            except Exception:
                continue        # a grid the translations or the crystal system exclude
            checked += 1
            n_hexagonal += sg.uses_hexagonal_axes()
            n_odd += any(n % 2 for n in nx)
            for problem in check_orbits(sg, hall, nx):
                failed.append(f"Hall {hall} on {nx}: {problem}")
            orbit_of = np.array(sg.get_full_to_reduced_map())
            for name in n_bases:
                sgb = SpaceGroup(nx, hall)
                try:
                    getattr(sgb, enable[name])()
                except Exception:
                    continue    # a basis the group or the grid does not admit
                n_bases[name] += 1
                for problem in check_physical_basis(sgb, name, hall, nx, orbit_of, sg.get_n_reduced_basis_full()):
                    failed.append(f"Hall {hall} on {nx}, {name} basis: {problem}")
        n_grids += checked
        if checked == 0:
            failed.append(f"Hall {hall}: no grid of the list is accepted")
        sg = SpaceGroup([12, 12, 12], hall)
        if sg.get_crystal_system() == "Trigonal" and not sg.uses_hexagonal_axes():
            try:
                SpaceGroup([12, 12, 18], hall)
                failed.append(f"Hall {hall} (rhombohedral axes) accepted the grid [12, 12, 18]")
            except Exception as e:
                n_refused += 1
                if "does not map" not in str(e):
                    failed.append(f"Hall {hall} (rhombohedral axes): unexpected error {e}")

    print(f"530 Hall numbers on {n_grids} grids ({n_hexagonal} with hexagonal axes, {n_odd} with an odd size); "
          f"physical bases {n_bases}; {n_refused} rhombohedral-axes settings refused [12, 12, 18]")
    # The numbers of cases of this list of grids
    expected = (2033, 197, 199, {"pmmm": 49, "m3": 534, "z-mirror": 356}, 7)
    if (n_grids, n_hexagonal, n_odd, n_bases, n_refused) != expected:
        failed.append(f"case counts {(n_grids, n_hexagonal, n_odd, n_bases, n_refused)}, expected {expected}")
    for f in failed[:40]:
        print("FAILED:", f)
    if failed:
        print(f"{len(failed)} failures")
        sys.exit(1)
    print("Space-group orbits and physical bases: OK")


if __name__ == "__main__":
    main()
