"""Initial fields (seeds) of common phases of AB block copolymers with a minority A block.

An SCFT run converges its seed, so a seed needs only the symmetry, the rough shape and
roughly the period of its phase. The scripts in this folder import this module directly;
those in phases/, phase_diagram/ and crystal_systems/ add this folder to the module path:

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import seeds

Each phase function returns {"w": {"A": w_A, "B": w_B}, "lx": lx}, the initial fields on
the grid nx and the cell they belong to.
"""

import os

import numpy as np
import scipy.io
import scipy.ndimage


def resample(field, nx):
    """A periodic field interpolated to the grid nx.

    Both grids sample the same cell at the centres of their grid cells. Cubic splines with
    periodic wrapping (scipy.ndimage.zoom with grid_mode=True and mode="grid-wrap") keep the
    field periodic; zoom's default corner-aligned, zero-padded interpolation does not. A field
    already on the grid nx is returned unchanged.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.shape == tuple(nx):
        return field
    return scipy.ndimage.zoom(field, np.array(nx)/np.array(field.shape), grid_mode=True, mode="grid-wrap")


def from_file(file_name, nx, axes=None):
    """The fields w_<type> of a .mat file (an SCFT result, or another file with nx, lx and the
    flattened fields), resampled to the grid nx, with the cell of the file. axes reorders the
    axes of the file first (numpy.transpose), e.g. a file in the order [c, a, b]."""
    data = scipy.io.loadmat(file_name, squeeze_me=True)
    nx_file = [int(n) for n in np.atleast_1d(data["nx"])]
    lx = [float(x) for x in np.atleast_1d(data["lx"])]
    w = {}
    for key in data:
        if key.startswith("w_"):
            field = np.reshape(data[key], nx_file)
            if axes is not None:
                field = np.transpose(field, axes)
            w[key[2:]] = resample(field, nx)
    if axes is not None:
        lx = [lx[axis] for axis in axes]
    return {"w": w, "lx": lx}


def lamellae(nx, lx):
    """One lamellar period along the single axis of a 1D grid."""
    w = np.cos(2*np.pi*np.arange(nx[0])/nx[0])
    return {"w": {"A": w, "B": -w}, "lx": list(lx)}


def wells(nx, lx, positions):
    """Smoothed wells of w_A at the given fractional positions (A domains: spheres, or
    cylinders on a 2D grid), the grid points nearest to them."""
    w_A = np.zeros(nx)
    for position in positions:
        index = tuple(np.round(np.array(position)*np.array(nx)).astype(int) % np.array(nx))
        w_A[index] = -1/(np.prod(lx)/np.prod(nx))
    w_A = scipy.ndimage.gaussian_filter(w_A, sigma=np.min(nx)/15, mode="wrap")
    return {"w": {"A": w_A, "B": np.zeros(nx)}, "lx": list(lx)}


def cylinders(nx, lx):
    """Hexagonally packed A cylinders in a 2D rectangular cell with sides in the
    ratio sqrt(3), which holds two of them."""
    return wells(nx, lx, [(0.0, 0.0), (0.5, 0.5)])


def bcc_spheres(nx, lx):
    """A spheres on a body-centred cubic lattice."""
    return wells(nx, lx, [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])


def oblique_cylinder(nx, lx, gamma, depth):
    """One A cylinder at the origin of a cell with a = b and the angle gamma (degrees) between a
    and b, with distances measured in the metric of that cell, so that the cylinder is round for
    any gamma; w_A reaches -depth at its centre. A third axis of nx runs along the cylinder."""
    u = np.arange(nx[0])[:, None] / nx[0]
    v = np.arange(nx[1])[None, :] / nx[1]
    u, v = u - np.round(u), v - np.round(v)
    r = np.sqrt(np.maximum(u**2 + v**2 + 2 * np.cos(np.radians(gamma)) * u * v, 0.0))
    w_A = -scipy.ndimage.gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
    w_A *= depth / np.abs(w_A).max()
    if len(nx) == 3:
        w_A = np.repeat(w_A[:, :, None], nx[2], axis=2)
    return {"w": {"A": w_A, "B": -w_A}, "lx": list(lx)}


def gyroid(nx, lx, periods=1):
    """Double gyroid of A networks from the level-set approximation of Matsen and Bates
    (Macromolecules 29, 1091 (1996)), with the given number of unit cells along each axis."""
    x, y, z = np.meshgrid(*[periods*np.arange(1, n + 1)*2*np.pi/n for n in nx], indexing="ij", sparse=True)
    c1 = np.sqrt(8.0/3.0)*(np.cos(x)*np.sin(y)*np.sin(2.0*z) +
                           np.cos(y)*np.sin(z)*np.sin(2.0*x) + np.cos(z)*np.sin(x)*np.sin(2.0*y))
    c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*x)*np.cos(2.0*y) +
                           np.cos(2.0*y)*np.cos(2.0*z) + np.cos(2.0*z)*np.cos(2.0*x))
    w_A = -0.3164*c1 + 0.1074*c2
    return {"w": {"A": w_A, "B": -w_A}, "lx": list(lx)}


def double_gyroid(nx, lx=None):
    """Converged double gyroid fields of phases/DG.mat, interpolated to nx, in their cell or in a
    cubic cell of side lx."""
    seed = from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phases", "DG.mat"), nx)
    return seed if lx is None else {"w": seed["w"], "lx": list(lx)}
