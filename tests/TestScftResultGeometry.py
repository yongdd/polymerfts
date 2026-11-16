#!/usr/bin/env python3
"""SCFT results and callbacks retain fixed as well as variable lattice angles.

Compare VTK coordinates exported from the result with the native solver's
Cartesian grid. Include fixed hexagonal cells with and without a space group,
a fixed oblique 3D cell, and orthogonal/1D compatibility controls.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
from polymerfts import SCFT, PlatformSelector, save_vtk
from TestSaveVtk import read_vtk
from cuda_part import cuda_part_missing


def check_geometry(platform, directory, nx, lx, angles=None, **options):
    params = {
        "nx": nx, "lx": lx, "chain_model": "continuous", "ds": 0.05,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 10.0},
        "polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": platform, "relax_cell": False, "max_iter": 1, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                      "mix_min": 0.1, "mix_init": 0.1},
    }
    if angles is not None:
        params["angles"] = angles
    params.update(options)
    snapshots = []
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SCFT(params)
        result = sim.run({name: np.zeros(int(np.prod(nx))) for name in ("A", "B")},
                         return_result=True, on_iteration=lambda info: snapshots.append(info))
    assert result.converged
    assert len(snapshots) == 1
    assert sim.prop_solver.check_total_partition()
    np.testing.assert_allclose(np.mean(result.phi["A"] + result.phi["B"]), 1.0, rtol=0, atol=2e-12)
    expected_angles = list(sim.prop_solver.get_angles_degrees())
    if len(nx) == 1:
        assert result.angles is None and snapshots[0].angles is None
    else:
        np.testing.assert_allclose(result.angles, expected_angles, rtol=0, atol=1e-12)
        np.testing.assert_allclose(snapshots[0].angles, expected_angles, rtol=0, atol=1e-12)

    path = directory / "geometry.vtk"
    save_vtk(path, result.phi, result.lx, nx=result.nx, angles=result.angles, space_group=sim.sg)
    header, points, fields = read_vtk(path)
    for name in ("A", "B"):
        np.testing.assert_allclose(fields[name], result.phi[name].reshape(nx).ravel(order="F"),
                                   rtol=1e-7, atol=1e-8)
    if np.allclose(expected_angles, 90.0, rtol=0, atol=1e-12):
        assert header["DATASET"] == ["STRUCTURED_POINTS"]
    else:
        assert header["DATASET"] == ["STRUCTURED_GRID"] and points is not None
        coordinates = sim.prop_solver.get_grid_points()
        expected = np.column_stack([axis.ravel(order="F") for axis in coordinates])
        np.testing.assert_allclose(points[:, :len(nx)], expected, rtol=0, atol=1e-9)
    print(f"[PASS] {platform}: nx={nx}, angles={expected_angles}, "
          f"relax_cell={sim.relax_cell}, space_group={params.get('space_group')}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        for platform in PlatformSelector.avail_platforms():
            root = Path(directory)
            check_geometry(platform, root, [8], [2.0])
            check_geometry(platform, root, [8, 8], [2.0, 2.0])
            for relax_cell in (False, True):
                check_geometry(platform, root, [8, 8], [2.0, 2.0], [120.0], relax_cell=relax_cell)
            check_geometry(platform, root, [8, 8, 8], [2.0, 2.0, 2.3], [90.0, 90.0, 120.0],
                           space_group={"symbol": "P6_3/mmc", "hall_number": 488})
            check_geometry(platform, root, [8, 8, 8], [1.7, 2.0, 2.3], [80.0, 100.0, 110.0])
            check_geometry(platform, root, [8, 8, 8], [1.7, 2.0, 2.3], [90.0, 90.0, 90.0],
                           crystal_system="Triclinic", relax_cell=True)
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
