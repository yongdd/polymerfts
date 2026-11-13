#!/usr/bin/env python3
"""VTK exports preserve distinct fields and small nonorthogonal cell angles."""

from pathlib import Path
import tempfile

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver, save_vtk
from TestSaveVtk import read_vtk
from cuda_part import cuda_part_missing


def field_names(root):
    path = root / "fields.vtk"
    fields = {"A B": np.arange(4, dtype=float), "C_D": np.arange(4, dtype=float) + 10.0}
    save_vtk(path, fields, [2.0], nx=[4])
    _, _, saved = read_vtk(path)
    assert set(saved) == {"A_B", "C_D"}
    np.testing.assert_array_equal(saved["A_B"], fields["A B"])
    np.testing.assert_array_equal(saved["C_D"], fields["C_D"])
    contents = path.read_bytes()
    for names in (("A B", "A_B"), ("A_B", "A B"), (1, "1")):
        conflicting = {names[0]: np.ones(4), names[1]: np.full(4, 2.0)}
        for target in (path, root / "new.vtk"):
            try:
                save_vtk(target, conflicting, [2.0], nx=[4])
            except ValueError as error:
                assert "collide" in str(error), error
            else:
                raise AssertionError("A VTK field silently overwrote another field")
        assert path.read_bytes() == contents, "Rejected export changed the previous file"
        assert not (root / "new.vtk").exists(), "Rejected export created a partial file"
        np.testing.assert_array_equal(conflicting[names[0]], np.ones(4))
        np.testing.assert_array_equal(conflicting[names[1]], np.full(4, 2.0))


def small_shear(root, platform):
    for dim in (2, 3):
        nx, lx = [3, 4, 2][:dim], [2.0, 3.0, 4.0][:dim]
        fields = np.arange(np.prod(nx), dtype=float).reshape(nx)
        for angle_index in ([2] if dim == 2 else range(3)):
            for delta in (0.0, 1e-12, -0.0005, 0.0005, 1e-6):
                angles = [90.0] * 3
                angles[angle_index] += delta
                solver = PropagatorSolver(nx=nx, lx=lx, ds=0.25, bond_lengths={"A": 1.0},
                                          bc=["periodic"] * (2 * dim), chain_model="continuous",
                                          angles=angles, platform=platform)
                solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
                path = root / "geometry.vtk"
                save_vtk(path, fields, lx, angles=[angles[2]] if dim == 2 else angles)
                header, points, saved = read_vtk(path)
                np.testing.assert_array_equal(saved["phi"], fields.ravel(order="F"))
                if solver.computation_box.is_orthogonal():
                    assert header["DATASET"] == ["STRUCTURED_POINTS"]
                    assert points is None
                else:
                    assert header["DATASET"] == ["STRUCTURED_GRID"]
                    expected = np.column_stack([axis.ravel(order="F") for axis in solver.get_grid_points()])
                    np.testing.assert_allclose(points[:, :dim], expected, rtol=0, atol=1e-9)
                    assert abs(delta) > 1e-12
        print(f"[PASS] {platform}: {dim}D VTK small shear and orthogonal controls")


def main():
    with tempfile.TemporaryDirectory(prefix="polymerfts_vtk_integrity_") as directory:
        root = Path(directory)
        field_names(root)
        for platform in PlatformSelector.avail_platforms():
            small_shear(root, platform)
    return int(cuda_part_missing())


if __name__ == "__main__":
    raise SystemExit(main())
