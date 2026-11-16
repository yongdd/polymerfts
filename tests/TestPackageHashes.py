#!/usr/bin/env python3
"""The source hashes of the metadata: the package, also from a copy of utils/common.py, and the file of a
simulation class that lies outside the package.

QuickStart tells users to copy a module next to their script and to point its imports at the
installed package. The package hash must then still walk the package: a walk from the copy's own
folder reads whatever lies around the user's project. A modified copy of scft.py (or a subclass in a
script) runs code that the package hash does not cover, so the metadata hashes that file as well.
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import pathlib
import re
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

from polymerfts import SCFT, PlatformSelector
from polymerfts.utils import common


def copy_as_in_quickstart(module, folder, project, name):
    """A module of the package next to a script, with the substitutions of QuickStart."""
    text = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    text = re.sub(r"^( *)from \.\. import _core", r"\1from polymerfts import _core", text, flags=re.M)
    text = re.sub(r"^( *)from \.\.([a-z_.]+) import", r"\1from polymerfts.\2 import", text, flags=re.M)
    text = re.sub(r"^( *)from \.([a-z_]+) import", rf"\1from polymerfts.{folder}.\2 import", text, flags=re.M)
    path = project / (name + ".py")
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    copy = importlib.util.module_from_spec(spec)
    sys.modules[name] = copy   # as an import of the copy does
    spec.loader.exec_module(copy)
    return copy, path


def scft_parameters(platform):
    return {"nx": [8], "lx": [2.0], "ds": 0.1, "chain_model": "continuous", "platform": platform,
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 10.0},
            "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                {"type": "B", "length": 0.5}]}],
            "relax_cell": False, "optimizer": {"name": "am", "max_hist": 5, "start_error": 0.01,
                                               "mix_min": 0.1, "mix_init": 0.1}}


class ScriptSCFT(SCFT):
    """A subclass in a script (this file)."""


def main():
    import polymerfts.scft.scft as scft_module
    platform = PlatformSelector.avail_platforms()[0]
    with tempfile.TemporaryDirectory() as root:
        project = pathlib.Path(root) / "my_project"
        project.mkdir()
        (project.parent / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
        copy, _ = copy_as_in_quickstart(common, "utils", project, "my_common")
        assert copy._package_hashes() == common._package_hashes(), "The copy hashed files outside the package"

        my_scft, path = copy_as_in_quickstart(scft_module, "scft", project, "my_scft")
        with contextlib.redirect_stdout(io.StringIO()):
            package, modified, script = (cls(scft_parameters(platform)) for cls in (SCFT, my_scft.SCFT, ScriptSCFT))
        package, modified, script = (common.simulation_metadata(sim) for sim in (package, modified, script))
        assert "simulation_source_sha256" not in package, "A class of the package needs no hash of its own"
        assert modified["simulation_source"] == str(path.resolve())
        assert modified["simulation_source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert script["simulation_source_sha256"] == hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        for metadata in (modified, script):
            assert metadata["python_source_sha256"] == package["python_source_sha256"]
    print("The hash of a copy of utils/common.py covers the package, and a simulation class outside it is hashed: OK")


if __name__ == "__main__":
    main()
