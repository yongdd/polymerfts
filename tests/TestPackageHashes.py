#!/usr/bin/env python3
"""The source hash of the metadata covers the package, also from a copy of utils/common.py.

QuickStart tells users to copy a module next to their script and to point its imports at the
installed package. The hash must then still walk the package: a walk from the copy's own folder
reads whatever lies around the user's project.
"""

import importlib.util
import pathlib
import re
import tempfile

from polymerfts.utils import common


def copy_as_in_quickstart(project):
    """utils/common.py next to a script, with the substitutions of QuickStart."""
    text = pathlib.Path(common.__file__).read_text(encoding="utf-8")
    text = re.sub(r"^( *)from \.\. import _core", r"\1from polymerfts import _core", text, flags=re.M)
    text = re.sub(r"^( *)from \.\.([a-z_.]+) import", r"\1from polymerfts.\2 import", text, flags=re.M)
    text = re.sub(r"^( *)from \.([a-z_]+) import", r"\1from polymerfts.utils.\2 import", text, flags=re.M)
    path = project / "my_common.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("my_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    with tempfile.TemporaryDirectory() as root:
        project = pathlib.Path(root) / "my_project"
        project.mkdir()
        (project.parent / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
        copy = copy_as_in_quickstart(project)
        assert copy._package_hashes() == common._package_hashes(), "The copy hashed files outside the package"
    print("The hash of a copy of utils/common.py covers the package: OK")


if __name__ == "__main__":
    main()
