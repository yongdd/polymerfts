"""The version number is written in four files and must be the same in all of them.

`polymerfts.__version__` comes from the native module (CMakeLists.txt), the wheel metadata from
pyproject.toml, the conda package from conda-recipe/meta.yaml and the citation from CITATION.cff.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def version_in(path, pattern):
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"no version found in {path}"
    return ".".join(match.groups())


def test_version_numbers_agree():
    versions = {
        "pyproject.toml": version_in("pyproject.toml", r'^version = "(\d+)\.(\d+)\.(\d+)"'),
        "CMakeLists.txt": version_in(
            "CMakeLists.txt",
            r"^SET\(POLYMERFTS_VERSION_MAJOR (\d+)\)\s*^SET\(POLYMERFTS_VERSION_MINOR (\d+)\)\s*"
            r"^SET\(POLYMERFTS_VERSION_PATCH (\d+)\)"),
        "conda-recipe/meta.yaml": version_in("conda-recipe/meta.yaml", r'^\{% set version = "(\d+)\.(\d+)\.(\d+)" %\}'),
        "CITATION.cff": version_in("CITATION.cff", r"^version: (\d+)\.(\d+)\.(\d+)$"),
    }
    assert len(set(versions.values())) == 1, versions
