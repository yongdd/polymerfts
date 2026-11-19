"""The platforms of a Python test.

cpu_platform() is the platform of a test, or of a part of a test, that runs on one platform:
the first CPU platform of the build, as PlatformSelector lists them (cpu-fftw when the build has
both CPU backends; the MKL-only CI job runs these tests on cpu-mkl), else CUDA.

PlatformSelector.avail_platforms() leaves out "cuda" when no device is visible, so a
test that runs every available platform would drop its CUDA part and pass, which
reads as a validated CUDA path. As the C++ tests do, such a test runs its CPU parts
and then fails (cuda_part_missing). A build without the CUDA backend has no CUDA part to run.
"""

from polymerfts import _core


def cuda_part_missing():
    """True, after printing why, if this build has the CUDA backend but no CUDA device is visible."""
    if _core.build_info["cuda"] == "TRUE" and "cuda" not in _core.PlatformSelector.avail_platforms():
        print("FAILED: this build has the CUDA backend, but no CUDA device is visible: "
              "the CUDA part of this test did not run.")
        return True
    return False


def cpu_platform():
    """The first CPU platform of the build, else CUDA. Without any platform the test ends: it
    fails (exit 1) in a build with the CUDA backend but no visible device, else it is skipped
    (exit 77)."""
    platforms = _core.PlatformSelector.avail_platforms()
    if not platforms:
        raise SystemExit(int(cuda_part_missing()) or 77)
    return next((name for name in platforms if name.startswith("cpu")), platforms[0])
