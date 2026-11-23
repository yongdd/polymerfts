"""Counted checks of a Python test, the twin of TestChecks.h, and inputs that several tests share.

A test that reports every failed check, rather than stopping at its first failure (plain assert),
calls check(condition, message) for each check: it prints [PASS] or [FAIL] with the message and
counts the check. summary() ends the test: it prints the counts and the failed checks, and returns
the exit code. A test whose checks print under its own headers sets checks.indent.
expect_validation_error() expects the input error of a construction. smooth_field() and smooth()
make the smooth random fields of several tests.

Importing this module loads no native library and changes nothing; fts_checks, the home of what
the tests of simulation runs share, imports the simulation classes and turns off their drawings.
"""

import contextlib
import io

import numpy as np

indent = ""   # printed before [PASS] and [FAIL]
passed = 0
failed = []   # the messages of the failed checks


def check(condition, message):
    """Prints [PASS] or [FAIL] and message, counts the check and returns whether it passed."""
    global passed
    print(f"{indent}[{'PASS' if condition else 'FAIL'}] {message}")
    if condition:
        passed += 1
    else:
        failed.append(message)
    return bool(condition)


def summary(expected=None):
    """Prints the counts and the failed checks. Returns the exit code: 1 if a check failed or, with
    expected, if the test made another number of checks, else 0."""
    count = "" if expected is None else f", {passed + len(failed)} checks (expected {expected})"
    print(f"\n{passed} passed, {len(failed)} failed{count}")
    for message in failed:
        print(f"FAILED: {message}")
    return int(bool(failed) or expected is not None and passed + len(failed) != expected)


def expect_validation_error(make, message):
    """make() raises a ValidationError whose message contains message; what make() prints is discarded."""
    from polymerfts import ValidationError   # here, so that importing this module loads no native library
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            make()
    except ValidationError as error:
        assert message in str(error), f"expected '{message}', got '{error}'"
        return
    raise AssertionError(f"expected a ValidationError with '{message}'")


def smooth_field(rng, nx, lx):
    """A sum of four products of cosines along the three axes of the box lx (wave numbers 1 or 2,
    random amplitudes and phases from rng) on the grid nx: mean 0, standard deviation 5, flat."""
    x = [np.arange(n) * l / n for n, l in zip(nx, lx)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    f = np.zeros(nx)
    for _ in range(4):
        kx, ky, kz = rng.integers(1, 3, size=3)
        f += rng.normal() * (np.cos(2 * np.pi * kx * X / lx[0] + rng.uniform(0, 2 * np.pi))
                             * np.cos(2 * np.pi * ky * Y / lx[1] + rng.uniform(0, 2 * np.pi))
                             * np.cos(2 * np.pi * kz * Z / lx[2] + rng.uniform(0, 2 * np.pi)))
    f = f - f.mean()
    return (f / f.std() * 5.0).flatten()


def smooth(x, nx, std):
    """The field x on the grid nx times exp(-0.08 m^2) in Fourier space (m: the integer wave numbers),
    shifted and scaled to mean 0 and standard deviation std, flat."""
    k = np.meshgrid(*[np.fft.fftfreq(m) * m for m in nx], indexing="ij")
    y = np.real(np.fft.ifftn(np.fft.fftn(np.reshape(x, nx)) * np.exp(-0.08 * sum(kk ** 2 for kk in k)))).ravel()
    return std * (y - y.mean()) / y.std()
