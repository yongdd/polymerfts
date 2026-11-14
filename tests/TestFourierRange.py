"""Finite Hermitian projections retain very large and subnormal components."""

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def transform(platform, nx):
    solver = PropagatorSolver(nx=nx, lx=[2.0] * len(nx), ds=0.1,
                              bond_lengths={"A": 1.0}, bc=["periodic"] * (2 * len(nx)),
                              chain_model="continuous", platform=platform)
    return solver, solver.fourier


def relative(actual, expected):
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.asarray(actual) / expected, 1.0, rtol=5e-13, atol=0.0)


def check(platform):
    tiny = np.nextafter(0.0, 1.0)
    for n in (15, 16):
        solver, ft = transform(platform, [n])
        for dc in (1e100, 1e308, 16 * tiny):
            coefficients = np.zeros((2, n // 2 + 1), complex)
            coefficients[:, 0] = (dc, -dc)
            before = coefficients.copy()
            actual = ft.backward(coefficients)
            relative(actual[0], dc / n)
            relative(actual[1], -dc / n)
            np.testing.assert_array_equal(coefficients, before)
        if n % 2 == 0:
            coefficients = np.zeros(n // 2 + 1, complex)
            coefficients[-1] = 1e308 + 1e308j
            relative(ft.backward(coefficients), (1e308 / n) * (-1.0) ** np.arange(n))
        for multiplier, field in ((1e100, 1e-100), (1e308, 1e-308), (tiny, 1e308 / n)):
            coefficients = np.full(n // 2 + 1, multiplier)
            before = coefficients.copy()
            handle = ft.set_multiplier(coefficients)
            relative(ft.apply([(handle, np.full(n, field))]), multiplier * field)
            np.testing.assert_array_equal(coefficients, before)

    # A non-self-conjugate pair averages both large components before acting
    # on small fields. Scale the reference operands before any FFT/products.
    solver, ft = transform(platform, [4, 4])
    multiplier = np.zeros((4, 3), complex)
    multiplier[1, 0] = complex(1.2e308, 1.0e308)
    multiplier[3, 0] = complex(0.8e308, -1.4e308)
    original = multiplier.copy()
    unit_fields = np.random.default_rng(12).normal(0.0, 5.0, size=(4, 4))
    normalized = np.zeros((4, 3), complex)
    normalized[1, 0] = 1.0 + 1.2j
    normalized[3, 0] = 1.0 - 1.2j
    expected = np.fft.irfftn(normalized * np.fft.rfftn(unit_fields), s=(4, 4), axes=(0, 1)).ravel()
    actual = ft.apply([(ft.set_multiplier(multiplier), (unit_fields * 1e-308).ravel())])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-14)
    np.testing.assert_array_equal(multiplier, original)
    print(f"Finite Hermitian projection: {platform} passed")


if __name__ == "__main__":
    for platform in PlatformSelector.avail_platforms():
        check(platform)
    raise SystemExit(cuda_part_missing())
