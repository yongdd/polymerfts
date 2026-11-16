#!/usr/bin/env python3
"""Anderson residual least squares for real and complex fields on every backend.

The complex Gram matrix must use the Hermitian inner product. A residual
proportional to [1, i] has nonzero norm although its bilinear square vanishes.
Nonproportional complex histories also check conjugation of cached cross
products and the right-hand side. Longer sequences compare with a rectangular
least-squares solve through history wrap-around and reset, for both field types
and both memory modes. The real-field cases preserve the existing algorithm.
"""

import numpy as np

from polymerfts import PlatformSelector
from cuda_part import cuda_part_missing


def mixer(platform, reduced, field_type, n_var, max_hist=3):
    factory = PlatformSelector.create_factory(platform, reduced, field_type)
    return factory.create_anderson_mixing(n_var, max_hist, 10.0, 0.1, 0.1)


def step(am, field, residual):
    return np.asarray(am.calculate_new_fields(field, residual, 1.0, 1.0)).ravel()


def test_isotropic_residual(platform, reduced):
    # Simple mixing contracts this residual by 0.798 on the first call.
    # The bilinear complex Gram used to vanish, leaving subsequent updates
    # alternating between w and -w. Least squares reaches zero on call two.
    for field_type, initial in (("complex", np.array([1.0, 1.0j])),
                                ("real", np.array([1.0, 0.0, 0.0, 1.0]))):
        am = mixer(platform, reduced, field_type, initial.size)
        first = step(am, initial, -2.0 * initial)
        np.testing.assert_allclose(first, 0.798 * initial, rtol=0, atol=1e-15)
        second = step(am, first, -2.0 * first)
        np.testing.assert_allclose(second, 0.0, rtol=0, atol=2e-13)


def test_complex_cross_history(platform, reduced):
    am = mixer(platform, reduced, "complex", 3)
    field = np.zeros(3, dtype=complex)
    oldest = np.array([1.0, 1.0, 2.0], dtype=complex)
    previous = np.array([0.0, 1.0j, 1.0])
    current = np.array([1.0 + 1.0j, 1.0, 1.0j])
    step(am, field, oldest)
    # One history: the least-squares coefficient is -i/4.
    np.testing.assert_allclose(step(am, field, previous),
                               [-0.25j, -0.25 + 0.75j, 1.0 - 0.25j], rtol=0, atol=2e-13)
    # Two histories: D^H D = [[6, 4+2i], [4-2i, 6]], D^H r = [4, 2-3i].
    # a = [5/8+i/2, -1/4-5i/8]; the cached old cross product is 2-i.
    expected = np.array([2.0 + 1.0j, -1.0 + 1.0j, -1.0j]) / 8.0
    np.testing.assert_allclose(step(am, field, current), expected, rtol=0, atol=2e-13)


def test_real_baseline(platform, reduced):
    am = mixer(platform, reduced, "real", 3)
    field = np.zeros(3)
    step(am, field, np.array([1.0, 1.0, 2.0]))
    step(am, field, np.array([0.0, 1.0, 1.0]))
    # Coefficients [1, -1/2] remove the first and third residual components.
    np.testing.assert_allclose(step(am, field, np.array([1.0, 1.0, 0.0])),
                               [0.0, 1.0, 0.0], rtol=0, atol=2e-13)


def test_history_wrap_and_reset(platform, reduced, field_type):
    rng = np.random.default_rng(530)
    fields = rng.normal(0.0, 5.0, (11, 7))
    residuals = rng.normal(0.0, 5.0, (11, 7))
    if field_type == "complex":
        fields = fields + 1j * rng.normal(0.0, 5.0, fields.shape)
        residuals = residuals + 1j * rng.normal(0.0, 5.0, residuals.shape)
    max_hist = 3
    am = mixer(platform, reduced, field_type, fields.shape[1], max_hist)
    outputs = []
    for k, (field, residual) in enumerate(zip(fields, residuals)):
        if k == 0:
            expected = field + 0.101 * residual
        else:
            ages = list(range(k - 1, max(-1, k - max_hist - 1), -1))
            differences = np.column_stack([residual - residuals[j] for j in ages])
            # Solve the original rectangular problem, independently of the
            # native implementation's cached Gram matrix and history indices.
            coefficients = np.linalg.lstsq(differences, residual, rcond=1e-12)[0]
            expected = field + residual
            for coefficient, j in zip(coefficients, ages):
                expected = expected + coefficient * (fields[j] + residuals[j] - field - residual)
        result = step(am, field, residual)
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-11,
                                   err_msg=f"{field_type}, history step {k}")
        outputs.append(result.copy())

    # After several complete ring-buffer wraps, reset must recover exactly
    # the same trajectory when the same backend receives the same inputs.
    am.reset_count()
    for field, residual, expected in zip(fields, residuals, outputs):
        np.testing.assert_array_equal(step(am, field, residual), expected)


def test_complex_non_finite_history(platform, reduced):
    am = mixer(platform, reduced, "complex", 3)
    field = np.array([0.5 + 0.25j, -0.5 + 0.5j, 0.25 - 0.5j])
    step(am, field, np.full(3, 1e200 + 1e200j))
    step(am, field, np.array([1.0 + 1.0j, -1.0j, -0.5]))
    residual = np.array([-0.5j, 0.5 + 1.0j, 0.25])
    # The oldest self-product overflows, so the unusable two-entry history
    # must still fall back to the current field plus its residual.
    np.testing.assert_array_equal(step(am, field, residual), field + residual)


def main():
    platforms = PlatformSelector.avail_platforms()
    for platform in platforms:
        for reduced in (False, True):
            test_isotropic_residual(platform, reduced)
            test_complex_cross_history(platform, reduced)
            test_real_baseline(platform, reduced)
            for field_type in ("real", "complex"):
                test_history_wrap_and_reset(platform, reduced, field_type)
            test_complex_non_finite_history(platform, reduced)
            print(f"PASS: Anderson least squares, {platform}, reduce_memory={reduced}")
    if cuda_part_missing():
        return 1
    return 0 if platforms else 77


if __name__ == "__main__":
    raise SystemExit(main())
