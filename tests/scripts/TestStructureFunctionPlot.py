"""Execute the notebook's aggregation cell against independently known full spectra."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_structure_function.ipynb'
CELL = ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][0]['source'])


def aggregate(tmp_path, monkeypatch, spectra, counts=None, lx=(1., 1., 1.),
              pair=("A", "A"), suffix="A_A", **extra):
    """Write exactly the four windows selected by the notebook, then run its actual code."""
    directory = tmp_path / 'data_simulation'
    directory.mkdir()
    for step, full, count in zip(range(100000, 500000, 100000), spectra, counts or [None] * 4):
        nx = full.shape
        data = {'nx': nx, 'lx': list(lx), **extra,
                'structure_function_' + suffix: full[:, :, :nx[2] // 2 + 1]}
        if count is not None:
            data['sample_count'] = count
        savemat(directory / f'structure_function_{step:06d}.mat', data)
    monkeypatch.chdir(tmp_path)
    result = {}
    cell = CELL.replace('type_pair = ("A", "A")', 'type_pair = ' + repr(pair))
    exec(compile(cell, str(NOTEBOOK) + ':cell0', 'exec'), result)
    return result['k2_unique'], result['y']


@pytest.mark.parametrize('nx', ([4, 4, 4], [5, 5, 5], [4, 5, 6]))
def test_half_spectrum_matches_full_fourier_shells(tmp_path, monkeypatch, nx):
    # A real field gives a valid Hermitian spectrum, including all Nyquist planes.
    field = np.random.default_rng(193).normal(size=nx)
    full = np.abs(np.fft.fftn(field)) ** 2
    k2, observed = aggregate(tmp_path, monkeypatch, [full] * 4, [3] * 4)
    modes = np.meshgrid(*(np.fft.fftfreq(n) * n for n in nx), indexing='ij')
    full_k2 = np.round(sum(k * k for k in modes), 7)
    expected = np.array([np.mean(full[full_k2 == shell]) for shell in k2])
    expected[0] = 0.
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize('axis', [0, 2])
def test_axis_rotation_does_not_change_shell_average(tmp_path, monkeypatch, axis):
    full = np.zeros((8, 8, 8))
    for dimension in range(3):
        for sign in (-1, 1):
            index = [0, 0, 0]
            index[dimension] = sign
            full[tuple(index)] = 10. if dimension == axis else 1.
    k2, mean = aggregate(tmp_path, monkeypatch, [full] * 4)
    assert mean[k2 == 1.].item() == pytest.approx(4.)


def test_measurement_windows_are_weighted_by_sample_count(tmp_path, monkeypatch):
    # sample_interval=30000 and statistics_save_interval=100000 give these counts.
    spectra = [np.full((4, 4, 4), value) for value in (1., 1., 10., 1.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra, [3, 3, 4, 3])
    np.testing.assert_allclose(mean[k2 > 0], 49 / 13)


def test_legacy_windows_keep_equal_file_weights(tmp_path, monkeypatch):
    spectra = [np.full((4, 4, 4), value) for value in (1., 1., 10., 1.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra)
    np.testing.assert_allclose(mean[k2 > 0], 3.25)


@pytest.mark.parametrize('pair, suffix', [(("A", "B_C"), "pair__first"),
                                         (("A_B", "C"), "pair__second")])
def test_pair_mapping_selects_distinct_underscore_names(tmp_path, monkeypatch, pair, suffix):
    mapping = {"pair__first": ["A", "B_C"], "pair__second": ["A_B", "C"]}
    spectra = [np.full((4, 4, 4), value) for value in (1., 1., 10., 1.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra, pair=pair, suffix=suffix,
                         pair_key_format_version=1, pair_keys_json=json.dumps(mapping))
    np.testing.assert_allclose(mean[k2 > 0], 3.25)


@pytest.mark.parametrize('metadata', [
    {"pair_key_format_version": 1},
    {"pair_keys_json": '{"A_A":["A","A"]}'},
    {"pair_key_format_version": 2, "pair_keys_json": '{"A_A":["A","A"]}'},
    {"pair_key_format_version": 1, "pair_keys_json": '{}'},
    {"pair_key_format_version": 1, "pair_keys_json": '{"one":["A","A"],"two":["A","A"]}'},
])
def test_invalid_pair_mapping_does_not_fall_back_to_legacy(tmp_path, monkeypatch, metadata):
    with pytest.raises(ValueError, match='pair-name mapping|no unique entry'):
        aggregate(tmp_path, monkeypatch, [np.ones((4, 4, 4))] * 4, **metadata)


def test_legacy_and_counted_windows_cannot_be_mixed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='with and without sample_count'):
        aggregate(tmp_path, monkeypatch, [np.ones((4, 4, 4))] * 4, [3, None, 4, 3])


def test_empty_windows_are_skipped(tmp_path, monkeypatch):
    spectra = [np.full((4, 4, 4), value) for value in (np.nan, 1., np.nan, 4.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra, [0, 2, 0, 4])
    np.testing.assert_allclose(mean[k2 > 0], 3.)


def test_all_empty_windows_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='contain no samples'):
        aggregate(tmp_path, monkeypatch, [np.zeros((4, 4, 4))] * 4, [0] * 4)


@pytest.mark.parametrize('count', [-1, .5, np.nan, np.inf, [1, 2], 1 + 1j])
def test_invalid_sample_counts_are_rejected(tmp_path, monkeypatch, count):
    with pytest.raises(ValueError, match='finite nonnegative integer'):
        aggregate(tmp_path, monkeypatch, [np.ones((4, 4, 4))] * 4, [count, 1, 1, 1])


def smallest_k(nx, lx, angles):
    """|k| of every index of the full spectrum: the alias m + a n of the smallest m^T G^-1 m, with the
    metric G_ij = l_i l_j cos(angle_ij) of the cell, by brute force over a window of aliases that
    holds the minimum (an axis of one grid point has no aliases, as in the solvers)."""
    alpha, beta, gamma = np.deg2rad(angles)
    cosines = np.array([[1, np.cos(gamma), np.cos(beta)], [np.cos(gamma), 1, np.cos(alpha)],
                        [np.cos(beta), np.cos(alpha), 1]])
    metric = np.outer(lx, lx)*cosines
    inverse = np.linalg.inv(metric)
    m = np.stack(np.meshgrid(*(np.arange(n) for n in nx), indexing='ij'), axis=-1)
    m = np.where(m > np.array(nx)//2, m - np.array(nx), m)
    form = lambda q: np.einsum('...i,ij,...j->...', q, inverse, q)
    # |q_d| <= sqrt(f G_dd) for the minimal alias q, with f the form of the centred index
    window = [0 if n == 1 else int(np.sqrt(form(m).max()*metric[d, d])/n) + 2 for d, n in enumerate(nx)]
    best = np.full(nx, np.inf)
    for shift in itertools.product(*(range(-w, w + 1) for w in window)):
        best = np.minimum(best, form(m + np.array(shift)*np.array(nx)))
    return 2*np.pi*np.sqrt(best)


@pytest.mark.parametrize('nx, lx, angles, source', [
    ((6, 6, 4), (1.0, 1.0, 1.5), [90.0, 90.0, 120.0], 'angles'),           # hexagonal
    ((6, 6, 4), (1.0, 1.0, 1.5), [90.0, 90.0, 120.0], 'initial_params'),   # older files: the input
    ((4, 32, 6), (1.0, 1.0, 1.0), [90.0, 90.0, 120.0], 'angles'),          # anisotropic spacing
    ((6, 6, 6), (1.0, 1.0, 1.0), [60.0, 60.0, 60.0], 'angles'),            # aliases along z too
    ((6, 6, 6), (1.0, 1.2, 1.4), [120.0, 120.0, 90.0], 'angles'),
    ((6, 5, 4), (1.0, 1.2, 1.4), [100.0, 80.0, 110.0], 'angles'),         # triclinic, every angle its own
    ((4, 4, 16), (1.0, 1.0, 1.0), [60.0, 60.0, 60.0], 'angles'),
    ((8, 8, 1), (1.0, 1.0, 0.3), [60.0, 60.0, 60.0], 'angles'),            # one grid point along z
])
def test_non_orthogonal_cell_groups_modes_of_equal_wavenumber(tmp_path, monkeypatch, nx, lx, angles, source):
    # A spectrum that depends on |k| alone must come back as that function on every shell
    k = smallest_k(nx, lx, angles)
    full = np.exp(-k**2/400)
    extra = {'angles': angles} if source == 'angles' else {'initial_params': {'nx': list(nx), 'angles': angles}}
    shells, mean = aggregate(tmp_path, monkeypatch, [full]*4, [1]*4, lx=lx, **extra)
    x = np.sqrt(shells)*2*np.pi
    # The shells are labelled by k^2/(2 pi)^2 rounded to 7 decimals, so x is exact to about 1e-7
    np.testing.assert_allclose(mean[shells > 0], np.exp(-x[shells > 0]**2/400), rtol=1e-6)
    # Every distinct |k| of the spectrum is one shell
    assert len(shells) == len(np.unique(np.round(k[:, :, :nx[2]//2 + 1]/(2*np.pi), 6)))
