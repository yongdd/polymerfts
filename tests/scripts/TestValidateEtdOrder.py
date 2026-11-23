"""tests/validation/validate_etd_order.py still starts: its command line and a few steps of each scheme.

The study itself runs as separate jobs (docs/validation/NumericalValidation.md); this only keeps the
script in step with the integrators it imports. It needs the built package.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'tests/validation/validate_etd_order.py'


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120)


def test_help():
    pytest.importorskip('polymerfts')
    out = run_cli('--help')
    assert out.returncode == 0, out.stderr
    assert '--scheme' in out.stdout and '--collect' in out.stdout


@pytest.mark.parametrize('scheme', ['euler', 'lm', 'etd', 'etd1'])
def test_few_steps(scheme):
    pytest.importorskip('polymerfts')
    out = run_cli('--scheme', scheme, '--dt', 0.1, '--chains', 20, '--relaxation-time', 0.1, '--sampling-time', 0.2)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result['scheme'] == scheme and result['chains'] == 20
    assert not result['diverged'] and abs(result['bias']) < 10.0
