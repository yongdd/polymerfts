#!/usr/bin/env python3
"""chain_fields: the field of a random copolymer is the mean of its monomer types, in double precision.

SCFT.compute_concentrations hands the fields of its caller to chain_fields, so the sum must not take
their type: integer fields are valid input, and a float32 field must not lower the precision.
"""

from types import SimpleNamespace

import numpy as np

from polymerfts.theory.chains import chain_fields


def main():
    sim = SimpleNamespace(monomer_types=["A", "B"], random_fraction={"R": {"A": 0.25, "B": 0.75}},
                          prop_solver=SimpleNamespace(n_grid=3))
    expected = np.array([0.25 * 1 + 0.75 * 4, 0.25 * 2 + 0.75 * 5, 0.25 * 3 + 0.75 * 6])
    fields = [[1, 2, 3], [4, 5, 6]]
    for dtype, kind in ((np.int64, np.float64), (np.float32, np.float64), (np.float64, np.float64), (np.complex128, np.complex128)):
        w = np.array(fields, dtype=dtype)
        result = chain_fields(sim, w)
        assert result["R"].dtype == kind, (dtype, result["R"].dtype)
        assert np.allclose(result["R"], expected, rtol=0, atol=1e-12), dtype
        assert result["A"] is not None and np.array_equal(result["B"], w[1])
    print("chain_fields: OK")


if __name__ == "__main__":
    main()
