#!/usr/bin/env python3
"""Errors of the propagator computation reach Python once, as RuntimeError.

An error found in the native layer names the place where it was found in one
"File: ..." line: it passes through its callers unchanged. Checked for a missing field (compute_propagators), an unknown monomer
type (get_total_concentration) and a segment out of range (get_chain_propagator), for
continuous and discrete chains with full and reduced memory on every platform.
"""

import sys

import numpy as np
from polymerfts import _core
from cuda_part import cuda_part_missing


def error_of(call):
    try:
        call()
    except Exception as exc:
        return exc
    raise AssertionError("The invalid call was accepted")


def check_error(call, ending):
    exc = error_of(call)
    text = str(exc)
    assert type(exc) is RuntimeError, (type(exc).__name__, text)
    assert text.count("File: ") == 1, text
    assert text.endswith(ending), text


def check_platform(platform, chain_model, reduce_memory):
    factory = _core.PlatformSelector.create_factory(platform, reduce_memory, "real")
    box = factory.create_computation_box([8], [4.0])
    molecules = factory.create_molecules_information(chain_model, 0.1, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    computation = factory.create_propagator_computation(box, molecules, optimizer,
                                                        "rqm4" if chain_model == "continuous" else "ps")

    check_error(lambda: computation.compute_propagators({"A": np.zeros(8)}),
                'monomer_type "B" is not in w_input.')
    computation.compute_propagators({"A": np.zeros(8), "B": np.zeros(8)})
    computation.compute_concentrations()
    check_error(lambda: computation.get_total_concentration("C"), "Key 'C' not found in the monomer types")
    check_error(lambda: computation.get_chain_propagator(0, 0, 1, 99), "must be in range [" +
                ("0" if chain_model == "continuous" else "1") + ", 5]")


def main():
    for platform in _core.PlatformSelector.avail_platforms():
        for chain_model in ("continuous", "discrete"):
            for reduce_memory in (False, True):
                check_platform(platform, chain_model, reduce_memory)
        print(f"{platform}: passed")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    sys.exit(main())
