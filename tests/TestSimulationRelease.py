#!/usr/bin/env python3
"""A simulation frees its solver as soon as it is dropped.

The objects of a simulation (sim.saddle, sim.cell, sim.thermodynamics, sim.structure_functions)
point back to it. With strong references every simulation would be a cycle, kept alive with its
propagators and FFT plans, on the GPU too, until the cyclic garbage collector happens to run; that
collector counts Python objects and does not see native memory, so a sweep of simulations would
pile them up. The check runs with the collector off, so only reference counting can free a solver.
"""

import contextlib
import gc
import io
import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import weakref

from polymerfts import CLFTS, LFTS, SCFT, PlatformSelector

NX, LX = [8], [3.0]
CHAINS = {"nx": NX, "lx": LX, "ds": 0.1, "chain_model": "discrete", "segment_lengths": {"A": 1.0, "B": 1.0},
          "chi_n": {"A,B": 12.0}, "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                                  {"type": "B", "length": 0.5}]}]}
FIELDS = {"A": 0.1 * np.cos(2 * np.pi * (np.arange(8) + 0.5) / 8), "B": -0.1 * np.cos(2 * np.pi * (np.arange(8) + 0.5) / 8)}


def recording():
    return {"dir": tempfile.mkdtemp(), "checkpoint_interval": 1, "sample_interval": 1, "statistics_save_interval": 1}


def scft(platform, relax_cell=True):
    return SCFT({**CHAINS, "platform": platform, "relax_cell": relax_cell, "max_iter": 2, "tolerance": 1e-3,
                 "optimizer": {"name": "am", "max_hist": 5, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1}})


def lfts(platform, compressor):
    return LFTS({**CHAINS, "platform": platform, "langevin": {"nbar": 1e4, "dt": 0.5, "max_step": 1}, "recording": recording(),
                 "compressor": {"name": compressor, "max_hist": 5, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
                 "saddle": {"max_iter": 100, "tolerance": 1e-3}, "verbose_level": 0}, random_seed=1)


def clfts(platform, integrator="lm", charges=False):
    params = {**CHAINS, "platform": platform, "langevin": {"nbar": 1e4, "dt": 0.5, "max_step": 1, "integrator": integrator},
              "recording": recording(), "verbose_level": 0}
    if charges:
        params["electrostatics"] = {"charges": {"A": 0.2, "B": -0.2}, "coupling": 100.0, "psi_dt_scaling": 0.8}
        params["smearing"] = {"type": "gaussian", "a_int": 0.2}
    return CLFTS(params, random_seed=1)


def main():
    platforms = [p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")]
    if not platforms:
        return 77
    platform = platforms[0]
    cases = {"SCFT": lambda: scft(platform), "SCFT, fixed cell": lambda: scft(platform, False),
             "LFTS, am": lambda: lfts(platform, "am"), "LFTS, lr": lambda: lfts(platform, "lr"),
             "LFTS, lram": lambda: lfts(platform, "lram"), "CLFTS": lambda: clfts(platform),
             "CLFTS, etd": lambda: clfts(platform, "etd"), "CLFTS, charges": lambda: clfts(platform, charges=True)}
    # The first drawing of the chains imports matplotlib, and its frames stay until a collection
    with contextlib.redirect_stdout(io.StringIO()):
        lfts(platform, "am").run(FIELDS)
    gc.collect()
    gc.disable()
    try:
        for name, make in cases.items():
            for run in (False, True):
                with contextlib.redirect_stdout(io.StringIO()):
                    sim = make()
                    if run:
                        sim.run(FIELDS)
                solver = weakref.ref(sim.prop_solver)
                del sim
                assert solver() is None, f"{name}: the solver outlives its simulation (run: {run})"
    finally:
        gc.enable()
    print(f"Solvers are freed with their simulations: {len(cases)} kinds of simulation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
