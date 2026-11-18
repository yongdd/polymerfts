#!/usr/bin/env python3
"""Input checks of the bindings, reduced-basis single steps and Anderson mixing
of non-finite histories.

- Monomer names ending with an underscore give the same results as plain names.
- NaN box lengths and angles, and NaN or infinite contour steps, are rejected.
- advance_propagator_single_segment works in the reduced basis when the
  computation has a space group, like compute_propagators, for continuous and
  discrete chains with and without reduce_memory, on fields rescaled to a
  standard deviation of 5 after symmetrization. It refuses to run before
  compute_propagators() and after a cell change, whose fields and tables it uses.
- Anderson mixing of a history whose residual products overflow uses no
  history and calls no LAPACK routine, which would print DLASCL warnings.
- A solver of real fields rejects complex arrays instead of dropping their
  imaginary parts, with a ValueError of the native binding: through PropagatorSolver,
  and given directly to its native computation box, propagator computation (fields
  and q_init), Anderson mixing, Fourier transform and space group, and as a mask.
  The message names field_type="complex" for the fields only, not for space-group
  fields and masks, which are real for every field type. Integer and boolean arrays
  still convert to real fields.
"""

import subprocess
import sys

import numpy as np
from polymerfts import _core
from polymerfts.theory.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing


def rejects(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("Invalid input was accepted")


def underscore_names(platform, chain_model, names):
    a, b = names
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    mol = fac.create_molecules_information(chain_model, 0.1, {a: 1.0, b: 1.5})
    mol.add_polymer(0.5, [[a, 0.5, 0, 1], [b, 0.5, 1, 2]])
    # A star with two identical arms: its keys are aggregated
    mol.add_polymer(0.5, [[a, 0.4, 0, 1], [a, 0.4, 0, 2], [b, 0.3, 0, 3]])
    opt = fac.create_propagator_computation_optimizer(mol, True)
    cb = fac.create_computation_box([16], [4.0])
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    w = np.random.default_rng(41).normal(0, 5, (2, 16))
    pc.compute_propagators({a: w[0], b: w[1]})
    pc.compute_concentrations()
    assert pc.check_total_partition()
    return np.array([pc.get_total_partition(p) for p in range(2)] +
                    [x for m in names for x in pc.get_total_concentration(m)])


def test_underscore_names(platform):
    for chain_model in ("continuous", "discrete"):
        plain = underscore_names(platform, chain_model, ("A", "B"))
        for names in (("A_", "B_"), ("A_", "_")):
            np.testing.assert_allclose(underscore_names(platform, chain_model, names), plain,
                                       rtol=1e-12, atol=0)


def test_nan_box(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    nan = float("nan")
    rejects(lambda: fac.create_computation_box([16], [nan]))
    rejects(lambda: fac.create_computation_box([8]*3, [2.0, nan, 2.0]))
    rejects(lambda: fac.create_computation_box([8]*3, [2.0]*3, [90.0, 90.0, nan]))
    cb = fac.create_computation_box([16], [4.0])
    rejects(lambda: cb.set_lx([nan]))
    assert cb.get_lx() == [4.0]
    # A NaN contour step used to give one segment per block and a plausible Q
    for chain_model in ("continuous", "discrete"):
        for ds in (nan, float("inf")):
            rejects(lambda: fac.create_molecules_information(chain_model, ds, {"A": 1.0}))


# Isolate null arguments: a regression must fail this test, not kill the test runner.
FACTORY_CHILD = r"""
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
from polymerfts import PlatformSelector

for field_type in ("real", "complex"):
    for reduced in (False, True):
        factory = PlatformSelector.create_factory(sys.argv[1], reduced, field_type)
        for aggregate in (False, True):
            try:
                factory.create_propagator_computation_optimizer(None, aggregate)
            except (TypeError, RuntimeError):
                pass
            else:
                raise AssertionError("A null optimizer molecules argument was accepted")
        box = factory.create_computation_box([4], [1.0])
        for model, method in (("continuous", "rqm4"), ("discrete", "ps")):
            molecules = factory.create_molecules_information(model, 0.1, {"A": 1.0})
            molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
            optimizer = factory.create_propagator_computation_optimizer(molecules, False)
            for args in ((None, molecules, optimizer), (box, None, optimizer), (box, molecules, None)):
                try:
                    factory.create_propagator_computation(*args, method)
                except RuntimeError as exc:
                    assert "null pointer" in str(exc), str(exc)
                else:
                    raise AssertionError("A null factory argument was accepted")
            # Rejected construction leaves the inputs usable for a valid solver.
            solver = factory.create_propagator_computation(box, molecules, optimizer, method)
            del solver
"""


def test_null_factory_arguments(platform):
    child = subprocess.run([sys.executable, "-c", FACTORY_CHILD, platform],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert child.returncode == 0, f"exit={child.returncode}\n{child.stdout}{child.stderr}"


def test_single_segment_reduced_basis(platform):
    nx = [8]*3
    sg = _core.SpaceGroup(nx, "Im-3m", 529)
    rng = np.random.default_rng(42)

    # Symmetrizing averages the random values over the group; rescale to std 5
    def symmetric_field():
        x = sg.symmetrize(rng.normal(0, 5, 512))
        return 5.0 * x / np.std(x)
    w = {m: symmetric_field() for m in "AB"}
    q_in = symmetric_field()
    for chain_model in ("continuous", "discrete"):
        for reduce_memory in (False, True):
            fac = _core.PlatformSelector.create_factory(platform, reduce_memory, "real")
            q_out = []
            for group in (None, sg):
                basis = (lambda x: x) if group is None else sg.to_reduced_basis
                cb = fac.create_computation_box(nx, [3.0]*3)
                mol = fac.create_molecules_information(chain_model, 0.05, {"A": 1.0, "B": 1.0})
                mol.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
                opt = fac.create_propagator_computation_optimizer(mol, False)
                pc = fac.create_propagator_computation(cb, mol, opt, "rqm4", group)
                rejects(lambda: pc.advance_propagator_single_segment(basis(q_in), 0, 0, 1))
                pc.compute_propagators({m: basis(w[m]) for m in "AB"})
                out = pc.advance_propagator_single_segment(basis(q_in), 0, 0, 1)
                assert out.shape == (cb.get_n_basis(),)
                q_out.append(out)
            # With a space group the full grid is rejected, like in compute_propagators
            rejects(lambda: pc.advance_propagator_single_segment(q_in, 0, 0, 1))
            # A cell change needs a new compute_propagators() before a single step
            cb.set_lx([3.1] * 3)
            pc.update_laplacian_operator()
            rejects(lambda: pc.advance_propagator_single_segment(sg.to_reduced_basis(q_in), 0, 0, 1))
            np.testing.assert_allclose(q_out[1], sg.to_reduced_basis(q_out[0]), rtol=1e-11, atol=1e-11,
                                       err_msg=f"{chain_model}, reduce_memory={reduce_memory}")


# Run in a child process, whose stdout and stderr capture what LAPACK prints
ANDERSON_CHILD = r"""
import sys
import numpy as np
from polymerfts import _core

n = 16
fac = _core.PlatformSelector.create_factory(sys.argv[1], False, "real")
am = fac.create_anderson_mixing(n, 5, 1e3, 0.1, 0.1)
w = np.linspace(0.1, 1.0, n)
# The squared norm of the oldest residual overflows, so in the third step the
# least-squares matrix of the two-entry history has one infinite element
residuals = [np.full(n, 1e200), np.linspace(-1.0, 1.0, n), np.cos(np.arange(n))]
for d in residuals:
    w_new = am.calculate_new_fields(w, d, 1.0, 0.5)
# No usable history: the new field is the newest field plus its residual
assert np.array_equal(w_new, w + residuals[-1])
"""


def test_anderson_non_finite(platform):
    child = subprocess.run([sys.executable, "-c", ANDERSON_CHILD, platform],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    output = child.stdout + child.stderr
    assert child.returncode == 0, output
    assert "DLASCL" not in output and "illegal value" not in output, output


def test_complex_into_real(platform):
    solver = PropagatorSolver(nx=[8, 8], lx=[2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0}, bc=["periodic"] * 4,
                              chain_model="continuous", platform=platform)
    solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    w = np.random.default_rng(3).normal(0.0, 5.0, 64)
    sg_solver = PropagatorSolver(nx=[8, 8, 8], lx=[2.0, 2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0},
                                 bc=["periodic"] * 6, chain_model="continuous", platform=platform,
                                 space_group=_core.SpaceGroup([8, 8, 8], "Im-3m", 529))
    reduced = sg_solver.to_reduced_basis(np.ones(512))
    solver.compute_propagators({"A": w})
    box, computation = solver.computation_box, solver.propagator_computation
    anderson = solver.create_anderson_mixing(64, 5, 0.1, 0.1, 0.1)
    grafted = PropagatorSolver(nx=[8, 8], lx=[2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0}, bc=["periodic"] * 4,
                               chain_model="continuous", platform=platform)
    grafted.add_polymer(1.0, [["A", 1.0, 0, 1]], {0: "G"})
    sg = sg_solver.space_group
    # Real cosine coefficients of a box with reflecting walls
    cosine = PropagatorSolver(nx=[8, 8], lx=[2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0}, bc=["reflecting"] * 4,
                              chain_model="continuous", numerical_method="cn-adi2", platform=platform).fourier
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    # Fields, which a solver of complex fields takes complex: the message names field_type
    fields = (lambda: solver.compute_propagators({"A": w + 1j * w}), lambda: solver.mean(w + 1j * w),
              lambda: solver.integral(w + 0j), lambda: solver.inner_product(w, w + 1j * w),
              lambda: solver.zero_mean(w + 1j * w),
              # The native objects themselves
              lambda: box.integral(w + 0j), lambda: box.zero_mean(w + 1j * w),
              lambda: computation.compute_propagators({"A": w + 1j * w}),
              lambda: grafted.compute_propagators({"A": w}, q_init={"G": w + 0j}),
              lambda: anderson.calculate_new_fields(w + 1j * w, w, 1.0, 1.0),
              lambda: anderson.calculate_new_fields(w, w + 1j * w, 1.0, 1.0),
              lambda: solver.fourier.forward(w + 1j * w), lambda: cosine.backward(np.ones((8, 8)) + 1j))
    # Space-group fields and masks, real for every field type: no field_type in the message
    always_real = (lambda: sg_solver.to_reduced_basis(np.ones(512) + 1j),
                   lambda: sg_solver.from_reduced_basis(reduced + 1j),
                   lambda: sg.symmetrize(np.ones(512) + 1j),
                   lambda: sg.reduced_to_irreducible(reduced + 1j),
                   lambda: sg.irreducible_to_reduced(sg.reduced_to_irreducible(reduced) + 1j),
                   lambda: fac.create_computation_box([8, 8], [2.0, 2.0], mask=np.ones(64) + 0j))
    for calls, names_field_type in ((fields, True), (always_real, False)):
        for call in calls:
            try:
                call()
            except ValueError as exc:
                assert "takes real values" in str(exc), exc
                assert ("field_type" in str(exc)) == names_field_type, exc
                continue
            raise AssertionError("A complex array was accepted for real fields")
    # Integer and boolean arrays convert
    integral = box.integral(np.ones(64, dtype=int))
    assert np.isclose(integral, box.integral(np.ones(64))) and np.isclose(box.integral(np.ones(64, dtype=bool)), integral)
    computation.compute_propagators({"A": np.zeros(64, dtype=int)})
    assert np.isclose(computation.get_total_partition(0), 1.0)
    new_fields = anderson.calculate_new_fields(np.zeros(64, dtype=bool), np.ones(64, dtype=int), 1.0, 1.0)
    assert new_fields.dtype == np.float64 and np.all(np.isfinite(new_fields))


def main():
    for platform in _core.PlatformSelector.avail_platforms():
        test_null_factory_arguments(platform)
        test_complex_into_real(platform)
        test_underscore_names(platform)
        test_nan_box(platform)
        test_single_segment_reduced_basis(platform)
        test_anderson_non_finite(platform)
        print("Input checks:", platform, "OK")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    sys.exit(main())
