#!/usr/bin/env python3
"""Deep valid chains must not exhaust the native stack while generating keys.

Check construction and contour-length remapping in separate child processes,
so either native crash becomes a test failure. A small POSIX stack reproduces
the original failure with modest heap use; other systems still check the keys.
"""

import subprocess
import sys


def limit_stack():
    try:
        import resource
    except ImportError:
        print("POSIX stack limit unavailable; checking keys without it", flush=True)
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    limit = 256 * 1024
    if hard != resource.RLIM_INFINITY and hard < limit:
        print("Hard stack limit below 256 KiB; retaining it", flush=True)
        return
    resource.setrlimit(resource.RLIMIT_STACK, (limit, hard))


def check_linear_keys(molecules, count, mapped):
    polymer = molecules.get_polymer(0)
    assert polymer.get_n_blocks() == count
    assert abs(polymer.get_alpha() - 1.0) < 1e-12
    # For a linear chain, each next block wraps the preceding complete code.
    # The two contour lengths have segment counts 2/3 and mapped indices 1/2.
    for forward in (True, False):
        code = ""
        indices = range(count) if forward else range(count - 1, -1, -1)
        label = "left" if forward else "right"
        for i in indices:
            monomer = "A" if i % 2 == 0 else "B"
            number = (1 if mapped else 2) + i % 2
            prefix = "(" + code + ")" if code else "{" + label + "}"
            key = prefix + monomer + ("+0" if mapped else "")
            v, u = (i, i + 1) if forward else (i + 1, i)
            assert polymer.get_propagator_key(v, u) == key, (mapped, v, u)
            code = prefix + monomer + str(number)


def check_branch_keys(factory):
    # The repeated B branches merge; entering one replicated A branch also
    # depends on the remaining 1.5 copies. A graft label must survive both.
    molecules = factory.create_molecules_information("continuous", 0.1, {"A": 1.0, "B": 1.0})
    molecules.add_polymer(1.0, [["A", 0.2, 0, 1, 2.5],
                               ["B", 0.3, 0, 2], ["B", 0.3, 0, 3]], {1: "G"})
    expected = {(0, 1): "(B3:2{G}A2:1.5)A", (1, 0): "{G}A",
                (0, 2): "(B3{G}A2:2.5)B", (2, 0): "B"}
    for edge, key in expected.items():
        assert molecules.get_polymer(0).get_propagator_key(*edge) == key
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    expected = {(0, 1): "(B2:2{G}A1:1.5)A+0", (1, 0): "{G}A+0",
                (0, 2): "(B2{G}A1:2.5)B+0", (2, 0): "B+0"}
    for edge, key in expected.items():
        assert molecules.get_polymer(0).get_propagator_key(*edge) == key
    assert optimizer.get_computation_propagators()


def child(mode):
    import faulthandler
    from polymerfts import PlatformSelector

    faulthandler.enable()
    platforms = PlatformSelector.avail_platforms()
    if not platforms:
        return 77
    # The optimizer is platform-neutral; this test allocates no solver or grid.
    factory = PlatformSelector.create_factory(platforms[0], False, "real")
    count = 2048
    ds = 1.0 / (2.5 * count)
    molecules = factory.create_molecules_information("continuous", ds, {"A": 1.0, "B": 1.0})
    blocks = [["A" if i % 2 == 0 else "B", (2 + i % 2) * ds, i, i + 1]
              for i in range(count)]
    if mode == "construction":
        limit_stack()
    molecules.add_polymer(1.0, blocks, {0: "left", count: "right"})
    check_linear_keys(molecules, count, False)
    if mode == "remapping":
        limit_stack()
        optimizer = factory.create_propagator_computation_optimizer(molecules, False)
        check_linear_keys(molecules, count, True)
        key = molecules.get_polymer(0).get_propagator_key(count - 1, count)
        assert optimizer.get_computation_propagator(key).max_n_segment == 3
    check_branch_keys(factory)
    print(f"Deep propagator {mode}: exact linear, branch, copy and graft keys passed", flush=True)
    return 0


def main():
    if len(sys.argv) > 1:
        return child(sys.argv[1])
    for mode in ("construction", "remapping"):
        result = subprocess.run([sys.executable, __file__, mode], capture_output=True,
                                text=True, timeout=90)
        if result.returncode == 77:
            print(result.stdout, end="")
            return 77
        assert result.returncode == 0, (
            f"{mode} process exit={result.returncode}\n{result.stdout}{result.stderr}")
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
