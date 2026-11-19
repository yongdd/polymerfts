"""Malformed public keys raise exceptions without terminating the Python process."""

import subprocess
import sys


CHILD = r'''
from polymerfts import PropagatorComputationOptimizer as Optimizer

# Unbalanced brackets and labels anywhere in the key, also after other characters
bad =(")", "]A+0", "(A1))B+0", "(A1]B+0", "[A1)B+0", "(A1B+0",
       "[A+0;1))B+0", "[A+0;1", "[A+0;1]B+0]", "{G1", "}A+0",
       "({G1}A1", "({G1A1)B+0")
for key in bad:
    try:
        Optimizer.get_deps_from_key(key)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError(f"Malformed key was accepted: {key!r}")

good = {
    "A+0": [],
    "(A1)B+0": [("A", 1, 1.0)],
    "(A1B2:2.5)C+0": [("A", 1, 1.0), ("B", 2, 2.5)],
    "({G12}A1B2)C+0": [("{G12}A", 1, 1.0), ("B", 2, 1.0)],
    "[{G12}A1:2.5,B2]C+0": [("{G12}A", 1, 2.5), ("B", 2, 1.0)],
    "((A1B2)C3D4)E+0": [("(A1B2)C", 3, 1.0), ("D", 4, 1.0)],
}
for key, expected in good.items():
    assert Optimizer.get_deps_from_key(key) == expected, key
print("Malformed keys rejected; valid leaf, junction, aggregate and graft keys preserved")
'''


if __name__ == "__main__":
    result = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Parser process exit={result.returncode}\n{result.stdout}{result.stderr}"
    print(result.stdout, end="")
