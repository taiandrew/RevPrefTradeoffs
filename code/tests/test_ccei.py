"""Sanity checks for the CCEI calculation.

Run with ``python tests/test_ccei.py`` (from ``code/``) or ``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccei import ccei
from garp import cross_expenditure, satisfies_garp
from test_clustering import CONSISTENT_PAIR, THREE_CYCLE, VIOLATING_PAIR


def bisect_ccei(E, tol=1e-9):
    """Reference implementation: plain bisection on the efficiency level."""
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if satisfies_garp(E, efficiency=mid):
            lo = mid
        else:
            hi = mid
    return lo


def test_consistent_data_gives_one():
    x, p = CONSISTENT_PAIR
    assert ccei(x, p) == 1.0


def test_violating_pair_exact_value():
    # E = [[4, 1], [3.6, 4]]: the 2-cycle needs both weak relations, and the
    # binding one (3.6 <= 4e) turns on at e = 0.9, so CCEI = 0.9 exactly.
    x, p = VIOLATING_PAIR
    value = ccei(x, p)
    assert abs(value - 0.9) < 1e-12
    E = cross_expenditure(x, p)
    assert satisfies_garp(E, efficiency=0.9 - 1e-9)
    assert not satisfies_garp(E, efficiency=0.9 + 1e-9)


def test_matches_bisection_on_random_data():
    rng = np.random.default_rng(99)
    for _ in range(5):
        n, j = 12, 3
        p = pd.DataFrame(rng.lognormal(0, 0.3, size=(n, j)))
        x = pd.DataFrame(rng.lognormal(0, 1.0, size=(n, j)))
        value = ccei(x, p)
        assert 0.0 <= value <= 1.0
        assert abs(value - bisect_ccei(cross_expenditure(x, p))) < 1e-8


def test_subset_input():
    # Fewer observations than the clustering data is fine: a single
    # observation is trivially consistent.
    x, p = VIOLATING_PAIR
    assert ccei(x.iloc[:1], p.iloc[:1]) == 1.0


def test_three_cycle_knife_edge():
    # This dataset's violating cycle only completes at e = 1 exactly (the
    # relation 1 R0 2 holds with equality, E[1,1] = E[1,2] = 18), so relaxed
    # GARP holds for every e < 1 and the supremum is 1.0 even though the data
    # violate standard GARP.
    x, p = THREE_CYCLE
    E = cross_expenditure(x, p)
    assert not satisfies_garp(E)
    assert ccei(x, p) == 1.0
    assert satisfies_garp(E, efficiency=1 - 1e-9)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
