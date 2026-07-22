"""Sanity checks for the Varian index computation.

Run with ``python tests/test_varian.py`` (from ``code/``) or ``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccei import ccei
from garp import cross_expenditure, satisfies_garp
from varian import varian
from test_clustering import CONSISTENT_PAIR, THREE_CYCLE, VIOLATING_PAIR


def random_data(seed, n=12, j=3):
    rng = np.random.default_rng(seed)
    p = pd.DataFrame(rng.lognormal(0, 0.4, size=(n, j)))
    x = pd.DataFrame(rng.lognormal(0, 0.8, size=(n, j)))
    return x, p


def test_consistent_data_gives_one():
    x, p = CONSISTENT_PAIR
    for method in ('exact', 'heuristic'):
        avg, e = varian(x, p, method=method, return_multipliers=True)
        assert avg == 1.0
        assert (e == 1.0).all()


def test_violating_pair_exact_value():
    # E = [[4, 1], [3.6, 4]]: the only violation is the 0-1 cycle. Keeping
    # e_0 = 1 requires e_1 < 0.9 (sup 0.9), and sacrificing e_0 instead would
    # cost far more (e_0 <= 0.25), so the index is (1 + 0.9) / 2 = 0.95.
    x, p = VIOLATING_PAIR
    for method in ('exact', 'heuristic'):
        avg, e = varian(x, p, method=method, return_multipliers=True)
        assert abs(avg - 0.95) < 1e-6, method
        assert abs(e.iloc[0] - 1.0) < 1e-6 and abs(e.iloc[1] - 0.9) < 1e-6
    # Strictly better than the common-multiplier CCEI (0.9)
    assert ccei(x, p) == 0.9


def test_three_cycle_knife_edge():
    # CCEI is 1.0 (violation only completes at e = 1 exactly), so the Varian
    # index under the same sup convention is 1.0 as well.
    x, p = THREE_CYCLE
    for method in ('exact', 'heuristic'):
        assert varian(x, p, method=method) == 1.0


def test_random_data_bounds_and_feasibility():
    for seed in range(6):
        x, p = random_data(seed)
        E = cross_expenditure(x, p)
        exact, e_exact = varian(x, p, method='exact', return_multipliers=True)
        heur, e_heur = varian(x, p, method='heuristic', return_multipliers=True)

        # Varian index dominates the CCEI; the heuristic is a lower bound
        # on the exact index
        assert exact >= ccei(x, p) - 1e-9
        assert heur <= exact + 1e-9
        assert 0.0 <= heur and exact <= 1.0

        # Returned multipliers are feasible (GARP just below them, per the
        # sup convention) — also exercises vector efficiency in satisfies_garp
        for e in (e_exact, e_heur):
            shrunk = (1 - 1e-9) * e.to_numpy()
            assert satisfies_garp(E, efficiency=shrunk[:, None])


def test_auto_method_switch():
    x, p = random_data(3, n=8)
    assert abs(varian(x, p, method='auto', exact_max_n=8)
               - varian(x, p, method='exact')) < 1e-9
    assert abs(varian(x, p, method='auto', exact_max_n=4)
               - varian(x, p, method='heuristic')) < 1e-9


def test_validation_errors():
    x, p = VIOLATING_PAIR
    zero = x.copy()
    zero.iloc[0] = 0.0
    with pytest.raises(ValueError):
        varian(zero, p)  # zero expenditure
    with pytest.raises(ValueError):
        varian(x, p, method='bogus')


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
