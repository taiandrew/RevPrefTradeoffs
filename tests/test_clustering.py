"""Sanity checks on synthetic data for the GARP utilities and the C&P
lower/upper bound clustering algorithms.

Run with ``python tests/test_clustering.py`` or ``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clusteringLowerBd import cluster_lower_bound
from clusteringUpperBd import cluster_upper_bound
from garp import cross_expenditure, pairwise_garp_violation, satisfies_garp


def make_dfs(prices, quantities):
    p = pd.DataFrame(prices, dtype=float)
    x = pd.DataFrame(quantities, dtype=float)
    return x, p


# Two observations at identical prices, distinct bundles: consistent.
CONSISTENT_PAIR = make_dfs(
    prices=[[1, 1], [1, 1]],
    quantities=[[2, 0], [0, 1]],
)

# Crossing budgets, each bundle strictly affordable at the other's prices:
# a mutual (2-cycle) GARP violation.
VIOLATING_PAIR = make_dfs(
    prices=[[1, 1], [0.9, 4]],
    quantities=[[4, 0], [0, 1]],
)

# Two internally consistent "types" (a1, a2) and (b1, b2); every cross pair
# violates GARP, so the minimal partition is exactly the two blocks.
TWO_BLOCKS = make_dfs(
    prices=[[1, 1], [1, 1.1], [0.9, 4], [1, 4]],
    quantities=[[4, 0], [4.4, 0], [0, 1], [0, 1.2]],
)

# Three goods, three observations: 0 R0 1, 1 R0 2, 2 R0 0 (a 3-cycle caught
# only through the transitive closure) with no pairwise 2-cycle violations.
THREE_CYCLE = make_dfs(
    prices=[[2, 5, 2], [3, 1, 4], [4, 1, 2]],
    quantities=[[0, 4, 4], [4, 2, 1], [3, 5, 1]],
)


def test_garp_basics():
    x, p = CONSISTENT_PAIR
    assert satisfies_garp(cross_expenditure(x, p))

    x, p = VIOLATING_PAIR
    E = cross_expenditure(x, p)
    assert not satisfies_garp(E)
    assert pairwise_garp_violation(E, 0, 1)

    x, p = THREE_CYCLE
    E = cross_expenditure(x, p)
    assert not satisfies_garp(E)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert not pairwise_garp_violation(E, i, j)
        assert satisfies_garp(E, [i, j])


def test_consistent_pair():
    x, p = CONSISTENT_PAIR
    for seed in range(5):
        assert cluster_lower_bound(x, p, seed=seed) == 1
        labels = cluster_upper_bound(x, p, seed=seed)
        assert labels.nunique() == 1


def test_violating_pair():
    x, p = VIOLATING_PAIR
    for seed in range(5):
        assert cluster_lower_bound(x, p, seed=seed) == 2
        labels = cluster_upper_bound(x, p, seed=seed)
        assert labels.nunique() == 2


def test_two_blocks():
    x, p = TWO_BLOCKS
    E = cross_expenditure(x, p)
    # Every cross pair violates, both within-block pairs are consistent.
    for i in (0, 1):
        for j in (2, 3):
            assert pairwise_garp_violation(E, i, j)
    assert satisfies_garp(E, [0, 1]) and satisfies_garp(E, [2, 3])

    for seed in range(10):
        assert cluster_lower_bound(x, p, seed=seed) == 2
        labels = cluster_upper_bound(x, p, seed=seed)
        assert labels.nunique() == 2
        assert labels.iloc[0] == labels.iloc[1]
        assert labels.iloc[2] == labels.iloc[3]
        assert labels.index.equals(x.index)
        assert set(labels) == {1, 2}


def test_three_cycle():
    x, p = THREE_CYCLE
    # No pairwise violations, so one iteration of the lower bound finds 1;
    # the data need 2 types, so the upper bound finds 2. The truth (2) lies
    # within [lower, upper].
    for seed in range(5):
        assert cluster_lower_bound(x, p, seed=seed) == 1
        labels = cluster_upper_bound(x, p, seed=seed)
        assert labels.nunique() == 2


def test_random_data_properties():
    rng = np.random.default_rng(12345)
    n, j = 30, 3
    p = pd.DataFrame(rng.lognormal(0, 0.4, size=(n, j)))
    x = pd.DataFrame(rng.lognormal(0, 0.8, size=(n, j)))
    E = cross_expenditure(x, p)

    lower = cluster_lower_bound(x, p, seed=7)
    labels = cluster_upper_bound(x, p, seed=7)

    # Every group produced by the upper bound must itself satisfy GARP,
    # and one-iteration bounds must bracket: lower <= number of groups.
    for g in labels.unique():
        members = np.flatnonzero(labels.to_numpy() == g)
        assert satisfies_garp(E, list(members))
    assert 1 <= lower <= labels.nunique()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
