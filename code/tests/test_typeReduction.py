"""Sanity checks for the type-reduction algorithm.

Run with ``python tests/test_typeReduction.py`` (from ``code/``) or
``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccei import ccei
from garp import cross_expenditure
from typeReduction import _ccei_idx, type_reduction
from test_clustering import TWO_BLOCKS


def test_ccei_idx_matches_ccei():
    rng = np.random.default_rng(11)
    n, j = 15, 3
    p = pd.DataFrame(rng.lognormal(0, 0.4, size=(n, j)))
    x = pd.DataFrame(rng.lognormal(0, 0.8, size=(n, j)))
    E = cross_expenditure(x, p)
    for idx in [np.arange(n), np.array([0, 3, 7]), np.arange(5, 12)]:
        assert _ccei_idx(E, idx) == ccei(x.iloc[idx], p.iloc[idx])


def test_two_blocks_reduction():
    x, p = TWO_BLOCKS
    partition = pd.Series([1, 1, 2, 2], index=x.index)
    parts, curve = type_reduction(x, p, partition)

    # k=2 is the fully rationalizing start: zero loss, labels as given
    assert curve.loc[2, 'total_loss'] == 0.0
    assert parts['k2'].tolist() == [1, 1, 2, 2]

    # k=1 pools everything: loss = N * (1 - CCEI of the whole sample) > 0
    expected = len(x) * (1 - ccei(x, p))
    assert abs(curve.loc[1, 'total_loss'] - expected) < 1e-12
    assert expected > 0
    assert parts['k1'].nunique() == 1
    assert curve.loc[1, 'merged_groups'] == '1+2'


def test_curve_and_partition_properties():
    # Random data: partition from a fine grouping down to one type
    rng = np.random.default_rng(5)
    n, j = 20, 3
    p = pd.DataFrame(rng.lognormal(0, 0.4, size=(n, j)))
    x = pd.DataFrame(rng.lognormal(0, 0.8, size=(n, j)))
    # Singleton groups are trivially GARP-consistent
    partition = pd.Series(np.arange(1, n + 1), index=x.index)

    parts, curve = type_reduction(x, p, partition)

    assert curve.loc[n, 'total_loss'] == 0.0
    # Weakly increasing loss as types are removed (index runs K..1)
    assert (curve['total_loss'].diff().dropna() >= -1e-12).all()
    assert (curve['merge_cost'].dropna() >= -1e-12).all()

    for k in range(n, 0, -1):
        col = parts[f'k{k}']
        assert col.nunique() == k
        if k < n:
            # Nested coarsening: households together at k+1 stay together at k
            prev = parts[f'k{k + 1}']
            grouped = pd.crosstab(prev, col)
            assert ((grouped > 0).sum(axis=1) == 1).all()


def test_nan_households_stay_nan():
    x, p = TWO_BLOCKS
    partition = pd.Series([1, 1, 2, np.nan], index=x.index)
    parts, curve = type_reduction(x, p, partition)
    assert parts.index.equals(x.index)
    assert parts.loc[3].isna().all()
    assert parts.loc[[0, 1, 2]].notna().all().all()
    assert list(curve.index) == [2, 1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
