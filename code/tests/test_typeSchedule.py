"""Sanity checks for the fixed-k greedy type schedule.

Run with ``python tests/test_typeSchedule.py`` (from ``code/``) or
``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccei import ccei
from garp import _SHRINK, ccei_subset, cross_expenditure, satisfies_garp
from typeSchedule import ccei_schedule_esearch, type_schedule
from varian import varian
from test_clustering import TWO_BLOCKS


def random_data(seed, n=20, j=3):
    rng = np.random.default_rng(seed)
    p = pd.DataFrame(rng.lognormal(0, 0.4, size=(n, j)))
    x = pd.DataFrame(rng.lognormal(0, 0.8, size=(n, j)))
    return x, p


def test_ccei_subset_bounds():
    # hi/lo-restricted search agrees with the unrestricted CCEI
    x, p = random_data(7, n=14)
    E = cross_expenditure(x, p)
    full = ccei(x, p)
    assert ccei_subset(E) == full
    assert ccei_subset(E, hi=full) == full           # quick accept at hi
    assert ccei_subset(E, hi=(full + 1) / 2) == full  # search below hi
    assert ccei_subset(E, lo=full) is None            # give-up threshold
    assert ccei_subset(E, lo=full / 2) == full


def test_two_blocks_recovered():
    x, p = TWO_BLOCKS
    for measure in ('ccei', 'varian'):
        parts, sched, detail, _ = type_schedule(x, p, k_max=2,
                                                measure=measure,
                                                n_restarts=3)
        # k=2 recovers the two rational blocks: full rationality
        assert sched.loc[2, 'rationality'] == 1.0
        col = parts['k2']
        assert col[0] == col[1] and col[2] == col[3] and col[0] != col[2]
        # k=1 pools everything
        assert parts['k1'].nunique() == 1
        assert sched.loc[1, 'rationality'] < 1.0


def test_k1_matches_pooled_indices():
    x, p = random_data(3)
    n = len(x)
    parts, sched, _, mult = type_schedule(x, p, k_max=1, measure='ccei')
    # At k=1 the reported rationality is the pooled CCEI
    assert sched.loc[1, 'rationality'] == ccei(x, p)
    assert sched.loc[1, 'best_restart'] == -1
    assert mult is None

    # Varian at k=1 with refinement: exact MILP at this size
    _, sched_v, detail_v, mult_v = type_schedule(x, p, k_max=1,
                                                 measure='varian',
                                                 refine=True)
    exact = varian(x, p, method='exact')
    assert abs(sched_v.loc[1, 'rationality'] - exact) < 1e-6
    assert detail_v.loc[0, 'method'] == 'exact'
    assert abs(mult_v['k1'].mean() - sched_v.loc[1, 'rationality']) < 1e-12

    # Without refinement the tracked multipliers are feasible, so the
    # reported rationality is a lower bound on the exact index
    _, sched_t, detail_t, mult_t = type_schedule(x, p, k_max=1,
                                                 measure='varian',
                                                 refine=False)
    assert detail_t.loc[0, 'method'] == 'tracked'
    assert sched_t.loc[1, 'rationality'] <= exact + 1e-9
    assert abs(mult_t['k1'].mean() - sched_t.loc[1, 'rationality']) < 1e-12


def test_k_equals_n_full_rationality():
    x, p = random_data(1, n=8)
    for measure in ('ccei', 'varian'):
        parts, sched, _, _ = type_schedule(x, p, k_max=8, measure=measure,
                                           n_restarts=2)
        assert sched.loc[8, 'rationality'] == 1.0
        assert sched.loc[8, 'total_loss'] == 0.0
        # Rationality is weakly increasing in k by construction
        assert (sched['rationality'].diff().dropna() >= 0).all()


def test_partitions_are_valid():
    x, p = random_data(2, n=15)
    E = cross_expenditure(x, p)
    parts, sched, detail, _ = type_schedule(x, p, k_max=4, measure='ccei',
                                            n_restarts=2)
    for k in range(1, 5):
        col = parts[f'k{k}']
        assert col.notna().all()
        assert col.nunique() == sched.loc[k, 'n_groups_used'] <= k
        # Each group's recorded efficiency is its actual pooled CCEI, and
        # the reported rationality is the minimum across groups
        effs = []
        for g, rows in detail[detail['k'] == k].groupby('group'):
            idx = np.flatnonzero(col.to_numpy() == g)
            assert len(idx) == rows['size'].iloc[0]
            assert abs(ccei_subset(E, idx) - rows['efficiency'].iloc[0]) < 1e-12
            effs.append(rows['efficiency'].iloc[0])
        assert sched.loc[k, 'rationality'] == min(effs)


def test_varian_multipliers_feasible_and_bounded():
    x, p = random_data(4, n=16)
    E = cross_expenditure(x, p)
    parts, sched, detail, mult = type_schedule(x, p, k_max=3,
                                               measure='varian',
                                               n_restarts=2, refine=False)
    for k in range(1, 4):
        col = parts[f'k{k}'].to_numpy()
        e_all = mult[f'k{k}'].to_numpy()
        # Reported rationality is the simple average of the multipliers
        assert abs(sched.loc[k, 'rationality'] - e_all.mean()) < 1e-12
        total_exact = 0.0
        for g in np.unique(col):
            idx = np.flatnonzero(col == g)
            # Tracked multipliers are feasible on their group (sup)
            shrunk = (_SHRINK * e_all[idx])[:, None]
            assert satisfies_garp(E[np.ix_(idx, idx)], efficiency=shrunk)
            avg = varian(x.iloc[idx], p.iloc[idx], method='exact')
            total_exact += len(idx) * (1 - avg)
        assert sched.loc[k, 'total_loss'] >= total_exact - 1e-6


def test_reference_partition_used_at_k_max():
    # With one restart the greedy order may miss the fully rational
    # 2-partition; the reference guarantees the schedule ends at 1
    x, p = TWO_BLOCKS
    reference = pd.Series([1, 1, 2, 2], index=x.index)
    for measure in ('ccei', 'varian'):
        parts, sched, detail, _ = type_schedule(
            x, p, k_max=2, measure=measure, n_restarts=1,
            reference=reference)
        assert sched.loc[2, 'rationality'] == 1.0
        col = parts['k2']
        assert col[0] == col[1] and col[2] == col[3] and col[0] != col[2]

    # A non-rationalizing reference is ignored with a warning
    bad = pd.Series([1, 2, 1, 2], index=x.index)
    _, sched, _, _ = type_schedule(x, p, k_max=2, measure='ccei',
                                   n_restarts=3, reference=bad)
    assert sched.loc[2, 'rationality'] == 1.0  # restarts still find it


def test_esearch_two_blocks_and_pooled():
    x, p = TWO_BLOCKS
    parts, sched, detail, mult = ccei_schedule_esearch(x, p, k_max=2,
                                                       n_restarts=3)
    assert mult is None
    assert sched.loc[2, 'rationality'] == 1.0
    col = parts['k2']
    assert col[0] == col[1] and col[2] == col[3] and col[0] != col[2]
    assert sched.loc[1, 'rationality'] == ccei(x, p)


def test_esearch_properties_and_dominance():
    x, p = random_data(2, n=15)
    E = cross_expenditure(x, p)
    k_max = 5
    parts_e, sched_e, detail_e, _ = ccei_schedule_esearch(
        x, p, k_max, n_restarts=2)
    parts_m, sched_m, _, _ = type_schedule(x, p, k_max, measure='ccei',
                                           n_restarts=2)
    for k in range(1, k_max + 1):
        # Valid partition; recorded efficiencies are the actual group
        # CCEIs and rationality is their minimum
        col = parts_e[f'k{k}']
        assert col.nunique() == sched_e.loc[k, 'n_groups_used'] <= k
        effs = []
        for g, rows in detail_e[detail_e['k'] == k].groupby('group'):
            idx = np.flatnonzero(col.to_numpy() == g)
            assert abs(ccei_subset(E, idx) - rows['efficiency'].iloc[0]) < 1e-12
            effs.append(rows['efficiency'].iloc[0])
        assert sched_e.loc[k, 'rationality'] == min(effs)
    assert (sched_e['rationality'].diff().dropna() >= 0).all()
    # The e-search targets the reported statistic directly, so it should
    # do at least as well as the max-min greedy on this small fixture
    assert (sched_e['rationality'] >= sched_m['rationality'] - 1e-9).all()


def test_validation_errors():
    x, p = random_data(0, n=6)
    with pytest.raises(ValueError):
        type_schedule(x, p, k_max=2, measure='bogus')
    with pytest.raises(ValueError):
        type_schedule(x, p, k_max=0)
    with pytest.raises(ValueError):
        type_schedule(x, p, k_max=7)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
