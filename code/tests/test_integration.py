"""Integration test: the type schedule on a random 400-household subset of
the real milk data. Skipped when the working_data files are not present.

Run with ``python tests/test_integration.py`` (from ``code/``) or
``pytest tests/``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccei import ccei
from garp import _SHRINK, cross_expenditure, satisfies_garp
from typeSchedule import type_schedule

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'working_data')

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'milk_partitions.parquet')),
    reason="working_data files not available")

N_SUBSET = 400
SEED = 7


def _subset():
    quantities = pd.read_parquet(os.path.join(DATA, 'milk_quantities.parquet'))
    prices = pd.read_parquet(os.path.join(DATA, 'milk_prices.parquet'))
    members = pd.read_parquet(os.path.join(DATA, 'milk_partitions.parquet'))[
        'group_all'].dropna().astype(int)
    rng = np.random.default_rng(SEED)
    hhs = members.index[rng.choice(len(members), N_SUBSET, replace=False)]
    hhs = members.loc[hhs].index
    return quantities.loc[hhs], prices.loc[hhs], members.loc[hhs]


def test_ccei_schedule_on_milk_subset():
    q, p, members = _subset()
    k_max = members.nunique()
    parts, sched, detail, mult = type_schedule(
        q, p, k_max, measure='ccei', n_restarts=1, reference=members)
    assert mult is None
    # k=1 is the pooled CCEI; k=k_max hits full rationality (subsets of
    # GARP-consistent groups stay GARP-consistent, so the restricted
    # reference partition still rationalizes); monotone in between
    assert sched.loc[1, 'rationality'] == ccei(q, p)
    assert sched.loc[k_max, 'rationality'] == 1.0
    assert (sched['rationality'].diff().dropna() >= 0).all()
    # Reported rationality is the minimum group CCEI
    for k in (2, k_max // 2):
        rows = detail[detail['k'] == k]
        assert sched.loc[k, 'rationality'] == rows['efficiency'].min()


def test_varian_schedule_on_milk_subset():
    q, p, members = _subset()
    k_max = members.nunique()
    E = cross_expenditure(q, p)
    parts, sched, detail, mult = type_schedule(
        q, p, k_max, measure='varian', n_restarts=1, reference=members,
        refine=False)
    assert sched.loc[k_max, 'rationality'] == 1.0
    assert (sched['rationality'].diff().dropna() >= 0).all()
    for k in (1, k_max // 2):
        e_all = mult[f'k{k}'].to_numpy()
        assert abs(sched.loc[k, 'rationality'] - e_all.mean()) < 1e-12
        # Tracked multipliers are feasible on their groups (sup convention)
        col = parts[f'k{k}'].to_numpy()
        for g in np.unique(col):
            idx = np.flatnonzero(col == g)
            shrunk = (_SHRINK * e_all[idx])[:, None]
            assert satisfies_garp(E[np.ix_(idx, idx)], efficiency=shrunk)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
    print("All tests passed.")
