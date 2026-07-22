"""Varian efficiency index (average of per-observation multipliers).

Varian's (1990) refinement of the Afriat/CCEI index gives every observation
its own efficiency multiplier e_t in [0, 1], relaxing the direct revealed
preference relation to x^t R0 x^s iff e_t * (p^t . x^t) >= p^t . x^s. The
index maximizes the average of the e_t subject to the data satisfying GARP
under the relaxed relations. It equals 1 for GARP-consistent data, is never
below the CCEI (which constrains all multipliers to be equal), and relates to
the loss in notes.lyx by L_Varian(G) = |G| * (1 - average index).

Computing the index is NP-hard, and no constant-factor polynomial-time
approximation exists unless P = NP (Smeulders, Spieksma, Cherchye & De Rock
2014; see the survey by Smeulders, Crama & Spieksma 2019, EJOR 272, Section
5.1). Practical strategy, following that literature:

- exact mixed-integer linear program for small samples (binary variables for
  the transitive closure of the relaxed relations, big-M links to the e_t);
- greedy coordinate ascent for larger samples: start every multiplier at the
  CCEI (a feasible point) and repeatedly raise each e_t as far as GARP
  allows, until a full sweep makes no improvement. This returns a feasible
  multiplier vector, hence a lower bound on the true index.

Both methods use the sup convention of ccei.py: the reported multipliers are
suprema, approached from below but not necessarily attained.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from garp import cross_expenditure, satisfies_garp

_SHRINK = 1 - 1e-9  # uniform shrink implementing the sup convention


def _feasible(E: np.ndarray, e: np.ndarray) -> bool:
    """GARP under per-observation multipliers, just below ``e`` (sup)."""
    return satisfies_garp(E, efficiency=(_SHRINK * e)[:, None])


def _varian_exact(ratios: np.ndarray) -> np.ndarray:
    """Maximize sum(e) by MILP over the ratio matrix r_ts = E_ts / E_tt.

    Variables: e_t (continuous, [0,1]) and, for each ordered pair (t, s),
    a binary x_ts indicating x^t R x^s in the transitive closure. Constraints
    (big-M = 1 works since all variables live in [0, 1]):

    - forcing:      e_t <= r_ts + x_ts        (relation on whenever e_t > r_ts;
                                                equality with x = 0 is allowed,
                                                which is the sup convention)
    - transitivity: x_tk + x_ks - x_ts <= 1
    - GARP:         x_ts = 1  =>  e_s <= r_st  (no strict reverse preference)
    """
    n = len(ratios)
    pair_idx = {}
    for t in range(n):
        for s in range(n):
            if t != s:
                pair_idx[(t, s)] = n + len(pair_idx)
    n_var = n + len(pair_idx)

    rows, cols, vals, ub = [], [], [], []

    def add_row(entries, bound):
        r = len(ub)
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        ub.append(bound)

    for (t, s), x in pair_idx.items():
        # Forcing (vacuous when r_ts >= 1 since e_t <= 1)
        if ratios[t, s] < 1:
            add_row([(t, 1.0), (x, -1.0)], ratios[t, s])
        # GARP link (vacuous when r_st >= 1)
        if ratios[s, t] < 1:
            add_row([(s, 1.0), (x, 1.0)], ratios[s, t] + 1.0)

    for t in range(n):
        for k in range(n):
            if k == t:
                continue
            for s in range(n):
                if s == t or s == k:
                    continue
                add_row([(pair_idx[(t, k)], 1.0), (pair_idx[(k, s)], 1.0),
                         (pair_idx[(t, s)], -1.0)], 1.0)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ub), n_var))
    c = np.concatenate([-np.ones(n), np.zeros(n_var - n)])
    integrality = np.concatenate([np.zeros(n), np.ones(n_var - n)])

    res = milp(c=c,
               constraints=LinearConstraint(A, -np.inf, np.array(ub)),
               integrality=integrality,
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP for the Varian index failed: {res.message}")
    return res.x[:n]


def _varian_heuristic(E: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """Greedy coordinate ascent from the CCEI (always feasible)."""
    n = len(ratios)
    # Per observation, the only values worth trying are its own ratios and 1
    candidates = [np.unique(np.append(
        ratios[t][(ratios[t] > 0) & (ratios[t] < 1)], 1.0)) for t in range(n)]

    diag = np.diag(E)
    if satisfies_garp(E):
        start = 1.0
    else:
        rr = E / diag[:, None]
        off = rr[~np.eye(n, dtype=bool)]
        cand = np.unique(off[(off > 0) & (off < 1)])
        bounds = np.concatenate(([0.0], cand, [1.0]))
        # ccei-style flip point (reuse ccei's search via garp on E directly)
        lo, hi = 0, len(bounds) - 2
        if not satisfies_garp(E, efficiency=(bounds[0] + bounds[1]) / 2):
            start = 0.0
        else:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if satisfies_garp(E, efficiency=(bounds[mid] + bounds[mid + 1]) / 2):
                    lo = mid
                else:
                    hi = mid - 1
            start = float(bounds[lo + 1])

    e = np.full(n, start)
    improved = True
    while improved:
        improved = False
        for t in range(n):
            cands = candidates[t][candidates[t] > e[t]]
            if len(cands) == 0:
                continue
            # Raising e_t only adds relations, so feasibility is monotone in
            # e_t and binary search finds the largest feasible candidate
            trial = e.copy()
            lo, hi = 0, len(cands) - 1
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                trial[t] = cands[mid]
                if _feasible(E, trial):
                    best = cands[mid]
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is not None:
                e[t] = best
                improved = True
    return e


def varian(quantities: pd.DataFrame, prices: pd.DataFrame,
           method: str = 'auto', exact_max_n: int = 30,
           return_multipliers: bool = False):
    """Calculate the (average) Varian index for the given consumption data.

    Returns the average of the per-observation efficiency multipliers (a
    float in [0, 1]); with ``return_multipliers=True``, returns
    ``(average, Series of multipliers)`` indexed like ``quantities``.

    ``method``: 'exact' (MILP), 'heuristic' (greedy ascent, a lower bound on
    the true index), or 'auto' (exact up to ``exact_max_n`` observations,
    heuristic beyond).
    """
    if method not in ('auto', 'exact', 'heuristic'):
        raise ValueError(f"unknown method {method!r}")

    E = cross_expenditure(quantities, prices)
    diag = np.diag(E)
    if (diag <= 0).any():
        raise ValueError("every observation must have positive expenditure")
    n = len(E)

    if satisfies_garp(E):
        e = np.ones(n)
    else:
        ratios = E / diag[:, None]
        if method == 'exact' or (method == 'auto' and n <= exact_max_n):
            e = _varian_exact(ratios)
        else:
            e = _varian_heuristic(E, ratios)

    average = float(e.mean())
    if return_multipliers:
        return average, pd.Series(e, index=quantities.index, name='e')
    return average
