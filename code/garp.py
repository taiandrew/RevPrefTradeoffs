"""Shared revealed-preference utilities: GARP checks per Varian (1982).

Data conventions used throughout the project:
- ``quantities``: DataFrame with N rows (observations) and j columns (products);
  entries are quantities purchased.
- ``prices``: DataFrame of the same shape; entries are the prices at which the
  corresponding bundle was purchased.
"""

import numpy as np
import pandas as pd


def cross_expenditure(quantities: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    """
    Return the N x N cross-expenditure matrix E with E[t, s] = p^t . x^s.

    Row t prices every observed bundle at observation t's prices, so the
    diagonal E[t, t] is observed expenditure at t.
    """
    if quantities.shape != prices.shape:
        raise ValueError(
            f"quantities {quantities.shape} and prices {prices.shape} must have the same shape"
        )
    if not quantities.index.equals(prices.index):
        raise ValueError("quantities and prices must share the same index")

    x = quantities.to_numpy(dtype=float)
    p = prices.to_numpy(dtype=float)
    if np.isnan(x).any() or np.isnan(p).any():
        raise ValueError("quantities and prices must not contain NaNs")
    if (p <= 0).any():
        raise ValueError("all prices must be strictly positive")
    if (x < 0).any():
        raise ValueError("quantities must be non-negative")

    return p @ x.T


def _bit_masks(m: int):
    """Byte index and bit mask of each column in a packbits row."""
    cols = np.arange(m)
    return cols >> 3, (0x80 >> (cols & 7)).astype(np.uint8)


def transitive_closure(R0: np.ndarray) -> np.ndarray:
    """Boolean transitive closure of a relation matrix (bitset Warshall:
    rows are packed 8 columns per byte, so each row update is a vectorized
    byte-wise OR)."""
    m = len(R0)
    R8 = np.packbits(R0, axis=1)
    byte_idx, bit = _bit_masks(m)
    for k in range(m):
        mask = (R8[:, byte_idx[k]] & bit[k]) != 0
        if mask.any():
            np.bitwise_or(R8, R8[k], out=R8, where=mask[:, None])
    return np.unpackbits(R8, axis=1, count=m).astype(bool)


def satisfies_garp(E: np.ndarray, idx=None, efficiency: float = 1.0) -> bool:
    """
    Check whether the observations ``idx`` (default: all) satisfy GARP.

    Direct relations on the subset: x^t R0 x^s iff e*E[t,t] >= E[t,s], and
    x^t P0 x^s iff e*E[t,t] > E[t,s], where e is the Afriat ``efficiency``
    level (e = 1 gives standard GARP; e < 1 relaxes the relations by treating
    only bundles costing at most an e-fraction of observed expenditure as
    affordable). R is the transitive closure of R0 (Warshall). GARP holds iff
    there is no pair with x^t R x^s and x^s P0 x^t.

    The closure runs on bit-packed rows, and since relations only grow
    during the closure, a violation found part-way is final: checking
    periodically lets violating data exit early.
    """
    if idx is not None:
        E = E[np.ix_(idx, idx)]

    diag = efficiency * np.diag(E)[:, None]
    R8 = np.packbits(E <= diag, axis=1)   # R0
    P0T8 = np.packbits(E.T < diag.T, axis=1)  # P0 transposed

    m = E.shape[0]
    byte_idx, bit = _bit_masks(m)
    for k in range(m):
        mask = (R8[:, byte_idx[k]] & bit[k]) != 0
        if mask.any():
            np.bitwise_or(R8, R8[k], out=R8, where=mask[:, None])
        if (k & 127) == 127 and (R8 & P0T8).any():
            return False

    return not (R8 & P0T8).any()


_SHRINK = 1 - 1e-9  # test just below an efficiency level (sup convention)


def ccei_subset(E: np.ndarray, idx=None, hi: float = 1.0, lo: float = 0.0):
    """CCEI of the observations ``idx`` (default: all), given the full
    cross-expenditure matrix E, searching only flip candidates in (lo, hi].

    ``hi`` is a known upper bound on the CCEI — e.g. a group's CCEI before
    adding an observation, since adding observations can only lower it. The
    common case (the CCEI stays at ``hi``) is decided by a single GARP check
    just below ``hi``. ``lo`` is a give-up threshold: returns None when the
    CCEI is <= lo (only checked when lo > 0), letting callers skip the
    search once a group cannot beat a bound. As in ccei.ccei, the returned
    value is a supremum: e-GARP holds just below it but not necessarily at
    it.
    """
    sub = E if idx is None else E[np.ix_(idx, idx)]
    diag = np.diag(sub)
    if (diag <= 0).any():
        raise ValueError("every observation must have positive expenditure")

    if satisfies_garp(sub, efficiency=(1.0 if hi >= 1.0 else hi * _SHRINK)):
        return min(hi, 1.0)

    ratios = sub / diag[:, None]
    off_diag = ratios[~np.eye(len(sub), dtype=bool)]
    candidates = np.unique(off_diag[(off_diag > lo) & (off_diag < hi)])
    bounds = np.concatenate(([lo], candidates, [hi]))

    lo_i, hi_i = 0, len(bounds) - 2
    if not satisfies_garp(sub, efficiency=(bounds[0] + bounds[1]) / 2):
        return None if lo > 0 else 0.0
    while lo_i < hi_i:
        mid = (lo_i + hi_i + 1) // 2
        if satisfies_garp(sub, efficiency=(bounds[mid] + bounds[mid + 1]) / 2):
            lo_i = mid
        else:
            hi_i = mid - 1
    return float(bounds[lo_i + 1])


def pairwise_garp_violation(E: np.ndarray, i: int, j: int) -> bool:
    """True iff observations i and j violate GARP as a pair (a 2-cycle:
    each bundle revealed weakly preferred to the other, at least one strictly)."""
    r_ij = E[i, i] >= E[i, j]  # x^i R0 x^j
    r_ji = E[j, j] >= E[j, i]  # x^j R0 x^i
    p_ij = E[i, i] > E[i, j]  # x^i P0 x^j
    p_ji = E[j, j] > E[j, i]  # x^j P0 x^i
    return (r_ij and p_ji) or (r_ji and p_ij)
