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


def satisfies_garp(E: np.ndarray, idx=None, efficiency: float = 1.0) -> bool:
    """
    Check whether the observations ``idx`` (default: all) satisfy GARP.

    Direct relations on the subset: x^t R0 x^s iff e*E[t,t] >= E[t,s], and
    x^t P0 x^s iff e*E[t,t] > E[t,s], where e is the Afriat ``efficiency``
    level (e = 1 gives standard GARP; e < 1 relaxes the relations by treating
    only bundles costing at most an e-fraction of observed expenditure as
    affordable). R is the transitive closure of R0 (Warshall). GARP holds iff
    there is no pair with x^t R x^s and x^s P0 x^t.
    """
    if idx is not None:
        E = E[np.ix_(idx, idx)]

    diag = efficiency * np.diag(E)[:, None]
    R = E <= diag  # R0
    P0 = E < diag

    m = E.shape[0]
    for k in range(m):
        R |= R[:, [k]] & R[[k], :]

    return not (R & P0.T).any()


def pairwise_garp_violation(E: np.ndarray, i: int, j: int) -> bool:
    """True iff observations i and j violate GARP as a pair (a 2-cycle:
    each bundle revealed weakly preferred to the other, at least one strictly)."""
    r_ij = E[i, i] >= E[i, j]  # x^i R0 x^j
    r_ji = E[j, j] >= E[j, i]  # x^j R0 x^i
    p_ij = E[i, i] > E[i, j]  # x^i P0 x^j
    p_ji = E[j, j] > E[j, i]  # x^j P0 x^i
    return (r_ij and p_ji) or (r_ji and p_ij)
