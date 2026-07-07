"""Critical cost efficiency index (CCEI / Afriat efficiency index).

CCEI = sup{e in [0, 1] : the data satisfy GARP under the relaxed relation
x^t R^e x^s iff p^t . x^s <= e * p^t . x^t}. It equals 1 for GARP-consistent
data; otherwise it is the least budget shrinking under which GARP holds.
"""

import numpy as np
import pandas as pd

from garp import cross_expenditure, satisfies_garp


def ccei(quantities: pd.DataFrame, prices: pd.DataFrame) -> float:
    """Calculate the CCEI for the given consumption data.

    GARP under efficiency e only gains relations as e grows, so consistency is
    monotone in e and the CCEI is found exactly by locating the flip point
    among the critical cost ratios c_ts = (p^t . x^s) / (p^t . x^t): the data
    satisfy relaxed GARP on every efficiency level strictly below the returned
    value and violate it strictly above.
    """
    E = cross_expenditure(quantities, prices)
    diag = np.diag(E)
    if (diag <= 0).any():
        raise ValueError("every observation must have positive expenditure")

    if satisfies_garp(E):
        return 1.0

    # Candidate flip points: cost ratios in (0, 1); GARP-under-e status is
    # constant between consecutive candidates.
    ratios = E / diag[:, None]
    off_diag = ratios[~np.eye(len(E), dtype=bool)]
    candidates = np.unique(off_diag[(off_diag > 0) & (off_diag < 1)])
    bounds = np.concatenate(([0.0], candidates, [1.0]))

    # Binary search over the open intervals between candidates for the last
    # one on which GARP holds; the CCEI is that interval's right endpoint.
    lo, hi = 0, len(bounds) - 2  # interval k = (bounds[k], bounds[k+1])
    if not satisfies_garp(E, efficiency=(bounds[0] + bounds[1]) / 2):
        return 0.0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if satisfies_garp(E, efficiency=(bounds[mid] + bounds[mid + 1]) / 2):
            lo = mid
        else:
            hi = mid - 1
    return float(bounds[lo + 1])
