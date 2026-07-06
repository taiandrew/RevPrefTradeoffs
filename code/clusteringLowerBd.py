"""One iteration of Crawford & Pendakur's practical lower bound algorithm.

Builds a set of observations that pairwise violate GARP: no two of them can
share a utility function, so the size of the set is a lower bound on the
number of preference types needed to rationalize the data.
"""

import numpy as np
import pandas as pd

from garp import cross_expenditure, pairwise_garp_violation


def cluster_lower_bound(
    quantities: pd.DataFrame, prices: pd.DataFrame, seed=None
) -> int:
    """Run ONE iteration of the C&P practical lower bound algorithm.

    Observations are visited in a random order; an observation is retained
    only if it pairwise violates GARP with every observation already retained.

    Returns the size of the retained set: a lower bound on the number of
    rationalizing groups from this random order.
    """
    E = cross_expenditure(quantities, prices)
    n = len(quantities)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    retained = [order[0]]
    for i in order[1:]:
        if all(pairwise_garp_violation(E, i, j) for j in retained):
            retained.append(i)

    k = len(retained)
    print(f"Lower bound (one iteration): {k} group(s) — "
          f"{k} pairwise-incompatible observation(s) found out of {n}.")
    return k
