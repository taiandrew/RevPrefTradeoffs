"""One iteration of Crawford & Pendakur's practical upper bound algorithm.

Greedily partitions the observations into GARP-consistent groups: each group
can be rationalized by a single utility function, so the number of groups is
an upper bound on the number of preference types needed.
"""

import numpy as np
import pandas as pd

from garp import cross_expenditure, satisfies_garp


def cluster_upper_bound(
    quantities: pd.DataFrame, prices: pd.DataFrame, seed=None
) -> pd.Series:
    """Run ONE iteration of the C&P practical upper bound algorithm.

    Observations are visited in a random order. Each observation is placed in
    the first existing group (tried largest first) that remains GARP-consistent
    with it; if none accepts it, a new group is opened.

    Returns a Series (indexed like ``quantities``) of integer group labels
    1..K.
    """
    E = cross_expenditure(quantities, prices)
    n = len(quantities)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    groups: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        # Largest group first; ties broken by creation order.
        by_size = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), g))
        for g in by_size:
            if satisfies_garp(E, groups[g] + [i]):
                groups[g].append(i)
                break
        else:
            groups.append([i])

    labels = np.empty(n, dtype=int)
    for g, members in enumerate(groups, start=1):
        labels[members] = g
    result = pd.Series(labels, index=quantities.index, name="group")

    sizes = result.value_counts().sort_index()
    print(f"Upper bound (one iteration): {len(groups)} group(s) out of {n} observations.")
    print("Group sizes: " + ", ".join(f"{g}: {s}" for g, s in sizes.items()))
    return result
