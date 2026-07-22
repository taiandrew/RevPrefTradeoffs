"""CCEI-based type reduction: the trade-off between types and
rationalizability loss, with loss measured by the common-multiplier CCEI.

Implements "Algorithm for type reduction II" from notes.lyx. For a set of
households G, the loss of treating all of G as one preference type is

    L(G) = |G| * (1 - e_G),

where e_G is the CCEI (Afriat efficiency: one common multiplier for the
whole group) of pooling G's observations. typeReductionVarian.py runs the
same algorithm under the per-observation Varian loss. The total loss of a
partition is the sum of its group losses. Starting from a fully rationalizing
partition P_K (each group satisfies GARP, so total loss is 0), the algorithm
greedily merges the pair of groups with the smallest incremental cost

    L(Ga u Gb) - L(Ga) - L(Gb)

down to a single type, tracing out the distortion curve of number of types
against total loss. Merging can only lower a group's CCEI (a violating cycle
in a subset is also a cycle in the superset), so every merge cost is
non-negative and the curve is weakly increasing as types are removed. Merges
are nested (a merge tree), so the greedy curve is an upper bound on the
minimal loss at each k.
"""

import numpy as np
import pandas as pd

from garp import cross_expenditure, satisfies_garp


def _ccei_idx(E: np.ndarray, idx: np.ndarray) -> float:
    """CCEI of the observations ``idx``, given the full cross-expenditure
    matrix E. Same flip-point search as ccei.ccei, on the subset."""
    sub = E[np.ix_(idx, idx)]
    diag = np.diag(sub)
    if (diag <= 0).any():
        raise ValueError("every observation must have positive expenditure")

    if satisfies_garp(sub):
        return 1.0

    ratios = sub / diag[:, None]
    off_diag = ratios[~np.eye(len(sub), dtype=bool)]
    candidates = np.unique(off_diag[(off_diag > 0) & (off_diag < 1)])
    bounds = np.concatenate(([0.0], candidates, [1.0]))

    lo, hi = 0, len(bounds) - 2
    if not satisfies_garp(sub, efficiency=(bounds[0] + bounds[1]) / 2):
        return 0.0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if satisfies_garp(sub, efficiency=(bounds[mid] + bounds[mid + 1]) / 2):
            lo = mid
        else:
            hi = mid - 1
    return float(bounds[lo + 1])


def type_reduction_ccei(quantities: pd.DataFrame, prices: pd.DataFrame,
                        partition: pd.Series):
    """Greedily merge the groups of ``partition`` down to one type,
    with group loss measured by the CCEI.

    ``partition`` holds integer group labels indexed like ``quantities``
    (e.g. one column of milk_partitions); NaN marks households outside the
    sample and stays NaN in the output.

    Returns ``(partitions_by_k, loss_curve)``: a DataFrame with one Int64
    column per number of types (``k{K}`` .. ``k1``; a merged group keeps the
    smaller of the two labels, so labels are traceable across columns and in
    the loss curve), and a DataFrame indexed by k with the total loss, the
    merged pair, its incremental cost and the merged group's CCEI.
    """
    members = partition.dropna()
    sample = members.index
    E = cross_expenditure(quantities.loc[sample], prices.loc[sample])

    labels = members.to_numpy()
    groups = {int(g): np.flatnonzero(labels == g) for g in np.unique(labels)}
    k = len(groups)

    ccei_g = {g: _ccei_idx(E, idx) for g, idx in groups.items()}
    loss = {g: len(groups[g]) * (1 - ccei_g[g]) for g in groups}
    for g, e in ccei_g.items():
        if e < 1:
            print(f"Warning: starting group {g} has CCEI {e:.4f} < 1; "
                  "the partition is not fully rationalizing.")
    total_loss = sum(loss.values())

    assignment = pd.Series(labels, index=sample)
    columns = {f'k{k}': assignment.copy()}
    curve = [{'k': k, 'total_loss': total_loss, 'merged_groups': None,
              'merge_cost': np.nan, 'ccei_merged': np.nan}]

    costs = {}  # (a, b) with a < b -> (incremental cost, ccei of the union)
    while len(groups) > 1:
        for a in groups:
            for b in groups:
                if a < b and (a, b) not in costs:
                    e_ab = _ccei_idx(E, np.concatenate([groups[a], groups[b]]))
                    l_ab = (len(groups[a]) + len(groups[b])) * (1 - e_ab)
                    costs[(a, b)] = (l_ab - loss[a] - loss[b], e_ab)

        (a, b), (cost, e_ab) = min(costs.items(), key=lambda kv: (kv[1][0], kv[0]))
        groups[a] = np.concatenate([groups[a], groups[b]])
        ccei_g[a] = e_ab
        loss[a] = len(groups[a]) * (1 - e_ab)
        del groups[b], ccei_g[b], loss[b]
        costs = {pair: v for pair, v in costs.items()
                 if a not in pair and b not in pair}
        total_loss += cost

        k = len(groups)
        assignment[assignment == b] = a
        columns[f'k{k}'] = assignment.copy()
        curve.append({'k': k, 'total_loss': total_loss,
                      'merged_groups': f'{a}+{b}', 'merge_cost': cost,
                      'ccei_merged': e_ab})
        print(f"k={k}: merged {a}+{b} "
              f"(cost {cost:.4f}, merged CCEI {e_ab:.4f}), "
              f"total loss {total_loss:.4f}", flush=True)

    partitions_by_k = pd.DataFrame(columns).reindex(partition.index) \
                        .astype('Int64')
    loss_curve = pd.DataFrame(curve).set_index('k')
    return partitions_by_k, loss_curve
