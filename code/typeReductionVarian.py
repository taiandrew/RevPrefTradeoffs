"""Varian-based type reduction: the trade-off between types and
rationalizability loss, with loss measured by per-observation multipliers.

Implements "Algorithm III" from notes.lyx: same greedy merging as
typeReductionCCEI.py, but the loss of treating a set of households G as one
preference type uses the Varian efficiency vector,

    L_Varian(G) = sum_{i in G} (1 - e_i) = |G| * (1 - average Varian index),

where the e_i maximize the average multiplier subject to GARP holding on the
pooled group under the relaxed relations (see varian.py). Because one bad
violator no longer drags down every member's multiplier, Varian loss prices
merges less severely than CCEI loss.

Starting from a fully rationalizing partition P_K (total loss 0), the pair of
groups with the smallest incremental cost L(Ga u Gb) - L(Ga) - L(Gb) is merged
repeatedly down to one type, tracing the distortion curve of number of types
against total loss. An optimal multiplier vector for a union is feasible on
each part, so exact merge costs are non-negative and the curve is weakly
increasing as types are removed; groups larger than ``exact_max_n`` are
evaluated with varian.py's greedy heuristic, whose loss is an upper bound on
the true Varian loss (so small non-monotonicities are possible there). As in
the CCEI version, merges are nested, making the greedy curve an upper bound
on the minimal loss at each k.
"""

import numpy as np
import pandas as pd

from garp import cross_expenditure, satisfies_garp
from varian import _varian_exact, _varian_heuristic


def _varian_avg_idx(E: np.ndarray, idx: np.ndarray, exact_max_n: int) -> float:
    """Average Varian index of the observations ``idx``, given the full
    cross-expenditure matrix E (exact MILP up to exact_max_n, else heuristic)."""
    sub = E[np.ix_(idx, idx)]
    diag = np.diag(sub)
    if (diag <= 0).any():
        raise ValueError("every observation must have positive expenditure")

    if satisfies_garp(sub):
        return 1.0

    ratios = sub / diag[:, None]
    if len(idx) <= exact_max_n:
        e = _varian_exact(ratios)
    else:
        e = _varian_heuristic(sub, ratios)
    return float(e.mean())


def type_reduction_varian(quantities: pd.DataFrame, prices: pd.DataFrame,
                          partition: pd.Series, exact_max_n: int = 30):
    """Greedily merge the groups of ``partition`` down to one type,
    with group loss measured by the Varian efficiency vector.

    ``partition`` holds integer group labels indexed like ``quantities``
    (e.g. one column of milk_partitions); NaN marks households outside the
    sample and stays NaN in the output.

    Returns ``(partitions_by_k, loss_curve)``: a DataFrame with one Int64
    column per number of types (``k{K}`` .. ``k1``; a merged group keeps the
    smaller of the two labels, so labels are traceable across columns and in
    the loss curve), and a DataFrame indexed by k with the total loss, the
    merged pair, its incremental cost and the merged group's average Varian
    index.
    """
    members = partition.dropna()
    sample = members.index
    E = cross_expenditure(quantities.loc[sample], prices.loc[sample])

    labels = members.to_numpy()
    groups = {int(g): np.flatnonzero(labels == g) for g in np.unique(labels)}
    k = len(groups)

    varian_g = {g: _varian_avg_idx(E, idx, exact_max_n)
                for g, idx in groups.items()}
    loss = {g: len(groups[g]) * (1 - varian_g[g]) for g in groups}
    for g, e in varian_g.items():
        if e < 1:
            print(f"Warning: starting group {g} has average Varian index "
                  f"{e:.4f} < 1; the partition is not fully rationalizing.")
    total_loss = sum(loss.values())

    assignment = pd.Series(labels, index=sample)
    columns = {f'k{k}': assignment.copy()}
    curve = [{'k': k, 'total_loss': total_loss, 'merged_groups': None,
              'merge_cost': np.nan, 'varian_merged': np.nan}]

    costs = {}  # (a, b) with a < b -> (incremental cost, avg Varian of union)
    while len(groups) > 1:
        for a in groups:
            for b in groups:
                if a < b and (a, b) not in costs:
                    e_ab = _varian_avg_idx(
                        E, np.concatenate([groups[a], groups[b]]), exact_max_n)
                    l_ab = (len(groups[a]) + len(groups[b])) * (1 - e_ab)
                    costs[(a, b)] = (l_ab - loss[a] - loss[b], e_ab)

        (a, b), (cost, e_ab) = min(costs.items(), key=lambda kv: (kv[1][0], kv[0]))
        groups[a] = np.concatenate([groups[a], groups[b]])
        varian_g[a] = e_ab
        loss[a] = len(groups[a]) * (1 - e_ab)
        del groups[b], varian_g[b], loss[b]
        costs = {pair: v for pair, v in costs.items()
                 if a not in pair and b not in pair}
        total_loss += cost

        k = len(groups)
        assignment[assignment == b] = a
        columns[f'k{k}'] = assignment.copy()
        curve.append({'k': k, 'total_loss': total_loss,
                      'merged_groups': f'{a}+{b}', 'merge_cost': cost,
                      'varian_merged': e_ab})
        print(f"k={k}: merged {a}+{b} "
              f"(cost {cost:.4f}, merged avg Varian {e_ab:.4f}), "
              f"total loss {total_loss:.4f}", flush=True)

    partitions_by_k = pd.DataFrame(columns).reindex(partition.index) \
                        .astype('Int64')
    loss_curve = pd.DataFrame(curve).set_index('k')
    return partitions_by_k, loss_curve
