"""Fixed-k greedy type schedule: number of types vs rationalizability loss.

Crawford & Pendakur-appendix-style construction. For each number of types
k = 1..k_max, the observations are visited in a random order and greedily
assigned to at most k groups. Each k is best-of-``n_restarts`` random
orders. Unlike the superseded merge-based reduction (typeReduction*.py,
retired to code/tags/), partitions are rebuilt from scratch at every k,
so they are not nested across k.

Measures (assignment rule matches the reported statistic):
- 'ccei':   reported rationality at k is the MINIMUM group CCEI — the
            common efficiency level at which every group passes e-GARP
            (a partition-level CCEI). Assignment is max-min: each
            observation goes where the resulting minimum group CCEI is
            highest (then the highest host CCEI, largest host, lowest
            label), and while fewer than k groups are open, starting a
            new group is scored as a candidate placement like any other.
            This spreads violators across groups instead of funneling
            them into whichever group is already worst (which is what
            marginal least-harm rules do, pinning the minimum).
- 'varian': each observation joins the group granting it the highest
            individual multiplier e_i (existing members keep theirs),
            ties to the largest group; a new group is opened only while
            slots remain and every existing group would grant e_i < 1.
            Reported rationality at k is the simple average of the e_i
            across all observations. At k = k_max this reduces to C&P's
            practical upper-bound algorithm.

The object of interest is the rationality schedule against the number of
types, weakly increasing in k by construction — a partition using fewer
than k groups is feasible when k are allowed, so whenever the greedy
result at k is worse than the one at k - 1, the k - 1 partition is
carried forward.

Performance. Candidate evaluations dominate the cost, so each group caches
the transitive closure of its (relaxed) revealed-preference relations and
"does observation i fit group G at its current efficiency?" is answered by
an O(|G|^2) single-node closure extension instead of a full O(|G|^3) GARP
check:
- CCEI: adding an observation can only lower the pooled CCEI, so the
  incremental loss is at least 1 - e_G; groups are visited in order of
  this bound and the scan stops once no remaining group can beat the best
  candidate. A flip-point search (garp.ccei_subset, restricted to the
  interval where the group could still win) runs only when the quick fit
  test fails. At k = 1 the pooled CCEI is order-independent, so it is
  computed directly without restarts.
- Varian: existing members keep their current multipliers (warm start) and
  only the newcomer's multiplier is optimized, which upper-bounds the true
  incremental loss. Reported group losses are recomputed at the end of
  each restart: exact Demuynck-Rehbeck MILP for groups up to
  ``exact_max_n``, greedy ascent (better of the warm start and the CCEI
  start) up to ``sweep_max_n``, and the warm-start loss as-is beyond
  (flagged 'warm' in the group detail, since a full ascent sweep on
  1,000+ observations takes hours).
"""

import numpy as np
import pandas as pd

from garp import (_SHRINK, ccei_subset, cross_expenditure, satisfies_garp,
                  transitive_closure)
from varian import _ascent, _varian_exact, _varian_heuristic


def _build_caches(E, members, eff_vec):
    """Closure R and strict direct relation P0 of ``members`` under the
    per-member efficiency levels ``eff_vec`` (already shrunk as needed)."""
    sub = E[np.ix_(members, members)]
    lim = (eff_vec * np.diag(sub))[:, None]
    return transitive_closure(sub <= lim), sub < lim


def _fits(R, P0, to_i, sto_i, from_i, sfrom_i):
    """Would a new node with direct (strict) edge vectors to_i/sto_i (into
    it) and from_i/sfrom_i (out of it) keep GARP, given the cached closure
    R and strict relation P0 of the current members (which satisfy GARP)?

    O(m^2). Returns (ok, reach_to, reach_from); the reach vectors are the
    new node's column/row in the extended closure, letting the caller
    extend the caches on acceptance.
    """
    reach_to = to_i | (R & to_i[None, :]).any(axis=1)
    reach_from = (from_i | R[from_i].any(axis=0)) if from_i.any() else from_i

    # New member-member relations (t thru i to v) meeting an old strict
    # reverse preference P0[v, t]
    tmp = (P0 & reach_to[None, :]).any(axis=1)
    if (tmp & reach_from).any():
        return False, None, None
    # t reaches i, and i directly strictly prefers t
    if (reach_to & sfrom_i).any():
        return False, None, None
    # i reaches v, and v directly strictly prefers i
    if (reach_from & sto_i).any():
        return False, None, None
    return True, reach_to, reach_from


def _extend_caches(R, P0, reach_to, reach_from, to_i, sto_i, sfrom_i):
    """Grow the caches by one accepted node (last position)."""
    m = len(R)
    R2 = np.empty((m + 1, m + 1), dtype=bool)
    R2[:m, :m] = R | (reach_to[:, None] & reach_from[None, :])
    R2[:m, m] = reach_to
    R2[m, :m] = reach_from
    R2[m, m] = bool((reach_from & to_i).any())
    P2 = np.empty((m + 1, m + 1), dtype=bool)
    P2[:m, :m] = P0
    P2[:m, m] = sto_i
    P2[m, :m] = sfrom_i
    P2[m, m] = False
    return R2, P2


class _CCEIGroup:
    """A group under the common-multiplier CCEI loss L = |G| * (1 - e)."""

    def __init__(self, E, label, i):
        self.E = E
        self.label = label
        self.members = [i]
        self.e = 1.0
        self.R, self.P0 = _build_caches(E, self.members, np.array([1.0]))

    def _eff(self):
        # e is a supremum: for e < 1 the group satisfies GARP just below e,
        # not necessarily at it (for e = 1 it satisfies GARP at 1 exactly)
        return self.e if self.e >= 1.0 else self.e * _SHRINK

    def evaluate(self, i, lo=0.0):
        """Pooled CCEI of the group with observation i added, or
        (None, None) when it is provably <= ``lo`` (give-up threshold)."""
        E, mem, eff = self.E, self.members, self._eff()
        diag = E[mem, mem]
        col, row = E[mem, i], E[i, mem]
        to_i, sto_i = col <= eff * diag, col < eff * diag
        lim_i = eff * E[i, i]
        from_i, sfrom_i = row <= lim_i, row < lim_i

        ok, r_to, r_from = _fits(self.R, self.P0, to_i, sto_i, from_i, sfrom_i)
        if ok:
            return self.e, ('quick', r_to, r_from, to_i, sto_i, sfrom_i)

        # The pooled CCEI drops below e: search flip points in (lo, e]
        e_new = ccei_subset(E, self.members + [i], hi=self.e, lo=lo)
        if e_new is None:
            return None, None
        return e_new, ('rebuild', e_new)

    def add(self, i, payload):
        if payload[0] == 'quick':
            _, r_to, r_from, to_i, sto_i, sfrom_i = payload
            self.R, self.P0 = _extend_caches(
                self.R, self.P0, r_to, r_from, to_i, sto_i, sfrom_i)
            self.members.append(i)
        else:
            self.members.append(i)
            self.e = payload[1]
            eff = np.full(len(self.members), self._eff())
            self.R, self.P0 = _build_caches(self.E, self.members, eff)

    def loss(self):
        return len(self.members) * (1.0 - self.e)


class _VarianGroup:
    """A group under the per-observation Varian loss L = sum(1 - e_i).

    Members keep the multipliers they were assigned on entry (warm start);
    only the newcomer's multiplier is optimized, holding the others fixed,
    so the group's multiplier vector is feasible throughout and its loss is
    an upper bound on the true Varian loss.
    """

    def __init__(self, E, label, i):
        self.E = E
        self.label = label
        self.members = [i]
        self.e_vec = np.array([1.0])
        self.R, self.P0 = _build_caches(E, self.members,
                                        _SHRINK * self.e_vec)

    def evaluate(self, i):
        """Largest multiplier e_i consistent with the group (existing
        members keep theirs); returns (1 - e_i, payload)."""
        E, mem = self.E, self.members
        diag = E[mem, mem]
        col, row = E[mem, i], E[i, mem]
        lim = _SHRINK * self.e_vec * diag
        to_i, sto_i = col <= lim, col < lim

        def try_c(c):
            lim_i = _SHRINK * c * E[i, i]
            return _fits(self.R, self.P0, to_i, sto_i,
                         row <= lim_i, row < lim_i), (row < lim_i)

        (ok, r_to, r_from), sfrom_i = try_c(1.0)
        if ok:
            return 0.0, (1.0, r_to, r_from, to_i, sto_i, sfrom_i)

        # Largest feasible multiplier among i's own flip candidates
        # (feasibility is monotone: raising e_i only adds relations; the
        # smallest candidate is always feasible since below it observation
        # i has no outgoing relations and so can close no cycle)
        ratios = row / E[i, i]
        cands = np.unique(ratios[(ratios > 0) & (ratios < 1)])
        best = None
        lo_i, hi_i = 0, len(cands) - 1
        while lo_i <= hi_i:
            mid = (lo_i + hi_i) // 2
            (ok, r_to, r_from), sfrom_i = try_c(cands[mid])
            if ok:
                best = (cands[mid], r_to, r_from, to_i, sto_i, sfrom_i)
                lo_i = mid + 1
            else:
                hi_i = mid - 1
        if best is None:
            raise AssertionError(
                "smallest own-ratio candidate should always be feasible")
        return 1.0 - best[0], best

    def add(self, i, payload):
        e_i, r_to, r_from, to_i, sto_i, sfrom_i = payload
        self.R, self.P0 = _extend_caches(
            self.R, self.P0, r_to, r_from, to_i, sto_i, sfrom_i)
        self.members.append(i)
        self.e_vec = np.append(self.e_vec, e_i)

    def loss(self):
        return float((1.0 - self.e_vec).sum())


def _choose_ccei(groups, i, slots_remain, next_label):
    """Max-min placement of observation i: maximize the resulting minimum
    group CCEI, then the host group's resulting CCEI, then host size, then
    the lowest label. While slots remain, opening a new group is scored as
    a candidate like any other (its host CCEI is 1 and it leaves every
    existing group untouched), so violators are isolated rather than
    funneled into whichever group is already worst.

    Returns ``(group, payload)``; group None means open a new group.
    """
    es = [g.e for g in groups]
    lows = sorted(es)[:2]

    def m_other(g):
        # minimum CCEI among the *other* groups (1.0 when there are none)
        if g.e > lows[0] or es.count(lows[0]) > 1:
            return lows[0]
        return lows[1] if len(lows) > 1 else 1.0

    best = None  # (score, group, payload); score sorts descending
    if slots_remain:
        best = ((lows[0] if es else 1.0, 1.0, 0, -next_label), None, None)

    ordered = sorted(
        groups, reverse=True,
        key=lambda g: (min(g.e, m_other(g)), g.e, len(g.members), -g.label))
    for g in ordered:
        mo = m_other(g)
        ub = (min(g.e, mo), g.e, len(g.members), -g.label)
        # The group's CCEI can only fall, so ub bounds its score; the sort
        # order makes every remaining group's bound no better either
        if best is not None and ub <= best[0]:
            break
        lo = 0.0
        if best is not None:
            bf, bs = best[0][0], best[0][1]
            # To beat best: exceed bf outright, or tie the resulting
            # minimum (only possible when mo == bf) and beat bs
            lo = max(0.0, (bs if mo <= bf else bf) - 1e-9)
        e_new, payload = g.evaluate(i, lo)
        if e_new is None:
            continue
        score = (min(e_new, mo), e_new, len(g.members), -g.label)
        if best is None or score > best[0]:
            best = (score, g, payload)
    return (None, None) if best is None else (best[1], best[2])


def _choose_varian(groups, i):
    """Least-harm placement of observation i: the group granting the
    highest individual multiplier, ties to the largest group, then the
    lowest label. Returns ``(group, payload, dl)``."""
    best = None  # (dl, -size, label, group, payload)
    by_size = sorted(
        ((-len(g.members), g.label, g) for g in groups))
    for neg_size, label, g in by_size:
        # Once a group takes i harmlessly (dl == 0), no later (smaller)
        # group can win the tie
        if best is not None and best[0] == 0.0:
            break
        dl, payload = g.evaluate(i)
        cand = (dl, neg_size, label, g, payload)
        if best is None or cand[:3] < best[:3]:
            best = cand
    return (None, None, None) if best is None else (best[3], best[4], best[0])


def _run_restart(E, n, k, measure, group_cls, rng):
    """One greedy assignment pass in a random order; returns the groups."""
    order = rng.permutation(n)
    groups = []
    next_label = 1
    for i in order:
        if measure == 'ccei':
            g, payload = _choose_ccei(groups, i, len(groups) < k, next_label)
        else:
            g, payload, dl = _choose_varian(groups, i)
            if g is not None and dl > 0 and len(groups) < k:
                g = None  # a strictly harmed placement loses to a new group
        if g is None:
            groups.append(group_cls(E, next_label, i))
            next_label += 1
        else:
            g.add(i, payload)
    return groups


def _final_varian_loss(E, members, e_vec, exact_max_n, sweep_max_n,
                       sweep_key=None):
    """Recompute a Varian group's multipliers for reporting: exact MILP for
    small groups, greedy ascent from the warm-start multipliers for medium
    ones, the warm-start multipliers as-is for the largest. Returns
    ``(multipliers aligned with members, how)``."""
    m = len(members)
    if (e_vec == 1.0).all():
        return e_vec, 'consistent'
    # The ascent sweeps coordinates in order and its local optimum depends
    # on it: sweeping similar households contiguously does markedly better
    # than a random or sample order, so order by the reference type when
    # available (``sweep_key``), then by sample position
    members = np.asarray(members)
    order = (np.lexsort((members, sweep_key[members]))
             if sweep_key is not None else np.argsort(members))
    sub = E[np.ix_(members[order], members[order])]
    ratios = sub / np.diag(sub)[:, None]
    if m <= exact_max_n:
        e, how = _varian_exact(ratios), 'exact'
    elif m <= sweep_max_n:
        # Ascent only ever raises multipliers, so the local optimum depends
        # on the start; the warm start can be beaten by the CCEI start
        # (which rebalances high early entrants against low late ones)
        e_warm = _ascent(sub, ratios, e_vec[order].copy())
        e_ccei = _varian_heuristic(sub, ratios)
        e, how = (e_warm if e_warm.sum() >= e_ccei.sum() else e_ccei), 'ascent'
    else:
        return e_vec, 'warm'
    aligned = np.empty(m)
    aligned[order] = e
    return aligned, how


def _finalize(groups, measure, E, exact_max_n, sweep_max_n, sweep_key=None,
              refine=True):
    """Total loss and per-group detail rows (relabeled 1.. by size)."""
    rows = []
    for g in sorted(groups, key=lambda g: (-len(g.members), g.label)):
        row = {'group': len(rows) + 1, 'size': len(g.members),
               'loss': g.loss(), 'members': g.members}
        if measure == 'ccei':
            row.update(efficiency=g.e, method='ccei')
        else:
            if refine:
                e, how = _final_varian_loss(
                    E, g.members, g.e_vec, exact_max_n, sweep_max_n,
                    sweep_key)
            else:
                e, how = g.e_vec, 'tracked'
            row.update(loss=float((1.0 - e).sum()), multipliers=e,
                       method=how)
            row['efficiency'] = 1.0 - row['loss'] / len(g.members)
        rows.append(row)
    return sum(r['loss'] for r in rows), rows


def _rationality(measure, n, total, rows):
    """The reported rationality of a partition: minimum group CCEI, or the
    simple average of the individual Varian multipliers."""
    if measure == 'ccei':
        return min(r['efficiency'] for r in rows)
    return 1.0 - total / n


def type_schedule(quantities: pd.DataFrame, prices: pd.DataFrame,
                  k_max: int, measure: str = 'ccei', n_restarts: int = 1,
                  seed: int = 42, exact_max_n: int = 80,
                  sweep_max_n: int = 500, reference: pd.Series = None,
                  refine: bool = True):
    """Trace the schedule of number of types k = 1..k_max against
    rationality, rebuilding the partition greedily at each k.

    ``quantities``/``prices`` are the (already sample-restricted) data.
    ``reference`` is an optional fully rationalizing partition (integer
    labels indexed like ``quantities``, e.g. the saved C&P upper-bound
    partition): at k equal to its number of groups it is scored as a
    known fully-rational candidate, so the schedule ends at exactly 1
    even when no random order happens to reproduce a rationalizing
    partition. Its labels also set the coordinate order of the final
    Varian ascent sweeps (similar households contiguous), which markedly
    improves the ascent's local optimum. ``refine=False`` reports the
    Varian multipliers exactly as tracked during assignment (each
    household keeps the multiplier it received on entry) instead of
    re-optimizing them per group at the end.

    Returns ``(partitions_by_k, schedule, groups_detail,
    multipliers_by_k)``:

    - partitions_by_k: DataFrame indexed like ``quantities`` with one Int64
      column per k (``k1`` .. ``k{k_max}``); labels are 1..k by group size
      (1 = largest) within each column and are not comparable across k;
    - schedule: DataFrame indexed by k with rationality (minimum group
      CCEI, or average Varian multiplier; weakly increasing in k: a worse
      greedy result than at k - 1 is replaced by the k - 1 partition,
      recognizable by n_groups_used < k), total_loss, avg_loss,
      n_groups_used and best_restart (-1 where the result did not come
      from a random restart: CCEI at k = 1, or the reference partition);
    - groups_detail: long DataFrame (k, group, size, efficiency, loss,
      method) for the best restart at each k; ``method`` records how a
      Varian group's reported multipliers were computed ('exact',
      'ascent', 'warm', 'tracked', or 'consistent'; always 'ccei' for the
      CCEI measure, and 'reference' where the reference partition was
      used);
    - multipliers_by_k: for the Varian measure, DataFrame indexed like
      ``quantities`` with each household's multiplier at every k (None
      for CCEI).
    """
    if measure not in ('ccei', 'varian'):
        raise ValueError(f"unknown measure {measure!r}")
    group_cls = _CCEIGroup if measure == 'ccei' else _VarianGroup

    E = cross_expenditure(quantities, prices)
    n = len(quantities)
    if not (1 <= k_max <= n):
        raise ValueError(f"k_max must be in 1..{n}")

    ref_rows, ref_k, sweep_key = None, None, None
    if reference is not None:
        if not reference.index.equals(quantities.index):
            raise ValueError("reference must be indexed like quantities")
        labels = reference.to_numpy()
        sweep_key = labels  # sweep similar households contiguously
        ref_groups = [np.flatnonzero(labels == g).tolist()
                      for g in np.unique(labels)]
        if all(satisfies_garp(E, idx) for idx in ref_groups):
            ref_k = len(ref_groups)
            ref_groups.sort(key=len, reverse=True)
            ref_rows = [{'group': i + 1, 'size': len(idx), 'efficiency': 1.0,
                         'loss': 0.0, 'method': 'reference', 'members': idx,
                         'multipliers': np.ones(len(idx))}
                        for i, idx in enumerate(ref_groups)]
        else:
            print("Warning: reference partition is not fully rationalizing; "
                  "ignoring it.")

    part_cols, mult_cols, sched_rows, detail_rows = {}, {}, [], []
    for k in range(1, k_max + 1):
        if measure == 'ccei' and k == 1:
            # Pooled CCEI is a set function: no assignment order involved
            e1 = ccei_subset(E)
            total, best_r = n * (1.0 - e1), -1
            rows = [{'group': 1, 'size': n, 'efficiency': e1,
                     'loss': total, 'method': 'ccei',
                     'members': list(range(n))}]
            rat = e1
        else:
            best_r, rat, total, rows = None, None, None, None
            for r in range(n_restarts):
                rng = np.random.default_rng([seed, k, r])
                groups = _run_restart(E, n, k, measure, group_cls, rng)
                total_r, rows_r = _finalize(
                    groups, measure, E, exact_max_n, sweep_max_n, sweep_key,
                    refine)
                rat_r = _rationality(measure, n, total_r, rows_r)
                if n_restarts > 1:
                    print(f"  k={k} restart {r}: rationality {rat_r:.4f} "
                          f"({len(groups)} groups)", flush=True)
                if best_r is None or rat_r > rat:
                    best_r, rat, total, rows = r, rat_r, total_r, rows_r
            if k == ref_k and rat < 1.0:
                best_r, rat, total, rows = -1, 1.0, 0.0, ref_rows

        # Monotonicity: a partition into fewer groups is feasible with k
        # allowed, so carry the k - 1 result forward when it is better
        if sched_rows and sched_rows[-1]['rationality'] > rat:
            prev = sched_rows[-1]
            rat, total, best_r, rows = (prev['rationality'],
                                        prev['total_loss'],
                                        prev['best_restart'], prev['rows'])
            print(f"k={k}: greedy result is worse than k={k - 1}; "
                  "carrying the previous partition forward.", flush=True)

        labels = np.empty(n, dtype=int)
        for row in rows:
            labels[row['members']] = row['group']
        part_cols[f'k{k}'] = pd.Series(labels, index=quantities.index)
        if measure == 'varian':
            mult = np.empty(n)
            for row in rows:
                mult[row['members']] = row['multipliers']
            mult_cols[f'k{k}'] = pd.Series(mult, index=quantities.index)

        for row in rows:
            detail_rows.append({'k': k, **{c: row[c] for c in
                                ('group', 'size', 'efficiency', 'loss',
                                 'method')}})
        sched_rows.append({'k': k, 'rationality': rat, 'total_loss': total,
                           'avg_loss': total / n,
                           'n_groups_used': len(rows),
                           'best_restart': best_r, 'rows': rows})
        print(f"k={k}: rationality {rat:.4f} "
              f"({len(rows)} groups, best restart {best_r})", flush=True)

    schedule = pd.DataFrame(sched_rows).drop(columns='rows').set_index('k')

    partitions_by_k = pd.DataFrame(part_cols).astype('Int64')
    groups_detail = pd.DataFrame(detail_rows)
    multipliers_by_k = pd.DataFrame(mult_cols) if measure == 'varian' else None
    return partitions_by_k, schedule, groups_detail, multipliers_by_k
