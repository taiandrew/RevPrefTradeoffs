"""Estimates a group-specific AIDS on each GARP-consistent group.

Identical to runQUAIDS.py but for the linear almost ideal demand system
(Deaton & Muellbauer 1980), i.e. QUAIDS without the quadratic expenditure
term (quadratic=False). Useful as the nested benchmark: comparing the AIDS and
QUAIDS fits shows whether the quadratic Engel-curve terms are needed.

Loads the household-level price and quantity dataframes produced by
imputePrices.py and the best upper-bound partitions saved by runClustering.py,
and estimates AIDS on each group large enough to support it.

Prices and expenditure are normalised to medians computed once over the whole
sample (not per group), so the level coefficients are evaluated at a common
constraint and are comparable across groups.
"""

#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from quaids import quaids

# AIDS has 11 free parameters across 3 equations for K=4 goods (one fewer per
# equation than QUAIDS); keep the same floor as runQUAIDS for comparability
MIN_AIDS_GROUP = 15

# Readable labels for the usdafoodcat4 milk categories
MILK_LABELS = {1002: 'Whole', 1004: '2%', 1006: '1%', 1008: 'Skim'}


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet') \
               .rename(columns=MILK_LABELS)
prices = pd.read_parquet('working_data/milk_prices.parquet') \
           .rename(columns=MILK_LABELS)
partitions = pd.read_parquet('working_data/milk_partitions.parquet')


#%% AIDS per upper-bound group, for each partition

results = {}

for sample, label in [('all', 'All households'),
                      ('multi', 'Multiple-milk-type households')]:
    groups = partitions[f'group_{sample}'].dropna()
    print(f"\n=== {label}: {len(groups)} households in "
          f"{groups.nunique()} groups ===")

    # Common normalisation constraint: medians over the whole sample
    sample_idx = groups.index
    price_medians = prices.loc[sample_idx].median()
    exp_median = (prices.loc[sample_idx] * quantities.loc[sample_idx]) \
        .sum(axis=1).median()

    for g in sorted(groups.unique()):
        idx = groups.index[groups == g]
        if len(idx) < MIN_AIDS_GROUP:
            print(f"Group {g}: {len(idx)} households "
                  f"< {MIN_AIDS_GROUP}, skipping AIDS.")
            continue
        print(f"\n--- {label}, group {g} ({len(idx)} households) ---")
        try:
            results[(sample, g)] = quaids(
                quantities.loc[idx], prices.loc[idx], quadratic=False,
                price_medians=price_medians, exp_median=exp_median)
        except (ValueError, np.linalg.LinAlgError) as err:
            print(f"Group {g}: AIDS failed ({err}).")


#%% Save coefficients and standard errors, one row per estimated group

goods = list(quantities.columns)

rows = []
for (sample, g), res in results.items():
    row = {'sample': sample, 'group': g, 'n_households': res.n_obs,
           'converged': res.converged}
    # AIDS has no quadratic term, so only alpha, beta and gamma are reported
    for coef, se, name in [(res.alpha, res.alpha_se, 'alpha'),
                           (res.beta, res.beta_se, 'beta')]:
        for good in goods:
            row[f'{name}_{good}'] = coef[good]
            row[f'{name}_{good}_se'] = se[good]
    # gamma is symmetric: keep each pair once
    for i, gi in enumerate(goods):
        for gj in goods[i:]:
            row[f'gamma_{gi}_{gj}'] = res.gamma.loc[gi, gj]
            row[f'gamma_{gi}_{gj}_se'] = res.gamma_se.loc[gi, gj]
    rows.append(row)

coefficients = pd.DataFrame(rows)
coefficients.to_csv('working_data/aids_coefficients.csv', index=False)
print(f"\nSaved working_data/aids_coefficients.csv "
      f"({len(coefficients)} groups, {coefficients.shape[1]} columns).")
