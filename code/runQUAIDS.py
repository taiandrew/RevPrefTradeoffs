"""Estimates a group-specific QUAIDS on each GARP-consistent group.

Loads the household-level price and quantity dataframes produced by
imputePrices.py and the best upper-bound partitions saved by runClustering.py
(columns group_all and group_multi of milk_partitions.parquet), and runs the
QUAIDS estimation of quaids.py on each group, as in Section 5 of the paper.
Groups too small to estimate are skipped (the paper drops its 8-observation
group; the model has 15 free parameters across 3 equations for K=4 goods).
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

MIN_QUAIDS_GROUP = 15

# Readable labels for the usdafoodcat4 milk categories
MILK_LABELS = {1002: 'Whole', 1004: '2%', 1006: '1%', 1008: 'Skim'}


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet') \
               .rename(columns=MILK_LABELS)
prices = pd.read_parquet('working_data/milk_prices.parquet') \
           .rename(columns=MILK_LABELS)
partitions = pd.read_parquet('working_data/milk_partitions.parquet')


#%% QUAIDS per upper-bound group, for each partition

results = {}

for sample, label in [('all', 'All households'),
                      ('multi', 'Multiple-milk-type households')]:
    groups = partitions[f'group_{sample}'].dropna()
    print(f"\n=== {label}: {len(groups)} households in "
          f"{groups.nunique()} groups ===")
    for g in sorted(groups.unique()):
        idx = groups.index[groups == g]
        if len(idx) < MIN_QUAIDS_GROUP:
            print(f"Group {g}: {len(idx)} households "
                  f"< {MIN_QUAIDS_GROUP}, skipping QUAIDS.")
            continue
        print(f"\n--- {label}, group {g} ({len(idx)} households) ---")
        try:
            results[(sample, g)] = quaids(quantities.loc[idx], prices.loc[idx])
        except (ValueError, np.linalg.LinAlgError) as err:
            print(f"Group {g}: QUAIDS failed ({err}).")


#%% Save coefficients and standard errors, one row per estimated group

goods = list(quantities.columns)

rows = []
for (sample, g), res in results.items():
    row = {'sample': sample, 'group': g, 'n_households': res.n_obs,
           'converged': res.converged}
    for coef, se, name in [(res.alpha, res.alpha_se, 'alpha'),
                           (res.beta, res.beta_se, 'beta'),
                           (res.lambda_, res.lambda_se, 'lambda')]:
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
coefficients.to_csv('working_data/quaids_coefficients.csv', index=False)
print(f"\nSaved working_data/quaids_coefficients.csv "
      f"({len(coefficients)} groups, {coefficients.shape[1]} columns).")
