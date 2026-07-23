"""Runs the Crawford & Pendakur clustering bounds on the milk data.

Loads the household-level price and quantity dataframes produced by
imputePrices.py and runs N_ITER random-order iterations of the practical
lower and upper bound algorithms (the bounds are the max of the lower bounds
and the min of the upper bounds across iterations) on the analysis sample:
all households with valid milk purchases (num_milk_types >= 1; households
whose purchases have no valid gram quantities are all-zero bundles that
carry no revealed-preference information and are excluded).

Saves the best (fewest-groups) upper-bound partition to
working_data/milk_partitions.{parquet,csv}: index = hhnum over all milk
households, column group_all with integer group labels (NaN for households
outside the sample).
"""

#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from clusteringLowerBd import cluster_lower_bound
from clusteringUpperBd import cluster_upper_bound

N_ITER = 5
SEED = 42


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
# Select the column so this is a Series: boolean filtering on the DataFrame
# would mask values instead of dropping rows
num_milk_types = pd.read_parquet(
    'working_data/milk_types.parquet')['num_milk_types']

samples = {
    'all': num_milk_types[num_milk_types >= 1].index,
}


#%% Run bounds: N_ITER iterations per sample, keep the best of each

best_partitions = {}

for name, hhs in samples.items():
    q, p = quantities.loc[hhs], prices.loc[hhs]
    print(f"\n=== Sample '{name}': {len(hhs)} households, {N_ITER} iterations ===")

    print("\nLower bound iterations:")
    lower_counts = [cluster_lower_bound(q, p, seed=SEED + i)
                    for i in range(N_ITER)]

    print("\nUpper bound iterations:")
    upper_partitions = [cluster_upper_bound(q, p, seed=SEED + i)
                        for i in range(N_ITER)]
    upper_counts = [g.nunique() for g in upper_partitions]
    best_partitions[name] = upper_partitions[int(np.argmin(upper_counts))]

    print(f"\nSample '{name}' recap:")
    print(f"  lower bound counts: {lower_counts} -> best (max): {max(lower_counts)}")
    print(f"  upper bound counts: {upper_counts} -> best (min): {min(upper_counts)}")
    print(f"  number of preference types is between "
          f"{max(lower_counts)} and {min(upper_counts)}.")


#%% Save best upper-bound partitions

partitions = pd.DataFrame(
    {'group_all': best_partitions['all']},
    index=num_milk_types.index,
).astype('Int64')
partitions.index.name = 'hhnum'

partitions.to_parquet('working_data/milk_partitions.parquet')
partitions.to_csv('working_data/milk_partitions.csv')

print("\nSaved working_data/milk_partitions.{parquet,csv}:")
print(partitions.notna().sum().rename('households classified').to_string())
