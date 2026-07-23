"""Runs the fixed-k type-schedule analysis on the milk data.

Supersedes the merge-based reduction (runReduction.py, retired to
code/tags/): instead of greedily
merging the C&P upper-bound partition, the partition is rebuilt from
scratch at every number of types k = 1..k_max by greedy least-harm
assignment (see typeSchedule.py), tracing the schedule of types against
rationality on the 'all' sample under both measures:

- 'ccei'  : each household joins the group whose pooled CCEI drops least;
            reported rationality at k is the MINIMUM group CCEI (the
            common efficiency at which every group passes e-GARP);
- 'varian': each household keeps the individual multiplier e_i it
            received on entry to its group; reported rationality at k is
            the simple average of the e_i across households.

k_max is the number of groups in the best C&P upper-bound partition saved
by runClustering.py (15 for 'all').

Saves, per method x sample:
- working_data/milk_schedule_{method}_{sample}.{parquet,csv}: partition at
  every number of types (one Int64 column per k, NaN outside the sample);
- working_data/schedule_loss_{method}_{sample}.csv: the rationality
  schedule (plus total/average loss by k);
- working_data/schedule_groups_{method}_{sample}.csv: per-group sizes and
  efficiency indices at each k;
- working_data/schedule_multipliers_varian_{sample}.{parquet,csv}: each
  household's Varian multiplier at every k.

The product graphs (rationality against the number of types, weakly
increasing in k by construction) are built from these files by graphs.py.

Runtime note: the exact Demuynck-Rehbeck Varian MILP is available in
varian.py (VARIAN_EXACT_MAX_N-sized groups solve in seconds), but with
VARIAN_REFINE = False the schedule reports the multipliers tracked during
assignment without re-optimizing them per group.
"""

#%% Preamble

import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from typeSchedule import type_schedule

N_RESTARTS = 1
SEED = 42
VARIAN_EXACT_MAX_N = 80
VARIAN_SWEEP_MAX_N = 500
VARIAN_REFINE = False  # report multipliers as tracked during assignment

METHODS = ['ccei', 'varian']
SAMPLE, SAMPLE_LABEL = 'all', 'All households'


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

members = partitions[f'group_{SAMPLE}'].dropna()
hhs = members.index
k_max = members.nunique()


#%% Schedule per method (cheapest runs first)

for method in METHODS:
    print(f"\n=== {method.upper()} schedule, {SAMPLE_LABEL}: "
          f"{len(hhs)} households, k = 1..{k_max} ===", flush=True)

    parts_by_k, schedule, groups_detail, mult_by_k = type_schedule(
        quantities.loc[hhs], prices.loc[hhs], k_max, measure=method,
        n_restarts=N_RESTARTS, seed=SEED,
        exact_max_n=VARIAN_EXACT_MAX_N, sweep_max_n=VARIAN_SWEEP_MAX_N,
        reference=members.astype(int), refine=VARIAN_REFINE)

    print(f"\nRationality schedule ({method}, {SAMPLE_LABEL}):")
    print(schedule.to_string())

    parts_by_k = parts_by_k.reindex(partitions.index).astype('Int64')
    parts_by_k.to_parquet(
        f'working_data/milk_schedule_{method}_{SAMPLE}.parquet')
    parts_by_k.to_csv(f'working_data/milk_schedule_{method}_{SAMPLE}.csv')
    schedule.to_csv(f'working_data/schedule_loss_{method}_{SAMPLE}.csv')
    groups_detail.to_csv(
        f'working_data/schedule_groups_{method}_{SAMPLE}.csv', index=False)
    if mult_by_k is not None:
        mult_by_k = mult_by_k.reindex(partitions.index)
        mult_by_k.to_parquet(
            f'working_data/schedule_multipliers_{method}_{SAMPLE}.parquet')
        mult_by_k.to_csv(
            f'working_data/schedule_multipliers_{method}_{SAMPLE}.csv')

print("\nDone. Build the rationality graphs with: python code/graphs.py")
