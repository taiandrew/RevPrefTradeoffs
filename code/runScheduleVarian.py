"""Runs the Varian type schedule on the milk data ('all' sample).

For each number of types k = 1..k_max, rebuilds the partition from scratch
by greedy assignment (typeSchedule.py): each household joins the group
granting it the highest individual multiplier e_i (existing members keep
theirs), and the reported rationality at k is the simple average of the
tracked multipliers across households. One random visit order per k
(N_RESTARTS). k_max is the number of groups in the best C&P upper-bound
partition saved by runClustering.py.

Saves:
- working_data/milk_schedule_varian_all.{parquet,csv}: partition at every
  k (one Int64 column per k, NaN outside the sample);
- working_data/schedule_loss_varian_all.csv: the rationality schedule;
- working_data/schedule_groups_varian_all.csv: per-group sizes and mean
  multipliers;
- working_data/schedule_multipliers_varian_all.{parquet,csv}: each
  household's multiplier at every k.

With REFINE = False the multipliers are reported exactly as tracked
during assignment; REFINE = True re-optimizes them per group at the end
(exact Demuynck-Rehbeck MILP up to EXACT_MAX_N, coordinate ascent up to
SWEEP_MAX_N — see varian.py). The rationality graph is built from these
files by graphs.py. runScheduleCCEI.py is the CCEI counterpart.
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
REFINE = False  # report multipliers as tracked during assignment
EXACT_MAX_N = 80
SWEEP_MAX_N = 500
SAMPLE, SAMPLE_LABEL = 'all', 'All households'


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

members = partitions[f'group_{SAMPLE}'].dropna()
hhs = members.index
k_max = members.nunique()


#%% Varian schedule

print(f"=== VARIAN schedule, {SAMPLE_LABEL}: {len(hhs)} households, "
      f"k = 1..{k_max} ===", flush=True)

parts_by_k, schedule, groups_detail, mult_by_k = type_schedule(
    quantities.loc[hhs], prices.loc[hhs], k_max, measure='varian',
    n_restarts=N_RESTARTS, seed=SEED, exact_max_n=EXACT_MAX_N,
    sweep_max_n=SWEEP_MAX_N, reference=members.astype(int), refine=REFINE)

print(f"\nRationality schedule (varian, {SAMPLE_LABEL}):")
print(schedule.to_string())

parts_by_k = parts_by_k.reindex(partitions.index).astype('Int64')
parts_by_k.to_parquet(f'working_data/milk_schedule_varian_{SAMPLE}.parquet')
parts_by_k.to_csv(f'working_data/milk_schedule_varian_{SAMPLE}.csv')
schedule.to_csv(f'working_data/schedule_loss_varian_{SAMPLE}.csv')
groups_detail.to_csv(f'working_data/schedule_groups_varian_{SAMPLE}.csv',
                     index=False)
mult_by_k = mult_by_k.reindex(partitions.index)
mult_by_k.to_parquet(
    f'working_data/schedule_multipliers_varian_{SAMPLE}.parquet')
mult_by_k.to_csv(f'working_data/schedule_multipliers_varian_{SAMPLE}.csv')

print("\nDone. Build the rationality graph with: python code/graphs.py")
