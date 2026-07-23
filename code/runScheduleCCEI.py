"""Runs the CCEI type schedule on the milk data ('all' sample).

For each number of types k = 1..k_max, rebuilds the partition from scratch
by max-min greedy assignment (typeSchedule.py): each household joins the
placement that maximizes the resulting MINIMUM group CCEI, which is also
the reported rationality at k — the common efficiency level at which every
group passes e-GARP. Best of N_RESTARTS random visit orders per k. k_max
is the number of groups in the best C&P upper-bound partition saved by
runClustering.py.

Saves:
- working_data/milk_schedule_ccei_all.{parquet,csv}: partition at every k
  (one Int64 column per k, NaN outside the sample);
- working_data/schedule_loss_ccei_all.csv: the rationality schedule;
- working_data/schedule_groups_ccei_all.csv: per-group sizes and CCEIs.

The rationality graph is built from these files by graphs.py.
runScheduleVarian.py is the Varian counterpart.
"""

#%% Preamble

import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from typeSchedule import type_schedule

N_RESTARTS = 5
SEED = 2026
SAMPLE, SAMPLE_LABEL = 'all', 'All households'


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

members = partitions[f'group_{SAMPLE}'].dropna()
hhs = members.index
k_max = members.nunique()


#%% CCEI schedule

print(f"=== CCEI schedule, {SAMPLE_LABEL}: {len(hhs)} households, "
      f"k = 1..{k_max} ===", flush=True)

parts_by_k, schedule, groups_detail, _ = type_schedule(
    quantities.loc[hhs], prices.loc[hhs], k_max, measure='ccei',
    n_restarts=N_RESTARTS, seed=SEED, reference=members.astype(int))

print(f"\nRationality schedule (ccei, {SAMPLE_LABEL}):")
print(schedule.to_string())

parts_by_k = parts_by_k.reindex(partitions.index).astype('Int64')
parts_by_k.to_parquet(f'working_data/milk_schedule_ccei_{SAMPLE}.parquet')
parts_by_k.to_csv(f'working_data/milk_schedule_ccei_{SAMPLE}.csv')
schedule.to_csv(f'working_data/schedule_loss_ccei_{SAMPLE}.csv')
groups_detail.to_csv(f'working_data/schedule_groups_ccei_{SAMPLE}.csv',
                     index=False)

print("\nDone. Build the rationality graph with: python code/graphs.py")
