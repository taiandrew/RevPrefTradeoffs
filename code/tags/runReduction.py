"""Runs the type-reduction analysis on the milk data.

SUPERSEDED by runSchedule.py: the loss exercise now rebuilds the partition
from scratch at every number of types (typeSchedule.py) instead of greedily
merging the upper-bound partition. Kept for comparison.

Loads the household-level price and quantity dataframes produced by
imputePrices.py and the best upper-bound partitions saved by runClustering.py,
and traces the distortion curve of number of types against total
rationalizability loss on each sample ('all' and 'multi'), under both loss
measures:

- 'ccei'  : common-multiplier CCEI loss (typeReductionCCEI.py, Algorithm II)
- 'varian': per-observation Varian loss (typeReductionVarian.py, Algorithm III)

Saves, per method x sample:
- working_data/milk_reduction_{method}_{sample}.{parquet,csv}: partition at
  every number of types (one Int64 column per k, NaN outside the sample);
- working_data/reduction_loss_{method}_{sample}.csv: the loss curve;
- graphs/reduction_distortion_{method}_{sample}.png (total loss) and
  graphs/reduction_avgloss_{method}_{sample}.png (total loss / households,
  on the 1 - efficiency scale).

Runtime warning: the Varian evaluations use varian.py's greedy heuristic for
groups above VARIAN_EXACT_MAX_N observations, whose cost grows roughly with
the cube of the group size per coordinate sweep. The multi sample takes about
a minute; the full 'all' sample (groups up to ~2,000 households) can take
many hours. The CCEI method on the 'all' sample takes ~30-60 minutes.
"""

#%% Preamble

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from typeReductionCCEI import type_reduction_ccei
from typeReductionVarian import type_reduction_varian

# Largest group evaluated with the exact Varian MILP; larger groups use the
# greedy heuristic (a lower bound on the index / upper bound on the loss)
VARIAN_EXACT_MAX_N = 30

METHODS = {
    'ccei': type_reduction_ccei,
    'varian': lambda q, p, part: type_reduction_varian(
        q, p, part, exact_max_n=VARIAN_EXACT_MAX_N),
}

SAMPLES = [('multi', 'Multiple-milk-type households'),
           ('all', 'All households')]


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

os.makedirs('graphs', exist_ok=True)
sns.set_theme(style='whitegrid')


#%% Type reduction per method and sample (cheapest runs first)

for method, reduce_fn in METHODS.items():
    for sample, label in SAMPLES:
        partition = partitions[f'group_{sample}']
        print(f"\n=== {method.upper()} reduction, {label}: "
              f"{partition.notna().sum()} households, "
              f"{partition.nunique()} starting groups ===")

        partitions_by_k, loss_curve = reduce_fn(quantities, prices, partition)

        print(f"\nDistortion curve ({method}, {label}):")
        print(loss_curve.to_string())

        partitions_by_k.to_parquet(
            f'working_data/milk_reduction_{method}_{sample}.parquet')
        partitions_by_k.to_csv(
            f'working_data/milk_reduction_{method}_{sample}.csv')
        loss_curve.to_csv(f'working_data/reduction_loss_{method}_{sample}.csv')

        # Distortion (total loss) and average-loss plots
        n_households = partition.notna().sum()
        plots = [
            (loss_curve['total_loss'], 'Total loss',
             f'reduction_distortion_{method}_{sample}.png', 'distortion curve'),
            (loss_curve['total_loss'] / n_households,
             'Average loss per household',
             f'reduction_avgloss_{method}_{sample}.png', 'average loss'),
        ]
        for values, ylabel, filename, kind in plots:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(loss_curve.index, values, marker='o')
            ax.set_xlabel('Number of types')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Type reduction ({method.upper()}): {kind} ({label})')
            fig.tight_layout()
            fig.savefig(f'graphs/{filename}', dpi=150)
            plt.close(fig)
            print(f"Saved graphs/{filename}")
