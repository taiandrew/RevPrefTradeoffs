"""Runs the type-reduction analysis on the milk data.

Loads the household-level price and quantity dataframes produced by
imputePrices.py and the best upper-bound partitions saved by runClustering.py,
and runs the greedy CCEI-merging algorithm of typeReduction.py on each sample
('all' and 'multi'), tracing the distortion curve of number of types against
total rationalizability loss.

Saves, per sample:
- working_data/milk_reduction_{sample}.{parquet,csv}: partition at every
  number of types (one Int64 column per k, NaN outside the sample);
- working_data/reduction_loss_{sample}.csv: the loss curve;
and per sample, a distortion-curve plot (total loss) and an average-loss
plot (total loss / households, on the 1 - CCEI scale) in
graphs/reduction_{distortion,avgloss}_{sample}.png.
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

from typeReduction import type_reduction


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
partitions = pd.read_parquet('working_data/milk_partitions.parquet')


#%% Type reduction per sample

loss_curves = {}

for sample, label in [('multi', 'Multiple-milk-type households'),
                      ('all', 'All households')]:
    partition = partitions[f'group_{sample}']
    print(f"\n=== {label}: {partition.notna().sum()} households, "
          f"{partition.nunique()} starting groups ===")

    partitions_by_k, loss_curve = type_reduction(quantities, prices, partition)
    loss_curves[sample] = loss_curve

    print(f"\nDistortion curve ({label}):")
    print(loss_curve.to_string())

    partitions_by_k.to_parquet(f'working_data/milk_reduction_{sample}.parquet')
    partitions_by_k.to_csv(f'working_data/milk_reduction_{sample}.csv')
    loss_curve.to_csv(f'working_data/reduction_loss_{sample}.csv')


#%% Plot distortion curves

os.makedirs('graphs', exist_ok=True)
sns.set_theme(style='whitegrid')
for sample, label in [('all', 'All households'),
                      ('multi', 'Multiple-milk-type households')]:
    curve = loss_curves[sample]
    n_households = partitions[f'group_{sample}'].notna().sum()
    plots = [
        ('total_loss', curve['total_loss'], 'Total loss',
         f'reduction_distortion_{sample}.png', 'distortion curve'),
        ('avg_loss', curve['total_loss'] / n_households,
         'Average loss per household',
         f'reduction_avgloss_{sample}.png', 'average loss'),
    ]
    for _, values, ylabel, filename, kind in plots:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(curve.index, values, marker='o')
        ax.set_xlabel('Number of types')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Type reduction: {kind} ({label})')
        fig.tight_layout()
        fig.savefig(f'graphs/{filename}', dpi=150)
        plt.close(fig)
        print(f"Saved graphs/{filename}")
