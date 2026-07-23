"""
Plots graphs related to type clustering and the type schedule.

Reads only saved outputs (working_data/*), so it can be rerun at any time
without recomputing anything:

- graphs/group_sizes.png: sizes of the C&P upper-bound groups
  (working_data/milk_partitions.parquet, runClustering.py);
- graphs/schedule_rationality_{ccei,varian}_all.png: the product — the
  rationality schedule against the number of types
  (working_data/schedule_loss_{method}_all.csv, runScheduleCCEI.py /
  runScheduleVarian.py). Methods whose schedule file does not exist yet
  are skipped.
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

os.makedirs('graphs', exist_ok=True)
sns.set_theme(style='whitegrid')

SAMPLE, SAMPLE_LABEL = 'all', 'All households'

# %% Plot of group sizes

partitions = pd.read_parquet('working_data/milk_partitions.parquet')

sizes = partitions.groupby(f'group_{SAMPLE}').agg(
    partition_size=(f'group_{SAMPLE}', 'count')
).reset_index()

n_households = sizes['partition_size'].sum()

plt.figure(figsize=(8, 6))
sns.barplot(data=sizes, x=f'group_{SAMPLE}', y='partition_size')
plt.title('Group Sizes, total = {}'.format(n_households))
plt.xlabel('Group')
plt.ylabel('Number of Households')
plt.xticks()
plt.savefig('graphs/group_sizes.png', dpi=300)
plt.close()
print("Saved graphs/group_sizes.png")

# %% Rationality schedules (the product)

Y_LABELS = {'ccei': 'Minimum group CCEI',
            'varian': 'Average Varian multiplier'}

for method, ylabel in Y_LABELS.items():
    path = f'working_data/schedule_loss_{method}_{SAMPLE}.csv'
    if not os.path.exists(path):
        print(f"Skipping {method}: {path} not found "
              "(run runSchedule.py first)")
        continue
    schedule = pd.read_csv(path, index_col='k')

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(schedule.index, schedule['rationality'], marker='o')
    ax.set_xlabel('Number of types')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(schedule.index)
    ax.set_title(f'Rationality by number of types '
                 f'({method.upper()}, {SAMPLE_LABEL})')
    fig.tight_layout()
    filename = f'schedule_rationality_{method}_{SAMPLE}.png'
    fig.savefig(f'graphs/{filename}', dpi=150)
    plt.close(fig)
    print(f"Saved graphs/{filename}")
