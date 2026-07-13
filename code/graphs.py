"""
Plots graphs related to type clustering.
"""

#%% Preamble

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

# %% Plot of group sizes

partitions = pd.read_parquet('working_data/milk_partitions.parquet')

partitions = partitions.groupby('group_all').agg(
    partition_size=('group_all', 'count')
).reset_index()

n_households = partitions['partition_size'].sum()

# Plot bars of partition sizes
plt.figure(figsize=(8, 6))
sns.barplot(data=partitions, x='group_all', y='partition_size')
plt.title('Group Sizes, total = {}'.format(n_households))
plt.xlabel('Group')
plt.ylabel('Number of Households')
plt.xticks()
plt.savefig('graphs/group_sizes.png', dpi=300)