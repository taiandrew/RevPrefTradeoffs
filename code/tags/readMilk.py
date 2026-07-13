#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

# Data directory
data_dir = os.path.join(working_dir, "data")

# %% Open parquet data

faps_fahitem_puf = pd.read_parquet('working_data/faps_fahitem_puf.parquet')
faps_fahnutrients = pd.read_parquet('working_data/faps_fahnutrients.parquet')


# %% Get milk data

'''
USDA Food Category 4 codes for milk:
1002 - Milk, fluid, whole + butterfat
1004 - Milk, fluid, reduced fat, 2% milkfat
1006 - Milk, fluid, lowfat, 1% milkfat
1008 - Milk, fluid, fat free, skim
'''

# Merge faps_fahnutrients with faps_fahitem_puf to get milk quantities and expenditures
milk = faps_fahnutrients.merge(
    faps_fahitem_puf,
    on=['hhnum', 'eventid', 'itemnum'],
    how='left')

milk = milk[
    milk['usdafoodcat4'].isin([1002, 1004, 1006, 1008])
    ]

milk.sort_values(by=['hhnum', 'eventid', 'itemnum'], inplace=True)
milk.reset_index(drop=True, inplace=True)

# Grams to be first of 4 columns nonempty
vol_cols = ['totgramsunadj_x', 'totgramsunadjimp_x', 'totgramsunadj_y', 'totgramsunadjimp_y']
milk['grams'] = milk[vol_cols].bfill(axis=1).iloc[:, 0]
milk.drop(columns=vol_cols, inplace=True)

# Sum grams and expenditures by household and food category
milk = milk.groupby(['hhnum', 'usdafoodcat4']).agg(
    grams=('grams', 'sum'),
    totitemexp=('totitemexp', 'sum')
).reset_index()

# Prices per unit
milk['price_per_100g'] = milk['totitemexp'] / milk['grams'] * 100
milk.loc[milk['price_per_100g'] <= 0, 'price_per_100g'] = np.nan
milk.loc[milk['grams'] <= 0, 'price_per_100g'] = np.nan



# %% 

# Keep only hhnum who purchase multiple types (usdafoodcat4) of milk
milk_types = milk.groupby('hhnum')['usdafoodcat4'].nunique().reset_index(name='num_milk_types')
milk = milk.merge(milk_types, on='hhnum', how='left')
milk = milk[milk['num_milk_types'] > 1]

# Save milk data to CSV
milk.to_csv('working_data/milk_data.csv', index=False)
