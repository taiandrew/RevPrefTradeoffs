"""
Builds household-level quantity and price dataframes for the milk analysis,
imputing missing prices so every household has a complete price vector.

Each final observation is a household: quantities and expenditures are summed
across all of a household's FAH events.

Imputation cells
----------------
Cells are imputed by:

    placetype x region x rural x nonmetro

For each household, missing prices are imputed as the median unit price
(per 100g) of that milk type across all item purchases in the household's
cell, where the household's placetype is the modal value over its own
milk-purchase events. Sparse cells back off to coarser groupings:

    1. placetype x region x rural x nonmetro
    2. placetype x rural x nonmetro
    3. placetype
    4. all purchases of the milk type (global median)

Outputs (working_data/, each as .parquet and .csv)
--------------------------------------------------
milk_hh_long   : hhnum x usdafoodcat4 long panel with grams,
                 expenditure, price, and price_source flag
milk_quantities: rows = households, cols = milk types, values = grams
milk_prices    : rows = households, cols = milk types, values = price
                 per 100g (observed or imputed, no missings)
milk_types     : rows = households, single column num_milk_types = number
                 of milk types purchased (0 if no valid gram quantities)
"""

#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

MILK_CATS = [1002, 1004, 1006, 1008]
'''
USDA Food Category 4 codes for milk:
1002 - Milk, fluid, whole + butterfat
1004 - Milk, fluid, reduced fat, 2% milkfat
1006 - Milk, fluid, lowfat, 1% milkfat
1008 - Milk, fluid, fat free, skim
'''


#%% Load data

faps_fahitem_puf = pd.read_parquet('working_data/faps_fahitem_puf.parquet')
faps_fahnutrients = pd.read_parquet('working_data/faps_fahnutrients.parquet')
faps_household_puf = pd.read_parquet('working_data/faps_household_puf.parquet')
faps_fahevent_puf = pd.read_parquet('working_data/faps_fahevent_puf.parquet')

#%% Item-level milk purchases with unit prices

milk = faps_fahnutrients.merge(
    faps_fahitem_puf,
    on=['hhnum', 'eventid', 'itemnum'],
    how='left')

milk = milk[milk['usdafoodcat4'].isin(MILK_CATS)]

# Grams -- first nonempty grams col
vol_cols = ['totgramsunadj_x', 'totgramsunadjimp_x',
            'totgramsunadj_y', 'totgramsunadjimp_y']
milk['grams'] = milk[vol_cols].bfill(axis=1).iloc[:, 0]
milk.drop(columns=vol_cols, inplace=True)

# Item-level unit price, valid only with positive grams and expenditure
milk['price_per_100g'] = np.where(
    (milk['grams'] > 0) & (milk['totitemexp'] > 0),
    milk['totitemexp'] / milk['grams'] * 100,
    np.nan)

# Attach place and geography variables
milk = milk.merge(
    faps_fahevent_puf[['hhnum', 'eventid', 'placetype']],
    on=['hhnum', 'eventid'], how='left')
milk = milk.merge(
    faps_household_puf[['hhnum', 'region', 'rural', 'nonmetro']],
    on='hhnum', how='left')


#%% Cell-median price tables (descending availability)

priced = milk[milk['price_per_100g'].notna()]

CELL_LEVELS = [
    ['placetype', 'region', 'rural', 'nonmetro'],
    ['placetype', 'rural', 'nonmetro'],
    ['placetype'],
]

cell_medians = [
    priced.groupby(keys + ['usdafoodcat4'])['price_per_100g']   # By cell level and milk type
          .median().rename(f'price_lvl{i}').reset_index()       # Take median; reset df
    for i, keys in enumerate(CELL_LEVELS)
]
global_median = priced.groupby('usdafoodcat4')['price_per_100g'] \
                      .median().rename('price_global').reset_index()


#%% Household cell assignment: finds most frequent shopping type for each household

hh_cell = (
    milk.groupby(['hhnum', 'placetype', 'region', 'rural', 'nonmetro'])
        .size().rename('n_events').reset_index()
        .sort_values(['hhnum', 'n_events', 'placetype'],
                     ascending=[True, False, True])
        .drop_duplicates('hhnum')
        .drop(columns='n_events')
)


#%% Household x milk-type panel (households as observations)

# Aggregate quantities and expenditures across all events per household
hh_milk = milk.groupby(['hhnum', 'usdafoodcat4']).agg(
    grams=('grams', 'sum'),
    totitemexp=('totitemexp', 'sum')
).reset_index()

hh_milk['price_observed'] = np.where(
    (hh_milk['grams'] > 0) & (hh_milk['totitemexp'] > 0),
    hh_milk['totitemexp'] / hh_milk['grams'] * 100,
    np.nan)

# Full balanced panel: every milk-buying household x every milk type
milk_hhs = sorted(milk['hhnum'].unique())
panel = pd.MultiIndex.from_product(
    [milk_hhs, MILK_CATS], names=['hhnum', 'usdafoodcat4']
).to_frame(index=False)

panel = panel.merge(hh_milk, on=['hhnum', 'usdafoodcat4'], how='left')
panel[['grams', 'totitemexp']] = panel[['grams', 'totitemexp']].fillna(0)

# Attach household cell and candidate imputed prices at each backoff level
panel = panel.merge(hh_cell, on='hhnum', how='left')
for i, keys in enumerate(CELL_LEVELS):
    panel = panel.merge(cell_medians[i], on=keys + ['usdafoodcat4'], how='left')
panel = panel.merge(global_median, on='usdafoodcat4', how='left')

# Price: observed where valid, otherwise first available backoff level
price_cols = ['price_observed'] + \
    [f'price_lvl{i}' for i in range(len(CELL_LEVELS))] + ['price_global']
panel['price_per_100g'] = panel[price_cols].bfill(axis=1).iloc[:, 0]

source_labels = ['observed', 'cell', 'placetype_rural_nonmetro',
                 'placetype', 'global']
panel['price_source'] = pd.Categorical.from_codes(
    panel[price_cols].notna().values.argmax(axis=1),
    categories=source_labels)

assert panel['price_per_100g'].notna().all(), "incomplete price vectors remain"

# Number of milk types purchased, for downstream filtering
panel['num_milk_types'] = panel.groupby('hhnum')['grams'] \
                               .transform(lambda g: (g > 0).sum())

panel = panel[['hhnum', 'usdafoodcat4', 'grams', 'totitemexp',
               'price_per_100g', 'price_source', 'num_milk_types']]


#%% Wide dataframes: rows = households, columns = milk types

milk_quantities = panel.pivot(
    index='hhnum', columns='usdafoodcat4', values='grams')
milk_prices = panel.pivot(
    index='hhnum', columns='usdafoodcat4', values='price_per_100g')

milk_types = panel.drop_duplicates('hhnum').set_index('hhnum')['num_milk_types']
milk_types = milk_types.to_frame(name='num_milk_types').sort_index()

#%% Save

panel.to_parquet('working_data/milk_hh_long.parquet', index=False)
milk_quantities.to_parquet('working_data/milk_quantities.parquet')
milk_prices.to_parquet('working_data/milk_prices.parquet')
milk_types.to_parquet('working_data/milk_types.parquet')

panel.to_csv('working_data/milk_hh_long.csv', index=False)
milk_quantities.to_csv('working_data/milk_quantities.csv')
milk_prices.to_csv('working_data/milk_prices.csv')
milk_types.to_csv('working_data/milk_types.csv')


#%% Diagnostics

print(f"Households: {len(milk_hhs)}")
print(f"Household x milk-type cells: {len(panel)}")
print("\nPrice source:")
print(panel['price_source'].value_counts())
print("\nMilk types purchased per household:")
print(panel.drop_duplicates('hhnum')['num_milk_types'].value_counts().sort_index())
print("\nMedian price per 100g by milk type and source:")
print(panel.groupby(['usdafoodcat4', 'price_source'], observed=True)
      ['price_per_100g'].median().unstack().round(3))
