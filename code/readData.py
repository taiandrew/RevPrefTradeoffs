"""
Reads in the FoodAPS data.

faps_fahitem_puf: quantities and expenditures of food items purchased by households
faps_fahnutrients: nutrient information for food items purchased by households
faps_fahevent_puf: time and place of purchase events
faps_household_puf: household information

Goal: 2 dfs
- quantities: rows = households, columns = food items, values = quantities purchased
- prices: rows = households, columns = food items, values = prices paid per unit
"""

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


# %% Read faps_fahitem_puf

faps_fahitem_puf = pd.read_csv(
    data_dir + "/faps_fahitem_puf.csv",
    low_memory=False,
    encoding='ISO-8859-1')


keep_cols = ['hhnum', 'eventid', 'itemnum',
             'totgramsunadj', 'totgramsunadjimp', 'quantity',
             'totitemexp']

faps_fahitem_puf = faps_fahitem_puf[keep_cols]

faps_fahitem_puf.to_parquet('working_data/faps_fahitem_puf.parquet', index=False)

#%% Read faps_fahnutrients

faps_fahnutrients = pd.read_csv(
    data_dir + "/faps_fahnutrients.csv",
    low_memory=False,
    encoding='ISO-8859-1')

keep_cols = ['hhnum', 'eventid', 'itemnum',
             'usdadescmain', 'usdafoodcat4',
             'totgramsunadj', 'totgramsunadjimp']
             #'d_total', 'd_milk', 'foodcode', 'usdafoodcat1', 'usdafoodcat2']

faps_fahnutrients = faps_fahnutrients[keep_cols]

faps_fahnutrients.to_parquet('working_data/faps_fahnutrients.parquet', index=False)

#%% Read faps_fahevent_puf

faps_fahevent_puf = pd.read_csv(
    data_dir + "/faps_fahevent_puf.csv",
    low_memory=False,
    encoding='ISO-8859-1')

keep_cols = ['hhnum', 'eventid',
             'date', 'daynum' ,'startmon',
             'placeid', 'placecateg', 'placecateg_ers', 'placetype', 'placesnaptype',
             'totalpaid']

faps_fahevent_puf = faps_fahevent_puf[keep_cols]

faps_fahevent_puf.to_parquet('working_data/faps_fahevent_puf.parquet', index=False)
             
#%% Read faps_household_puf

faps_household_puf = pd.read_csv(
    data_dir + "/faps_household_puf.csv",
    low_memory=False,
    encoding='ISO-8859-1')  

keep_cols = ['hhnum', 'targetgroup', 'region', 'rural', 'nonmetro']

faps_household_puf = faps_household_puf[keep_cols]

faps_household_puf.to_parquet('working_data/faps_household_puf.parquet', index=False)
