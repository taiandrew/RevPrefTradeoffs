"""Descriptive-statistics tables for the milk data, mirroring Tables 1 and 2
of Crawford & Pendakur (2013, EJ) minus the demographic material.

Loads the household-level milk types, prices, quantities and upper-bound
partitions produced by imputePrices.py / runClustering.py, and for the
analysis sample ('all') emits:

- tables/summary_{sample}.tex (C&P Table 1): budget shares, total expenditure
  and prices, each summarised by Mean, Min, Max and SD;
- tables/group_shares_{sample}.tex (C&P Table 2): average budget shares by
  upper-bound type -- a Pooled row plus one row per group (ordered largest
  first) giving the group size and its mean budget share of each milk type.

Conventions
-----------
Everything is derived from the price and quantity matrices exactly as the
revealed-preference analysis consumes them:

- Expenditure on type j is p_ij * q_ij; with prices stored per 100 g and
  quantities in grams, dollar expenditure is p_ij * q_ij / 100. (This equals
  realised milk expenditure for observed prices and repairs the raw
  totitemexp, which contains erroneous non-positive values.)
- Budget shares w_ij = p_ij q_ij / sum_k p_ik q_ik are summarised over every
  in-sample household, so single-variety buyers contribute shares of 0 and 1
  (as in C&P, whose shares range the full [0, 1]). As in C&P Table 2, the
  per-group averages mix the effects of preferences and budget constraints.
- Prices are summarised over purchasers only (q_ij > 0), i.e. genuine unit
  values, rather than the imputed cell medians assigned to non-purchasers.

Both outputs require \\usepackage{booktabs}.
"""

#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

# Readable labels for the usdafoodcat4 milk categories, in fat-content order
MILK_LABELS = {1002: 'Whole', 1004: '2\\%', 1006: '1\\%', 1008: 'Skim'}

SAMPLES = [('all', 'all households with valid milk purchases', 1)]

STAT_COLS = ['Mean', 'Min', 'Max', 'SD']


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet')
prices = pd.read_parquet('working_data/milk_prices.parquet')
num_milk_types = pd.read_parquet(
    'working_data/milk_types.parquet')['num_milk_types']
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

goods = list(quantities.columns)


#%% Budget shares (shared by both tables)

def budget_data(sample_index: pd.Index):
    """Return (prices, quantities, total expenditure, budget shares) for the
    households in ``sample_index``. Expenditure is in price*gram units;
    dividing by 100 gives dollars, but the /100 cancels in the shares."""
    q = quantities.loc[sample_index]
    p = prices.loc[sample_index]
    expenditure = p * q
    total = expenditure.sum(axis=1)
    shares = expenditure.div(total, axis=0)
    return p, q, total, shares


#%% Summary statistics

def _stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean/Min/Max/SD for each column of ``frame`` (population-style SD,
    ddof=1), indexed by the readable milk labels."""
    out = pd.DataFrame({
        'Mean': frame.mean(),
        'Min': frame.min(),
        'Max': frame.max(),
        'SD': frame.std(ddof=1),
    })
    out.index = [MILK_LABELS[g] for g in frame.columns]
    return out[STAT_COLS]


def summary_stats(sample_index: pd.Index):
    """Return (budget-share stats, expenditure stats, price stats) for the
    households in ``sample_index``."""
    p, q, total, shares = budget_data(sample_index)

    share_stats = _stats(shares)
    exp_stats = pd.DataFrame(
        {'Mean': [total.mean() / 100], 'Min': [total.min() / 100],
         'Max': [total.max() / 100], 'SD': [total.std(ddof=1) / 100]},
        index=['Total expenditure'])[STAT_COLS]

    # Prices over purchasers only (genuine unit values)
    price_stats = _stats(p.where(q > 0))

    return share_stats, exp_stats, price_stats


def group_share_table(sample_index: pd.Index, labels: pd.Series) -> pd.DataFrame:
    """Average budget share of each milk type by upper-bound group.

    Returns a DataFrame with columns [Type, N, <good labels>]: a Pooled row
    over the whole sample, then one row per group ordered by size (largest
    first, ties broken by group label)."""
    _, _, _, shares = budget_data(sample_index)
    labels = labels.loc[sample_index]

    cols = ['Type', 'N'] + [MILK_LABELS[g] for g in goods]
    rows = [['Pooled', len(sample_index)] + list(shares.mean().values)]

    order = (labels.value_counts()
             .sort_index(kind='stable')
             .sort_values(ascending=False, kind='stable'))
    for rank, (g, n) in enumerate(order.items(), start=1):
        members = labels.index[labels == g]
        rows.append([f'Type {rank}', int(n)] + list(shares.loc[members].mean().values))

    return pd.DataFrame(rows, columns=cols)


#%% LaTeX rendering

def _rows(stats: pd.DataFrame) -> str:
    lines = []
    for label, row in stats.iterrows():
        vals = ' & '.join(f'{v:.2f}' for v in row)
        lines.append(f'\\quad {label} & {vals} \\\\')
    return '\n'.join(lines)


def to_latex(share_stats, exp_stats, price_stats, sample_desc, n_obs) -> str:
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Descriptive statistics ({sample_desc}, $N = {n_obs}$)}}
\\label{{tab:summary_{sample_desc.split()[0]}}}
\\begin{{tabular}}{{l{'r' * len(STAT_COLS)}}}
\\toprule
 & {' & '.join(STAT_COLS)} \\\\
\\midrule
\\multicolumn{{{len(STAT_COLS) + 1}}}{{l}}{{\\textit{{Budget shares}} $\\{{w_i\\}}$}} \\\\
{_rows(share_stats)}
\\midrule
\\multicolumn{{{len(STAT_COLS) + 1}}}{{l}}{{\\textit{{Total expenditure (\\$)}}}} \\\\
{_rows(exp_stats)}
\\midrule
\\multicolumn{{{len(STAT_COLS) + 1}}}{{l}}{{\\textit{{Prices (\\$ per 100 g)}} $\\{{p_i\\}}$, purchasers only}} \\\\
{_rows(price_stats)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def to_latex_groups(table: pd.DataFrame, sample_desc: str) -> str:
    good_labels = [MILK_LABELS[g] for g in goods]

    def line(row):
        vals = ' & '.join([f'{int(row["N"])}']
                          + [f'{row[c]:.2f}' for c in good_labels])
        return f'{row["Type"]} & {vals} \\\\'

    pooled = line(table.iloc[0])
    types = '\n'.join(line(row) for _, row in table.iloc[1:].iterrows())
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Average budget shares by upper-bound type ({sample_desc})}}
\\label{{tab:groupshares_{sample_desc.split()[0]}}}
\\begin{{tabular}}{{lr{'r' * len(goods)}}}
\\toprule
 & $N$ & {' & '.join(good_labels)} \\\\
\\midrule
{pooled}
\\midrule
{types}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


#%% Build and save one table per sample

os.makedirs('tables', exist_ok=True)

for sample, desc, min_types in SAMPLES:
    idx = num_milk_types[num_milk_types >= min_types].index
    share_stats, exp_stats, price_stats = summary_stats(idx)
    latex = to_latex(share_stats, exp_stats, price_stats, desc, len(idx))

    path = f'tables/summary_{sample}.tex'
    with open(path, 'w') as fh:
        fh.write(latex + '\n')

    group_tbl = group_share_table(idx, partitions[f'group_{sample}'])
    group_latex = to_latex_groups(group_tbl, desc)
    group_path = f'tables/group_shares_{sample}.tex'
    with open(group_path, 'w') as fh:
        fh.write(group_latex + '\n')

    print(f"\n=== {desc}: {len(idx)} households ===")
    print("Budget shares:\n", share_stats.round(2).to_string())
    print("Total expenditure ($):\n", exp_stats.round(2).to_string())
    print("Prices ($/100g, purchasers only):\n", price_stats.round(2).to_string())
    print("Average budget shares by upper-bound type:\n",
          group_tbl.round(2).to_string(index=False))
    print(f"Saved {path} and {group_path}")
