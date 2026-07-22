"""AIDS levels and expenditure semi-elasticities by reduced type, formatted as
LaTeX tables akin to Crawford & Pendakur (2013) Table 3.

For each (reduction method, number of types) in CONFIGS, estimates AIDS
(quaids.py, quadratic=False) on each type of that type reduction
(typeReduction{CCEI,Varian}.py / runReduction.py) for the chosen sample, plus
a pooled row estimated on all sample households. Tabulates the two panels C&P
report -- levels a^j and expenditure semi-elasticities b^j -- with standard
errors, omitting the price matrix. Prices and expenditure use a common median
normalisation over the whole sample, so level coefficients are comparable
across types and across the two tables.

A milk type that no household in a group ever purchases has an identically
zero budget share; AIDS for that group is estimated on the goods it does buy
and the absent good is reported as 0.000 (--).

Caveat: unlike C&P's upper-bound groups, the reduced types are the
parsimonious partition and are not perfectly rationalisable (merging incurs
efficiency loss), so within-type residuals mix specification error with
residual preference heterogeneity.

Outputs tables/aids_types_{method}_{sample}.tex (requires \\usepackage{booktabs}).
"""

#%% Preamble

import numpy as np
import pandas as pd

import os

# Set working directory to the parent of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.dirname(script_dir)
os.chdir(working_dir)

from quaids import quaids

SAMPLE = 'all'                          # 'all' or 'multi'
CONFIGS = [('ccei', 8), ('varian', 5)]  # (reduction method, number of types)

METHOD_NAMES = {'ccei': 'CCEI', 'varian': 'Varian'}
SAMPLE_NAMES = {'all': 'all households',
                'multi': 'multiple-milk-type households'}

# Readable labels for the usdafoodcat4 milk categories, in fat-content order
MILK_LABELS = {1002: 'Whole', 1004: '2\\%', 1006: '1\\%', 1008: 'Skim'}


#%% Load data

quantities = pd.read_parquet('working_data/milk_quantities.parquet') \
               .rename(columns=MILK_LABELS)
prices = pd.read_parquet('working_data/milk_prices.parquet') \
           .rename(columns=MILK_LABELS)
partitions = pd.read_parquet('working_data/milk_partitions.parquet')

goods = list(quantities.columns)

# Common normalisation constraint: medians over the whole sample
sample_idx = partitions[f'group_{SAMPLE}'].dropna().index
price_medians = prices.loc[sample_idx].median()
exp_median = (prices.loc[sample_idx] * quantities.loc[sample_idx]) \
    .sum(axis=1).median()


#%% AIDS estimation helper

def estimate_row(label, member_idx):
    """Estimate AIDS on the given households and return a table-row dict.
    A never-bought good has a constant zero share and is dropped from the
    estimation, then reported as 0.000 (--)."""
    bought = [c for c in goods if (quantities.loc[member_idx, c] > 0).any()]
    print(f"\n--- {label} ({len(member_idx)} households, goods {bought}) ---")
    res = quaids(quantities.loc[member_idx, bought],
                 prices.loc[member_idx, bought], quadratic=False,
                 price_medians=price_medians, exp_median=exp_median)
    row = {'label': label, 'N': len(member_idx)}
    for good in goods:
        if good in bought:
            row[f'alpha_{good}'] = res.alpha[good]
            row[f'alpha_se_{good}'] = res.alpha_se[good]
            row[f'beta_{good}'] = res.beta[good]
            row[f'beta_se_{good}'] = res.beta_se[good]
        else:  # never bought: share is identically zero
            row[f'alpha_{good}'] = 0.0
            row[f'alpha_se_{good}'] = None
            row[f'beta_{good}'] = 0.0
            row[f'beta_se_{good}'] = None
    return row


# Pooled AIDS over the whole sample is identical across configs, so estimate
# it once
pooled_row = estimate_row('Pooled', sample_idx)


#%% LaTeX rendering (two panels, C&P Table 3 style)

def _coef(v):
    return f'{v:.3f}'


def _se(v):
    return '--' if v is None else f'({v:.3f})'


def _panel(rows, coef_key, se_key):
    lines = []
    for i, r in enumerate(rows):
        coefs = ' & '.join(_coef(r[f'{coef_key}_{g}']) for g in goods)
        ses = ' & '.join(_se(r[f'{se_key}_{g}']) for g in goods)
        lines.append(f"{r['label']} & {r['N']} & {coefs} \\\\")
        lines.append(f" & & {ses} \\\\")
        if i == 0:  # subtle gap after the Pooled row
            lines.append('\\addlinespace')
    return '\n'.join(lines)


def to_latex(rows, method, n_types):
    ncol = len(goods) + 2
    good_headers = ' & '.join(goods)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{AIDS levels and expenditure semi-elasticities by {METHOD_NAMES[method]}-reduced
type ({SAMPLE_NAMES[SAMPLE]}, {n_types} types). Standard errors in parentheses;
0.000 (--) marks a milk type the group never purchases.}}
\\label{{tab:aids_types_{method}_{SAMPLE}}}
\\begin{{tabular}}{{lr{'r' * len(goods)}}}
\\toprule
 & $N$ & {good_headers} \\\\
\\midrule
\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Levels}}, $a^j$}} \\\\
{_panel(rows, 'alpha', 'alpha_se')}
\\midrule
\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Expenditure semi-elasticities}}, $b^j$}} \\\\
{_panel(rows, 'beta', 'beta_se')}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


#%% Build one table per configuration

os.makedirs('tables', exist_ok=True)

for method, n_types in CONFIGS:
    reduction = pd.read_parquet(
        f'working_data/milk_reduction_{method}_{SAMPLE}.parquet')
    types = reduction[f'k{n_types}'].dropna().astype(int)
    order = (types.value_counts()
             .sort_index(kind='stable')
             .sort_values(ascending=False, kind='stable'))

    print(f"\n{'=' * 70}\n{METHOD_NAMES[method]} reduction, {n_types} types\n{'=' * 70}")
    type_rows = [estimate_row(f'Type {rank}', types.index[types == g])
                 for rank, (g, _) in enumerate(order.items(), start=1)]
    rows = [pooled_row] + type_rows

    path = f'tables/aids_types_{method}_{SAMPLE}.tex'
    with open(path, 'w') as fh:
        fh.write(to_latex(rows, method, n_types) + '\n')

    preview = pd.DataFrame([
        {'Type': r['label'], 'N': r['N'],
         **{f'a_{g}': round(r[f'alpha_{g}'], 3) for g in goods},
         **{f'b_{g}': round(r[f'beta_{g}'], 3) for g in goods}}
        for r in rows])
    print(f"\nAIDS by {METHOD_NAMES[method]}-reduced type "
          f"({SAMPLE_NAMES[SAMPLE]}, {n_types} types):")
    print(preview.to_string(index=False))
    print(f"Saved {path}")
