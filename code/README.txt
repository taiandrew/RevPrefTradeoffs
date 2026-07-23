Pipeline run order
==================

Run everything from the repo root (each script chdir's there on its own):

1. readData.py          Parse the raw FoodAPS csvs (data/) into working_data.
2. imputePrices.py      Build the household-level milk panel:
                        milk_quantities/milk_prices/milk_types parquets,
                        with missing prices imputed by cell medians.
3. runClustering.py     Crawford-Pendakur lower/upper bounds on the number
                        of types; saves the best upper-bound partition to
                        milk_partitions.parquet (column group_all). Sets
                        k_max for the schedules.
4. summarizeData.py     Descriptive tables (C&P Tables 1-2 style):
                        tables/summary_all.tex, tables/group_shares_all.tex.
                        Any time after step 3.
5. runScheduleCCEI.py   Rationality schedules: for each k = 1..k_max,
   runScheduleVarian.py rebuild the partition greedily and report
                        rationality (min group CCEI / average Varian
                        multiplier). Independent of each other; both need
                        step 3. Varian is the slow one.
6. graphs.py            All graphs, built only from saved outputs: group
                        sizes and the rationality-by-types graphs.
                        Skips any schedule not yet run.
7. runAIDS.py           AIDS (quaids.py, quadratic=False) per type;
                        saves working_data/aids_coefficients.csv.
8. tableAIDS.py         LaTeX tables of the AIDS estimates
                        (tables/aids_types_*.tex).

Tests:  cd code && python -m pytest tests/
        (includes an integration test on a 400-household subset of the
        real data; skipped if working_data is absent)

Retired scripts live in code/tags/ (merge-based type reduction, QUAIDS
driver, and their tests). notes.lyx documents the algorithms.
