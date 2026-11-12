# Predicting Active Users from Early Behavior: Evidence from NetEase Cloud Music

Code for an MSc thesis in Business Analytics at Nova School of Business and Economics. The thesis studies one month of
the content feed of NetEase Cloud Music (Zhang et al. 2020) and answers three questions:

1. What are the characteristics and preferences of active users?
2. How can the platform predict whether a user will be active or inactive from early actions such as clicks, likes and shares?
3. How can the recommender be designed to maximize the number of active users?

Every number and chart in the thesis is produced by the notebooks in `notebooks/`. Each notebook shows the results of
the last run on all users, so it can be read without the data.

## Contents

| Path | Content |
|---|---|
| `notebooks/` | the seven notebooks of the analysis, run in the order below |
| `src/thesis/` | shared code: paths and settings, the snapshot panel, statistics, the click model and the slow steps |
| `outputs/` | saved results as JSON and CSV, and the charts in `outputs/figures/` |
| `run_all.sh` | runs every notebook from top to bottom |
| `run_heavy.py` | runs the slow steps without the notebooks |

## Notebooks

| Notebook | What it does |
|---|---|
| `01_data` | converts the raw files, audits them, puts the feed position on one scale, explores the data and builds the panel of snapshots |
| `02_rq2_prediction` | predicts return and clicks, checks calibration, tests what early actions add, builds and evaluates the inactive label |
| `03_rq3_recommender` | fits the click model, measures the gain from re-ordering cards within a page, relates first-page content to return and simulates the policies |
| `04_rq1_active_users` | describes active users by what they do the following week, their characteristics and their preferences |
| `A1_model_checks` | compares learners and tests user weights, creator features and variety features |
| `A2_segmentation` | clusters users by how they use the feed |
| `05_figures` | draws every chart of the thesis from the saved results |

The notebooks run in the order 01, 02, 03, 04, A1, A2, 05. A snapshot is one user on one day the user visited the feed,
with the user's behavior up to that day and what the user did in the seven days after it.

## Data

The data is not included. It was released for research by NetEase Cloud Music and the Revenue Management and Pricing
Section of INFORMS and is subject to the provider's terms. Place the six CSV files, unchanged, in

    data/csv_data/   impression_data.csv, user_demographics.csv, mlog_demographics.csv, mlog_stats.csv,
                     creator_demographics.csv, creator_stats.csv

or point `THESIS_DATA` to any folder that contains `csv_data/`. Derived tables are written to `parquet/` in the same folder.

The raw feed position mixes three ways of numbering the first page. Notebook 01 therefore writes a copy of the
impressions with each card's place counted from 0 on every page (`src/thesis/positions.py`), and every later step
reads that copy.

## Running

Python 3.10 or newer is required, and on macOS `brew install libomp` for LightGBM.

    ./run_all.sh                  # creates .venv, installs the requirements and runs every notebook
    ./run_all.sh --from 03        # starts at a notebook (earlier results must exist)
    ./run_all.sh --only A1        # runs one notebook
    ./run_all.sh --sample 10      # quick test on 10% of users (results get the suffix _s10)

A full run takes about an hour and a half on a laptop. On a Mac, `caffeinate -i ./run_all.sh` keeps it awake.
`THESIS_MEM` (default 8GB) and `THESIS_THREADS` (default 8) set the resources of DuckDB.

The slow steps (model fits, resampling, scans of the impressions) are functions in `src/thesis/heavy/`. A notebook asks
for a step with `ensure("02.models")`, which reads the step's saved results from `outputs/` and runs the step only when
a result is missing. A notebook therefore reruns in seconds once its results exist. The large result files are not in
this repository, so the first run computes them.

    python run_heavy.py                   # every step whose results are missing
    python run_heavy.py --force 02.label  # recomputes a step even if its results are saved
    python run_heavy.py --list            # every step and whether its results are saved

Results are not recomputed when the code or the data changes. Use `--force` after a change, also for the steps that
read the changed results.

## Reproducibility

Users are assigned to the training, validation and test sets by an md5 hash of the user id. Every random step has a
fixed seed, ties in every ordering are broken by a row id, and every query that feeds a model has a full `ORDER BY`.
Package versions are pinned in `requirements.txt`.

## Reference

Zhang, D. J., Hu, M., Liu, X., Wu, Y., and Li, Y. (2020). NetEase Cloud Music Data. Available at SSRN 3554826.
