# Overview

This repository contains the code used for the paper [In Search of the True Greenium](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4744608) by Eskildsen, Ibert, Jensen, and Pedersen (forthcoming, *Journal of Financial Economics*). Please cite this paper if you are using the code:

```
@article{EskildsenIbertJensenPedersen2026,
    author = {Eskildsen, Marc Baert and Ibert, Markus and Jensen, Theis Ingerslev and Pedersen, Lasse Heje},
    title = {In Search of the True Greenium},
    journal = {Journal of Financial Economics},
    year = {2026}
}
```

Please send questions about the code to Theis I. Jensen at [theis.jensen@yale.edu](mailto:theis.jensen@yale.edu).

A single end-to-end run of the pipeline regenerates every figure and table in the paper and its Internet Appendix.


# How to run the code

Clone this repository to your local machine, set up the two language environments, place the raw data under `data/raw/` (see **Data** below), and run the pipeline.

## Requirements

- **Python 3.13+**, managed with [`uv`](https://docs.astral.sh/uv/) (exact versions pinned in `uv.lock`).
- **R 4.5+**, with packages pinned in `renv.lock` (managed by [`renv`](https://rstudio.github.io/renv/)).
- Roughly **120 GB of free disk** (≈30 GB of raw data plus pipeline intermediates) and **~64 GB of RAM** (the bootstrap steps are the most demanding).

## Environment setup

```bash
# Python
uv sync

# R (renv must be installed system-wide first)
Rscript -e 'install.packages("renv"); renv::restore(prompt = FALSE)'
```

## Data

The pipeline reads the raw inputs below. Each must be placed in the indicated
folder under `data/raw/` with the exact filename shown. `data/CHECKSUMS.sha256`
lists every expected file with its SHA-256 hash.

- `data/raw/jkp/`
    - `Characteristics.zip` — firm characteristics from the JKP Global Factor Data.
    - `return_cutoffs.csv` — extreme-return cutoffs.
    - `rf.csv` — risk-free rate.
    - _(source: TODO — authors to complete)_
- `data/raw/icc/`
    - `icc_us.csv`, `icc_comp.csv` — implied cost of capital, US and global.
    - `forward_eps_us.csv`, `forward_eps_comp.csv` — forward EPS forecasts, US and global.
    - _(source: TODO — authors to complete)_
- `data/raw/msci/`
    - MSCI ESG ratings ("ESG Ratings Timeseries Expanded"), annual files 2007–2022 (`.xlsx` / `.csv`).
    - _(source: TODO — authors to complete)_
- `data/raw/sustainalytics/`
    - `sust_indicator_all.zip`, `sust_legacy.zip`, `sust_ref_data.zip` — Sustainalytics ESG indicators and reference data.
    - `G_NAMES.csv`, `IDS_TABLE.csv` — identifier/name tables.
    - _(source: TODO — authors to complete)_
- `data/raw/trucost/`
    - `trucost.zip` — Trucost firm-level carbon-emissions data.
    - _(source: TODO — authors to complete)_
- `data/raw/morningstar/`
    - `Morningstar_holdings_consolidated.csv`, `Morningstar_holdings_consolidated_notArticle9.csv`, `Morningstar_holdings_consolidated_notArticle9_US.csv` — fund holdings.
    - `Sustainable_Funds_Processed.csv` — sustainable-funds classification.
    - _(source: TODO — authors to complete)_
- `data/raw/sers/`
    - `subjective_er.csv`, `value_line.csv`, `morningstar.csv` — subjective / analyst expected-return data.
    - _(source: TODO — authors to complete)_
- `data/raw/vilkov_options/`
    - `glb_daily.zip` — option-implied data (global daily).
    - `MartinWagnerBounds.zip` — Martin–Wagner expected-return bounds.
    - _(source: TODO — authors to complete)_
- `data/raw/bonds/`
    - `yields_traded_firm.csv` — corporate bond yields by firm.
    - _(source: TODO — authors to complete)_
- `data/raw/hoberg_phillips/`
    - `tnic3_data.zip` — Hoberg–Phillips TNIC3 text-based industry network.
    - _(source: TODO — authors to complete)_
- `data/raw/ccpi/`
    - `CCPI_data.xlsx` — Climate Change Performance Index country scores.
    - _(source: TODO — authors to complete)_
- `data/raw/linking/`
    - `WRDS_GVKEY.csv`, `WRDS_ISIN.csv` — WRDS identifier linking tables.
    - _(source: TODO — authors to complete)_
- `data/raw/other/`
    - `industry_codes.csv` — industry classification codes.
    - `ME_Breakpoints.csv` — NYSE size (market-equity) breakpoints.
    - `country_abbr.xlsx` — country abbreviations.
    - _(source: TODO — authors to complete)_

## Run

```bash
bash run.sh
```

This sources the determinism settings in `scripts/env.sh` and then runs the two pipeline stages:

1. `uv run python -u -m greenium.main` — the main Python pipeline (≈ 3.5 h on a first run; the bootstrap regeneration dominates).
2. `Rscript R/main.R` — the R post-processing (≈ 45 min).

Figures are written to `output/figures/` and tables to `output/tables/`. To reuse cached intermediates instead of regenerating them from scratch, set `FULL_RUN = False` in `greenium/config.py`.
