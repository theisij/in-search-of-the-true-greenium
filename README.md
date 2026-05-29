# In Search of the True Greenium — replication code

Replication code for **"In Search of the True Greenium"** by Eskildsen, Ibert, Jensen, and Pedersen (forthcoming, *Journal of Financial Economics*). A single end-to-end run of the pipeline regenerates every figure and table that the main paper (`main_JFE.tex`) and its Internet Appendix (`InternetAppendix.tex`) reference.

This repository was carved out of a larger codebase that produced two related papers; only the code needed for this paper is retained here. See **Provenance** below.

## Reproduces

61 figures and tables in total — the union of the main paper and Internet Appendix outputs. Highlights:

**Main paper:** estimate-by-correlation figure (`est_by_cor.pdf`), Figure 1 panels (`fig1_b.pdf`), projection loadings (`proj_loadings.pdf`), ICC figures (`icc_all.pdf` / `icc_us.pdf` / `icc_gexus.pdf`), green-variable comparison (`gvars_comb.pdf`), bond figures (`bonds.pdf`, `bonds_ratingcontrol.pdf`), WACC figures (`wacc.pdf` / `wacc_g.pdf` / `wacc_all.pdf`), green-tilts (`green_tilts3.pdf`), the random-composite figures (`random_icc_t.pdf`, `random_average_MT.pdf`), plus tables `esg_variables.tex`, `est_by_cor_bs.tex`, `ccpi_reg_new.tex`.

**Internet Appendix:** descriptive-statistics figures and tables, the `all_*` / `us_*` / `global_*` regression panels, the `bond_*` panels, individual-estimate tables, `realizedret.pdf`, `clustering.pdf`, the individual/decile/time-trend ICC figures, and `icc_hb.pdf`.

## Requirements

- **Python 3.13+** via [`uv`](https://docs.astral.sh/uv/) (pinned in `uv.lock`)
- **R 4.5+** with packages pinned in `renv.lock` (managed by `renv`)
- **poppler** (`pdftoppm`) for any rendering-based output checks
- **~120 GB free disk** (≈30 GB raw data + pipeline intermediates + working set)
- **~64 GB RAM** (the panel-OLS bootstraps are the heaviest step)
- Local execution only — there is no cluster/SLURM path.

## Setup

```bash
# Python environment
uv sync

# R environment (renv required system-wide)
Rscript -e 'install.packages("renv"); renv::restore(prompt = FALSE)'

# Place the raw inputs in data/raw/. They are not redistributed here;
# data/CHECKSUMS.sha256 lists every expected file with its SHA-256.
```

## Run

```bash
bash run.sh
```

`run.sh` sources determinism settings (`SOURCE_DATE_EPOCH=0`, `PYTHONHASHSEED=0`, pinned matplotlib rc params, `LC_ALL=C.UTF-8`) and then runs the Python pipeline followed by the R post-processing:

1. `uv run python -u -m greenium.main` (≈ 3.5 h on a first `FULL_RUN=True` run; the bootstrap regeneration dominates)
2. `Rscript R/main.R` (≈ 45 min)

Outputs are written to `output/figures/` and `output/tables/`. To reuse cached intermediates on subsequent runs, set `FULL_RUN = False` in `greenium/config.py`.

## Provenance

This repository was created by pruning the combined-project repository at commit `2998f124d806338c9965f6c7b1e455e8abcc9a9b` down to only the code paths that produce this paper's figures and tables. The pipeline is deterministic on the pinned environment (`pyproject.toml`, `uv.lock`, `renv.lock`); R's `pdf()` device stamps wall-clock dates in PDF metadata, so consecutive runs match visually/numerically but are not byte-identical. The full per-edit pruning log and the figure/table verification reports are maintained separately in the project's private build-tracking repository.
