# Raw data classification for the JFE replication package

Decided 2026-10-19 (Theis). Drives: pseudo-data generator (which folders to
mock), public-data packaging (which folders to upload), and README updates.

## Include as real data (redistributable)

| Path | Source | Note |
|---|---|---|
| `raw/ccpi/CCPI_data.xlsx` | CCPI annual reports | collected manually |
| `raw/hoberg_phillips/tnic3_data.zip` | Hoberg–Phillips data library | free for academic use |
| `raw/vilkov_options/glb_daily.zip`, `MartinWagnerBounds.zip` | Vilkov OSF | 529 MB |
| `raw/icc/icc_us.csv`, `icc_comp.csv`, `forward_eps_us.csv`, `forward_eps_comp.csv` | theisij/implied-cost-of-capital | derived estimates, 926 MB |
| `raw/other/ME_Breakpoints.csv` | Ken French data library | |
| `raw/other/countries.zip` | Natural Earth 110m admin-0 | public domain; must ship (the download branch in `main.py` is disabled). Added to README |
| `raw/other/gics_names.xlsx` | MSCI/S&P GICS structure | used in `R/green_tilts.R`. Added to README |
| `raw/other/country_abbr.xlsx` | ships in repo | already in README |

## Pseudo data (proprietary or derived from proprietary sources)

| Path | Underlying source |
|---|---|
| `raw/jkp/Characteristics.zip`, `return_cutoffs.csv`, `rf.csv` | CRSP / Compustat via JKP code |
| `raw/msci/*` (annual 2007–2022) | MSCI ESG |
| `raw/sustainalytics/*.zip`, `G_NAMES.csv`, `IDS_TABLE.csv` | Sustainalytics via WRDS |
| `raw/trucost/trucost.zip` | Trucost via WRDS |
| `raw/morningstar/*.csv` | Morningstar Direct |
| `raw/linking/WRDS_GVKEY.csv`, `WRDS_ISIN.csv` | Capital IQ via WRDS |
| `raw/sers/subjective_er.csv`, `value_line.csv`, `morningstar.csv` | I/B/E/S, Value Line, Morningstar |
| `raw/bonds/yields_traded_firm.csv` | Dick-Nielsen et al. (TRACE) |
| `raw/other/industry_codes.csv` | CRSP / Compustat SIC headers (firm-month panel) |
| `raw/other/crsp_dsenames.csv` | CRSP. Used in `R/green_tilts.R`. Added to README |

## Follow-ups for later items

- Item 5 (README): add the three `other/` files flagged **Not in README** above.
- Item 5 (README): note that `CHECKSUMS.sha256` applies to real data only.
- Package destination: Mendeley Data (~2.2 GB of real raw data).

## Package contents (data deposit)

Upload alongside a snapshot of the GitHub repository:

| Item | Source path | Note |
|---|---|---|
| Public raw data | `data/raw/icc/`, `data/raw/vilkov_options/`, `data/raw/hoberg_phillips/`, `data/raw/ccpi/`, `data/raw/other/{ME_Breakpoints.csv,countries.zip,gics_names.xlsx,country_abbr.xlsx}` | ≈1.6 GB; hashes in `data/CHECKSUMS.sha256` |
| Pseudo data | `data/raw_pseudo/` (output of `scripts/make_pseudo_data.py --seed 1`) | ≈100 MB; one file per proprietary input |
| Code | GitHub repository at the final commit | includes `data/pseudo_schema.json` and the two `scripts/*pseudo*` scripts |
