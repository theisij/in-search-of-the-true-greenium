#!/usr/bin/env python3
"""
export_JFE.py — organize the paper's figures/tables for Overleaf upload.

After running the pipeline (`bash run.sh`), this takes the already-generated
outputs in `output/` and copies the paper-referenced ones into two flat
folders, tagged by which Overleaf document uses them:

  output/JFE_main/<stem>_JFE_main.<ext>   -> referenced by main_JFE.tex
  output/JFE_ia/<stem>_JFE_ia.<ext>       -> referenced by InternetAppendix.tex

It only copies and renames existing files; it never regenerates anything.

The two file lists below are the JFE main-paper and Internet-Appendix
keep-sets. They are disjoint (no file feeds both) and have no basename
collisions, so flattening + the suffix is unambiguous.

Run with:  uv run python export_JFE.py   (pure standard library, no deps)
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Outputs referenced by main_JFE.tex (26).
JFE_MAIN = [
    "output/figures/maps/estimate.svg",   # \includesvg{estimate.svg} — global greenium map
    "output/figures/global/est_by_cor.pdf",
    "output/figures/fig1_b.pdf",
    "output/tables/des_stats/esg_variables.tex",
    "output/figures/global/proj_loadings.pdf",
    "output/figures/global/icc_all.pdf",
    "output/figures/us/icc_us.pdf",
    "output/figures/global/icc_gexus.pdf",
    "output/figures/global/gvars_comb.pdf",
    "output/figures/global/individual_vs_aggregate_tstats.pdf",
    "output/tables/global/est_by_cor_bs.tex",
    "output/figures/global/bm_t_unbalanced.pdf",
    "output/figures/random_icc_t.pdf",
    "output/figures/random_average_MT.pdf",
    "output/figures/global/valrat_all.pdf",
    "output/figures/us/er_options.pdf",
    "output/figures/us/subj_ret.pdf",
    "output/figures/global/icc_fm_all.pdf",
    "output/figures/global/icc_timetrend_all.pdf",
    "output/tables/global/ccpi_reg_new.tex",
    "output/figures/us/bonds.pdf",
    "output/figures/us/bonds_ratingcontrol.pdf",
    "output/figures/us/wacc.pdf",
    "output/figures/global/wacc_g.pdf",
    "output/figures/global/wacc_all.pdf",
    "output/figures/green_tilts3.pdf",
]

# Outputs referenced by Internet Appendix InternetAppendix.tex (36).
JFE_IA = [
    "output/figures/des_stats/us_sample_nfirms.pdf",
    "output/figures/des_stats/global_sample_nfirms.pdf",
    "output/figures/des_stats/us_sample_nfirms_measures.pdf",
    "output/figures/des_stats/global_sample_nfirms_measures.pdf",
    "output/tables/des_stats/esg_descriptives.tex",
    "output/tables/des_stats/agg_green_scores.tex",
    "output/tables/des_stats/firm_country_count.tex",
    "output/tables/des_stats/char_descriptives.tex",
    "output/tables/des_stats/other_descriptives.tex",
    "output/tables/des_stats/option_descriptives.tex",
    "output/tables/des_stats/bond_descriptives.tex",
    "output/figures/us/realizedret.pdf",
    "output/tables/all_0.tex",
    "output/tables/all_1.tex",
    "output/tables/all_2.tex",
    "output/tables/us_0.tex",
    "output/tables/us_1.tex",
    "output/tables/us_2.tex",
    "output/tables/global_0.tex",
    "output/tables/global_1.tex",
    "output/tables/global_2.tex",
    "output/tables/bond_0.tex",
    "output/tables/bond_1.tex",
    "output/tables/bond_2.tex",
    "output/tables/bond_3.tex",
    "output/tables/bond_4.tex",
    "output/tables/bond_5.tex",
    "output/tables/global/individual_estimates.tex",
    "output/tables/global/individual_estimates_unbalanced.tex",
    "output/tables/global/individual_estimates_unbalanced_raw.tex",
    "output/tables/global/est_by_cor_bs_unbalanced.tex",
    "output/figures/clustering.pdf",
    "output/figures/global/icc_all_indv.pdf",
    "output/figures/global/icc_timetrend_all_indv.pdf",
    "output/figures/global/icc_deciles_all.pdf",
    "output/figures/us/icc_hb.pdf",
]

GROUPS = {"JFE_main": JFE_MAIN, "JFE_ia": JFE_IA}


def export_group(name: str, sources: list[str]) -> int:
    dest_dir = REPO / "output" / name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)          # fresh each run; no stale leftovers
    dest_dir.mkdir(parents=True)

    missing, copied = [], 0
    for rel in sources:
        src = REPO / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = dest_dir / f"{src.stem}_{name}{src.suffix}"
        shutil.copy2(src, dst)
        copied += 1

    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} expected output(s) not found under output/. "
            f"Run the pipeline first (`bash run.sh`). Missing:\n  "
            + "\n  ".join(missing)
        )
    print(f"{name}: {copied} copied -> output/{name}")
    return copied


def main() -> int:
    for name, sources in GROUPS.items():
        export_group(name, sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
