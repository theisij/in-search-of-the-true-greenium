# importing packages and paths
import re
import zipfile

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from linearmodels.panel import PanelOLS  #, compare
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, t
from stargazer.stargazer import Stargazer
from statsmodels.iolib.summary2 import summary_col

from greenium.aggregate_green_score import (
    direct_green_scores,
    proj_bootstrap,
    proj_bootstrap_cis,
    proj_green_scores,
    proj_plot_loadings,
)
from greenium.benchmarking import (
    agg_indv_plot,
    agg_indv_regressions,
    bh_adjustment,
    bm_data,
    bm_plot_est,
    bm_plot_t,
    bm_regressions,
    bs_est_by_cor,
    df_impute_balanced,
    new_fig1a_old,
    plot_est_by_cor,
    plot_stat_ranked,
    plot_stat_ranked_combined,
    residual_correlations,
    table_bs_est_by_cor,
    table_individual_est,
    tidy_panel_results,
)
from greenium.config import (
    AGG_GREEN_VAR_MAIN,
    AGG_GREEN_VARS,
    BS_REPEATS,
    BS_SAMPLE_GROUP,
    BS_SEED,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_RAW,
    FULL_RUN,
    MAIN_CONTROLS,
    OUTPUT_FIGURES,
    OUTPUT_FIGURES_DESCRIPTIVES,
    OUTPUT_FIGURES_GLOBAL,
    OUTPUT_FIGURES_MAPS,
    OUTPUT_FIGURES_US,
    OUTPUT_TABLES,
    OUTPUT_TABLES_DESCRIPTIVES,
    OUTPUT_TABLES_GLOBAL,
    OUTPUT_TABLES_US,
    emission_factors,
    global_start,
    msci_factors,
    set_start,
    set_start_morningstar,
    sust_factors,
    to_test,
    to_test_g,
    to_test_g_dict,
    to_test_g_names,
    to_test_names,
    total_em_factors,
    tri_factors,
)
from greenium.data_initial import load_all_data
from greenium.utils import desc_table

# loading data (either load or generate & save for next load; to re-run code from scratch, delete pickled files)
try:
    # Try loading files
    df = pd.read_parquet(DATA_INTERIM / "df_main.parquet")
    df_bonds = pd.read_parquet(DATA_INTERIM / "df_bonds.parquet")
    holdings_agg = pl.read_parquet(DATA_INTERIM / "holdings_agg.parquet")
    holdings_agg_bf = pl.read_parquet(DATA_INTERIM / "holdings_agg_bf.parquet")
    print("Interim data files loaded successfully.")
except FileNotFoundError:
    # If files don't exist, generate them
    print("Interim data files not found. Running latest data pull.")
    df, df_bonds, holdings_agg, holdings_agg_bf = load_all_data()

#### NOTE: code includes plt.close("all") throughout in order to automatically close outputs created; can be deleted
# df = df[df.date>=global_start]
# df_bonds = df_bonds[df_bonds.date>=global_start]

# loading AP factors
def load_factors():
    # FF6 factors
    df_ff6 = pd.read_excel(DATA_RAW / "factors" / "FF6_monthly.xlsx", sheet_name='FF')
    df_ff6 = df_ff6.set_index(pd.to_datetime(df_ff6['date'], format='%Y%m'))
    df_ff6.index = df_ff6.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_ff6 = df_ff6.drop(['date'], axis=1)

    # q-factors
    df_q = pd.read_csv(DATA_RAW / "factors" / "q5_factors_monthly_2022.csv")
    df_q['date'] = df_q.year.astype(str) + "-" + df_q.month.astype(str)
    df_q = df_q.set_index(pd.to_datetime(df_q['date'], format='%Y-%m'))
    df_q.index = df_q.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_q = df_q.drop(['date'], axis=1)
    df_q = df_q.drop(['R_F', 'year', 'month'], axis=1)

    # country-level market returns
    df_gmkt = pd.read_csv(DATA_RAW / "jkp" / "market_returns.csv",
                          usecols=['excntry', 'eom', 'mkt_vw_exc'])
    df_gmkt = df_gmkt.rename(columns={'eom': 'date'})
    df_gmkt = df_gmkt.set_index(pd.to_datetime(df_gmkt['date'], format='%Y%m%d'))
    df_gmkt.index = df_gmkt.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_gmkt = df_gmkt.drop(['date'], axis=1)
    df_gmkt = df_gmkt.reset_index()
    df_gmkt['mkt_vw_exc'] = df_gmkt['mkt_vw_exc'] * 100

    # country-level factors
    df_gfactors = pd.read_csv(DATA_RAW / "jkp" / "[all_countries]_[all_factors]_[monthly]_[vw_cap].csv",
        usecols=['location', 'name', 'date', 'ret'])
    df_gfactors = df_gfactors.set_index(pd.to_datetime(df_gfactors['date'], format='%Y-%m-%d'))
    df_gfactors.index = df_gfactors.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_gfactors = df_gfactors.drop(['date'], axis=1)
    df_gfactors = df_gfactors.reset_index()
    df_gfactors = df_gfactors[df_gfactors.date >= '1990-01-01']
    df_gfactors = df_gfactors.rename(columns={'location': 'excntry'})
    df_gfactors['excntry'] = df_gfactors['excntry'].str.upper()
    df_gfactors['ret'] = df_gfactors['ret'] * 100
    df_gfactors = df_gfactors[
        df_gfactors.name.isin(['market_equity', 'be_me', 'ret_12_1', 'at_gr1', 'ope_be'])]  # keeping only main factors

    df_gfactors = df_gfactors.set_index(['date', 'name', 'excntry']).unstack(level=1)
    df_gfactors.columns = df_gfactors.columns.droplevel()
    df_gfactors = df_gfactors.reset_index()

    # renaming to avoid name overlap with firm-level characteristics
    df_gfactors = df_gfactors.rename(columns={'market_equity': 'size', 'be_me': 'value', 'ret_12_1': 'momentum',
                                              'at_gr1': 'investment', 'ope_be': 'profitability'})

    # combining global MRP and AP factors
    df_gfactors = df_gmkt.merge(df_gfactors, how='left', on=['date', 'excntry'])

    return df_ff6, df_q, df_gfactors
df_ff6, df_q, df_gfactors = load_factors()

# esg variable summary statistics (before standardizing)
desc_table(
    data = df[(df.date>=global_start)],
    cols = to_test_g,
    cols_names = to_test_g_names,
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "esg_descriptives.tex",
    include_skewness=True
)

# esg variable summary statistics (before standardizing -- with EPA)
desc_table(
    data = df[(df.date>=global_start)],
    cols = to_test,
    cols_names = to_test_names,
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "esg_descriptives_with_epa.tex",
    include_skewness=True
)

# Signing factors consistently such that they all reflect greenness
for i in (total_em_factors + emission_factors + tri_factors):
    df[i] = -df[i]

# Estimate individual greeniums before they are standardized
def raw_indv_greeniums(df, y, ctrls, gvars, gvars_dict, start):
    # Make sample more manageable
    df_input = (
            pl.from_pandas(df[['id', 'date', 'gics', 'excntry', 'country_ym_num']+[y]+ctrls+gvars])
            .lazy()
            .filter(pl.col('date') >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d"))
            .drop_nulls(subset=ctrls+[y]+['gics'])
            .collect()
            .to_pandas()
        )
    # Estimate regressions
    green_est = bm_regressions(df_input=df_input, y='icc_avg', controls=ctrls, 
                               agg_green_vars=gvars, cluster_se=True)
    green_est = tidy_panel_results(green_est)
    green_est = green_est[green_est['green_variable']==green_est['term']]
    green_est = green_est.reset_index()
    # Latex table
    tbl = green_est[
            ['green_variable', 'estimate', 't_statistic', 'r_squared_overall', 
            'n_obs']
        ].rename(columns={
            'green_variable': 'Variable',
            'n_obs': 'N',
            'estimate': 'Estimate',
            't_statistic': 't-stat',
            'r_squared_overall': r'$R^2$'
        }).assign(
            # Map names from gvars_dict, escape underscores for LaTeX
            Variable=lambda df: df['Variable'].map(gvars_dict).str.replace('_', r'\_', regex=False)
        )
    tbl.to_latex(OUTPUT_TABLES_GLOBAL / "individual_estimates_unbalanced_raw.tex", 
            float_format="%.2f", index=False, column_format = "lrrrrr")
raw_indv_greeniums(df=df, y='icc_avg', ctrls=MAIN_CONTROLS, gvars=to_test_g, 
                   gvars_dict=to_test_g_dict, start=global_start)

# Create standardized measures greeness measures 
def prepare_indv_esg(df):
    # ensuring EPA TRI data is US only
    for i in tri_factors:
        df.loc[df['excntry'] != 'USA', i] = np.nan

    # creating average measures for each of the three green score components
    df['trucost_avg'] = df[emission_factors].mean(axis=1) #average of 6 trucost measures
    df['msci_avg'] = df[['weighted_esg_score', 'e_score']].mean(axis=1) #average of 2 msci measures
    df['sust_avg'] = df[['total_esg_score', 'environment_score']].mean(axis=1) #average of 2 sustainalytics measures

    # cross-sectionally standardising the ESG data - doing for each month and country separately
    for i in (to_test + ['trucost_avg', 'msci_avg', 'sust_avg']):
        df[i] = df.groupby(['date', 'excntry'])[i].transform(lambda x: (x - x.mean()) / x.std(ddof=1))

    # Output
    return df

# Save raw scope1int_assets before standardization (needed for time-series plot in fama_macbeth_full())
df['raw_scope1int_assets'] = df['scope1int_assets'].copy()

df = prepare_indv_esg(df)

# Direct aggregate green scores
df_direct = direct_green_scores(
    df=df,
    holdings_agg=holdings_agg,
    trim_perc=(0.01, 0.99)
)
df = df.merge(df_direct.to_pandas(), on=['id', 'date'], how='left')

# Backfilled direct aggregate green scores (robustness test)
df_direct_bf = direct_green_scores(
    df=df,
    holdings_agg=holdings_agg_bf,
    trim_perc=(0.01, 0.99)
)
df_direct_bf = df_direct_bf.rename({
    'gs_direct_raw': 'gs_direct_raw_bf',
    'gs_direct_log': 'gs_direct_log_bf',
    'gs_direct_rank': 'gs_direct_rank_bf',
    'share_sust': 'share_sust_bf',
})
df = df.merge(df_direct_bf.to_pandas(), on=['id', 'date'], how='left')

# Projected aggregate green scores
proj_y_vars = ['share_raw', 'share_log', 'share_rank']
trim_perc = (0.01, 0.99)
lambda_reg = 0
res_df, df_proj, r_squared = proj_green_scores(
    proj_y_vars=proj_y_vars,
    df=df,
    holdings_agg=holdings_agg,
    gvars=to_test_g,
    ctrls=MAIN_CONTROLS,
    lambda_reg=lambda_reg,
    start_date = set_start,
    trim_perc=trim_perc
)
df = df.merge(df_proj.to_pandas(), on=['id', 'date'], how='left')

# Bootstrap samples for projection loadings
if FULL_RUN:
    bs_samples = proj_bootstrap(
        proj_y_vars=proj_y_vars, df=df, holdings_agg=holdings_agg,
        gvars=to_test_g, ctrls=MAIN_CONTROLS, lambda_reg=lambda_reg,
        start_date=set_start, trim_perc=trim_perc,
        sample_group=BS_SAMPLE_GROUP, repeats=BS_REPEATS, seed=BS_SEED
    )
    bs_samples.write_parquet(DATA_INTERIM / "proj_bootstrap_samples.parquet")
else:
    bs_samples = pl.read_parquet(DATA_INTERIM / "proj_bootstrap_samples.parquet")
bootstrap_cis = proj_bootstrap_cis(bs_samples=bs_samples, q_low=0.025, q_high=0.975)

# Plot of loadings
fig = proj_plot_loadings(res_df=res_df, green_vars=to_test_g,
                         green_vars_dict=to_test_g_dict,
                         r_squared=r_squared, bootstrap_cis=bootstrap_cis)
fig.savefig(OUTPUT_FIGURES_GLOBAL / "proj_loadings.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

# Equal weighted average of all individual green scores
df['gs_ew'] = df[to_test_g].mean(axis=1)

# Create full series of direct rank
df['gs_direct_rank_imp'] = np.where(
    df['date']>=pd.to_datetime(set_start_morningstar),
    df['gs_direct_rank'],
    df['gs_proj_rank'] 
)

# Standardize aggregate green scores
for i in AGG_GREEN_VARS:
    df[i] = df.groupby(['date', 'excntry'])[i].transform(lambda x: (x - x.mean()) / x.std(ddof=1))

# Aggregate green score comparison -----------------------------------------------------
def bm_main(df, y, controls, agg_green_vars, start, balanced):
    """
    Main function to run the benchmark analysis for green scores.
    It loads the data, prepares it, runs the regressions, and plots the results.
    """
    # Create data
    df_input = bm_data(df, y=y, controls=controls, green_vars=agg_green_vars, 
                    start=start, balanced=balanced)

    # Estimate regressions
    results = bm_regressions(df_input=df_input, y='icc_avg', controls=controls, 
                             agg_green_vars=agg_green_vars)
    
    # Transform to tidy format
    results_tidy = tidy_panel_results(results)
    
    # Figures 
    filename_suffix = "balanced" if balanced else "unbalanced"
    
    # Estimates and CIs
    fig = bm_plot_est(results_tidy=results_tidy, agg_green_vars=agg_green_vars)
    fig.savefig(OUTPUT_FIGURES_GLOBAL / f"bm_est_{filename_suffix}.pdf", 
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    # T-stats
    fig = bm_plot_t(results_tidy=results_tidy, agg_green_vars=agg_green_vars)
    fig.savefig(OUTPUT_FIGURES_GLOBAL / f"bm_t_{filename_suffix}.pdf", 
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Output
    return None

# Unblanced benchmarking
bm_main(df=df, y='icc_avg', controls=MAIN_CONTROLS, agg_green_vars=AGG_GREEN_VARS, 
        start=global_start, balanced=False)
# Balanced benchmarking
bm_main(df=df, y='icc_avg', controls=MAIN_CONTROLS, agg_green_vars=AGG_GREEN_VARS, 
        start=global_start, balanced=True)

# Set main aggregate green score variable
df['green_score'] = df[AGG_GREEN_VAR_MAIN]
df.loc[df['date'] < global_start, 'green_score'] = np.nan # removing green score observations prior to start date of sample
# Consider removing all AGG_GREEN_VARS except for AGG_GREEN_VAR_MAIN

# Compare green_score to van der Beck ESG portfolio weights
def van_der_beck_comparison(df, path, save_path):
    """Quarterly Spearman correlation between our share_sust and w_ESG from van der Beck."""
    esg = pd.read_csv(path, usecols=['date', 'permno', 'w_ESG'])
    esg['date'] = pd.to_datetime(esg['date']).dt.to_period('M').dt.to_timestamp('M')
    merged = df.loc[df['permno'].notna(), ['permno', 'date', 'share_sust']].merge(
        esg, on=['permno', 'date'], how='inner'
    ).dropna(subset=['share_sust', 'w_ESG'])
    corrs = merged.groupby('date').apply(
        lambda g: g['share_sust'].corr(g['w_ESG'], method='spearman')
    ).rename('spearman_corr')
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(corrs.index, corrs.values, color='green', linewidth=1)
    ax.axhline(0, color='black', alpha=0.4, linewidth=0.8)
    ax.set_ylabel('Spearman correlation')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Van der Beck comparison saved to {save_path} (mean ρ = {corrs.mean():.3f})")

van_der_beck_comparison(
    df=df,
    path=DATA_RAW / "other" / "ESG_Portfolio.csv",
    save_path=OUTPUT_FIGURES_US / "van_der_beck_comp.pdf"
)

# Individual versus aggregate green score (bivariate regressions)
def agg_indv_main(df, green_vars, green_vars_dict, controls):
    """
    Main function to run the aggregate vs individual regression and plot
    """
    # Run the regression
    results = agg_indv_regressions(df, green_vars, controls)
    # Tidy results
    results_tidy = tidy_panel_results(results)
    # Multiple adjustment testing 
    bh = bh_adjustment(results_tidy, 0.05)
    print("Benjamini-Hochberg adjusted p-values for multiple testing:", 
          bh[0].sum(), "out of", len(bh[0]), "tests are significant at 5% level.")
    # Create the plot
    fig = agg_indv_plot(results_tidy, green_vars, green_vars_dict)
    fig.savefig(OUTPUT_FIGURES_GLOBAL / "individual_vs_aggregate_tstats.pdf", 
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    # New figure 1a
    new_fig1a_old(results_tidy)
    
    return None

agg_indv_main(df=df, green_vars=to_test_g, green_vars_dict=to_test_g_dict, 
              controls=MAIN_CONTROLS)

# Individual greenium estimates explain by correlation with aggregate?
def indv_by_agg(df, y, agg_var, indv_vars, ctrls, start, gvars_dict,
                bs_sample_group, bs_repeats, bs_seed,
                unbalanced=False):
    gvars = indv_vars + [agg_var]
    if unbalanced:
        df_input = (
            pl.from_pandas(df[['id', 'date', 'gics', 'excntry', 'country_ym_num']+[y]+ctrls+gvars])
            .lazy()
            .filter(pl.col('date') >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d"))
            .filter(pl.col(agg_var).is_not_null())
            .drop_nulls(subset=ctrls+[y]+['gics'])
            .collect()
            .to_pandas()
        )
        file_suffix = '_unbalanced'
    else:
        # Imputed and balanced data
        df_input = df_impute_balanced(df=df, agg_var=agg_var, indv_vars=indv_vars, 
                                    gvars=gvars, ctrls=ctrls, y=y, start=start)
        df_input = df_input.to_pandas()
        file_suffix = ''
    # Estimate main regression for each gvars
    green_est = bm_regressions(df_input=df_input, y='icc_avg', controls=ctrls, 
                               agg_green_vars=gvars, cluster_se=True)
    green_est = tidy_panel_results(green_est)
    green_est = green_est[green_est['green_variable']==green_est['term']]
    green_est = green_est.reset_index()
    # Estimate residual correlations
    residual_cors = residual_correlations(df_input=df_input, agg_var=agg_var, 
                                      indv_vars=indv_vars, ctrls=ctrls)
    # Combine results
    cors_est = green_est.merge(residual_cors, on='green_variable', how='left')
    # Bootstrap estimates and correlations
    if FULL_RUN:
        cors_est_bs = bs_est_by_cor(df_input=df_input, sample_group=bs_sample_group, 
                                    repeats=bs_repeats, seed=bs_seed, agg_var=agg_var, 
                                    indv_vars=indv_vars, ctrls=ctrls)
        cors_est_bs.to_parquet(DATA_INTERIM / f"cors_est_bs{file_suffix}.parquet") # Save for faster experimentation
    else:
        cors_est_bs = pd.read_parquet(DATA_INTERIM / f"cors_est_bs{file_suffix}.parquet")
    # Table (estimates and bootstraped standard errors)
    tbl = table_bs_est_by_cor(cors_est=cors_est, cors_est_bs=cors_est_bs, agg_var=agg_var)    
    tbl.to_latex(OUTPUT_TABLES_GLOBAL / f"est_by_cor_bs{file_suffix}.tex", 
                float_format="%.2f", index=False, column_format = "lrrrrr")
    # Save individual estimates and correlations
    tbl = table_individual_est(cors_est=cors_est, gvars_dict=gvars_dict)
    tbl.to_latex(OUTPUT_TABLES_GLOBAL / f"individual_estimates{file_suffix}.tex",
                 float_format="%.2f", index=False, column_format = "lrrrrrr",
                 formatters={r'$R^2$': '{:.3f}'.format, r'$\Delta R^2$': '{:.3f}'.format})
    # Plots 
    fig = plot_est_by_cor(cors_est=cors_est, agg_var=agg_var)
    fig.savefig(OUTPUT_FIGURES_GLOBAL / f"est_by_cor{file_suffix}.pdf", format="pdf", bbox_inches="tight")
    if True:
        fig = plot_stat_ranked_combined(green_est=cors_est, green_vars_dict=gvars_dict)
        fig.savefig(OUTPUT_FIGURES_GLOBAL / f"gvars_comb{file_suffix}.pdf", format="pdf", bbox_inches="tight")
    else:
        fig = plot_stat_ranked(green_est=green_est, stat='estimate', x_label='Greenium estimate (bps)')
        fig.savefig(OUTPUT_FIGURES_GLOBAL / "gvars_est.pdf", format="pdf", bbox_inches="tight")
        fig = plot_stat_ranked(green_est=green_est, stat='t_statistic', x_label='Greenium $t$-statistic')
        fig.savefig(OUTPUT_FIGURES_GLOBAL / "gvars_t.pdf", format="pdf", bbox_inches="tight")
        fig = plot_stat_ranked(green_est=green_est, stat='r_squared_overall', x_label='Greenium R$^2$', revert_x=False)
        fig.savefig(OUTPUT_FIGURES_GLOBAL / "gvars_r2.pdf", format="pdf", bbox_inches="tight")

    return None

# Main: Balanced with imputations
indv_by_agg(df=df, y='icc_avg', agg_var='green_score', indv_vars=to_test_g, 
            gvars_dict=to_test_g_dict,
            ctrls=MAIN_CONTROLS, start=global_start, 
            bs_sample_group=BS_SAMPLE_GROUP, bs_repeats=BS_REPEATS, bs_seed=BS_SEED,
            unbalanced=False)
# Alternative: Unbalanced
indv_by_agg(df=df, y='icc_avg', agg_var='green_score', indv_vars=to_test_g, 
            gvars_dict=to_test_g_dict,
            ctrls=MAIN_CONTROLS, start=global_start, 
            bs_sample_group=BS_SAMPLE_GROUP, bs_repeats=BS_REPEATS, bs_seed=BS_SEED,
            unbalanced=True)
# Prepare some additional data
def prepare_additional(df, df_bonds):
    # creating green score decile dummies by country
    temp_data = df[df.green_score.notna()][['date', 'id', 'excntry', 'green_score']]
    temp_data['esg_rank'] = temp_data.groupby(['date', 'excntry'])['green_score'].rank(pct=True)
    temp_data['decile_esg'] = temp_data.groupby(['date', 'excntry'])['esg_rank'].transform(lambda x: pd.cut(x, 10, retbins=False, labels=['q_1_10', 'q_2_10', 'q_3_10', 'q_4_10', 'q_5_10',
                                                                                                                               'q_6_10', 'q_7_10', 'q_8_10', 'q_9_10', 'q_10_10']))
    decile_dummies = pd.get_dummies(temp_data['decile_esg'])
    temp_data = pd.concat([temp_data, decile_dummies], axis=1)
    temp_data = temp_data.drop(['green_score', 'esg_rank', 'decile_esg', 'excntry'], axis=1) #dropping unneeded columns
    df = df.merge(temp_data, how='left', on=['id', 'date'])

    # adding firm information to bond data
    df_bonds = df_bonds.merge(df[['date', 'gvkey', 'id', 'me_company', 'gics', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'ms_coe', 'vl_req_ret', 'icc_avg', 'green_score'] + total_em_factors + emission_factors + msci_factors + sust_factors], how='left', on=['date', 'gvkey'])
    df_bonds = df_bonds.drop_duplicates(['gvkey', 'date'], keep='first') # gvkey merge gives duplicates
    df_bonds['market_leverage'] = df_bonds['debt_mv'] / df_bonds['me_company'] * 100 # calculating implied debt to equity market value

    # calculate pre-tax WACC by weighting the adjusted yield and average ICC
    df_bonds['wacc_icc_avg'] = (df_bonds['yield_adj'] * df_bonds['debt_mv'] + df_bonds['icc_avg'] * df_bonds['me_company']) / (df_bonds['debt_mv'] + df_bonds['me_company'])

    return df, df_bonds

df, df_bonds = prepare_additional(df=df, df_bonds=df_bonds) # For Kelly Shue: df_bonds[['date', "gvkey", "debt_mv", "yield_adj", "me_company", "icc_avg", "wacc_icc_avg"]].to_csv('wacc.csv', index=False)

# ESG count table with descriptive statistics
print('Descriptive statistics...')
def esg_var_export():
    # extracting variable list with time periods, avg. count etc. to LaTeX (where we add description etc.)
    esg_var = pd.DataFrame(index=['green_score']+to_test_g, columns=['time period', 'Avg. N (US)', 'Avg. N (G)', 'source', 'reference'])
    esg_var['source'] = ['Several', 'Trucost', 'Trucost', 'Trucost', 'Trucost', 'Trucost', 'Trucost', 'Trucost', 'Trucost', 'Trucost',
                         'MSCI', 'MSCI', 'MSCI', 'MSCI', 'MSCI', 'MSCI', 'MSCI', 'MSCI',
                         'Sustainalytics', 'Sustainalytics'] # , 'EPA TRI', 'EPA TRI', 'EPA TRI', 'EPA TRI'

    # Loop through each index in esg_var
    for i in esg_var.index:
        # Filter the relevant data and sort by 'date'
        temp = df[df[i].notna()][['date', i]].drop_duplicates(subset='date').sort_values('date')

        # Extract the first and last dates, formatted as "YYYY-MM"
        first_row = temp['date'].iloc[0].strftime('%Y-%m')
        last_row = temp['date'].iloc[-1].strftime('%Y-%m')

        # Set the 'time period' for the current index in esg_var
        esg_var.at[i, 'time period'] = f"{first_row} to {last_row}"

        # Calculate 'Avg. N (US)' for the current index
        us_avg = df[(df['excntry'] == "USA") & (df[i].notna())].groupby('date')[i].count().mean()
        esg_var.at[i, 'Avg. N (US)'] = round(us_avg) if pd.notna(us_avg) else 0

        # Calculate 'Avg. N (G)' for the current index (Global except USA)
        global_avg = df[(df['excntry'] != "USA") & (df[i].notna())].groupby('date')[i].count().mean()
        esg_var.at[i, 'Avg. N (G)'] = round(global_avg) if pd.notna(global_avg) else 0

    # Ensure 'Avg. N (US)' and 'Avg. N (G)' columns are filled with 0 where NaN, and rounded to 0 decimal places
    esg_var['Avg. N (US)'] = esg_var['Avg. N (US)'].fillna(0).round(0).infer_objects(copy=False)
    esg_var['Avg. N (G)'] = esg_var['Avg. N (G)'].fillna(0).round(0).infer_objects(copy=False)

    esg_var.index = ['Aggregate Green Score'] + to_test_g_names

    # new_order = ['Aggregate Green Score', 'S1INT (Sales)', 'S1+2INT (Sales)', 'S1+2+3INT (Sales)', 'S1INT (Assets)', 'S1+2INT (Assets)', 'S1+2+3INT (Assets)',
    #              'Weighted ESG score', 'Environment score', 'Total ESG score', 'Environmental score',
    #              'LOG(S1TOT)', 'LOG(S1+2TOT)', 'LOG(S1+2+3TOT)', 'Ind.-adj. ESG score',
    #              'Greenness (PST)', 'E climate score', 'E nat. res. score', 'E waste score', 'E env. opps. score',
    #              'TRI (Sales)', 'TPWINT (Sales)', 'TRI (Assets)', 'TPWINT (Assets)']  # Desired order of index names
    # esg_var = esg_var.reindex(new_order)
    esg_var = esg_var.reset_index()
    esg_var.columns = ['Name', 'Time period', 'Avg. $N$ (US)', 'Avg. $N$ (Global ex-US)', 'Source', 'Reference']

    #esg_var = pd.read_csv(r"C:\Users\meskilds\Dropbox (CBS)\ESG_EJIP\esg_var_import.csv")
    pd.set_option('display.max_colwidth', None)
    esg_var.iloc[:, :-1].to_latex(OUTPUT_TABLES_DESCRIPTIVES / "esg_variables.tex", float_format="%.0f", index=False,
                     column_format='l c c c c p{5cm}')

    # exporting simplified version to slides
    esg_var.reset_index()[['Name', 'Source', 'Reference']].to_latex(
        OUTPUT_TABLES_DESCRIPTIVES / "esg_variables_ref.tex", float_format="%.0f", index=False)

    return None
esg_var_export()

# Descriptive statistics ---------------------------------------------------------------
# Aggregate green scores and holdings
df['share_sust_perc'] = df['share_sust'] * 100
desc_table(
    data = df[df['date'] >= global_start],
    cols = ['share_sust_perc', 'gs_direct_raw', 'gs_direct_log', 'gs_direct_rank',
            'gs_proj_raw', 'gs_proj_log', 'gs_proj_rank', 'green_score'],
    cols_names = ['Green share (\\%)', 'Green share', 'Log(green share)',
                  'Rank(green share)', 'IV(green share)', 'IV(log(green share))',
                  'IV(rank(green share))', 'Aggregate green score'],
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "agg_green_scores.tex",
    include_skewness=True,
    # Several columns here are standardized to mean zero; without this the
    # rounded mean sometimes renders as "-0.00" depending on the side of
    # zero the FP residual lands on.
    canonicalize_zero=True,
)
df = df.drop('share_sust_perc', axis = 1)
# Return and characteristics
desc_table(
    data = df[df.green_score.notna()],
    cols = ['ret_exc_lead1m', 'log_size', 'log_assets', 'log_be', 'debt_at', 'cash_at', 'netdebt_at', 'ebit_at', 'beta_252d'],
    cols_names = ['Exc. Ret. (leading)', 'LOG(ME)', 'LOG(Assets)', 'LOG(BE)', 'Debt/Assets', 'Cash/Assets', 'NetDebt/Assets', 'EBIT/Assets', 'Beta (252D)'],
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "char_descriptives.tex"
)
# ICC and valuation ratios
desc_table(
    data = df[df.green_score.notna()],
    cols = ['icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'ep_fwd0', 'ep_fwd1', 'ep_fwd2', 'ebitda_mev', 'be_me', 'bev_mev'],
    cols_names = ['ICC', 'ICC (GLS)', 'ICC (CT)', 'ICC (OJ)', 'ICC (PEG)', 'E/P (latest)', 'E/P (FY+1)', 'E/P (FY+2)', 'EBITDA/MEV', 'B/M', 'BEV/MEV'],
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "other_descriptives.tex"
)
# Options data
desc_table(
    data = df[df.green_score.notna()],
    cols = ['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182'],
    cols_names = ['SVIX (30D)', 'SVIX (91D)', 'SVIX (182D)', 'GLB (30D)', 'GLB (91D)', 'GLB (182D)'],
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "option_descriptives.tex"
)
# Bond data
desc_table(
    data = df_bonds[(df_bonds.log_size.notna()) & (df_bonds.green_score.notna())],
    cols = ['tmt', 'coupon', 'rating_num', 'yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'market_leverage'],
    cols_names = ['Time to maturity', 'Coupon rate', 'Rating (numeric)', 'Yield', 'Yield spread', 'Yield (adj.)', 'Yield spread (adj.)', 'Market leverage'],
    save_as = OUTPUT_TABLES_DESCRIPTIVES / "bond_descriptives.tex"
)
def descriptives_other():
    # ESG measure correlation - all variables separately with each other and then green_score at the top
    esg_corr_pearson = df[(df.date>=global_start) & (df.trucost_avg.notna()) & (df.msci_avg.notna()) & (df.sust_avg.notna()) & (df.green_score.notna())].groupby('date')[['green_score', 'trucost_avg', 'msci_avg', 'sust_avg']].corr(method='pearson').unstack(level=1).mean().unstack(level=0).astype(float).round(decimals=4)*100
    esg_corr_pearson.index = ['Agg. Green Score', 'Trucost', 'MSCI', 'Sustainalytics']
    esg_corr_pearson.columns = ['Agg. Green Score', 'Trucost', 'MSCI', 'Sustainalytics']
    column_format = 'l' + 'c' * (len(esg_corr_pearson.columns))  # 'l' for left, 'c' for center (centering number columns in output)
    esg_corr_pearson.to_latex(OUTPUT_TABLES_DESCRIPTIVES / "esg_corr_pearson.tex", float_format="%.1f", column_format = column_format)

    # adding country-level counts
    df_count = df[df.green_score.notna()].drop_duplicates(['gvkey']).groupby(['excntry'])['green_score'].count().reset_index()
    df_count = df_count.rename(columns={'green_score': 'N'})
    df_count = df_count[df_count.N >= 10]  # requiring at least 10 obs to show

    country_info = df[['excntry', 'Region', 'Continent', 'Country']].drop_duplicates().dropna()
    df_count = df_count.merge(country_info, how='left', on=['excntry'])
    df_count.columns = ['Exchange Country', '$N$', 'Region/Area', 'Continent', 'Country']
    # exporting to latex
    df_count.to_latex(OUTPUT_TABLES_DESCRIPTIVES / "firm_country_count.tex", index=False)

    return None
descriptives_other()

plt.close("all")
# descriptive plots with number of observations of key variables
def des_plots():
    # US sample: ESG measures
    plt.plot(df[df.excntry=="USA"].groupby('date')['green_score'].count(), label='Agg. Green Score', color='tab:green')
    plt.plot(df[df.excntry=="USA"].groupby('date')['weighted_esg_score'].count(), label='MSCI', color='tab:blue')
    plt.plot(df[df.excntry=="USA"].groupby('date')['total_esg_score'].count(), label='Sustainalytics', color='tab:orange')
    plt.plot(df[df.excntry=="USA"].groupby('date')['scope1int_sales'].count(), label='Trucost', color='tab:purple')
    #plt.plot(df[df.excntry=="USA"].groupby('date')['prod_waste_total_sales'].count(), label="EPA TRI", color='tab:red')
    plt.legend(loc='upper left')
    plt.xlim(pd.to_datetime('2007-01-01'), pd.to_datetime('2022-12-31'))
    plt.ylabel("Observations")
    plt.title('US Sample Coverage')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DESCRIPTIVES / "us_sample_nfirms.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # Global sample: ESG measures
    plt.plot(df[df.excntry!="USA"].groupby('date')['green_score'].count(), label='Aggregate Green Score', color='tab:green')
    plt.plot(df[df.excntry!="USA"].groupby('date')['weighted_esg_score'].count(), label='MSCI', color='tab:blue')
    plt.plot(df[df.excntry!="USA"].groupby('date')['total_esg_score'].count(), label='Sustainalytics', color='tab:orange')
    plt.plot(df[df.excntry!="USA"].groupby('date')['scope1int_sales'].count(), label='Trucost', color='tab:purple')
    plt.xlim(pd.to_datetime('2007-01-01'), pd.to_datetime('2022-12-31'))
    plt.title('Global Sample Coverage')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DESCRIPTIVES / "global_sample_nfirms.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # US sample: key dependent variables
    plt.plot(df[(df.excntry=="USA") & (df.green_score.notna())].groupby('date')['icc_avg'].count(), label='ICC (AVG)', color='tab:olive')
    plt.plot(df[(df.excntry=="USA") & (df.green_score.notna())].groupby('date')['ep_fwd1'].count(), label='E/P (FY+1)', color='tab:cyan')
    plt.plot(df[(df.excntry=="USA") & (df.green_score.notna())].groupby('date')['be_me'].count(), label='B/M', color='tab:brown')
    plt.legend()
    plt.xlim(pd.to_datetime('2007-01-01'), pd.to_datetime('2022-12-31'))
    plt.ylabel("Observations")
    plt.title('US Sample with Aggregate Green Score')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DESCRIPTIVES / "us_sample_nfirms_measures.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # Global sample:  sample: key dependent variables
    plt.plot(df[(df.excntry!="USA") & (df.green_score.notna())].groupby('date')['icc_avg'].count(), label='ICC (AVG)', color='tab:olive')
    plt.plot(df[(df.excntry!="USA") & (df.green_score.notna())].groupby('date')['ep_fwd1'].count(), label='E/P (FY+1)', color='tab:cyan')
    plt.plot(df[(df.excntry!="USA") & (df.green_score.notna())].groupby('date')['be_me'].count(), label='B/M', color='tab:brown')
    plt.xlim(pd.to_datetime('2007-01-01'), pd.to_datetime('2022-12-31'))
    plt.title('Global Sample with Aggregate Green Score')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DESCRIPTIVES / "global_sample_nfirms_measures.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
des_plots()

# US portfolio sorts using terciles (using full TS) - FF49 instead of GIC6 sort because only available since 2000
print('Running portfolio sorts...')
def portfolio_sorts(df, factor_test):
    df_ret_vw = pd.DataFrame(index=df_ff6.index)
    df_ret_vwcap = pd.DataFrame(index=df_ff6.index)
    df_ret_vwcap_indn = pd.DataFrame(index=df_ff6.index)

    # Looping over ESG variables to test
    for i in factor_test:
        temp_data = df[df[i].notna()]
        temp_data = temp_data[temp_data.ret_exc_lead1m.notna()]

        # Requiring minimum number of observations but otherwise using full time series available
        number_obs = temp_data.groupby(['date'], observed=False)[i].count().reset_index()
        number_obs = number_obs[number_obs[i] >= 100]  # at least 100 observations
        temp_data = temp_data[temp_data.date.isin(number_obs.date)]

        # vw
        temp_data['esg_rank'] = temp_data.groupby(['date'], observed=False)[i].rank(pct=True)
        temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
            lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

        # Exclude grouping columns when applying the function
        gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me']].apply(
            lambda x: (x['ret_exc_lead1m'] * x['me']).sum() / x['me'].sum()).unstack(level=1)

        gmb_factor = gmb_factor[3] - gmb_factor[1]

        # Off-setting index by 1 month due to returns being 1 month forward
        gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
        df_ret_vw = df_ret_vw.merge(gmb_factor.rename(i), how='left', left_index=True, right_index=True)

        # vw cap
        gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
            lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
        gmb_factor = gmb_factor[3] - gmb_factor[1]
        gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
        df_ret_vwcap = df_ret_vwcap.merge(gmb_factor.rename(i), how='left', left_index=True, right_index=True)

        # vw cap within industries
        temp_data['esg_rank'] = temp_data.groupby(['date', 'ff49'], observed=False)[i].rank(pct=True)
        temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
            lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

        gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
            lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
        gmb_factor = gmb_factor[3] - gmb_factor[1]
        gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
        df_ret_vwcap_indn = df_ret_vwcap_indn.merge(gmb_factor.rename(i), how='left', left_index=True, right_index=True)

    return df_ret_vw, df_ret_vwcap, df_ret_vwcap_indn
df_ret_vw, df_ret_vwcap, df_ret_vwcap_indn = portfolio_sorts(df[(df.ff49.notna()) & (df.excntry == "USA")], to_test + ['green_score'])

# US: regressing onto AP factors - can just have one function and call for each of the three dataframes? No need to repeat. Same above Also rename output
def portfolio_sorts_regs():
    out_vw = pd.DataFrame(
        columns=['SR', 'exc_ret', 't_exc_ret', 'p_exc_ret', 'a_capm', 't_a_capm', 'p_a_capm', 'a_ff3', 't_a_ff3',
                 'p_a_ff3', 'a_ff6', 't_a_ff6', 'p_a_ff6', 'a_q5', 't_a_q5', 'p_a_q5'], index=['green_score']+to_test)
    out_vwcap = pd.DataFrame(
        columns=['SR', 'exc_ret', 't_exc_ret', 'p_exc_ret', 'a_capm', 't_a_capm', 'p_a_capm', 'a_ff3', 't_a_ff3',
                 'p_a_ff3', 'a_ff6', 't_a_ff6', 'p_a_ff6', 'a_q5', 't_a_q5', 'p_a_q5'], index=['green_score']+to_test)
    out_vwcap_indn = pd.DataFrame(
        columns=['SR', 'exc_ret', 't_exc_ret', 'p_exc_ret', 'a_capm', 't_a_capm', 'p_a_capm', 'a_ff3', 't_a_ff3',
                 'p_a_ff3', 'a_ff6', 't_a_ff6', 'p_a_ff6', 'a_q5', 't_a_q5', 'p_a_q5'], index=['green_score']+to_test)

    # straight vw
    for i in out_vw.index:
        Y = df_ret_vw[i].dropna()
        constant_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])['const']).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        capm_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])).fit(cov_type='HAC',
                                                                                                    cov_kwds={
                                                                                                        'maxlags': 3})
        ff3_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)][['Mkt-RF', 'SMB', 'HML']])).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        ff6_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                         cov_kwds={'maxlags': 3})
        q5_results = sm.OLS(Y, sm.add_constant(df_q[df_q.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                    cov_kwds={'maxlags': 3})

        out_vw.loc[i, 'exc_ret'] = constant_results.params.iloc[0]
        out_vw.loc[i, 't_exc_ret'] = constant_results.tvalues.iloc[0]
        out_vw.loc[i, 'p_exc_ret'] = constant_results.pvalues.iloc[0]
        out_vw.loc[i, 'SR'] = (Y.mean() / Y.std()) * 12 ** 0.5
        out_vw.loc[i, 'a_capm'] = capm_results.params.iloc[0]
        out_vw.loc[i, 't_a_capm'] = capm_results.tvalues.iloc[0]
        out_vw.loc[i, 'p_a_capm'] = capm_results.pvalues.iloc[0]
        out_vw.loc[i, 'a_ff3'] = ff3_results.params.iloc[0]
        out_vw.loc[i, 't_a_ff3'] = ff3_results.tvalues.iloc[0]
        out_vw.loc[i, 'p_a_ff3'] = ff3_results.pvalues.iloc[0]
        out_vw.loc[i, 'a_ff6'] = ff6_results.params.iloc[0]
        out_vw.loc[i, 't_a_ff6'] = ff6_results.tvalues.iloc[0]
        out_vw.loc[i, 'p_a_ff6'] = ff6_results.pvalues.iloc[0]
        out_vw.loc[i, 'a_q5'] = q5_results.params.iloc[0]
        out_vw.loc[i, 't_a_q5'] = q5_results.tvalues.iloc[0]
        out_vw.loc[i, 'p_a_q5'] = q5_results.pvalues.iloc[0]

    # vw capped
    for i in out_vwcap.index:
        Y = df_ret_vwcap[i].dropna()
        constant_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])['const']).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        capm_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])).fit(cov_type='HAC',
                                                                                                    cov_kwds={
                                                                                                        'maxlags': 3})
        ff3_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)][['Mkt-RF', 'SMB', 'HML']])).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        ff6_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                         cov_kwds={'maxlags': 3})
        q5_results = sm.OLS(Y, sm.add_constant(df_q[df_q.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                    cov_kwds={'maxlags': 3})

        out_vwcap.loc[i, 'exc_ret'] = constant_results.params.iloc[0]
        out_vwcap.loc[i, 't_exc_ret'] = constant_results.tvalues.iloc[0]
        out_vwcap.loc[i, 'p_exc_ret'] = constant_results.pvalues.iloc[0]
        out_vwcap.loc[i, 'SR'] = (Y.mean() / Y.std()) * 12 ** 0.5
        out_vwcap.loc[i, 'a_capm'] = capm_results.params.iloc[0]
        out_vwcap.loc[i, 't_a_capm'] = capm_results.tvalues.iloc[0]
        out_vwcap.loc[i, 'p_a_capm'] = capm_results.pvalues.iloc[0]
        out_vwcap.loc[i, 'a_ff3'] = ff3_results.params.iloc[0]
        out_vwcap.loc[i, 't_a_ff3'] = ff3_results.tvalues.iloc[0]
        out_vwcap.loc[i, 'p_a_ff3'] = ff3_results.pvalues.iloc[0]
        out_vwcap.loc[i, 'a_ff6'] = ff6_results.params.iloc[0]
        out_vwcap.loc[i, 't_a_ff6'] = ff6_results.tvalues.iloc[0]
        out_vwcap.loc[i, 'p_a_ff6'] = ff6_results.pvalues.iloc[0]
        out_vwcap.loc[i, 'a_q5'] = q5_results.params.iloc[0]
        out_vwcap.loc[i, 't_a_q5'] = q5_results.tvalues.iloc[0]
        out_vwcap.loc[i, 'p_a_q5'] = q5_results.pvalues.iloc[0]

    # vw industry neutral
    for i in out_vwcap_indn.index:
        Y = df_ret_vwcap_indn[i].dropna()
        constant_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])['const']).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        capm_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)]['Mkt-RF'])).fit(cov_type='HAC',
                                                                                                    cov_kwds={
                                                                                                        'maxlags': 3})
        ff3_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)][['Mkt-RF', 'SMB', 'HML']])).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3})
        ff6_results = sm.OLS(Y, sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                         cov_kwds={'maxlags': 3})
        q5_results = sm.OLS(Y, sm.add_constant(df_q[df_q.index.isin(Y.index)])).fit(cov_type='HAC',
                                                                                    cov_kwds={'maxlags': 3})

        out_vwcap_indn.loc[i, 'exc_ret'] = constant_results.params.iloc[0]
        out_vwcap_indn.loc[i, 't_exc_ret'] = constant_results.tvalues.iloc[0]
        out_vwcap_indn.loc[i, 'p_exc_ret'] = constant_results.pvalues.iloc[0]
        out_vwcap_indn.loc[i, 'SR'] = (Y.mean() / Y.std()) * 12 ** 0.5
        out_vwcap_indn.loc[i, 'a_capm'] = capm_results.params.iloc[0]
        out_vwcap_indn.loc[i, 't_a_capm'] = capm_results.tvalues.iloc[0]
        out_vwcap_indn.loc[i, 'p_a_capm'] = capm_results.pvalues.iloc[0]
        out_vwcap_indn.loc[i, 'a_ff3'] = ff3_results.params.iloc[0]
        out_vwcap_indn.loc[i, 't_a_ff3'] = ff3_results.tvalues.iloc[0]
        out_vwcap_indn.loc[i, 'p_a_ff3'] = ff3_results.pvalues.iloc[0]
        out_vwcap_indn.loc[i, 'a_ff6'] = ff6_results.params.iloc[0]
        out_vwcap_indn.loc[i, 't_a_ff6'] = ff6_results.tvalues.iloc[0]
        out_vwcap_indn.loc[i, 'p_a_ff6'] = ff6_results.pvalues.iloc[0]
        out_vwcap_indn.loc[i, 'a_q5'] = q5_results.params.iloc[0]
        out_vwcap_indn.loc[i, 't_a_q5'] = q5_results.tvalues.iloc[0]
        out_vwcap_indn.loc[i, 'p_a_q5'] = q5_results.pvalues.iloc[0]

    out_vw = out_vw.astype(float)
    out_vwcap = out_vwcap.astype(float)
    out_vwcap_indn = out_vwcap_indn.astype(float)

    # saving to latex, excluding green score returns
    ret_vw = out_vw[1:][['SR', 'exc_ret', 't_exc_ret', 'a_capm', 't_a_capm',
       'a_ff3', 't_a_ff3', 'a_ff6', 't_a_ff6',
       'a_q5', 't_a_q5']].copy()
    ret_vwcap = out_vwcap[1:][['SR', 'exc_ret', 't_exc_ret', 'a_capm', 't_a_capm',
       'a_ff3', 't_a_ff3', 'a_ff6', 't_a_ff6',
       'a_q5', 't_a_q5']].copy()
    ret_vwcap_indn = out_vwcap_indn[1:][['SR', 'exc_ret', 't_exc_ret', 'a_capm', 't_a_capm',
       'a_ff3', 't_a_ff3', 'a_ff6', 't_a_ff6',
       'a_q5', 't_a_q5']].copy()

    ret_vw.index = to_test_names
    ret_vwcap.index = to_test_names
    ret_vwcap_indn.index = to_test_names

    # reorder rows
    # new_order = ['S1INT (Sales)', 'S1+2INT (Sales)', 'S1+2+3INT (Sales)', 'S1INT (Assets)', 'S1+2INT (Assets)', 'S1+2+3INT (Assets)',
    #              'Weighted ESG score', 'Environment score', 'Total ESG score', 'Environmental score',
    #              'LOG(S1TOT)', 'LOG(S1+2TOT)', 'LOG(S1+2+3TOT)', 'Ind.-adj. ESG score',
    #              'Greenness (PST)', 'E climate score', 'E nat. res. score', 'E waste score', 'E env. opps. score',
    #              'TRI (Sales)', 'TPWINT (Sales)', 'TRI (Assets)', 'TPWINT (Assets)']  # Desired order of index names
    # ret_vw = ret_vw.reindex(new_order)
    # ret_vwcap = ret_vwcap.reindex(new_order)
    # ret_vwcap_indn = ret_vwcap_indn.reindex(new_order)

    # column names for export
    column_names = [
        r"$SR$",
        r"$r$",
        r"$t(r)$",
        r"$\alpha^{CAPM}$",
        r"$t(\alpha^{CAPM})$",
        r"$\alpha^{FF3}$",
        r"$t(\alpha^{FF3})$",
        r"$\alpha^{FF6}$",
        r"$t(\alpha^{FF6})$",
        r"$\alpha^{q5}$",
        r"$t(\alpha^{q5})$"
    ]

    ret_vw.columns = column_names
    ret_vwcap.columns = column_names
    ret_vwcap_indn.columns = column_names

    ret_vw.to_latex(OUTPUT_TABLES_US / "pf_vw.tex", float_format="%.2f", column_format='lccccccccccc')
    ret_vwcap.to_latex(OUTPUT_TABLES_US / "pf_vwcap.tex", float_format="%.2f", column_format='lccccccccccc')
    ret_vwcap_indn.to_latex(OUTPUT_TABLES_US / "pf_vwcap_indn.tex", float_format="%.2f", column_format='lccccccccccc')

    return out_vw, out_vwcap, out_vwcap_indn
ret_vw, ret_vwcap, ret_vwcap_indn = portfolio_sorts_regs()

# Global, country-level results (min 10 firms each period, min 5 years data, using JKP vw cap factors)
def portfolio_sorts_global(df_input, esg_var):
    country_list = df_input['excntry'].drop_duplicates().sort_values()
    country_list = country_list[country_list != 'KEN']  # dropping Kenya, obvious huge outliers in returns
    ret_vwcap = []
    ret_vwcap_indn = []

    # VW cap
    for e in esg_var: # looping over chosen ESG variables
        temp_data = df_input[df_input[e].notna()]
        temp_data = temp_data[temp_data.ret_exc_lead1m.notna()]
        for c in country_list: # looping over each country
            temp_data_country = temp_data[temp_data.excntry == c]
            # requiring minimum 10 firms in each country-month pair
            number_obs = temp_data_country.groupby(['date'], observed=False)[e].count().reset_index()
            number_obs = number_obs[number_obs[e] >= 10]

            if len(number_obs) >= 60: # requiring time-series length of at least 5 years
                temp_data_country = temp_data_country[temp_data_country.date.isin(number_obs.date)]

                # rank-sorting based on ESG characteristic and then creating terciles
                temp_data_country['esg_rank'] = temp_data_country.groupby(['date'], observed=False)[e].rank(pct=True)
                temp_data_country['tercile_esg'] = temp_data_country.groupby(['date'], observed=False)['esg_rank'].transform(
                    lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

                # vw cap factor portfolio
                gmb_factor = temp_data_country.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
                    lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
                gmb_factor = gmb_factor.dropna()

                if len(gmb_factor) >= 60:  # require at least 5 years data
                    gmb_factor = gmb_factor[3] - gmb_factor[1]
                    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
                    gmb_factor = gmb_factor.dropna()
                    gmb_factor = gmb_factor.rename('factor_ret').reset_index()
                    gmb_factor['excntry'] = c
                    gmb_factor['const'] = 1
                    gmb_factor = gmb_factor.merge(df_gfactors, how='left', on=['date', 'excntry'])
                    gmb_factor = gmb_factor.dropna()

                    # obtaining factor results using JKP factors
                    Y = gmb_factor['factor_ret']
                    constant_results = sm.OLS(Y, gmb_factor['const']).fit(cov_type='HAC', cov_kwds={'maxlags': 3}) # Newey-West 3 lags
                    capm_results = sm.OLS(Y, gmb_factor[['const', 'mkt_vw_exc']]).fit(cov_type='HAC',
                                                                                      cov_kwds={'maxlags': 3})
                    ff3_results = sm.OLS(Y, gmb_factor[['const', 'mkt_vw_exc', 'size', 'value']]).fit(cov_type='HAC',
                                                                                                      cov_kwds={
                                                                                                          'maxlags': 3})
                    ff6_results = sm.OLS(Y, gmb_factor[
                        ['const', 'mkt_vw_exc', 'size', 'value', 'investment', 'profitability', 'momentum']]).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 3})

                    # saving output
                    ret_vwcap.append([e, c,
                                      constant_results.params.iloc[0], constant_results.tvalues.iloc[0], constant_results.pvalues.iloc[0],
                                      (Y.mean() / Y.std()) * 12 ** 0.5,
                                      capm_results.params.iloc[0], capm_results.tvalues.iloc[0], capm_results.pvalues.iloc[0],
                                      ff3_results.params.iloc[0], ff3_results.tvalues.iloc[0], ff3_results.pvalues.iloc[0],
                                      ff6_results.params.iloc[0], ff6_results.tvalues.iloc[0], ff6_results.pvalues.iloc[0]])

                # repeating for vw cap within industry
                temp_data_country['esg_rank'] = temp_data_country.groupby(['date', 'ff49'], observed=False)[e].rank(pct=True)
                temp_data_country['tercile_esg'] = temp_data_country.groupby(['date'], observed=False)['esg_rank'].transform(
                    lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

                # vw cap industry-neutral
                gmb_factor = temp_data_country.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
                    lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
                gmb_factor = gmb_factor.dropna()

                if len(gmb_factor) >= 60:  # require at least 5 years data with 10 obs in each month
                    gmb_factor = gmb_factor[3] - gmb_factor[1]
                    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
                    gmb_factor = gmb_factor.dropna()
                    gmb_factor = gmb_factor.rename('factor_ret').reset_index()
                    gmb_factor['excntry'] = c
                    gmb_factor['const'] = 1
                    gmb_factor = gmb_factor.merge(df_gfactors, how='left', on=['date', 'excntry'])
                    gmb_factor = gmb_factor.dropna()

                    # obtaining factor results using JKP factors
                    Y = gmb_factor['factor_ret']
                    constant_results = sm.OLS(Y, gmb_factor['const']).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                    capm_results = sm.OLS(Y, gmb_factor[['const', 'mkt_vw_exc']]).fit(cov_type='HAC',
                                                                                      cov_kwds={'maxlags': 3})
                    ff3_results = sm.OLS(Y, gmb_factor[['const', 'mkt_vw_exc', 'size', 'value']]).fit(cov_type='HAC',
                                                                                                      cov_kwds={
                                                                                                          'maxlags': 3})
                    ff6_results = sm.OLS(Y, gmb_factor[
                        ['const', 'mkt_vw_exc', 'size', 'value', 'investment', 'profitability', 'momentum']]).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 3})

                    # saving output
                    ret_vwcap_indn.append([e, c,
                                      constant_results.params.iloc[0], constant_results.tvalues.iloc[0], constant_results.pvalues.iloc[0],
                                      (Y.mean() / Y.std()) * 12 ** 0.5,
                                      capm_results.params.iloc[0], capm_results.tvalues.iloc[0], capm_results.pvalues.iloc[0],
                                      ff3_results.params.iloc[0], ff3_results.tvalues.iloc[0], ff3_results.pvalues.iloc[0],
                                      ff6_results.params.iloc[0], ff6_results.tvalues.iloc[0], ff6_results.pvalues.iloc[0]])

    # formatting output
    ret_vwcap = pd.DataFrame(ret_vwcap)
    ret_vwcap.columns = ['esg', 'excntry', 'exc_ret', 't_exc_ret', 'p_exc_ret', 'SR', 'a_capm', 't_a_capm', 'p_a_capm',
                         'a_ff3', 't_a_ff3', 'p_a_ff3', 'a_ff6', 't_a_ff6', 'p_a_ff6']

    ret_vwcap_indn = pd.DataFrame(ret_vwcap_indn)
    ret_vwcap_indn.columns = ['esg', 'excntry', 'exc_ret', 't_exc_ret', 'p_exc_ret', 'SR', 'a_capm', 't_a_capm', 'p_a_capm',
                         'a_ff3', 't_a_ff3', 'p_a_ff3', 'a_ff6', 't_a_ff6', 'p_a_ff6']

    return ret_vwcap, ret_vwcap_indn
ret_vwcap_g, ret_vwcap_indn_g = portfolio_sorts_global(df[(df.ff49.notna()) & (df.excntry!="USA")], to_test_g)

# Country-by-country green score factors
def portfolio_sorts_country(df):
    df_ret_vwcap = pd.DataFrame(index=df_ff6.index)
    df_ret_vwcap_indn = pd.DataFrame(index=df_ff6.index)
    df_ret_vwcap = df_ret_vwcap[df_ret_vwcap.index > '2009-08-31']
    df_ret_vwcap_indn = df_ret_vwcap_indn[df_ret_vwcap_indn.index > '2009-08-31']

    df = df[df['green_score'].notna()]
    df = df[df.ret_exc_lead1m.notna()]

    # Looping over country variables except Kenya
    for i in df[df.excntry!="KEN"].excntry.drop_duplicates():
        try:
            temp_data = df[df.excntry==i]
            # Requiring minimum number of observations but otherwise using full time series available
            number_obs = temp_data.groupby(['date'], observed=False)['green_score'].count().reset_index()
            number_obs = number_obs[number_obs['green_score'] >= 10]  # at least 10 observations
            temp_data = temp_data[temp_data.date.isin(number_obs.date)]

            # vw cap
            temp_data['esg_rank'] = temp_data.groupby(['date'], observed=False)['green_score'].rank(pct=True)
            temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
                lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))
            gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
                lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
            gmb_factor = gmb_factor[3] - gmb_factor[1]
            gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
            df_ret_vwcap = df_ret_vwcap.merge(gmb_factor.rename(i), how='left', left_index=True, right_index=True)

            # vw cap within industries
            temp_data['esg_rank'] = temp_data.groupby(['date', 'ff49'], observed=False)['green_score'].rank(pct=True)
            temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
                lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))
            gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
                lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
            gmb_factor = gmb_factor[3] - gmb_factor[1]
            gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
            df_ret_vwcap_indn = df_ret_vwcap_indn.merge(gmb_factor.rename(i), how='left', left_index=True, right_index=True)

        except:
            continue
    return df_ret_vwcap, df_ret_vwcap_indn
df_ret_vwcap_country, df_ret_vwcap_indn_country = portfolio_sorts_country(df[(df.ff49.notna())])

# exporting country-by-country green score factors
df_ret_vwcap_country.to_parquet(DATA_PROCESSED / "gs_factor_country_vwcap.parquet")
df_ret_vwcap_indn_country.to_parquet(DATA_PROCESSED / "gs_factor_country_vwcap_indn.parquet")

global_temp_1 = ret_vw.reset_index().copy()
global_temp_1 = global_temp_1.rename(columns={'index': 'esg'})
global_temp_1['ind_neut'] = 0
global_temp_1['excntry'] = "USA"

global_temp_2 = ret_vw.reset_index().copy()
global_temp_2 = global_temp_1.rename(columns={'index': 'esg'})
global_temp_2['ind_neut'] = 1
global_temp_2['excntry'] = "USA"

global_temp_3 = ret_vwcap_g.copy()
global_temp_3['ind_neut'] = 0

global_temp_4 = ret_vwcap_indn_g.copy()
global_temp_4['ind_neut'] = 1

df_temp = pd.concat([global_temp_1, global_temp_2, global_temp_3, global_temp_4], axis=0)

# 1. Drop the irrelevant columns
df_temp = df_temp.drop(columns=['SR', 'exc_ret', 'a_capm', 'a_ff3', 'a_ff6', 'a_q5'])

# 2. Separate the `t_` and `p_` columns, keeping `esg` and `excntry` as id_vars
df_t = df_temp.melt(id_vars=['esg', 'excntry'], value_vars=[col for col in df_temp.columns if col.startswith('t_')],
               var_name='metric', value_name='t_stat')
df_p = df_temp.melt(id_vars=['esg', 'excntry'], value_vars=[col for col in df_temp.columns if col.startswith('p_')],
               var_name='metric', value_name='p_value')

# 3. Clean up the column names to only contain the metric name (remove 't_' and 'p_')
df_t['metric'] = df_t['metric'].str.replace(r'^t_', '', regex=True)
df_p['metric'] = df_p['metric'].str.replace(r'^p_', '', regex=True)

# 4. Merge the two dataframes on 'esg', 'excntry', and 'metric' to get both t_stat and p_value in the same row
df_temp = pd.merge(df_t, df_p, on=['esg', 'excntry', 'metric'])

# Save US and Global-ex factor time series
df_ret_vwcap.to_parquet(DATA_PROCESSED / "factor_us_vwcap_ts.parquet")
df_ret_vwcap_indn.to_parquet(DATA_PROCESSED / "factor_us_vwcap_indn_ts.parquet")
ret_vwcap_g.to_parquet(DATA_PROCESSED / "factor_globalexus_vwcap_ts.parquet")
ret_vwcap_indn_g.to_parquet(DATA_PROCESSED / "factor_globalexus_vwcap_indn_ts.parquet")

# Exporting basic data (US and Global ex-US separately), for use in R
df.to_parquet(DATA_PROCESSED / "df_full.parquet")
df[(df.green_score.notna()) & (df.excntry=="USA")][['date', 'id', 'excntry', 'me', 'ret_exc_lead1m', 'gics', 'icc_avg', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be', 'green_score']+total_em_factors+emission_factors+msci_factors+sust_factors+tri_factors].to_csv(DATA_PROCESSED / "df_us.csv")
df[(df.green_score.notna()) & (df.excntry!="USA")][['date', 'id', 'excntry', 'me', 'ret_exc_lead1m', 'gics', 'icc_avg', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be', 'green_score']+total_em_factors+emission_factors+msci_factors+sust_factors].to_csv(DATA_PROCESSED / "df_gexus.csv")


# importing published t-stats
df_pub = pd.read_excel(DATA_RAW / "other" / "published_tstats.xlsx") #, usecols=['t-stat', 'N']
df_pub = df_pub.rename(columns={'t-stat': 't_stat'})
df_pub['p_val'] = 2 * (1 - t.cdf(abs(df_pub.t_stat), df_pub.N-1)) #converting t-stats to p-values using N-1 DoF
#df_pub = df_pub.drop('N', axis=1)

# df_pub.to_csv(DATA_INTERIM / "published_factors.csv")

# combining industry-agnostic and -neutral information
us_temp_alpha = pd.concat([ret_vwcap.loc[to_test], ret_vwcap_indn.loc[to_test]], axis=0)
us_tstat = us_temp_alpha.loc[to_test][['t_exc_ret', 't_a_capm', 't_a_ff3', 't_a_ff6', 't_a_q5']].melt()['value'].sort_values(ascending=True)
us_pval = us_temp_alpha.loc[to_test][['p_exc_ret', 'p_a_capm', 'p_a_ff3', 'p_a_ff6', 'p_a_q5']].melt()['value'].sort_values(ascending=True)

global_temp_alpha = pd.concat([ret_vwcap_g, ret_vwcap_indn_g], axis=0)
global_tstat = global_temp_alpha[['t_exc_ret', 't_a_capm', 't_a_ff3', 't_a_ff6']].melt()['value'].sort_values(ascending=True)
global_pval = global_temp_alpha[['p_exc_ret', 'p_a_capm', 'p_a_ff3', 'p_a_ff6']].melt()['value'].sort_values(ascending=True)

all_temp_alpha = pd.concat([us_temp_alpha, global_temp_alpha], axis=0)
all_tstat = all_temp_alpha[['t_exc_ret', 't_a_capm', 't_a_ff3', 't_a_ff6', 't_a_q5']].melt()['value'].dropna().sort_values(ascending=True)
all_pval = all_temp_alpha[['p_exc_ret', 'p_a_capm', 'p_a_ff3', 'p_a_ff6', 'p_a_q5']].melt()['value'].dropna().sort_values(ascending=True)

#all_pub_tstat = pd.concat([df_pub['t_stat'], all_tstat], axis=0).sort_values(ascending=True)
all_pub_pval = pd.concat([df_pub['p_val'], all_pval], axis=0).sort_values(ascending=True)


plt.close("all")
# histograms of t-stats
def realized_ret_hist():
    # US t-stat
    plt.hist(us_tstat, color='tab:green', alpha=0.8, bins=25, density=True)
    plt.xlim(-4.5, 4.5)
    plt.xlabel("$t$-statistic", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_us.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # Global ex-US t-stat
    plt.hist(global_tstat, color='tab:green', alpha=0.8, bins=25, density=True)
    plt.xlim(-4.5, 4.5)
    plt.xlabel("$t$-statistic", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_globalexus.pdf",
                format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # Global t-stat
    plt.hist(all_tstat, color='tab:green', alpha=0.8, bins=25, density=True)
    plt.xlim(-4.5, 4.5)
    plt.xlabel("$t$-statistic", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_global.pdf",
                format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    # Global t-stat
    plt.hist(df_pub.t_stat, color='tab:green', alpha=0.8, bins=25, density=True)
    plt.xlim(-5.5, 5.5)
    plt.xlabel("$t$-statistic", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_pub.pdf",
                format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    with mpl.rc_context(rc={'font.size': 14}):
        # Version 1: Published only
        plt.figure(figsize=(12, 6))
        plt.hist(df_pub.t_stat, color='tab:blue', alpha=0.8, bins=25, density=True, label='Published')
        plt.xlim(-5.5, 5.5)
        plt.ylim(0, 0.45)  # Fix y-axis so it doesn't jump between slides
        plt.xlabel("$t$-statistic", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_global_publ1.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        # Version 2: Both published and replication
        # Replication bars get a diagonal hatch pattern to distinguish them
        # from Published bars in print and to match the IA's published
        # styling. edgecolor='darkgreen' (not 'black') so hatch strokes are
        # greenish-dark rather than starkly black — matches the Overleaf
        # rendering much more closely than plain black.
        plt.figure(figsize=(12, 6))
        plt.hist(all_tstat, color='tab:green', alpha=0.8, bins=25, density=True,
                 label='Replication', hatch='//', edgecolor='darkgreen', linewidth=0)
        plt.hist(df_pub.t_stat, color='tab:blue', alpha=0.8, bins=25, density=True, label='Published')
        plt.xlim(-5.5, 5.5)
        plt.ylim(0, 0.45)  # Same y-axis as version 1
        plt.xlabel("$t$-statistic", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_GLOBAL / "tstat_global_publ2.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    plt.close("all")

    ########################### GLOBAL EX-US ###########################
    us_pval_bh = pd.DataFrame(us_pval)
    us_pval_bh.columns = ['p_value']
    alpha = 0.05
    m = len(us_pval_bh)  # Total number of tests
    us_pval_bh['rank'] = range(1, m + 1)  # Rank of p-values (1, 2, ..., m)
    us_pval_bh['critical_value'] = (us_pval_bh['rank'] / m) * alpha

    # Determine significant p-values
    us_pval_bh['significant'] = us_pval_bh['p_value'] <= us_pval_bh['critical_value']

    # Find the largest significant p-value rank
    max_significant = us_pval_bh[us_pval_bh['significant']]['rank'].max()
    if pd.notna(max_significant):
        us_pval_bh['final_decision'] = us_pval_bh['rank'] <= max_significant
    else:
        us_pval_bh['final_decision'] = False  # No significant results

    us_pval_bh['p_perc'] = us_pval_bh['rank'] / m

    plt.close("all")
    plt.scatter(us_pval_bh.p_perc, us_pval_bh.p_value, color='firebrick', s=4, label='observations', zorder=1)
    plt.plot(us_pval_bh.p_perc, us_pval_bh.p_perc, linestyle='dotted', color='blue', label='45-degree line',
             zorder=2)
    plt.plot(us_pval_bh.p_perc, us_pval_bh.critical_value, linestyle='dashed', color='black',
             label='B-H 5% threshold', zorder=3)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=-1)  # Customize grid appearance
    plt.legend()
    plt.ylabel("p-value", fontsize=14)
    plt.xlabel("p-value percentile", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "bh_us.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    ########################### GLOBAL EX-US ###########################
    global_pval_bh = pd.DataFrame(global_pval)
    global_pval_bh.columns = ['p_value']
    alpha = 0.05
    m = len(global_pval_bh)  # Total number of tests
    global_pval_bh['rank'] = range(1, m + 1)  # Rank of p-values (1, 2, ..., m)
    global_pval_bh['critical_value'] = (global_pval_bh['rank'] / m) * alpha

    # Determine significant p-values
    global_pval_bh['significant'] = global_pval_bh['p_value'] <= global_pval_bh['critical_value']

    # Find the largest significant p-value rank
    max_significant = global_pval_bh[global_pval_bh['significant']]['rank'].max()
    if pd.notna(max_significant):
        global_pval_bh['final_decision'] = global_pval_bh['rank'] <= max_significant
    else:
        global_pval_bh['final_decision'] = False  # No significant results

    global_pval_bh['p_perc'] = global_pval_bh['rank'] / m

    plt.close("all")
    plt.scatter(global_pval_bh.p_perc, global_pval_bh.p_value, color='firebrick', s=4, label='observations', zorder=1)
    plt.plot(global_pval_bh.p_perc, global_pval_bh.p_perc, linestyle='dotted', color='blue', label='45-degree line',
             zorder=2)
    plt.plot(global_pval_bh.p_perc, global_pval_bh.critical_value, linestyle='dashed', color='black',
             label='B-H 5% threshold', zorder=3)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=-1)  # Customize grid appearance
    plt.legend()
    plt.ylabel("p-value", fontsize=14)
    plt.xlabel("p-value percentile", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "bh_globalexus.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    ########################### GLOBAL ###########################
    all_pval_bh = pd.DataFrame(all_pval)
    all_pval_bh.columns = ['p_value']
    alpha = 0.05
    m = len(all_pval_bh)  # Total number of tests
    all_pval_bh['rank'] = range(1, m + 1)  # Rank of p-values (1, 2, ..., m)
    all_pval_bh['critical_value'] = (all_pval_bh['rank'] / m) * alpha

    # Determine significant p-values
    all_pval_bh['significant'] = all_pval_bh['p_value'] <= all_pval_bh['critical_value']

    # Find the largest significant p-value rank
    max_significant = all_pval_bh[all_pval_bh['significant']]['rank'].max()
    if pd.notna(max_significant):
        all_pval_bh['final_decision'] = all_pval_bh['rank'] <= max_significant
    else:
        all_pval_bh['final_decision'] = False  # No significant results

    all_pval_bh['p_perc'] = all_pval_bh['rank'] / m

    plt.close("all")
    plt.scatter(all_pval_bh.p_perc, all_pval_bh.p_value, color='firebrick', s=4, label='observations', zorder=1)
    plt.plot(all_pval_bh.p_perc, all_pval_bh.p_perc, linestyle='dotted', color='blue', label='45-degree line',
             zorder=2)
    plt.plot(all_pval_bh.p_perc, all_pval_bh.critical_value, linestyle='dashed', color='black',
             label='B-H 5% threshold', zorder=3)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=-1)  # Customize grid appearance
    plt.legend()
    plt.ylabel("p-value", fontsize=14)
    plt.xlabel("p-value percentile", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "bh_global.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    ########################### GLOBAL + PUBLISHED ###########################
    all_pub_pval_bh = pd.DataFrame(all_pub_pval)
    all_pub_pval_bh.columns = ['p_value']
    alpha = 0.05
    m = len(all_pub_pval_bh)  # Total number of tests
    all_pub_pval_bh['rank'] = range(1, m + 1)  # Rank of p-values (1, 2, ..., m)
    all_pub_pval_bh['critical_value'] = (all_pub_pval_bh['rank'] / m) * alpha

    # Determine significant p-values
    all_pub_pval_bh['significant'] = all_pub_pval_bh['p_value'] <= all_pub_pval_bh['critical_value']

    # Find the largest significant p-value rank
    max_significant = all_pub_pval_bh[all_pub_pval_bh['significant']]['rank'].max()
    if pd.notna(max_significant):
        all_pub_pval_bh['final_decision'] = all_pub_pval_bh['rank'] <= max_significant
    else:
        all_pub_pval_bh['final_decision'] = False  # No significant results

    all_pub_pval_bh['p_perc'] = all_pub_pval_bh['rank'] / m

    plt.close("all")
    # plt.plot(all_pub_pval_bh.p_perc, all_pub_pval_bh.p_value, marker='o', linestyle='none', color='blue', markersize=2, linewidth=0.5)
    plt.scatter(all_pub_pval_bh.p_perc, all_pub_pval_bh.p_value, color='firebrick', s=4, label='observations', zorder=1)
    plt.plot(all_pub_pval_bh.p_perc, all_pub_pval_bh.p_perc, linestyle='dotted', color='blue', label='45-degree line',
             zorder=2)
    plt.plot(all_pub_pval_bh.p_perc, all_pub_pval_bh.critical_value, linestyle='dashed', color='black',
             label='B-H 5% threshold', zorder=3)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=-1)  # Customize grid appearance
    plt.legend()
    plt.ylabel("p-value", fontsize=14)
    plt.xlabel("p-value percentile", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "bh_global_publ.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    ########################### PUBLISHED ###########################
    df_pub_bh = pd.DataFrame(df_pub['p_val'].sort_values(ascending=True))
    df_pub_bh.columns = ['p_value']
    alpha = 0.05
    m = len(df_pub_bh)  # Total number of tests
    df_pub_bh['rank'] = range(1, m + 1)  # Rank of p-values (1, 2, ..., m)
    df_pub_bh['critical_value'] = (df_pub_bh['rank'] / m) * alpha

    # Determine significant p-values
    df_pub_bh['significant'] = df_pub_bh['p_value'] <= df_pub_bh['critical_value']

    # Find the largest significant p-value rank
    max_significant = df_pub_bh[df_pub_bh['significant']]['rank'].max()
    if pd.notna(max_significant):
        df_pub_bh['final_decision'] = df_pub_bh['rank'] <= max_significant
    else:
        df_pub_bh['final_decision'] = False  # No significant results

    df_pub_bh['p_perc'] = df_pub_bh['rank'] / m

    plt.close("all")
    # plt.plot(df_pub_bh.p_perc, df_pub_bh.p_value, marker='o', linestyle='none', color='blue', markersize=2, linewidth=0.5)
    plt.scatter(df_pub_bh.p_perc, df_pub_bh.p_value, color='firebrick', s=4, label='observations', zorder=1)
    plt.plot(df_pub_bh.p_perc, df_pub_bh.p_perc, linestyle='dotted', color='blue', label='45-degree line',
             zorder=2)
    plt.plot(df_pub_bh.p_perc, df_pub_bh.critical_value, linestyle='dashed', color='black',
             label='B-H 5% threshold', zorder=3)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=-1)  # Customize grid appearance
    plt.legend()
    plt.ylabel("p-value", fontsize=14)
    plt.xlabel("p-value percentile", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "bh_publ.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
realized_ret_hist()

# Two-sample Kolmogorov-Smirnov test comparing published t-stats vs replicated t-stats
ks_stat, p_value = ks_2samp(df_pub['t_stat'], all_tstat)

# Compute ECDFs
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def ks_test():
    # Get ECDFs for both distributions
    x_rep, y_rep = ecdf(all_tstat)
    x_pub, y_pub = ecdf(df_pub['t_stat'])

    # Create figure
    with mpl.rc_context(rc={'font.size': 14}):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot ECDFs - using same colors as histogram
        ax.step(x_rep, y_rep, where='post', color='tab:green', linewidth=2, alpha=0.8, 
                linestyle='dashed', label=f'Replication (n={len(all_tstat):,})')
        ax.step(x_pub, y_pub, where='post', color='tab:blue', linewidth=2, alpha=0.8, 
                label=f'Published (n={len(df_pub)})')

        # Fill between the two CDFs to show difference region
        rep_interp = interp1d(x_rep, y_rep, kind='previous', bounds_error=False, fill_value=(0, 1))
        x_fill = np.sort(np.concatenate([x_rep, x_pub]))
        y_rep_fill = rep_interp(x_fill)
        pub_interp = interp1d(x_pub, y_pub, kind='previous', bounds_error=False, fill_value=(0, 1))
        y_pub_fill = pub_interp(x_fill)
        ax.fill_between(x_fill, y_rep_fill, y_pub_fill, alpha=0.3, color='tab:grey')

        # Add vertical lines at t = ±1.96
        ax.axvline(-1.96, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(1.96, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Find maximum difference point
        all_x = np.sort(np.concatenate([x_rep, x_pub]))
        rep_at_x = rep_interp(all_x)
        pub_at_x = pub_interp(all_x)
        diff = np.abs(rep_at_x - pub_at_x)
        max_idx = np.argmax(diff)
        x_max = all_x[max_idx]
        y_max_rep = rep_at_x[max_idx]
        y_max_pub = pub_at_x[max_idx]

        # Draw vertical line at max difference
        ax.plot([x_max, x_max], [y_max_rep, y_max_pub], color='black', linewidth=1.5)
        ax.scatter([x_max, x_max], [y_max_rep, y_max_pub], color='black', s=30, zorder=5)

        # Annotate max difference with text box
        ax.annotate(f'D = {ks_stat:.2f}', 
                    xy=(x_max, (y_max_rep + y_max_pub) / 2),
                    xytext=(x_max + 1.2, (y_max_rep + y_max_pub) / 2),
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

        # Add KS test results box with legend-like style
        textstr = f'Kolmogorov-Smirnov Test\nD = {ks_stat:.2f}\np < 0.001'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray'))

        # Labels and formatting
        ax.set_xlabel('$t$-statistic', fontsize=14)
        ax.set_ylabel('Cumulative probability', fontsize=14)
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=14, loc='lower right')

        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_GLOBAL / "ks_test_ecdf.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()
ks_test()



#exporting to csv
ret_vwcap.to_csv(DATA_PROCESSED / "factor_us_vwcap.csv")
ret_vwcap_indn.to_csv(DATA_PROCESSED / "factor_us_vwcap_indn.csv")
ret_vwcap_g.to_csv(DATA_PROCESSED / "factor_globalexus_vwcap.csv")
ret_vwcap_indn_g.to_csv(DATA_PROCESSED / "factor_globalexus_vwcap_indn.csv")



# multiple fe regression: a function to be called by regressions below
print('Running main greenium regressions...')
def fe_reg_function(df_input, dep_var, controls, industry_fe):
    df_out = pd.DataFrame(index=dep_var, columns=['Greenium', 'S.E. (Greenium)', 'Beta', 'S.E. (Beta)',
                                                  'LOG(BE)', 'S.E. (LOG(BE))', 'EBIT/AT', 'S.E. (EBIT/AT)',
                                                  'NETDEBT/AT', 'S.E. (NETDEBT/AT)', 'N', 'R2', 'R2 (Total)'])

    # data requirements: must have industry code, green score and control variables
    test_data = df_input[df_input.gics6.notna()]
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.beta_252d.notna()]
    test_data = test_data[test_data.log_be.notna()]
    test_data = test_data[test_data.ebit_at.notna()]
    test_data = test_data[test_data.netdebt_at.notna()]
    
    # setting index to firm-id and country-date (to use as FE input)
    test_data = test_data.set_index(['id', 'country_ym_num'])

    for j in dep_var:
        test_data[j] = test_data[j]*100 #scaling
        if (j == 'ret_exc_lead1m'):
            test_data[j] = test_data[j] * 12 #annualizing returns
        if (controls is None):
            X = test_data['green_score']
        else:
            X = pd.concat([test_data['green_score'], test_data[controls]], axis=1)
        X = sm.add_constant(X)

        if (industry_fe is None):  # only time FE
            reg = PanelOLS(test_data[j], X, entity_effects=False, time_effects=True,
                           other_effects=None, drop_absorbed=True, check_rank=False).fit(cov_type="clustered",
                                                                                             clusters=test_data[['date', 'gics']],
                                                                                             cluster_entity=False,
                                                                                             cluster_time=False)
        elif (industry_fe is not None):  # industry and time FE
            reg = PanelOLS(test_data[j], X, entity_effects=False, time_effects=False,
                           other_effects=test_data[industry_fe], drop_absorbed=True, check_rank=False).fit(cov_type="clustered",
                                                                                             clusters=test_data[['date', 'gics']],
                                                                                             cluster_entity=False,
                                                                                             cluster_time=False)
        df_out.loc[j, 'Greenium'] = reg.params.loc['green_score']
        df_out.loc[j, 'S.E. (Greenium)'] = reg.std_errors.loc['green_score']

        df_out.loc[j, 'N'] = reg.nobs
        df_out.loc[j, 'R2'] = reg.rsquared*100
        df_out.loc[j, 'R2 (Total)'] = reg.rsquared_inclusive*100

        # trying for other coefficients that might not be included
        try:
            df_out.loc[j, 'Beta'] = reg.params.loc['beta_252d']
        except KeyError:
            df_out.loc[j, 'Beta'] = np.nan

        try:
            df_out.loc[j, 'LOG(BE)'] = reg.params.loc['log_be']
        except KeyError:
            df_out.loc[j, 'LOG(BE)'] = np.nan

        try:
            df_out.loc[j, 'EBIT/AT'] = reg.params.loc['ebit_at']
        except KeyError:
            df_out.loc[j, 'EBIT/AT'] = np.nan

        try:
            df_out.loc[j, 'NETDEBT/AT'] = reg.params.loc['netdebt_at']
        except KeyError:
            df_out.loc[j, 'NETDEBT/AT'] = np.nan

        try:
            df_out.loc[j, 'S.E. (Beta)'] = reg.std_errors.loc['beta_252d']
        except KeyError:
            df_out.loc[j, 'S.E. (Beta)'] = np.nan

        try:
            df_out.loc[j, 'S.E. (LOG(BE))'] = reg.std_errors.loc['log_be']
        except KeyError:
            df_out.loc[j, 'S.E. (LOG(BE))'] = np.nan

        try:
            df_out.loc[j, 'S.E. (EBIT/AT)'] = reg.std_errors.loc['ebit_at']
        except KeyError:
            df_out.loc[j, 'S.E. (EBIT/AT)'] = np.nan

        try:
            df_out.loc[j, 'S.E. (NETDEBT/AT)'] = reg.std_errors.loc['netdebt_at']
        except KeyError:
            df_out.loc[j, 'S.E. (NETDEBT/AT)'] = np.nan

    df_out = df_out.astype(float)
    df_out['N'] = df_out['N'].astype(int)

    return df_out

# using above reg-function to create all results for US, global-ex US, and all (0 = no controls; 1 = controls; 2 = controls and industry-time FE)
def main_results(df_us, df_global, df_all):
    us_factors = ['ret_exc_lead1m', 'icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg',
                       'ep_fwd0', 'ep_fwd1', 'ep_fwd2', 'log_be_me',
                       'log_bev_mev', 'ebitda_mev',
                       'coc_avg_options',
                       'mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182',
                       'subj_rr_avg', 'vl_req_ret', 'ms_coe', 'er_ibes', 'er_vl', 'er_ms'
                       ]
    
    global_factors = ['ret_exc_lead1m', 'icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg',
                       'ep_fwd0', 'ep_fwd1', 'ep_fwd2', 'log_be_me',
                       'log_bev_mev', 'ebitda_mev']
    
    control = ['beta_252d', 'ebit_at', 'netdebt_at', 'log_be']

    #us with time FE; appending yield results
    us_0 = fe_reg_function(df_us[df_us.date>=global_start], us_factors, None, None)
    us_0_wacc = fe_reg_function(df_bonds[df_bonds.date>=global_start], ['wacc_icc_avg'], None, None)
    us_0 = pd.concat([us_0, us_0_wacc], axis=0) #merging with bond results
    us_0 = us_0.dropna(axis=1) #dropping control columns

    #us with time FE and controls
    us_1 = fe_reg_function(df_us[df_us.date>=global_start], us_factors, control, None)
    us_1_wacc = fe_reg_function(df_bonds[df_bonds.date>=global_start], ['wacc_icc_avg'], control, None)
    us_1 = pd.concat([us_1, us_1_wacc], axis=0)

    #us with industry-time FE and controls
    us_2 = fe_reg_function(df_us[df_us.date>=global_start], us_factors, control, 'country_ym_gics6')
    us_2_wacc = fe_reg_function(df_bonds[df_bonds.date>=global_start], ['wacc_icc_avg'], control, 'country_ym_gics6')
    us_2 = pd.concat([us_2, us_2_wacc], axis=0)

    #global ex-us with time FE
    global_0 = fe_reg_function(df_global[df_global.date>=global_start], global_factors, None, None)
    global_0 = global_0.dropna(axis=1) #dropping control columns

    #global ex-us with time FE and controls
    global_1 = fe_reg_function(df_global[df_global.date>=global_start], global_factors, control, None)

    #global ex-us with industry-time FE and controls
    global_2 = fe_reg_function(df_global[df_global.date>=global_start], global_factors, control, 'country_ym_gics6')

    # global (all stocks including US) — restricted to global_factors only.
    # US-only return measures (options-implied, SVIX, GLB, analyst-provided,
    # WACC from bonds) live in us_factors / df_bonds and are shown in the
    # us_X tables. They were previously also being computed against df_all
    # (where they're populated only for US firms anyway) and included in
    # all_X, which made all_X.tex include rows that were effectively
    # US-only despite the table caption claiming "Global stocks".
    all_0 = fe_reg_function(df_all[df_all.date>=global_start], global_factors, None, None)
    all_0 = all_0.dropna(axis=1) #dropping control columns

    all_1 = fe_reg_function(df_all[df_all.date>=global_start], global_factors, control, None)

    all_2 = fe_reg_function(df_all[df_all.date>=global_start], global_factors, control, 'country_ym_gics6')

    index_names = {'ret_exc_lead1m': 'Exc. Ret. (leading)', 'icc_avg': 'ICC', 'icc_gls': 'ICC (GLS)', 'icc_ct': 'ICC (CT)', 'icc_oj': 'ICC (OJ)', 'icc_peg': 'ICC (PEG)',
                       'ep_fwd0': 'E/P (latest)', 'ep_fwd1': 'E/P (FY+1)', 'ep_fwd2': 'E/P (FY+2)',
                       'log_be_me': 'LOG(B/M)', 'log_bev_mev': 'LOG(BEV/MEV)', 'ebitda_mev': 'EBITDA/MEV (latest)',
                       'coc_avg_options': 'Exp. Ret. (options)', 'mw30': 'Exp. Ret. (SVIX 30D)', 'mw91': 'Exp. Ret. (SVIX 91D)', 'mw182': 'Exp. Ret. (SVIX 182D)',
                       'glb2_D30': 'Exp. Ret. (GLB 30D)', 'glb2_D91': 'Exp. Ret. (GLB 91D)', 'glb2_D182': 'Exp. Ret. (GLB 182D)',
                       'subj_rr_avg': 'Req. Ret. (VL and MS)', 'vl_req_ret': 'Req. Ret. (ValueLine)', 'ms_coe': 'Req. Ret. (Morningstar)',
                       'er_ibes': 'Exp. Ret. (IBES)', 'er_vl': 'Exp. Ret. (ValueLine)', 'er_ms': 'Exp. Ret. (Morningstar)', 'wacc_icc_avg': 'WACC'}
    column_names = {'S.E. (Greenium)': 'S.E.', 'S.E. (Beta)': 'S.E.', 'S.E. (LOG(BE))': 'S.E.',
                                'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.',
                                'LOG(BE)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'}

    #exporting
    us_0.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "us_0.tex", float_format="%.2f")
    us_1.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "us_1.tex", float_format="%.2f")
    us_2.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "us_2.tex", float_format="%.2f")
    global_0.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "global_0.tex", float_format="%.2f")
    global_1.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "global_1.tex", float_format="%.2f")
    global_2.rename(columns=column_names,
                index=index_names).to_latex(OUTPUT_TABLES / "global_2.tex", float_format="%.2f")
    all_0.rename(columns=column_names,
                 index=index_names).to_latex(OUTPUT_TABLES / "all_0.tex", float_format="%.2f")
    all_1.rename(columns=column_names,
                 index=index_names).to_latex(OUTPUT_TABLES / "all_1.tex", float_format="%.2f")
    all_2.rename(columns=column_names,
                 index=index_names).to_latex(OUTPUT_TABLES / "all_2.tex", float_format="%.2f")

    return us_0, us_1, us_2, global_0, global_1, global_2, all_0, all_1, all_2
us_0, us_1, us_2, global_0, global_1, global_2, all_0, all_1, all_2 = main_results(df[df.excntry=="USA"], df[df.excntry!="USA"], df)
# Save as parquet files 
files = {
    "us_0": us_0,
    "us_1": us_1,
    "us_2": us_2,
    "global_0": global_0,
    "global_1": global_1,
    "global_2": global_2,
    "all_0": all_0,
    "all_1": all_1,
    "all_2": all_2
}
for name, file in files.items():
    file.to_parquet(DATA_INTERIM / f"{name}.parquet")   

# multiple fe regression with the possibility of including rating-time FE
print('Running main bond regressions...')
def fe_reg_function_bonds(input_df, dep_var, controls, industry_fe, rating_fe):
    df_out = pd.DataFrame(index=dep_var, columns=['Greenium', 'S.E. (Greenium)',
                                                  'LOG(Assets)', 'S.E. (LOG(Assets))', 'EBIT/AT', 'S.E. (EBIT/AT)',
                                                  'NETDEBT/AT', 'S.E. (NETDEBT/AT)',
                                                  'LOG(FV)', 'S.E. (LOG(FV))',
                                                  'TMT', 'S.E. (TMT)', 'N', 'R2', 'R2 (Total)'])

    # data requirements
    test_data = input_df[input_df.gics6.notna()]
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.tmt.notna()]
    test_data = test_data[test_data.log_assets.notna()]
    test_data = test_data[test_data.ebit_at.notna()]
    test_data = test_data[test_data.netdebt_at.notna()]
    test_data = test_data[test_data.log_debt_fv.notna()]

    # creating rating-by-time fe
    test_data['rating_num'] = test_data['rating_num'].round(0) # round the firm-level VW rating to nearest integer
    test_data['eom'] = test_data.date.astype(str).str[0:4] + test_data.date.astype(str).str[5:7]
    test_data['rating_ym'] = np.where(test_data['rating_num'].notna(), test_data.rating_num.astype(str) + test_data.eom.astype(str), np.nan)
    test_data['rating_ym'] = pd.factorize(test_data.rating_ym)[0]

    # choosing index for fixed effects use in regressions
    if rating_fe:
        test_data = test_data.set_index(['id', 'rating_ym']) #replacing "date" with rating-year/month as a numerical variable (i.e. rating-time FE)
    elif not rating_fe:
        test_data = test_data.set_index(['id', 'country_ym_num'])  # replacing "date" with country-year/month as a numerical variable (i.e. country-time FE)

    for j in dep_var:
        test_data[j] = test_data[j]*100 #scaling
        if (controls is None):
            X = test_data['green_score']
        else:
            X = pd.concat([test_data['green_score'], test_data[controls]], axis=1)
        X = sm.add_constant(X)

        if (industry_fe is None):  # only time FE
            reg = PanelOLS(test_data[j], X, entity_effects=False, time_effects=True,
                           other_effects=None, drop_absorbed=True, check_rank=False).fit(cov_type="clustered",
                                                                                         clusters=test_data[['date', 'gics']],
                                                                                         cluster_entity=False,
                                                                                         cluster_time=False)
        elif (industry_fe is not None):  # industry and time FE
            if rating_fe: #if rating-time FE, need to include time FE when also industry-time
                reg = PanelOLS(test_data[j], X, entity_effects=False, time_effects=True,
                               other_effects=test_data[industry_fe], drop_absorbed=True, check_rank=False).fit(cov_type="clustered",
                                                                                         clusters=test_data[['date', 'gics']],
                                                                                         cluster_entity=False,
                                                                                         cluster_time=False)
            elif not rating_fe:
                reg = PanelOLS(test_data[j], X, entity_effects=False, time_effects=False,
                               other_effects=test_data[industry_fe], drop_absorbed=True, check_rank=False).fit(cov_type="clustered",
                                                                                         clusters=test_data[['date', 'gics']],
                                                                                         cluster_entity=False,
                                                                                         cluster_time=False)

        df_out.loc[j, 'Greenium'] = reg.params.loc['green_score']
        df_out.loc[j, 'S.E. (Greenium)'] = reg.std_errors.loc['green_score']
        df_out.loc[j, 'N'] = reg.nobs
        df_out.loc[j, 'R2'] = reg.rsquared*100
        df_out.loc[j, 'R2 (Total)'] = reg.rsquared_inclusive*100

        # trying for other coefficients that might not be included
        try:
            df_out.loc[j, 'LOG(Assets)'] = reg.params.loc['log_assets']
        except KeyError:
            df_out.loc[j, 'LOG(Assets)'] = np.nan

        try:
            df_out.loc[j, 'EBIT/AT'] = reg.params.loc['ebit_at']
        except KeyError:
            df_out.loc[j, 'EBIT/AT'] = np.nan

        try:
            df_out.loc[j, 'NETDEBT/AT'] = reg.params.loc['netdebt_at']
        except KeyError:
            df_out.loc[j, 'NETDEBT/AT'] = np.nan

        try:
            df_out.loc[j, 'LOG(FV)'] = reg.params.loc['log_debt_fv']
        except KeyError:
            df_out.loc[j, 'LOG(FV)'] = np.nan

        try:
            df_out.loc[j, 'TMT'] = reg.params.loc['tmt']
        except KeyError:
            df_out.loc[j, 'TMT'] = np.nan

        try:
            df_out.loc[j, 'S.E. (LOG(Assets))'] = reg.std_errors.loc['log_assets']
        except KeyError:
            df_out.loc[j, 'S.E. (LOG(Assets))'] = np.nan

        try:
            df_out.loc[j, 'S.E. (EBIT/AT)'] = reg.std_errors.loc['ebit_at']
        except KeyError:
            df_out.loc[j, 'S.E. (EBIT/AT)'] = np.nan

        try:
            df_out.loc[j, 'S.E. (NETDEBT/AT)'] = reg.std_errors.loc['netdebt_at']
        except KeyError:
            df_out.loc[j, 'S.E. (NETDEBT/AT)'] = np.nan

        try:
            df_out.loc[j, 'S.E. (LOG(FV))'] = reg.std_errors.loc['log_debt_fv']
        except KeyError:
            df_out.loc[j, 'S.E. (LOG(FV))'] = np.nan

        try:
            df_out.loc[j, 'S.E. (TMT)'] = reg.std_errors.loc['tmt']
        except KeyError:
            df_out.loc[j, 'S.E. (TMT)'] = np.nan

    df_out = df_out.astype(float)
    df_out['N'] = df_out['N'].astype(int)
    return df_out
fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], ['rating_num'], None, 'gics6_ym', False)

# using above bond reg-function to create all bond regressions
def bond_main_results():
    test_1 = ['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']
    test_2 = ['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']
    control = ['log_assets', 'netdebt_at', 'ebit_at', 'tmt', 'log_debt_fv']

    # time-FE
    bond_0 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_1, None, None, False)
    bond_0 = bond_0.dropna(axis=1)  # dropping control columns

    # time FE and controls
    bond_1 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_1, control, None, False)

    # industry-time FE and controls
    bond_2 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_1, control, 'gics6_ym', False)

    # rating-time FE
    bond_3 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_2, None, None, True)
    bond_3 = bond_3.dropna(axis=1)  # dropping control columns

    # rating-time FE and controls
    bond_4 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_2, control, None, True)

    # rating-time and industry-time FE and controls
    bond_5 = fe_reg_function_bonds(df_bonds[df_bonds.date >= global_start], test_2, control, 'gics6_ym', True)

    # exporting
    bond_0.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_0.tex", float_format="%.2f")
    bond_1.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_1.tex", float_format="%.2f")
    bond_2.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_2.tex", float_format="%.2f")
    bond_3.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_3.tex", float_format="%.2f")
    bond_4.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_4.tex", float_format="%.2f")
    bond_5.rename(columns={'S.E. (Greenium)': 'S.E.', 'S.E. (LOG(Assets))': 'S.E.',
                         'S.E. (NETDEBT/AT)': 'S.E.', 'S.E. (EBIT/AT)': 'S.E.', 'S.E. (LOG(FV))': 'S.E.', 'S.E. (TMT)': 'S.E.',
                         'LOG(Assets)': 'Size', 'NETDEBT/AT': 'Leverage', 'EBIT/AT': 'Profitability'},
                index={'yield': 'Yield', 'yield_spread': 'Yield Spread', 'yield_adj': 'Adj. Yield', 'yield_spread_adj': 'Adj. Yield Spread', 'rating_num': 'Credit Rating'}).to_latex(
        OUTPUT_TABLES / "bond_5.tex", float_format="%.2f")

    return bond_0, bond_1, bond_2, bond_3, bond_4, bond_5
bond_0, bond_1, bond_2, bond_3, bond_4, bond_5 = bond_main_results()

plt.close("all")
# US and Global: realized returns FE regression output
print('Plotting...')
def realized_ret_plot():
    fig = plt.figure(figsize=(8, 2))
    gs = gridspec.GridSpec(1, 3)

    # details on ICC components
    data_labels = ['US', 'Global ex-US', 'Global'][::-1]

    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze([us_0.loc['ret_exc_lead1m']['Greenium'], global_0.loc['ret_exc_lead1m']['Greenium'], all_0.loc['ret_exc_lead1m']['Greenium']]).transpose()[::-1]
    standard_errors = np.squeeze([us_0.loc['ret_exc_lead1m']['S.E. (Greenium)'], global_0.loc['ret_exc_lead1m']['S.E. (Greenium)'], all_0.loc['ret_exc_lead1m']['S.E. (Greenium)']]).transpose()[
                      ::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze([us_1.loc['ret_exc_lead1m']['Greenium'], global_1.loc['ret_exc_lead1m']['Greenium'], all_1.loc['ret_exc_lead1m']['Greenium']]).transpose()[::-1]
    standard_errors = np.squeeze([us_1.loc['ret_exc_lead1m']['S.E. (Greenium)'], global_1.loc['ret_exc_lead1m']['S.E. (Greenium)'], all_1.loc['ret_exc_lead1m']['S.E. (Greenium)']]).transpose()[
                      ::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze([us_2.loc['ret_exc_lead1m']['Greenium'], global_2.loc['ret_exc_lead1m']['Greenium'], all_2.loc['ret_exc_lead1m']['Greenium']]).transpose()[::-1]
    standard_errors = np.squeeze([us_2.loc['ret_exc_lead1m']['S.E. (Greenium)'], global_2.loc['ret_exc_lead1m']['S.E. (Greenium)'], all_2.loc['ret_exc_lead1m']['S.E. (Greenium)']]).transpose()[
                      ::-1]

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-100, 300)
    ax2.set_xlim(-100, 300)
    ax3.set_xlim(-100, 300)

    ax1.set_ylim(0.5, 3.5)
    ax2.set_ylim(0.5, 3.5)
    ax3.set_ylim(0.5, 3.5)

    ax1.set_title('Country-Time FE')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_US / "realizedret.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
realized_ret_plot()

plt.close("all")

# ICC plot: individual components (global)
def icc_plot_indv():    
    ### GLOBAL INCL. US (ICC AVERAGE AND FOUR COMPONENTS)
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 4])

    # first row is overall ICC
    data_labels = ['ICC']
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(all_0.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['icc_avg']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(232)
    data_list = np.squeeze(all_1.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(233)
    data_list = np.squeeze(all_2.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)

    # details on ICC components
    data_labels = [r'ICC$^{GLS}$', r'ICC$^{CT}$', r'ICC$^{OJ}$', r'ICC$^{PEG}$'][::-1]

    ax4 = fig.add_subplot(234)
    data_list = np.squeeze(all_0.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(all_0.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['S.E. (Greenium)']).transpose()[::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax4.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax5 = fig.add_subplot(235)
    data_list = np.squeeze(all_1.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(all_1.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['S.E. (Greenium)']).transpose()[::-1]

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax5.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax6 = fig.add_subplot(236)
    data_list = np.squeeze(all_2.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(all_2.loc[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']]['S.E. (Greenium)']).transpose()[::-1]

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax6.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax4.set_xlim(-75, 25)
    ax5.set_xlim(-75, 25)
    ax6.set_xlim(-75, 25)

    ax4.set_ylim(0.5, 4.5)
    ax5.set_ylim(0.5, 4.5)
    ax6.set_ylim(0.5, 4.5)

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_all_indv.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
icc_plot_indv()

# ICC plot (global with details; US; global ex-US)
def icc_plot():
    ### GLOBAL INCL. US (ICC AVERAGE AND FOUR COMPONENTS)
    fig = plt.figure(figsize=(8, 1.35))
    gs = gridspec.GridSpec(1, 3)

    # first row is overall ICC
    data_labels = ['ICC']
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(all_0.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['icc_avg']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(232)
    data_list = np.squeeze(all_1.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(233)
    data_list = np.squeeze(all_2.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)

    plt.tight_layout()



    fig = plt.figure(figsize=(8, 1.35))
    gs = gridspec.GridSpec(1, 3)

    xticks = np.arange(-75, 26, 25) #for axis

    data_labels = ['ICC']
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(all_0.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['icc_avg']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.set_ylabel('ICC$^{PEG}$', color='white') # hidden label to adjust LHS
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(all_1.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(all_2.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax3.set_xticks(xticks)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_all.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    ### US (ICC AVERAGE)
    fig = plt.figure(figsize=(8, 1.35))
    gs = gridspec.GridSpec(1, 3)

    xticks = np.arange(-75, 26, 25) #for axis

    data_labels = ['ICC']
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(us_0.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(us_0.loc[['icc_avg']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.set_ylabel('ICC$^{PEG}$', color='white') # hidden label to adjust LHS
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(us_1.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(us_1.loc[['icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(us_2.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(us_2.loc[['icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax3.set_xticks(xticks)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_US / "icc_us.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    ### Global ex-US (ICC AVERAGE)
    fig = plt.figure(figsize=(8, 1.35))
    gs = gridspec.GridSpec(1, 3)

    xticks = np.arange(-75, 26, 25) #for axis

    data_labels = ['ICC']
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(global_0.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(global_0.loc[['icc_avg']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.set_ylabel('ICC$^{PEG}$', color='white')  # hidden label to adjust LHS
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(global_1.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(global_1.loc[['icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(global_2.loc[['icc_avg']]['Greenium'])
    standard_errors = np.squeeze(global_2.loc[['icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax3.set_xticks(xticks)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_gexus.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return None
icc_plot()

plt.close("all")
# Equity valuation ratio plot
def valrat_plot():
    ### GLOBAL INCL. US
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

    data_labels = ['E/P (latest)', 'E/P (FY+1)', 'E/P (FY+2)'][::-1]
    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(all_0.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['S.E. (Greenium)'])

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)
    #ax1.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax2 = fig.add_subplot(232)
    data_list = np.squeeze(all_1.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)
    #ax2.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax3 = fig.add_subplot(233)
    data_list = np.squeeze(all_2.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['ep_fwd0', 'ep_fwd1', 'ep_fwd2']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)
    #ax3.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)

    #formatting vertical location
    ax1.set_ylim(0.5, 3.5)
    ax2.set_ylim(0.5, 3.5)
    ax3.set_ylim(0.5, 3.5)

    # Log B/E
    data_labels = ['LOG (B/M)']

    ax4 = fig.add_subplot(234)
    data_list = np.squeeze(all_0.loc[['log_be_me']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['log_be_me']]['S.E. (Greenium)'])

    positions = [1]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax5 = fig.add_subplot(235)
    data_list = np.squeeze(all_1.loc[['log_be_me']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['log_be_me']]['S.E. (Greenium)'])

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax6 = fig.add_subplot(236)
    data_list = np.squeeze(all_2.loc[['log_be_me']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['log_be_me']]['S.E. (Greenium)'])

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax4.set_xlim(-25, 10)
    ax5.set_xlim(-25, 10)
    ax6.set_xlim(-25, 10)

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "valrat_all.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
valrat_plot()

plt.close("all")
# Option-implied expected returns plot
def options_plot():
    fig = plt.figure(figsize=(8, 4))

    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 6])

    data_labels = ['ER']

    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(us_0.loc[['coc_avg_options']]['Greenium'])
    standard_errors = np.squeeze(us_0.loc[['coc_avg_options']]['S.E. (Greenium)'])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Country-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(232)
    data_list = np.squeeze(us_1.loc[['coc_avg_options']]['Greenium'])
    standard_errors = np.squeeze(us_1.loc[['coc_avg_options']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nTime FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(233)
    data_list = np.squeeze(us_2.loc[['coc_avg_options']]['Greenium'])
    standard_errors = np.squeeze(us_2.loc[['coc_avg_options']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nIndustry-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    # details on ICC components
    data_labels = [r'ER$^{SVIX}_{30D}$', r'ER$^{SVIX}_{91D}$', r'ER$^{SVIX}_{182D}$', r'ER$^{GLB}_{30D}$', r'ER$^{GLB}_{91D}$', r'ER$^{GLB}_{182D}$'][::-1]

    ax4 = fig.add_subplot(234)
    data_list = np.squeeze(us_0.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(us_0.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['S.E. (Greenium)']).transpose()[
                      ::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax4.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax5 = fig.add_subplot(235)
    data_list = np.squeeze(us_1.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(us_1.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['S.E. (Greenium)']).transpose()[
                      ::-1]

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax5.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax6 = fig.add_subplot(236)
    data_list = np.squeeze(us_2.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['Greenium']).transpose()[::-1]
    standard_errors = np.squeeze(us_2.loc[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']]['S.E. (Greenium)']).transpose()[
                      ::-1]

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax6.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax1.set_xlim(-200, 50)
    ax2.set_xlim(-100, 25)
    ax3.set_xlim(-75, 25)

    ax4.set_xlim(-200, 50)
    ax5.set_xlim(-100, 25)
    ax6.set_xlim(-75, 25)

    ax4.set_ylim(0.5, 6.5)
    ax5.set_ylim(0.5, 6.5)
    ax6.set_ylim(0.5, 6.5)

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2)  # Adjust margins and spacing
    plt.savefig(OUTPUT_FIGURES_US / "er_options.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
options_plot()

plt.close("all")
# Subjective expected and required returns
def subj_plot():
    fig = plt.figure(figsize=(10, 3))

    gs = gridspec.GridSpec(2, 4, height_ratios=[2, 3])

    data_labels = [r'RR$^{VL}$', r'RR$^{MS}$'][::-1]

    ax1 = fig.add_subplot(gs[0,0])
    data_list = np.squeeze(us_0.loc[['vl_req_ret', 'ms_coe']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_0.loc[['vl_req_ret', 'ms_coe']]['S.E. (Greenium)'].transpose())[::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    # ax1.set_title('Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax1.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax2 = fig.add_subplot(gs[0,1])
    data_list = np.squeeze(us_1.loc[['vl_req_ret', 'ms_coe']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_1.loc[['vl_req_ret', 'ms_coe']]['S.E. (Greenium)'].transpose())[::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    # ax2.set_title('Controls +\nTime FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax2.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax3 = fig.add_subplot(gs[0,2])
    data_list = np.squeeze(us_2.loc[['vl_req_ret', 'ms_coe']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_2.loc[['vl_req_ret', 'ms_coe']]['S.E. (Greenium)'].transpose())[::-1]

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    # ax3.set_title('Controls +\nIndustry-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax3.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax1.set_title('Time FE')
    ax2.set_title('Controls +\nTime FE')
    ax3.set_title('Controls +\nIndustry-Time FE')

    ax1.set_xlim(-75, 25)
    ax2.set_xlim(-75, 25)
    ax3.set_xlim(-75, 25)

    ax1.set_ylim(0.5, 2.5)
    ax2.set_ylim(0.5, 2.5)
    ax3.set_ylim(0.5, 2.5)

    ### errorplot - ICC
    data_labels = [r'ER$^{IBES}$', r'ER$^{VL}$', r'ER$^{MS}$'][::-1]

    ax4 = fig.add_subplot(gs[1,0])
    data_list = np.squeeze(us_0.loc[['er_ibes', 'er_vl', 'er_ms']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_0.loc[['er_ibes', 'er_vl', 'er_ms']]['S.E. (Greenium)'].transpose())[::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    # ax4.set_title('Time FE')
    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax4.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax5 = fig.add_subplot(gs[1,1])
    data_list = np.squeeze(us_1.loc[['er_ibes', 'er_vl', 'er_ms']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_1.loc[['er_ibes', 'er_vl', 'er_ms']]['S.E. (Greenium)'].transpose())[::-1]

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    # ax5.set_title('Controls +\nTime FE')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax5.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax6 = fig.add_subplot(gs[1,2])
    data_list = np.squeeze(us_2.loc[['er_ibes', 'er_vl', 'er_ms']]['Greenium'].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        us_2.loc[['er_ibes', 'er_vl', 'er_ms']]['S.E. (Greenium)'].transpose())[::-1]

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    # ax6.set_title('Controls +\nIndustry-Time FE')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax6.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax4.set_xlim(-480, 100)
    ax5.set_xlim(-480, 100)
    ax6.set_xlim(-480, 100)

    ax4.set_ylim(0.5, 3.5)
    ax5.set_ylim(0.5, 3.5)
    ax6.set_ylim(0.5, 3.5)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_US / "subj_ret.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
subj_plot()

# firm-level bond yields
def bond_plot():
    ### with time FE
    fig = plt.figure(figsize=(8, 3))
    data_labels = ['Yield (bps)', 'Yield Spread (bps)', 'Adj. Yield (bps)', 'Adj. Yield Spread (bps)', 'Credit Rating (num)'][::-1]

    ax1 = fig.add_subplot(131)
    data_list = np.squeeze(bond_0.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_0.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['S.E. (Greenium)'].transpose())[::-1]

    positions = range(1, len(data_list) + 1)
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=1, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(bond_1.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_1.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['S.E. (Greenium)'].transpose())[::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax2.set_title('Controls +\nTime FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=1, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(bond_2.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_2.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj', 'rating_num']]['S.E. (Greenium)'].transpose())[::-1]

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nIndustry-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=1, linewidth=1)

    ax1.set_xlim(-120, 25)
    ax2.set_xlim(-60, 25)
    ax3.set_xlim(-60, 25)

    ax1.set_ylim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    ax3.set_ylim(0.5, 5.5)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)

    plt.savefig(OUTPUT_FIGURES_US / "bonds.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # with rating-time fe
    ### errorplot - yields and ratings
    fig = plt.figure(figsize=(8, 3))
    data_labels = ['Yield (bps)', 'Yield Spread (bps)', 'Adj. Yield (bps)', 'Adj. Yield Spread (bps)'][::-1]

    ax1 = fig.add_subplot(131)
    data_list = np.squeeze(bond_3.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_3.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['S.E. (Greenium)'].transpose())[::-1]

    positions = range(1, len(data_list) + 1)
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Rating-Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    # ax1.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')
    ax1.axvline(0, color='black', alpha=1, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(bond_4.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_4.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['S.E. (Greenium)'].transpose())[::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax2.set_title('Controls +\nRating-Time FE')
    ax2.set_yticks([])
    # ax2.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')
    ax2.axvline(0, color='black', alpha=1, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(bond_5.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['Greenium'].transpose())[::-1]
    standard_errors = np.squeeze(bond_5.loc[['yield', 'yield_spread', 'yield_adj', 'yield_spread_adj']]['S.E. (Greenium)'].transpose())[::-1]

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls + Rating-Time FE\n+ Industry-Time FE')
    ax3.set_yticks([])
    # ax3.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')
    ax3.axvline(0, color='black', alpha=1, linewidth=1)

    ax1.set_xlim(-30, 15)
    ax2.set_xlim(-30, 15)
    ax3.set_xlim(-30, 15)

    ax1.set_ylim(0.5, 4.5)
    ax2.set_ylim(0.5, 4.5)
    ax3.set_ylim(0.5, 4.5)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)

    plt.savefig(OUTPUT_FIGURES_US / "bonds_ratingcontrol.pdf", format="pdf", bbox_inches="tight")
    plt.show()



    return None
bond_plot()

plt.close("all")
# global decile regression
def icc_deciles():
    test_data = df[df.date >= global_start]
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.gics6.notna()]
    test_data = test_data.dropna(subset=['beta_252d', 'ebit_at', 'netdebt_at', 'log_be'])
    test_data = test_data.set_index(['id', 'country_ym_num'])

    # decile regression, where constant represents left-out group 5/10
    reg = PanelOLS(test_data['icc_avg']*100, sm.add_constant(
        test_data[['q_1_10', 'q_2_10', 'q_3_10', 'q_4_10', 'q_6_10', 'q_7_10', 'q_8_10', 'q_9_10', 'q_10_10']]),
                   entity_effects=False,
                   time_effects=True, other_effects=None,
                   drop_absorbed=True,
                   check_rank=False).fit(cov_type="clustered", clusters=test_data[['date', 'gics']], cluster_entity=False, cluster_time=False)
    reg1_coef = pd.DataFrame(reg.params[1:10]).transpose()
    reg1_stde = pd.DataFrame(reg.std_errors[1:10]).transpose()

    reg = PanelOLS(test_data['icc_avg']*100,
                   sm.add_constant(test_data[
                                       ['q_1_10', 'q_2_10', 'q_3_10', 'q_4_10', 'q_6_10', 'q_7_10', 'q_8_10', 'q_9_10', 'q_10_10',
                                        'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']]),
                   entity_effects=False, time_effects=True, other_effects=None,
                   drop_absorbed=True,
                   check_rank=False).fit(cov_type="clustered", clusters=test_data[['date', 'gics']], cluster_entity=False, cluster_time=False)
    reg2_coef = pd.DataFrame(reg.params[1:10]).transpose()
    reg2_stde = pd.DataFrame(reg.std_errors[1:10]).transpose()

    reg = PanelOLS(test_data['icc_avg']*100,
                   sm.add_constant(test_data[
                                       ['q_1_10', 'q_2_10', 'q_3_10', 'q_4_10', 'q_6_10', 'q_7_10', 'q_8_10', 'q_9_10', 'q_10_10',
                                        'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']]),
                   entity_effects=False, time_effects=False, other_effects=test_data['country_ym_gics6'],
                   drop_absorbed=True,
                   check_rank=False).fit(cov_type="clustered", clusters=test_data[['date', 'gics']], cluster_entity=False, cluster_time=False)
    reg3_coef = pd.DataFrame(reg.params[1:10]).transpose()
    reg3_stde = pd.DataFrame(reg.std_errors[1:10]).transpose()

    ### errorplot - ICC
    fig = plt.figure(figsize=(8, 4))

    gs = gridspec.GridSpec(1, 3)

    ax1 = fig.add_subplot(gs[0])
    data_labels = ['1/10', '2/10', '3/10', '4/10', '6/10', '7/10', '8/10', '9/10', '10/10'][::-1]

    data_list = np.squeeze(reg1_coef.transpose())[::-1]
    standard_errors = np.squeeze(reg1_stde.transpose())[::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(132)
    data_list = np.squeeze(reg2_coef.transpose())[::-1]
    standard_errors = np.squeeze(reg2_stde.transpose())[::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(133)
    data_list = np.squeeze(reg3_coef.transpose())[::-1]
    standard_errors = np.squeeze(reg3_stde.transpose())[::-1]

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-150, 100)
    ax2.set_xlim(-150, 100)
    ax3.set_xlim(-150, 100)

    ax1.set_title('Country-Time FE')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')

    ax = fig.add_subplot(111, frame_on=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.grid(False)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_deciles_all.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    print("point estimates with country-time FE")
    print(reg2_coef.transpose())

    return None
icc_deciles()

# fama macbeth regression function
def fama_macbeth(data, dep_var, esg_var, controls, fixed_effect):
    # setting data requirements
    data = data.dropna(subset=[esg_var], axis=0)
    data = data.dropna(subset=[dep_var], axis=0)
    data = data.dropna(subset=controls, axis=0)
    # scaling dependent variable
    Y = data[dep_var]*100
    # adding constant and defining X variables
    X = sm.add_constant(data[[esg_var] + controls])
    if fixed_effect is not None:
        X = pd.concat([X, pd.get_dummies(data[fixed_effect], drop_first=True).astype(int)], axis=1)
    try:
        model = sm.OLS(Y, X, missing='drop').fit()
    except:
        return pd.Series([np.nan])
    return pd.Series([model.params.loc[esg_var]])

plt.close("all")

# fama macbeth outputs
print('Running greenium over time analysis...')
def fama_macbeth_full(esg_var, fig_name, trend_baseline=None, trend_slope=None):
    results = df[(df.date >= global_start) & (df.gics6.notna())].groupby(['date'], observed=False).apply(
        fama_macbeth, dep_var='icc_avg', esg_var=esg_var,
        controls=['beta_252d', 'ebit_at', 'log_be', 'netdebt_at'],
        fixed_effect='excntry').rename(columns={0: 'icc_avg'}).dropna()

    print(f"FM ({esg_var}): first={results['icc_avg'].iloc[0]:.2f}, avg={results['icc_avg'].mean():.2f}, last={results['icc_avg'].iloc[-1]:.2f}")

    plt.plot(
        results.icc_avg.rolling(12).mean(),
        color='tab:green',
        linestyle='dashed'
    )

    # trend line from pooled panel regression (icc_timetrend_data)
    if trend_baseline is not None and trend_slope is not None:
        dates = results.index
        time_trend = np.linspace(0, 1, len(dates))
        trend_line = pd.Series(trend_baseline + trend_slope * time_trend, index=dates)
        plt.plot(trend_line, color='tab:red', linewidth=1.5, label='Full sample linear time trend')
        print(f"FM ({esg_var}): pooled time trend implied greenium by end of sample = {trend_baseline + trend_slope:.2f}")

    # formatting and saving
    plt.axhline(0, linewidth=0.8, color='black', alpha=0.5)
    plt.ylabel("12M rolling average estimate (bps)")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / f"{fig_name}.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None

plt.close("all")
# global icc timetrend
def icc_timetrend_data():
    test_data = df.copy()
    test_data = test_data[test_data.green_score.notna()].copy()
    test_data = test_data[test_data.gics6.notna()]
    test_data = test_data[test_data.date >= global_start]
    time_index = test_data['date'].drop_duplicates().sort_values().to_frame()
    time_index['time_trend'] = 1
    time_index['time_trend'] = (time_index['time_trend'].cumsum()-1)
    time_index['time_trend'] = time_index['time_trend'] / time_index['time_trend'].max() # scaling to [0,1]
    test_data = test_data.merge(time_index, how='left', on=['date'])
    test_data['green_score_timetrend'] = test_data['green_score'] * test_data['time_trend']
    test_data = test_data.set_index(['id', 'country_ym_num'])

    params_timetrend_baseline = pd.DataFrame(
        index=['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'icc_avg'],
        columns=['slope', 'std_error'])
    params_timetrend_trend = pd.DataFrame(
        index=['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'icc_avg'],
        columns=['slope', 'std_error'])

    X = pd.concat([test_data['green_score'], test_data['green_score_timetrend'], test_data[['beta_252d', 'ebit_at', 'netdebt_at', 'log_be']]], axis=1)
    X = sm.add_constant(X)
    for i in params_timetrend_baseline.index:
        reg = PanelOLS(test_data[i]*100, X, entity_effects=False, time_effects=True, other_effects=None,
                       drop_absorbed=True,
                       check_rank=False).fit(cov_type="clustered", clusters=test_data[['date', 'gics']], cluster_entity=False, cluster_time=False)

        params_timetrend_baseline.loc[i, 'slope'] = reg.params['green_score']
        params_timetrend_trend.loc[i, 'slope'] = reg.params['green_score_timetrend']
        params_timetrend_baseline.loc[i, 'std_error'] = reg.std_errors['green_score']
        params_timetrend_trend.loc[i, 'std_error'] = reg.std_errors['green_score_timetrend']

    print("baseline in trend")
    print(params_timetrend_baseline)
    print("time trend")
    print(params_timetrend_trend)

    return params_timetrend_baseline, params_timetrend_trend

# Diagnostics: firm count over time (to understand difference between FMB and pooled estimates)
def fm_diagnostics_firm_count():
    temp = df[df.green_score.notna() & df.gics6.notna() & (df.date >= global_start)]
    temp = temp.dropna(subset=['beta_252d', 'ebit_at', 'netdebt_at', 'log_be', 'icc_avg'])
    firm_counts = temp.groupby('date')['id'].nunique()
    print(f"Firms per month — Min: {firm_counts.min()}, Max: {firm_counts.max()}, Ratio: {firm_counts.max()/firm_counts.min():.1f}x")
    plt.plot(firm_counts)
    plt.ylabel("Number of firms")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "diag_firm_count_over_time.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
fm_diagnostics_firm_count()

# Estimate pooled time trend (needed for trend line in FM plots)
icc_timetrend_baseline, icc_timetrend_trend = icc_timetrend_data()
print(f"icc_timetrend_baseline slope: {float(icc_timetrend_baseline.loc['icc_avg', 'slope'])}")
print(f"icc_timetrend_trend slope: {float(icc_timetrend_trend.loc['icc_avg', 'slope'])}")
icc_timetrend_baseline.to_parquet(DATA_INTERIM / 'icc_timetrend_baseline.parquet')
icc_timetrend_trend.to_parquet(DATA_INTERIM / 'icc_timetrend_trend.parquet')


fama_macbeth_full(esg_var='green_score', fig_name='icc_fm_all',
                  trend_baseline=float(icc_timetrend_baseline.loc['icc_avg', 'slope']),
                  trend_slope=float(icc_timetrend_trend.loc['icc_avg', 'slope']))
# Fama-MacBeth with raw (unstandardized) scope 1 emission intensity
fama_macbeth_full(esg_var='raw_scope1int_assets', fig_name='icc_fm_raw_scope1int_assets')

# Last-period cross-sectional regression with clustered standard errors
def last_period_regression(df, dep_var, esg_var, controls, cluster_var):
    """OLS regression on the last available period with industry-clustered SEs."""
    last_date = df.loc[df[esg_var].notna() & df[dep_var].notna(), 'date'].max()
    data = df[df['date'] == last_date].dropna(subset=[esg_var, dep_var] + controls).copy()
    Y = data[dep_var] * 100
    X = sm.add_constant(data[[esg_var] + controls])
    X = pd.concat([X, pd.get_dummies(data['excntry'], drop_first=True).astype(int)], axis=1)
    model = sm.OLS(Y, X, missing='drop').fit(
        cov_type='cluster', cov_kwds={'groups': data[cluster_var]}
    )
    print(f"\nLast-period regression ({last_date.strftime('%Y-%m-%d')}, "
          f"n={len(data)}, clustered by {cluster_var}):")
    print(f"  {esg_var}: coef={model.params[esg_var]:.4f}, se={model.bse[esg_var]:.4f}, "
          f"t={model.tvalues[esg_var]:.2f}, p={model.pvalues[esg_var]:.4f}")
    return model

last_period_model = last_period_regression(
    df=df[df.gics6.notna()], dep_var='icc_avg', esg_var='green_score',
    controls=['beta_252d', 'ebit_at', 'log_be', 'netdebt_at'], cluster_var='gics'
)

def icc_timetrend(icc_timetrend_base, icc_timetrend_trend):
    ### errorplot - ICC
    fig = plt.figure(figsize=(8, 2))

    gs = gridspec.GridSpec(1, 2)

    data_labels = ['ICC']

    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(icc_timetrend_baseline.loc['icc_avg'][['slope']])
    standard_errors = np.squeeze(icc_timetrend_baseline.loc['icc_avg'][['std_error']])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Baseline coefficient (' + 'g₁' + ")") #β₁
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(gs[1])
    data_list = np.squeeze(icc_timetrend_trend.loc['icc_avg'][['slope']])
    standard_errors = np.squeeze(icc_timetrend_trend.loc['icc_avg'][['std_error']])

    positions = [1]
    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax2.set_title('Time-trend interaction (' + 'g₂' + ")") #β₂
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax1.set_xlim(-40, 20)
    ax2.set_xlim(-100, 25)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_timetrend_all.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None

def icc_timetrend_indv(icc_timetrend_base, icc_timetrend_trend):
    ### errorplot - ICC
    fig = plt.figure(figsize=(6, 3.5))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])

    data_labels = ['ICC']

    ax1 = fig.add_subplot(gs[0])
    data_list = np.squeeze(icc_timetrend_baseline.loc['icc_avg'][['slope']])
    standard_errors = np.squeeze(icc_timetrend_baseline.loc['icc_avg'][['std_error']])

    positions = [1]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Baseline coefficient (' + 'g₁' + ")") #β₁
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(222)
    data_list = np.squeeze(icc_timetrend_trend.loc['icc_avg'][['slope']])
    standard_errors = np.squeeze(icc_timetrend_trend.loc['icc_avg'][['std_error']])

    positions = [1]
    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax2.set_title('Time-trend interaction (' + 'g₂' + ")") #β₂
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    data_labels = [r'ICC$^{GLS}$', r'ICC$^{CT}$', r'ICC$^{OJ}$', r'ICC$^{PEG}$'][::-1]

    ax3 = fig.add_subplot(223)
    data_list = np.squeeze(icc_timetrend_baseline.iloc[:-1][['slope']])[::-1]
    standard_errors = np.squeeze(icc_timetrend_baseline.iloc[:-1][['std_error']])[::-1]

    positions = range(1, len(data_list) + 1)
    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax3.set_yticks(positions)
    ax3.set_yticklabels(data_labels)
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)
    ax3.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax4 = fig.add_subplot(224)
    data_list = np.squeeze(icc_timetrend_trend.iloc[:-1][['slope']])[::-1]
    standard_errors = np.squeeze(icc_timetrend_trend.iloc[:-1][['std_error']])[::-1]

    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax4.set_yticks([])
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)
    ax4.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax1.set_xlim(-20, 35)
    ax2.set_xlim(-150, 25)
    ax3.set_xlim(-20, 35)
    ax4.set_xlim(-150, 25)

    ax3.set_ylim(0.5, 4.5)
    ax4.set_ylim(0.5, 4.5)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "icc_timetrend_all_indv.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None

# Time trend with overall ICC
icc_timetrend(icc_timetrend_baseline, icc_timetrend_trend)
# Time trend with individual ICC measures
icc_timetrend_indv(icc_timetrend_baseline, icc_timetrend_trend)

plt.close("all")
# US firm-level
print('Running firm-level greenium analysis...')
def wacc():
    fig = plt.figure(figsize=(8, 3))

    gs = gridspec.GridSpec(3, 3)

    data_labels = ['WACC']

    ax1 = fig.add_subplot(gs[0,0])
    data_list = np.squeeze(us_0.loc[['wacc_icc_avg']]['Greenium'])
    standard_errors = np.squeeze(us_0.loc[['wacc_icc_avg']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax1.set_title('Time FE')
    # ax1.set_yticks(positions)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(gs[0,1])
    data_list = np.squeeze(us_1.loc[['wacc_icc_avg']]['Greenium'])
    standard_errors =  np.squeeze(us_1.loc[['wacc_icc_avg']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_title('Controls +\nTime FE')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(gs[0,2])
    data_list = np.squeeze(us_2.loc[['wacc_icc_avg']]['Greenium'])
    standard_errors =  np.squeeze(us_2.loc[['wacc_icc_avg']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_title('Controls +\nIndustry-Time FE')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    ### errorplot - ICC
    data_labels = ['EBITDA/MEV']

    ax4 = fig.add_subplot(gs[1,0])
    data_list = np.squeeze(us_0.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(us_0.loc[['ebitda_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    #ax4.set_title('Time FE')
    # ax4.set_yticks(positions)
    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax5 = fig.add_subplot(gs[1,1])
    data_list = np.squeeze(us_1.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(us_1.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    #ax5.set_title('Controls +\nTime FE')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax6 = fig.add_subplot(gs[1,2])
    data_list = np.squeeze(us_2.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(us_2.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    #ax6.set_title('Controls +\nIndustry-Time FE')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)

    ### errorplot - ICC
    data_labels = ['LOG(BEV/MEV)']

    ax7 = fig.add_subplot(gs[2,0])
    data_list = np.squeeze(us_0.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(us_0.loc[['log_bev_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax7.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    #ax7.set_title('Time FE')
    # ax7.set_yticks(positions)
    ax7.set_yticks(positions)
    ax7.set_yticklabels(data_labels)
    ax7.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax8 = fig.add_subplot(gs[2,1])
    data_list = np.squeeze(us_1.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(us_1.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax8.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    #ax8.set_title('Controls +\nTime FE')
    ax8.set_yticks([])
    ax8.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax9 = fig.add_subplot(gs[2,2])
    data_list = np.squeeze(us_2.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(us_2.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax9.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    #ax9.set_title('Controls +\nIndustry-Time FE')
    ax9.set_yticks([])
    ax9.axvline(0, color='black', alpha=0.5, linewidth=1)


    ax1.set_xlim(-50, 10)
    ax2.set_xlim(-50, 10)
    ax3.set_xlim(-50, 10)

    ax4.set_xlim(-230, 75)
    ax5.set_xlim(-230, 75)
    ax6.set_xlim(-230, 75)

    ax7.set_xlim(-35, 10)
    ax8.set_xlim(-35, 10)
    ax9.set_xlim(-35, 10)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_US / "wacc.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
wacc()

# Global-ex US: WACC
def wacc_global():
    fig = plt.figure(figsize=(8, 2))

    gs = gridspec.GridSpec(2, 3)

    # ebitda/mev
    data_labels = ['EBITDA/MEV']

    ax1 = fig.add_subplot(gs[0, 0])
    data_list = np.squeeze(global_0.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(global_0.loc[['ebitda_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(gs[0, 1])
    data_list = np.squeeze(global_1.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(global_1.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(gs[0, 2])
    data_list = np.squeeze(global_2.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(global_2.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    # log(bev/mev)
    data_labels = ['LOG(BEV/MEV)']

    ax4 = fig.add_subplot(gs[1, 0])
    data_list = np.squeeze(global_0.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(global_0.loc[['log_bev_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax5 = fig.add_subplot(gs[1, 1])
    data_list = np.squeeze(global_1.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(global_1.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5,
                  linestyle='None')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax6 = fig.add_subplot(gs[1, 2])
    data_list = np.squeeze(global_2.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(global_2.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5,
                  linestyle='None')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)

    ### formatting
    ax1.set_title('Country-Time FE')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')

    ax1.set_xlim(-230, 75)
    ax2.set_xlim(-230, 75)
    ax3.set_xlim(-230, 75)

    ax4.set_xlim(-35, 10)
    ax5.set_xlim(-35, 10)
    ax6.set_xlim(-35, 10)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "wacc_g.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
wacc_global()

# Global incl. US: WACC
def wacc_all():
    fig = plt.figure(figsize=(8, 2))

    gs = gridspec.GridSpec(2, 3)

    # ebitda/mev
    data_labels = ['EBITDA/MEV']

    ax1 = fig.add_subplot(gs[0, 0])
    data_list = np.squeeze(all_0.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['ebitda_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax2 = fig.add_subplot(gs[0, 1])
    data_list = np.squeeze(all_1.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax3 = fig.add_subplot(gs[0, 2])
    data_list = np.squeeze(all_2.loc[['ebitda_mev']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['ebitda_mev']]['S.E. (Greenium)'])

    ax3.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', alpha=0.5, linewidth=1)

    # log(bev/mev)
    data_labels = ['LOG(BEV/MEV)']

    ax4 = fig.add_subplot(gs[1, 0])
    data_list = np.squeeze(all_0.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(all_0.loc[['log_bev_mev']]['S.E. (Greenium)'])

    positions = [1]  # [1, 2, 3, 4]
    ax4.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax4.set_yticks(positions)
    ax4.set_yticklabels(data_labels)
    ax4.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax5 = fig.add_subplot(gs[1, 1])
    data_list = np.squeeze(all_1.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(all_1.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5,
                  linestyle='None')
    ax5.set_yticks([])
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax6 = fig.add_subplot(gs[1, 2])
    data_list = np.squeeze(all_2.loc[['log_bev_mev']]['Greenium'])
    standard_errors = np.squeeze(all_2.loc[['log_bev_mev']]['S.E. (Greenium)'])

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5,
                  linestyle='None')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)

    ### formatting
    ax1.set_title('Country-Time FE')
    ax2.set_title('Controls +\nCountry-Time FE')
    ax3.set_title('Controls +\nCountry-Ind.-Time FE')

    ax1.set_xlim(-230, 75)
    ax2.set_xlim(-230, 75)
    ax3.set_xlim(-230, 75)

    ax4.set_xlim(-35, 10)
    ax5.set_xlim(-35, 10)
    ax6.set_xlim(-35, 10)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_GLOBAL / "wacc_all.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    return None
wacc_all()

pd.options.display.float_format = '{:.2f}'.format # changing float format to 2 decimals

plt.close("all")
# global icc map
print('Greenium by country analysis...')
def icc_country_map():
    #icc by country with time FE (min 20 obs per country in each period) - icc slope scale capped to 80th percentile for visual purposes
    test_data = df[df.date >= global_start]
    test_data['country_ym'] = np.where(test_data['excntry'].notna(), test_data.excntry.astype(str) + test_data.eom.astype(str), np.nan)
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.icc_avg.notna()]
    test_data = test_data[test_data.gics6.notna()]
    test_data = test_data.dropna(subset=['beta_252d', 'ebit_at', 'netdebt_at', 'log_be'])
    test_data['country_ym_count'] = test_data.groupby('country_ym')['country_ym'].transform('count')
    test_data = test_data[test_data.country_ym_count >= 20] #requiring at least 20 observations in a country-date to be included
    test_data['country_date_count'] = test_data.groupby('excntry')['date'].transform('nunique')
    test_data = test_data[test_data.country_date_count >= 12]  # requiring at least 12 unique dates per country

    test_data['icc_avg'] = test_data['icc_avg']*100 #scaling to bps

    test_data = test_data.set_index(['id', 'date'])
    out = pd.DataFrame(index=test_data.excntry.drop_duplicates().sort_values(), columns=['estimate', 'stderror', 'tstat'])
    for i in out.index:
        print(i)
        temp_data = test_data[test_data.excntry==i].copy()
        reg = PanelOLS(temp_data['icc_avg'], sm.add_constant(temp_data[['green_score', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']]), entity_effects=False, time_effects=True,
                   other_effects=None, drop_absorbed=True, check_rank=False).fit(cov_type="clustered", cluster_entity=False, cluster_time=True, clusters=temp_data['gics'])

        out.loc[i, 'estimate'] = reg.params.loc['green_score']
        out.loc[i, 'stderror'] = reg.std_errors.loc['green_score']
        out.loc[i, 'tstat'] = reg.tstats.loc['green_score']

    out = out.reset_index()

    def maps(out):
        ##### global heatmaps of emission coefficients
        country_list = pd.read_excel(DATA_RAW / "other" / "country_abbr.xlsx", sheet_name="Sheet1", usecols=['iso_3', 'Country'])

        country_list = country_list.rename(
            columns={'iso_3': 'excntry'})

        out = out.merge(country_list, how='left', on=['excntry'])

        # renaming to conform with data
        out = out.replace('Czech Republic', 'Czechia')
        out = out.replace('United States', 'United States of America')
        out = out.replace('Korea, South', 'South Korea')
        out = out.replace('Hong Kong (China)', 'Hong Kong')

        # plotting coefficients and t-stats on global map
        for loop_var in ['estimate', 'tstat']:
            # Load a GeoDataFrame containing countries' geometry
            if False:
                # Download and save to data/raw
                import requests
                
                url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
                zip_path = DATA_RAW / "other" / "countries.zip"
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            f.write(chunk)

                print(f"Downloaded countries ZIP to {zip_path}")
                
                # Old approach - download directly in script
                # url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
                # world = gpd.read_file(url)
            world = gpd.read_file(DATA_RAW / "other" / "countries.zip")
            world = world[world.CONTINENT != "Antarctica"] #removing Antarctica

            # now merging map data with estimates country-by-country
            world = world.merge(out, how='left', left_on='NAME', right_on='Country')

            # Define the colormap
            #colormap = plt.cm.RdYlBu
            colormap = plt.cm.BrBG.reversed() #reversed brown-to-green map
            # Normalize the colormap
            norm = mcolors.Normalize(vmin=world[loop_var].min(), vmax=world[loop_var].max())

            sym_value = max(abs(world[loop_var].min()), abs(world[loop_var].max()))
            norm = mcolors.Normalize(vmin=-sym_value, vmax=sym_value)   # symmetric

            # capping axis at 80% percentile
            if loop_var=='estimate':
                norm = mcolors.Normalize(vmin=-abs(out[loop_var]).quantile(0.8),
                                     vmax=abs(out[loop_var]).quantile(0.8))  # symmetric

            # Function to apply color based on value
            def apply_color(value):
                if value == "nan":
                    return 'lightgrey'  # Color for zero values
                else:
                    return mcolors.to_hex(colormap(norm(value)))  # Color for non-zero values

            # Apply the color mapping function
            world['color'] = world[loop_var].apply(apply_color)

            world['color'] = world['color'].replace("#000000", 'lightgrey')

            # Plot the merged DataFrame using the colors defined
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            world.plot(color=world['color'], ax=ax, legend=True)

            # titles = {
            #     'estimate': 'estimate: icc_avg',
            #     'tstat': 'tstat: icc_avg',
            # }

            #title = titles.get(loop_var, "Default Title")  # Fallback to "Default Title" if not found

            # Create a colorbar as a legend
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm._A = []  # Empty array for the data range
            #cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1) 
            fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()

            file_path = OUTPUT_FIGURES_MAPS / "{}.svg".format(loop_var)

            plt.savefig(file_path)

            plt.show()
            plt.clf()
        return None
    maps(out)
    return None
icc_country_map()

# green/brown country regression using CCPI
def icc_country_ccpi():
    # loading ccpi data
    df_ccpi = pd.read_excel(DATA_RAW / "ccpi" / "CCPI_data.xlsx", sheet_name='ccpi_import')
    df_ccpi = df_ccpi.drop('country', axis=1)
    df_ccpi = df_ccpi[df_ccpi.country_code.notna()]
    df_ccpi = df_ccpi.melt(id_vars=['country_code'], var_name='year', value_name='ccpi_index')
    df_ccpi = df_ccpi.sort_values(['country_code', 'year'])
    df_ccpi = df_ccpi.rename(columns={'country_code': 'excntry'})
    df_ccpi.year = df_ccpi.year.astype(int)

    # loading test data
    test_data = df.copy()
    test_data = test_data[(test_data.date >= global_start)]
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.icc_avg.notna()]
    test_data = test_data[test_data.gics6.notna()]
    test_data = test_data.dropna(subset=['beta_252d', 'ebit_at', 'netdebt_at', 'log_be'])
    test_data['country_ym_count'] = test_data.groupby('country_ym')['country_ym'].transform('count')

    # adding ccpi and creating dummy interaction for country having score above median ("green country")
    test_data = test_data.merge(df_ccpi, how='left', on=['year', 'excntry'])

    inspect = test_data[['excntry', 'year', 'ccpi_index']].drop_duplicates(subset=['excntry', 'year']).dropna()
    inspect['ccpi_median'] = inspect.groupby('year')['ccpi_index'].transform('median')
    inspect['GreenCountry'] = np.where(inspect['ccpi_index'] > inspect['ccpi_median'], 1, 0)
    test_data = test_data.merge(inspect[['excntry', 'year', 'GreenCountry']], how='left', on=['year', 'excntry'])
    test_data['GreenScore*GreenCountry'] = test_data['green_score'] * test_data['GreenCountry']

    # preparing for reg's
    test_data = test_data.set_index(['id', 'country_ym_num'])

    # global greenium
    reg_1 = PanelOLS(test_data['icc_avg'] * 100, test_data[['green_score', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']],
                     entity_effects=False, time_effects=True,
                     other_effects=None, drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=False, cluster_time=False, clusters=test_data[['date', 'gics']])
    print(reg_1)

    # basic regression (ccpi index > median dummy)
    reg_2 = PanelOLS(test_data['icc_avg'] * 100,
                     test_data[['green_score', 'GreenScore*GreenCountry'] + ['beta_252d', 'ebit_at', 'netdebt_at', 'log_be']],
                     entity_effects=False, time_effects=True,
                     other_effects=None, drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", cluster_entity=False, cluster_time=False, clusters=test_data[['date', 'gics']])
    print(reg_2)

    # Alternative regression
    # test_data['GreenScore*CCPI'] = test_data['green_score'] * test_data['ccpi_index']
    # reg_3 = PanelOLS(test_data['icc_avg'] * 100,
    #                 test_data[['green_score', 'GreenScore*CCPI'] + ['beta_252d', 'ebit_at', 'netdebt_at', 'log_be']],
    #                 entity_effects=False, time_effects=True,
    #                 other_effects=None, drop_absorbed=True, check_rank=False).fit(
    #     cov_type="clustered", cluster_entity=False, cluster_time=False, clusters=test_data[['date', 'gics']])
    # print(reg_3)

    # comparison = compare({'(1)': reg_1, '(2)': reg_2})
    # print(comparison)
    # # Write the LaTeX code to a file
    # with open(OUTPUT_TABLES_GLOBAL / "ccpi_reg_new.tex",
    #           'w') as file:
    #     file.write(comparison.summary.as_latex())

    # Change to nicer formating

    # --- build the table ---
    stargazer = Stargazer([reg_1, reg_2])

    #stargazer.title("Global cost of capital and CCPI")
    #stargazer.custom_columns(["ICC","ICC"], [1,1])
    stargazer.significant_digits(2)

    # only show these two covariates, in this order:
    stargazer.covariate_order(["green_score","GreenScore*GreenCountry"])

    # rename them for display
    stargazer.rename_covariates({
        "green_score":               "$s_t^n$",
        "GreenScore*GreenCountry":   "$s_t^n \\times 1_{(\\text{CCPI}_{c,t}>\\text{median}_t)}$"
    })

    # hide the built-in DOF note and the significance footnote
    stargazer.show_degrees_of_freedom(False)
    #stargazer.append_notes(False)

    # now add exactly the summary rows you want:
    stargazer.add_line("\\hline", ["",""])
    stargazer.add_line(
        "$N$",
        [
        f"{int(reg_1.nobs):,}",
        f"{int(reg_2.nobs):,}"
        ]
    )
    stargazer.add_line(
        "$R$$^2$",
        [
        f"{reg_1.rsquared_overall:.0%}".replace("%", r"\%"),
        f"{reg_2.rsquared_overall:.0%}".replace("%", r"\%")
        ]
    )
    stargazer.add_line("\\hline", ["",""])
    stargazer.add_line("Country-Time FE", ["Yes","Yes"])
    stargazer.add_line("Controls",        ["Yes","Yes"])
    stargazer.add_line("\\bottomrule", ["",""])

    # overwrite the std‐errors with the t‐statistics
    for md, reg in zip(stargazer.model_data, [reg_1, reg_2]):
        for cov in md['cov_names']:
            # reg.tstats[cov] is the t‐stat for that covariate
            md['cov_std_err'][cov] = reg.tstats[cov]

    # render the LaTeX
    latex = stargazer.render_latex()

    # remove any leftover summary‐stat lines
    patterns = [
        r".*F Statistic.*\n",
        r".*Observations.*\n",
        r".*Residual Std\. Error.*\n",
        r".*N\. of groups.*\n",         # your “N. of groups” line
        r".*\$R\^2\$.*\n",              # any literal $R^2$ row
        r".*\\textit{Note:}.*\n"    # extra hline with spacing
    ]
    for pat in patterns:
        latex = re.sub(pat, "", latex)

    # this will delete any occurrence of $^{*}$, $^{**}$ or $^{***}$
    latex = re.sub(
        r"\$\^\{\*{1,3}\}\$",
        "",
        latex
    )

    # remove the three \hline lines immediately before \end{table}
    latex = re.sub(
        r"(?:^\\hline.*\n){3}(?=\s*\\end\{tabular\})",
        "",
        latex,
        flags=re.MULTILINE
    )

    # remove \begin{table}[…]
    latex = re.sub(r"(?m)^\\begin\{table\}.*\n", "", latex)

    # remove \caption{…}
    latex = re.sub(r"(?m)^\\caption\{.*?\}\n",    "", latex)

    # table end
    latex = re.sub(r"\n\\end\{table\}", "", latex)

    # Replace hline ends
    latex = re.sub(r"\\hline &  &  \\\\\n", r"\\hline\\addlinespace% \n", latex)
    # Replace bottomrule ends
    latex = re.sub(r"\\bottomrule &  &  \\\\\n", r"\\bottomrule\\addlinespace% \n", latex)
    # Replace icc_avg with ICC
    latex = re.sub(r"\\textit{Dependent variable: icc_avg}", r"$\\text{ICC}^n_{t}$", latex)

    print(latex)
    # Write the LaTeX code to a file
    with open(OUTPUT_TABLES_GLOBAL / "ccpi_reg_new.tex",
                'w') as file:
        file.write(latex)
    
    # Return numeric output for further analysis
    return pd.Series([reg_1.params['green_score'], reg_1.std_errors['green_score'],
                      reg_2.params['green_score'], reg_2.std_errors['green_score'],
                      reg_2.params['GreenScore*GreenCountry'], reg_2.std_errors['GreenScore*GreenCountry']])
green_country_summary = icc_country_ccpi()
green_country_summary.to_pickle(DATA_INTERIM / 'green_country_summary.pkl')

plt.close("all")
# Hoberg-Phillips similarity scores: closest match (clustering by firm-firm pair, not industry)
print('Robustness...')
def hoberg_phillips():

    #loading HB data
    zf = zipfile.ZipFile(DATA_RAW / "hoberg_phillips" / "tnic3_data.zip")
    hb = pd.read_csv(zf.open('tnic3_data.txt', 'r'), sep="\t")
    zf.close()

    hb = hb.dropna() #drop if any data point missing

    #sorting by max similarity for each firm in each year
    idx = hb.groupby(['gvkey1', 'year'])['score'].idxmax()

    # Use the index to filter the DataFrame
    hb = hb.loc[idx]

    #merging info on gvkey1 and gvkey2 (using year in first to get all 12 months in a year, then second merge is using this date)
    hb = hb.merge(df[['gvkey', 'date', 'year', 'icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'green_score', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']], how='left', left_on=['gvkey1', 'year'], right_on=['gvkey', 'year'])
    hb = hb.merge(df[['gvkey', 'date', 'year', 'icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'green_score', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']], how='left', left_on=['gvkey2', 'date'], right_on=['gvkey', 'date'])

    hb['firm_firm'] = hb['gvkey1'].astype(str) + hb['gvkey2'].astype(str)

    ###could be double-counting observations if the highest similarity scores between two firms are the same (both are each others' max)

    reg1_coef = pd.DataFrame(columns=['icc_avg_diff', 'icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff'], index=['green_score_diff'])
    reg1_stde = pd.DataFrame(columns=['icc_avg_diff', 'icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff'], index=['green_score_diff'])

    reg2_coef = pd.DataFrame(columns=['icc_avg_diff', 'icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff'], index=['green_score_diff'])
    reg2_stde = pd.DataFrame(columns=['icc_avg_diff', 'icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff'], index=['green_score_diff'])

    for i in ['icc_avg', 'icc_gls', 'icc_ct', 'icc_oj', 'icc_peg', 'green_score', 'beta_252d', 'ebit_at', 'netdebt_at', 'log_be']:
        hb[i+"_diff"] = hb[i+"_x"] - hb[i+"_y"]

    hb = hb.dropna(subset=['icc_avg_diff', 'green_score_diff', 'beta_252d_diff', 'ebit_at_diff', 'netdebt_at_diff', 'log_be_diff'], how='any') #drop if any data point missing

    hb = hb.set_index(['firm_firm', 'date'])

    for i in reg1_coef.columns:
        reg = PanelOLS(hb[i]*100, hb[
            ['green_score_diff']],
                       entity_effects=False, time_effects=True, drop_absorbed=True,
                       check_rank=False).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        reg1_coef.loc['green_score_diff', i] = reg.params.loc['green_score_diff']
        reg1_stde.loc['green_score_diff', i] = reg.std_errors.loc['green_score_diff']

        reg = PanelOLS(hb[i]*100, hb[
            ['green_score_diff', 'beta_252d_diff', 'ebit_at_diff', 'netdebt_at_diff', 'log_be_diff']],
                       entity_effects=False, time_effects=True, drop_absorbed=True,
                       check_rank=False).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        reg2_coef.loc['green_score_diff', i] = reg.params.loc['green_score_diff']
        reg2_stde.loc['green_score_diff', i] = reg.std_errors.loc['green_score_diff']

    ### errorplot - ICC

    fig = plt.figure(figsize=(7, 3.5))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])

    data_labels = [r'${\Delta}$ICC']

    ax5 = fig.add_subplot(gs[0])
    data_list = np.squeeze(reg1_coef[['icc_avg_diff']])
    standard_errors = np.squeeze(reg1_stde[['icc_avg_diff']])

    positions = [1]  # [1, 2, 3, 4]
    ax5.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    ax5.set_title('Time FE')
    ax5.set_yticks(positions)
    ax5.set_yticklabels(data_labels)
    ax5.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax6 = fig.add_subplot(222)
    data_list = np.squeeze(reg2_coef[['icc_avg_diff']])
    standard_errors = np.squeeze(reg2_stde[['icc_avg_diff']])

    ax6.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax6.set_title('Controls + Time FE')
    ax6.set_yticks([])
    ax6.axvline(0, color='black', alpha=0.5, linewidth=1)

    ax5.set_xlim(-30, 10)
    ax6.set_xlim(-30, 10)

    data_labels = [r'${\Delta}$ICC$^{GLS}$', r'${\Delta}$ICC$^{CT}$', r'${\Delta}$ICC$^{OJ}$', r'${\Delta}$ICC$^{PEG}$'][::-1]

    ax1 = fig.add_subplot(223)
    data_list = np.squeeze(reg1_coef[['icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff']].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        reg1_stde[['icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff']].transpose())[::-1]

    positions = range(1, len(data_list) + 1)  # [1, 2, 3, 4]
    ax1.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')

    # ax1.set_title('Time FE')
    ax1.set_yticks(positions)
    ax1.set_yticklabels(data_labels)
    ax1.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax1.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax2 = fig.add_subplot(224)
    data_list = np.squeeze(reg2_coef[['icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff']].transpose())[
                ::-1]
    standard_errors = np.squeeze(
        reg2_stde[['icc_gls_diff', 'icc_ct_diff', 'icc_oj_diff', 'icc_peg_diff']].transpose())[::-1]

    ax2.errorbar(data_list, positions, xerr=1.96 * standard_errors, fmt='o', color='green', capsize=5, linestyle='None')
    ax2.set_yticks([])
    ax2.axvline(0, color='black', alpha=0.5, linewidth=1)
    # ax2.axvline(data_list.mean(), color='orange', linewidth=1.5, linestyle='dotted')

    ax1.set_xlim(-45, 10)
    ax2.set_xlim(-45, 10)

    ax1.set_ylim(0.5, 4.5)
    ax2.set_ylim(0.5, 4.5)

    ax = fig.add_subplot(111, frame_on=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.grid(False)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2)
    plt.savefig(OUTPUT_FIGURES_US / "icc_hb.pdf", format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close()

    print("with time-FE")
    print(reg1_coef)

    print("with controls and time-FE")
    print(reg2_coef)

    return None
hoberg_phillips()

plt.close("all")
# figure 1 panel a
def fig_1_a_old():
    # plotting US VW CAP industry-agnostic and -neutral cumulative returns since sample start (Aug-2009)
    with mpl.rc_context(rc={'font.size': 14}): #to make the font size bigger
        plt.figure(figsize=(12, 4))
        # vwcap industry agnostic
        temp_data = df_ret_vwcap.copy()
        cols_to_replace = temp_data.loc['2009-08-31'].notna()

        # Replace values at '2009-08' with 0 in those columns
        temp_data.loc['2009-08-31', cols_to_replace] = 0

        # plotting individual factors (plotting last separately in order to add data label in legend)
        for i in to_test[:-1]:
            plt.plot(temp_data[temp_data.index>='2009-08-31'][i].cumsum(), linewidth=0.75, color='tab:grey', alpha=0.5)
        plt.plot(temp_data[temp_data.index >= '2009-08-31'][to_test[-1:]].cumsum(), linewidth=0.75, color='tab:grey', alpha=0.5, label='GMB')

        # industry neutral
        temp_data = df_ret_vwcap_indn.copy()  # vwcap industry agnostic
        cols_to_replace = temp_data.loc['2009-08-31'].notna()

        # Replace values at '2009-4' with 0 in those columns
        temp_data.loc['2009-08-31', cols_to_replace] = 0
        # plotting individual factors (plotting last separately in order to add data label in legend)
        for i in to_test[:-1]:
            plt.plot(temp_data[temp_data.index>='2009-08-31'][i].cumsum(), linewidth=0.75, color='tab:grey', alpha=0.5, linestyle='dotted')
        plt.plot(temp_data[temp_data.index >= '2009-08-31'][to_test[-1:]].cumsum(), linewidth=0.75, color='tab:grey', alpha=0.5, linestyle='dotted', label='GMB (industry-neutral)')

        plt.axhline(0, alpha=0.8, color='black', linewidth=1)
        plt.legend(fontsize=14, framealpha=1, loc='upper left')
        plt.xlim(pd.to_datetime('2009-08-31'), pd.to_datetime('2022-12-31'))
        plt.xticks(['2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01'],
                   ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'])
        plt.tight_layout()
        plt.ylabel("Cumulative sum (%)")
        plt.ylim(-65, 65)
        plt.savefig(OUTPUT_FIGURES / "fig1_a.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()

    return None
fig_1_a_old()

# Realized return of GMB portfolios in the US
def plot_gmb_ret_us():
    # plotting US VW CAP cumulative returns since sample start (Aug-2009)
    with mpl.rc_context(rc={'font.size': 14}):
        start = '2009-08-31'
        end = '2022-12-31'
        tick_locs = pd.to_datetime([
            '2010-01-01','2011-01-01','2012-01-01','2013-01-01','2014-01-01',
            '2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01',
            '2020-01-01','2021-01-01','2022-01-01'
        ])
        tick_labels = ['2010','2011','2012','2013','2014','2015','2016',
                    '2017','2018','2019','2020','2021','2022']

        # -------- Figure 1: Raw (industry-agnostic) --------
        plt.figure(figsize=(12, 4))
        temp_data = df_ret_vwcap.copy()
        cols_to_replace = temp_data.loc[start].notna()
        temp_data.loc[start, cols_to_replace] = 0

        for i in to_test[:-1]:
            plt.plot(temp_data[temp_data.index >= start][i].cumsum(),
                    linewidth=1, color='tab:grey', alpha=1)
        plt.plot(temp_data[temp_data.index >= start][to_test[-1:]].cumsum(),
                linewidth=1, color='tab:grey', alpha=1, label='GMB')

        plt.axhline(0, alpha=0.8, color='black', linewidth=1)
        plt.legend(fontsize=14, framealpha=1, loc='upper left')
        plt.xlim(pd.to_datetime(start), pd.to_datetime(end))
        plt.xticks(tick_locs, tick_labels)
        plt.ylabel("Cumulative sum (%)")
        plt.ylim(-65, 65)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_US / "us_gmb_ret.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        # -------- Figure 2: Industry-neutral (industry-adjusted) --------
        plt.figure(figsize=(12, 4))
        temp_data = df_ret_vwcap_indn.copy()
        cols_to_replace = temp_data.loc[start].notna()
        temp_data.loc[start, cols_to_replace] = 0

        for i in to_test[:-1]:
            plt.plot(temp_data[temp_data.index >= start][i].cumsum(),
                    linewidth=1, color='tab:grey', alpha=1, linestyle='solid')
        plt.plot(temp_data[temp_data.index >= start][to_test[-1:]].cumsum(),
                linewidth=1, color='tab:grey', alpha=1, linestyle='solid',
                label='GMB (industry-neutral)')

        plt.axhline(0, alpha=0.8, color='black', linewidth=1)
        plt.legend(fontsize=14, framealpha=1, loc='upper left')
        plt.xlim(pd.to_datetime(start), pd.to_datetime(end))
        plt.xticks(tick_locs, tick_labels)
        plt.ylabel("Cumulative sum (%)")
        plt.ylim(-65, 65)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_US / "us_gmb_ret_industry.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    return None
plot_gmb_ret_us()

# figure 1 panel b-c
def fig_1_bc_old():
    # Panel b: Global (baseline); US (baseline); Brown Countries; Green Countries; Global Start Sample; Global End Sample
    # icc_timetrend_base, icc_timetrend_trend = icc_timetrend()
    # green_country_summary = icc_country_ccpi()

    plt.close("all")
    categories = ['Global\n(Baseline)', 'US\n(Baseline)', 'Brown\nCountries', 'Green\nCountries', 'Global\nStart Sample', 'Global\nEnd Sample']

    values = [-green_country_summary.iloc[0],
              -us_1.loc['icc_avg', 'Greenium'],
              -green_country_summary.iloc[2],
              -green_country_summary.iloc[4],
              -icc_timetrend_baseline.loc['icc_avg', 'slope'],
              -icc_timetrend_trend.loc['icc_avg', 'slope']]
    errors = [1.96 * green_country_summary.iloc[1],
              1.96 * us_1.loc['icc_avg', 'S.E. (Greenium)'],
              1.96 * green_country_summary.iloc[3],
              1.96 * green_country_summary.iloc[5],
              1.96 * icc_timetrend_baseline.loc['icc_avg', 'std_error'],
              1.96 * icc_timetrend_trend.loc['icc_avg', 'std_error']]

    # Creating the bar chart
    with mpl.rc_context(rc={'font.size': 14}):
        plt.figure(figsize=(12, 4))

        # Plotting the first bar for Category A separately
        plt.bar(categories[0], values[0], yerr=errors[0], capsize=5, color='skyblue') # US
        plt.bar(categories[1], values[1], yerr=errors[1], capsize=5, color='skyblue') # Global

        # Plotting the stacked bars for Category B and Category C
        plt.bar(categories[2], values[2], yerr=errors[2], capsize=5, color='slategrey') #Brown countries
        plt.bar(categories[3], values[2], yerr=None, capsize=5, color='slategrey', alpha=0.2)  # Brown countries base
        plt.bar(categories[3], values[3], bottom=values[2], yerr=errors[3], capsize=5, color='slategrey') #Green countries on top

        # Plotting the first bar for Category A separately
        plt.bar(categories[4], values[4], yerr=errors[4], capsize=5, color='orange') #Baseline
        plt.bar(categories[5], values[4], yerr=None, capsize=5, color='orange', alpha=0.2)  #Shaded baseline
        plt.bar(categories[5], values[5], bottom=values[4], yerr=errors[5], capsize=5, color='orange') #End of sample

        # Adding a vertical line to separate the two sections
        plt.axvline(x=1.5, color='grey', linestyle='--')
        plt.axvline(x=3.5, color='grey', linestyle='--')
        plt.axhline(0, color='black', linewidth=1)

        #plt.yticks(ticks=plt.gca().get_yticks(), labels=[-abs(int(tick)) for tick in plt.gca().get_yticks()])
        plt.yticks(ticks=plt.gca().get_yticks(), labels=[-(int(tick)) for tick in plt.gca().get_yticks()])
        plt.ylabel('Greenium (annual bps)')
        #plt.ylim(0, 40)
        # plt.xticks(rotation=45)  # Rotate category labels for better readability
        plt.tight_layout()  # Adjust the layout to fit everything nicely
        plt.savefig(OUTPUT_FIGURES / "fig1_b_old.pdf",
                    format="pdf",
                    bbox_inches="tight")
        plt.show()
        plt.close()

    # Panel c: equity greenium robustness and other asses classes (all US) : E/P, options, Subj, bonds, wacc, sov bond
    plt.close("all")
    categories = ['E/P\n(Latest)', 'Exp. Return\n(Options)',
                  'Subj. Req. Return\n(MS and VL)', 'Corp. Bonds\n(US)', 'WACC\n(US)',
                  'Sov. Bonds\n(DK & DE)']

    # plotting greenium estimates
    values = [-us_1.loc['ep_fwd0', 'Greenium'],
              -us_1.loc['coc_avg_options', 'Greenium'],
              -us_1.loc['subj_rr_avg', 'Greenium'],
              -bond_1.loc['yield_spread_adj', 'Greenium'],
              -us_1.loc['wacc_icc_avg', 'Greenium'],
              3.235] # from separate excel file
    # plotting standard errors
    errors = [1.96 * us_1.loc['ep_fwd0', 'S.E. (Greenium)'],
              1.96 * us_1.loc['coc_avg_options', 'S.E. (Greenium)'],
              1.96 * us_1.loc['subj_rr_avg', 'S.E. (Greenium)'],
              1.96 * bond_1.loc['yield_spread_adj', 'S.E. (Greenium)'],
              1.96 * us_1.loc['wacc_icc_avg', 'S.E. (Greenium)'],
              1.96 * 0.6510]  # from separate excel file

    # Creating the bar chart
    with mpl.rc_context(rc={'font.size': 14}):
        plt.figure(figsize=(12, 4))

        # Plotting the first bar for Category A separately
        plt.bar(categories[0], values[0], yerr=errors[0], capsize=5, color='steelblue')
        plt.bar(categories[1], values[1], yerr=errors[1], capsize=5, color='steelblue')
        plt.bar(categories[2], values[2], yerr=errors[2], capsize=5, color='steelblue')
        plt.bar(categories[3], values[3], yerr=errors[3], capsize=5, color='chocolate')
        plt.bar(categories[4], values[4], yerr=errors[4], capsize=5, color='chocolate')
        plt.bar(categories[5], values[5], yerr=errors[5], capsize=5, color='chocolate')
        plt.axvline(x=2.5, color='grey', linestyle='--')
        plt.axhline(0, color='black', linewidth=1)
        plt.yticks(ticks=plt.gca().get_yticks(), labels=[-(int(tick)) for tick in plt.gca().get_yticks()])
        plt.ylabel('Greenium (annual bps)')
        plt.tight_layout()  # Adjust the layout to fit everything nicely
        plt.savefig(OUTPUT_FIGURES / "fig1_c_old.pdf",
                    format="pdf",
                    bbox_inches="tight")
        plt.show()
        plt.close()

    return None
fig_1_bc_old()

def fig_1_b():
    plt.close("all")
    categories = ['Global\n(Baseline)', 
                  'Brown\nCountries', 'Green\nCountries', 
                  'Global\nStart Sample', 'Global\nEnd Sample',
                  'Corp. Bonds\n(US)', 'WACC\n(US)']

    values = [-green_country_summary.iloc[0],
              -green_country_summary.iloc[2],
              -green_country_summary.iloc[4],
              -icc_timetrend_baseline.loc['icc_avg', 'slope'],
              -icc_timetrend_trend.loc['icc_avg', 'slope'],
              -bond_1.loc['yield_spread_adj', 'Greenium'],
              -us_1.loc['wacc_icc_avg', 'Greenium']]
    errors = [1.96 * green_country_summary.iloc[1],
              1.96 * green_country_summary.iloc[3],
              1.96 * green_country_summary.iloc[5],
              1.96 * icc_timetrend_baseline.loc['icc_avg', 'std_error'],
              1.96 * icc_timetrend_trend.loc['icc_avg', 'std_error'],
              1.96 * bond_1.loc['yield_spread_adj', 'S.E. (Greenium)'],
              1.96 * us_1.loc['wacc_icc_avg', 'S.E. (Greenium)']]

    # Creating the bar chart
    with mpl.rc_context(rc={'font.size': 14}):
        plt.figure(figsize=(12, 4))

        # Main estimate
        plt.bar(categories[0], values[0], yerr=errors[0], capsize=5, color='skyblue') # Global

        # Green versus brown countries
        plt.bar(categories[1], values[1], yerr=errors[1], capsize=5, color='slategrey') #Brown countries
        plt.bar(categories[2], values[1], yerr=None, capsize=5, color='slategrey', alpha=0.2)  # Brown countries base
        plt.bar(categories[2], values[2], bottom=values[1], yerr=errors[2], capsize=5, color='slategrey') #Green countries on top

        # Over time 
        plt.bar(categories[3], values[3], yerr=errors[4], capsize=5, color='orange') #Baseline
        plt.bar(categories[4], values[3], yerr=None, capsize=5, color='orange', alpha=0.2)  #Shaded baseline
        plt.bar(categories[4], values[4], bottom=values[3], yerr=errors[4], capsize=5, color='orange') #End of sample

        # Corporate bonds and WACC
        plt.bar(categories[5], values[5], yerr=errors[5], capsize=5, color='chocolate')
        plt.bar(categories[6], values[6], yerr=errors[6], capsize=5, color='chocolate')

        # Adding a vertical line to separate the three sections
        plt.axvline(x=0.5, color='grey', linestyle='--')
        plt.axvline(x=2.5, color='grey', linestyle='--')
        plt.axvline(x=4.5, color='grey', linestyle='--')
        plt.axhline(0, color='black', linewidth=1)

        #plt.yticks(ticks=plt.gca().get_yticks(), labels=[-abs(int(tick)) for tick in plt.gca().get_yticks()])
        plt.yticks(ticks=plt.gca().get_yticks(), labels=[-(int(tick)) for tick in plt.gca().get_yticks()])
        plt.ylabel('Greenium (annual bps)')
        #plt.ylim(0, 40)
        # plt.xticks(rotation=45)  # Rotate category labels for better readability
        plt.tight_layout()  # Adjust the layout to fit everything nicely
        plt.savefig(OUTPUT_FIGURES / "fig1_b.pdf",
                    format="pdf",
                    bbox_inches="tight")
        plt.show()
        plt.close()
    return None
fig_1_b()

plt.close("all")
# PST GMB vs. MCCC
print('Some pure replications...')
def gmb_extension():
    # importing L/S factor returns from  OpenAssetPricing
    zf = zipfile.ZipFile(DATA_RAW / "ac_factors" / "PredictorAltPorts_QuintilesVW.zip")
    df_signal_ret = pd.read_csv(zf.open('PredictorAltPorts_QuintilesVW.csv', 'r'),
                                usecols=['date', 'signalname', 'port', 'ret'])
    zf.close()

    # extracting two main factors
    df_signal_ret = df_signal_ret[df_signal_ret.signalname.isin(['AnnouncementReturn', 'AnalystRevision'])]

    # keeping only L/S portfolios
    df_signal_ret = df_signal_ret[df_signal_ret.port == 'LS']

    # creating datetime index
    df_signal_ret = df_signal_ret.set_index(pd.to_datetime(df_signal_ret['date'], format='%Y-%m-%d'))
    df_signal_ret.index = df_signal_ret.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_signal_ret = df_signal_ret.drop(['date'], axis=1)
    df_signal_ret = df_signal_ret.reset_index()
    df_signal_ret = df_signal_ret.drop('port', axis=1)
    df_signal_ret = df_signal_ret.pivot(index='date', columns='signalname', values='ret')

    # loading MCCC (media climate change concerns index)
    df_mccc = pd.read_excel(DATA_RAW / "mccc" / "Sentometrics_US_Media_Climate_Change_Index.xlsx",
                            sheet_name='2023 update monthly', skiprows=6, usecols=['Date', 'Aggregate'])
    df_mccc = df_mccc.rename(columns={'Date': 'date', 'Aggregate': 'mccc'})

    df_mccc = df_mccc.set_index('date')
    df_mccc.index = df_mccc.index.to_period('M').to_timestamp('M')
    df_mccc = df_mccc.reset_index()

    # estimating rolling 36 month AR(1) shocks to MCCC
    def estimate_ar1(data):
        Y = data[1:]
        X = data[:-1]
        X = sm.add_constant(X)  # adds a constant (intercept) to the predictor
        model = sm.OLS(Y, X).fit()
        return model.params[0], model.params[1]  # intercept, slope

    # Initialize lists to store results
    intercepts = []
    slopes = []

    window_size = 36
    for start in range(0, len(df_mccc) - window_size + 1):
        window_data = df_mccc['mccc'].iloc[start:start + window_size].values
        intercept, slope = estimate_ar1(window_data)
        intercepts.append(intercept)
        slopes.append(slope)

    # Store results in the original dataframe (aligned with the end of each window)
    df_mccc['intercept'] = np.nan
    df_mccc['slope'] = np.nan
    df_mccc.loc[df_mccc.index[window_size - 1:], 'intercept'] = intercepts
    df_mccc.loc[df_mccc.index[window_size - 1:], 'slope'] = slopes

    df_mccc['mccc_pred'] = df_mccc['intercept'] + df_mccc['mccc'] * df_mccc['slope']

    df_mccc[['mccc_pred']] = df_mccc[['mccc_pred']].shift(1)
    df_mccc['mccc_pred_error'] = df_mccc['mccc'] - df_mccc['mccc_pred']
    df_mccc = df_mccc.set_index('date')
    df_mccc = df_mccc.rename(columns={'mccc_pred_error': 'mccc_shock'})
    df_mccc['mccc_shock_lag'] = df_mccc['mccc_shock'].shift(1)
    df_mccc = df_mccc[['mccc_shock', 'mccc_shock_lag']]

    # replicating PST table 4 (LHS using excess returns)
    Y = df_ret_vw['greenness'].dropna()
    X = sm.add_constant(df_ff6[df_ff6.index.isin(Y.index)][['Mkt-RF', 'SMB', 'HML']])
    X = X.merge(df_mccc, how='left', left_index=True, right_index=True)
    X = X.dropna() #MCCC only until Aug-2022
    Y = Y[Y.index.isin(X.index)]
    X = X.merge(df_signal_ret, how='left', left_index=True, right_index=True) # adding signal factors to X variables

    # full time-series estimates
    results = sm.OLS(Y, X[['const', 'Mkt-RF', 'SMB', 'HML']]).fit()
    Y_alpha_full = results.resid + results.params.iloc[0]  # residual + alpha = new Y

    # Nov-2012 to Jun 2018 (PST)
    Y_pst = Y[(Y.index >= '2012-11-01') & (Y.index <= '2018-06-30')]
    X_pst = X[X.index.isin(Y_pst.index)]
    results = sm.OLS(Y_pst, X_pst[['const', 'Mkt-RF', 'SMB', 'HML']]).fit()
    Y_alpha_pst = results.resid + results.params.iloc[0]  # residual + alpha = new Y

    # Nov-2012 to Aug-2022
    Y_pstext = Y[(Y.index >= '2012-11-01')]
    X_pstext = X[X.index.isin(Y_pstext.index)]
    results = sm.OLS(Y_pstext, X_pstext[['const', 'Mkt-RF', 'SMB', 'HML']]).fit()
    Y_alpha_pstext = results.resid + results.params.iloc[0]  # residual + alpha = new Y

    # Running regressions using excess returns as LHS
    reg1 = sm.OLS(Y_pst, X_pst[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg2 = sm.OLS(Y_pst, X_pst[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')
    reg3 = sm.OLS(Y_pstext, X_pstext[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg4 = sm.OLS(Y_pstext, X_pstext[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')
    reg5 = sm.OLS(Y, X[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg6 = sm.OLS(Y, X[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')

    # Running regressions using FF3 alpha as LHS
    reg7 = sm.OLS(Y_alpha_pst, X_pst[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg8 = sm.OLS(Y_alpha_pst, X_pst[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')
    reg9 = sm.OLS(Y_alpha_pstext, X_pstext[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg10 = sm.OLS(Y_alpha_pstext, X_pstext[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')
    reg11 = sm.OLS(Y_alpha_full, X[['const', 'mccc_shock', 'mccc_shock_lag']]).fit(cov_type='HC3')
    reg12 = sm.OLS(Y_alpha_full, X[['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn', 'AnalystRevision']]).fit(cov_type='HC3')

    # outputting
    dfoutput = summary_col([reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12], stars=True,
                           regressor_order=['const', 'mccc_shock', 'mccc_shock_lag', 'AnnouncementReturn',
                                            'AnalystRevision'],
                           model_names=['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)', '(11)', '(12)'],
                           info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))}, float_format='%.2f')
    dfoutput.add_title('GMB ExcRet (1-6) / GMB Alpha (7-12): 2012-2018 (1-2 & 7-8) / 2012-2022 (3-4 & 9-1) / 2007-2022 (5-6 & 11-12)')
    print(dfoutput)

    test = dfoutput.as_latex()
    text_file = open(OUTPUT_TABLES_US / "pst_table4_extended.tex", "w")
    text_file.write(test)
    text_file.close()

    return df_mccc
df_mccc = gmb_extension()

# Zhang and PST replication
def replication_tables():
    ################## PST replication ##################
    Y = df_ret_vw['greenness'].dropna()
    X = df_ff6[df_ff6.index.isin(Y.index)]
    X = sm.add_constant(X)

    Y_1 = Y[(Y.index >= '2012-11-01') & (Y.index <= '2020-12-31')]
    X_1 = X[(X.index >= '2012-11-01') & (X.index <= '2020-12-31')]

    Y_2 = Y[(Y.index >= '2012-11-01')]
    X_2 = X[(X.index >= '2012-11-01')]

    # regression functions - first PST sample period
    reg1 = sm.OLS(Y_1, X_1[['const']]).fit(cov_type='HC3')
    reg2 = sm.OLS(Y_1, X_1[['const', 'Mkt-RF']]).fit(cov_type='HC3')
    reg3 = sm.OLS(Y_1, X_1[['const', 'Mkt-RF', 'SMB', 'HML']]).fit(cov_type='HC3')
    reg4 = sm.OLS(Y_1, X_1[['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]).fit(cov_type='HC3')

    # Extending PST to 2022
    reg5 = sm.OLS(Y_2, X_2[['const']]).fit(cov_type='HC3')
    reg6 = sm.OLS(Y_2, X_2[['const', 'Mkt-RF']]).fit(cov_type='HC3')
    reg7 = sm.OLS(Y_2, X_2[['const', 'Mkt-RF', 'SMB', 'HML']]).fit(cov_type='HC3')
    reg8 = sm.OLS(Y_2, X_2[['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]).fit(cov_type='HC3')

    # Full sample period
    reg9 = sm.OLS(Y, X[['const']]).fit(cov_type='HC3')
    reg10 = sm.OLS(Y, X[['const', 'Mkt-RF']]).fit(cov_type='HC3')
    reg11 = sm.OLS(Y, X[['const', 'Mkt-RF', 'SMB', 'HML']]).fit(cov_type='HC3')
    reg12 = sm.OLS(Y, X[['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]).fit(cov_type='HC3')

    dfoutput = summary_col([reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12], stars=True,
                           regressor_order=['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'],
                           model_names=['Exc. Ret.\n(1)', 'CAPM\n(2)', 'FF3\n(3)', 'FF6\n(4)',
                                        'Exc. Ret.\n(5)', 'CAPM\n(6)', 'FF3\n(7)', 'FF6\n(8)',
                                        'Exc. Ret.\n(9)', 'CAPM\n(10)', 'FF3\n(11)', 'FF6\n(12)'],
                           info_dict={'N': lambda x: "{0:d}".format(int(x.nobs))}, float_format='%.2f')
    dfoutput.add_title('2012-2020 (1-4) / 2012-2022 (5-8) / 2007-2022 (9-12)')
    print(dfoutput)

    test = dfoutput.as_latex()
    text_file = open(OUTPUT_TABLES_US / "gmb_factor_reg.tex", "w")
    text_file.write(test)
    text_file.close()

    # loading MCCC (media climate change concerns index)
    df_mccc = pd.read_excel(DATA_RAW / "mccc" / "Sentometrics_US_Media_Climate_Change_Index.xlsx",
                            sheet_name='2023 update monthly', skiprows=6, usecols=['Date', 'Aggregate'])
    df_mccc = df_mccc.rename(columns={'Date': 'date', 'Aggregate': 'mccc'})

    df_mccc = df_mccc.set_index('date')
    df_mccc.index = df_mccc.index.to_period('M').to_timestamp('M')
    df_mccc = df_mccc.reset_index()

    # estimating rolling 36 month AR(1) shocks to MCCC
    def estimate_ar1(data):
        Y = data[1:]
        X = data[:-1]
        X = sm.add_constant(X)  # adds a constant (intercept) to the predictor
        model = sm.OLS(Y, X).fit()
        return model.params[0], model.params[1]  # intercept, slope

    # Initialize lists to store results
    intercepts = []
    slopes = []

    window_size = 36
    for start in range(0, len(df_mccc) - window_size + 1):
        window_data = df_mccc['mccc'].iloc[start:start + window_size].values
        intercept, slope = estimate_ar1(window_data)
        intercepts.append(intercept)
        slopes.append(slope)

    # Store results in the original dataframe (aligned with the end of each window)
    df_mccc['intercept'] = np.nan
    df_mccc['slope'] = np.nan
    df_mccc.loc[df_mccc.index[window_size - 1:], 'intercept'] = intercepts
    df_mccc.loc[df_mccc.index[window_size - 1:], 'slope'] = slopes

    df_mccc['mccc_pred'] = df_mccc['intercept'] + df_mccc['mccc'] * df_mccc['slope']

    df_mccc[['mccc_pred']] = df_mccc[['mccc_pred']].shift(1)
    df_mccc['mccc_pred_error'] = df_mccc['mccc'] - df_mccc['mccc_pred']
    df_mccc = df_mccc.set_index('date')
    df_mccc = df_mccc.rename(columns={'mccc_pred_error': 'mccc_shock'})
    df_mccc['mccc_shock_lag'] = df_mccc['mccc_shock'].shift(1)
    df_mccc = df_mccc[['mccc_shock', 'mccc_shock_lag']]
    df_mccc = df_mccc[df_mccc.index>= '2012-11-01']['mccc_shock']

    # PST replication chart
    gmb_factor = df_ret_vw['greenness'].dropna()
    gmb_factor = gmb_factor[gmb_factor.index >= '2012-11-01']
    plt.plot(gmb_factor.cumsum(), color='blue', label='GMB factor')
    plt.plot(df_mccc.cumsum()*6, color='green', linestyle='dotted', label='MCCC shocks')
    plt.axvline(pd.to_datetime('2020-12-31'), label='Dec-2020', color='black', linestyle='dashed')
    plt.xlim(pd.to_datetime('2012-11-30'), pd.to_datetime('2022-12-31'))
    plt.legend()
    plt.ylabel("Cumulative returns (%)")
    plt.title('PST replication: greenness')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "pst.pdf",
                format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close("all")

    ################## Zhang replication ##################
    gmb_factor = df_ret_vw.scope1int_sales.dropna()

    plt.plot(gmb_factor.cumsum(), color='blue')
    plt.axvline(pd.to_datetime('2021-12-31'), label='Dec-2021', color='black', linestyle='dashed')
    plt.xlim(pd.to_datetime('2009-06-30'), pd.to_datetime('2022-12-31'))
    plt.legend()
    plt.ylabel("Cumulative returns (%)")
    plt.title('Zhang replication: Scope 1 intensity (sales)')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "zhang.pdf",
                format="pdf",
                bbox_inches="tight")
    plt.show()
    plt.close("all")

    # Zhang excess returns until 2022
    #constant_results = sm.OLS(gmb_factor,
    #                          sm.add_constant(df_ff6[df_ff6.index.isin(gmb_factor.index)]['Mkt-RF'])['const']).fit()
    #print(constant_results.summary())
    # Zhang excess returns until 2021
    #constant_results = sm.OLS(gmb_factor[gmb_factor.index <= '2021-12-31'], sm.add_constant(
    #    df_ff6[df_ff6.index.isin(gmb_factor[gmb_factor.index <= '2021-12-31'].index)]['Mkt-RF'])[
    #    'const']).fit()  # cov_type='HAC', cov_kwds={'maxlags': 3}
    #print(constant_results.summary())
    # Zhang ff6 alpha until 2022
    #ff6_results = sm.OLS(gmb_factor, sm.add_constant(df_ff6[df_ff6.index.isin(gmb_factor.index)])).fit()
    #print(ff6_results.summary())
    # Zhang ff6 alpha until 2021
    #ff6_results = sm.OLS(gmb_factor[gmb_factor.index <= '2021-12-31'], sm.add_constant(
    #    df_ff6[df_ff6.index.isin(gmb_factor[gmb_factor.index <= '2021-12-31'].index)])).fit()
    #print(ff6_results.summary())


    return None
replication_tables()

# print US and global mean green score spreads using VW and VW cap weights
print('Green score spreads and validation...')
def green_score_spreads(df_us, df):
    ########## US GMB green score spread ##########
    temp_data = df_us[(df_us['green_score'].notna()) & (df_us['ret_exc_lead1m'].notna())]

    # ranking to create terciles
    temp_data['esg_rank'] = temp_data.groupby(['date'], observed=False)['green_score'].rank(
        pct=True)
    temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
        lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

    # VW cap mean green score
    gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'green_score', 'me_cap']].apply(
        lambda x: (x['green_score'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
    gmb_factor = gmb_factor[3] - gmb_factor[1]
    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    print("US: VW cap")
    print(gmb_factor.describe()['mean'].round(2))

    # VW mean green score
    gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'green_score', 'me']].apply(
        lambda x: (x['green_score'] * x['me']).sum() / x['me'].sum()).unstack(level=1)
    gmb_factor = gmb_factor[3] - gmb_factor[1]
    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    print("US: VW")
    print(gmb_factor.describe()['mean'].round(2))

    ########## Global GMB green score spread ##########
    temp_data = df[(df['green_score'].notna()) & (df['ret_exc_lead1m'].notna())]

    # ranking to create terciles
    temp_data['esg_rank'] = temp_data.groupby(['date'], observed=False)['green_score'].rank(
        pct=True)
    temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
        lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

    # VW cap mean green score
    gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'green_score', 'me_cap']].apply(
        lambda x: (x['green_score'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)
    gmb_factor = gmb_factor[3] - gmb_factor[1]
    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    print("Global: VW cap")
    print(gmb_factor.describe()['mean'].round(2))

    # VW mean green score
    gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[['date', 'tercile_esg', 'green_score', 'me']].apply(
        lambda x: (x['green_score'] * x['me']).sum() / x['me'].sum()).unstack(level=1)
    gmb_factor = gmb_factor[3] - gmb_factor[1]
    gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    print("Global: VW")
    print(gmb_factor.describe()['mean'].round(2))

    return None
green_score_spreads(df[df.excntry=="USA"], df)

# 1-year esg convergence to aggregate green score
def esg_convergence():
    # leading delta esg regressed on (green_score-esg). No FE

    esg_conv = {} # dictionary to store results

    # data requirements
    test_data = df[df.date >= global_start]
    test_data = test_data[test_data.green_score.notna()]
    test_data = test_data[test_data.gics6.notna()]
    test_data = test_data.sort_values(by=['id', 'date'])
    test_data = test_data.set_index(['id', 'country_ym_num'])

    # loop through each ESG measure
    for esg_var in to_test:
        test_data['esg_mean_diff'] = test_data['green_score'] - test_data[esg_var] #difference in green score and esg variable
        test_data['esg_diff'] = test_data.groupby(level=0)[esg_var].diff(12) # 12 month change in esg variable
        test_data['esg_diff_lead'] = test_data.groupby(level=0)['esg_diff'].shift(-12) # 12 month ahead change in esg variable
        test_data_filtered = test_data.dropna(subset=['esg_diff_lead', 'esg_mean_diff'])
        Y = test_data_filtered['esg_diff_lead']
        X = test_data_filtered[['esg_mean_diff']]

        model = PanelOLS(Y, sm.add_constant(X), entity_effects=False, time_effects=False)
        results = model.fit(cov_type='clustered', cluster_entity=False, cluster_time=False, clusters=test_data_filtered[['gics', 'date']])

        # Store coefficient and standard error
        coef = results.params['esg_mean_diff']
        tstat = results.tstats['esg_mean_diff']
        esg_conv[esg_var] = (coef, tstat)

    # converting to dataframe and exporting to LaTeX
    esg_conv = pd.DataFrame.from_dict(esg_conv, orient='index', columns=['coef', 't-stat'])
    esg_conv.index = to_test_names

    # new_order = ['S1INT (Sales)', 'S1+2INT (Sales)', 'S1+2+3INT (Sales)', 'S1INT (Assets)', 'S1+2INT (Assets)',
    #              'S1+2+3INT (Assets)',
    #              'Weighted ESG score', 'Environment score', 'Total ESG score', 'Environmental score',
    #              'LOG(S1TOT)', 'LOG(S1+2TOT)', 'LOG(S1+2+3TOT)', 'Ind.-adj. ESG score',
    #              'Greenness (PST)', 'E climate score', 'E nat. res. score', 'E waste score', 'E env. opps. score',
    #              'TRI (Sales)', 'TPWINT (Sales)', 'TRI (Assets)', 'TPWINT (Assets)']  # Desired order of index names
    # esg_conv = esg_conv.reindex(new_order)

    esg_conv.columns = ['Coeff. estimate', '$t$-statistic']

    column_format = 'l' + 'c' + 'c'  # 'l' for left, 'c' for center (centering number columns in output)

    esg_conv.to_latex(OUTPUT_TABLES_DESCRIPTIVES / "esg_convergence.tex", float_format="%.2f", column_format=column_format)

    return None
esg_convergence()

# benjamini-hochberg multiple testing adjustment
# inspect = pd.concat([ret_vwcap.loc[to_test], ret_vwcap_indn.loc[to_test]], axis=0)
# inspect = inspect[['p_a_q5', 'p_a_ff6', 'p_a_ff3', 'p_a_capm', 'p_exc_ret']].melt().sort_values('value').dropna()
# inspect['factor_count'] = 1
# inspect['factor_count'] = inspect['factor_count'].cumsum()
# inspect['bh_adj'] = 0.05 * inspect['factor_count'] / len(inspect)
# inspect['sign'] = inspect['value'] <= inspect.bh_adj
# inspect = inspect.reset_index(drop=True)
# print(inspect[inspect['sign']].index.max() + 1)  # number of significant factors
# import scipy
# -scipy.stats.norm.ppf(inspect.iloc[inspect[inspect['sign']].index.max()]['bh_adj'] / 2)
# from statsmodels.stats.multitest import multipletests
# multipletests(inspect['value'], method='fdr_bh')[0]  # [1] gives adjusted pval

########
# i = 'green_score'
# temp_data = df[(df[i].notna()) & (df.excntry=="USA")]
# temp_data = temp_data[temp_data.ret_exc_lead1m.notna()]

# # Requiring minimum number of observations but otherwise using full time series available
# number_obs = temp_data.groupby(['date'], observed=False)[i].count().reset_index()
# number_obs = number_obs[number_obs[i] >= 100]  # at least 100 observations
# temp_data = temp_data[temp_data.date.isin(number_obs.date)]

# # vw
# temp_data['esg_rank'] = temp_data.groupby(['date'], observed=False)[i].rank(pct=True)
# temp_data['tercile_esg'] = temp_data.groupby(['date'], observed=False)['esg_rank'].transform(
#     lambda x: pd.cut(x, 3, retbins=False, labels=[1, 2, 3]))

# # Exclude grouping columns when applying the function
# gmb_factor = temp_data.groupby(['date', 'tercile_esg'], observed=False)[
#     ['date', 'tercile_esg', 'ret_exc_lead1m', 'me_cap']].apply(
#     lambda x: (x['ret_exc_lead1m'] * x['me_cap']).sum() / x['me_cap'].sum()).unstack(level=1)

# gmb_factor = gmb_factor[3] - gmb_factor[1]

# # Off-setting index by 1 month due to returns being 1 month forward
# gmb_factor.index = gmb_factor.index + pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
# gmb_factor.describe()
# 1.49052*12**0.5

# ####
# test_1_10 = df[df.q_1_10==1].groupby(['date'], observed=False)[['date', 'green_score', 'me_cap']].apply(
#     lambda x: (x['green_score'] * x['me_cap']).sum() / x['me_cap'].sum())
# test_10_10 = df[df.q_10_10==1].groupby(['date'], observed=False)[['date', 'green_score', 'me_cap']].apply(
#     lambda x: (x['green_score'] * x['me_cap']).sum() / x['me_cap'].sum())
# test_1_10.mean()
# test_10_10.mean()

