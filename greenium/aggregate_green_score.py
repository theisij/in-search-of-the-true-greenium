# importing packages and paths
import time
from datetime import date

import ibis
import ibis.selectors as s
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from ibis import _

# register the operator on pd.DataFrame
from janitor.functions import conditional_join  # noqa: F401
from scipy.optimize import minimize
from tqdm import trange

from greenium.config import DATA_RAW


# Create fund info ------------
def load_fund_info( 
    morningstar_start: date,
    impute_start: bool,
    impute_start_date: date
) -> pl.DataFrame:
    """
    Reads and cleans the Morningstar sustainable funds CSV.
    
    - Selects & renames columns
    - Builds EOM as the last day of each month
    - Cleans fund_size
    - Fills missing or early sus_fund_start with impute_start
    """
    fund_info = pl.read_csv(DATA_RAW / "morningstar" / "Sustainable_Funds_Processed.csv")

    # Select and rename columns
    fund_info = fund_info.select([
        pl.col("secid"),
        pl.col("ym"),
        pl.col("fund_size"),
        pl.col("domicile"),
        pl.col("sustainable_investment_overall").alias("sys_fund"),
        pl.col("sustainable_investment_overall_start_date").alias("sus_fund_start")
    ])

    # Create EOM column
    fund_info = (
        fund_info
        .with_columns(
            eom=(
                pl.concat_str([
                    pl.col("ym").str.slice(0, 4),
                    pl.lit("-"),
                    pl.col("ym").str.slice(5, 2),
                    pl.lit("-01")
                ])
                .str.strptime(pl.Date, format="%Y-%m-%d")
                .dt.month_end()
            )
        )
        .drop("ym")
    )

    # Remove commas from fund_size and convert to numeric
    fund_info = (
        fund_info
        .with_columns(
            fund_size = (
                pl.col("fund_size")
                .str.replace_all(",", "")
                .cast(pl.Float64)
            )
        )
    )

    # Handle start data
    fund_info = (
        fund_info
        .with_columns(
            sus_fund_start = pl.col("sus_fund_start").cast(pl.Date)  # Ensure sus_fund_start is a date type
        )
    )
    if impute_start:
        # If start date is missing or equal to morningstar_start, set it to impute_start
        fund_info = (
            fund_info
            .with_columns(
                sus_fund_start = pl.when(pl.col("sus_fund_start").is_null() | (pl.col("sus_fund_start") == morningstar_start))
                .then(impute_start_date)
                .otherwise(pl.col("sus_fund_start"))
            )
        )
    else:
        # Else, exclude observations before sus_fund_start (around 4% of obs)
        fund_info = (
            fund_info
            .filter((pl.col('sus_fund_start')>=morningstar_start) & pl.col('sus_fund_start').is_not_null())
        )

    return fund_info

# Prepare holdings data ----------------------------------------------------------------
def load_holdings_raw(path: str) -> pl.DataFrame:
    """
    Reads a single Morningstar holdings CSV and returns per-fund raw holdings.

    - Filters for equity funds
    - Unpivots to long format
    - Adds EOM column, converts weights to decimals
    - Returns: isin, secid, date, eom, weight
    """
    id_cols = ["isin", "secid"]
    holdings = (
        ibis.duckdb.connect()
        .read_csv(path, table_name=str(path))
        .rename(lambda col: col.replace(" 00:00:00", "").lower().replace(" ", "_"))
        .filter(_.detail_holding_type == "EQUITY")
        .drop(["coupon_%", "secondary_sector", "detail_holding_type", "maturity_date",
                "name", "country", "currency"])
        .mutate(
            s.across(
                ~s.cols(*id_cols),
                lambda col: col.cast("double")
            )
        )
        .pivot_longer(
            ~s.cols(*id_cols),
            names_to="date",
            values_to="weight",
        )
        .cast({"weight": "double", "date": "date"})
        .drop_null("weight")
        .to_polars()
    )

    holdings = (
        holdings.lazy()
        .with_columns(eom=pl.col("date").dt.month_end())
        .cast({"weight": pl.Float64})
        .with_columns(weight=pl.col("weight") / 100)
        .collect()
    )

    return holdings


def aggregate_holdings(holdings_raw: pl.DataFrame, fund_info: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates raw per-fund holdings using fund_info.

    - Joins with fund_info to get fund_size and sus_fund_start
    - Filters: eom >= sus_fund_start
    - Computes holding = weight * fund_size
    - Removes duplicates by (secid, isin, date)
    - Aggregates by (isin, eom): sum(holding) / 1e6
    """
    holdings = (
        holdings_raw.lazy()
        .join(
            fund_info.lazy().select(['secid', 'eom', 'fund_size', 'sus_fund_start']),
            on=["secid", "eom"],
            how="left"
        )
        .filter(pl.col("eom") >= pl.col("sus_fund_start"))
        .with_columns(holding=pl.col("weight") * pl.col("fund_size"))
        .drop("weight", "fund_size", "sus_fund_start")
        .filter(pl.col("holding").is_not_null() & pl.col("isin").is_not_null())
        .with_columns(n=pl.len().over(["secid", "isin", "date"]))
        .filter(pl.col("n") == 1)
        .drop("n")
        .group_by(["isin", "eom"])
        .agg(holding=pl.col("holding").sum() / 1e6)
        .collect()
    )

    return holdings


def load_all_holdings_raw() -> pl.DataFrame:
    """Load all 3 Morningstar holdings CSVs and concatenate raw per-fund holdings."""
    paths = [
        DATA_RAW / "morningstar" / "Morningstar_holdings_consolidated.csv",
        DATA_RAW / "morningstar" / "Morningstar_holdings_consolidated_notArticle9.csv",
        DATA_RAW / "morningstar" / "Morningstar_holdings_consolidated_notArticle9_US.csv",
    ]
    raw_parts = []
    for path in paths:
        print(f"Loading {path}...")
        raw_parts.append(load_holdings_raw(path))
    return pl.concat(raw_parts)


def finalize_holdings_agg(holdings_raw, fund_info, df, isin_gvkey) -> pl.DataFrame:
    """
    From raw per-fund holdings, aggregate using fund_info, then merge with
    isin_gvkey and firm data to produce the final holdings_agg DataFrame.
    """
    start = time.perf_counter()

    # Aggregate per file-chunk, then re-aggregate across files
    holdings_agg = aggregate_holdings(holdings_raw, fund_info)
    holdings_agg = (
        holdings_agg
        .group_by(["isin", "eom"])
        .agg(holding=pl.col("holding").sum())
    )

    # Add gvkey (need to do conditional join in pandas..)
    holdings_agg = (
        holdings_agg
        .to_pandas()
        .astype({"eom": "datetime64[ns]"})
        .conditional_join(
            isin_gvkey,
            ('isin', 'ISSUER_ISIN', '=='),
            ('eom', 'startdate', '>='),
            ('eom', 'enddate', '<='),
            how="inner",
            df_columns    = slice(None),
            right_columns = ["gvkey"]
        )
    )
    # Add total market cap and excntry
    holdings_agg = (
        holdings_agg
        .rename(columns={'eom': 'date'})
        .merge(
            df[['excntry', 'gvkey', 'id', 'date', 'me']],
            on=['gvkey', 'date'],
            how='left'
        )
    )
    # Sustainable ownership share
    holdings_agg['share'] = holdings_agg['holding'] / holdings_agg['me']
    # Exclude observations with missing share
    holdings_agg = holdings_agg[holdings_agg['share'].notna()]
    # Keep only the ISIN with the highest holding per firm-date (occasionally multiple ISINs map to one id)
    holdings_agg = (
        holdings_agg
        .sort_values(
            by=["excntry", "id", "date", "holding"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["excntry", "id", "date"], keep="first")
        .reset_index(drop=True)
    )

    end = time.perf_counter()
    print(f"Total elapsed: {end - start:.2f} s")

    return pl.from_pandas(holdings_agg)


def direct_green_scores(df, holdings_agg, trim_perc) -> pl.DataFrame:
    """
    Generate direct green scores from holdings data.
    """
    df_direct = (
        holdings_agg
        .select(['id', 'date', 'excntry', 'share'])
        # Filters
        .filter(
            (pl.col('share') >= pl.col('share').quantile(trim_perc[0])) &
            (pl.col('share') <= pl.col('share').quantile(trim_perc[1]))
        )
        # Log and rank transformation
        .with_columns(
            share_raw = pl.col('share'),  # Keep raw share for later
            share_log = pl.col('share').log(),
            share_rank = pl.col('share').rank(method='ordinal').over(['excntry', 'date'])
        )
        # Cast for join
        .cast({'id': pl.Int64})
        # Join with df (to get proper subset to standardize over)
        .join(
            pl.from_pandas(df[['id', 'date']]),
            on = ['id', 'date'],
            how = 'inner'
        )
        # Standardize
        .with_columns(
            (
                (pl.col(var) - pl.col(var).mean().over(['excntry', 'date'])) / pl.col(var).std().over(['excntry', 'date'])
            ).alias(var)
            for var in ['share_raw', 'share_log', 'share_rank']  # Variables to standardize
        )
        # Rename columns
        .rename({'share': 'share_sust',
                 'share_raw': 'gs_direct_raw', 
                 'share_log': 'gs_direct_log', 
                 'share_rank': 'gs_direct_rank'})
        # Drop excntry
        .drop('excntry')
    )

    return df_direct

# Generated projected green score ------------------------------------------------------
# Ensure that df is prepare EXCATLY in the projection and extrapolate step
def proj_df(df, ctrls, gvars):
    data_main = pl.from_pandas(df[['id', 'date', 'gics', 'excntry']+ctrls+gvars]) 
    # Impute missing values in green vars
    data_main = (
        data_main
        # Require at least one of gvars to be non-null
        .filter(pl.any_horizontal([pl.col(var).is_not_null() for var in gvars]))
        # Impute missing values in green vars
        .with_columns([
            pl.col(var).fill_null(pl.col(var).mean()).over(["excntry", 'date'])
            for var in gvars  
        ])
    )
    # Output
    return data_main

def proj_data(df, holdings_agg, ctrls, gvars, proj_y, start_date, trim_perc) -> pl.DataFrame:
    # Create X variables in polars data frame
    data_main = proj_df(df=df, ctrls=ctrls, gvars=gvars)
    # Add y variable
    data_main = (
        data_main
        .join(
            holdings_agg.select(['id', 'date', 'share']).cast({'id': pl.Int64}),
            on = ['id', 'date'],
            how = 'left'
        )
    )

    # Prepare for regression
    data_main = (
        data_main
        # Filters
        .filter(
            (pl.col('share') >= pl.col('share').quantile(trim_perc[0])) &
            (pl.col('share') <= pl.col('share').quantile(trim_perc[1]))
        )
        # .filter((pl.col('share') > 0) & (pl.col('share') <= 1))  
        .filter(pl.col('date') >= start_date)
        .filter(pl.all_horizontal([pl.col(ctrl).is_not_null() for ctrl in ctrls]))  # Keep only complete controls
        # Helper columns
        .with_columns(
            excntry_date = pl.concat_str([pl.col('excntry'), pl.lit('_'), pl.col('date')])
        )
        # log and rank transformation of share
        .with_columns(
            share_raw = pl.col('share'),
            share_log = pl.col("share").log(),  
            share_rank = pl.col('share').rank(method='ordinal').over('excntry_date')
        )
        # Cross-sectionally standardize y
        .with_columns(
            (
                (pl.col(var) - pl.col(var).mean().over('excntry_date')) / pl.col(var).std().over('excntry_date')
            ).alias(var)
            for var in [proj_y]   
        )
        # Filter out rows with NaN/NULL/infinite values in controls, green vars, and y variables
        .filter(
            pl.all_horizontal([
                pl.col(col).is_not_null() & 
                pl.col(col).is_not_nan() & 
                pl.col(col).is_finite()
                for col in (ctrls + gvars + [proj_y])
            ])
        )
    )

    # Create a ridge regression with non-negative weights on green variables ---------------
    # Subtract the fixed effect
    df_dm = (
        data_main
        # Subtract the fixed effects
        .with_columns([
            # X variables (controls + green vars)
            (pl.col(var) - pl.col(var).mean().over("excntry_date")).alias(var)
            for var in (ctrls + gvars + [proj_y])
        ])
    )
    # Output
    return df_dm

# Objective function for non-negative ridge regression
def obj_ridge_pos(beta, y, X, lambda_reg, ctrls):
    """
    Objective function: MSE + λ * sum(beta^2)
    """
    r = y - X @ beta
    mse = np.mean(r**2)  # 1/n * sum(r^2)
    penalty = lambda_reg * np.sum(beta[len(ctrls):]**2) # Penalty only on green variables (skip first len(ctrls) variables)
    return mse + penalty

# Ridge with non-negative green parameters
def ridge_nonneg(y, X, init, bounds, lambda_reg, ctrls, gvars):
    """
    Estimate non-negative ridge regression using L-BFGS-B
    """
    # Optimize
    result = minimize(
        fun=obj_ridge_pos,
        x0=init,
        args=(y, X, lambda_reg, ctrls),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # Create results DataFrame
    results_df = pl.DataFrame({
        'variable': (ctrls + gvars),  # Use your green variable names
        'estimate': result.x
    })
    
    return results_df, result

def proj_estimate(df_dm, proj_y, ctrls, gvars, lambda_reg, verbose=True) -> tuple:
    # Extract y and X matrices
    y = df_dm.select(proj_y).to_numpy().flatten()
    X = df_dm.select((ctrls + gvars)).to_numpy()

    # Set up bounds
    p = X.shape[1]  # number of columns
    if verbose:
        print(f"Number of variables: {p}")
        print(f"Green variables: {len(gvars)}")
        print(f"Control variables: {len(ctrls)}")

    # Bounds: non-negative for green vars, unbounded for controls
    # Since we only have green vars in X (no controls), all should be non-negative
    lower_bounds = np.concatenate([
        np.full(len(ctrls), -np.inf),  # Controls can be negative (unbounded below)
        np.zeros(p - len(ctrls))       # Green variables >= 0
    ])
    upper_bounds = np.full(p, np.inf)  # All unbounded above

    # Create bounds list for scipy
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(p)]

    # Initial guess
    init = np.zeros(p)

    # Optimization
    if verbose:
        print(f"\nRunning ridge regression with lambda = {lambda_reg} and y = {proj_y}...")

    results_df, optimization_result = ridge_nonneg(y=y, X=X, init=init, bounds=bounds, lambda_reg=lambda_reg, ctrls=ctrls, gvars=gvars)

    if verbose:
        print(f"\nOptimization successful: {optimization_result.success}")
        print(f"Objective value: {optimization_result.fun:.6f}")
        print(f"Number of iterations: {optimization_result.nit}")

        print("\nParameter estimates:")
        results_pretty = (
            results_df
            .filter(
                (~pl.col("variable").is_in(ctrls)) &
                (pl.col("estimate") != 0)
            )
            .with_columns(
                weight = pl.col('estimate') / pl.col('estimate').sum()
            )
            .sort('estimate', descending=True)
        )
        print(results_pretty)

    # Compute R²
    y_hat = X @ optimization_result.x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Output
    return results_df, r_squared

def proj_green_score(df, results_df, ctrls, gvars) -> pl.DataFrame:
    """
    Extrapolate the projection to the full dataset, and create aggregate green score.
    """
    # Create aggregate green score
    ids = ['excntry', 'id', 'date']
    data_main = proj_df(df=df, ctrls=ctrls, gvars=gvars)
    df_proj = (
        data_main.lazy()
        .select(ids + gvars)  
        .unpivot(
            index = ids,
            on = gvars
        )
        .join(results_df.lazy(), on="variable", how="left")
        .filter(
            (pl.col("value").is_not_null()) & 
            (pl.col("variable").is_in(gvars)) & # Not neccesary, just a safeguard and for clarity 
            (pl.col("estimate") != 0)
        )
        .with_columns(
            weight = pl.col('estimate') / pl.col('estimate').sum().over(['id', 'date'])
        )
        .group_by(ids)
        .agg(
            raw = (pl.col('value') * pl.col('weight')).sum()
        )
        .with_columns(
            green_score = (pl.col('raw') - pl.col('raw').mean().over(['excntry', 'date'])) / pl.col('raw').std().over(['excntry', 'date'])
        )
    )
    # Output
    return df_proj.collect()

def projection(proj_y, df, holdings_agg, gvars, ctrls, lambda_reg, start_date, trim_perc):
    """
    Main function to run the projection.
    """
    # Generate projected data
    df_dm = proj_data(df=df, holdings_agg=holdings_agg, ctrls=ctrls, gvars=gvars,
                      proj_y=proj_y, start_date = start_date, trim_perc=trim_perc)

    # Estimate projection
    results_df, r_squared = proj_estimate(df_dm=df_dm, proj_y=proj_y, ctrls=ctrls,
                                          gvars=gvars, lambda_reg=lambda_reg)

    # Generate green score
    df_proj = proj_green_score(df=df, results_df=results_df, gvars=gvars, ctrls=ctrls)

    return results_df, df_proj, r_squared

def proj_green_scores(proj_y_vars, df, holdings_agg, gvars, ctrls, lambda_reg, start_date, trim_perc):
    # containers for each run's outputs
    results_list = []
    proj_list    = []
    r_squared_dict = {}

    for y in proj_y_vars:
        # run your projection for this y‐variable
        res_df, df_proj, r_squared = projection(
            proj_y=y,
            df=df,
            holdings_agg=holdings_agg,
            gvars=gvars,
            ctrls=ctrls,
            lambda_reg=lambda_reg,
            start_date=start_date,
            trim_perc=trim_perc
        )

        # tag each DataFrame with the y‐variable used
        res_df   = res_df.with_columns(pl.lit(y).alias('proj_y'))
        df_proj  = df_proj.with_columns(pl.lit(y).alias('proj_y'))

        # collect
        results_list.append(res_df)
        proj_list.append(df_proj)
        r_squared_dict[y] = r_squared

    # Concatenate into two "long" DataFrames
    est_proj   = pl.concat(results_list)
    df_proj   = pl.concat(proj_list)
    # Cast df proj to wide
    df_proj = (
        df_proj
        .select(['id', 'date', 'proj_y', 'green_score'])
        .pivot(
            index=['id', 'date'],
            on='proj_y',
            values='green_score'
        )
        .rename({
            'share_raw': 'gs_proj_raw',
            'share_log': 'gs_proj_log',
            'share_rank': 'gs_proj_rank'
        })
    )

    # Output
    return est_proj, df_proj, r_squared_dict
    
def proj_bootstrap(proj_y_vars, df, holdings_agg, gvars, ctrls, lambda_reg, start_date, trim_perc,
                   sample_group, repeats, seed):
    """Bootstrap normalized projection loadings. Returns per-replicate samples."""
    # Pre-split data by group for faster sampling
    groups = sorted(df[sample_group].unique())
    df_by_group = {g: df[df[sample_group] == g] for g in groups}

    np.random.seed(seed)
    bs_loadings = []

    for s in trange(repeats, desc='Bootstrap projection loadings'):
        # Sample groups with replacement
        bs_groups = np.random.choice(groups, size=len(groups), replace=True)

        # Build bootstrap sample by concatenating group data
        df_bs = pd.concat([df_by_group[g] for g in bs_groups], ignore_index=True)

        for y in proj_y_vars:
            try:
                df_dm = proj_data(df=df_bs, holdings_agg=holdings_agg, ctrls=ctrls, gvars=gvars,
                                  proj_y=y, start_date=start_date, trim_perc=trim_perc)
                results_df, _ = proj_estimate(df_dm=df_dm, proj_y=y, ctrls=ctrls,
                                              gvars=gvars, lambda_reg=lambda_reg, verbose=False)

                # Normalize loadings (green vars only)
                green_loadings = results_df.filter(pl.col('variable').is_in(gvars))
                total = green_loadings['estimate'].sum()
                if total > 0:
                    green_loadings = green_loadings.with_columns(
                        loading=pl.col('estimate') / total
                    )
                else:
                    green_loadings = green_loadings.with_columns(
                        loading=pl.lit(0.0)
                    )
                green_loadings = green_loadings.with_columns(
                    pl.lit(y).alias('proj_y'),
                    pl.lit(s).alias('bs_sample')
                )
                bs_loadings.append(green_loadings.select(['variable', 'proj_y', 'bs_sample', 'loading']))
            except Exception as e:
                print(f"Bootstrap sample {s}, y={y} failed: {e}")
                continue

    # Combine all bootstrap loadings (columns: variable, proj_y, bs_sample, loading)
    return pl.concat(bs_loadings)

def proj_bootstrap_cis(bs_samples, q_low=0.01, q_high=0.99):
    """Compute percentile CIs from bootstrap samples."""
    return (
        bs_samples
        .group_by(['variable', 'proj_y'])
        .agg(
            ci_low=pl.col('loading').quantile(q_low),
            ci_high=pl.col('loading').quantile(q_high)
        )
    )

# Plot projection loadings
def proj_plot_loadings(res_df, green_vars, green_vars_dict,
                       r_squared=None, bootstrap_cis=None) -> plt.Figure:
    """
    Plot the projection loadings with optional R² in titles and bootstrap CI whiskers.
    """
    sub = (
        res_df
        .filter(pl.col('variable').is_in(green_vars))  # Filter for green vars with non-zero estimates
        .with_columns(
            estimate = pl.col('estimate') / pl.col('estimate').sum().over(pl.col('proj_y'))
        )
        .pivot(
            index='variable',
            on='proj_y',
            values='estimate'
        )
        .sort(['share_rank', 'share_raw'])
    )

    # Create the plot with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(6.5, 4),   # ← match LaTeX width
        sharey=True, sharex=True
    )

    # Get positions
    positions = range(len(sub))
    labels = sub['variable'].to_pandas().map(green_vars_dict).to_list()

    # Panel config: (column_name, axis, base_title)
    panels = [
        ('share_raw', ax1, 'Green share'),
        ('share_log', ax2, 'Log(green share)'),
        ('share_rank', ax3, 'Rank(green share)'),
    ]

    for col, ax, base_title in panels:
        values = sub[col].to_numpy()

        # Compute error bars from bootstrap CIs
        xerr = None
        if bootstrap_cis is not None:
            ci_panel = bootstrap_cis.filter(pl.col('proj_y') == col)
            ci_ordered = sub.select(['variable']).join(ci_panel, on='variable', how='left')
            ci_low = ci_ordered['ci_low'].to_numpy()
            ci_high = ci_ordered['ci_high'].to_numpy()
            xerr = np.array([
                np.maximum(values - ci_low, 0),
                np.maximum(ci_high - values, 0)
            ])

        ax.barh(positions, values, xerr=xerr, color='green', alpha=0.8,
                ecolor=(0, 0, 0, 0.4), capsize=2)

        ax.set_title(base_title)
        ax.set_xlabel('Loading')
        ax.grid(True, alpha=0.3, axis='x')

        # R² annotation in bottom-left corner
        if r_squared is not None and col in r_squared:
            ax.text(0.95, 0.05, f'$R^2 = {r_squared[col]:.2f}$',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right')

    # Set y-axis labels (only on left panel since sharey=True)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(labels)
    ax1.tick_params(labelsize=10)

    # Adjust layout
    plt.tight_layout()

    return fig
