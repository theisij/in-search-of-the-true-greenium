import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from statsmodels.stats.multitest import multipletests
from tqdm import trange

from greenium.config import OUTPUT_FIGURES_GLOBAL


def bm_data(df, y, controls, green_vars, start, balanced):
    # Prepare data 
    df_input = df[['id', 'date', 'excntry', 'gics6', 'gics', 'country_ym_num'] +
                   [y] + controls + green_vars].copy()
    # Filters
    df_input = df_input[df_input['date'] >= start]
    print('Before dropping NAs:')
    print(df_input[green_vars].count()) 
    print('After dropping NA y:')
    df_input = df_input[df_input[y].notna()]
    print(df_input[green_vars].count())  
    if balanced:          
        print('After dropping any NAs:')
        df_input = df_input.dropna()
        print(df_input[green_vars].count())
        # Re-standardize for parameters to be comparable
        for var in green_vars:
            df_input[var] = df_input.groupby('country_ym_num')[var].transform(
                lambda x: (x - x.mean()) / x.std(ddof=1)
            )

    # Output
    return df_input


def bm_regressions(df_input, y, controls, agg_green_vars, cluster_se=True, verbose=True):
    """
    Benchmark regression for green scores.
    
    Parameters:
        df_input (DataFrame): Input DataFrame containing the data.
        y (str): Dependent variable (typically ICC).
        controls (list): List of control variables.
        agg_green_vars (list): List of aggregate green score variables.
        start (str): Start date for the analysis.
        
    Returns:
        Two plots showing the estimate and t-stats of {agg_green_vars}.
    """
    # setting index to firm-id and country-date (to use as FE input)
    df_input = df_input.set_index(['id', 'country_ym_num'])

    # Make percentages
    df_input[y] = df_input[y]*100 #scaling

    # Loop through each green score variable
    results = []
    for green_var in agg_green_vars:
        if verbose:
            print(f"Running regression for {green_var}...")
        
        # Create X matrix with constant and current green score variable
        X = pd.concat([df_input[green_var], df_input[controls]], axis=1)
        X = sm.add_constant(X)
        
        # Run PanelOLS regression
        reg = PanelOLS(
            df_input[y], 
            X, 
            entity_effects=False, 
            time_effects=True,
            other_effects=None, 
            drop_absorbed=True, 
            check_rank=False
        )
        if cluster_se:
            # Fit with clustered standard errors
            reg = reg.fit(
                cov_type="clustered",
                clusters=df_input[['date', 'gics']],
                cluster_entity=False,
                cluster_time=False
            )
        else:
            # Fit without clustering
            reg = reg.fit()
        
        # Controls-only regression on the same sample for marginal R².
        # NOTE: build the controls-only design + outcome by column selection
        # then dropna; do NOT do `df_input[y].loc[X.dropna().index]`. Line 55
        # set a MultiIndex on (id, country_ym_num), and in the bootstrap path
        # (bs_est_by_cor) the same (id, country_ym_num) appears once per
        # bootstrap repeat. A label-based .loc[] then returns every matching
        # row and dependent/exog lengths diverge — PanelOLS raises
        # "dependent and exog must have the same number of observations".
        # Column selection + dropna is immune to duplicate index labels.
        df_ctrl = df_input[[y, green_var, *controls]].dropna()
        X_ctrl = sm.add_constant(df_ctrl[controls])
        reg_ctrl = PanelOLS(
            df_ctrl[y],
            X_ctrl,
            entity_effects=False,
            time_effects=True,
            other_effects=None,
            drop_absorbed=True,
            check_rank=False
        ).fit()

        # Store the result with a descriptive name
        results.append((green_var, reg, reg_ctrl.rsquared_overall))

        if verbose:
            print(f"Completed {green_var}: N = {reg.nobs}")

    # Output
    return results

# Function to extract tidy results from PanelOLS regression
def tidy_panel_results(results_list):
    """
    Extract parameter estimates and statistics from PanelOLS results
    Similar to broom::tidy() in R
    """
    tidy_results = []
    
    for item in results_list:
        if len(item) == 3:
            green_var, reg, r2_controls = item
        else:
            green_var, reg = item
            r2_controls = None
        # Extract parameter estimates
        params = reg.params
        std_errors = reg.std_errors
        tvalues = reg.tstats
        pvalues = reg.pvalues
        conf_int = reg.conf_int()

        # Create a DataFrame for this regression
        for param_name in params.index:
            row = {
                'green_variable': green_var,
                'term': param_name,
                'estimate': params[param_name],
                'std_error': std_errors[param_name],
                't_statistic': tvalues[param_name],
                'p_value': pvalues[param_name],
                'conf_low': conf_int.loc[param_name, 'lower'],
                'conf_high': conf_int.loc[param_name, 'upper'],
                'n_obs': reg.nobs,
                'r_squared': reg.rsquared,
                'r_squared_overall': reg.rsquared_overall,
                'marginal_r_squared': reg.rsquared_overall - r2_controls if r2_controls is not None else np.nan
            }
            tidy_results.append(row)
    
    return pd.DataFrame(tidy_results)

def bm_plot_est(results_tidy, agg_green_vars):
    # Filter the data for green variables
    green_data = results_tidy[results_tidy['term'].isin(agg_green_vars)].copy()

    # Rename for clarity
    name_map = {
        'gs_ew':               'Equal weighted',
        'gs_direct_rank_imp':  'Aggregate green score',
        'gs_direct_raw':       'Green holding',
        'gs_direct_log':       'log(green holding)',
        'gs_direct_rank':      'Ranked green holding',
        'gs_direct_raw_bf':    'BF(green holding)',
        'gs_direct_log_bf':    'BF(log(green holding))',
        'gs_direct_rank_bf':   'BF(ranked green holding)',
        'gs_proj_raw':         'IV(green holding)',
        'gs_proj_log':         'IV(log(green holding))',
        'gs_proj_rank':        'IV(ranked green holding)'
    }
    green_data['green_variable'] = green_data['green_variable'].replace(name_map)

    # Create the plot similar to icc_plot
    fig = plt.figure(figsize=(6.5, 4))  # Match LaTeX width

    # Create labels for y-axis (reverse order for plotting from top to bottom)
    data_labels = green_data['green_variable'].tolist()[::-1]

    # Get the data (reverse order to match labels)
    estimates = green_data['estimate'].values[::-1]
    conf_low = green_data['conf_low'].values[::-1]
    conf_high = green_data['conf_high'].values[::-1]

    # Calculate error bars (distance from estimate to confidence bounds)
    lower_errors = estimates - conf_low
    upper_errors = conf_high - estimates

    # Create positions for each variable
    positions = range(1, len(estimates) + 1)

    # Create the plot
    plt.errorbar(estimates, positions, 
                xerr=[lower_errors, upper_errors], 
                fmt='o', color='green', capsize=5, linestyle='None',
                markersize=6)

    # Customize the plot
    plt.yticks(positions, data_labels)
    plt.axvline(0, color='black', alpha=0.5, linewidth=1)
    plt.xlabel('Greenium estimate (bps)')
    # plt.ylabel('', fontsize=12)
    #plt.title('Greenium estimates for aggregate green score proxies', fontsize=14)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    
    return fig

def bm_plot_t(results_tidy, agg_green_vars):
    # Filter the data for green variables
    green_data = results_tidy[results_tidy['term'].isin(agg_green_vars)].copy()

    # Rename for clarity
    name_map = {
        'gs_ew':               'Equal weighted',
        'gs_direct_rank_imp':  'Aggregate green score',
        'gs_direct_raw':       'Green share',
        'gs_direct_log':       'Log(green share)',
        'gs_direct_rank':      'Rank(green share)',
        'gs_direct_raw_bf':    'BF(green share)',
        'gs_direct_log_bf':    'BF(log(green share))',
        'gs_direct_rank_bf':   'BF(rank(green share))',
        'gs_proj_raw':         'IV(green share)',
        'gs_proj_log':         'IV(log(green share))',
        'gs_proj_rank':        'IV(rank(green share))'
    }
    green_data['green_variable'] = green_data['green_variable'].replace(name_map)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6.5, 4))  # Match LaTeX width
    
    # Get the data - sorted by t_statistic for better visualization
    # Keep ascending=True for y-axis order (most negative at bottom)
    green_data_sorted = green_data.sort_values('t_statistic', ascending=True)
    
    # Create labels and data
    data_labels = green_data_sorted['green_variable'].tolist()
    t_statistics = green_data_sorted['t_statistic'].values
    
    # Create positions for each variable
    positions = range(len(t_statistics))
    
    # Create the bar chart
    ax.barh(positions, t_statistics, color='green', alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(positions)
    ax.set_yticklabels(data_labels, size=7)
    ax.tick_params(axis='x', labelsize=7) 
    ax.axvline(0, color='black', alpha=0.5, linewidth=1)
    ax.set_xlabel('Greenium $t$-statistic', size=8)
    #ax.set_ylabel('', fontsize=14)
    #ax.set_title('Greenium t-stats for aggregate green score proxies', fontsize=14)
    
    # REVERT X-AXIS: Invert the x-axis so it goes from high to low (right to left)
    ax.invert_xaxis()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations for ALL bars
    for i, (pos, t_stat, label) in enumerate(zip(positions, t_statistics, data_labels)):
        text_x = t_stat

        ax.text(
            text_x, pos,
            f"{t_stat:.1f}",
            size = 7,
            ha="right", va="center",        # center horizontally & vertically
            # fontweight="bold",                # bold text
            color="white"                     # white text
        )
        
    
    # Adjust layout
    plt.tight_layout()
    
    # Output
    return fig

# Individual/aggregate comparison ------------------------------------------------------
def agg_indv_regressions(df, green_vars, controls):
    # Get relevant columns
    sub = (
        df[['id', 'date', 'excntry', 'country_ym_num', 'gics', 'icc_avg', 'green_score'] + green_vars + controls]
        .copy()
        # Require that green score is not missing
        .dropna(subset=['green_score'])
        # Set index for fixed effects
        .set_index(['id', 'country_ym_num'])
    )
    # Split by us, global ex-us and global

    # Do bivariate regressions for each green score variable
    results = []
    for green_var in green_vars:
        print(f"Running regression for {green_var}...")
        
        # Create X matrix with constant and current green score variable
        X = pd.concat([sub['green_score'], sub[green_var], sub[controls]], axis=1)
        X = sm.add_constant(X)
        
        # Run PanelOLS regression
        reg = PanelOLS(
            sub['icc_avg'], 
            X, 
            entity_effects=False, 
            time_effects=True,
            other_effects=None, 
            drop_absorbed=True, 
            check_rank=False
        ).fit(
            cov_type="clustered",
            clusters=sub[['date', 'gics']],
            cluster_entity=False,
            cluster_time=False
        )
        
        # Store the result with a descriptive name
        results.append((green_var, reg))
        
        print(f"...Completed {green_var}: N = {reg.nobs}")

    # Output
    return results

def bh_adjustment(results_tidy, alpha = 0.05):
    pvals = (
        results_tidy[results_tidy['green_variable']==results_tidy['term']]['p_value']
        .values
    )
    bh_op = multipletests(
        pvals,
        alpha=alpha,
        method='fdr_bh'        # options: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by', ...
    )
    return bh_op

def agg_indv_plot(results_tidy, green_vars, green_vars_dict):
    """
    Create a single grouped bar chart comparing individual green‐score t‐stats
    against the aggregate green‐score t‐stat for each variable.
    """
    # 1) prepare your data exactly as before up through `plot_df`
    plot_data = results_tidy[
        results_tidy['term'].isin(['green_score'] + green_vars)
    ].copy()
    pivot_data = plot_data.pivot_table(
        index='green_variable',
        columns='term',
        values='t_statistic',
        aggfunc='first'
    ).rename(columns={'green_score':'aggregate_t_stat'})
    # extract individual t‐stat into its own column
    pivot_data['individual_t_stat'] = pivot_data[green_vars].stack().droplevel(1)
    pivot_data = pivot_data.reset_index()[['green_variable','aggregate_t_stat','individual_t_stat']]

    # 2) sort by aggregate descending
    pivot_data = pivot_data.sort_values('aggregate_t_stat', ascending=False)
    
    # 3) plot
    labels = pivot_data['green_variable'].map(green_vars_dict).tolist()
    agg   = pivot_data['aggregate_t_stat']
    ind   = pivot_data['individual_t_stat']
    y_pos = np.arange(len(labels))
    height = 0.4

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(y_pos - height/2, agg, height, color='green',  label='Aggregate green score')
    ax.barh(y_pos + height/2, ind, height, color='orange', label='Individual green scores')

    # lines at 0 and ±1.96 *not meaningfull with MT adjustment
    # for v in [0, -1.96, 1.96]:
    #     ax.axvline(v, color='black', linestyle='--' if abs(v)==1.96 else '-', linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('$t$‐statistic')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    return fig

# Create new figure 1a -----------------------------------------------------------------
def new_fig1a_old(results_tidy):
    """
    Create new figure 1a comparing aggregate green score vs individual green scores.
    """
    # Average of agggregate estimates
    agg_avg = results_tidy[results_tidy['term']=='green_score']['t_statistic'].mean()
    agg_avg = pd.DataFrame({
        'term': ['green_score'],
        't_statistic': [np.abs(agg_avg)],
        'type': ['Aggregate']
    })
    # Individual estimates
    indv = results_tidy[results_tidy['term']==results_tidy['green_variable']]
    indv = indv[['term', 't_statistic']]
    indv['type'] = 'Individual'
    indv['t_statistic'] = np.abs(indv['t_statistic'])
    indv = indv.sort_values('t_statistic', ascending=False)
    indv['term'] = [f"i{n}" for n in range(1, len(indv)+1)]
    # Combine
    sub = pd.concat([agg_avg, indv])

    # Create plot 
    x = sub['term']
    y = sub['t_statistic']
    colors = {'Aggregate': 'green', 'Individual': 'orange'}

    plt.figure(figsize=(12, 4))
    plt.bar(
        x, 
        y, 
        color = sub['type'].map(colors)
    )
    legend_handles = [
        mpatches.Patch(color='green', label = 'Aggregate green score'),
        mpatches.Patch(color='orange', label = 'Individual green scores')
    ]
    plt.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=14                   
    )
    plt.xlabel('')
    plt.ylabel(
        '|Greenium $t$-statistic|',
        fontsize=14                   
    )
    plt.tight_layout()
    # Remove x axis text
    plt.xticks([])
    plt.yticks(fontsize=14)

    plt.savefig(OUTPUT_FIGURES_GLOBAL / "new_fig1a_old.pdf", format="pdf", bbox_inches="tight")
    plt.close('all')

# Individual versus aggregate green scores ---------------------------------------------
# Create a balanced data by imputing missing green scores
def df_impute_balanced(df, agg_var, indv_vars, gvars, ctrls, y, start):
    df_input = (
        pl.from_pandas(df[['id', 'date', 'gics', 'excntry', 'country_ym_num']+[y]+ctrls+gvars])
        .lazy()
        .filter(pl.col('date') >= pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d"))
        .filter(pl.col(agg_var).is_not_null())
        .with_columns([
            pl.col(var).fill_null(pl.col(var).mean()).over(["excntry", 'date'])
            for var in indv_vars  
        ])
        # .drop_nulls(subset=ctrls+[y])
        .with_columns(
            (
                (pl.col(var) - pl.col(var).mean().over('country_ym_num')) / 
                (pl.col(var).std().over('country_ym_num'))
            ).alias(var)
            for var in indv_vars   
        )
        # Fill nulls and NAs with zero
        .with_columns(
            [
                pl.when(
                    (pl.col(col).is_null())
                    | (pl.col(col).is_nan())
                    | (pl.col(col).is_infinite())
                )
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
                for col in indv_vars
            ]
        )
        # Drop cases with missing values in controls or y
        .drop_nulls(subset=ctrls+[y]+['gics'])
        .collect()
    )
    return df_input

# Estimate residual correlations (not working!!)
def residualize_old(df_input, res_var, controls):
    """
    Estimate residuals of a variable after controlling main FEs and controls
    """
    df_input = df_input.set_index(['id', 'country_ym_num'])
    y = df_input[res_var]
    X = df_input[controls]
    X = sm.add_constant(X)
    reg = PanelOLS(
            y, 
            X, 
            entity_effects=False, 
            time_effects=True,
            other_effects=None, 
            drop_absorbed=True, 
            check_rank=False
        ).fit()
    # Return residuals with NANs preserved
    mask = y.notna().to_numpy()
    resid_full = np.full(len(y), np.nan)
    resid_full[mask] = reg.resids.to_numpy()

    return resid_full

def residualize(df_input, res_var, controls):
    """
    Residualize res_var on controls + time FE (country_ym_num),
    returning residuals same length/order as df_input, with NaNs preserved.
    Works with duplicate (id, country_ym_num) due to bootstrap.
    """
    cols = [res_var] + controls + ['country_ym_num']
    d = df_input[cols].copy()

    # Absorb time FE by demeaning within time
    grp = d.groupby('country_ym_num', observed=True)
    y_tilde = d[res_var] - grp[res_var].transform('mean')
    X_tilde = d[controls] - grp[controls].transform('mean')

    # add constant for safety, though the demeaned regression doesn't need it
    X_tilde = sm.add_constant(X_tilde, has_constant='add')

    # OLS on demeaned data; keep only rows where y and all X are present
    valid = y_tilde.notna()
    for c in X_tilde.columns:
        valid &= X_tilde[c].notna()

    # allocate full-length residuals as NaN
    resid_full = np.full(len(d), np.nan)

    if valid.any():
        res = sm.OLS(y_tilde[valid].to_numpy(), X_tilde.loc[valid].to_numpy()).fit()
        resid_full[valid.to_numpy()] = res.resid  # positional write-back

    return resid_full



def residual_correlations(df_input, agg_var, indv_vars, ctrls):
    """
    Estimate residual correlation between aggregate green score and individual green scores.
    """
    agg_res = residualize(df_input=df_input, res_var=agg_var, controls=ctrls)
    residual_cors = []
    for green_var in indv_vars:
        indv_res = residualize(df_input=df_input, res_var=green_var, controls=ctrls)
        mask = ~np.isnan(agg_res) & ~np.isnan(indv_res)
        cor_res = pd.DataFrame({
            "green_variable": [green_var],
            "cor_res": np.corrcoef(agg_res[mask], indv_res[mask])[0,1]
        })
        residual_cors.append(cor_res)
    residual_cors = pd.concat(residual_cors, ignore_index=True)
    return residual_cors

def plot_est_by_cor(cors_est, agg_var):
    base_est = float(cors_est[cors_est['green_variable'] == agg_var]['estimate'])
    print(f"Base estimate for {agg_var}: {base_est}")
    plot_data = cors_est[cors_est['green_variable'] != agg_var].copy()
    x = plot_data['cor_res'].to_numpy()
    y = plot_data['estimate'].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    # scatter points
    ax.scatter(x, y, color='orange')

    # fit a simple linear regression line
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    xx_full = np.linspace(0, 1, 200)
    yy = intercept + slope * xx
    ax.plot(xx, yy, linewidth=1.5, color='black', label='OLS fit')

    # annotate the green point at (1, base_est)
    ax.scatter(1, base_est, color='green')

    # add the dashed reference line: y = -base_est * x
    ax.plot(xx_full, base_est * xx_full, linestyle='--', color='gray', linewidth=1)

    # annotate the green point at (1, base_est)
    ax.scatter(1, base_est, color='green')
    
    # reverse the y‐axis 
    ax.set_ylim(0, base_est - 2)

    # labels & ticks
    ax.set_xlabel("Correlation with aggregate green score", fontsize=14)
    ax.set_ylabel('Greenium (annual bps)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    legend_handles = [
        mpatches.Patch(color='green', label = 'Aggregate green score'),
        mpatches.Patch(color='orange', label = 'Individual green scores')
    ]
    plt.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=14                   
    )

    plt.tight_layout()

    return fig

def plot_stat_ranked(green_est, stat, x_label, revert_x=True, figsize=(6.5, 4), drop_y_ticks=False):
    # Prepare data
    plot_data = green_est.sort_values(stat, ascending=revert_x)
    
    x = plot_data['green_variable'].tolist()
    y = plot_data[stat].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "green" if val == "green_score" else "orange"
        for val in x
    ]
    legend_handles = [
        mpatches.Patch(color='green', label = 'Aggregate green score'),
        mpatches.Patch(color='orange', label = 'Individual green scores')
    ]
    plt.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=11                   
    )

    # Draw horizontal bars: names on the y‐axis, estimates on the x‐axis
    ax.barh(
        x,
        y,
        color=colors
    )
    # Invert axis
    ax.invert_yaxis()
    if revert_x:
        ax.invert_xaxis()
    
    if stat == 'r_squared_overall':
        ax.set_xlim(0.05, y.max()+0.0025)
    
    # Layout
    ax.set_xlabel(x_label)
    if drop_y_ticks:
        ax.tick_params(axis="y", labelleft=False)
    
    # ax.set_ylabel("")           # remove the y‐axis title

    plt.tight_layout()

    return fig

# Combine everything from above in one figure
def plot_stat_ranked_combined(green_est, green_vars_dict):
    """
    Create a combined plot with four panels showing different statistics of green variables.
    """
    # Fill missing cor_res for aggregate green score (correlation with itself = 1)
    green_est = green_est.copy()
    green_est['cor_res'] = green_est['cor_res'].fillna(1)

    # Define the panels with their respective statistics and labels
    # Each tuple contains: (statistic name, x-axis label, revert x-axis, show legend)
    panels = [
        ("estimate",           "Greenium estimate",           True,  False),
        ("t_statistic",        "Greenium $t$-statistic",      True,  False),
        ("marginal_r_squared", r"Greenium $\Delta R^2$",      False, True),
        ("cor_res",            r"$\rho$ with aggregate",      False, True),
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(8.5, 4),
        sharey=True,
        constrained_layout=True
    )

    for ax, (stat, x_label, revert_x, show_leg) in zip(axes, panels):
        # sort & extract
        plot_data = green_est.sort_values(stat, ascending=revert_x)
        labels    = plot_data["green_variable"].map(green_vars_dict).tolist()
        values    = plot_data[stat]

        # draw bars
        colors = ["green" if v == "Aggregate green score" else "orange" for v in labels]
        ax.barh(labels, values, color=colors)
        if revert_x:
            ax.invert_xaxis()
        if stat in ("marginal_r_squared", "cor_res"):
            ax.set_xlim(0, values.max() * 1.05)

        # x‐axis label
        ax.set_xlabel(x_label)

        # y‐tick labels: only on the *first* (Panel A)
        if ax is axes[0]:
            ax.tick_params(axis="y", which="both", labelleft=True)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

    # invert y-axis once (not per-panel, since sharey=True would toggle it)
    axes[0].invert_yaxis()

    # no ylabel text ─ labels are the bar names themselves
    plt.show()

    return fig

# Bootstrapped standard errors
def bs_est_by_cor(df_input, sample_group, repeats, seed, agg_var, indv_vars, ctrls):
    # Sort df_input to have a deterministic strategy for pruning
    bs_data = (
        df_input
        .copy()
        .sort_values(by=['excntry', 'date', 'id'])
        .reset_index(drop=True)
    )
    # Create groups to sample from
    bs_groups = (
        bs_data
        .groupby(sample_group)
        .size()
        .reset_index(name='N')
        .sort_values(by=sample_group)
    )
    n_orig = len(bs_data)
    # Split groups for faster sampling
    bs_data = dict(tuple(bs_data.groupby(sample_group)))
    # Estimate parameters over bootstrapped samples
    np.random.seed(seed)
    bs_op_list = []
    for s in trange(0, repeats, desc='Bootstrapping'):
        # Create bootstrap sample
        bs_obs = []
        n_bs = 0
        while n_bs < n_orig:
            samp = bs_groups.sample(n=1).iloc[0]
            bs_obs.append(samp['excntry'])
            n_bs += samp['N']
        sub = [bs_data[label] for label in bs_obs]
        sub = pd.concat(sub, ignore_index=True)
        sub = sub[:n_orig] # Prune to original sample size
        # Estimate greenium
        green_est = bm_regressions(df_input=sub, y='icc_avg', controls=ctrls, 
                                   agg_green_vars=indv_vars, cluster_se=False, verbose=False)
        green_est = tidy_panel_results(green_est)
        green_est = green_est[green_est['green_variable']==green_est['term']]
        green_est = green_est.reset_index(drop=True)
        # Estimate residual correlations
        residual_cors = residual_correlations(df_input=sub, agg_var=agg_var, 
                                              indv_vars=indv_vars, ctrls=ctrls)
        residual_cors = residual_cors.reset_index(drop=True)
        # Combine and save
        cors_est = green_est.merge(residual_cors, on='green_variable', how='left')
        cors_est['bs_sample'] = s
        bs_op_list.append(cors_est)

    bs_op = pd.concat(bs_op_list, ignore_index=True)
    return bs_op

def table_bs_est_by_cor(cors_est, cors_est_bs, agg_var):
    # Calculate the estimates on the original data
    agg_est = cors_est[cors_est['green_variable'] == agg_var]["estimate"].iloc[0]
    indv_est = cors_est[cors_est['green_variable'] != agg_var].copy()
    act_fit = smf.ols("estimate ~ cor_res", data=indv_est).fit()
    adj_r2 = act_fit.rsquared_adj
    print(f"Adjusted R2 from regression (include in paper) = {adj_r2:.2f}")
    
    # Calculate bootstrapped standard errors
    def est_by_cor_params(df):
        """Fit OLS on one group and return a Series of its params."""
        m = smf.ols("estimate ~ cor_res", data=df).fit()
        return pd.Series({
            "Intercept": m.params["Intercept"],
            "Slope_cor_res": m.params["cor_res"]
        })
    bs_estimates = (
        cors_est_bs
        .groupby("bs_sample")
        .apply(est_by_cor_params)
        .reset_index(drop=True)
    )
    bs_ses = bs_estimates.std(ddof=1)
    # Build pretty output table
    summary = pd.DataFrame({
        "Parameter": ["Intercept", "Slope"],
        "Estimate": [act_fit.params["Intercept"], act_fit.params["cor_res"]],
        "SE (OLS)": [act_fit.bse["Intercept"], act_fit.bse["cor_res"]],
        "SE (BS)":  [bs_ses['Intercept'], bs_ses['Slope_cor_res']]
    })
    ho_green = f"$t$-stat $(H_0$: ${agg_est:.0f})$"
    summary['$t$-stat $(H_0$: $0)$'] = (summary['Estimate'] - 0) / summary['SE (BS)']
    summary[ho_green] = (summary['Estimate'] - agg_est) / summary['SE (BS)']
    # Set t > aggregate test to empty for intercept
    summary.at[0, ho_green] = ""

    return summary 

def table_individual_est(cors_est, gvars_dict):
    tbl = cors_est[
        ['green_variable', 'estimate', 't_statistic', 'r_squared_overall',
         'marginal_r_squared', 'cor_res', 'n_obs']
    ].rename(columns={
        'green_variable': 'Variable',
        'n_obs': 'N',
        'estimate': 'Estimate',
        't_statistic': 't-stat',
        'r_squared_overall': r'$R^2$',
        'marginal_r_squared': r'$\Delta R^2$',
        'cor_res': r'$\rho$'
    }).assign(
        # Map names from gvars_dict, escape underscores for LaTeX
        Variable=lambda df: df['Variable'].map(gvars_dict).str.replace('_', r'\_', regex=False),
        **{r'$\rho$': lambda df: df[r'$\rho$'].mask(cors_est['green_variable'] == 'green_score', 1)}
    )
    
    return tbl