# importing packages and paths

import numpy as np
import pandas as pd

import polars as pl

from greenium.aggregate_green_score import (
    finalize_holdings_agg,
    load_all_holdings_raw,
    load_fund_info,
)
from greenium.config import (
    DATA_INTERIM,
    IMPUTE_START,
    IMPUTE_START_DATE,
    global_start,
    set_start_morningstar,
)
from greenium.expected_returns import bond_data, market_data, options_er, subjective
from greenium.green_measures import isin_gvkey_load, msci, sustainalytics, trucost


# merge data and then clean
def merge_and_clean(df, df_msci, df_trucost, df_sust_indicator, df_sust_legacy,
                    df_icc, df_ms, df_vl, df_valrat, df_subj_er, df_options_glb, df_options_mwb):

    # adding sales/assets/MV info to trucost using fiscal date + 4 months (same lag as in JKP)
    df_trucost['date'] = (df_trucost['date'] + pd.DateOffset(months=4)).dt.to_period('M').dt.to_timestamp('M')
    df_trucost = df_trucost.merge(df.drop_duplicates(['gvkey', 'date', 'sales'], keep='first')[['gvkey', 'date', 'sales', 'assets', 'me_company']], how='left', on=['gvkey', 'date'])

    df_trucost = df_trucost[df_trucost.sales > 0]
    df_trucost = df_trucost[df_trucost.assets > 0]

    df_trucost['scope1int_sales'] = (df_trucost['scope1tot'] / df_trucost['sales'])
    df_trucost['scope2int_sales'] = (df_trucost['scope2tot'] / df_trucost['sales'])
    df_trucost['scope3int_sales'] = (df_trucost['scope3tot'] / df_trucost['sales'])
    df_trucost['scope12int_sales'] = df_trucost['scope1int_sales'] + df_trucost['scope2int_sales']
    df_trucost['scope123int_sales'] = df_trucost['scope12int_sales'] + df_trucost['scope3int_sales']

    df_trucost['scope1int_assets'] = (df_trucost['scope1tot'] / df_trucost['assets'])
    df_trucost['scope2int_assets'] = (df_trucost['scope2tot'] / df_trucost['assets'])
    df_trucost['scope3int_assets'] = (df_trucost['scope3tot'] / df_trucost['assets'])
    df_trucost['scope12int_assets'] = df_trucost['scope1int_assets'] + df_trucost['scope2int_assets']
    df_trucost['scope123int_assets'] = df_trucost['scope12int_assets'] + df_trucost['scope3int_assets']

    df_trucost['scope12tot'] = df_trucost['scope1tot'] + df_trucost['scope2tot']
    df_trucost['scope123tot'] = df_trucost['scope12tot'] + df_trucost['scope3tot']

    # scaling variables
    for i in ['scope1int_sales', 'scope12int_sales', 'scope123int_sales',
              'scope1int_assets', 'scope12int_assets', 'scope123int_assets']:
        df_trucost[i] = df_trucost[i] / 100

    for i in ['scope1tot', 'scope12tot', 'scope123tot']:
        log_string = "log_"+i
        df_trucost[log_string] = np.log(df_trucost[i]+1)

    #removing 4-month lagged date and instead using the date when available
    df_trucost = df_trucost.drop(['date'], axis=1)
    df_trucost = df_trucost.rename(columns={'date_eff': 'date'})

    df = df.merge(df_trucost[['date', 'gvkey', 'log_scope1tot', 'log_scope12tot', 'log_scope123tot',
                              'scope1int_sales', 'scope12int_sales', 'scope123int_sales',
                              'scope1int_assets', 'scope12int_assets', 'scope123int_assets']],
                  how='left', on=['date', 'gvkey'])

    # frontfilling trucost data 3 years
    df = df.sort_values(['id', 'date'])
    for col in ['log_scope1tot', 'log_scope12tot', 'log_scope123tot', 'scope1int_sales', 'scope12int_sales', 'scope123int_sales',
                              'scope1int_assets', 'scope12int_assets', 'scope123int_assets']:
        df[col] = df.groupby(['id'])[col].fillna(method='ffill', limit=35)

    # Add MSCI data
    df = df.merge(df_msci[['gvkey', 'date', 'industry_esg_score', 'weighted_esg_score', 'e_score',
                           'e_climate_score', 'e_natres_score', 'e_waste_score', 'e_envopps_score',
                           'greenness']], how='left', on=['gvkey', 'date'])

    # Add Sustainalytics data: both legacy and new indicator data. Then frontfilling both 1 year. Then setting cutoff to transition in 2019
    df = df.merge(df_sust_legacy[['gvkey', 'date', 'total_esg_score', 'environment_score']], how='left', on=['gvkey', 'date'])
    df = df.merge(df_sust_indicator[['gvkey', 'date', 'total_esg_score', 'environment_score']], how='left', on=['gvkey', 'date'])

    df = df.sort_values(['id', 'date'])
    for col in ['total_esg_score_x', 'total_esg_score_y', 'environment_score_x', 'environment_score_y']:
        df[col] = df.groupby(['id'])[col].fillna(method='ffill', limit=11)

    df['total_esg_score_x'] = np.where(df['date'] > '2019-12-31', np.nan, df['total_esg_score_x'])
    df['environment_score_x'] = np.where(df['date'] > '2019-12-31', np.nan, df['environment_score_x'])
    df['total_esg_score_y'] = np.where(df['date'] <= '2019-12-31', np.nan, df['total_esg_score_y'])
    df['environment_score_y'] = np.where(df['date'] <= '2019-12-31', np.nan, df['environment_score_y'])

    df['total_esg_score'] = df['total_esg_score_x'].fillna(df['total_esg_score_y'])
    df['environment_score'] = df['environment_score_x'].fillna(df['environment_score_y'])

    df = df.drop(['total_esg_score_x', 'total_esg_score_y', 'environment_score_x', 'environment_score_y'], axis=1)


    # frontfilling msci/sustainalytics/tri info 1 year (using id as opposed to id because unique and global data)
    df = df.sort_values(['id', 'date'])
    for col in ['industry_esg_score', 'weighted_esg_score', 'e_score', 'greenness',
                'e_climate_score', 'e_natres_score', 'e_waste_score', 'e_envopps_score',
                'release_total_sales', 'prod_waste_total_sales', 'release_total_assets', 'prod_waste_total_assets']:
        df[col] = df.groupby(['id'])[col].fillna(method='ffill', limit=11)

    # define factors
    total_em_factors = ['log_scope1tot', 'log_scope12tot', 'log_scope123tot']

    emission_factors = ['scope1int_sales', 'scope12int_sales', 'scope123int_sales',
                        'scope1int_assets', 'scope12int_assets', 'scope123int_assets']

    msci_factors = ['industry_esg_score', 'weighted_esg_score', 'e_score', 'greenness',
                    'e_climate_score', 'e_natres_score', 'e_waste_score', 'e_envopps_score']

    sust_factors = ['total_esg_score', 'environment_score']

    to_test = total_em_factors + emission_factors + msci_factors + sust_factors

    # removing esg info in months where the number of observations is low
    min_obs = 100  # number of observations required in a given month for a given ESG variable
    for col in to_test:
        valid_obs = df.groupby('date')[col].transform('count')
        df[col] = np.where(valid_obs >= min_obs, df[col], np.nan)

    # restricting sample to standard assumptions (but only doing now to avoid merging and frontfilling issues)
    for i in ['assets', 'sales', 'book_equity', 'me_company']:
        df = df[df[i] > 0]

    # creating a few variables after imposing requirements on size
    df['netdebt_at'] = df['debt_at'] - df['cash_at']
    df['log_size'] = np.log(df.me_company)
    df['log_assets'] = np.log(df.assets)
    df['log_be'] = np.log(df.book_equity)

    #removing valuation ratios with negative earnings
    df['be_me'] = np.where(df['be_me'] > 0, df['be_me'], np.nan)
    df['bev_mev'] = np.where(df['bev_mev'] > 0, df['bev_mev'], np.nan)
    df['ebitda_mev'] = np.where(df['ebitda_mev'] > 0, df['ebitda_mev'], np.nan)

    # removing infinity valuation ratios
    df['be_me'] = df['be_me'].replace([np.inf, -np.inf], np.nan)
    df['bev_mev'] = df['bev_mev'].replace([np.inf, -np.inf], np.nan)
    df['ni_me'] = df['ni_me'].replace([np.inf, -np.inf], np.nan)
    df['ebitda_mev'] = df['ebitda_mev'].replace([np.inf, -np.inf], np.nan)

    # creating country-time, country-industry and country-industry-time numerical value (industry based on gic)
    df['country_ym'] = np.where(df['excntry'].notna(), df.excntry.astype(str) + df.eom.astype(str), np.nan)
    df['country_ym_num'] = pd.factorize(df.country_ym)[0]
    df['country_gics6'] = np.where((df['excntry'].notna() & (df['gics6'].notna())),
                                    df.excntry.astype(str) + df.gics6.astype(str), np.nan)
    df['country_gics6'] = pd.factorize(df.country_gics6)[0]
    df['country_ym_gics6'] = np.where((df['excntry'].notna() & (df['gics6'].notna())),
                                    df.excntry.astype(str) + df.gics6.astype(str) + df.eom.astype(str), np.nan)
    df['country_ym_gics6'] = pd.factorize(df.country_ym_gics6)[0]

    # adding global icc data
    df = df.merge(df_icc, how='left', on=['date', 'id'])

    # adding global valuation ratios
    df = df.merge(df_valrat, how='left', on=['date', 'id'])

    # winsorizing cross-sectionally by month
    def winsorize_group(group):
        for variable in emission_factors + ['be_me', 'debt_at',
                                             'ep_fwd0', 'ep_fwd1', 'ep_fwd2', 'ep_fwd0_all', 'ep_fwd1_all', 'ep_fwd2_all',
                                             'ebit_at', 'bev_mev', 'ni_me', 'ebitda_mev',
                                             'ret_12_1', 'beta_252d', 'cash_at', 'netdebt_at',
                                             'icc_ct', 'icc_gls', 'icc_peg', 'icc_oj']:
            group[variable] = group[variable].clip(lower=group[variable].quantile(0.01),
                                                   upper=group[variable].quantile(0.99))
        return group

    df = df.groupby('date').apply(winsorize_group).reset_index(drop=True)

    # scaling earnings measures
    for i in ['ep_fwd0', 'ep_fwd1', 'ep_fwd2', 'ebitda_mev', 'ep_fwd0_all', 'ep_fwd1_all', 'ep_fwd2_all']:
        df[i] = df[i]*100

    #taking log valuation ratios after winz
    df['log_be_me'] = np.log(df['be_me'])
    df['log_bev_mev'] = np.log(df['bev_mev'])

    # calculating average ICC and overall of all 4 measures - after winsorizing
    df['icc_avg'] = df[['icc_gls', 'icc_ct', 'icc_oj', 'icc_peg']].mean(axis=1)

    # subjective and valuation measures (US only). Winsorize all except value-line
    df = df.merge(df_ms, how='left', on=['date', 'permno'])
    df = df.merge(df_vl, how='left', on=['date', 'permno'])
    df = df.merge(df_subj_er, how='left', on=['date', 'permno'])

    # merging to main dataframe
    df = df.merge(df_options_glb[['permno', 'date', 'glb2_D30', 'glb3_D30', 'glb2_D91', 'glb3_D91', 'glb2_D182', 'glb3_D182']], how='left', on=['permno', 'date'])
    df = df.merge(df_options_mwb[['permno', 'date', 'mw30', 'mw91', 'mw182']], how='left', on=['permno', 'date'])

    # winsorizing cross-sectionally by month (not icc measures which are global and winsorized above)
    def winsorize_group(group):
        for variable in ['ms_coe', 'glb2_D30', 'glb3_D30', 'glb2_D91', 'glb3_D91', 'glb2_D182', 'glb3_D182', 'mw30', 'mw91', 'mw182']:
            group[variable] = group[variable].clip(lower=group[variable].quantile(0.01),
                                                   upper=group[variable].quantile(0.99))
        return group

    df = df.groupby('date').apply(winsorize_group).reset_index(drop=True)

    # creating average measures of options and subjective data after winsorizing
    df['coc_avg_options'] = df[['mw30', 'mw91', 'mw182', 'glb2_D30', 'glb2_D91', 'glb2_D182']].mean(axis=1) #across the most liquid maturities
    df['subj_rr_avg'] = df[['ms_coe', 'vl_req_ret']].mean(axis=1)

    df = df.drop(df[(df['ret_exc_lead1m'].isna()) & (
                df['date'] != '2022-12-31')].index)  # dropping missing leading return, unless Dec-2022

    df = df.drop(df[(df['ret_exc_lead1m'].isna()) & (
                df['date'] != '2022-12-31')].index)  # dropping missing leading return, unless Dec-2022

    # winsorizing returns globally ex-US (only Compustat data)
    df['ret_exc_lead1m'] = np.where(((df['ret_exc_lead1m'] > df['ret_exc_99_9_lead1m']) & (df.excntry!='USA')), df['ret_exc_99_9_lead1m'], df['ret_exc_lead1m'])
    df['ret_exc_lead1m'] = np.where(((df['ret_exc_lead1m'] < df['ret_exc_0_1_lead1m']) & (df.excntry!='USA')), df['ret_exc_0_1_lead1m'], df['ret_exc_lead1m'])

    df = df.drop(['ret_exc_99_9_lead1m', 'ret_exc_0_1_lead1m'], axis=1)

    # drop rows where 'excntry' is 'USA' and permno is missing
    df.drop(df[(df['excntry'] == 'USA') & (df['permno'].isna())].index, inplace=True)
    # Drop rows where 'excntry' is 'USA' and 'crsp_shrcd' is not 10 or 11
    df.drop(df[(df['excntry'] == 'USA') & (~df['crsp_shrcd'].isin([10, 11]))].index, inplace=True)

    # handling duplicates: Create a boolean column to indicate if 'excntry' is 'USA'
    df['is_usa'] = df['excntry'] == 'USA'

    # sort by 'gvkey', 'date', prioritize 'USA', and then by market cap ('me') in descending order
    df.sort_values(by=['gvkey', 'date', 'is_usa', 'me'], ascending=[True, True, False, False], inplace=True)

    # drop duplicates by 'gvkey' and 'date', keeping the first occurrence (which will be the 'USA' if present and largest 'me')
    df.drop_duplicates(subset=['gvkey', 'date'], keep='first', inplace=True)

    # drop the helper 'is_usa' column if you don't need it
    df.drop(columns=['is_usa'], inplace=True)

    return df

def prepare_save_data(df, df_bonds):
    total_em_factors = ['log_scope1tot', 'log_scope12tot', 'log_scope123tot']

    emission_factors = ['scope1int_sales', 'scope12int_sales', 'scope123int_sales',
                        'scope1int_assets', 'scope12int_assets', 'scope123int_assets']

    msci_factors = ['industry_esg_score', 'weighted_esg_score', 'e_score', 'greenness',
                    'e_climate_score', 'e_natres_score', 'e_waste_score', 'e_envopps_score']

    sust_factors = ['total_esg_score', 'environment_score']

    # All ESG factors
    to_test = total_em_factors + emission_factors + msci_factors + sust_factors

    # removing esg information for months where number of observations is low
    min_obs = 100  # number of observations required in a given month for a given ESG variable
    for col in to_test:
        valid_obs = df.groupby('date')[col].transform('count')
        df[col] = np.where(valid_obs >= min_obs, df[col], np.nan)

    # adding firm-level data to bond-information which has only one observation per month (VW data) (adding ESG info later)
    df_bonds = df_bonds.merge(df[['date', 'gvkey', 'ff49', 'log_assets', 'log_be', 'be_me', 'ebit_at',
                                  'ret_12_1', 'beta_252d', 'netdebt_at', 'log_size',
                                  'gics6', 'gics6_ym', 'country_ym', 'country_ym_num', 'country_ym_gics6', 'country_gics6']],
                              how='left', on=['date', 'gvkey'])
    df_bonds = df_bonds.drop_duplicates(['gvkey', 'date'], keep='first') #need to do this because gvkey merge

    # Saving interim data 
    df.to_parquet(DATA_INTERIM / "df_main.parquet")
    df_bonds.to_parquet(DATA_INTERIM / "df_bonds.parquet")

    return df, df_bonds

def load_all_data():
    # loading or generating data (larger files are stored when code run once)
    isin_gvkey = isin_gvkey_load()
    print("ISIN-GVKEY bridge created")
    df_msci = msci(isin_gvkey)
    print("MSCI data loaded")
    df_trucost = trucost()
    print("Trucost data loaded")
    df_sust_indicator, df_sust_legacy = sustainalytics()
    print("Sustainalytics data loaded")
    df = market_data()
    print("Market data loaded")
    df_bonds = bond_data()
    print("Bond data loaded")
    df_icc, df_ms, df_vl, df_valrat, df_subj_er = subjective()
    print("ICC data loaded")
    df_options_glb, df_options_mwb = options_er()
    print("Option data loaded")

    # combined data
    df = merge_and_clean(df, df_msci, df_trucost, df_sust_indicator, df_sust_legacy,
                         df_icc, df_ms, df_vl, df_valrat, df_subj_er, df_options_glb, df_options_mwb)
    print("Main data combined and cleaned")

    # set minimum data requirement for variables and save
    df, df_bonds = prepare_save_data(df, df_bonds)
    print("Main data prepared and saved")

    # Create holdings data ---
    # Generate fund info
    fund_info = load_fund_info(
        impute_start = IMPUTE_START,
        impute_start_date = IMPUTE_START_DATE,
        morningstar_start = set_start_morningstar
    )
    # Load raw holdings once
    holdings_raw = load_all_holdings_raw()
    # Generate baseline aggregated holdings
    holdings_agg = finalize_holdings_agg(holdings_raw, fund_info, df, isin_gvkey)
    holdings_agg.write_parquet(DATA_INTERIM / "holdings_agg.parquet")
    print("Sustainable ownership data aggregated and saved")

    # Generate backfilled aggregated holdings (robustness test)
    from datetime import date as _date
    fund_info_bf = fund_info.with_columns(
        pl.when(pl.col('sus_fund_start') == set_start_morningstar)
        .then(pl.lit(_date.fromisoformat(global_start)))
        .otherwise(pl.col('sus_fund_start'))
        .alias('sus_fund_start')
    )
    holdings_agg_bf = finalize_holdings_agg(holdings_raw, fund_info_bf, df, isin_gvkey)
    holdings_agg_bf.write_parquet(DATA_INTERIM / "holdings_agg_bf.parquet")
    print("Backfilled sustainable ownership data aggregated and saved")

    return df, df_bonds, holdings_agg, holdings_agg_bf