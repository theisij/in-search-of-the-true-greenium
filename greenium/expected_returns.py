# importing packages and paths
#from config import *
import zipfile

import numpy as np
import pandas as pd

from greenium.config import DATA_RAW


# subjective measures
def subjective():
    # ICC US: some values 1700%, some negative. Winsorize later
    df_icc = pd.read_csv(DATA_RAW / "icc" / "icc_us.csv")
    df_icc = df_icc.rename(columns={'datadate': 'date'})
    df_icc = df_icc.set_index(pd.to_datetime(df_icc['date'], format='%Y-%m-%d'))
    df_icc.index = df_icc.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_icc = df_icc.drop(['date'], axis=1).reset_index()
    df_icc = df_icc[df_icc.id.notna()]
    df_icc[['icc_ct', 'icc_gls', 'icc_peg', 'icc_oj']] = df_icc[['icc_ct', 'icc_gls', 'icc_peg', 'icc_oj']] * 100

    # ICC: Global
    df_icc_g = pd.read_csv(DATA_RAW / "icc" / "icc_comp.csv")
    df_icc_g = df_icc_g.rename(columns={'datadate': 'date'})
    df_icc_g = df_icc_g.set_index(pd.to_datetime(df_icc_g['date'], format='%Y-%m-%d'))
    df_icc_g.index = df_icc_g.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_icc_g = df_icc_g.drop(['date'], axis=1).reset_index()
    df_icc_g = df_icc_g[df_icc_g.id.notna()]
    df_icc_g[['icc_ct', 'icc_gls', 'icc_peg', 'icc_oj']] = df_icc_g[['icc_ct', 'icc_gls', 'icc_peg', 'icc_oj']] * 100

    # ICC: merging to one dataframe
    df_icc = pd.concat([df_icc, df_icc_g], axis=0)

    # morningstar - also needs to be winsorized from massive outliers. Removing risk-free rate (will not be fixed by time FE in WACC analysis)
    df_ms = pd.read_csv(DATA_RAW / "sers" / "morningstar.csv", usecols=['permno', 'eom', 'ms_coe_exc'])
    df_ms = df_ms.rename(columns={'eom': 'date'})
    df_ms = df_ms.set_index(pd.to_datetime(df_ms['date'], format='%Y-%m-%d'))
    df_ms.index = df_ms.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_ms = df_ms.drop(['date'], axis=1).reset_index()
    df_ms = df_ms[df_ms.permno.notna()]
    df_ms[['ms_coe_exc']] = df_ms[['ms_coe_exc']] * 100

    df_rf = pd.read_csv(DATA_RAW / "jkp" / "rf.csv")
    df_rf = df_rf.set_index(pd.to_datetime(df_rf['date'], format='%Y%m'))
    df_rf.index = df_rf.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_rf = df_rf.drop(['date'], axis=1).reset_index()
    df_ms = df_ms.merge(df_rf, how='left', on=['date'])
    df_ms['ms_coe'] = df_ms['ms_coe_exc'] + df_ms['rf'] #adding back RF
    df_ms = df_ms.drop(['rf', 'ms_coe_exc'], axis=1)

    # value line - less need to winsorize it seems
    df_vl = pd.read_csv(DATA_RAW / "sers" / "value_line.csv", usecols=['eom', 'permno', 'vl_req_ret'])
    df_vl = df_vl.rename(columns={'eom': 'date'})
    df_vl = df_vl.set_index(pd.to_datetime(df_vl['date'], format='%Y-%m-%d'))
    df_vl.index = df_vl.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_vl = df_vl.drop(['date'], axis=1).reset_index()
    df_vl = df_vl[df_vl.permno.notna()]
    df_vl[['vl_req_ret']] = df_vl[['vl_req_ret']] * 100

    # valuation ratios - needs to be winsorized later (similar to be/me already done). Combining global and US
    df_valrat = pd.read_csv(DATA_RAW / "icc" / "forward_eps_us.csv")
    df_valrat_g = pd.read_csv(DATA_RAW / "icc" / "forward_eps_comp.csv")
    df_valrat = pd.concat([df_valrat, df_valrat_g], axis=0)
    df_valrat = df_valrat.rename(columns={'datadate': 'date'})
    df_valrat = df_valrat.set_index(pd.to_datetime(df_valrat['date'], format='%Y-%m-%d'))
    df_valrat.index = df_valrat.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_valrat = df_valrat.drop(['date'], axis=1).reset_index()
    df_valrat = df_valrat[df_valrat.id.notna()]
    df_valrat = df_valrat.rename(columns={'eps0_p': 'ep_fwd0', 'eps1_p': 'ep_fwd1', 'eps2_p': 'ep_fwd2'})

    # removing valuation ratios with negative earnings (and those with infinity) ("all" keeps negative observations)
    df_valrat['ep_fwd0_all'] = df_valrat['ep_fwd0']
    df_valrat['ep_fwd1_all'] = df_valrat['ep_fwd1']
    df_valrat['ep_fwd2_all'] = df_valrat['ep_fwd2']
    df_valrat['ep_fwd0'] = np.where(df_valrat['ep_fwd0'] > 0, df_valrat['ep_fwd0'], np.nan)
    df_valrat['ep_fwd1'] = np.where(df_valrat['ep_fwd1'] > 0, df_valrat['ep_fwd1'], np.nan)
    df_valrat['ep_fwd2'] = np.where(df_valrat['ep_fwd2'] > 0, df_valrat['ep_fwd2'], np.nan)
    df_valrat = df_valrat.replace([np.inf, -np.inf], np.nan)
    df_valrat = df_valrat.sort_values(['id', 'date'])

    #subj expected returns: IBES/MS/VL are 1/3/4 year. Not winsorizing.
    df_subj_er = pd.read_csv(DATA_RAW / "sers" / "subjective_er.csv")
    df_subj_er = df_subj_er.rename(columns={'eom': 'date'})
    df_subj_er = df_subj_er.set_index(pd.to_datetime(df_subj_er['date'], format='%Y-%m-%d'))
    df_subj_er.index = df_subj_er.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_subj_er = df_subj_er.drop(['date'], axis=1).reset_index()
    df_subj_er = df_subj_er[df_subj_er.permno.notna()]
    df_subj_er['er_ms'] = (df_subj_er['er_ms'] + 1) ** (1 / 3) - 1
    df_subj_er['er_vl'] = (df_subj_er['er_vl'] + 1) ** (1 / 4) - 1
    df_subj_er['er_ibes'] = df_subj_er['er_ibes']*100
    df_subj_er['er_ms'] = df_subj_er['er_ms'] * 100
    df_subj_er['er_vl'] = df_subj_er['er_vl'] * 100

    return df_icc, df_ms, df_vl, df_valrat, df_subj_er

# option-implied expected returns
def options_er():
    #### testing more options data
    zf = zipfile.ZipFile(DATA_RAW / "vilkov_options" / "glb_daily.zip")
    df_options_glb = pd.read_csv(zf.open('glb_daily.csv', 'r'))
    zf.close()

    zf = zipfile.ZipFile(DATA_RAW / "vilkov_options" / "MartinWagnerBounds.zip")
    df_options_mwb = pd.read_csv(zf.open('MartinWagnerBounds.csv', 'r'))
    zf.close()

    # renaming
    df_options_glb = df_options_glb.rename(columns={'id': 'permno'})  # renaming before merging
    df_options_mwb = df_options_mwb.rename(columns={'id': 'permno'})  # renaming before merging

    # creating year-month variable to take mean over
    df_options_glb['date'] = df_options_glb['date'].astype(str).str[0:7]
    df_options_mwb['date'] = df_options_mwb['date'].astype(str).str[0:7]

    # monthly averages
    df_options_glb = df_options_glb.groupby(['permno', 'date'])[['glb2_D30', 'glb3_D30', 'glb2_D91', 'glb3_D91',
                                                                 'glb2_D182', 'glb3_D182', 'glb2_D273', 'glb3_D273',
                                                                 'glb2_D365', 'glb3_D365']].mean().reset_index()
    df_options_glb = df_options_glb.drop_duplicates(['permno', 'date'], keep='last')

    df_options_mwb = df_options_mwb.groupby(['permno', 'date'])[
        ['mw30', 'mw91', 'mw182', 'mw273', 'mw365']].mean().reset_index()
    df_options_mwb = df_options_mwb.drop_duplicates(['permno', 'date'], keep='last')

    # creating date-time
    df_options_glb = df_options_glb.set_index(pd.to_datetime(df_options_glb['date'], format='%Y-%m'))
    df_options_glb.index = df_options_glb.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_options_glb = df_options_glb.drop(['date'], axis=1)
    df_options_glb = df_options_glb.reset_index()
    df_options_glb[['glb2_D30', 'glb3_D30', 'glb2_D91', 'glb3_D91', 'glb2_D182', 'glb3_D182',
                    'glb2_D273', 'glb3_D273', 'glb2_D365']] = df_options_glb[['glb2_D30', 'glb3_D30', 'glb2_D91', 'glb3_D91',
                                                                   'glb2_D182', 'glb3_D182',
                                                                   'glb2_D273', 'glb3_D273', 'glb2_D365']] * 100

    df_options_mwb = df_options_mwb.set_index(pd.to_datetime(df_options_mwb['date'], format='%Y-%m'))
    df_options_mwb.index = df_options_mwb.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_options_mwb = df_options_mwb.drop(['date'], axis=1)
    df_options_mwb = df_options_mwb.reset_index()
    df_options_mwb[['mw30', 'mw91', 'mw182', 'mw273', 'mw365']] = df_options_mwb[
                                                                      ['mw30', 'mw91', 'mw182', 'mw273', 'mw365']] * 100

    return df_options_glb, df_options_mwb

# market data (CRSP/Compustat from JKP)
def market_data():
    cols = ['id', 'eom', 'gvkey', 'permno', 'excntry', 'crsp_shrcd',
            'me', 'me_lag1', 'me_company', 'book_equity', 'sales', 'assets', 'bev_mev', 'ni_me', 'ebitda_mev',
            'size_grp', 'ff49', 'sic', 'gics', 'ret_exc', 'ret_exc_lead1m',
            'be_me', 'debt_at', 'ebit_at', 'ni_be', 'ret_12_1', 'beta_252d', 'cash_at']

    # load all countries
    ziptrain = zipfile.ZipFile(DATA_RAW / "jkp" / "Characteristics.zip")
    train = []
    train = [pd.read_csv(ziptrain.open(f), usecols=cols) for f in ziptrain.namelist()]
    df = pd.concat(train)
    del (ziptrain, train)

    # setting start date to reduce size of file
    df = df[df.eom >= 19910000]

    # requiring gvkey
    df = df[df.gvkey.notna()]

    # df.permno = df.permno.astype(int)
    df.gvkey = df.gvkey.astype(int)
    df['year'] = df['eom'].astype(str).str[0:4].astype(int)
    df['month'] = df['eom'].astype(str).str[4:6].astype(int)
    df['date'] = pd.to_datetime(df['eom'], format='%Y%m%d')
    df['ret_exc_lead1m'] = df['ret_exc_lead1m'] * 100  # scaling returns

    # return cutoffs
    ret_cutoffs = pd.read_csv(DATA_RAW / "jkp" / "return_cutoffs.csv", usecols=['eom', 'ret_exc_0_1', 'ret_exc_99_9'])
    ret_cutoffs['date'] = pd.to_datetime(ret_cutoffs['eom'], format='%Y%m%d')
    ret_cutoffs['ret_exc_0_1'] = ret_cutoffs['ret_exc_0_1']*100 # scaling
    ret_cutoffs['ret_exc_99_9'] = ret_cutoffs['ret_exc_99_9'] * 100 # scaling
    ret_cutoffs['ret_exc_0_1_lead1m'] = ret_cutoffs['ret_exc_0_1'].shift(-1)
    ret_cutoffs['ret_exc_99_9_lead1m'] = ret_cutoffs['ret_exc_99_9'].shift(-1)

    # merging ret cutoffs
    df = df.merge(ret_cutoffs[['date', 'ret_exc_0_1_lead1m', 'ret_exc_99_9_lead1m']], how='left', on=['date'])

    # merging better industry data from Theis
    df_industry = pd.read_csv(DATA_RAW / "other" / "industry_codes.csv")
    df_industry['date'] = pd.to_datetime(df_industry['eom'], format='%Y-%m-%d')
    df_industry = df_industry.dropna()

    #dropping existing industry codes and merging new ones with better coverage and 2022 covered too
    df = df.drop(['sic', 'ff49'], axis=1)
    df = df.merge(df_industry[['id', 'date', 'sic', 'ff49']], how='left', on=['date', 'id'])

    # adding nyse breakpoints
    nyse_bp = pd.read_csv(DATA_RAW / "other" / "ME_Breakpoints.csv")
    nyse_bp = nyse_bp.set_index(pd.to_datetime(nyse_bp['date'], format='%Y%m'))
    nyse_bp.index = nyse_bp.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    nyse_bp = nyse_bp.drop(['date'], axis=1)
    nyse_bp = nyse_bp.reset_index()

    df = df.merge(nyse_bp[['date', 'nyse_50', 'nyse_80']], how='left', on='date')
    df['big'] = np.where((df.me > df.nyse_50), 1, 0)
    df = df.drop('nyse_50', axis=1)

    # creating cap value weight at 80th NYSE percentile
    df['me_cap'] = np.where(df['me'] > df['nyse_80'], df['nyse_80'], df['me'])

    # creating GIC industry codes including interacted with time
    df['gics6'] = np.where(df['gics'].notna(), df.gics.astype(str).str[0:6], np.nan)
    df['gics6_ym'] = np.where(df['gics6'].notna(), df.gics6.astype(str) + df.eom.astype(str), np.nan)

    # converting to smaller dtypes to save space
    for i in ['me', 'me_lag1', 'me_company',
              'ret_exc', 'ret_exc_lead1m', 'book_equity', 'assets', 'sales',
              'ret_12_1', 'bev_mev', 'ni_me', 'ebitda_mev',
              'ebit_at', 'debt_at', 'cash_at',
              'be_me', 'beta_252d', 'nyse_80', 'me_cap']:
        df[i] = df[i].astype(np.float32)

    # adding region
    country_list = pd.read_excel(DATA_RAW / "other" / "country_abbr.xlsx", sheet_name="Sheet1")
    country_list = country_list.rename(columns={'iso_3': 'excntry'})  # renaming to merge

    # some manual edits to Regional names
    country_list['Region'] = country_list['Region'].replace("Eastern Africa", "Africa")
    country_list['Region'] = country_list['Region'].replace("Indian Ocean", "Africa")
    country_list['Region'] = country_list['Region'].replace("Northern Africa", "Africa")
    country_list['Region'] = country_list['Region'].replace("Southern Africa", "Africa")
    country_list['Region'] = country_list['Region'].replace("Western Africa", "Africa")

    country_list['Region'] = country_list['Region'].replace("Central America", "CS America")
    country_list['Region'] = country_list['Region'].replace("South America", "CS America")

    country_list['Region'] = country_list['Region'].replace("South East Europe", "Eastern Europe")
    country_list['Region'] = country_list['Region'].replace("South West Europe", "Western Europe")

    country_list['Region'] = country_list['Region'].replace("Central Europe", "CE Europe")
    country_list['Region'] = country_list['Region'].replace("Eastern Europe", "CE Europe")

    # Moving 3 countries to Western Europe
    country_list['Region'] = np.where(country_list['excntry'] == "ITA", "Western Europe", country_list['Region'])
    country_list['Region'] = np.where(country_list['excntry'] == "CHE", "Western Europe", country_list['Region'])
    country_list['Region'] = np.where(country_list['excntry'] == "AUT", "Western Europe", country_list['Region'])

    df = df.merge(country_list[['Region', 'Continent', 'Country', 'excntry']], how='left', on=['excntry'])

    #df.to_pickle("market_data.pkl")

    return df

# bond data
def bond_data():
    df_bonds = pd.read_csv(DATA_RAW / "bonds" / "yields_traded_firm.csv")
    df_bonds = df_bonds.drop(['ret_exc_lead'], axis=1)
    df_bonds = df_bonds.rename(columns={'eom': 'date'})

    # requiring permno
    df_bonds = df_bonds[df_bonds.gvkey.notna()]

    # fixing date format
    df_bonds = df_bonds.set_index(pd.to_datetime(df_bonds['date'], format='%Y-%m-%d'))
    df_bonds.index = df_bonds.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_bonds = df_bonds.drop(['date'], axis=1).reset_index()

    df_bonds = df_bonds[df_bonds.debt_mv.notna()]
    df_bonds = df_bonds[df_bonds.debt_fv.notna()]

    df_bonds = df_bonds.dropna(subset=['rating_num', 'tmt'], axis=0)  # a few are missing a rating
    #df_bonds = df_bonds[df_bonds.rating_num <= 18]  # otherwise too speculative

    # scaling to millions and require size at least 1m
    df_bonds['debt_mv'] = df_bonds['debt_mv'] / 1000000
    df_bonds['debt_fv'] = df_bonds['debt_fv'] / 1000000

    # require positive market value and at least 1M face value outstanding
    df_bonds = df_bonds[df_bonds.debt_mv > 0]
    df_bonds = df_bonds[df_bonds.debt_fv >= 1]

    df_bonds['log_debt_mv'] = np.log(df_bonds.debt_mv)
    df_bonds['log_debt_fv'] = np.log(df_bonds.debt_fv)

    # scaling yields to basis points
    df_bonds['yield'] = df_bonds['yield'] * 100
    df_bonds['yield_spread'] = df_bonds['yield_spread'] * 100
    df_bonds['yield_adj'] = df_bonds['yield_adj'] * 100
    df_bonds['yield_spread_adj'] = df_bonds['yield_spread_adj'] * 100

    #df_bonds.to_pickle("bond_data.pkl")

    return df_bonds