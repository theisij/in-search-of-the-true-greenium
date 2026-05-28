# importing packages and paths
import os
import zipfile

import numpy as np
import pandas as pd

from greenium.config import DATA_RAW


# ISIN-GVKEY linking table
def isin_gvkey_load():
    # loading gvkey and isin information from wrds
    wrds_gvkey = pd.read_csv(DATA_RAW / "linking" / "WRDS_GVKEY.csv")
    wrds_isin = pd.read_csv(DATA_RAW / "linking" / "WRDS_ISIN.csv")

    wrds_gvkey = wrds_gvkey[wrds_gvkey['gvkey'].notna()]
    wrds_isin = wrds_isin[wrds_isin['isin'].notna()]

    # replacing beginning "B" / end "E" values with early and late dates
    wrds_gvkey['startdate'] = wrds_gvkey['startdate'].replace("B", '19000101')
    wrds_gvkey['enddate'] = wrds_gvkey['enddate'].replace("E", '20231231')
    wrds_isin['startdate'] = wrds_isin['startdate'].replace("B", '19000101')
    wrds_isin['enddate'] = wrds_isin['enddate'].replace("E", '20231231')

    # renaming before merge
    wrds_gvkey = wrds_gvkey.rename(
        columns={'startdate': 'startdate_gvkey', 'enddate': 'enddate_gvkey', 'companyname': 'cname_gvkey'})
    wrds_isin = wrds_isin.rename(
        columns={'startdate': 'startdate_isin', 'enddate': 'enddate_isin', 'companyname': 'cname_isin'})

    isin_gvkey = wrds_gvkey.merge(wrds_isin, how='left', on=['companyid'])
    isin_gvkey = isin_gvkey[isin_gvkey['gvkey'].notna()]
    isin_gvkey = isin_gvkey[isin_gvkey['isin'].notna()]

    # making into date-time for lookup
    isin_gvkey['startdate_gvkey'] = pd.to_datetime(isin_gvkey['startdate_gvkey'], format='%Y%m%d')
    isin_gvkey['enddate_gvkey'] = pd.to_datetime(isin_gvkey['enddate_gvkey'], format='%Y%m%d')
    isin_gvkey['startdate_isin'] = pd.to_datetime(isin_gvkey['startdate_isin'], format='%Y%m%d')
    isin_gvkey['enddate_isin'] = pd.to_datetime(isin_gvkey['enddate_isin'], format='%Y%m%d')

    # creating start and end dates (max and min, respectively of gvkey and isin)
    isin_gvkey['startdate'] = isin_gvkey[['startdate_isin', 'startdate_gvkey']].max(axis=1)
    isin_gvkey['enddate'] = isin_gvkey[['enddate_isin', 'enddate_gvkey']].min(axis=1)

    # removing those where startdate > enddate and only keeping key info
    isin_gvkey = isin_gvkey[isin_gvkey.startdate <= isin_gvkey.enddate]
    isin_gvkey = isin_gvkey[['gvkey', 'isin', 'startdate', 'enddate']]

    isin_gvkey.gvkey = isin_gvkey.gvkey.astype(int)
    isin_gvkey = isin_gvkey.rename(columns={'isin': 'ISSUER_ISIN'})  # renaming before merging

    return isin_gvkey

# MSCI ESG
def msci(isin_gvkey):
    # loading ESG data (first two files are summary files and will overlap with expanded data, so must keep latter when overlap)
    esg_2007_to_2012 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2007 to 2012.xlsx', usecols=range(30))
    esg_2013 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2013.xlsx', usecols=range(30))
    esg_2014 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2014.xlsx', usecols=range(30))
    esg_2015 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2015.xlsx', usecols=range(30))
    esg_2016 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2016.xlsx', usecols=range(30))
    esg_2017 = pd.read_excel(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2017.xlsx', usecols=range(30))
    esg_2018 = pd.read_csv(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2018 - CSV.csv', usecols=range(30))
    esg_2019 = pd.read_csv(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2019 - CSV.csv', usecols=range(30))
    esg_2020 = pd.read_csv(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2020 - CSV.csv', usecols=range(30))
    esg_2021 = pd.read_csv(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2021 - CSV.csv', usecols=range(30))
    esg_2022 = pd.read_csv(DATA_RAW / 'msci' / 'ESG Ratings Timeseries Expanded 2022.csv', usecols=range(30))

    # need to convert iva_rating_date before consolidating because date formats are different
    esg_2007_to_2012['IVA_RATING_DATE'] = pd.to_datetime(esg_2007_to_2012['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2013['IVA_RATING_DATE'] = esg_2013['IVA_RATING_DATE'].replace(19000100, 20130101)  # two values are strange ...
    esg_2013['IVA_RATING_DATE'] = pd.to_datetime(esg_2013['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2014['IVA_RATING_DATE'] = pd.to_datetime(esg_2014['IVA_RATING_DATE'], format='%m/%d/%y')
    esg_2015['IVA_RATING_DATE'] = pd.to_datetime(esg_2015['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2016['IVA_RATING_DATE'] = pd.to_datetime(esg_2016['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2017['IVA_RATING_DATE'] = pd.to_datetime(esg_2017['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2018['IVA_RATING_DATE'] = pd.to_datetime(esg_2018['IVA_RATING_DATE'], format='%Y%m%d')
    esg_2019['IVA_RATING_DATE'] = pd.to_datetime(esg_2019['IVA_RATING_DATE'], format='%m/%d/%Y')
    esg_2020['IVA_RATING_DATE'] = pd.to_datetime(esg_2020['IVA_RATING_DATE'], format='%m/%d/%Y')
    esg_2021['IVA_RATING_DATE'] = pd.to_datetime(esg_2021['IVA_RATING_DATE'], format='%m/%d/%Y')
    esg_2022['IVA_RATING_DATE'] = pd.to_datetime(esg_2022['IVA_RATING_DATE'], format='%m/%d/%Y')

    #appending all information to one dataframe
    df_msci = pd.concat([esg_2007_to_2012, esg_2013, esg_2014, esg_2015, esg_2016, esg_2017,
         esg_2018, esg_2019, esg_2020, esg_2021, esg_2022], axis=0, ignore_index=True)

    # sorting info
    df_msci = df_msci[df_msci.ISSUER_ISIN.notna()] #requiring ISIN
    df_msci = df_msci.set_index(pd.to_datetime(df_msci['AS_OF_DATE'], format='%Y%m%d')) #creating date-time format
    df_msci.index = df_msci.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_msci = df_msci.drop(['AS_OF_DATE'], axis=1)
    df_msci = df_msci.reset_index()
    df_msci = df_msci.sort_values(by=['ISSUER_ISIN', 'AS_OF_DATE'], ascending=True)
    df_msci = df_msci.rename(columns={'AS_OF_DATE': 'date'})
    df_msci = df_msci.drop(['ISSUER_NAME', 'ISSUERID', 'ISSUER_TICKER', 'ISSUER_CUSIP',
       'ISSUER_SEDOL', 'ISSUER_CNTRY_DOMICILE', 'IVA_INDUSTRY', 'IVA_RATING_TREND', 'IVA_PREVIOUS_RATING', 'HUMAN_CAPITAL_THEME_SCORE'], axis=1) #dropping unnecessary info

    # renaming to meaningful names
    df_msci = df_msci.rename(columns={'ENVIRONMENTAL_PILLAR_SCORE': 'e_score'})
    df_msci = df_msci.rename(columns={'ENVIRONMENTAL_PILLAR_WEIGHT': 'e_weight'})
    df_msci = df_msci.rename(columns={'SOCIAL_PILLAR_SCORE': 's_score'})
    df_msci = df_msci.rename(columns={'SOCIAL_PILLAR_WEIGHT': 's_weight'})
    df_msci = df_msci.rename(columns={'GOVERNANCE_PILLAR_SCORE': 'g_score'})
    df_msci = df_msci.rename(columns={'GOVERNANCE_PILLAR_WEIGHT': 'g_weight'})
    df_msci = df_msci.rename(columns={'WEIGHTED_AVERAGE_SCORE': 'weighted_esg_score'})
    df_msci = df_msci.rename(columns={'INDUSTRY_ADJUSTED_SCORE': 'industry_esg_score'})
    df_msci = df_msci.rename(columns={'CLIMATE_CHANGE_THEME_SCORE': 'e_climate_score'})
    df_msci = df_msci.rename(columns={'CLIMATE_CHANGE_THEME_WEIGHT': 'e_climate_weight'})
    df_msci = df_msci.rename(columns={'NATURAL_RES_USE_THEME_SCORE': 'e_natres_score'})
    df_msci = df_msci.rename(columns={'NATURAL_RES_USE_THEME_WEIGHT': 'e_natres_weight'})
    df_msci = df_msci.rename(columns={'WASTE_MGMT_THEME_SCORE': 'e_waste_score'})
    df_msci = df_msci.rename(columns={'WASTE_MGMT_THEME_WEIGHT': 'e_waste_weight'})
    df_msci = df_msci.rename(columns={'ENVIRONMENTAL_OPPS_THEME_SCORE': 'e_envopps_score'})
    df_msci = df_msci.rename(columns={'ENVIRONMENTAL_OPPS_THEME_WEIGHT': 'e_envopps_weight'})

    #a few weighted esg scores obviously missing but data is available
    df_msci['weighted_esg_score'] = df_msci['weighted_esg_score'].fillna((df_msci['e_score']*df_msci['e_weight']+
                                            df_msci['s_score']*df_msci['s_weight']+
                                            df_msci['g_score']*df_msci['g_weight'])/100)

    #because of some overlap of early files, must drop duplicates - the early file has same values but no weighted_esg_score
    df_msci = df_msci.sort_values(['ISSUER_ISIN', 'date', 'weighted_esg_score'])
    df_msci = df_msci.drop_duplicates(['ISSUER_ISIN', 'date'], keep='first')

    #fixing a few wrong observations to maximum valuess
    df_msci['e_natres_score'] = np.where(df_msci['e_natres_score'] > 10, 10, df_msci['e_natres_score'])
    df_msci['e_climate_score'] = np.where(df_msci['e_climate_score'] > 10, 10, df_msci['e_climate_score'])

    # using isin-gvkey bridge
    df_msci = df_msci.merge(isin_gvkey, how='left', on=['ISSUER_ISIN'])
    df_msci = df_msci[(df_msci.date>=df_msci.startdate) & (df_msci.date<=df_msci.enddate)]

    df_msci = df_msci.drop(['startdate', 'enddate'], axis=1)

    # cleaning df
    df_msci = df_msci[~df_msci.gvkey.isna()]  # removing without gvkey link
    df_msci = df_msci[~df_msci.industry_esg_score.isna()]  # removing based on missing industry score (weighted score n/a early sample for all obs)
    df_msci = df_msci.drop_duplicates(['date', 'gvkey', 'industry_esg_score'], keep='first')  # removing duplicates with same score (multiple ISIN lookup)
    df_msci = df_msci.sort_values(['date', 'gvkey', 'IVA_RATING_DATE']) #sorting such that for firm-month, latest rating is last
    df_msci = df_msci.drop_duplicates(['date', 'gvkey'], keep='last')  # removing remaining duplicates - keeping last score (double/triple ISIN lookup, most are similar)

    df_msci.gvkey = df_msci.gvkey.astype(float)  # converting to float to remove leading 00 etc. similar to df_crsp

    # calculating PST greenness
    df_msci['greenness'] = -(10 - df_msci['e_score']) * df_msci['e_weight'] / 100 + 10

    return df_msci

# Trucost
def trucost():
    zf = zipfile.ZipFile(DATA_RAW / "trucost" / "trucost.zip")
    df_trucost = pd.read_csv(zf.open('nfyldit54txkyele.csv', 'r'),
                             usecols=['gvkey', 'periodenddate', 'di_323615_text', 'di_319407',
                                      'di_319413', 'di_319408', 'di_319414', 'di_319409',
                                      'di_319415', 'di_319403_text', 'di_329689_text', 'di_368758_text'])
    zf.close()

    df_trucost = df_trucost.rename(columns={'periodenddate': 'date', 'di_323615_text': 'date_eff',
                                            'di_319407': 'scope1int', 'di_319413': 'scope1tot',
                                            'di_319408': 'scope2int', 'di_319414': 'scope2tot',
                                            'di_319409': 'scope3int', 'di_319415': 'scope3tot',
                                            'di_319403_text': 'scope1_discl',
                                            'di_329689_text': 'scope2_discl',
                                            'di_368758_text': 'scope3_discl'})

    df_trucost = df_trucost[df_trucost.gvkey.notna()]
    df_trucost = df_trucost[df_trucost.scope1int.notna()]  # only a few missing
    df_trucost = df_trucost[df_trucost.date_eff.notna()]  # only a few missing

    # fiscal date
    df_trucost = df_trucost.set_index(pd.to_datetime(df_trucost['date'], format='%Y-%m-%d'))
    df_trucost.index = df_trucost.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_trucost = df_trucost.drop(['date'], axis=1)
    df_trucost = df_trucost.reset_index()

    # effective date of environmental calculation (nearly all have this)
    df_trucost['date_eff'] = df_trucost['date_eff'].astype(str).str[0:11]
    df_trucost = df_trucost.set_index(pd.to_datetime(df_trucost['date_eff']))
    df_trucost.index = df_trucost.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_trucost = df_trucost.drop(['date_eff'], axis=1)
    df_trucost = df_trucost.reset_index()

    df_trucost = df_trucost.drop_duplicates(['gvkey', 'date', 'scope1int'],
                                            keep='first')  # if duplicates by fiscal date with same scope1int, keep one (very few)
    df_trucost = df_trucost.drop_duplicates(['gvkey', 'date'],
                                            keep=False)  # if still duplicates, drop both as entity not clear or different estimation dates (only few obs)

    # now handling cases where estimated data is same datapoint across years (dropping duplicates by gvkey-estimate date and only keeping latest)
    df_trucost = df_trucost.sort_values(['gvkey', 'date_eff', 'date'])
    df_trucost = df_trucost.drop_duplicates(['gvkey', 'date_eff'], keep='last')
    df_trucost['date_diff'] = (df_trucost['date_eff'] - df_trucost['date']).dt.days / 365
    # some only shifting by 1-2 months. Let's ensure minimum 4 month lag floor
    df_trucost['date_eff_lag'] = (df_trucost['date'] + pd.DateOffset(months=4)).dt.to_period('M').dt.to_timestamp('M')
    df_trucost['date_eff_use'] = np.where(df_trucost.date_eff < df_trucost.date_eff_lag, df_trucost.date_eff_lag,
                                          df_trucost.date_eff)
    df_trucost['date_diff_use'] = (df_trucost['date_eff_use'] - df_trucost['date']).dt.days / 365
    df_trucost = df_trucost[
        df_trucost.date_diff_use <= 5]  # requiring max 5 year data lag (only few obs and perhaps won't even be used)

    # dropping not needed columns and renaming date_eff_use to date_eff
    df_trucost = df_trucost[['date_eff_use', 'date', 'scope1int', 'scope3int', 'scope2int',
                             'scope1tot', 'scope2tot', 'scope3tot', 'gvkey']]

    df_trucost = df_trucost.rename(columns={'date_eff_use': 'date_eff'})

    # because we have lagged, we need to one more time drop duplicates in case lag overlaps with existing
    df_trucost = df_trucost.sort_values(['gvkey', 'date_eff', 'date'])
    df_trucost = df_trucost.drop_duplicates(['gvkey', 'date_eff'], keep='last')

    df_trucost.gvkey = df_trucost.gvkey.astype(int)

    return df_trucost

# Sustainalytics
def sustainalytics():
    # loading new sustainalytics data: re-creating old score system by focusing on E/S/G sub-information
    zf = zipfile.ZipFile(DATA_RAW / "sustainalytics" / "sust_indicator_all.zip")
    df_sust_indicator = pd.read_csv(zf.open('ipcyxtqygf6aicwh.csv', 'r'),
                                    usecols=['EntityId', 'FieldName', 'FieldValue', 'File_date', 'FieldDate'],
                                    encoding='unicode_escape')
    zf.close()

    df_sust_indicator = df_sust_indicator[df_sust_indicator.FieldName != 'EntityId']  # already a separate column
    df_sust_indicator = df_sust_indicator[df_sust_indicator.FieldName != 'Constant-Weight-RR']
    df_sust_indicator = df_sust_indicator[df_sust_indicator.FieldName != 'Constant-Weighted Score-RR']

    df_sust_indicator = df_sust_indicator.drop_duplicates(keep='first')  # a few exact duplicates, just keeping first

    # there are other duplicated (same dates etc., but two values. We take mean as approximation)
    df_sust_indicator = df_sust_indicator.pivot_table(index=['EntityId', 'File_date'], columns='FieldName',
                                                      values='FieldValue', aggfunc='mean')  # creating column-form frame

    df_sust_indicator = df_sust_indicator.dropna(axis=0, how='all')  # dropping if all values n/a for a firm-time pair

    # summing weights and scores separately for all E/S/G (first creating column names, then doing the sum)
    e_weight_cols = [col for col in df_sust_indicator.columns if
                     (col.startswith('E.') & col.endswith('-Weight-Overall-RR'))]
    s_weight_cols = [col for col in df_sust_indicator.columns if
                     (col.startswith('S.') & col.endswith('-Weight-Overall-RR'))]
    g_weight_cols = [col for col in df_sust_indicator.columns if
                     (col.startswith('G.') & col.endswith('-Weight-Overall-RR'))]

    e_score_cols = [col for col in df_sust_indicator.columns if
                    (col.startswith('E.') & col.endswith('-Weighted Score-Overall-RR'))]
    s_score_cols = [col for col in df_sust_indicator.columns if
                    (col.startswith('S.') & col.endswith('-Weighted Score-Overall-RR'))]
    g_score_cols = [col for col in df_sust_indicator.columns if
                    (col.startswith('G.') & col.endswith('-Weighted Score-Overall-RR'))]

    # Step 2: Sum these columns for each row
    df_sust_indicator['e_weight'] = df_sust_indicator[e_weight_cols].sum(axis=1)
    df_sust_indicator['s_weight'] = df_sust_indicator[s_weight_cols].sum(axis=1)
    df_sust_indicator['g_weight'] = df_sust_indicator[g_weight_cols].sum(axis=1)
    df_sust_indicator['e_score'] = df_sust_indicator[e_score_cols].sum(axis=1)
    df_sust_indicator['s_score'] = df_sust_indicator[s_score_cols].sum(axis=1)
    df_sust_indicator['g_score'] = df_sust_indicator[g_score_cols].sum(axis=1)

    # creating sums of weights and scores
    df_sust_indicator['esg_weight'] = df_sust_indicator[['e_weight', 's_weight', 'g_weight']].sum(axis=1)
    df_sust_indicator['esg_score'] = df_sust_indicator[['e_score', 's_score', 'g_score']].sum(axis=1)

    df_sust_indicator['e_pillar'] = df_sust_indicator['e_score'] / df_sust_indicator['e_weight']
    df_sust_indicator['s_pillar'] = df_sust_indicator['s_score'] / df_sust_indicator['s_weight']
    df_sust_indicator['g_pillar'] = df_sust_indicator['g_score'] / df_sust_indicator['g_weight']
    df_sust_indicator['esg_pillar'] = df_sust_indicator['esg_score'] / df_sust_indicator['esg_weight']

    # only keeping E/S/G values needed and dropping missing values
    df_sust_indicator = df_sust_indicator.iloc[:, -12:]
    df_sust_indicator = df_sust_indicator.dropna() # dropping missing values

    # converting to end-of-month dates
    df_sust_indicator = df_sust_indicator.reset_index()
    df_sust_indicator = df_sust_indicator.rename(columns={'File_date': 'date'})
    df_sust_indicator = df_sust_indicator.set_index(pd.to_datetime(df_sust_indicator['date'], format='%Y-%m-%d'))
    df_sust_indicator.index = df_sust_indicator.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_sust_indicator = df_sust_indicator.drop('date', axis=1)
    df_sust_indicator = df_sust_indicator.reset_index()

    # loading capIQ id's
    df_capiqid = pd.read_csv(DATA_RAW / "sustainalytics" / "IDS_TABLE.csv", encoding='latin1')
    df_capiqid.columns = ['CIQ_ID', 'CompanyId_old', 'EntityId', 'CompanyName']

    # merging together to get CapIQ ID
    df_sust_indicator = df_sust_indicator.merge(df_capiqid, how='left', on=['EntityId'])

    # loading other reference data
    zf = zipfile.ZipFile(DATA_RAW / "sustainalytics" / "sust_ref_data.zip")
    df_sust_ref = pd.read_csv(zf.open('vqywhnnzjul6s0cp.csv', 'r'), encoding='unicode_escape')
    zf.close()

    df_sust_ref = df_sust_ref.reset_index()
    df_sust_ref = df_sust_ref.rename(columns={'ExtractionDate': 'date'})
    df_sust_ref = df_sust_ref.set_index(pd.to_datetime(df_sust_ref['date'], format='%Y-%m-%d'))
    df_sust_ref.index = df_sust_ref.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_sust_ref = df_sust_ref.drop('date', axis=1)
    df_sust_ref = df_sust_ref.reset_index()

    # merging with reference data to see if it provides more info
    df_sust_indicator = df_sust_indicator.merge(df_sust_ref, how='left', on=['EntityId', 'date'])

    # creating capIQ id to gvkey bridge using historical information
    def ciq_gvkey():
        # loading gvkey and isin information from wrds
        wrds_gvkey = pd.read_csv(DATA_RAW / "linking" / "WRDS_GVKEY.csv")

        wrds_gvkey = wrds_gvkey[wrds_gvkey['companyid'].notna()]
        wrds_gvkey = wrds_gvkey[wrds_gvkey['gvkey'].notna()]

        # replacing beginning / end values with early and late dates
        wrds_gvkey['startdate'] = wrds_gvkey['startdate'].replace("B", '19000101')
        wrds_gvkey['enddate'] = wrds_gvkey['enddate'].replace("E", '20231231')

        wrds_gvkey['startdate'] = pd.to_datetime(wrds_gvkey['startdate'], format='%Y%m%d')
        wrds_gvkey['enddate'] = pd.to_datetime(wrds_gvkey['enddate'], format='%Y%m%d')

        wrds_gvkey = wrds_gvkey[
            (wrds_gvkey.startdate <= wrds_gvkey.enddate)]  # a few with dates too far away to be needed

        wrds_gvkey.gvkey = wrds_gvkey.gvkey.astype(int)

        wrds_gvkey = wrds_gvkey.rename(columns={'companyid': 'CIQ_ID'})  # renaming before merging
        wrds_gvkey['CIQ_ID'] = "IQ" + wrds_gvkey['CIQ_ID'].astype(str)

        return wrds_gvkey
    wrds_gvkey = ciq_gvkey()

    df_sust_indicator['year'] = df_sust_indicator.date.dt.year

    # merging to get GVKEY
    df_sust_indicator = df_sust_indicator.merge(wrds_gvkey, how='left', on=['CIQ_ID'])

    # keep only if date >= start_date and <= end_date of link (but only for those with a match)
    df_sust_indicator.loc[df_sust_indicator['date'] < df_sust_indicator['startdate'], 'gvkey'] = np.nan
    df_sust_indicator.loc[df_sust_indicator['date'] > df_sust_indicator['enddate'], 'gvkey'] = np.nan

    # for those missing, using bigger header table
    df_gnames = pd.read_csv(DATA_RAW / "sustainalytics" / "G_NAMES.csv")  # only global names and likely header values
    df_gnames = df_gnames.rename(columns={'isin': 'ISIN'})  # renaming before merging
    df_gnames = df_gnames[df_gnames.gvkey.notna()]
    df_gnames = df_gnames[df_gnames.year1.notna()]
    df_gnames = df_gnames[df_gnames.year2.notna()]
    df_gnames = df_gnames[df_gnames['ISIN'].notna()]
    df_gnames[['year1', 'year2']] = df_gnames[['year1', 'year2']].astype(int)

    # merging gnames
    df_sust_indicator = df_sust_indicator.merge(df_gnames[['gvkey', 'ISIN', 'conm', 'year1', 'year2']], how='left',
                                                on=['ISIN'])
    df_sust_indicator.loc[df_sust_indicator['year'] < df_sust_indicator['year1'], 'gvkey_y'] = np.nan
    df_sust_indicator.loc[df_sust_indicator['year'] > df_sust_indicator['year2'], 'gvkey_y'] = np.nan

    # checking disagreements between the two databases (only very few)
    (df_sust_indicator[(df_sust_indicator.gvkey_x.notna()) & (df_sust_indicator.gvkey_y.notna())]['gvkey_x'] ==
     df_sust_indicator[(df_sust_indicator.gvkey_x.notna()) & (df_sust_indicator.gvkey_y.notna())][
         'gvkey_y']).value_counts()

    # prefer to use historical bridge, otherwise header
    df_sust_indicator['gvkey'] = df_sust_indicator['gvkey_x'].fillna(df_sust_indicator['gvkey_y'])

    # delete missing gvkeys, dropping duplicates, keeping info key info
    df_sust_indicator = df_sust_indicator[df_sust_indicator.gvkey.notna()]
    df_sust_indicator = df_sust_indicator.drop_duplicates(['EntityId', 'date', 'gvkey'])
    df_sust_indicator = df_sust_indicator[['gvkey', 'date', 'e_pillar', 'esg_pillar']]
    df_sust_indicator['gvkey'] = df_sust_indicator['gvkey'].astype(int)
    df_sust_indicator = df_sust_indicator.rename(
        columns={'esg_pillar': 'total_esg_score', 'e_pillar': 'environment_score'})  # renaming before merging
    df_sust_indicator = df_sust_indicator.drop_duplicates() #dropping exact duplicates

    df_sust_indicator = df_sust_indicator.drop_duplicates(['gvkey', 'date'], keep=False)

    # now using legacy data: more straightforward as just need ESG and E score plus a gvkey
    cols = ['Date', 'Company', 'ISIN', 'CapitalIQ_ID', 'CompanyId', 'Country', 'total_esg_score', 'environment_score']

    zf = zipfile.ZipFile(DATA_RAW / "sustainalytics" / "sust_legacy.zip")
    df_sust_legacy = pd.read_csv(zf.open('orvxvkrcejp13hs6.csv', 'r'), usecols=cols, encoding='unicode_escape')
    zf.close()

    df_sust_legacy = df_sust_legacy[df_sust_legacy.total_esg_score.notna()]
    df_sust_legacy = df_sust_legacy[df_sust_legacy.environment_score.notna()]

    df_sust_legacy = df_sust_legacy[df_sust_legacy.CompanyId.str[0:2] != "IQ"]  # dropping a few wrongly shifted obs

    df_sust_legacy = df_sust_legacy.rename(columns={'CapitalIQ_ID': 'CIQ_ID', 'Date': 'date'})  # renaming before merging

    # creating time index
    df_sust_legacy = df_sust_legacy.set_index(pd.to_datetime(df_sust_legacy['date'], format='%Y-%m-%d'))
    df_sust_legacy.index = df_sust_legacy.index.to_period('M').to_timestamp('M')  # converting to end-of-month dates
    df_sust_legacy = df_sust_legacy.drop('date', axis=1)
    df_sust_legacy = df_sust_legacy.reset_index()
    df_sust_legacy['year'] = df_sust_legacy.date.dt.year

    # merging gvkey bridge based on CapIQ ID
    df_sust_legacy = df_sust_legacy.merge(wrds_gvkey, how='left', on=['CIQ_ID'])

    # keep only if date >= start_date and <= end_date of link (but only for those with a match)
    df_sust_legacy.loc[df_sust_legacy['date'] < df_sust_legacy['startdate'], 'gvkey'] = np.nan
    df_sust_legacy.loc[df_sust_legacy['date'] > df_sust_legacy['enddate'], 'gvkey'] = np.nan

    # merging alternative gvkey table using ISIN
    df_sust_legacy = df_sust_legacy.merge(df_gnames[['gvkey', 'ISIN', 'conm', 'year1', 'year2']], how='left', on=['ISIN'])
    df_sust_legacy.loc[df_sust_legacy['year'] < df_sust_legacy['year1'], 'gvkey_y'] = np.nan
    df_sust_legacy.loc[df_sust_legacy['year'] > df_sust_legacy['year2'], 'gvkey_y'] = np.nan

    # again preferring historical bridge to header bridge
    df_sust_legacy['gvkey'] = df_sust_legacy['gvkey_x'].fillna(df_sust_legacy['gvkey_y'])

    df_sust_legacy = df_sust_legacy[df_sust_legacy.gvkey.notna()]

    # handling duplicates
    df_sust_legacy = df_sust_legacy.drop_duplicates(subset=['gvkey', 'date', 'total_esg_score'], keep='last')
    df_sust_legacy = df_sust_legacy.drop_duplicates(subset=['gvkey', 'date'], keep='last')

    df_sust_legacy = df_sust_legacy[['date', 'gvkey', 'total_esg_score', 'environment_score']]
    df_sust_legacy['gvkey'] = df_sust_legacy['gvkey'].astype(int)
    df_sust_legacy['total_esg_score'] = df_sust_legacy['total_esg_score'].astype(float)

    return df_sust_indicator, df_sust_legacy

# EPA TRI
    return epa_tri