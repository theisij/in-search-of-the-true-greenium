######## packages ########
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
from linearmodels.shared.exceptions import MissingValueWarning

# import janitor as jn # for conditional_join
# from scipy.optimize import minimize

######## hiding certain warnings and choosing float output format ########
warnings.filterwarnings('ignore', category=MissingValueWarning) # suppress missing value warning in regressions
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning) #suppressing import warning on dtypes
warnings.filterwarnings("ignore", category=FutureWarning) # suppressing future package changes (remove to see errors)
pd.options.display.float_format = '{:.5f}'.format

# General settings
FULL_RUN = True  # Reproducibility baseline: regenerate all intermediates from raw inputs. Set to False during local iteration to reuse cached data/interim parquets.
global_start = '2009-08-01' # setting the global start date of the analysis
IMPUTE_START = False
IMPUTE_START_DATE = date(2009, 8, 1) #'2009-08-01' # Start date for global analysis 
set_start = date(2012, 12, 31) # Start date for the analysis
#set_start = '2012-12-31' # Start date for the analysis
set_start_morningstar = date(2018, 10, 31) # Start date for MS sus_fund

# esg factors to be tested
total_em_factors = ['log_scope1tot', 'log_scope12tot', 'log_scope123tot']
emission_factors = ['scope1int_sales', 'scope12int_sales', 'scope123int_sales',
                    'scope1int_assets', 'scope12int_assets', 'scope123int_assets']
msci_factors = ['industry_esg_score', 'weighted_esg_score', 'e_score', 'greenness',
                'e_climate_score', 'e_natres_score', 'e_waste_score', 'e_envopps_score']
sust_factors = ['total_esg_score', 'environment_score']

# choosing main factors to test (including BK's total emissions in return analysis, but excluding later in ICC etc.)
to_test = total_em_factors + emission_factors + msci_factors + sust_factors
to_test_g = total_em_factors + emission_factors + msci_factors + sust_factors

# name of factors for outputs
to_test_g_names = ['LOG(S1TOT)', 'LOG(S1+2TOT)', 'LOG(S1+2+3TOT)', 'S1INT (Sales)', 'S1+2INT (Sales)', 'S1+2+3INT (Sales)', 'S1INT (Assets)', 'S1+2INT (Assets)', 'S1+2+3INT (Assets)',
                         'Ind.-adj. ESG score', 'Weighted ESG score', 'Environment score', 'Greenness (PST)', 'E climate score', 'E nat. res. score', 'E waste score', 'E env. opps. score',
                         'Total ESG score', 'Environmental score']
# Dictionary to map factors to their names
to_test_g_dict = dict(zip(to_test_g, to_test_g_names))
to_test_g_dict['green_score'] = 'Aggregate green score'

# Main controls 
MAIN_CONTROLS = ['beta_252d', 'ebit_at', 'netdebt_at', 'log_be']
# Aggregate green score variables to consider for benchmarking
AGG_GREEN_VARS = ['gs_ew',
                  'gs_direct_rank_imp',
                  'gs_direct_raw', 'gs_direct_log', 'gs_direct_rank',
                  'gs_direct_raw_bf', 'gs_direct_log_bf', 'gs_direct_rank_bf',
                  'gs_proj_raw', 'gs_proj_log', 'gs_proj_rank']
# Main green score variable
AGG_GREEN_VAR_MAIN = 'gs_direct_rank_imp' 
# Bootstrap settings
BS_SEED = 1
BS_REPEATS = 100
BS_SAMPLE_GROUP = 'excntry' # Group variable to sample bootstrap samples from. 
                            # For example, 'excntry' will sample countries.

# Relative project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Data/
DATA = PROJECT_ROOT / "data" 
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
# Output/
OUTPUT = PROJECT_ROOT / "output"
OUTPUT_TABLES = OUTPUT / "tables"
OUTPUT_TABLES_DESCRIPTIVES = OUTPUT_TABLES / "des_stats"
OUTPUT_TABLES_GLOBAL = OUTPUT_TABLES / "global"
OUTPUT_FIGURES = OUTPUT / "figures"
OUTPUT_FIGURES_DESCRIPTIVES = OUTPUT_FIGURES / "des_stats"
OUTPUT_FIGURES_GLOBAL = OUTPUT_FIGURES / "global"
OUTPUT_FIGURES_US = OUTPUT_FIGURES / "us"
OUTPUT_FIGURES_MAPS = OUTPUT_FIGURES / "maps"
