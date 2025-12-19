"""
Configuration module for all the framework.

This file centralizes all fixed settings used across the experimental
pipeline, including dataset selection rules, NaN-handling thresholds,
hyperparameter optimization parameters, reproducibility seeds and the
curated dataset/model lists used in the paper.

The goal is to guarantee full experimental reproducibility 
No logic for training or evaluation is implemented here; only static,
transparent configuration values.
"""

from pmlb import classification_dataset_names, regression_dataset_names
import random
import numpy as np


# ============================================================================
#                             GENERAL SETTINGS
# ============================================================================

# Exclude datasets whose names start with any of these prefixes
EXCLUDED_DS_START = ("_deprecated_", "feynman", "first_principle")

TARGET_NAME = 'target'


# ============================================================================
#                         NaN HANDLING CONFIGURATION
# ============================================================================

# Drop columns with more than this fraction of NaN values
MAX_NAN_SINGLE_COLUMN = 0.3

# If more than this fraction of columns is removed → discard entire dataset
REMAINING_COL_AFTER_NAN_COL_ELIMINATION = 0.5

# After dropping all row-level NaN, if fewer than this fraction remain → discard dataset
REMAINING_OBSERVATIONS_AFTER_NAN_DROP = 0.5


# ============================================================================
#                           DATASET FILTERING RULES
# ============================================================================

# Minimum required number of observations
MIN_OBSERVATIONS = 20

# Restrict classification datasets to binary targets
CLF_ONLY_BINARY = True


# ============================================================================
#                           REPRODUCIBILITY SETTINGS
# ============================================================================

SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

# Number of splits for cross-validation or repeated sampling
K_SPLITS = 4


# ============================================================================
#                 OOS EVALUATION CONFIGURATION (CLASSIFICATION)
# ============================================================================

F_OOS_CLF = 1 / 2
L_OOS_CLF = 1 / 8
U_OOS_CLF = 1 / 2


# ============================================================================
#              HYPERPARAMETER OPTIMIZATION (OPTUNA) SETTINGS
# ============================================================================

# Number of trials for Optuna optimization
N_TRIALS = 50

# Fraction of the data passed to Optuna reserved as internal TEST set
OPT_SINGLE_TEST_SIZE = 0.33


# ============================================================================
#                       PMLB DATASET CURATION (STATIC LISTS)
# ============================================================================
# NOTE:
# The lists below (original_clf_dataset_name, original_reg_dataset_name, full_datasets_names) represent 
# the full PMLB datasets available after automatic filtering by name.
# These lists are not used in the experiments included in the paper.
# For reproducibility reasons, the experiments rely on the curated hard-coded 
# lists `clf_dataset_names` and `regr_dataset_names`, which correspond to the 
# exact datasets tested and reported in the manuscript.

# Filter original lists
original_clf_dataset_name = [
    name for name in classification_dataset_names
    if not name.startswith(EXCLUDED_DS_START)
]

original_reg_dataset_name = [
    name for name in regression_dataset_names
    if not name.startswith(EXCLUDED_DS_START)
]

# Combined list (not used directly but kept for reference)
full_datasets_names = original_clf_dataset_name + original_reg_dataset_name


# ---------------------------------------------------------------------------
# Classification dataset list (fixed order for reproducibility)
# ---------------------------------------------------------------------------
clf_dataset_names = [
    'GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1',
    'GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1',
    'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1',
    'GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1',
    'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001',
    'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001',
    'Hill_Valley_with_noise', 'Hill_Valley_without_noise',
    'adult', 'agaricus_lepiota', 'analcatdata_aids',
    'analcatdata_asbestos', 'analcatdata_bankruptcy',
    'analcatdata_boxing1', 'analcatdata_boxing2',
    'analcatdata_creditscore', 'analcatdata_cyyoung8092',
    'analcatdata_cyyoung9302', 'analcatdata_fraud',
    'analcatdata_japansolvent', 'analcatdata_lawsuit',
    'appendicitis', 'backache', 'banana', 'biomed',
    'breast_cancer', 'breast_cancer_wisconsin_diagnostic',
    'breast_cancer_wisconsin_original', 'bupa', 'chess',
    'churn', 'clean1', 'clean2', 'coil2000',
    'congressional_voting_records', 'corral',
    'credit_approval_australia', 'credit_approval_germany',
    'dis', 'glass2', 'haberman', 'heart_disease_cleveland',
    'heart_disease_hungarian', 'heart_disease_va_long_beach',
    'heart_disease_zurich', 'hepatitis', 'hypothyroid',
    'ionosphere', 'irish', 'kr_vs_kp', 'labor', 'lupus',
    'magic', 'mofn_3_7_10', 'molecular_biology_promoters',
    'monk1', 'monk2', 'monk3', 'mushroom', 'mux6',
    'parity5', 'parity5+5', 'phoneme',
    'postoperative_patient_data', 'prnn_crabs', 'prnn_synth',
    'profb', 'ring', 'saheart', 'sonar', 'spambase',
    'spect', 'spectf', 'threeOf9', 'tic_tac_toe',
    'titanic', 'tokyo1', 'twonorm', 'xd6'
]


# ---------------------------------------------------------------------------
# Regression dataset list (fixed order for reproducibility)
# ---------------------------------------------------------------------------
regr_dataset_names = [
    '1027_ESL', '1028_SWD', '1029_LEV', '1030_ERA', '1089_USCrime',
    '1096_FacultySalaries', '1191_BNG_pbc', '1193_BNG_lowbwt',
    '1196_BNG_pharynx', '1199_BNG_echoMonths', '1201_BNG_breastTumor',
    '1203_BNG_pwLinear', '1595_poker', '192_vineyard', '197_cpu_act',
    '201_pol', '210_cloud', '215_2dplanes', '218_house_8L',
    '225_puma8NH', '227_cpu_small', '228_elusage', '229_pwLinear',
    '230_machine_cpu', '294_satellite_image', '344_mv',
    '4544_GeographicalOriginalofMusic', '485_analcatdata_vehicle',
    '503_wind', '505_tecator', '519_vinnie', '522_pm10',
    '523_analcatdata_neavote', '527_analcatdata_election2000',
    '529_pollen', '537_houses', '542_pollution', '547_no2',
    '556_analcatdata_apnea2', '557_analcatdata_apnea1', '560_bodyfat',
    '561_cpu', '562_cpu_small', '564_fried', '573_cpu_act',
    '574_house_16H', '579_fri_c0_250_5', '581_fri_c3_500_25',
    '582_fri_c1_500_25', '583_fri_c1_1000_50', '584_fri_c4_500_25',
    '586_fri_c3_1000_25', '588_fri_c4_1000_100', '589_fri_c2_1000_25',
    '590_fri_c0_1000_50', '591_fri_c1_100_10', '592_fri_c4_1000_25',
    '593_fri_c1_1000_10', '594_fri_c2_100_5', '595_fri_c0_1000_10',
    '596_fri_c2_250_5', '597_fri_c2_500_5', '598_fri_c0_1000_25',
    '599_fri_c2_1000_5', '601_fri_c1_250_5', '602_fri_c3_250_10',
    '603_fri_c0_250_50', '604_fri_c4_500_10', '605_fri_c2_250_25',
    '606_fri_c2_1000_10', '607_fri_c4_1000_50', '608_fri_c3_1000_10',
    '609_fri_c0_1000_5', '611_fri_c3_100_5', '612_fri_c1_1000_5',
    '613_fri_c3_250_5', '615_fri_c4_250_10', '616_fri_c4_500_50',
    '617_fri_c3_500_5', '618_fri_c3_1000_50', '620_fri_c1_1000_25',
    '621_fri_c0_100_10', '622_fri_c2_1000_50', '623_fri_c4_1000_10',
    '624_fri_c0_100_5', '626_fri_c2_500_50', '627_fri_c2_500_10',
    '628_fri_c3_1000_5', '631_fri_c1_500_5', '633_fri_c0_500_25',
    '634_fri_c2_100_10', '635_fri_c0_250_10', '637_fri_c1_500_50',
    '641_fri_c1_500_10', '643_fri_c2_500_25', '644_fri_c4_250_25',
    '645_fri_c3_500_50', '646_fri_c3_500_10', '647_fri_c1_250_10',
    '648_fri_c1_250_50', '649_fri_c0_500_5', '650_fri_c0_500_50',
    '651_fri_c0_100_25', '653_fri_c0_250_25', '654_fri_c0_500_10',
    '656_fri_c1_100_5', '657_fri_c2_250_10', '658_fri_c3_250_25',
    '659_sleuth_ex1714', '663_rabe_266', '665_sleuth_case2002',
    '666_rmftsa_ladata', '678_visualizing_environmental',
    '687_sleuth_ex1605', '690_visualizing_galaxy', '695_chatfield_4',
    '706_sleuth_case1202', '712_chscase_geyser1',
    'auto_insurance_losses', 'auto_insurance_price',
    'nikuradse_1', 'nikuradse_2', 'solar_flare',
    'strogatz_bacres1', 'strogatz_bacres2', 'strogatz_barmag1',
    'strogatz_barmag2', 'strogatz_glider1', 'strogatz_glider2',
    'strogatz_lv1', 'strogatz_lv2', 'strogatz_predprey1',
    'strogatz_predprey2', 'strogatz_shearflow1',
    'strogatz_shearflow2', 'strogatz_vdp1', 'strogatz_vdp2'
]


# ============================================================================
#                           MODEL NAME CONFIGURATION
# ============================================================================

MODELS_NAME = {
    "clf": [
        'knn',
        'decision_tree',
        'logistic_regression',
        'gosdt',
        'naive_bayes',
        'ebm',
        'igann'
    ],
    "regr": [
        'linear_regression',
        'decision_tree_regressor',
        'symbolic_regression',
        'knn',
        'ebm',
        'poly_lasso',
        'glm',
        'lasso',
        'igann'
    ],
}


# ============================================================================
#                               LOGGER SETTINGS
# ============================================================================

import logging

logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] %(message)s',
)

logger = logging.getLogger(__name__)
