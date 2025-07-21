"""Default configuration values for dist-s1.

This module provides a single source of default values used across
the data models and CLI to ensure consistency.
"""

from pathlib import Path


# =============================================================================
# Algorithm Configuration Defaults
# =============================================================================

# Device and performance settings
DEFAULT_DEVICE = 'best'  # Resolved to best available device
DEFAULT_MEMORY_STRATEGY = 'high'
DEFAULT_TQDM_ENABLED = True
DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION = 8
DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION = 32
DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION = 16
DEFAULT_N_WORKERS_FOR_DESPECKLING = 8

# Algorithm strategy settings
DEFAULT_LOOKBACK_STRATEGY = 'multi_window'
DEFAULT_MODEL_COMPILATION = False
DEFAULT_MAX_PRE_IMGS_PER_BURST_MW = [5, 5]
DEFAULT_DELTA_LOOKBACK_DAYS_MW = [730, 365]

# Confidence thresholds and limits
DEFAULT_MODERATE_CONFIDENCE_THRESHOLD = 3.5
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 5.5
DEFAULT_NODAYLIMIT = 18
DEFAULT_MAX_OBS_NUM_YEAR = 253
DEFAULT_CONF_UPPER_LIM = 32000
DEFAULT_CONF_THRESH = 3**2 * 3.5  # 31.5
DEFAULT_METRIC_VALUE_UPPER_LIM = 100.0

# Model settings
DEFAULT_MODEL_SOURCE = 'transformer_optimized'
DEFAULT_MODEL_CFG_PATH = None
DEFAULT_MODEL_WTS_PATH = None
DEFAULT_APPLY_LOGIT_TO_INPUTS = True
DEFAULT_APPLY_DESPECKLING = True
DEFAULT_INTERPOLATION_METHOD = 'bilinear'
DEFAULT_MODEL_DTYPE = 'float32'
DEFAULT_USE_DATE_ENCODING = False

# =============================================================================
# Run Configuration Defaults
# =============================================================================

# Directory and file paths
DEFAULT_DST_DIR = Path('out')
DEFAULT_PRODUCT_DST_DIR = None
DEFAULT_WATER_MASK_PATH = None
DEFAULT_ALGO_CONFIG_PATH = None

# Processing settings
DEFAULT_APPLY_WATER_MASK = True
DEFAULT_CHECK_INPUT_PATHS = True

# AWS/Cloud settings
DEFAULT_BUCKET = None
DEFAULT_BUCKET_PREFIX = None

# Prior product settings
DEFAULT_PRIOR_DIST_S1_PRODUCT = None

# =============================================================================
# CLI-specific Defaults (not in data models)
# =============================================================================

# Workflow parameters
DEFAULT_POST_DATE_BUFFER_DAYS = 1
DEFAULT_INPUT_DATA_DIR = None

# =============================================================================
# String versions for CLI parsing
# =============================================================================

# For CLI options that need string representations
DEFAULT_DST_DIR_STR = 'out/'
