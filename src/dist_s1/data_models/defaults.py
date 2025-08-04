"""Default configuration values for dist-s1.

This module provides a single source of default values used across
the data models and CLI to ensure consistency.
"""

from pathlib import Path


# =============================================================================
# Algorithm Configuration Defaults
# =============================================================================

# Enumeration settings - how to curate the baseline
DEFAULT_LOOKBACK_STRATEGY = 'multi_window'
DEFAULT_MAX_PRE_IMGS_PER_BURST_MW = None
DEFAULT_DELTA_LOOKBACK_DAYS_MW = None
DEFAULT_POST_DATE_BUFFER_DAYS = 1
DEFAULT_N_ANNIVERSARIES_FOR_MW = 3

# Despeckling settings
DEFAULT_N_WORKERS_FOR_DESPECKLING = 8
DEFAULT_INTERPOLATION_METHOD = 'bilinear'
DEFAULT_APPLY_DESPECKLING = True

# Inference settings - how to run the model
DEFAULT_DEVICE = 'best'
DEFAULT_MEMORY_STRATEGY = 'high'
DEFAULT_TQDM_ENABLED = True
DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION = 8
DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION = 32
DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION = 16

# Model settings
DEFAULT_MODEL_SOURCE = 'transformer_optimized'
DEFAULT_MODEL_CFG_PATH = None
DEFAULT_MODEL_WTS_PATH = None
DEFAULT_APPLY_LOGIT_TO_INPUTS = True
DEFAULT_MODEL_COMPILATION = False
DEFAULT_MODEL_DTYPE = 'float32'
DEFAULT_USE_DATE_ENCODING = False

# Alert Disturbance Settings - Confidence thresholds and limits
DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD = 3.5
DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD = 5.5

# Confirmation Settings - Confidence thresholds and limits
DEFAULT_NO_DAY_LIMIT = 30
DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST = True
DEFAULT_PERCENT_RESET_THRESH = 10
DEFAULT_NO_COUNT_RESET_THRESH = 7
DEFAULT_MAX_OBS_NUM_YEAR = 253
DEFAULT_CONFIDENCE_UPPER_LIM = 32000
DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD = 3**2 * 3.5  # 31.5
DEFAULT_METRIC_VALUE_UPPER_LIM = 100.0


# =============================================================================
# Run Configuration Defaults for Filepaths Handling
# =============================================================================

# Directory and file paths
DEFAULT_DST_DIR = Path('out')
DEFAULT_INPUT_DATA_DIR = None
DEFAULT_PRODUCT_DST_DIR = None
DEFAULT_WATER_MASK_PATH = None
DEFAULT_ALGO_CONFIG_PATH = None

# Water Mask settings
DEFAULT_APPLY_WATER_MASK = True
DEFAULT_CHECK_INPUT_PATHS = True

# AWS/Cloud settings - if using Hyp3
DEFAULT_BUCKET = None
DEFAULT_BUCKET_PREFIX = None

# Prior product settings for confirmation
DEFAULT_PRIOR_DIST_S1_PRODUCT = None
