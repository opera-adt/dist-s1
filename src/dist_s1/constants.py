import numpy as np
import pandas as pd


PRODUCT_VERSION = '0.1'

# Tolerance constants for layer comparison
MAX_FLOAT_LAYER_DIFF = 2e-5
MAX_INT_LAYER_DIFF = 0

# Confirmation
BASE_DATE_FOR_CONFIRMATION = pd.Timestamp('2020-12-31', tz='UTC')

# Disturbance labels
DISTLABEL2VAL = {
    'nodata': 255,
    'no_disturbance': 0,
    'first_low_conf_disturbance': 1,
    'provisional_low_conf_disturbance': 2,
    'confirmed_low_conf_disturbance': 3,
    'first_high_conf_disturbance': 4,
    'provisional_high_conf_disturbance': 5,
    'confirmed_high_conf_disturbance': 6,
    'confirmed_low_conf_disturbance_finished': 7,
    'confirmed_high_conf_disturbance_finished': 8,
}
DISTVAL2LABEL = {v: k for k, v in DISTLABEL2VAL.items()}

TIF_LAYER_DTYPES = {
    'GEN-DIST-STATUS': 'uint8',
    'GEN-METRIC': 'float32',
    'GEN-DIST-STATUS-ACQ': 'uint8',
    'GEN-METRIC-MAX': 'float32',
    'GEN-DIST-CONF': 'float32',
    'GEN-DIST-DATE': 'int16',
    'GEN-DIST-COUNT': 'uint8',
    'GEN-DIST-PERC': 'uint8',
    'GEN-DIST-DUR': 'int16',
    'GEN-DIST-LAST-DATE': 'int16',
}
TIF_LAYER_NODATA_VALUES = {
    'GEN-DIST-STATUS': 255,
    'GEN-DIST-STATUS-ACQ': 255,
    'GEN-METRIC': np.nan,
    'GEN-METRIC-MAX': np.nan,
    'GEN-DIST-CONF': np.nan,
    'GEN-DIST-DATE': -1,
    'GEN-DIST-COUNT': 255,
    'GEN-DIST-PERC': 255,
    'GEN-DIST-DUR': -1,
    'GEN-DIST-LAST-DATE': -1,
}
TIF_LAYER_DESCRIPTIONS = {
    'GEN-DIST-STATUS': 'Status of the generic disturbance classification (see the disturbance labels table for more '
    'details on the status labels).',
    'GEN-METRIC': 'Metric value for the generic disturbance classification. Can be viewed as number of standard '
    'devations from the mean. Value is a non-negative real number.',
    'GEN-DIST-STATUS-ACQ': 'Status of the generic disturbance classification with respect to the latest acquisition '
    'date (see disturbance labels table for more details on the status labels)',
    'GEN-METRIC-MAX': 'Maximum metric value for the generic disturbance classification over all acquisition dates since'
    'first disturbance. Reset to 0 when a new disturbance is detected. Value is a non-negative real number.',
    'GEN-DIST-CONF': 'Confidence level for the generic disturbance classification. Value is a non-negative real number.'
    ' Reset to 0 when a new disturbance is detected. Nan is nodata or no acquisition data available over previous '
    'dates.',
    'GEN-DIST-DATE': 'Date of the generic disturbance classification. Value is a non-negative integer and is the number'
    f' of days from {BASE_DATE_FOR_CONFIRMATION.strftime("%Y-%m-%d")}. -1 is nodata or no acquisition data available.'
    'over previous dates.',
    'GEN-DIST-COUNT': 'The number of generic disturbances since first detection. Value is a non-negative integer.',
    'GEN-DIST-PERC': 'Percentage of the generic disturbance disturbance since first detection.',
    'GEN-DIST-DUR': 'Duration of the generic disturbance classification since first detection in days.',
    'GEN-DIST-LAST-DATE': 'Latest generic disturbance detection.',
}
TIF_LAYERS = TIF_LAYER_DTYPES.keys()
EXPECTED_FORMAT_STRING = (
    'OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{PRODUCT_VERSION}'
)


# Colormaps
DIST_STATUS_CMAP = {
    0: (18, 18, 18, 255),  # No disturbance
    1: (0, 85, 85, 255),  # First low
    2: (137, 127, 78, 255),  # Provisional low
    3: (222, 224, 67, 255),  # Confrimed low
    4: (0, 136, 136, 255),  # First high
    5: (228, 135, 39, 255),  # Provisional high
    6: (224, 27, 7, 255),  # Confirmed high
    7: (119, 119, 119, 255),  # Confirmed low finished
    8: (221, 221, 221, 255),  # Confirmed high finished
    255: (0, 0, 0, 255),  # No data
}

DIST_STATUS_LABEL_DESCRIPTIONS = {
    0: 'No disturbance detected.',
    1: 'First low disturbance detected.',
    2: 'Provisional low disturbance detected meaning 2 disturbances were detected in the allowable '
    'confirmation window.',
    3: 'Confirmed low disturbance detected meaning *at least* 3 disturbances were detected in the allowable '
    'confirmation window.',
    4: 'First high disturbance detected.',
    5: 'Provisional high disturbance detected meaning 2 disturbances were detected in the allowable '
    'confirmation window.',
    6: 'Confirmed high disturbance detected meaning *at least* 3 disturbances were detected in the allowable '
    'confirmation window.',
    7: 'Confirmed low disturbance finished meaning the disturbance has been confirmed and the pixel has returned to '
    'baseline status.',
    8: 'Confirmed high disturbance finished meaning the disturbance has been confirmed and the pixel has returned to '
    'baseline status.',
    255: 'No data available.',
}

DISTLABEL2CBARLABEL = {
    'nodata': 'No Data',
    'no_disturbance': 'No Dist',
    'first_low_conf_disturbance': '1st Low',
    'provisional_low_conf_disturbance': 'Prov Low',
    'confirmed_low_conf_disturbance': 'Conf Low',
    'first_high_conf_disturbance': '1st High',
    'provisional_high_conf_disturbance': 'Prov High',
    'confirmed_high_conf_disturbance': 'Conf High',
    'confirmed_low_conf_disturbance_finished': 'Low Fin',
    'confirmed_high_conf_disturbance_finished': 'High Fin',
}
DISTVAL2CBARLABEL = {k: DISTLABEL2CBARLABEL[v] for k, v in DISTVAL2LABEL.items()}
