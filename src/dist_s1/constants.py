import pandas as pd


PRODUCT_VERSION = '0.1'

MODEL_CONTEXT_LENGTH = 10

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


# Colormaps
DIST_CMAP = {
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
