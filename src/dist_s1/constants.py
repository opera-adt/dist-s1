PRODUCT_VERSION = '0.1'

MODEL_CONTEXT_LENGTH = 10

DISTLABEL2VAL = {
    'nodata': 255,
    'no_disturbance': 0,
    'first_moderate_conf_disturbance': 1,
    'provisional_moderate_conf_disturbance': 2,
    'confirmed_moderate_conf_disturbance': 3,
    'first_high_conf_disturbance': 4,
    'provisional_high_conf_disturbance': 5,
    'confirmed_high_conf_disturbance': 6,
}
DISTVAL2LABEL = {v: k for k, v in DISTLABEL2VAL.items()}

COLORBLIND_DIST_CMAP = {
    0: (255, 255, 255, 255),  # No disturbance (White)
    1: (173, 216, 230, 255),  # First low (Light Blue)
    2: (100, 149, 237, 255),  # Provisional low (Cornflower Blue)
    3: (25, 25, 112, 255),  # Confirmed low (Midnight Blue)
    7: (10, 10, 60, 255),  # Confirmed low finished (Very Dark Blue)
    4: (255, 182, 193, 255),  # First high (Light Pink)
    5: (255, 99, 71, 255),  # Provisional high (Tomato Red)
    6: (178, 34, 34, 255),  # Confirmed high (Firebrick Red)
    8: (139, 0, 0, 255),  # Confirmed high finished (Dark Red)
    255: (128, 128, 128, 255),  # No data (Grey)
}

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
