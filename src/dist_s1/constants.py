MODEL_CONTEXT_LENGTH = 10
N_LOOKBACKS = 3

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

MODERATE_CONF_THRESHOLD = 2.5
HIGH_CONF_THRESHOLD = 4.5
