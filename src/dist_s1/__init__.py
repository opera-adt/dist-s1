import warnings
from importlib.metadata import PackageNotFoundError, version

from .data_models.runconfig_model import RunConfigData
from .workflows import (
    run_burst_disturbance_workflow,
    run_despeckle_workflow,
    run_dist_s1_localization_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
    run_disturbance_merge_workflow,
    run_normal_param_estimation_workflow,
)


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = None
    warnings.warn(
        'package is not installed!\n'
        'Install in editable/develop mode via (from the top of this repo):\n'
        '   python -m pip install -e .\n',
        RuntimeWarning,
    )


__all__ = [
    'RunConfigData',
    'run_dist_s1_workflow',
    'run_dist_s1_sas_workflow',
    'run_dist_s1_localization_workflow',
    'run_normal_param_estimation_workflow',
    'run_burst_disturbance_workflow',
    'run_despeckle_workflow',
    'run_disturbance_merge_workflow',
]
