import warnings
from importlib.metadata import PackageNotFoundError, version

from dist_s1.workflows import (
    run_burst_disturbance_workflow,
    run_confirmation_of_dist_product_workflow,
    run_despeckle_workflow,
    run_dist_s1_localization_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
    run_disturbance_merge_workflow,
    run_sequential_confirmation_of_dist_products_workflow,
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
    'run_dist_s1_workflow',
    'run_dist_s1_sas_workflow',
    'run_dist_s1_localization_workflow',
    'run_burst_disturbance_workflow',
    'run_despeckle_workflow',
    'run_disturbance_merge_workflow',
    'run_confirmation_of_dist_product_workflow',
    'run_sequential_confirmation_of_dist_products_workflow',
]
