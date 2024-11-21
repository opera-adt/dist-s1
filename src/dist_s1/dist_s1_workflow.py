from pathlib import Path

import numpy as np
import rasterio

from .data_models.output_models import LAYER_DTYPES
from .data_models.runconfig_model import RunConfigData


def _one_one_arr(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def compute_metric_for_one_burst(
    pre_rtc_crosspol_paths: list[str],
    post_rtc_crosspol_path: str,
):
    pre_arrs = [_one_one_arr(path) for path in pre_rtc_crosspol_paths]
    pre_arrs_stacked = np.stack(pre_arrs, axis=0)
    pre_arrs_median = np.median(pre_arrs_stacked, axis=0)

    post_arr = _one_one_arr(post_rtc_crosspol_path)
    dist = np.log10(post_arr) - np.log10(pre_arrs_median)
    return dist


def run_dist_s1_workflow(run_config: RunConfigData) -> Path:
    # Get the first burst in the time series
    sample_burst_id = sorted(list(run_config.time_series_by_burst.keys()))[0]

    dist = compute_metric_for_one_burst(
        run_config.time_series_by_burst[sample_burst_id]['pre_rtc_crosspol'],
        run_config.time_series_by_burst[sample_burst_id]['post_rtc_crosspol'][0],
    )

    dummy_data = (dist < -1).astype(np.uint8)

    with rasterio.open(run_config.time_series_by_burst[sample_burst_id]['post_rtc_copol'][0]) as ds:
        profile = ds.profile

    output_prod_data = run_config.product_dir_data
    for layer, path in output_prod_data.layer_path_dict.items():
        profile['dtype'] = LAYER_DTYPES[layer]
        profile['nodata'] = None
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(dummy_data.astype(LAYER_DTYPES[layer]), 1)

    return output_prod_data
