from pathlib import Path

import numpy as np
import rasterio

from .input_data_model import RunConfigModel


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


def run_dist_s1_workflow(run_config: RunConfigModel):
    sample_burst_id = sorted(list(run_config.time_series_by_burst.keys()))[0]

    dist = compute_metric_for_one_burst(
        run_config.time_series_by_burst[sample_burst_id]['pre_rtc_crosspol'],
        run_config.time_series_by_burst[sample_burst_id]['post_rtc_crosspol'][0],
    )

    dist_map = dist < -1

    with rasterio.open(run_config.time_series_by_burst[sample_burst_id]['post_rtc_copol'][0]) as ds:
        profile = ds.profile

    mgrs_tile_id = run_config.mgrs_tile_id
    output_path = run_config.output_product_dir / f'disturbance_{mgrs_tile_id}.tif'
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dist_map.astype(np.uint8), 1)
