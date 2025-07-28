import datetime
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import rasterio.errors
from rasterio.transform import from_origin

from dist_s1.confirmation import compute_tile_disturbance_using_previous_product_and_serialize


def generate_status_test_series() -> tuple[list[np.ndarray], dict[str, tuple[int, int]]]:
    # Configuration: Durations required to trigger states
    PROV_DURATION = 2
    CONF_DURATION = 10
    FIN_QUIET_DURATION = 10

    num_arrays = CONF_DURATION + FIN_QUIET_DURATION + 1
    final_t = num_arrays - 1

    size = 10
    arrays = []
    base = np.zeros((size, size), dtype=np.float32)

    # Define a unique coordinate for each status's representative pixel.
    coords = {
        'NODIST': (0, 0),  # No disturbance
        'NODATA': (0, 1),  # No data (will be set to NaN)
        'FIRSTHI': (1, 0),  # First high anomaly
        'FIRSTLO': (1, 1),  # First low anomaly
        'PROVHI': (2, 0),  # Provisional high
        'PROVLO': (2, 1),  # Provisional low
        'CONFHI': (3, 0),  # Confirmed high
        'CONFLO': (3, 1),  # Confirmed low
        'CONFHIFIN': (4, 0),  # Finalized confirmed high
        'CONFLOFIN': (4, 1),  # Finalized confirmed low
    }

    # High and low anomaly values.
    HIGH_ANOMALY_VALUE = 6.0
    LOW_ANOMALY_VALUE = 4.0

    for t in range(num_arrays):
        frame = base.copy()

        if final_t - PROV_DURATION < t <= final_t:
            frame[coords['PROVHI']] = HIGH_ANOMALY_VALUE

        if final_t - PROV_DURATION < t <= final_t:
            frame[coords['PROVLO']] = LOW_ANOMALY_VALUE

        if final_t - CONF_DURATION < t <= final_t:
            frame[coords['CONFHI']] = HIGH_ANOMALY_VALUE

        if final_t - CONF_DURATION < t <= final_t:
            frame[coords['CONFLO']] = LOW_ANOMALY_VALUE

        if 0 <= t < CONF_DURATION:
            frame[coords['CONFHIFIN']] = HIGH_ANOMALY_VALUE

        if 0 <= t < CONF_DURATION:
            frame[coords['CONFLOFIN']] = LOW_ANOMALY_VALUE

        if t == final_t:
            frame[coords['FIRSTHI']] = HIGH_ANOMALY_VALUE
            frame[coords['FIRSTLO']] = LOW_ANOMALY_VALUE
            frame[coords['NODATA']] = np.nan

        arrays.append(frame)

    return arrays, coords


def write_raster(path: str | Path, array: np.ndarray, dtype: str = 'float32', nodata_val: int = -9999) -> None:
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs='EPSG:4326',
        transform=from_origin(0, 0, 1, 1),
        nodata=nodata_val,
    ) as dst:
        dst.write(np.nan_to_num(array, nan=nodata_val).astype(dtype), 1)


def read_raster(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def test_disturbance_status_series(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
        arrays, slices = generate_status_test_series()
        base_date = datetime.datetime(2023, 7, 1)

        # Paths to track previous data
        previous_paths = None
        output_products = [tmp_path / f'product_{i}' for i in range(len(arrays))]

        for t, frame in enumerate(arrays):
            date = base_date + datetime.timedelta(days=t)
            date_tag = date.strftime('%Y%m%dT000000Z')
            metric_path = tmp_path / f'dist_metric_dummy1_dummy2_{date_tag}_metric.tif'
            write_raster(metric_path, frame)

            output_paths = [tmp_path / f'out_status_{t}_{i}.tif' for i in range(8)]

            compute_tile_disturbance_using_previous_product_and_serialize(
                metric_path,
                metric_path,
                [str(p) for p in output_paths],
                previous_dist_arr_path_list=[str(p) for p in previous_paths] if previous_paths else None,
                base_date_for_confirmation=base_date,
                alert_low_conf_thresh=3.5,
                alert_high_conf_thresh=5.5,
            )

            # Prepare next timestep input
            previous_paths = output_paths
            output_products[t] = output_paths[0]  # status band

        # Final status map
        final_status = read_raster(output_products[-1])

        expected_codes = {
            'FIRSTLO': 1,
            'PROVLO': 2,
            'CONFLO': 3,
            'FIRSTHI': 4,
            'PROVHI': 5,
            'CONFHI': 6,
            'CONFLOFIN': 7,
            'CONFHIFIN': 8,
        }

        for label, expected_val in expected_codes.items():
            r, c = slices[label]
            sub = final_status[r, c]
            assert (sub == expected_val).any(), f'{label} block missing expected code {expected_val}'
