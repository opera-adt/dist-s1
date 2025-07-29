import datetime
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from rasterio.transform import from_origin

from dist_s1.dist_processing import label_one_disturbance, label_alert_intermediate
from dist_s1.packaging import generate_default_dist_arrs_from_metric_status
from dist_s1.confirmation import confirm_disturbance_arr
from dist_s1.constants import DISTLABEL2VAL


def generate_status_test_series(
    provisional_duration: int = 2,
    confirmed_duration: int = 10,
    fin_quiet_duration: int = 10,
    high_anomaly_value: float = 6.0,
    low_anomaly_value: float = 4.0,
) -> tuple[list[np.ndarray], dict[str, tuple[int, int]]]:
    # Configuration: Durations required to trigger states

    num_arrays = confirmed_duration + fin_quiet_duration + 1
    final_t = num_arrays - 1

    arr_size = 10
    metric_ts = []
    arr_base = np.zeros((arr_size, arr_size), dtype=np.float32)

    # Define a unique coordinate for each status's representative pixel.
    label2coords = {
        'no_disturbance': (0, 0),  # No disturbance
        'nodata': (0, 1),  # No data (will be set to NaN)
        'first_high_conf_disturbance': (1, 0),  # First high anomaly
        'first_low_conf_disturbance': (1, 1),  # First low anomaly
        'provisional_high_conf_disturbance': (2, 0),  # Provisional high
        'provisional_low_conf_disturbance': (2, 1),  # Provisional low
        'confirmed_high_conf_disturbance': (3, 0),  # Confirmed high
        'confirmed_low_conf_disturbance': (3, 1),  # Confirmed low
        'finalized_confirmed_high_conf_disturbance': (4, 0),  # Finalized confirmed high
        'finalized_confirmed_low_conf_disturbance': (4, 1),  # Finalized confirmed low
    }
    assert set(label2coords.keys()) == set(DISTLABEL2VAL.keys())

    for t in range(num_arrays):
        metric_at_t = arr_base.copy()

        if final_t - provisional_duration < t <= final_t:
            metric_at_t[label2coords['provisional_high_conf_disturbance']] = high_anomaly_value

        if final_t - provisional_duration < t <= final_t:
            metric_at_t[label2coords['provisional_low_conf_disturbance']] = low_anomaly_value

        if final_t - confirmed_duration < t <= final_t:
            metric_at_t[label2coords['confirmed_high_conf_disturbance']] = high_anomaly_value

        if final_t - confirmed_duration < t <= final_t:
            metric_at_t[label2coords['confirmed_low_conf_disturbance']] = low_anomaly_value

        if 0 <= t < confirmed_duration:
            metric_at_t[label2coords['finalized_confirmed_high_conf_disturbance']] = high_anomaly_value

        if 0 <= t < confirmed_duration:
            metric_at_t[label2coords['finalized_confirmed_low_conf_disturbance']] = low_anomaly_value

        if t == final_t:
            metric_at_t[label2coords['first_high_conf_disturbance']] = high_anomaly_value
            metric_at_t[label2coords['first_low_conf_disturbance']] = low_anomaly_value
            metric_at_t[label2coords['nodata']] = np.nan

        metric_ts.append(metric_at_t)

    return metric_ts, label2coords


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


def test_disturbance_status_series(
    tmp_path: Path, moderate_confidence_threshold: float = 3.5, high_confidence_threshold: float = 5.5
) -> None:
    metric_ts, label2coords = generate_status_test_series()
    t0_ref_date = datetime.datetime(2023, 7, 1)

    zipped_metric_ts = list(enumerate(metric_ts))
    t, metric_ts_at_t = zipped_metric_ts[0]
    X_dist_status_ = label_one_disturbance(metric_ts_at_t, moderate_confidence_threshold, high_confidence_threshold)
    X_dist_status = label_alert_intermediate(
        X_dist_status_,
        moderate_confidence_label=moderate_confidence_threshold,
        high_confidence_label=high_confidence_threshold,
    )

    for t, metric_at_t in zipped_metric_ts[1:]:
        current_date = t0_ref_date + datetime.timedelta(days=t)

        out_dist_arr_dict = confirm_disturbance_arr(
            current_metric=metric_at_t,
            current_date=current_date,
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
            r, c = label2coords[label]
            sub = final_status[r, c]
            assert (sub == expected_val).any(), f'{label} block missing expected code {expected_val}'
