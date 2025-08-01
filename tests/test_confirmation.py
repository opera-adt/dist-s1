import numpy as np
import pandas as pd

from dist_s1.confirmation import confirm_disturbance_arr
from dist_s1.constants import BASE_DATE_FOR_CONFIRMATION, DISTLABEL2VAL
from dist_s1.dist_processing import label_alert_status_from_metric
from dist_s1.packaging import generate_default_dist_arrs_from_metric_and_alert_status


def generate_metric_ts_and_expected_disturbances(
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
        'first_low_conf_disturbance': (1, 0),  # First low anomaly
        'provisional_low_conf_disturbance': (1, 1),  # Provisional low
        'confirmed_low_conf_disturbance': (2, 0),  # Confirmed low
        'first_high_conf_disturbance': (2, 1),  # First high anomaly
        'provisional_high_conf_disturbance': (3, 0),  # Provisional high
        'confirmed_high_conf_disturbance': (3, 1),  # Confirmed high
        'confirmed_low_conf_disturbance_finished': (4, 0),  # Confirmed low finished
        'confirmed_high_conf_disturbance_finished': (4, 1),  # Confirmed high finished
    }
    assert set(label2coords.keys()) == set(DISTLABEL2VAL.keys())

    for t in range(num_arrays):
        metric_at_t = arr_base.copy()
        metric_at_t[label2coords['nodata']] = np.nan

        if final_t - provisional_duration < t <= final_t:
            metric_at_t[label2coords['provisional_high_conf_disturbance']] = high_anomaly_value

        if final_t - provisional_duration < t <= final_t:
            metric_at_t[label2coords['provisional_low_conf_disturbance']] = low_anomaly_value

        if final_t - confirmed_duration < t <= final_t:
            metric_at_t[label2coords['confirmed_high_conf_disturbance']] = high_anomaly_value

        if final_t - confirmed_duration < t <= final_t:
            metric_at_t[label2coords['confirmed_low_conf_disturbance']] = low_anomaly_value

        if 0 <= t < confirmed_duration:
            metric_at_t[label2coords['confirmed_high_conf_disturbance_finished']] = high_anomaly_value

        if 0 <= t < confirmed_duration:
            metric_at_t[label2coords['confirmed_low_conf_disturbance_finished']] = low_anomaly_value

        if t == final_t:
            metric_at_t[label2coords['first_high_conf_disturbance']] = high_anomaly_value
            metric_at_t[label2coords['first_low_conf_disturbance']] = low_anomaly_value

        metric_ts.append(metric_at_t)

    return metric_ts, label2coords


def test_disturbance_status_series(
    moderate_confidence_threshold: float = 3.5, high_confidence_threshold: float = 5.5
) -> None:
    metric_ts, label2coords = generate_metric_ts_and_expected_disturbances()
    t0_ref_date = pd.Timestamp('2023-07-01', tz='UTC')

    zipped_metric_ts = list(enumerate(metric_ts))
    t, X_metric_t = zipped_metric_ts[0]
    X_dist_status = label_alert_status_from_metric(
        X_metric_t,
        low_confidence_alert_threshold=moderate_confidence_threshold,
        high_confidence_alert_threshold=high_confidence_threshold,
    )
    prior_dist_arr_dict = generate_default_dist_arrs_from_metric_and_alert_status(
        X_metric_t, X_dist_status, t0_ref_date
    )
    status_series = []
    status_series.append(X_dist_status)

    for t, X_metric_t in zipped_metric_ts[1:]:
        current_date = t0_ref_date + pd.Timedelta(days=t)
        current_date_days_from_base_date = (current_date - BASE_DATE_FOR_CONFIRMATION).days

        current_dist_arr_dict = confirm_disturbance_arr(
            current_metric=X_metric_t,
            current_date_days_from_base_date=current_date_days_from_base_date,
            prior_alert_status=prior_dist_arr_dict['GEN-DIST-STATUS'],
            prior_max_metric=prior_dist_arr_dict['GEN-METRIC-MAX'],
            prior_confidence=prior_dist_arr_dict['GEN-DIST-CONF'],
            prior_date=prior_dist_arr_dict['GEN-DIST-DATE'],
            prior_count=prior_dist_arr_dict['GEN-DIST-COUNT'],
            prior_percent=prior_dist_arr_dict['GEN-DIST-PERC'],
            prior_duration=prior_dist_arr_dict['GEN-DIST-DUR'],
            prior_last_obs=prior_dist_arr_dict['GEN-DIST-LAST-DATE'],
            alert_low_conf_thresh=moderate_confidence_threshold,
            alert_high_conf_thresh=high_confidence_threshold,
            exclude_consecutive_no_dist=False,
            percent_reset_thresh=10,
            no_count_reset_thresh=7,
            no_day_limit=30,
            max_obs_num_year=253,
            conf_upper_lim=32000,
            conf_thresh=3**2 * 3.5,
            metric_value_upper_lim=100,
        )
        current_status = current_dist_arr_dict['GEN-DIST-STATUS']
        status_series.append(current_status)
        prior_dist_arr_dict = current_dist_arr_dict

    for label, status_expected in DISTLABEL2VAL.items():
        r, c = label2coords[label]
        status_actual = current_status[r, c]

        # Debugging
        status_ts_pixel = [s[r, c] for s in status_series]
        metric_ts_pixel = [m[r, c] for m in metric_ts]
        assert status_ts_pixel
        assert metric_ts_pixel

        assert (status_actual == status_expected).any(), f'{label} block missing expected code {status_expected}'
