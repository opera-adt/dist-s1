from pathlib import Path

import numpy as np
import rasterio

from dist_s1.constants import (
    BASE_DATE_FOR_CONFIRMATION,
    DISTLABEL2VAL,
    DIST_STATUS_CMAP,
    TIF_LAYERS,
    TIF_LAYER_NODATA_VALUES,
)
from dist_s1.data_models.output_models import TIF_LAYER_DTYPES, DistS1ProductDirectory
from dist_s1.dist_processing import label_alert_status_from_metric
from dist_s1.packaging import update_profile
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds


def confirm_disturbance_arr(
    *,
    current_metric: np.ndarray,
    current_date_days_from_base_date: int,
    prior_alert_status: np.ndarray,
    prior_max_metric: np.ndarray,
    prior_confidence: np.ndarray,
    prior_date: np.ndarray,
    prior_count: np.ndarray,
    prior_percent: np.ndarray,
    prior_duration: np.ndarray,
    prior_last_obs: np.ndarray,
    alert_low_conf_thresh: float,
    alert_high_conf_thresh: float,
    exclude_consecutive_no_dist: bool,
    percent_reset_thresh: int,
    no_count_reset_thresh: int,
    no_day_limit: int,
    max_obs_num_year: int,
    conf_upper_lim: int,
    conf_thresh: float,
    metric_value_upper_lim: float,
) -> dict[np.ndarray]:
    # Status codes
    no_disturbance_label = DISTLABEL2VAL['no_disturbance']
    first_dist_low_label = DISTLABEL2VAL['first_low_conf_disturbance']
    prov_dist_low_label = DISTLABEL2VAL['provisional_low_conf_disturbance']
    conf_dist_low_label = DISTLABEL2VAL['confirmed_low_conf_disturbance']
    first_dist_high_label = DISTLABEL2VAL['first_high_conf_disturbance']
    prov_dist_high_label = DISTLABEL2VAL['provisional_high_conf_disturbance']
    conf_dist_high_label = DISTLABEL2VAL['confirmed_high_conf_disturbance']
    conf_dist_low_fin_label = DISTLABEL2VAL['confirmed_low_conf_disturbance_finished']
    conf_dist_high_fin_label = DISTLABEL2VAL['confirmed_high_conf_disturbance_finished']
    nodata_label = DISTLABEL2VAL['nodata']

    # Masks
    valid_data_mask = ~np.isnan(current_metric)
    # Reset if 365-day timeout, or previous status is finished and current anomaly is above low threshold
    reset_mask = ((current_date_days_from_base_date - prior_date) > 365) | (
        (prior_alert_status > 6) & (current_metric >= alert_low_conf_thresh)
    )
    reset_mask &= valid_data_mask

    # Initializization
    current_percent = prior_percent.copy()
    current_count = prior_count.copy()
    current_duration = prior_duration.copy()
    current_last_obs = prior_last_obs.copy()
    current_max_metric = prior_max_metric.copy()
    current_confidence = prior_confidence.copy()
    current_date = prior_date.copy()
    current_last_obs = prior_last_obs.copy()
    current_alert_status = prior_alert_status.copy()

    current_alert_status[reset_mask] = no_disturbance_label
    current_percent[reset_mask] = 255
    current_count[reset_mask] = 0
    current_max_metric[reset_mask] = 0
    current_confidence[reset_mask] = 0
    current_date[reset_mask] = 0
    current_last_obs[reset_mask] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        prior_no_count = np.where(
            (current_percent > 0) & (current_percent <= 100),
            ((100.0 - current_percent) / current_percent * current_count).astype(np.int32),
            0,
        )

    current_disturbed_mask = (current_metric >= alert_low_conf_thresh) & valid_data_mask

    # New disturbance detection logic
    new_detection = current_disturbed_mask & (
        (current_alert_status == no_disturbance_label) | (current_alert_status == nodata_label)
    )
    current_date[new_detection] = current_date_days_from_base_date
    current_max_metric[new_detection] = current_metric[new_detection]
    current_percent[new_detection] = 100
    current_count[new_detection] = 1
    current_duration[new_detection] = 1

    # Ongoing disturbance detection logic
    continuing_dist_mask = current_disturbed_mask & ~new_detection
    current_max_metric[continuing_dist_mask] = np.maximum(
        current_max_metric[continuing_dist_mask], current_metric[continuing_dist_mask]
    )
    can_increment = continuing_dist_mask & (current_count < max_obs_num_year)
    current_count[can_increment] += 1
    current_percent[can_increment] = (
        (current_count[can_increment] * 100.0) / (current_count[can_increment] + prior_no_count[can_increment])
    ).astype(np.uint8)

    current_duration[continuing_dist_mask] = current_date_days_from_base_date - current_date[continuing_dist_mask] + 1

    # Track valid obs but not anomalous or not above low threshold (adjust percent)
    current_not_disturbed_mask = (~current_disturbed_mask) & valid_data_mask
    adjust_percent = (
        current_not_disturbed_mask
        & (current_percent > 0)
        & (current_percent <= 100)
        & (current_count < max_obs_num_year + 1)
    )
    current_percent[adjust_percent] = (
        (current_count[adjust_percent] * 100.0) / (current_count[adjust_percent] + prior_no_count[adjust_percent] + 1)
    ).astype(np.uint8)

    # Reset status for pixels that were NODATA and are now not disturbed
    status_reset_mask = current_not_disturbed_mask & (current_alert_status == nodata_label)
    current_alert_status[status_reset_mask] = no_disturbance_label
    current_percent[status_reset_mask] = 255
    current_count[status_reset_mask] = 0
    current_max_metric[status_reset_mask] = 0
    current_confidence[status_reset_mask] = 0
    current_date[status_reset_mask] = 0
    current_last_obs[status_reset_mask] = 0

    # Update confidence
    update_conf = (current_confidence > 0) & (current_alert_status <= conf_dist_high_label) & valid_data_mask
    current_metric_conf = np.minimum(current_metric, metric_value_upper_lim)
    prior_mean = np.zeros_like(current_confidence, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        prior_mean[update_conf] = current_confidence[update_conf].astype(np.float64) / (
            current_count[update_conf].astype(np.float64) ** 2
        )
    current_mean = np.zeros_like(prior_mean)
    denom = current_count + prior_no_count + 1
    with np.errstate(divide='ignore', invalid='ignore'):
        current_mean[update_conf] = (
            prior_mean[update_conf] * (current_count[update_conf] + prior_no_count[update_conf])
            + current_metric_conf[update_conf]
        ) / denom[update_conf]
    tempconf = (current_mean * current_count.astype(np.float64) * current_count.astype(np.float64)).astype(np.int32)
    current_confidence[update_conf] = np.clip(tempconf[update_conf], 0, conf_upper_lim)

    # Update confidence for new disturbances
    conf_update_mask = (
        (current_alert_status == no_disturbance_label) | (current_alert_status == nodata_label)
    ) & current_disturbed_mask
    current_confidence[conf_update_mask] = np.minimum(current_metric[conf_update_mask], metric_value_upper_lim)

    latest_dist_date = current_date + current_duration - 1

    # consecutive_no_dist_count can be 0, 1, or 2 value
    consecutive_no_dist_count = (current_last_obs > latest_dist_date).astype(np.uint8) + (
        current_metric < alert_low_conf_thresh
    ).astype(np.uint8)

    # Mask for active disturbances
    updating_current = (current_alert_status <= conf_dist_high_label) | (current_alert_status == nodata_label)

    # High threshold disturbances
    high_dist_mask = updating_current & (current_max_metric >= alert_high_conf_thresh)
    confirmed_hi_mask = high_dist_mask & (current_confidence >= conf_thresh)
    first_hi_mask = high_dist_mask & (current_duration == 1)
    prov_hi_mask = (
        high_dist_mask & ~(confirmed_hi_mask | first_hi_mask) & (current_alert_status != conf_dist_high_label)
    )
    current_alert_status[confirmed_hi_mask] = conf_dist_high_label
    current_alert_status[first_hi_mask] = first_dist_high_label
    current_alert_status[prov_hi_mask] = prov_dist_high_label

    # Low threshold disturbances
    low_dist_mask = (
        updating_current & (current_max_metric >= alert_low_conf_thresh) & (current_max_metric < alert_high_conf_thresh)
    )
    confirmed_lo_mask = low_dist_mask & (current_confidence >= conf_thresh)
    first_lo_mask = low_dist_mask & (current_duration == 1)
    prov_lo_mask = low_dist_mask & ~(confirmed_lo_mask | first_lo_mask) & (current_alert_status != conf_dist_low_label)
    current_alert_status[confirmed_lo_mask] = conf_dist_low_label
    current_alert_status[first_lo_mask] = first_dist_low_label
    current_alert_status[prov_lo_mask] = prov_dist_low_label

    # Reset ongoing disturbances if max_anom drops below lowthresh
    current_alert_status[updating_current & (current_max_metric < alert_low_conf_thresh)] = no_disturbance_label

    # Initialize must_finish_conditions list
    must_finish_conditions = []

    # Condition A: Observation gap too large
    must_finish_conditions.append(
        ((current_date_days_from_base_date - latest_dist_date) >= no_day_limit) & (latest_dist_date > 0)
    )
    # Condition B: Short disturbance with low current metric
    must_finish_conditions.append((current_duration == 1) & (current_metric < alert_low_conf_thresh))
    # Condition C: Conditional nocount logic (used if `consecutive_nodist` is set to True)
    if not exclude_consecutive_no_dist:
        must_finish_conditions.append(consecutive_no_dist_count == 2)
    # Condition D: Percent below threshold
    # If the percent of disturbed observations drops below threshold, it triggers reset.
    must_finish_conditions.append(current_percent < percent_reset_thresh)
    # Condition E: Number of non-disturbed observations (prevnocount) below threshold
    # If the number of non-disturbed observations in a series (prevnocount) is high,
    # the disturbance resets.
    must_finish_conditions.append(prior_no_count >= no_count_reset_thresh)

    combined_must_finish_criteria = np.logical_or.reduce(must_finish_conditions)

    must_finish = (current_alert_status <= conf_dist_high_label) & combined_must_finish_criteria

    # Apply finished status
    current_alert_status_before_fin = current_alert_status.copy()
    current_alert_status[must_finish & (current_alert_status_before_fin == conf_dist_low_label)] = (
        conf_dist_low_fin_label
    )
    current_alert_status[must_finish & (current_alert_status_before_fin == conf_dist_high_label)] = (
        conf_dist_high_fin_label
    )

    # Reset other finished pixels to NODIST
    finished_reset_mask = must_finish & ~(
        (current_alert_status_before_fin == conf_dist_low_label)
        | (current_alert_status_before_fin == conf_dist_high_label)
    )
    current_alert_status[finished_reset_mask] = no_disturbance_label
    current_percent[finished_reset_mask] = 0
    current_count[finished_reset_mask] = 0
    current_max_metric[finished_reset_mask] = 0
    current_confidence[finished_reset_mask] = 0
    current_date[finished_reset_mask] = 0
    current_duration[finished_reset_mask] = 0

    # Update last observation date for all valid pixels
    current_last_obs[valid_data_mask] = current_date_days_from_base_date

    alert_acq_status = label_alert_status_from_metric(
        current_metric,
        low_confidence_alert_threshold=alert_low_conf_thresh,
        high_confidence_alert_threshold=alert_high_conf_thresh,
    )

    return {
        'GEN-METRIC': current_metric.astype(TIF_LAYER_DTYPES['GEN-METRIC']),
        'GEN-METRIC-MAX': current_max_metric.astype(TIF_LAYER_DTYPES['GEN-METRIC-MAX']),
        'GEN-DIST-STATUS': current_alert_status.astype(TIF_LAYER_DTYPES['GEN-DIST-STATUS']),
        'GEN-DIST-STATUS-ACQ': alert_acq_status.astype(TIF_LAYER_DTYPES['GEN-DIST-STATUS-ACQ']),
        'GEN-DIST-CONF': current_confidence.astype(TIF_LAYER_DTYPES['GEN-DIST-CONF']),
        'GEN-DIST-DATE': current_date.astype(TIF_LAYER_DTYPES['GEN-DIST-DATE']),
        'GEN-DIST-COUNT': current_count.astype(TIF_LAYER_DTYPES['GEN-DIST-COUNT']),
        'GEN-DIST-PERC': current_percent.astype(TIF_LAYER_DTYPES['GEN-DIST-PERC']),
        'GEN-DIST-DUR': current_duration.astype(TIF_LAYER_DTYPES['GEN-DIST-DUR']),
        'GEN-DIST-LAST-DATE': current_last_obs.astype(TIF_LAYER_DTYPES['GEN-DIST-LAST-DATE']),
    }


def confirm_disturbance_with_prior_product_and_serialize(
    current_dist_s1_product: DistS1ProductDirectory | str | Path,
    prior_dist_s1_product: DistS1ProductDirectory | str | Path,
    dst_dist_product_parent: str | Path | None,
    alert_low_conf_thresh: float = 3.5,
    alert_high_conf_thresh: float = 5.5,
    exclude_consecutive_no_dist: bool = False,
    percent_reset_thresh: int = 10,
    no_count_reset_thresh: int = 7,
    no_day_limit: int = 30,
    max_obs_num_year: int = 253,
    confidence_upper_lim: int = 32000,
    confidence_thresh: float = 3**2 * 3.5,
    metric_value_upper_lim: float = 100,
    product_tags: dict | None = None,
) -> DistS1ProductDirectory:
    """Perform the confirmation and packaging of a DIST-S1 product and the prior product."""
    if not isinstance(current_dist_s1_product, DistS1ProductDirectory):
        current_dist_s1_product = DistS1ProductDirectory.from_product_path(current_dist_s1_product)

    if not isinstance(prior_dist_s1_product, DistS1ProductDirectory):
        prior_dist_s1_product = DistS1ProductDirectory.from_product_path(prior_dist_s1_product)
    prior_product_name = prior_dist_s1_product.product_name

    if dst_dist_product_parent is None:
        dst_dist_product_parent = current_dist_s1_product.product_dir_path.parent
    else:
        dst_dist_product_directory = Path(dst_dist_product_parent) / current_dist_s1_product.product_name
        dst_dist_product_directory.mkdir(parents=True, exist_ok=True)

    dst_dist_product_directory = DistS1ProductDirectory(
        dst_dir=dst_dist_product_parent,
        product_name=current_dist_s1_product.product_name,
    )
    if product_tags is None:
        with rasterio.open(current_dist_s1_product.layer_path_dict['GEN-METRIC']) as src:
            product_tags = src.tags()
        if product_tags is None:
            raise ValueError('No product tags found in current product; not using correctly formatted product.')

    # Get dist_date from a sample path pattern
    current_date_ts = current_dist_s1_product.acq_datetime
    current_date_days_from_base_date = (current_date_ts - BASE_DATE_FOR_CONFIRMATION).days

    # Load product arrays
    prior_arr_dict = {
        layer_name: open_one_ds(prior_dist_s1_product.layer_path_dict[layer_name])[0] for layer_name in TIF_LAYERS
    }
    current_metric, anom_prof = open_one_ds(current_dist_s1_product.layer_path_dict['GEN-METRIC'])

    # Core Confirmation Logic
    confirmed_arr_dict = confirm_disturbance_arr(
        current_metric=current_metric,
        prior_alert_status=prior_arr_dict['GEN-DIST-STATUS'],
        prior_max_metric=prior_arr_dict['GEN-METRIC-MAX'],
        prior_confidence=prior_arr_dict['GEN-DIST-CONF'],
        prior_date=prior_arr_dict['GEN-DIST-DATE'],
        prior_count=prior_arr_dict['GEN-DIST-COUNT'],
        prior_percent=prior_arr_dict['GEN-DIST-PERC'],
        prior_duration=prior_arr_dict['GEN-DIST-DUR'],
        prior_last_obs=prior_arr_dict['GEN-DIST-LAST-DATE'],
        current_date_days_from_base_date=current_date_days_from_base_date,
        alert_low_conf_thresh=alert_low_conf_thresh,
        alert_high_conf_thresh=alert_high_conf_thresh,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        no_day_limit=no_day_limit,
        max_obs_num_year=max_obs_num_year,
        conf_upper_lim=confidence_upper_lim,
        conf_thresh=confidence_thresh,
        metric_value_upper_lim=metric_value_upper_lim,
    )

    # Path Accounting
    out_paths_dict = {layer_name: dst_dist_product_directory.layer_path_dict[layer_name] for layer_name in TIF_LAYERS}

    # Update profiles
    out_profiles_dict = {
        layer_name: update_profile(anom_prof, TIF_LAYER_DTYPES[layer_name], TIF_LAYER_NODATA_VALUES[layer_name])
        for layer_name in TIF_LAYERS
    }

    # Serialize output
    out_product_tags = product_tags.copy()
    out_product_tags['prior_product_name'] = prior_product_name
    for layer_name in TIF_LAYERS:
        if layer_name in ['GEN-DIST-STATUS', 'GEN-DIST-STATUS-ACQ']:
            cmap = DIST_STATUS_CMAP
        else:
            cmap = None
        serialize_one_2d_ds(
            confirmed_arr_dict[layer_name],
            out_profiles_dict[layer_name],
            out_paths_dict[layer_name],
            colormap=cmap,
            cog=True,
            tags=out_product_tags,
        )
    return dst_dist_product_directory
