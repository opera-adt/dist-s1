import datetime
from pathlib import Path

import numpy as np

from dist_s1.constants import BASE_DATE_FOR_CONFIRMATION, DISTLABEL2VAL
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds


def compute_tile_disturbance_arr(
    *,
    prior_: np.ndarray,
    prior_max_metric: np.ndarray,
    prior_confidence: np.ndarray,
    prior_date: np.ndarray,
    count: np.ndarray,
    percent: np.ndarray,
    duration: np.ndarray,
    last_obs: np.ndarray,
    current_metric: np.ndarray,
    current_date_days_from_base_date: int,
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
    no_disturbance = DISTLABEL2VAL['no_disturbance']
    first_dist_low = DISTLABEL2VAL['first_low_conf_disturbance']
    prov_dist_low = DISTLABEL2VAL['provisional_low_conf_disturbance']
    conf_dist_low = DISTLABEL2VAL['confirmed_low_conf_disturbance']
    first_dist_high = DISTLABEL2VAL['first_high_conf_disturbance']
    prov_dist_high = DISTLABEL2VAL['provisional_high_conf_disturbance']
    conf_dist_high = DISTLABEL2VAL['confirmed_high_conf_disturbance']
    conf_dist_low_fin = DISTLABEL2VAL['confirmed_low_conf_finished']
    conf_dist_high_fin = DISTLABEL2VAL['confirmed_high_conf_finished']
    nodata = DISTLABEL2VAL['nodata']

    valid = current_metric >= 0

    # Reset if 365-day timeout, or previous status is finished and current anomaly is above low threshold
    reset_mask = ((current_date_days_from_base_date - prior_date) > 365) | (
        (prior_ > 6) & (current_metric >= alert_low_conf_thresh)
    )
    reset_mask &= valid
    prior_[reset_mask] = no_disturbance
    percent[reset_mask] = 255
    count[reset_mask] = 0
    prior_max_metric[reset_mask] = 0
    prior_confidence[reset_mask] = 0
    prior_date[reset_mask] = 0
    duration[reset_mask] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        prevnocount = np.where(
            (percent > 0) & (percent <= 100), ((100.0 - percent) / percent * count).astype(np.int32), 0
        )

    disturbed = (current_metric >= alert_low_conf_thresh) & valid

    # New disturbance detection logic
    new_detection = disturbed & ((prior_ == no_disturbance) | (prior_ == nodata))
    prior_date[new_detection] = current_date_days_from_base_date
    prior_max_metric[new_detection] = current_metric[new_detection]
    percent[new_detection] = 100
    count[new_detection] = 1

    # Ongoing disturbance detection logic
    continuing = disturbed & ~new_detection
    prior_max_metric[continuing] = np.maximum(prior_max_metric[continuing], current_metric[continuing])
    can_increment = continuing & (count < max_obs_num_year)
    count[can_increment] += 1
    percent[can_increment] = (
        (count[can_increment] * 100.0) / (count[can_increment] + prevnocount[can_increment])
    ).astype(np.uint8)

    duration[disturbed] = current_date_days_from_base_date - prior_date[disturbed] + 1

    # Track valid obs but not anomalous or not above low threshold (adjust percent)
    not_disturbed = (~disturbed) & valid
    adjust_percent = not_disturbed & (percent > 0) & (percent <= 100) & (count < max_obs_num_year + 1)
    prevcount = count.copy()
    percent[adjust_percent] = (
        (count[adjust_percent] * 100.0) / (prevcount[adjust_percent] + prevnocount[adjust_percent] + 1)
    ).astype(np.uint8)

    # Reset status for pixels that were NODATA and are now not disturbed
    status_reset = not_disturbed & (prior_ == nodata)
    prior_[status_reset] = no_disturbance
    percent[status_reset] = 255
    count[status_reset] = 0
    prior_max_metric[status_reset] = 0
    prior_confidence[status_reset] = 0
    prior_date[status_reset] = 0
    duration[status_reset] = 0

    # Update confidence
    update_conf = (prior_confidence > 0) & (prior_ <= conf_dist_high) & valid
    curr_metric_conf = np.minimum(current_metric, metric_value_upper_lim)
    prevmean = np.zeros_like(prior_confidence, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        prevmean[update_conf] = prior_confidence[update_conf].astype(np.float64) / (
            count[update_conf].astype(np.float64) ** 2
        )
    mean = np.zeros_like(prevmean)
    denom = count + prevnocount + 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mean[update_conf] = (
            prevmean[update_conf] * (count[update_conf] + prevnocount[update_conf]) + curr_metric_conf[update_conf]
        ) / denom[update_conf]
    tempconf = (mean * count.astype(np.float64) * count.astype(np.float64)).astype(np.int32)
    prior_confidence[update_conf] = np.clip(tempconf[update_conf], 0, conf_upper_lim)

    # Update confidence for new disturbances
    new_conf = ((prior_ == no_disturbance) | (prior_ == nodata)) & disturbed
    prior_confidence[new_conf] = np.minimum(current_metric[new_conf], metric_value_upper_lim)

    last_metric_date = prior_date + duration - 1

    nocount = (last_obs > last_metric_date).astype(np.int8) + (current_metric < alert_low_conf_thresh).astype(np.int8)

    # Updates for active disturbances
    # High threshold disturbances
    updating_current = (prior_ <= conf_dist_high) | (prior_ == nodata)
    hi_mask = updating_current & (prior_max_metric >= alert_high_conf_thresh)
    conf_hi = hi_mask & (prior_confidence >= conf_thresh)
    first_hi = hi_mask & (duration == 1)
    prov_hi = hi_mask & ~(conf_hi | first_hi) & (prior_ != conf_dist_high)
    prior_[conf_hi] = conf_dist_high
    prior_[first_hi] = first_dist_high
    prior_[prov_hi] = prov_dist_high
    # Low threshold disturbances
    lo_mask = (
        updating_current & (prior_max_metric >= alert_low_conf_thresh) & (prior_max_metric < alert_high_conf_thresh)
    )
    conf_lo = lo_mask & (prior_confidence >= conf_thresh)
    first_lo = lo_mask & (duration == 1)
    prov_lo = lo_mask & ~(conf_lo | first_lo) & (prior_ != conf_dist_low)
    prior_[conf_lo] = conf_dist_low
    prior_[first_lo] = first_dist_low
    prior_[prov_lo] = prov_dist_low

    # Reset ongoing disturbances if max_anom drops below lowthresh
    prior_[updating_current & (prior_max_metric < alert_low_conf_thresh)] = no_disturbance

    # Update must finish disturbances
    status_at_finish_check = prior_.copy()  # Capture status before final finish logic

    # Initialize must_finish_conditions list
    must_finish_conditions = []

    # Condition A: Observation gap too large
    must_finish_conditions.append(
        ((current_date_days_from_base_date - last_metric_date) >= no_day_limit) & (last_metric_date > 0)
    )
    # Condition B: Short disturbance with low current metric
    must_finish_conditions.append((duration == 1) & (current_metric < alert_low_conf_thresh))
    # Condition C: Conditional nocount logic (used if `consecutive_nodist` is set to True)
    if not exclude_consecutive_no_dist:
        must_finish_conditions.append(nocount == 2)
    # Condition D: Percent below threshold
    # If the percent of disturbed observations drops below threshold, it triggers reset.
    must_finish_conditions.append(percent < percent_reset_thresh)
    # Condition E: Number of non-disturbed observations (prevnocount) below threshold
    # If the number of non-disturbed observations in a series (prevnocount) is high,
    # the disturbance resets.
    must_finish_conditions.append(prevnocount >= no_count_reset_thresh)

    # Combine all must_finish_conditions with OR
    if must_finish_conditions:
        combined_must_finish_criteria = np.logical_or.reduce(must_finish_conditions)
    else:
        combined_must_finish_criteria = np.full(prior_.shape, False, dtype=bool)

    must_finish = (status_at_finish_check <= conf_dist_high) & combined_must_finish_criteria

    # Apply finished status
    prior_[must_finish & (status_at_finish_check == conf_dist_low)] = conf_dist_low_fin
    prior_[must_finish & (status_at_finish_check == conf_dist_high)] = conf_dist_high_fin

    # Reset other finished pixels to NODIST
    reset_finish = must_finish & ~(
        (status_at_finish_check == conf_dist_low) | (status_at_finish_check == conf_dist_high)
    )
    prior_[reset_finish] = no_disturbance
    percent[reset_finish] = 0
    count[reset_finish] = 0
    prior_max_metric[reset_finish] = 0
    prior_confidence[reset_finish] = 0
    prior_date[reset_finish] = 0
    duration[reset_finish] = 0

    # Update last observation date for all valid pixels
    last_obs[valid] = current_date_days_from_base_date

    return {
        'status': prior_,
        'max_metric': prior_max_metric,
        'confidence': prior_confidence,
        'date': prior_date,
        'count': count,
        'percent': percent,
        'duration': duration,
        'last_obs': last_obs,
    }


def compute_tile_disturbance_using_previous_product_and_serialize(
    current_dist_s1_product: DistS1ProductDirectory | str | Path,
    prior_dist_s1_product: DistS1ProductDirectory | str | Path,
    dst_dist_product: str | Path | DistS1ProductDirectory = None,
    alert_low_conf_thresh: float = 3.5,
    alert_high_conf_thresh: float = 5.5,
    exclude_consecutive_no_dist: bool = False,
    percent_reset_thresh: int = 10,
    no_count_reset_thresh: int = 7,
    no_day_limit: int = 30,
    max_obs_num_year: int = 253,
    conf_upper_lim: int = 32000,
    conf_thresh: float = 3**2 * 3.5,
    metric_value_upper_lim: float = 100,
    base_date_for_confirmation: datetime.datetime | None = None,
) -> None:
    if not isinstance(current_dist_s1_product, DistS1ProductDirectory):
        current_dist_s1_product = DistS1ProductDirectory.from_product_path(current_dist_s1_product)

    if not isinstance(prior_dist_s1_product, DistS1ProductDirectory):
        prior_dist_s1_product = DistS1ProductDirectory.from_product_path(prior_dist_s1_product)

    if dst_dist_product is None:
        dst_dist_product = current_dist_s1_product
    else:
        if not isinstance(dst_dist_product, DistS1ProductDirectory):
            dst_dist_product = DistS1ProductDirectory.from_product_path(dst_dist_product)

    if base_date_for_confirmation is None:
        base_date_for_confirmation = BASE_DATE_FOR_CONFIRMATION

    # Get dist_date from a sample path pattern
    dist_date = current_dist_s1_product.acq_datetime
    current_date_days_from_base_date = (dist_date - base_date_for_confirmation).days

    # Load product arrays
    current_metric, anom_prof = open_one_ds(current_dist_s1_product.layer_path_dict['GEN-METRIC'])
    prior_status, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-STATUS'])
    prior_max_metric, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-METRIC-MAX'])
    prior_confidence, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-CONF'])
    prior_date, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-DATE'])
    prior_count, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-COUNT'])
    prior_percent, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-PERC'])
    prior_duration, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-DUR'])
    prior_last_obs, _ = open_one_ds(prior_dist_s1_product.layer_path_dict['GEN-DIST-LAST-DATE'])

    # Core Confirmation Logic
    dist_arr_dict = compute_tile_disturbance_arr(
        current_metric=current_metric,
        prior_=prior_status,
        prior_max_metric=prior_max_metric,
        prior_confidence=prior_confidence,
        prior_date=prior_date,
        prior_count=prior_count,
        prior_percent=prior_percent,
        prior_duration=prior_duration,
        prior_last_obs=prior_last_obs,
        current_date_days_from_base_date=current_date_days_from_base_date,
        alert_low_conf_thresh=alert_low_conf_thresh,
        alert_high_conf_thresh=alert_high_conf_thresh,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        no_day_limit=no_day_limit,
        max_obs_num_year=max_obs_num_year,
        conf_upper_lim=conf_upper_lim,
        conf_thresh=conf_thresh,
        metric_value_upper_lim=metric_value_upper_lim,
    )
    # Prepare profiles for serialization
    p_dist_int8 = anom_prof.copy()
    p_dist_int8['nodata'] = 255
    p_dist_int8['dtype'] = np.uint8

    p_dist_int16 = anom_prof.copy()
    p_dist_int16['nodata'] = 255
    p_dist_int16['dtype'] = np.int16

    # Serialize output
    out_status_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-STATUS']
    out_max_metric_path = dst_dist_product_dir.layer_path_dict['GEN-METRIC-MAX']
    out_confidence_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-CONF']
    out_date_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-DATE']
    out_count_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-COUNT']
    out_percent_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-PERC']
    out_duration_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-DUR']
    out_last_obs_path = dst_dist_product_dir.layer_path_dict['GEN-DIST-LAST-DATE']

    serialize_one_2d_ds(dist_arr_dict['status'], p_dist_int8, out_status_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['max_metric'], anom_prof, out_max_metric_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['confidence'], p_dist_int16, out_confidence_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['date'], p_dist_int16, out_date_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['count'], p_dist_int8, out_count_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['percent'], p_dist_int8, out_percent_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['duration'], p_dist_int16, out_duration_path, cog=True)
    serialize_one_2d_ds(dist_arr_dict['last_obs'], p_dist_int16, out_last_obs_path, cog=True)
