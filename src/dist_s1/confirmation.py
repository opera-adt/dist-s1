import datetime
from pathlib import Path

import numpy as np

from dist_s1.constants import BASE_DATE_FOR_CONFIRMATION
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds



def compute_tile_disturbance_using_previous_product_and_serialize(
    dist_metric_path: Path,
    dist_metric_date: Path,
    out_path_list: list[str],
    lowthresh: float = 3.5,
    highthresh: float = 5.5,
    consecutive_nodist: bool = False,
    percent_reset_thresh: int = 10,
    nocount_reset_thresh: int = 7, 
    nodaylimit: int = 30,
    max_obs_num_year: int = 253,
    conf_upper_lim: int = 32000,
    conf_thresh: float = 3**2 * 3.5,
    metric_value_upper_lim: float = 100,
    base_date_for_confirmation: datetime.datetime | None = None,
    previous_dist_arr_path_list: list[str | None] | None = None,
) -> None:
    # Status codes
    NODIST, FIRSTLO, PROVLO, CONFLO, FIRSTHI, PROVHI, CONFHI, CONFLOFIN, CONFHIFIN, NODATA = range(10)
    NODATA = 255  # right now this function is not used, check the function.
    if base_date_for_confirmation is None:
        base_date_for_confirmation = BASE_DATE_FOR_CONFIRMATION

    # Get dist_date from a sample path pattern
    dist_date = datetime.datetime.strptime(Path(dist_metric_date).name.split('_')[4], '%Y%m%dT%H%M%SZ')

    # Load current metric and profile
    current_metric, anom_prof = open_one_ds(dist_metric_path)
    rows, cols = current_metric.shape
    current_date = (dist_date - base_date_for_confirmation).days

    # Initialize or load previous arrays
    if previous_dist_arr_path_list is None:
        print('This is a first product tile with date', dist_date)
        status = np.full((rows, cols), 255, dtype=np.uint8)
        max_metric = np.full((rows, cols), -1, dtype=np.int16)
        confidence = np.zeros((rows, cols), dtype=np.int16)
        date = np.zeros((rows, cols), dtype=np.int16)
        count = np.zeros((rows, cols), dtype=np.uint8)
        percent = np.full((rows, cols), 200, dtype=np.uint8)
        duration = np.zeros((rows, cols), dtype=np.int16)
        last_obs = np.zeros((rows, cols), dtype=np.int16)
    elif all(p is not None for p in previous_dist_arr_path_list):
        status, _ = open_one_ds(previous_dist_arr_path_list[0])
        max_metric, _ = open_one_ds(previous_dist_arr_path_list[1])
        confidence, _ = open_one_ds(previous_dist_arr_path_list[2])
        date, _ = open_one_ds(previous_dist_arr_path_list[3])
        count, _ = open_one_ds(previous_dist_arr_path_list[4])
        percent, _ = open_one_ds(previous_dist_arr_path_list[5])
        duration, _ = open_one_ds(previous_dist_arr_path_list[6])
        last_obs, _ = open_one_ds(previous_dist_arr_path_list[7])
    else:
        raise ValueError('Missing previous product data for non-first acquisitions.')

    valid = current_metric >= 0

    # Reset if 365-day timeout, or previous status is finished and current anomaly is above low threshold
    reset_mask = ((current_date - date) > 365) | ((status > 6) & (current_metric >= lowthresh))
    reset_mask &= valid
    status[reset_mask] = NODIST
    percent[reset_mask] = 255
    count[reset_mask] = 0
    max_metric[reset_mask] = 0
    confidence[reset_mask] = 0
    date[reset_mask] = 0
    duration[reset_mask] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        prevnocount = np.where(
            (percent > 0) & (percent <= 100), ((100.0 - percent) / percent * count).astype(np.int32), 0
        )

    disturbed = (current_metric >= lowthresh) & valid

    # New disturbance detection logic
    new_detection = disturbed & ((status == NODIST) | (status == NODATA))
    date[new_detection] = current_date
    max_metric[new_detection] = current_metric[new_detection]
    percent[new_detection] = 100
    count[new_detection] = 1

    # Ongoing disturbance detection logic
    continuing = disturbed & ~new_detection
    max_metric[continuing] = np.maximum(max_metric[continuing], current_metric[continuing])
    can_increment = continuing & (count < max_obs_num_year)
    count[can_increment] += 1
    percent[can_increment] = (
        (count[can_increment] * 100.0) / (count[can_increment] + prevnocount[can_increment])
    ).astype(np.uint8)

    duration[disturbed] = current_date - date[disturbed] + 1

    # Track valid obs but not anomalous or not above low threshold (adjust percent)
    not_disturbed = (~disturbed) & valid
    adjust_percent = not_disturbed & (percent > 0) & (percent <= 100) & (count < max_obs_num_year + 1)
    prevcount = count.copy()
    percent[adjust_percent] = (
        (count[adjust_percent] * 100.0) / (prevcount[adjust_percent] + prevnocount[adjust_percent] + 1)
    ).astype(np.uint8)
    
    # Reset status for pixels that were NODATA and are now not disturbed 
    status_reset = not_disturbed & (status == NODATA)
    status[status_reset] = NODIST
    percent[status_reset] = 255
    count[status_reset] = 0
    max_metric[status_reset] = 0
    confidence[status_reset] = 0
    date[status_reset] = 0
    duration[status_reset] = 0

    # Update confidence
    update_conf = (confidence > 0) & (status <= CONFHI) & valid
    curr_metric_conf = np.minimum(current_metric, metric_value_upper_lim)
    prevmean = np.zeros_like(confidence, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        prevmean[update_conf] = (
            confidence[update_conf].astype(np.float64) /
            (count[update_conf].astype(np.float64) ** 2)
        )
    mean = np.zeros_like(prevmean)
    denom = count + prevnocount + 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mean[update_conf] = (prevmean[update_conf] * (count[update_conf] + prevnocount[update_conf]) +
                             curr_metric_conf[update_conf]) / denom[update_conf]
    tempconf = (mean * count.astype(np.float64) * count.astype(np.float64)).astype(np.int32)
    confidence[update_conf] = np.clip(tempconf[update_conf], 0, conf_upper_lim)

    # Update confidence for new disturbances
    new_conf = ((status == NODIST) | (status == NODATA)) & disturbed
    confidence[new_conf] = np.minimum(current_metric[new_conf], metric_value_upper_lim)

    last_metric_date = date + duration - 1

    nocount = ((last_obs > last_metric_date).astype(np.int8) +
               (current_metric < lowthresh).astype(np.int8))

    # Updates for active disturbances
    # High threshold disturbances
    updating_current = (status <= CONFHI) | (status == NODATA)
    hi_mask = updating_current & (max_metric >= highthresh)
    conf_hi = hi_mask & (confidence >= conf_thresh)
    first_hi = hi_mask & (duration == 1)
    prov_hi = hi_mask & ~(conf_hi | first_hi) & (status != CONFHI)
    status[conf_hi] = CONFHI
    status[first_hi] = FIRSTHI
    status[prov_hi] = PROVHI
    # Low threshold disturbances
    lo_mask = updating_current & (max_metric >= lowthresh) & (max_metric < highthresh)
    conf_lo = lo_mask & (confidence >= conf_thresh)
    first_lo = lo_mask & (duration == 1)
    prov_lo = lo_mask & ~(conf_lo | first_lo) & (status != CONFLO)
    status[conf_lo] = CONFLO
    status[first_lo] = FIRSTLO
    status[prov_lo] = PROVLO

    # Reset ongoing disturbances if max_anom drops below lowthresh
    status[updating_current & (max_metric < lowthresh)] = NODIST

    # Update must finish disturbances
    status_at_finish_check = status.copy() # Capture status before final finish logic

    # Initialize must_finish_conditions list
    must_finish_conditions = []

    # Condition A: Observation gap too large
    must_finish_conditions.append(((current_date - last_metric_date) >= nodaylimit) & (last_metric_date > 0))
    # Condition B: Short disturbance with low current metric
    must_finish_conditions.append((duration == 1) & (current_metric < lowthresh))
    # Condition C: Conditional nocount logic (used if `consecutive_nodist` is set to True)
    if consecutive_nodist:
        must_finish_conditions.append(nocount == 2)
    # Condition D: Percent below threshold
    # If the percent of disturbed observations drops below threshold, it triggers reset.
    must_finish_conditions.append(percent < percent_reset_thresh)
    # Condition E: Number of non-disturbed observations (prevnocount) below threshold
    # If the number of non-disturbed observations in a series (prevnocount) is high,
    # the disturbance resets.
    must_finish_conditions.append(prevnocount >= nocount_reset_thresh)

    # Combine all must_finish_conditions with OR
    if must_finish_conditions:
        combined_must_finish_criteria = np.logical_or.reduce(must_finish_conditions)
    else:
        combined_must_finish_criteria = np.full(status.shape, False, dtype=bool)

    must_finish = (status_at_finish_check <= CONFHI) & combined_must_finish_criteria

    # Apply finished status
    status[must_finish & (status_at_finish_check == CONFLO)] = CONFLOFIN
    status[must_finish & (status_at_finish_check == CONFHI)] = CONFHIFIN

    # Reset other finished pixels to NODIST
    reset_finish = must_finish & ~((status_at_finish_check == CONFLO) | (status_at_finish_check == CONFHI))
    status[reset_finish] = NODIST
    percent[reset_finish] = 0
    count[reset_finish] = 0
    max_metric[reset_finish] = 0
    confidence[reset_finish] = 0
    date[reset_finish] = 0
    duration[reset_finish] = 0

    # Update last observation date for all valid pixels
    last_obs[valid] = current_date

    # Prepare profiles for serialization
    p_dist_int8 = anom_prof.copy()
    p_dist_int8['nodata'] = 255
    p_dist_int8['dtype'] = np.uint8

    p_dist_int16 = anom_prof.copy()
    p_dist_int16['nodata'] = 255
    p_dist_int16['dtype'] = np.int16

    # Serialize output
    serialize_one_2d_ds(status, p_dist_int8, out_path_list[0])
    serialize_one_2d_ds(max_metric, p_dist_int16, out_path_list[1])
    serialize_one_2d_ds(confidence, p_dist_int16, out_path_list[2])
    serialize_one_2d_ds(date, p_dist_int16, out_path_list[3])
    serialize_one_2d_ds(count, p_dist_int8, out_path_list[4])
    serialize_one_2d_ds(percent, p_dist_int8, out_path_list[5])
    serialize_one_2d_ds(duration, p_dist_int16, out_path_list[6])
    serialize_one_2d_ds(last_obs, p_dist_int16, out_path_list[7])