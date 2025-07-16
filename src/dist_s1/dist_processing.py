import datetime
from pathlib import Path

import numpy as np
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from distmetrics.model_load import load_transformer_model
from distmetrics.rio_tools import merge_categorical_arrays, merge_with_weighted_overlap
from distmetrics.tf_inference import estimate_normal_params
from scipy.special import logit

from dist_s1.constants import BASE_DATE, DISTLABEL2VAL, DIST_CMAP
from dist_s1.rio_tools import check_profiles_match, get_mgrs_profile, open_one_ds, serialize_one_2d_ds


def compute_logit_mdist(arr_logit: np.ndarray, mean_logit: np.ndarray, sigma_logit: np.ndarray) -> np.ndarray:
    return np.abs(arr_logit - mean_logit) / sigma_logit


def label_one_disturbance(
    mdist: np.ndarray, moderate_confidence_threshold: float, high_confidence_threshold: float
) -> np.ndarray:
    nodata_mask = np.isnan(mdist)
    dist_labels = np.zeros_like(mdist)
    dist_labels[mdist >= moderate_confidence_threshold] = 1
    dist_labels[mdist >= high_confidence_threshold] = 2
    dist_labels[nodata_mask] = 255
    return dist_labels


def label_alert_intermediate(
    intermediate_labels: np.ndarray,
    *,
    moderate_confidence_label: float,
    high_confidence_label: float,
) -> np.ndarray:
    # Note nodata is taken care of in the intermediate labels
    dist_labels = np.zeros_like(intermediate_labels, dtype=np.uint8)
    dist_labels[intermediate_labels == moderate_confidence_label] = DISTLABEL2VAL['first_moderate_conf_disturbance']
    dist_labels[intermediate_labels == high_confidence_label] = DISTLABEL2VAL['first_high_conf_disturbance']
    return dist_labels


def compute_burst_disturbance_and_serialize(
    *,
    pre_copol_paths: list[Path | str],
    pre_crosspol_paths: list[Path | str],
    post_copol_path: Path | str,
    post_crosspol_path: Path | str,
    acq_dts: list[datetime.datetime],
    out_dist_path: Path,
    moderate_confidence_threshold: float,
    high_confidence_threshold: float,
    out_metric_path: Path | None = None,
    utilize_acq_dts: bool = False,
    use_logits: bool = True,
    model_source: str | Path | None = 'transformer_optimized',
    model_cfg_path: str | Path | None = None,
    model_wts_path: str | Path | None = None,
    memory_strategy: str = 'high',
    stride: int = 16,
    batch_size: int = 32,
    device: str = 'best',
    model_compilation: bool = False,
    fill_value: float = 1e-7,
    dtype: np.dtype = 'float32',
) -> None:
    model = load_transformer_model(
        model_token=model_source,
        dtype=dtype,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        device=device,
        model_compilation=model_compilation,
        batch_size=batch_size,
    )

    if utilize_acq_dts:
        print(f'Using acq_dts {acq_dts}')

    pre_copol_data = [open_one_ds(path) for path in pre_copol_paths]
    pre_crosspol_data = [open_one_ds(path) for path in pre_crosspol_paths]
    post_copol_data = open_one_ds(post_copol_path)
    post_crosspol_data = open_one_ds(post_crosspol_path)

    pre_copol_arrs, pre_copol_profs = zip(*pre_copol_data)
    pre_crosspol_arrs, pre_crosspol_profs = zip(*pre_crosspol_data)
    post_copol_arr, post_copol_prof = post_copol_data
    post_crosspol_arr, post_crosspol_prof = post_crosspol_data

    if fill_value <= 0:
        fill_value = 1e-7

    # Preserve nodata values for metric
    mask_2d = np.isnan(post_copol_arr) | np.isnan(post_crosspol_arr)
    # Fill nodata values with fill_value for model
    pre_copol_arrs = [np.where(np.isnan(arr), fill_value, arr) for arr in pre_copol_arrs]
    pre_crosspol_arrs = [np.where(np.isnan(arr), fill_value, arr) for arr in pre_crosspol_arrs]
    post_copol_arr = np.where(np.isnan(post_copol_arr), fill_value, post_copol_arr)
    post_crosspol_arr = np.where(np.isnan(post_crosspol_arr), fill_value, post_crosspol_arr)

    if use_logits:
        # TODO: Remove logit transformation from model application
        pre_copol_arrs = [logit(arr) for arr in pre_copol_arrs]
        pre_crosspol_arrs = [logit(arr) for arr in pre_crosspol_arrs]
        post_copol_arr = logit(post_copol_arr)
        post_crosspol_arr = logit(post_crosspol_arr)

    p_ref = pre_copol_profs[0].copy()
    [
        check_profiles_match(p_ref, p)
        for p in list(pre_copol_profs) + list(pre_crosspol_profs) + [post_copol_prof] + [post_crosspol_prof]
    ]

    mu_2d, sigma_2d = estimate_normal_params(
        model,
        pre_copol_arrs,
        pre_crosspol_arrs,
        memory_strategy=memory_strategy,
        device=device,
        stride=stride,
        batch_size=batch_size,
    )

    post_arr_2d = np.stack([post_copol_arr, post_crosspol_arr], axis=0)
    z_score_per_channel = np.abs(post_arr_2d - mu_2d) / sigma_2d
    metric = np.nanmax(z_score_per_channel, axis=0)
    metric[mask_2d] = np.nan

    # Intermediate (single comparison with baseline using moderate/high confidence thresholds):
    # 0 - No disturbance
    # 1 - Moderate confidence
    # 2 - High confidence
    # 255 - Nodata
    intermediate_labels = label_one_disturbance(metric, moderate_confidence_threshold, high_confidence_threshold)

    # Translates intermediate labels to disturbance labels dictated in constants.py
    # See labels (0, 1, 2) provided in previuos function
    alert_intermediate = label_alert_intermediate(
        intermediate_labels,
        moderate_confidence_label=1,
        high_confidence_label=2,
    )

    p_dist_ref = p_ref.copy()
    p_dist_ref['nodata'] = 255
    p_dist_ref['dtype'] = np.uint8

    p_metric_ref = p_ref.copy()
    p_metric_ref['nodata'] = np.nan
    p_metric_ref['dtype'] = np.float32

    serialize_one_2d_ds(alert_intermediate, p_dist_ref, out_dist_path, colormap=DIST_CMAP)
    serialize_one_2d_ds(metric, p_metric_ref, out_metric_path)


def compute_tile_disturbance_using_previous_product_and_serialize(
    dist_metric_path: Path,
    dist_metric_date: Path,
    out_path_list: list[str],
    lowthresh: float = 3.5,
    highthresh: float = 5.5,
    nodaylimit: int = 18,
    max_obs_num_year: int = 253,
    conf_upper_lim: int = 32000,
    conf_thresh: float = 3**2 * 3.5,
    metric_value_upper_lim: float = 100,
    base_date: datetime.datetime | None = None,
    previous_dist_arr_path_list: list[str | None] | None = None,
) -> None:
    # Status codes
    NODIST, FIRSTLO, PROVLO, CONFLO, FIRSTHI, PROVHI, CONFHI, CONFLOFIN, CONFHIFIN, NODATA = range(10)
    NODATA = 255  # right now this function is not used, check the function.
    if base_date is None:
        base_date = BASE_DATE

    # Get dist_date from a sample path pattern
    dist_date = datetime.datetime.strptime(Path(dist_metric_date).name.split('_')[4], '%Y%m%dT%H%M%SZ')

    # Load current metric and profile
    currAnom, anom_prof = open_one_ds(dist_metric_path)
    rows, cols = currAnom.shape
    currDate = (dist_date - base_date).days

    # Initialize or load previous arrays
    if previous_dist_arr_path_list is None:
        print('This is a first product tile with date', dist_date)
        status = np.full((rows, cols), 255, dtype=np.uint8)
        max_anom = np.full((rows, cols), -1, dtype=np.int16)
        conf = np.zeros((rows, cols), dtype=np.int16)
        date = np.zeros((rows, cols), dtype=np.int16)
        count = np.zeros((rows, cols), dtype=np.uint8)
        percent = np.full((rows, cols), 200, dtype=np.uint8)
        dur = np.zeros((rows, cols), dtype=np.int16)
        lastObs = np.zeros((rows, cols), dtype=np.int16)
    elif all(p is not None for p in previous_dist_arr_path_list):
        status, _ = open_one_ds(previous_dist_arr_path_list[0])
        max_anom, _ = open_one_ds(previous_dist_arr_path_list[1])
        conf, _ = open_one_ds(previous_dist_arr_path_list[2])
        date, _ = open_one_ds(previous_dist_arr_path_list[3])
        count, _ = open_one_ds(previous_dist_arr_path_list[4])
        percent, _ = open_one_ds(previous_dist_arr_path_list[5])
        dur, _ = open_one_ds(previous_dist_arr_path_list[6])
        lastObs, _ = open_one_ds(previous_dist_arr_path_list[7])
    else:
        raise ValueError('Missing previous product data for non-first acquisitions.')

    valid = currAnom >= 0

    reset_mask = ((currDate - date) > 365) | ((status > 6) & (currAnom >= lowthresh))
    reset_mask &= valid
    status[reset_mask] = NODIST
    percent[reset_mask] = 200
    count[reset_mask] = 0
    max_anom[reset_mask] = 0
    conf[reset_mask] = 0
    date[reset_mask] = 0
    dur[reset_mask] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        prevnocount = np.where(
            (percent > 0) & (percent <= 100), ((100 - percent) / percent * count).astype(np.int32), 0
        )

    disturbed = (currAnom >= lowthresh) & valid
    new_detection = disturbed & ((status == NODIST) | (status == NODATA))
    date[new_detection] = currDate
    max_anom[new_detection] = currAnom[new_detection]
    percent[new_detection] = 100
    count[new_detection] = 1

    continuing = disturbed & ~new_detection
    max_anom[continuing] = np.maximum(max_anom[continuing], currAnom[continuing])
    can_increment = continuing & (count < max_obs_num_year)
    count[can_increment] += 1
    percent[can_increment] = (
        (count[can_increment] * 100) / (count[can_increment] + prevnocount[can_increment])
    ).astype(np.uint8)

    dur[disturbed] = currDate - date[disturbed] + 1

    not_disturbed = (~disturbed) & valid
    adjust_percent = not_disturbed & (percent > 0) & (percent <= 100) & (count < max_obs_num_year + 1)
    prevcount = count.copy()
    percent[adjust_percent] = (
        (count[adjust_percent] * 100) / (prevcount[adjust_percent] + prevnocount[adjust_percent] + 1)
    ).astype(np.uint8)

    status_reset = not_disturbed & (status == NODATA)
    status[status_reset] = NODIST
    percent[status_reset] = 200
    count[status_reset] = 0
    max_anom[status_reset] = 0
    conf[status_reset] = 0
    date[status_reset] = 0
    dur[status_reset] = 0

    update_conf = (conf > 0) & (status <= 6) & valid
    currAnomConf = np.minimum(currAnom, metric_value_upper_lim)
    prevmean = np.zeros_like(conf, dtype=np.float64)
    prevmean[update_conf] = conf[update_conf] / (count[update_conf] ** 2)
    mean = np.zeros_like(prevmean)
    denom = count + prevnocount + 1
    mean[update_conf] = (
        prevmean[update_conf] * (count[update_conf] + prevnocount[update_conf]) + currAnomConf[update_conf]
    ) / denom[update_conf]
    tempconf = (mean * count * count).astype(np.int32)
    conf[update_conf] = np.clip(tempconf[update_conf], 0, conf_upper_lim)

    new_conf = ((status == NODIST) | (status == NODATA)) & disturbed
    conf[new_conf] = np.minimum(currAnom[new_conf], metric_value_upper_lim)

    lastAnomDate = date + dur - 1
    nocount = (lastObs > lastAnomDate).astype(np.int8) + (currAnom < lowthresh).astype(np.int8)

    updating = (status <= 6) | (status == NODATA)
    must_finish = updating & (
        (nocount == 2)
        | ((currDate - lastAnomDate) >= nodaylimit) & (lastAnomDate > 0)
        | ((dur == 1) & (currAnom < lowthresh))
    )

    status[must_finish & (status == CONFLO)] = CONFLOFIN
    status[must_finish & (status == CONFHI)] = CONFHIFIN

    reset_finish = must_finish & ~((status == CONFLO) | (status == CONFHI))
    status[reset_finish] = NODIST
    percent[reset_finish] = 0
    count[reset_finish] = 0
    max_anom[reset_finish] = 0
    conf[reset_finish] = 0
    date[reset_finish] = 0
    dur[reset_finish] = 0

    hi_mask = updating & (max_anom >= highthresh)
    conf_hi = hi_mask & (conf >= conf_thresh)
    first_hi = hi_mask & (dur == 1)
    prov_hi = hi_mask & ~(conf_hi | first_hi) & (status != CONFHI)
    status[conf_hi] = CONFHI
    status[first_hi] = FIRSTHI
    status[prov_hi] = PROVHI

    lo_mask = updating & (max_anom >= lowthresh) & (max_anom < highthresh)
    conf_lo = lo_mask & (conf >= conf_thresh)
    first_lo = lo_mask & (dur == 1)
    prov_lo = lo_mask & ~(conf_lo | first_lo) & (status != CONFLO)
    status[conf_lo] = CONFLO
    status[first_lo] = FIRSTLO
    status[prov_lo] = PROVLO

    status[updating & (max_anom < lowthresh)] = NODIST
    lastObs[valid] = currDate

    p_dist_int8 = anom_prof.copy()
    p_dist_int8['nodata'] = 255
    p_dist_int8['dtype'] = np.uint8

    p_dist_int16 = anom_prof.copy()
    p_dist_int16['nodata'] = 255
    p_dist_int16['dtype'] = np.int16

    # Serialize output
    serialize_one_2d_ds(status, p_dist_int8, out_path_list[0])
    serialize_one_2d_ds(max_anom, p_dist_int16, out_path_list[1])
    serialize_one_2d_ds(conf, p_dist_int16, out_path_list[2])
    serialize_one_2d_ds(date, p_dist_int16, out_path_list[3])
    serialize_one_2d_ds(count, p_dist_int8, out_path_list[4])
    serialize_one_2d_ds(percent, p_dist_int8, out_path_list[5])
    serialize_one_2d_ds(dur, p_dist_int16, out_path_list[6])
    serialize_one_2d_ds(lastObs, p_dist_int16, out_path_list[7])


def merge_burst_disturbances_and_serialize(
    burst_disturbance_paths: list[Path], dst_path: Path, mgrs_tile_id: str
) -> None:
    data = [open_one_ds(path) for path in burst_disturbance_paths]
    X_dist_burst_l, profs = zip(*data)

    X_merged, p_merged = merge_categorical_arrays(X_dist_burst_l, profs, exterior_mask_dilation=20, merge_method='max')
    X_merged[0, ...] = X_merged

    p_mgrs = get_mgrs_profile(mgrs_tile_id)
    X_dist_mgrs, p_dist_mgrs = reproject_arr_to_match_profile(X_merged, p_merged, p_mgrs, resampling='nearest')
    # From BIP back to 2D array
    X_dist_mgrs = X_dist_mgrs[0, ...]
    serialize_one_2d_ds(X_dist_mgrs, p_dist_mgrs, dst_path, colormap=DIST_CMAP)


def merge_burst_metrics_and_serialize(burst_metrics_paths: list[Path], dst_path: Path, mgrs_tile_id: str) -> None:
    data = [open_one_ds(path) for path in burst_metrics_paths]
    X_metric_burst_l, profs = zip(*data)
    X_metric_merged, p_merged = merge_with_weighted_overlap(
        X_metric_burst_l,
        profs,
        exterior_mask_dilation=20,
        distance_weight_exponent=1.0,
        use_distance_weighting_from_exterior_mask=True,
    )

    p_mgrs = get_mgrs_profile(mgrs_tile_id)
    X_dist_mgrs, p_dist_mgrs = reproject_arr_to_match_profile(X_metric_merged, p_merged, p_mgrs, resampling='bilinear')
    # From BIP back to 2D array
    X_dist_mgrs = X_dist_mgrs[0, ...]
    serialize_one_2d_ds(X_dist_mgrs, p_dist_mgrs, dst_path)
