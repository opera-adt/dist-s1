import datetime
from pathlib import Path

import numpy as np
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from distmetrics.model_load import load_transformer_model
from distmetrics.rio_tools import merge_categorical_arrays, merge_with_weighted_overlap
from distmetrics.tf_inference import estimate_normal_params
from scipy.special import logit

from dist_s1.constants import DISTLABEL2VAL, DIST_CMAP
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
    use_date_encoding: bool = False,
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
    model_dtype: np.dtype = 'float32',
    raw_data_for_nodata_mask: Path | str | None = None,
) -> None:
    model = load_transformer_model(
        lib_model_token=model_source,
        dtype=model_dtype,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        device=device,
        model_compilation=model_compilation,
        batch_size=batch_size,
    )

    if use_date_encoding:
        print(f'Using acq_dts {acq_dts}')

    pre_copol_data = [open_one_ds(path) for path in pre_copol_paths]
    pre_crosspol_data = [open_one_ds(path) for path in pre_crosspol_paths]
    post_copol_data = open_one_ds(post_copol_path)
    post_crosspol_data = open_one_ds(post_crosspol_path)

    pre_copol_arrs, pre_copol_profs = zip(*pre_copol_data)
    pre_crosspol_arrs, pre_crosspol_profs = zip(*pre_crosspol_data)
    post_copol_arr, post_copol_prof = post_copol_data
    post_crosspol_arr, post_crosspol_prof = post_crosspol_data

    if raw_data_for_nodata_mask is not None:
        arr, _ = open_one_ds(raw_data_for_nodata_mask)
        mask_2d = np.isnan(arr)
    else:
        mask_2d = np.isnan(post_copol_arr) | np.isnan(post_crosspol_arr)

    if fill_value <= 0:
        fill_value = 1e-7

    # Preserve nodata values for metric
    # Fill nodata values with fill_value for model that is 1e-7
    pre_copol_arrs = [np.where(np.isnan(arr), fill_value, arr) for arr in pre_copol_arrs]
    pre_crosspol_arrs = [np.where(np.isnan(arr), fill_value, arr) for arr in pre_crosspol_arrs]
    post_copol_arr = np.where(np.isnan(post_copol_arr), fill_value, post_copol_arr)
    post_crosspol_arr = np.where(np.isnan(post_crosspol_arr), fill_value, post_crosspol_arr)

    if use_logits:
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
        dtype=model_dtype,
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
