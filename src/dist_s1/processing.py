from pathlib import Path

import numpy as np
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from distmetrics.despeckle import despeckle_rtc_arrs_with_tv
from distmetrics.rio_tools import merge_categorical_arrays, merge_with_weighted_overlap
from distmetrics.transformer import estimate_normal_params_of_logits, load_transformer_model
from scipy.special import logit
from tqdm import tqdm

from dist_s1.constants import COLORBLIND_DIST_CMAP, DISTLABEL2VAL
from dist_s1.rio_tools import check_profiles_match, get_mgrs_profile, open_one_ds, serialize_one_2d_ds


def despeckle_and_serialize_rtc_s1(
    rtc_s1_paths: list[Path],
    dst_paths: list[Path],
    batch_size: int = 100,
    tqdm_enabled: bool = True,
    n_workers: int = 5,
) -> list[Path]:
    # Cast to Path
    dst_paths = list(map(Path, dst_paths))
    # Make sure the parent directories exist
    [p.parent.mkdir(exist_ok=True, parents=True) for p in dst_paths]

    n_batches = int(np.ceil(len(rtc_s1_paths) / batch_size))
    for k in tqdm(range(n_batches), desc='Despeckling batch', disable=not tqdm_enabled):
        paths_subset = rtc_s1_paths[k * batch_size : (k + 1) * batch_size]
        dst_paths_subset = dst_paths[k * batch_size : (k + 1) * batch_size]

        # don't overwrite existing data
        dst_paths_subset_to_create = [dst_p for dst_p in dst_paths_subset if not dst_p.exists()]
        paths_subset_to_create = [src_p for (src_p, dst_p) in zip(paths_subset, dst_paths_subset) if not dst_p.exists()]

        # open
        if dst_paths_subset_to_create:
            data = list(map(open_one_ds, paths_subset_to_create))
            arrs, ps = zip(*data)
            # despeckle
            arrs_d = despeckle_rtc_arrs_with_tv(arrs, tqdm_enabled=tqdm_enabled, n_jobs=n_workers)
            # serialize
            [serialize_one_2d_ds(arr, prof, dst_path) for (arr, prof, dst_path) in zip(arrs_d, ps, dst_paths_subset)]

    return dst_paths


def compute_normal_params_per_burst_and_serialize(
    pre_copol_paths_dskpl_paths: list[Path],
    pre_crosspol_paths_dskpl_paths: list[Path],
    out_path_mu_copol: Path,
    out_path_mu_crosspol: Path,
    out_path_sigma_copol: Path,
    out_path_sigma_crosspol: Path,
    memory_strategy: str = 'high',
    device: str = 'best',
) -> Path:
    if device not in ('cpu', 'cuda', 'mps', 'best'):
        raise ValueError(f'Invalid device: {device}')
    # For distmetrics, None is how we choose the "best" available device
    if device == 'best':
        device = None
    model = load_transformer_model(device=device)

    copol_data = [open_one_ds(path) for path in pre_copol_paths_dskpl_paths]
    crosspol_data = [open_one_ds(path) for path in pre_crosspol_paths_dskpl_paths]
    arrs_copol, profs_copol = zip(*copol_data)
    arrs_crosspol, profs_crosspol = zip(*crosspol_data)

    if len(arrs_copol) != len(arrs_crosspol):
        raise ValueError('Length of Copolar and crosspolar arrays do not match')
    p_ref = profs_copol[0]
    for p_copol, p_crosspol in zip(profs_copol, profs_crosspol):
        check_profiles_match(p_ref, p_copol)
        check_profiles_match(p_ref, p_crosspol)

    logits_mu, logits_sigma = estimate_normal_params_of_logits(
        model, arrs_copol, arrs_crosspol, memory_strategy=memory_strategy, device=device
    )
    logits_mu_copol, logits_mu_crosspol = logits_mu[0, ...], logits_mu[1, ...]
    logits_sigma_copol, logits_sigma_crosspol = logits_sigma[0, ...], logits_sigma[1, ...]

    serialize_one_2d_ds(logits_mu_copol, p_ref, out_path_mu_copol)
    serialize_one_2d_ds(logits_mu_crosspol, p_ref, out_path_mu_crosspol)
    serialize_one_2d_ds(logits_sigma_copol, p_ref, out_path_sigma_copol)
    serialize_one_2d_ds(logits_sigma_crosspol, p_ref, out_path_sigma_crosspol)


def compute_logit_mdist(arr_logit: np.ndarray, mean_logit: np.ndarray, sigma_logit: np.ndarray) -> np.ndarray:
    return np.abs(arr_logit - mean_logit) / sigma_logit


def label_one_disturbance(
    mdist: np.ndarray, moderate_confidence_threshold: float, high_confidence_threshold: float
) -> np.ndarray:
    nodata_mask = np.isnan(mdist)
    arr = np.zeros_like(mdist)
    arr[mdist > moderate_confidence_threshold] = 1
    arr[mdist > high_confidence_threshold] = 2
    arr[nodata_mask] = 255
    return arr


def aggregate_disturbance_over_time(
    disturbance_one_look_l: list[np.ndarray],
    moderate_confidence_label: float = 1,
    high_confidence_label: float = 2,
    max_lookbacks: int | None = None,
    nodata_value: int = 255,
) -> np.ndarray:
    n_looks = len(disturbance_one_look_l)

    # Mask - easier to handle 255 ignore via numpy builtins of nan
    disturbance_stack = np.stack(disturbance_one_look_l, axis=0).astype(np.float32)
    disturbance_stack[disturbance_stack == nodata_value] = np.nan
    # Right now, we require the n_looks to have disturbances for all Deltas - any nodata will be no disturbance
    nodata_mask = np.any(np.isnan(disturbance_stack), axis=0)

    if (max_lookbacks is not None) and (n_looks > max_lookbacks) or (n_looks == 0):
        raise ValueError(
            f'Number of looks ({n_looks}) exceeds maximum number of lookbacks ({max_lookbacks}) or is zero.'
        )
    if n_looks == 1:
        X_agg = disturbance_stack.squeeze(0)
        X_agg[X_agg == moderate_confidence_label] = DISTLABEL2VAL['first_moderate_conf_disturbance']
        X_agg[X_agg == high_confidence_label] = DISTLABEL2VAL['first_high_conf_disturbance']
    elif len(disturbance_one_look_l) > 1:
        X_dist_min = np.nanmin(disturbance_stack, axis=0)
        X_dist_count = np.nansum((disturbance_stack != 0), axis=0).astype(np.uint8)
        X_agg = np.zeros_like(X_dist_min)
        # Requires all looks to have data to be labeled otherwise X_dist_count < n_looks
        ind_moderate = (X_dist_count == n_looks) & (X_dist_min == moderate_confidence_label)
        ind_high = (X_dist_count == n_looks) & (X_dist_min == high_confidence_label)
        if n_looks == 2:
            X_agg[ind_moderate] = DISTLABEL2VAL['provisional_moderate_conf_disturbance']
            X_agg[ind_high] = DISTLABEL2VAL['provisional_high_conf_disturbance']
        elif n_looks == 3:
            X_agg[ind_moderate] = DISTLABEL2VAL['confirmed_moderate_conf_disturbance']
            X_agg[ind_high] = DISTLABEL2VAL['confirmed_high_conf_disturbance']
        else:
            raise NotImplementedError(f'Number of scenes ({n_looks}) is not implemented.')
    X_agg[nodata_mask] = nodata_value
    X_agg = X_agg.astype(np.uint8)
    return X_agg


def compute_burst_disturbance_for_lookback_group_and_serialize(
    *,
    copol_paths: list[Path],
    crosspol_paths: list[Path],
    logit_mean_copol_path: Path,
    logit_mean_crosspol_path: Path,
    logit_sigma_copol_path: Path,
    logit_sigma_crosspol_path: Path,
    out_dist_path: Path,
    max_lookbacks: int,
    moderate_confidence_threshold: float,
    high_confidence_threshold: float,
    out_metric_path: Path | None = None,
) -> None:
    if len(copol_paths) != len(crosspol_paths):
        raise ValueError('Length of Copolar and crosspolar arrays do not match')
    if len(copol_paths) > max_lookbacks:
        raise ValueError(
            f'Number of looks ({len(copol_paths)}) exceeds maximum number of lookbacks ({max_lookbacks}) for '
            f'paths: {copol_paths} and {crosspol_paths}'
        )
    copol_data = [open_one_ds(path) for path in copol_paths]
    crosspol_data = [open_one_ds(path) for path in crosspol_paths]

    arrs_copol, profs_copol = zip(*copol_data)
    arrs_crosspol, profs_crosspol = zip(*crosspol_data)

    logit_arrs_copol = [logit(arr) for arr in arrs_copol]
    logit_arrs_crosspol = [logit(arr) for arr in arrs_crosspol]

    logit_mean_copol, p_mean_copol = open_one_ds(logit_mean_copol_path)
    logit_mean_crosspol, p_mean_crosspol = open_one_ds(logit_mean_crosspol_path)
    logit_sigma_copol, p_sigma_copol = open_one_ds(logit_sigma_copol_path)
    logit_sigma_crosspol, p_sigma_crosspol = open_one_ds(logit_sigma_crosspol_path)

    [
        check_profiles_match(p_mean_copol, p)
        for p in [p_mean_crosspol, p_sigma_copol, p_sigma_crosspol, profs_crosspol[0], profs_copol[0]]
    ]

    mdist_copol_l = [compute_logit_mdist(arr, logit_mean_copol, logit_sigma_copol) for arr in logit_arrs_copol]
    mdist_crosspol_l = [
        compute_logit_mdist(arr, logit_mean_crosspol, logit_sigma_crosspol) for arr in logit_arrs_crosspol
    ]

    mdist_l = [
        np.nanmax(np.stack([mdist_copol, mdist_crosspol], axis=0), axis=0)
        for mdist_copol, mdist_crosspol in zip(mdist_copol_l, mdist_crosspol_l)
    ]

    # Intermediate (single comparison with baseline using moderate/high confidence thresholds):
    # 0 - No disturbance
    # 1 - Moderate confidence
    # 2 - High confidence
    # 255 - Nodata
    disturbance_one_look_l = [
        label_one_disturbance(mdist, moderate_confidence_threshold, high_confidence_threshold) for mdist in mdist_l
    ]

    # Translates intermediate labels to disturbance labels dictated in constants.py
    disturbance_temporal_agg = aggregate_disturbance_over_time(disturbance_one_look_l, max_lookbacks=max_lookbacks)

    p_dist_ref = profs_copol[0].copy()
    p_dist_ref['nodata'] = 255
    p_dist_ref['dtype'] = np.uint8

    serialize_one_2d_ds(disturbance_temporal_agg, p_dist_ref, out_dist_path)
    if out_metric_path is not None:
        serialize_one_2d_ds(mdist_l[-1], profs_copol[0], out_metric_path)


def aggregate_disturbance_over_lookbacks(X_delta_l: list[np.ndarray]) -> np.ndarray:
    X_agg = np.zeros_like(X_delta_l[0])
    # nodata is where any lookback group had nodata
    nodata_mask = np.any(np.stack(X_delta_l, axis=0) == 255, axis=0)
    # priority is to largest lookback where we can confirm disturbances
    for X_delta in X_delta_l:
        ind = ~np.isin(X_delta, [0, 255])
        X_agg[ind] = X_delta[ind]
    X_agg[nodata_mask] = 255
    return X_agg


def aggregate_burst_disturbance_over_lookbacks_and_serialize(
    disturbance_paths: list[Path], out_path: Path, max_lookbacks: int
) -> None:
    stems = [Path(p).stem for p in disturbance_paths]
    # Make sure the paths are in order of lookback, delta key is the last token
    assert 'delta' in stems[0].split('_')[-1]
    if sorted(stems, key=lambda x: x.split('_')[-1]) != stems:
        raise ValueError('Disturbance paths must be supplied in order of lookback.')
    if len(disturbance_paths) != max_lookbacks:
        raise ValueError(f'Expected {max_lookbacks} disturbance paths, got {len(disturbance_paths)}.')

    data = [open_one_ds(path) for path in disturbance_paths]
    X_delta_l, profs = zip(*data)
    for p in profs[1:]:
        check_profiles_match(profs[0], p)

    X_time_agg = aggregate_disturbance_over_lookbacks(X_delta_l)

    p_ref = profs[0]
    serialize_one_2d_ds(X_time_agg, p_ref, out_path)


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
    serialize_one_2d_ds(X_dist_mgrs, p_dist_mgrs, dst_path, colormap=COLORBLIND_DIST_CMAP)


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
