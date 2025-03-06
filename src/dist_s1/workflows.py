from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import torch.multiprocessing as tmp
from tqdm.auto import tqdm

from dist_s1.aws import upload_product_to_s3
from dist_s1.constants import MODEL_CONTEXT_LENGTH
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.localize_rtc_s1 import localize_rtc_s1
from dist_s1.packaging import generate_browse_image, package_disturbance_tifs
from dist_s1.processing import (
    aggregate_burst_disturbance_over_lookbacks_and_serialize,
    compute_burst_disturbance_for_lookback_group_and_serialize,
    compute_normal_params_per_burst_and_serialize,
    despeckle_and_serialize_rtc_s1,
    merge_burst_disturbances_and_serialize,
    merge_burst_metrics_and_serialize,
)


# Use spawn for multiprocessing
tmp.set_start_method('spawn', force=True)


def curate_input_burst_rtc_input_for_dist(
    copol_paths: list[str], crosspol_paths: list[str], lookback: int
) -> tuple[list[Path], list[Path]]:
    """Curate the paths to the correct length for mdist estimation."""
    if len(copol_paths) != len(crosspol_paths):
        raise ValueError('The number of copol and crosspol paths must be the same')
    dates_copol = [Path(path).stem.split('_')[4] for path in copol_paths]
    dates_crosspol = [Path(path).stem.split('_')[4] for path in crosspol_paths]
    if dates_copol != dates_crosspol:
        raise ValueError('The copol and crosspol paths must have the same dates')
    n_imgs = len(copol_paths)
    start = max(n_imgs - lookback - 1, 0)
    stop = n_imgs
    copol_paths_lookback_group = copol_paths[start:stop]
    crosspol_paths_lookback_group = crosspol_paths[start:stop]
    return copol_paths_lookback_group, crosspol_paths_lookback_group


def curate_input_burst_rtc_s1_paths_for_normal_param_est(
    copol_paths: list[str], crosspol_paths: list[str], lookback: int
) -> tuple[list[Path], list[Path]]:
    """Curate the paths to the correct length for normal param estimation."""
    if len(copol_paths) != len(crosspol_paths):
        raise ValueError('The number of copol and crosspol paths must be the same')
    dates_copol = [Path(path).stem.split('_')[4] for path in copol_paths]
    dates_crosspol = [Path(path).stem.split('_')[4] for path in crosspol_paths]
    if dates_copol != dates_crosspol:
        raise ValueError('The copol and crosspol paths must have the same dates')

    n_imgs = len(copol_paths)
    start = max(n_imgs - MODEL_CONTEXT_LENGTH - lookback - 1, 0)
    stop = min(n_imgs - lookback - 1, n_imgs - 1)
    copol_paths_pre = copol_paths[start:stop]
    crosspol_paths_pre = crosspol_paths[start:stop]
    return copol_paths_pre, crosspol_paths_pre


def curate_paths_for_normal_param_est_via_burst_id_and_lookback(
    df_inputs: pd.DataFrame, df_burst_distmetrics: pd.DataFrame, burst_id: str, lookback: int
) -> tuple[list[Path], list[Path]]:
    """Curate the paths to the correct length for normal param estimation."""
    indices_input = (df_inputs.jpl_burst_id == burst_id) & (df_inputs.input_category == 'pre')
    df_burst_input_data = df_inputs[indices_input].reset_index(drop=True)
    indices_input = (df_inputs.jpl_burst_id == burst_id) & (df_inputs.input_category == 'pre')
    df_burst_input_data = df_inputs[indices_input].reset_index(drop=True)
    df_metric = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)

    copol_paths = df_burst_input_data.loc_path_copol_dspkl.tolist()
    crosspol_paths = df_burst_input_data.loc_path_crosspol_dspkl.tolist()

    # curate the paths to the correct length
    copol_paths_pre, crosspol_paths_pre = curate_input_burst_rtc_s1_paths_for_normal_param_est(
        copol_paths, crosspol_paths, lookback
    )

    output_mu_copol_l = df_metric[f'loc_path_normal_mean_delta{lookback}_copol'].tolist()
    output_mu_crosspol_l = df_metric[f'loc_path_normal_mean_delta{lookback}_crosspol'].tolist()
    output_sigma_copol_l = df_metric[f'loc_path_normal_std_delta{lookback}_copol'].tolist()
    output_sigma_crosspol_l = df_metric[f'loc_path_normal_std_delta{lookback}_crosspol'].tolist()

    assert (
        len(output_mu_copol_l)
        == len(output_mu_crosspol_l)
        == len(output_sigma_copol_l)
        == len(output_sigma_crosspol_l)
        == 1
    )

    output_mu_copol_path = output_mu_copol_l[0]
    output_mu_crosspol_path = output_mu_crosspol_l[0]
    output_sigma_copol_path = output_sigma_copol_l[0]
    output_sigma_crosspol_path = output_sigma_crosspol_l[0]
    return dict(
        copol_paths_pre=copol_paths_pre,
        crosspol_paths_pre=crosspol_paths_pre,
        output_mu_copol_path=output_mu_copol_path,
        output_mu_crosspol_path=output_mu_crosspol_path,
        output_sigma_copol_path=output_sigma_copol_path,
        output_sigma_crosspol_path=output_sigma_crosspol_path,
    )


def check_dates_of_normal_param_files(pre_paths: list[str], normal_out_path: Path) -> bool:
    last_date_of_pre_paths = pre_paths[-1].split('/')[-1].split('_')[4]
    # Format in these paths is YYYYMMDDTHHMMSS - will be within 1 day of the normal param output
    ts_pre = pd.Timestamp(last_date_of_pre_paths, tz='UTC')

    # Format in this path is YYYY-MM-DD
    ts_normal_out = pd.Timestamp(normal_out_path.name.split('_')[-2], tz='UTC')
    ts_agree = (ts_pre - ts_normal_out).days < 1
    return ts_agree


def run_dist_s1_localization_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    post_date_buffer_days: int = 1,
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    apply_water_mask: bool = True,
    water_mask_path: str | Path | None = None,
) -> RunConfigData:
    # Localize inputs
    run_config = localize_rtc_s1(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        apply_water_mask=apply_water_mask,
        water_mask_path=water_mask_path,
    )

    return run_config


def run_despeckle_workflow(run_config: RunConfigData) -> None:
    """Despeckle by burst/polarization and then serializes.

    Parameters
    ----------
    run_config : RunConfigData

    Notes
    -----
    - All input and output paths are in the run_config.
    """
    # Table has input copol/crosspol paths and output despeckled paths
    df_inputs = run_config.df_inputs

    # Inputs
    copol_paths = df_inputs.loc_path_copol.tolist()
    crosspol_paths = df_inputs.loc_path_crosspol.tolist()

    # Outputs
    dspkl_copol_paths = df_inputs.loc_path_copol_dspkl.tolist()
    dspkl_crosspol_paths = df_inputs.loc_path_crosspol_dspkl.tolist()

    assert len(copol_paths) == len(dspkl_copol_paths) == len(crosspol_paths) == len(dspkl_crosspol_paths)

    # The copol/crosspol paths must be in the same order
    rtc_paths = copol_paths + crosspol_paths
    dst_paths = dspkl_copol_paths + dspkl_crosspol_paths

    despeckle_and_serialize_rtc_s1(
        rtc_paths,
        dst_paths,
        n_workers=run_config.n_workers_for_despeckling,
        batch_size=run_config.batch_size_for_despeckling,
    )


def _process_normal_params(path_data: dict, memory_strategy: str, device: str) -> None:
    return compute_normal_params_per_burst_and_serialize(
        path_data['copol_paths_pre'],
        path_data['crosspol_paths_pre'],
        path_data['output_mu_copol_path'],
        path_data['output_mu_crosspol_path'],
        path_data['output_sigma_copol_path'],
        path_data['output_sigma_crosspol_path'],
        memory_strategy=memory_strategy,
        device=device,
    )


def run_normal_param_estimation_workflow(run_config: RunConfigData) -> None:
    """Compute normal params per burst and serialize.

    Parameters
    ----------
    run_config : RunConfigData
    """
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    tqdm_disable = not run_config.tqdm_enabled
    burst_id_lookback_pairs = [
        (burst_id, lookback)
        for burst_id in df_inputs.jpl_burst_id.unique()
        for lookback in range(run_config.n_lookbacks)
    ]
    norm_param_paths = [
        curate_paths_for_normal_param_est_via_burst_id_and_lookback(df_inputs, df_burst_distmetrics, burst_id, lookback)
        for burst_id, lookback in burst_id_lookback_pairs
    ]

    if run_config.n_workers_for_norm_param_estimation == 1:
        for path_data in tqdm(
            norm_param_paths,
            disable=tqdm_disable,
            desc='Normal param estimation for burst/lookback pairs',
            dynamic_ncols=True,
            leave=False,
        ):
            compute_normal_params_per_burst_and_serialize(
                path_data['copol_paths_pre'],
                path_data['crosspol_paths_pre'],
                path_data['output_mu_copol_path'],
                path_data['output_mu_crosspol_path'],
                path_data['output_sigma_copol_path'],
                path_data['output_sigma_crosspol_path'],
                memory_strategy=run_config.memory_strategy,
                device=run_config.device,
            )
    else:
        if run_config.device in ('cuda', 'mps'):
            raise NotImplementedError('Multi-GPU processing is not supported yet')

        # Create a partial function with the memory strategy and device
        worker_fn = partial(
            _process_normal_params, memory_strategy=run_config.memory_strategy, device=run_config.device
        )

        # Start a pool of workers
        with tmp.Pool(processes=run_config.n_workers_for_norm_param_estimation) as pool:
            # Map the work to the pool and show progress
            list(
                tqdm(
                    pool.imap(worker_fn, norm_param_paths),
                    total=len(norm_param_paths),
                    desc='Normal param estimation for burst/lookback pairs',
                    dynamic_ncols=True,
                )
            )


def run_burst_disturbance_workflow(run_config: RunConfigData) -> None:
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    tqdm_disable = not run_config.tqdm_enabled
    for burst_id in tqdm(df_inputs.jpl_burst_id.unique(), disable=tqdm_disable, desc='Burst disturbance'):
        indices_input = df_inputs.jpl_burst_id == burst_id
        df_burst_input_data = df_inputs[indices_input].reset_index(drop=True)
        df_metric_burst = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)

        assert df_metric_burst.shape[0] == 1

        copol_paths = sorted(df_burst_input_data.loc_path_copol_dspkl.tolist())
        crosspol_paths = sorted(df_burst_input_data.loc_path_crosspol_dspkl.tolist())

        for lookback in tqdm(range(run_config.n_lookbacks), disable=tqdm_disable, desc='Lookbacks'):
            logit_mean_copol_path = df_metric_burst[f'loc_path_normal_mean_delta{lookback}_copol'].iloc[0]
            logit_mean_crosspol_path = df_metric_burst[f'loc_path_normal_mean_delta{lookback}_crosspol'].iloc[0]
            logit_sigma_copol_path = df_metric_burst[f'loc_path_normal_std_delta{lookback}_copol'].iloc[0]
            logit_sigma_crosspol_path = df_metric_burst[f'loc_path_normal_std_delta{lookback}_crosspol'].iloc[0]

            dist_path_lookback_l = df_metric_burst[f'loc_path_disturb_delta{lookback}'].tolist()
            assert len(dist_path_lookback_l) == 1
            output_dist_path = dist_path_lookback_l[0]

            copol_paths_lookback_group, crosspol_paths_lookback_group = curate_input_burst_rtc_input_for_dist(
                copol_paths, crosspol_paths, lookback
            )
            # breakpoint()
            output_metric_path = None
            if lookback == 0:
                output_metric_path = df_metric_burst[f'loc_path_metric_delta{lookback}'].iloc[0]

            # Computes the disturbance for a a single lookback group and serlialize
            # Delta_0, Delta_1, ..., Delta_N_LOOKBACKS
            # Labels will be 0 for no disturbance, 1 for moderate confidence disturbance,
            # 2 for high confidence disturbance, and 255 for nodata
            compute_burst_disturbance_for_lookback_group_and_serialize(
                copol_paths=copol_paths_lookback_group,
                crosspol_paths=crosspol_paths_lookback_group,
                logit_mean_copol_path=logit_mean_copol_path,
                logit_mean_crosspol_path=logit_mean_crosspol_path,
                logit_sigma_copol_path=logit_sigma_copol_path,
                logit_sigma_crosspol_path=logit_sigma_crosspol_path,
                out_dist_path=output_dist_path,
                out_metric_path=output_metric_path,
                max_lookbacks=run_config.n_lookbacks,
                moderate_confidence_threshold=run_config.moderate_confidence_threshold,
                high_confidence_threshold=run_config.high_confidence_threshold,
            )
        # Aggregate over lookbacks
        time_aggregated_disturbance_path = df_metric_burst['loc_path_disturb_time_aggregated'].iloc[0]
        disturbance_paths = [
            df_metric_burst[f'loc_path_disturb_delta{lookback}'].iloc[0] for lookback in range(run_config.n_lookbacks)
        ]
        # Aggregate the disturbances maps for all the lookbacks computed above
        # This will have the labels of the final disturbance map (see constants.py and the function itself)
        aggregate_burst_disturbance_over_lookbacks_and_serialize(
            disturbance_paths, time_aggregated_disturbance_path, run_config.n_lookbacks
        )


def run_disturbance_merge_workflow(run_config: RunConfigData) -> None:
    dst_tif_paths = run_config.final_unformatted_tif_paths

    # Metrics
    metric_burst_paths = run_config.df_burst_distmetrics['loc_path_metric_delta0'].tolist()
    dst_metric_path = dst_tif_paths['metric_status_path']
    merge_burst_metrics_and_serialize(metric_burst_paths, dst_metric_path, run_config.mgrs_tile_id)

    # Disturbance
    dist_burst_paths = run_config.df_burst_distmetrics['loc_path_disturb_time_aggregated'].tolist()
    dst_dist_path = dst_tif_paths['alert_status_path']
    merge_burst_disturbances_and_serialize(dist_burst_paths, dst_dist_path, run_config.mgrs_tile_id)

    for lookback in range(run_config.n_lookbacks):
        dist_burst_paths_delta0 = run_config.df_burst_distmetrics[f'loc_path_disturb_delta{lookback}'].tolist()
        dst_last_pass_path = dst_tif_paths[f'alert_delta{lookback}_path']
        merge_burst_disturbances_and_serialize(dist_burst_paths_delta0, dst_last_pass_path, run_config.mgrs_tile_id)


def run_dist_s1_processing_workflow(run_config: RunConfigData) -> RunConfigData:
    # Despeckle by burst
    run_despeckle_workflow(run_config)

    # Compute normal params for logit transformed data per burst
    run_normal_param_estimation_workflow(run_config)

    # Compute disturbance per burst and all possible lookbacks
    run_burst_disturbance_workflow(run_config)

    # Merge the burst-wise products
    run_disturbance_merge_workflow(run_config)

    return run_config


def run_dist_s1_packaging_workflow(run_config: RunConfigData) -> Path:
    package_disturbance_tifs(run_config)
    generate_browse_image(run_config)

    product_data = run_config.product_data_model
    product_data.validate_tif_layer_dtypes()
    product_data.validate_layer_paths()


def run_dist_s1_sas_prep_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    post_date_buffer_days: int = 1,
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    memory_strategy: str = 'high',
    moderate_confidence_threshold: float = 3.5,
    high_confidence_threshold: float = 5.5,
    tqdm_enabled: bool = True,
    apply_water_mask: bool = True,
    n_lookbacks: int = 3,
    water_mask_path: str | Path | None = None,
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    device: str = 'best',
    batch_size_for_despeckling: int = 25,
    n_workers_for_norm_param_estimation: int = 1,
) -> RunConfigData:
    run_config = run_dist_s1_localization_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        apply_water_mask=apply_water_mask,
        water_mask_path=water_mask_path,
    )
    run_config.memory_strategy = memory_strategy
    run_config.tqdm_enabled = tqdm_enabled
    run_config.apply_water_mask = apply_water_mask
    run_config.moderate_confidence_threshold = moderate_confidence_threshold
    run_config.high_confidence_threshold = high_confidence_threshold
    run_config.n_lookbacks = n_lookbacks
    run_config.water_mask_path = water_mask_path
    run_config.product_dst_dir = product_dst_dir
    run_config.bucket = bucket
    run_config.bucket_prefix = bucket_prefix
    run_config.n_workers_for_despeckling = n_workers_for_despeckling
    run_config.batch_size_for_despeckling = batch_size_for_despeckling
    run_config.n_workers_for_norm_param_estimation = n_workers_for_norm_param_estimation
    run_config.device = device
    return run_config


def run_dist_s1_sas_workflow(run_config: RunConfigData) -> Path:
    _ = run_dist_s1_processing_workflow(run_config)
    _ = run_dist_s1_packaging_workflow(run_config)

    # Upload to S3 if bucket is provided
    if run_config.bucket is not None:
        upload_product_to_s3(run_config.product_directory, run_config.bucket, run_config.bucket_prefix)
    return run_config


def run_dist_s1_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    post_date_buffer_days: int = 1,
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    memory_strategy: str = 'high',
    moderate_confidence_threshold: float = 3.5,
    high_confidence_threshold: float = 5.5,
    water_mask_path: str | Path | None = None,
    tqdm_enabled: bool = True,
    apply_water_mask: bool = True,
    n_lookbacks: int = 3,
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    batch_size_for_despeckling: int = 25,
    n_workers_for_norm_param_estimation: int = 1,
    device: str = 'best',
) -> Path:
    run_config = run_dist_s1_sas_prep_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        memory_strategy=memory_strategy,
        moderate_confidence_threshold=moderate_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        tqdm_enabled=tqdm_enabled,
        apply_water_mask=apply_water_mask,
        n_lookbacks=n_lookbacks,
        water_mask_path=water_mask_path,
        product_dst_dir=product_dst_dir,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        n_workers_for_despeckling=n_workers_for_despeckling,
        batch_size_for_despeckling=batch_size_for_despeckling,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
        device=device,
    )
    _ = run_dist_s1_sas_workflow(run_config)

    return run_config
