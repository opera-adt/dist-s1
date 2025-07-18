from datetime import datetime
from pathlib import Path

import torch.multiprocessing as torch_mp
from tqdm.auto import tqdm

from dist_s1.aws import upload_product_to_s3
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.despeckling import despeckle_and_serialize_rtc_s1
from dist_s1.dist_processing import (
    compute_burst_disturbance_and_serialize,
    compute_tile_disturbance_using_previous_product_and_serialize,
    merge_burst_disturbances_and_serialize,
    merge_burst_metrics_and_serialize,
)
from dist_s1.localize_rtc_s1 import localize_rtc_s1
from dist_s1.packaging import (
    generate_browse_image,
    package_conf_db_disturbance_tifs,
    package_disturbance_tifs_no_confirmation,
)


# Use spawn for multiprocessing
torch_mp.set_start_method('spawn', force=True)


def run_dist_s1_localization_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    lookback_strategy: str = 'multi_window',
    post_date_buffer_days: int = 1,
    max_pre_imgs_per_burst_mw: list[int] = [5, 5],
    delta_lookback_days_mw: list[int] = [730, 365],
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    apply_water_mask: bool = True,
    water_mask_path: str | Path | None = None,
    device: str = 'best',
    interpolation_method: str = 'none',
    apply_despeckling: bool = True,
    apply_logit_to_inputs: bool = True,
) -> RunConfigData:
    """Run the DIST-S1 localization workflow.

    This function handles both data localization and algorithm parameter assignment.
    It separates the core data localization from algorithm parameter configuration
    for better maintainability.
    """
    # Localize inputs - only passing essential parameters for data enumeration/localization
    run_config = localize_rtc_s1(
        mgrs_tile_id,
        post_date,
        track_number,
        lookback_strategy=lookback_strategy,
        post_date_buffer_days=post_date_buffer_days,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        # Configuration and algorithm parameters removed from localize_rtc_s1
    )

    # Assign configuration parameters after localization
    # These assignments will trigger validation due to validate_assignment=True
    run_config.apply_water_mask = apply_water_mask
    run_config.water_mask_path = water_mask_path

    # Assign algorithm parameters after localization
    # These assignments will trigger validation due to validate_assignment=True
    run_config.device = device
    run_config.interpolation_method = interpolation_method
    run_config.apply_despeckling = apply_despeckling
    run_config.apply_logit_to_inputs = apply_logit_to_inputs

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
        interpolation_method=run_config.interpolation_method,
    )


def run_burst_disturbance_workflow(run_config: RunConfigData) -> None:
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    tqdm_disable = not run_config.tqdm_enabled
    for burst_id in tqdm(df_inputs.jpl_burst_id.unique(), disable=tqdm_disable, desc='Burst disturbance'):
        df_burst_input_data = df_inputs[df_inputs.jpl_burst_id == burst_id].reset_index(drop=True)
        df_burst_input_data.sort_values(by='acq_dt', inplace=True, ascending=True)
        df_metric_burst = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)
        if run_config.apply_despeckling:
            copol_path_column = 'loc_path_copol_dspkl'
            crosspol_path_column = 'loc_path_crosspol_dspkl'
        else:
            copol_path_column = 'loc_path_copol'
            crosspol_path_column = 'loc_path_crosspol'

        df_pre = df_burst_input_data[df_burst_input_data.input_category == 'pre'].reset_index(drop=True)
        df_post = df_burst_input_data[df_burst_input_data.input_category == 'post'].reset_index(drop=True)
        pre_copol_paths = df_pre[copol_path_column].tolist()
        pre_crosspol_paths = df_pre[crosspol_path_column].tolist()
        post_copol_paths = df_post[copol_path_column].tolist()
        post_crosspol_paths = df_post[crosspol_path_column].tolist()
        acq_dts = df_pre['acq_dt'].tolist() + df_post['acq_dt'].tolist()
        # Assert the number of copol and crosspol paths are the same and are 1
        assert len(post_copol_paths) == len(post_crosspol_paths) == 1
        # Assert the number of paths is the same as the number of dates
        assert len(acq_dts) == len(pre_copol_paths) + len(post_copol_paths)
        # Assert dates are unique
        assert len(list(set(acq_dts))) == len(acq_dts)

        assert df_metric_burst.shape[0] == 1

        dist_path_l = df_metric_burst['loc_path_dist_alert_burst'].tolist()
        assert len(dist_path_l) == 1
        output_dist_path = dist_path_l[0]
        output_metric_path = df_metric_burst['loc_path_metric'].iloc[0]

        # Use the original copol post path to compute the nodata mask
        copol_post_path = df_post['loc_path_copol'].iloc[0]

        # Computes the disturbance for a a single baseline and serlialize.
        # Labels will be 0 for no disturbance, 1 for moderate confidence disturbance,
        # 2 for high confidence disturbance, and 255 for nodata
        compute_burst_disturbance_and_serialize(
            pre_copol_paths=pre_copol_paths,
            pre_crosspol_paths=pre_crosspol_paths,
            post_copol_path=post_copol_paths[0],
            post_crosspol_path=post_crosspol_paths[0],
            acq_dts=acq_dts,
            out_dist_path=output_dist_path,
            out_metric_path=output_metric_path,
            moderate_confidence_threshold=run_config.moderate_confidence_threshold,
            high_confidence_threshold=run_config.high_confidence_threshold,
            use_logits=run_config.apply_logit_to_inputs,
            model_compilation=run_config.model_compilation,
            model_source=run_config.model_source,
            model_cfg_path=run_config.model_cfg_path,
            model_wts_path=run_config.model_wts_path,
            memory_strategy=run_config.memory_strategy,
            stride=run_config.stride_for_norm_param_estimation,
            batch_size=run_config.batch_size_for_norm_param_estimation,
            device=run_config.device,
            raw_data_for_nodata_mask=copol_post_path,
        )


def run_disturbance_merge_workflow(run_config: RunConfigData) -> None:
    dst_tif_paths = run_config.final_unformatted_tif_paths

    # Metric
    metric_burst_paths = run_config.df_burst_distmetrics['loc_path_metric'].tolist()
    dst_metric_path = dst_tif_paths['metric_status_path']
    merge_burst_metrics_and_serialize(metric_burst_paths, dst_metric_path, run_config.mgrs_tile_id)

    # Disturbance
    dist_burst_paths = run_config.df_burst_distmetrics['loc_path_dist_alert_burst'].tolist()
    dst_dist_path = dst_tif_paths['alert_status_path']
    merge_burst_disturbances_and_serialize(dist_burst_paths, dst_dist_path, run_config.mgrs_tile_id)


def run_disturbance_confirmation(run_config: RunConfigData) -> None:
    print('Running disturbance confirmation')
    # Use previous DIST-S1 product to confirm the disturbance
    df_prior_dist_products = run_config.df_prior_dist_products

    if df_prior_dist_products.empty:
        print('No previous product found for confirmation. Assuming this is the first product.')
        prev_prod_paths = None
    else:
        ordered_columns = [
            'path_dist_status',
            'path_dist_max',
            'path_dist_conf',
            'path_dist_date',
            'path_dist_count',
            'path_dist_perc',
            'path_dist_dur',
            'path_dist_last_date',
        ]
        prev_prod_paths = df_prior_dist_products[ordered_columns].values.flatten().tolist()

    final_unformated_conf_tif_paths = [
        run_config.final_unformatted_tif_paths['dist_status_path'],
        run_config.final_unformatted_tif_paths['dist_max_path'],
        run_config.final_unformatted_tif_paths['dist_conf_path'],
        run_config.final_unformatted_tif_paths['dist_date_path'],
        run_config.final_unformatted_tif_paths['dist_count_path'],
        run_config.final_unformatted_tif_paths['dist_perc_path'],
        run_config.final_unformatted_tif_paths['dist_dur_path'],
        run_config.final_unformatted_tif_paths['dist_last_date_path'],
    ]
    out_pattern_sample = run_config.product_data_model.layer_path_dict['GEN-DIST-STATUS']
    compute_tile_disturbance_using_previous_product_and_serialize(
        dist_metric_path=run_config.final_unformatted_tif_paths['metric_status_path'],
        dist_metric_date=out_pattern_sample,
        out_path_list=final_unformated_conf_tif_paths,
        previous_dist_arr_path_list=prev_prod_paths,
    )


def run_dist_s1_processing_workflow(run_config: RunConfigData) -> RunConfigData:
    if run_config.apply_despeckling:
        run_despeckle_workflow(run_config)

    run_burst_disturbance_workflow(run_config)

    run_disturbance_merge_workflow(run_config)

    return run_config


def run_dist_s1_packaging_workflow(run_config: RunConfigData) -> Path:
    if not run_config.confirmation:
        print('No confirmation requested, skipping confirmation step')
        package_disturbance_tifs_no_confirmation(run_config)

    if run_config.confirmation:
        print('Using previous product for confirmation')
        run_disturbance_confirmation(run_config)
        package_conf_db_disturbance_tifs(run_config)

    product_data = run_config.product_data_model
    product_data.validate_conf_db_tif_layer_dtypes()
    product_data.validate_conf_db_layer_paths()

    generate_browse_image(run_config)


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
    lookback_strategy: str = 'multi_window',
    max_pre_imgs_per_burst_mw: list[int] = [5, 5],
    delta_lookback_days_mw: list[int] = [730, 365],
    water_mask_path: str | Path | None = None,
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    device: str = 'best',
    n_workers_for_norm_param_estimation: int = 1,
    model_source: str | None = None,
    model_cfg_path: str | Path | None = None,
    model_wts_path: str | Path | None = None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    interpolation_method: str = 'none',
    apply_despeckling: bool = True,
    apply_logit_to_inputs: bool = True,
    model_compilation: bool = False,
    algo_config_path: str | Path | None = None,
    prior_dist_s1_product: str | Path | None = None,
) -> RunConfigData:
    run_config = run_dist_s1_localization_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        lookback_strategy,
        post_date_buffer_days,
        max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        apply_water_mask=apply_water_mask,
        water_mask_path=water_mask_path,
        device=device,
    )
    run_config.memory_strategy = memory_strategy
    run_config.tqdm_enabled = tqdm_enabled
    run_config.apply_water_mask = apply_water_mask
    run_config.moderate_confidence_threshold = moderate_confidence_threshold
    run_config.high_confidence_threshold = high_confidence_threshold
    run_config.lookback_strategy = lookback_strategy
    run_config.water_mask_path = water_mask_path
    run_config.product_dst_dir = product_dst_dir
    run_config.bucket = bucket
    run_config.bucket_prefix = bucket_prefix
    run_config.n_workers_for_despeckling = n_workers_for_despeckling
    run_config.n_workers_for_norm_param_estimation = n_workers_for_norm_param_estimation
    run_config.device = device
    run_config.model_source = model_source
    run_config.model_cfg_path = model_cfg_path
    run_config.model_wts_path = model_wts_path
    run_config.stride_for_norm_param_estimation = stride_for_norm_param_estimation
    run_config.batch_size_for_norm_param_estimation = batch_size_for_norm_param_estimation
    run_config.model_compilation = model_compilation
    run_config.interpolation_method = interpolation_method
    run_config.apply_despeckling = apply_despeckling
    run_config.apply_logit_to_inputs = apply_logit_to_inputs
    run_config.algo_config_path = algo_config_path
    run_config.prior_dist_s1_product = prior_dist_s1_product
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
    lookback_strategy: str = 'multi_window',
    max_pre_imgs_per_burst_mw: list[int] = [5, 5],
    delta_lookback_days_mw: list[int] = [730, 365],
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    n_workers_for_norm_param_estimation: int = 1,
    device: str = 'best',
    model_source: str | None = None,
    model_cfg_path: str | Path | None = None,
    model_wts_path: str | Path | None = None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    model_compilation: bool = False,
    interpolation_method: str = 'none',
    apply_despeckling: bool = True,
    apply_logit_to_inputs: bool = True,
    algo_config_path: str | Path | None = None,
    prior_dist_s1_product: str | Path | None = None,
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
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        water_mask_path=water_mask_path,
        product_dst_dir=product_dst_dir,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        n_workers_for_despeckling=n_workers_for_despeckling,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
        device=device,
        model_source=model_source,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        stride_for_norm_param_estimation=stride_for_norm_param_estimation,
        batch_size_for_norm_param_estimation=batch_size_for_norm_param_estimation,
        model_compilation=model_compilation,
        interpolation_method=interpolation_method,
        apply_despeckling=apply_despeckling,
        apply_logit_to_inputs=apply_logit_to_inputs,
        algo_config_path=algo_config_path,
        prior_dist_s1_product=prior_dist_s1_product,
    )
    _ = run_dist_s1_sas_workflow(run_config)

    return run_config
