import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as torch_mp
from distmetrics.model_load import get_model_context_length
from tqdm.auto import tqdm


try:
    torch_mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

from dist_s1.aws import upload_product_to_s3
from dist_s1.confirmation import confirm_disturbance_with_prior_product_and_serialize
from dist_s1.data_models.data_utils import get_max_pre_imgs_per_burst_mw
from dist_s1.data_models.defaults import (
    DEFAULT_CONFIDENCE_UPPER_LIM,
    DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_METRIC_VALUE_UPPER_LIM,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_NO_COUNT_RESET_THRESH,
    DEFAULT_NO_DAY_LIMIT,
    DEFAULT_N_ANNIVERSARIES_FOR_MW,
    DEFAULT_PERCENT_RESET_THRESH,
)
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.despeckling import despeckle_and_serialize_rtc_s1
from dist_s1.dist_processing import (
    compute_burst_disturbance_and_serialize,
    merge_burst_disturbances_and_serialize,
    merge_burst_metrics_and_serialize,
)
from dist_s1.localize_rtc_s1 import localize_rtc_s1
from dist_s1.packaging import (
    generate_browse_image,
    get_product_tags,
    package_disturbance_tifs_no_confirmation,
)


@dataclass
class DistBurstProcessingArgs:
    pre_copol_paths: list[str]
    pre_crosspol_paths: list[str]
    post_copol_path: str
    post_crosspol_path: str
    acq_dts: list
    out_dist_path: str
    out_metric_path: str
    low_confidence_alert_threshold: float
    high_confidence_alert_threshold: float
    use_logits: bool
    model_compilation: bool
    model_source: str
    model_cfg_path: str | None
    model_wts_path: str | None
    memory_strategy: str
    stride_for_norm_param_estimation: int
    batch_size_for_norm_param_estimation: int
    device: str
    raw_data_for_nodata_mask: str
    model_dtype: str
    use_date_encoding: bool


def _dist_processing_one_burst_wrapper(args: DistBurstProcessingArgs) -> str:
    compute_burst_disturbance_and_serialize(
        pre_copol_paths=args.pre_copol_paths,
        pre_crosspol_paths=args.pre_crosspol_paths,
        post_copol_path=args.post_copol_path,
        post_crosspol_path=args.post_crosspol_path,
        acq_dts=args.acq_dts,
        out_dist_path=args.out_dist_path,
        out_metric_path=args.out_metric_path,
        low_confidence_alert_threshold=args.low_confidence_alert_threshold,
        high_confidence_alert_threshold=args.high_confidence_alert_threshold,
        use_logits=args.use_logits,
        model_compilation=args.model_compilation,
        model_source=args.model_source,
        model_cfg_path=args.model_cfg_path,
        model_wts_path=args.model_wts_path,
        memory_strategy=args.memory_strategy,
        stride=args.stride_for_norm_param_estimation,
        batch_size=args.batch_size_for_norm_param_estimation,
        device=args.device,
        raw_data_for_nodata_mask=args.raw_data_for_nodata_mask,
        model_dtype=args.model_dtype,
        use_date_encoding=args.use_date_encoding,
    )
    return args.out_dist_path  # Return something to track completion


def run_dist_s1_localization_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
    post_date_buffer_days: int = 1,
    max_pre_imgs_per_burst_mw: tuple[int, int] | None = DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    delta_lookback_days_mw: tuple[int, int] | None = DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    n_anniversaries_for_mw: int = DEFAULT_N_ANNIVERSARIES_FOR_MW,
    model_context_length: int = 10,
) -> RunConfigData:
    """Run the DIST-S1 localization workflow.

    This function handles data localization only. Algorithm parameter assignment
    is handled separately by the calling workflow for better separation of concerns.
    """
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
        n_anniversaries_for_mw=n_anniversaries_for_mw,
        model_context_length=model_context_length,
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
        n_workers=run_config.algo_config.n_workers_for_despeckling,
        interpolation_method=run_config.algo_config.interpolation_method,
    )


def run_burst_disturbance_workflow(run_config: RunConfigData) -> None:
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    # Collect all burst processing arguments for potential parallel processing
    burst_args_list = []

    for burst_id in df_inputs.jpl_burst_id.unique():
        df_burst_input_data = df_inputs[df_inputs.jpl_burst_id == burst_id].reset_index(drop=True)
        df_burst_input_data.sort_values(by='acq_dt', inplace=True, ascending=True)
        df_metric_burst = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)

        if run_config.algo_config.apply_despeckling:
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

        # Create BurstProcessingArgs object for this burst
        burst_args = DistBurstProcessingArgs(
            pre_copol_paths=pre_copol_paths,
            pre_crosspol_paths=pre_crosspol_paths,
            post_copol_path=post_copol_paths[0],
            post_crosspol_path=post_crosspol_paths[0],
            acq_dts=acq_dts,
            out_dist_path=output_dist_path,
            out_metric_path=output_metric_path,
            low_confidence_alert_threshold=run_config.algo_config.low_confidence_alert_threshold,
            high_confidence_alert_threshold=run_config.algo_config.high_confidence_alert_threshold,
            use_logits=run_config.algo_config.apply_logit_to_inputs,
            model_compilation=run_config.algo_config.model_compilation,
            model_source=run_config.algo_config.model_source,
            model_cfg_path=run_config.algo_config.model_cfg_path,
            model_wts_path=run_config.algo_config.model_wts_path,
            memory_strategy=run_config.algo_config.memory_strategy,
            stride_for_norm_param_estimation=run_config.algo_config.stride_for_norm_param_estimation,
            batch_size_for_norm_param_estimation=run_config.algo_config.batch_size_for_norm_param_estimation,
            device=run_config.algo_config.device,
            raw_data_for_nodata_mask=copol_post_path,
            model_dtype=run_config.algo_config.model_dtype,
            use_date_encoding=run_config.algo_config.use_date_encoding,
        )
        burst_args_list.append(burst_args)

    # Process bursts in parallel or sequentially based on configuration
    tqdm_disable = not run_config.algo_config.tqdm_enabled

    if run_config.algo_config.n_workers_for_norm_param_estimation == 1 or len(burst_args_list) == 1:
        for args in tqdm(burst_args_list, disable=tqdm_disable, desc='Burst disturbance'):
            _dist_processing_one_burst_wrapper(args)
    else:
        pool = None
        try:
            pool = torch_mp.Pool(processes=run_config.algo_config.n_workers_for_norm_param_estimation)
            list(
                tqdm(
                    pool.imap(_dist_processing_one_burst_wrapper, burst_args_list),
                    total=len(burst_args_list),
                    disable=tqdm_disable,
                    desc='Burst disturbance',
                )
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()


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


def run_confirmation_of_dist_product_workflow(
    run_config: RunConfigData,
) -> None:
    current_dist_s1_product = run_config.product_data_model_no_confirmation
    prior_dist_s1_product = run_config.prior_dist_s1_product
    dst_dist_product_parent = run_config.product_data_model.product_dir_path.parent
    no_day_limit = run_config.algo_config.no_day_limit
    exclude_consecutive_no_dist = run_config.algo_config.exclude_consecutive_no_dist
    percent_reset_thresh = run_config.algo_config.percent_reset_thresh
    no_count_reset_thresh = run_config.algo_config.no_count_reset_thresh
    confidence_upper_lim = run_config.algo_config.confidence_upper_lim
    confidence_threshold = run_config.algo_config.confirmation_confidence_threshold
    metric_value_upper_lim = run_config.algo_config.metric_value_upper_lim
    product_tags = get_product_tags(run_config)

    confirm_disturbance_with_prior_product_and_serialize(
        current_dist_s1_product=current_dist_s1_product,
        prior_dist_s1_product=prior_dist_s1_product,
        dst_dist_product_parent=dst_dist_product_parent,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        confidence_upper_lim=confidence_upper_lim,
        confidence_thresh=confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
        product_tags=product_tags,
    )
    # Generate browse image for the final product
    generate_browse_image(run_config.product_data_model, run_config.water_mask_path)


def run_sequential_confirmation_of_dist_products_workflow(
    directory_of_dist_s1_products: Path | str,
    dst_dist_product_parent: Path | str,
    no_day_limit: int = DEFAULT_NO_DAY_LIMIT,
    exclude_consecutive_no_dist: bool = DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    percent_reset_thresh: int = DEFAULT_PERCENT_RESET_THRESH,
    no_count_reset_thresh: int = DEFAULT_NO_COUNT_RESET_THRESH,
    confidence_upper_lim: int = DEFAULT_CONFIDENCE_UPPER_LIM,
    confidence_thresh: float = DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    metric_value_upper_lim: float = DEFAULT_METRIC_VALUE_UPPER_LIM,
    tqdm_enabled: bool = True,
) -> None:
    if isinstance(directory_of_dist_s1_products, str):
        directory_of_dist_s1_products = Path(directory_of_dist_s1_products)
    if isinstance(dst_dist_product_parent, str):
        dst_dist_product_parent = Path(dst_dist_product_parent)
        dst_dist_product_parent.mkdir(parents=True, exist_ok=True)

    # Sorted is important here as we assume earlier products are older
    product_dirs = sorted(list(directory_of_dist_s1_products.glob('OPERA*')))
    product_dirs = list(Path(p) for p in product_dirs)
    product_dirs = list(filter(lambda x: x.is_dir(), product_dirs))

    if len(product_dirs) == 0:
        raise ValueError(f'No product directories found in the product directory {directory_of_dist_s1_products}.')
    if len(product_dirs) == 1:
        raise ValueError(f'Only one product directory in the product directory {directory_of_dist_s1_products}.')

    for k, current_dist_s1_product in tqdm(
        enumerate(product_dirs),
        desc=f'Confirming {len(product_dirs)} products',
        total=len(product_dirs),
        disable=not tqdm_enabled,
    ):
        if k == 0:
            dst_dist_product_directory = dst_dist_product_parent / product_dirs[0].name
            shutil.copytree(product_dirs[0], dst_dist_product_directory, dirs_exist_ok=True)
            dst_dist_product_directory = DistS1ProductDirectory.from_product_path(dst_dist_product_directory)
            prior_confirmed_dist_s1_prod = dst_dist_product_directory
        else:
            dst_dist_product_directory = confirm_disturbance_with_prior_product_and_serialize(
                current_dist_s1_product=current_dist_s1_product,
                prior_dist_s1_product=prior_confirmed_dist_s1_prod,
                dst_dist_product_parent=dst_dist_product_parent,
                no_day_limit=no_day_limit,
                exclude_consecutive_no_dist=exclude_consecutive_no_dist,
                percent_reset_thresh=percent_reset_thresh,
                no_count_reset_thresh=no_count_reset_thresh,
                confidence_upper_lim=confidence_upper_lim,
                confidence_thresh=confidence_thresh,
                metric_value_upper_lim=metric_value_upper_lim,
            )
            prior_confirmed_dist_s1_prod = dst_dist_product_parent / current_dist_s1_product.name
        generate_browse_image(dst_dist_product_directory)


def run_dist_s1_processing_workflow(run_config: RunConfigData) -> RunConfigData:
    if run_config.algo_config.apply_despeckling:
        run_despeckle_workflow(run_config)

    run_burst_disturbance_workflow(run_config)

    run_disturbance_merge_workflow(run_config)

    return run_config


def run_dist_s1_packaging_workflow_no_confirmation(run_config: RunConfigData) -> Path:
    package_disturbance_tifs_no_confirmation(run_config)
    generate_browse_image(run_config.product_data_model_no_confirmation, run_config.water_mask_path)

    product_data = run_config.product_data_model_no_confirmation
    product_data.validate_layer_paths()
    product_data.validate_tif_layer_dtypes()

    return product_data.product_dir_path


def run_dist_s1_sas_prep_workflow(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    post_date_buffer_days: int = 1,
    dst_dir: str | Path = 'out',
    input_data_dir: str | Path | None = None,
    memory_strategy: str = 'high',
    low_confidence_alert_threshold: float = 3.5,
    high_confidence_alert_threshold: float = 5.5,
    tqdm_enabled: bool = True,
    apply_water_mask: bool = True,
    lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
    max_pre_imgs_per_burst_mw: tuple[int, ...] | None = None,
    delta_lookback_days_mw: tuple[int, ...] | None = None,
    water_mask_path: str | Path | None = None,
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    device: str = 'best',
    n_workers_for_norm_param_estimation: int = 1,
    model_source: str = DEFAULT_MODEL_SOURCE,
    model_cfg_path: str | Path | None = None,
    model_wts_path: str | Path | None = None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    interpolation_method: str = 'bilinear',
    apply_despeckling: bool = True,
    apply_logit_to_inputs: bool = True,
    model_compilation: bool = False,
    algo_config_path: str | Path | None = None,
    prior_dist_s1_product: str | Path | None = None,
    run_config_path: str | Path | None = None,
    model_dtype: str = 'float32',
    use_date_encoding: bool = False,
    n_anniversaries_for_mw: int = DEFAULT_N_ANNIVERSARIES_FOR_MW,
) -> RunConfigData:
    model_context_length = get_model_context_length(model_source, model_cfg_path)
    if max_pre_imgs_per_burst_mw is None:
        max_pre_imgs_per_burst_mw = get_max_pre_imgs_per_burst_mw(model_context_length, n_anniversaries_for_mw)
    if delta_lookback_days_mw is None:
        delta_lookback_days_mw = tuple(365 * n for n in range(n_anniversaries_for_mw, 0, -1))

    assert len(max_pre_imgs_per_burst_mw) == n_anniversaries_for_mw == len(delta_lookback_days_mw)

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
        n_anniversaries_for_mw=n_anniversaries_for_mw,
        model_context_length=model_context_length,
    )
    run_config.algo_config.memory_strategy = memory_strategy
    run_config.algo_config.tqdm_enabled = tqdm_enabled
    run_config.apply_water_mask = apply_water_mask
    run_config.algo_config.low_confidence_alert_threshold = low_confidence_alert_threshold
    run_config.algo_config.high_confidence_alert_threshold = high_confidence_alert_threshold
    run_config.algo_config.lookback_strategy = lookback_strategy
    run_config.water_mask_path = water_mask_path
    run_config.product_dst_dir = product_dst_dir
    run_config.bucket = bucket
    run_config.bucket_prefix = bucket_prefix
    run_config.algo_config.n_workers_for_despeckling = n_workers_for_despeckling
    run_config.algo_config.n_workers_for_norm_param_estimation = n_workers_for_norm_param_estimation
    run_config.algo_config.device = device
    run_config.algo_config.model_source = model_source
    run_config.algo_config.model_cfg_path = model_cfg_path
    run_config.algo_config.model_wts_path = model_wts_path
    run_config.algo_config.stride_for_norm_param_estimation = stride_for_norm_param_estimation
    run_config.algo_config.batch_size_for_norm_param_estimation = batch_size_for_norm_param_estimation
    run_config.algo_config.model_compilation = model_compilation
    run_config.algo_config.interpolation_method = interpolation_method
    run_config.algo_config.apply_despeckling = apply_despeckling
    run_config.algo_config.apply_logit_to_inputs = apply_logit_to_inputs
    run_config.prior_dist_s1_product = prior_dist_s1_product
    run_config.algo_config.model_dtype = model_dtype
    run_config.algo_config.use_date_encoding = use_date_encoding
    # This appears last because it will overwrite all the parameters above
    run_config.algo_config_path = algo_config_path
    if run_config_path is not None:
        run_config.to_yaml(run_config_path, algo_param_path=algo_config_path)

    return run_config


def run_dist_s1_sas_workflow(run_config: RunConfigData) -> Path:
    _ = run_dist_s1_processing_workflow(run_config)
    _ = run_dist_s1_packaging_workflow_no_confirmation(run_config)

    if run_config.confirmation:
        run_confirmation_of_dist_product_workflow(run_config)
    else:
        src = run_config.product_data_model_no_confirmation.product_dir_path
        dst = run_config.product_data_model.product_dir_path
        shutil.copytree(src, dst, dirs_exist_ok=True)

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
    low_confidence_alert_threshold: float = 3.5,
    high_confidence_alert_threshold: float = 5.5,
    water_mask_path: str | Path | None = None,
    tqdm_enabled: bool = True,
    apply_water_mask: bool = True,
    lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
    max_pre_imgs_per_burst_mw: tuple[int, int] | None = None,
    delta_lookback_days_mw: tuple[int, int] | None = None,
    product_dst_dir: str | Path | None = None,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = 5,
    n_workers_for_norm_param_estimation: int = 1,
    device: str = 'best',
    model_source: str = DEFAULT_MODEL_SOURCE,
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
    model_dtype: str = 'float32',
    use_date_encoding: bool = False,
) -> Path:
    run_config = run_dist_s1_sas_prep_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        memory_strategy=memory_strategy,
        low_confidence_alert_threshold=low_confidence_alert_threshold,
        high_confidence_alert_threshold=high_confidence_alert_threshold,
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
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
    )
    _ = run_dist_s1_sas_workflow(run_config)

    return run_config
