import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import click
from distmetrics.model_load import ALLOWED_MODELS

from dist_s1.confirmation import confirm_disturbance_with_prior_product_and_serialize
from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_CONFIDENCE_UPPER_LIM,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_DELTA_WINDOW_DAYS,
    DEFAULT_DEVICE,
    DEFAULT_DST_DIR,
    DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_MEMORY_STRATEGY,
    DEFAULT_METRIC_VALUE_UPPER_LIM,
    DEFAULT_MODEL_COMPILATION,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_NO_COUNT_RESET_THRESH,
    DEFAULT_NO_DAY_LIMIT,
    DEFAULT_N_ANNIVERSARIES_FOR_MW,
    DEFAULT_N_WORKERS_FOR_DESPECKLING,
    DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_PERCENT_RESET_THRESH,
    DEFAULT_POST_DATE_BUFFER_DAYS,
    DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_TQDM_ENABLED,
    DEFAULT_USE_DATE_ENCODING,
)
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.workflows import (
    run_dist_s1_sas_prep_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
    run_sequential_confirmation_of_dist_products_workflow,
)


P = ParamSpec('P')  # Captures all parameter types
R = TypeVar('R')  # Captures the return type


def parse_int_list_with_none(ctx: click.Context, param: click.Parameter, value: str) -> tuple[int, ...]:
    if isinstance(value, str):
        if value == 'none':
            return None
        else:
            try:
                return tuple(int(x.strip()) for x in value.split(','))
            except Exception:
                raise click.BadParameter(
                    f'Invalid tuple format: {value}. Expected comma-separated integers (e.g., 4,4,2).'
                )
    else:
        raise click.BadParameter('Expected a string of comma-separated integers (e.g., 4,4,2) or "none".')


@click.group()
def cli() -> None:
    """CLI for dist-s1 workflows."""
    pass


def common_algo_options_for_confirmation_workflows(func: Callable) -> Callable:
    @click.option(
        '--no_day_limit',
        type=int,
        required=False,
        default=DEFAULT_NO_DAY_LIMIT,
        help='No day limit in the confirmation logic - logic is constrained to this number.',
    )
    @click.option(
        '--exclude_consecutive_no_dist',
        type=bool,
        required=False,
        default=DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
        help='Whether to exclude consecutive no disturbance.',
    )
    @click.option(
        '--percent_reset_thresh',
        type=int,
        required=False,
        default=DEFAULT_PERCENT_RESET_THRESH,
        help='Percent reset threshold.',
    )
    @click.option(
        '--no_count_reset_thresh',
        type=int,
        required=False,
        default=DEFAULT_NO_COUNT_RESET_THRESH,
        help='No count reset threshold.',
    )
    @click.option(
        '--confidence_upper_lim',
        type=int,
        required=False,
        default=DEFAULT_CONFIRMATION_CONFIDENCE_UPPER_LIM,
        help='Confidence upper limit.',
    )
    @click.option(
        '--confirmation_confidence_threshold',
        type=float,
        required=False,
        default=DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
        help='Confidence threshold.',
    )
    @click.option(
        '--metric_value_upper_lim',
        type=float,
        required=False,
        default=DEFAULT_METRIC_VALUE_UPPER_LIM,
        help='Metric value upper limit.',
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper


def common_options_for_dist_workflows(func: Callable) -> Callable:
    @click.option('--mgrs_tile_id', type=str, required=True, help='MGRS tile ID.')
    @click.option('--post_date', type=str, required=True, help='Post acquisition date.')
    @click.option(
        '--track_number',
        type=int,
        required=True,
        help='Sentinel-1 Track Number; Supply one from the group of bursts collected from a pass; '
        'Near the dateline you may have two sequential track numbers.',
    )
    @click.option(
        '--post_date_buffer_days',
        type=int,
        default=DEFAULT_POST_DATE_BUFFER_DAYS,
        required=False,
        help='Buffer days around post-date.',
    )
    @click.option(
        '--dst_dir',
        type=str,
        default=str(DEFAULT_DST_DIR),
        required=False,
        help='Path to intermediate data products; this will also be where the final products are stored if '
        'product_dst_dir is not provided.',
    )
    @click.option(
        '--memory_strategy',
        type=click.Choice(['high', 'low']),
        required=False,
        default=DEFAULT_MEMORY_STRATEGY,
        help='Memory strategy to use for GPU inference. Options: high, low.',
    )
    @click.option(
        '--low_confidence_alert_threshold',
        type=float,
        required=False,
        default=DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
        help='Low confidence alert threshold.',
    )
    @click.option(
        '--high_confidence_alert_threshold',
        type=float,
        required=False,
        default=DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
        help='High confidence alert threshold.',
    )
    @click.option('--tqdm_enabled', type=bool, required=False, default=DEFAULT_TQDM_ENABLED, help='Enable tqdm.')
    @click.option(
        '--input_data_dir',
        type=str,
        default=DEFAULT_INPUT_DATA_DIR,
        required=False,
        help='Input data directory. If None, uses `dst_dir`. Default None.',
    )
    @click.option(
        '--src_water_mask_path',
        type=str,
        default=None,
        required=False,
        help='Path to water mask file.',
    )
    @click.option(
        '--apply_water_mask',
        type=bool,
        default=DEFAULT_APPLY_WATER_MASK,
        required=False,
        help='Apply water mask to the data.',
    )
    @click.option(
        '--lookback_strategy',
        type=click.Choice(['multi_window', 'immediate_lookback']),
        required=False,
        default=DEFAULT_LOOKBACK_STRATEGY,
        help='Options to use for lookback strategy.',
    )
    @click.option(
        '--n_anniversaries_for_mw',
        type=int,
        required=False,
        default=DEFAULT_N_ANNIVERSARIES_FOR_MW,
        help='Number of anniversaries to use for multi-window lookback strategy.',
    )
    @click.option(
        '--delta_window_days',
        type=int,
        required=False,
        default=DEFAULT_DELTA_WINDOW_DAYS,
        help='The acceptable window of time to search for pre-image RTC-S1 data. Default is 60 days (or 2 months). '
        'This amounts to roughly `post_date - lookback_days - delta_window_days` to `post_date - lookback_days`. '
        'If lookback strategy is \'multi_window\', this means the maximum window of time to search for pre-images on each '
        'anniversary date where `post_date - n * lookback_days` are the anniversary dates for n = 1,...',
    )
    @click.option(
        '--max_pre_imgs_per_burst_mw',
        default=','.join(map(str, DEFAULT_MAX_PRE_IMGS_PER_BURST_MW))
        if DEFAULT_MAX_PRE_IMGS_PER_BURST_MW is not None
        else 'none',
        callback=parse_int_list_with_none,
        required=False,
        show_default=True,
        type=str,
        help='Comma-separated list of integers (e.g., --max_pre_imgs_per_burst_mw 4,4,2) or "none" to use defaults '
        'calculated from model context length and n_anniversaries_for_mw.',
    )
    @click.option(
        '--delta_lookback_days_mw',
        default=','.join(map(str, DEFAULT_DELTA_LOOKBACK_DAYS_MW))
        if DEFAULT_DELTA_LOOKBACK_DAYS_MW is not None
        else 'none',
        callback=parse_int_list_with_none,
        required=False,
        show_default=True,
        help='Comma-separated list of integers (e.g., --delta_lookback_days_mw 730,365,0) or "none" to use defaults '
        'calculated from n_anniversaries_for_mw and model context length. Provide list values in order of older to '
        'recent lookback days.',
    )
    @click.option(
        '--product_dst_dir',
        type=str,
        default=None,
        required=False,
        help='Path to save the final products. If not specified, uses `dst_dir`.',
    )
    @click.option(
        '--bucket',
        type=str,
        default=None,
        required=False,
        help='S3 bucket to upload the final products to.',
    )
    @click.option(
        '--n_workers_for_despeckling',
        type=int,
        default=DEFAULT_N_WORKERS_FOR_DESPECKLING,
        required=False,
        help='N CPUs to use for despeckling the bursts',
    )
    @click.option(
        '--bucket_prefix',
        type=str,
        default=None,
        required=False,
        help='S3 bucket prefix to upload the final products to.',
    )
    @click.option(
        '--device',
        type=click.Choice(['cpu', 'cuda', 'mps', 'best']),
        required=False,
        default=DEFAULT_DEVICE,
        help='Device to use for transformer model inference of normal parameters.',
    )
    @click.option(
        '--n_workers_for_norm_param_estimation',
        type=int,
        default=DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
        required=False,
        help='Number of CPUs to use for normal parameter estimation; error will be thrown if GPU is available and not'
        ' or set to something other than CPU.',
    )
    @click.option(
        '--model_source',
        type=click.Choice(['external'] + ALLOWED_MODELS),
        default=DEFAULT_MODEL_SOURCE,
        required=False,
        help='What model to load; external means load model from cfg and wts paths specified in parameters;'
        'see distmetrics.model_load.ALLOWED_MODELS for available models.',
    )
    @click.option(
        '--model_cfg_path',
        type=str,
        default=None,
        required=False,
        help='Path to Transformer model config file.',
    )
    @click.option(
        '--model_wts_path',
        type=str,
        default=None,
        required=False,
        help='Path to Transformer model weights file.',
    )
    @click.option(
        '--stride_for_norm_param_estimation',
        type=int,
        default=DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
        required=False,
        help='Batch size for norm param. Number of pixels the'
        ' convolutional filter moves across the input image at'
        ' each step.',
    )
    @click.option(
        '--batch_size_for_norm_param_estimation',
        type=int,
        default=DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
        required=False,
        help='Batch size for norm param estimation; Tune it according to resouces i.e. memory.',
    )
    @click.option(
        '--model_compilation',
        type=bool,
        default=DEFAULT_MODEL_COMPILATION,
        required=False,
        help='Flag to enable compilation duringe execution.',
    )
    @click.option(
        '--algo_config_path',
        type=str,
        default=None,
        required=False,
        help='Path to external algorithm configuration YAML file.',
    )
    @click.option(
        '--model_dtype',
        type=click.Choice(['float32', 'bfloat16', 'float16']),
        required=False,
        default=DEFAULT_MODEL_DTYPE,
        help='Data type for model inference. Options: float32, bfloat16, float16.',
    )
    @click.option(
        '--use_date_encoding',
        type=bool,
        default=DEFAULT_USE_DATE_ENCODING,
        required=False,
        help='Whether to use acquisition date encoding in processing.',
    )
    @click.option(
        '--prior_dist_s1_product',
        type=str,
        required=False,
        default=None,
        help='Path to prior DIST-S1 product. If provided, will be used for confirmation.',
    )
    @click.option(
        '--model_context_length',
        type=int,
        required=False,
        default=-1,
        help='Model context length. If not provided or set to -1, will be calculated using max allowed by model source '
        '(though not more than 20).',
    )
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper


# SAS Prep Workflow (No Internet Access)
@cli.command(
    name='run_sas_prep', help='Run SAS prep workflow to generate runconfig file and localize input OPERA RTC data.'
)
@common_options_for_dist_workflows
@common_algo_options_for_confirmation_workflows
@click.option(
    '--run_config_path',
    type=str,
    default=None,
    required=False,
    help='Path to yaml runconfig file that will be created. If not provided, no file will be created.',
)
def run_sas_prep(
    mgrs_tile_id: str,
    post_date: str,
    track_number: int,
    post_date_buffer_days: int,
    apply_water_mask: bool,
    memory_strategy: str,
    low_confidence_alert_threshold: float,
    high_confidence_alert_threshold: float,
    tqdm_enabled: bool,
    input_data_dir: str | Path | None,
    run_config_path: str | Path,
    lookback_strategy: str,
    delta_lookback_days_mw: tuple[int, ...],
    max_pre_imgs_per_burst_mw: tuple[int, ...],
    dst_dir: str | Path,
    src_water_mask_path: str | Path | None,
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int,
    batch_size_for_norm_param_estimation: int,
    model_compilation: bool,
    algo_config_path: str | Path | None,
    prior_dist_s1_product: str | Path | None,
    model_dtype: str,
    use_date_encoding: bool,
    n_anniversaries_for_mw: int,
    delta_window_days: int,
    no_day_limit: int,
    exclude_consecutive_no_dist: bool,
    percent_reset_thresh: int,
    no_count_reset_thresh: int,
    confidence_upper_lim: int,
    confirmation_confidence_threshold: float | None,
    metric_value_upper_lim: float,
    model_context_length: int,
) -> None:
    """Run SAS prep workflow."""
    if model_context_length == -1:
        model_context_length = None
    run_dist_s1_sas_prep_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        apply_water_mask=apply_water_mask,
        memory_strategy=memory_strategy,
        low_confidence_alert_threshold=low_confidence_alert_threshold,
        high_confidence_alert_threshold=high_confidence_alert_threshold,
        tqdm_enabled=tqdm_enabled,
        input_data_dir=input_data_dir,
        dst_dir=dst_dir,
        src_water_mask_path=src_water_mask_path,
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
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
        algo_config_path=algo_config_path,
        prior_dist_s1_product=prior_dist_s1_product,
        run_config_path=run_config_path,
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
        n_anniversaries_for_mw=n_anniversaries_for_mw,
        delta_window_days=delta_window_days,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        confirmation_confidence_upper_lim=confidence_upper_lim,
        confirmation_confidence_threshold=confirmation_confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
    )


# SAS Workflow (No Internet Access)
@cli.command(name='run_sas')
@click.option('--run_config_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
def run_sas(run_config_path: str | Path) -> None:
    """Run SAS workflow."""
    run_config = RunConfigData.from_yaml(run_config_path)
    run_dist_s1_sas_workflow(run_config)


@cli.command(name='run_one_confirmation', help='Run one confirmation of a single pair of DIST-S1 products.')
@click.option('--prior_dist_s1_product', type=str, required=True, help='Path to prior DIST-S1 product.')
@click.option(
    '--dst_dist_product_parent', type=str, required=True, help='Path to parent directory for new DIST-S1 product.'
)
@click.option(
    '--current_dist_s1_product',
    type=str,
    required=True,
    help='Path to current DIST-S1 product. Confirmed product inherits name from this product.',
)
@common_algo_options_for_confirmation_workflows
def run_one_confirmation(
    prior_dist_s1_product: str | Path,
    current_dist_s1_product: str | Path,
    dst_dist_product_parent: str | Path | None,
    no_day_limit: int,
    exclude_consecutive_no_dist: bool,
    percent_reset_thresh: int,
    no_count_reset_thresh: int,
    confidence_upper_lim: int,
    confirmation_confidence_threshold: float | None,
    metric_value_upper_lim: float,
) -> None:
    confirm_disturbance_with_prior_product_and_serialize(
        prior_dist_s1_product=prior_dist_s1_product,
        current_dist_s1_product=current_dist_s1_product,
        dst_dist_product_parent=dst_dist_product_parent,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        confirmation_confidence_upper_lim=confidence_upper_lim,
        confirmation_confidence_thresh=confirmation_confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
    )


@cli.command(
    name='run_sequential_confirmation',
    help='Run sequential confirmation of unconfirmed DIST-S1 products. Confirms products in order of oldest to newest.',
)
@click.option(
    '--unconfirmed_dist_s1_product_dir',
    type=str,
    required=True,
    help='Directory of OPERA products that are unconfirmed',
)
@click.option(
    '--dst_dist_product_parent', type=str, required=True, help='Path to parent directory for new DIST-S1 product.'
)
@common_algo_options_for_confirmation_workflows
def run_sequential_confirmation(
    unconfirmed_dist_s1_product_dir: str | Path,
    dst_dist_product_parent: str | Path | None,
    no_day_limit: int,
    exclude_consecutive_no_dist: bool,
    percent_reset_thresh: int,
    no_count_reset_thresh: int,
    confidence_upper_lim: int,
    confirmation_confidence_threshold: float | None,
    metric_value_upper_lim: float,
) -> None:
    run_sequential_confirmation_of_dist_products_workflow(
        dist_s1_data=unconfirmed_dist_s1_product_dir,
        dst_dist_product_parent=dst_dist_product_parent,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        confirmation_confidence_upper_lim=confidence_upper_lim,
        confirmation_confidence_thresh=confirmation_confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
    )


# Effectively runs the two workflows above in sequence
@cli.command(name='run', help='Run complete DIST-S1 workflow.')
@common_options_for_dist_workflows
@common_algo_options_for_confirmation_workflows
@click.option(
    '--run_config_path',
    type=str,
    default=None,
    required=False,
    help='Path to yaml runconfig file that will be created. If not provided, no file will be created.',
)
def run(
    mgrs_tile_id: str,
    post_date: str,
    track_number: int,
    post_date_buffer_days: int,
    memory_strategy: str,
    dst_dir: str | Path,
    low_confidence_alert_threshold: float,
    high_confidence_alert_threshold: float,
    tqdm_enabled: bool,
    input_data_dir: str | Path | None,
    src_water_mask_path: str | Path | None,
    apply_water_mask: bool,
    lookback_strategy: str,
    delta_lookback_days_mw: tuple[int, ...],
    max_pre_imgs_per_burst_mw: tuple[int, ...],
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int,
    batch_size_for_norm_param_estimation: int,
    model_compilation: bool,
    algo_config_path: str | Path | None,
    model_dtype: str,
    use_date_encoding: bool,
    n_anniversaries_for_mw: int,
    delta_window_days: int,
    run_config_path: str | Path | None,
    prior_dist_s1_product: str | Path | None,
    no_day_limit: int,
    exclude_consecutive_no_dist: bool,
    percent_reset_thresh: int,
    no_count_reset_thresh: int,
    confidence_upper_lim: int,
    confirmation_confidence_threshold: float | None,
    metric_value_upper_lim: float,
    model_context_length: int,
) -> str:
    """Localize data and run dist_s1_workflow."""
    if model_context_length == -1:
        model_context_length = None
    return run_dist_s1_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        apply_water_mask=apply_water_mask,
        memory_strategy=memory_strategy,
        low_confidence_alert_threshold=low_confidence_alert_threshold,
        high_confidence_alert_threshold=high_confidence_alert_threshold,
        tqdm_enabled=tqdm_enabled,
        input_data_dir=input_data_dir,
        dst_dir=dst_dir,
        src_water_mask_path=src_water_mask_path,
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
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
        algo_config_path=algo_config_path,
        n_anniversaries_for_mw=n_anniversaries_for_mw,
        delta_window_days=delta_window_days,
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
        run_config_path=run_config_path,
        prior_dist_s1_product=prior_dist_s1_product,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        confidence_upper_lim=confidence_upper_lim,
        confirmation_confidence_threshold=confirmation_confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
        model_context_length=model_context_length,
    )


@cli.command(name='check_equality', help='Check equality of two DIST-S1 products.')
@click.argument('dist-s1-product-0', type=click.Path(exists=True, file_okay=False))
@click.argument('dist-s1-product-1', type=click.Path(exists=True, file_okay=False))
def check_equality(dist_s1_product_0: str | Path, dist_s1_product_1: str | Path) -> None:
    product_0 = DistS1ProductDirectory.from_product_path(dist_s1_product_0)
    product_1 = DistS1ProductDirectory.from_product_path(dist_s1_product_1)
    if product_0 != product_1:
        print('Products are  NOT equal')
        sys.exit(1)
    else:
        print('Products are equal')
        sys.exit(0)


if __name__ == '__main__':
    cli()
