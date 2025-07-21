import functools
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import click
from distmetrics.model_load import ALLOWED_MODELS

from .data_models.defaults import (
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_DEVICE,
    DEFAULT_DST_DIR_STR,
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_MEMORY_STRATEGY,
    DEFAULT_MODEL_COMPILATION,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODERATE_CONFIDENCE_THRESHOLD,
    DEFAULT_N_WORKERS_FOR_DESPECKLING,
    DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_POST_DATE_BUFFER_DAYS,
    DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_TQDM_ENABLED,
    DEFAULT_USE_DATE_ENCODING,
)
from .data_models.runconfig_model import RunConfigData
from .workflows import run_dist_s1_sas_prep_workflow, run_dist_s1_sas_workflow, run_dist_s1_workflow


P = ParamSpec('P')  # Captures all parameter types
R = TypeVar('R')  # Captures the return type


def parse_int_list(ctx: click.Context, param: click.Parameter, value: str) -> list[int]:
    try:
        return [int(x.strip()) for x in value.split(',')]
    except Exception:
        raise click.BadParameter(f'Invalid list format: {value}. Expected comma-separated integers (e.g., 4,4,2).')


@click.group()
def cli() -> None:
    """CLI for dist-s1 workflows."""
    pass


def common_options(func: Callable) -> Callable:
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
        default=DEFAULT_DST_DIR_STR,
        required=False,
        help='Path to intermediate data products',
    )
    @click.option(
        '--memory_strategy',
        type=click.Choice(['high', 'low']),
        required=False,
        default=DEFAULT_MEMORY_STRATEGY,
        help='Memory strategy to use for GPU inference. Options: high, low.',
    )
    @click.option(
        '--moderate_confidence_threshold',
        type=float,
        required=False,
        default=DEFAULT_MODERATE_CONFIDENCE_THRESHOLD,
        help='Moderate confidence threshold.',
    )
    @click.option(
        '--high_confidence_threshold',
        type=float,
        required=False,
        default=DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
        help='High confidence threshold.',
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
        '--water_mask_path',
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
        '--max_pre_imgs_per_burst_mw',
        default=','.join(map(str, DEFAULT_MAX_PRE_IMGS_PER_BURST_MW)),
        callback=parse_int_list,
        required=False,
        show_default=True,
        help='Comma-separated list of integers (e.g., --max_pre_imgs_per_burst_mw 4,4,2).',
    )
    @click.option(
        '--delta_lookback_days_mw',
        default=','.join(map(str, DEFAULT_DELTA_LOOKBACK_DAYS_MW)),
        callback=parse_int_list,
        required=False,
        show_default=True,
        help='Comma-separated list of integers (e.g., --delta_lookback_days_mw 730,365,0). '
        'Provide list values in order of older to recent lookback days.',
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
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapper


# Load parameter as list of integers
@cli.command()
@common_options
def parse_pre_imgs_per_burst_mw(max_pre_imgs_per_burst_mw: list[int], **kwargs: dict[str, object]) -> None:
    print('Parsed list:', max_pre_imgs_per_burst_mw)


@cli.command()
@common_options
def parse_delta_lookback_days_mw(delta_lookback_days_mw: list[int], **kwargs: dict[str, object]) -> None:
    print('Parsed list:', delta_lookback_days_mw)


# SAS Prep Workflow (No Internet Access)
@cli.command(name='run_sas_prep')
@click.option(
    '--run_config_path',
    type=str,
    default=None,
    required=False,
    help='Path to yaml runconfig file that will be created. If not provided, no file will be created.',
)
@click.option(
    '--algo_config_path',
    type=str,
    default=None,
    required=False,
    help=(
        'Path to save algorithm parameters to a separate yml file. '
        'If provided, the main config will reference this file.'
    ),
)
@common_options
def run_sas_prep(
    mgrs_tile_id: str,
    post_date: str,
    track_number: int,
    post_date_buffer_days: int,
    apply_water_mask: bool,
    memory_strategy: str,
    moderate_confidence_threshold: float,
    high_confidence_threshold: float,
    tqdm_enabled: bool,
    input_data_dir: str | Path | None,
    run_config_path: str | Path,
    lookback_strategy: str,
    delta_lookback_days_mw: list[int],
    max_pre_imgs_per_burst_mw: list[int],
    dst_dir: str | Path,
    water_mask_path: str | Path | None,
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str | None,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    model_compilation: bool = False,
    algo_config_path: str | Path | None = None,
    model_dtype: str = 'float32',
    use_date_encoding: bool = False,
) -> None:
    """Run SAS prep workflow."""
    run_dist_s1_sas_prep_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        apply_water_mask=apply_water_mask,
        memory_strategy=memory_strategy,
        moderate_confidence_threshold=moderate_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        tqdm_enabled=tqdm_enabled,
        input_data_dir=input_data_dir,
        dst_dir=dst_dir,
        water_mask_path=water_mask_path,
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
        run_config_path=run_config_path,
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
    )


# SAS Workflow (No Internet Access)
@cli.command(name='run_sas')
@click.option('--run_config_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
def run_sas(run_config_path: str | Path, algo_config_path: str | Path | None = None) -> None:
    """Run SAS workflow."""
    run_config = RunConfigData.from_yaml(run_config_path)
    run_dist_s1_sas_workflow(run_config)


# Effectively runs the two workflows above in sequence
@cli.command(name='run')
@common_options
def run(
    mgrs_tile_id: str,
    post_date: str,
    track_number: int,
    post_date_buffer_days: int,
    memory_strategy: str,
    dst_dir: str | Path,
    moderate_confidence_threshold: float,
    high_confidence_threshold: float,
    tqdm_enabled: bool,
    input_data_dir: str | Path | None,
    water_mask_path: str | Path | None,
    apply_water_mask: bool,
    lookback_strategy: str,
    delta_lookback_days_mw: list[int],
    max_pre_imgs_per_burst_mw: list[int],
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str | None,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    model_compilation: bool = False,
    algo_config_path: str | Path | None = None,
    model_dtype: str = 'float32',
    use_date_encoding: bool = False,
) -> str:
    """Localize data and run dist_s1_workflow."""
    return run_dist_s1_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        apply_water_mask=apply_water_mask,
        memory_strategy=memory_strategy,
        moderate_confidence_threshold=moderate_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        tqdm_enabled=tqdm_enabled,
        input_data_dir=input_data_dir,
        dst_dir=dst_dir,
        water_mask_path=water_mask_path,
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
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
    )


if __name__ == '__main__':
    cli()
