import functools
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import click

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
        required=False,
        default=1,
        help='Sentinel-1 Track Number; Supply one from the group of bursts collected from a pass; '
        'Near the dateline you may have two sequential track numbers.',
    )
    @click.option('--post_date_buffer_days', type=int, default=1, required=False, help='Buffer days around post-date.')
    @click.option(
        '--dst_dir',
        type=str,
        default='out/',
        required=False,
        help='Path to intermediate data products',
    )
    @click.option(
        '--memory_strategy',
        type=click.Choice(['high', 'low']),
        required=False,
        default='high',
        help='Memory strategy to use for GPU inference. Options: high, low.',
    )
    @click.option(
        '--moderate_confidence_threshold',
        type=float,
        required=False,
        default=3.5,
        help='Moderate confidence threshold.',
    )
    @click.option(
        '--high_confidence_threshold', type=float, required=False, default=5.5, help='High confidence threshold.'
    )
    @click.option('--tqdm_enabled', type=bool, required=False, default=True, help='Enable tqdm.')
    @click.option(
        '--input_data_dir',
        type=str,
        default=None,
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
        default=True,
        required=False,
        help='Apply water mask to the data.',
    )
    @click.option(
        '--n_lookbacks',
        type=int,
        default=3,
        required=False,
        help='Number of lookbacks to use for change confirmation within SAS. Use value 1, to avoid SAS confirmation.',
    )
    @click.option(
        '--lookback_strategy',
        type=click.Choice(['multi_window', 'immediate_lookback']),
        required=False,
        default='immediate_lookback',
        help='Options to use for lookback strategy.',
    )
    @click.option(
        '--max_pre_imgs_per_burst_mw',
        default='5,5',
        callback=parse_int_list,
        required=False,
        show_default=True,
        help='Comma-separated list of integers (e.g., --max_pre_imgs_per_burst_mw 4,4,2).',
    )
    @click.option(
        '--delta_lookback_days_mw',
        default='730,365',
        callback=parse_int_list,
        required=False,
        show_default=True,
        help='Comma-separated list of integers (e.g., --delta_lookback_days_mw 730,365,0). '
        'Provide list values in order of older to recent lookback days.',
    )
    @click.option(
        '--confirmation_strategy',
        type=click.Choice(['compute_baseline', 'use_prev_product']),
        required=False,
        default='compute_baseline',
        help='Options to use for confirmation strategy.',
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
        default=8,
        required=False,
        help='N CPUs to use for despeckling the bursts',
    )
    @click.option(
        '--bucket_prefix',
        type=str,
        default='',
        required=False,
        help='S3 bucket prefix to upload the final products to.',
    )
    @click.option(
        '--device',
        type=click.Choice(['cpu', 'cuda', 'mps', 'best']),
        required=False,
        default='best',
        help='Device to use for transformer model inference of normal parameters.',
    )
    @click.option(
        '--batch_size_for_despeckling',
        type=int,
        default=25,
        required=False,
        help='Batch size for despeckling the bursts; i.e. how many arrays are loaded into CPU memory at once.',
    )
    @click.option(
        '--n_workers_for_norm_param_estimation',
        type=int,
        default=8,
        required=False,
        help='Number of CPUs to use for normal parameter estimation; error will be thrown if GPU is available and not'
        ' or set to something other than CPU.',
    )
    @click.option(
        '--model_source',
        type=click.Choice(['internal', 'external']),
        required=False,
        help='Where to load Transformer model from;'
        ' internal means load model stored inside docker image,'
        ' external means load model from cfg'
        ' and wts paths specified in parameters',
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
        default=16,
        required=False,
        help='Batch size for norm param. Number of pixels the'
        ' convolutional filter moves across the input image at'
        ' each step.',
    )
    @click.option(
        '--batch_size_for_norm_param_estimation',
        type=int,
        default=32,
        required=False,
        help='Batch size for norm param estimation; Tune it according to resouces i.e. memory.',
    )
    @click.option(
        '--optimize',
        type=bool,
        default=True,
        required=False,
        help='Flag to enable compilation duringe execution.',
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
    '--runconfig_path',
    type=str,
    default='run_config.yml',
    required=False,
    help='Path to yaml runconfig file that will be created.',
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
    runconfig_path: str | Path,
    n_lookbacks: int,
    lookback_strategy: str,
    delta_lookback_days_mw: list[int],
    max_pre_imgs_per_burst_mw: list[int],
    confirmation_strategy: str,
    dst_dir: str | Path,
    water_mask_path: str | Path | None,
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    batch_size_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str | None,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    optimize: bool = True,
) -> None:
    """Run SAS prep workflow."""
    run_config = run_dist_s1_sas_prep_workflow(
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
        n_lookbacks=n_lookbacks,
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        confirmation_strategy=confirmation_strategy,
        product_dst_dir=product_dst_dir,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        n_workers_for_despeckling=n_workers_for_despeckling,
        batch_size_for_despeckling=batch_size_for_despeckling,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
        device=device,
        model_source=model_source,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        stride_for_norm_param_estimation=stride_for_norm_param_estimation,
        batch_size_for_norm_param_estimation=batch_size_for_norm_param_estimation,
        optimize=optimize,
    )
    run_config.to_yaml(runconfig_path)


# SAS Workflow (No Internet Access)
@cli.command(name='run_sas')
@click.option('--runconfig_yml_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
def run_sas(runconfig_yml_path: str | Path) -> None:
    """Run SAS workflow."""
    run_config = RunConfigData.from_yaml(runconfig_yml_path)
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
    n_lookbacks: int,
    lookback_strategy: str,
    delta_lookback_days_mw: list[int],
    max_pre_imgs_per_burst_mw: list[int],
    confirmation_strategy: str,
    product_dst_dir: str | Path | None,
    bucket: str | None,
    bucket_prefix: str,
    n_workers_for_despeckling: int,
    batch_size_for_despeckling: int,
    n_workers_for_norm_param_estimation: int,
    device: str,
    model_source: str | None,
    model_cfg_path: str | Path | None,
    model_wts_path: str | Path | None,
    stride_for_norm_param_estimation: int = 16,
    batch_size_for_norm_param_estimation: int = 32,
    optimize: bool = True,
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
        n_lookbacks=n_lookbacks,
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        confirmation_strategy=confirmation_strategy,
        product_dst_dir=product_dst_dir,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        n_workers_for_despeckling=n_workers_for_despeckling,
        batch_size_for_despeckling=batch_size_for_despeckling,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
        device=device,
        model_source=model_source,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        stride_for_norm_param_estimation=stride_for_norm_param_estimation,
        batch_size_for_norm_param_estimation=batch_size_for_norm_param_estimation,
        optimize=optimize,
    )


if __name__ == '__main__':
    cli()
