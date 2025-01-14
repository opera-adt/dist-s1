from datetime import datetime
from pathlib import Path

import click

from .data_models.runconfig_model import RunConfigData
from .workflows import run_dist_s1_sas_workflow, run_dist_s1_workflow


def localize_data(mgrs_tile_id: str, post_date: str | datetime, track: int, post_buffer_days: int) -> dict:
    click.echo('Localizing data')
    return {'mgrs_tile_id': mgrs_tile_id, 'post_date': post_date, 'track': track, 'buffer_days': post_buffer_days}


@click.group()
def cli() -> None:
    """CLI for dist-s1 workflows."""
    pass


# SAS Workflow (No Internet Access)
@cli.command(name='run_sas')
@click.option('--runconfig_yml_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
def run_sas(runconfig_yml_path: str | Path) -> str:
    runconfig_data = RunConfigData.from_yaml(runconfig_yml_path)
    out_dir_data = run_dist_s1_sas_workflow(runconfig_data)
    click.echo(f'Writing to DIST-S1 product to directory: {out_dir_data.path}')
    return str(out_dir_data)


# MGRS Workflow with Internet Access
@cli.command(name='run')
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
@click.option('--post_date_buffer_days', type=int, required=True, help='Buffer days around post-date.')
def run(mgrs_tile_id: str, post_date: str, track_number: int, post_date_buffer_days: int) -> str:
    """Localize data and run dist_s1_workflow."""
    # Localize data
    run_config = run_dist_s1_workflow(mgrs_tile_id, post_date, track_number, post_date_buffer_days)
    return run_config


if __name__ == '__main__':
    cli()
