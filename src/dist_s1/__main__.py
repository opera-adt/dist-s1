from datetime import datetime
from pathlib import Path

import click

from .dist_s1_workflow import run_dist_s1_workflow
from .input_data_model import RunConfigModel


def localize_data(mgrs_tile_id: str, post_date: str | datetime, track: int, post_buffer_days: int):
    """Dummy function to localize data."""
    click.echo('Localizing data')
    return {'mgrs_tile_id': mgrs_tile_id, 'post_date': post_date, 'track': track, 'buffer_days': post_buffer_days}


@click.group()
def cli():
    """CLI for dist-s1 workflows."""
    pass


# SAS Workflow (No Internet Access)
@cli.command(name='run_sas')
@click.option('--runconfig_yml_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
def run_sas(runconfig_yml_path: str | Path):
    runconfig_data = RunConfigModel.from_yaml(runconfig_yml_path)
    run_dist_s1_workflow(runconfig_data)


# MGRS Workflow with Internet Access
@cli.command(name='run')
@click.option('--mgrs_tile_id', type=str, required=True, help='MGRS tile ID.')
@click.option('--post_date', type=str, required=True, help='Post acquisition date.')
@click.option('--track_number', type=int, required=True, help='Sentinel-1 Track number.')
@click.option('--post_buffer_days', type=int, required=True, help='Buffer days around post-date.')
def run(mgrs_tile_id: str, post_date: str, track_number: int, post_buffer_days: int):
    """Localize data and run dist_s1_workflow."""
    # Localize data
    _ = localize_data(mgrs_tile_id, post_date, track_number, post_buffer_days)
    # TODO: Run the workflow with localized data
    return 'output_path'


if __name__ == '__main__':
    cli()
