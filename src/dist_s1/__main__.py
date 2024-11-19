from pathlib import Path

import click

from .dist_s1_workflow import run_dist_s1_workflow
from .input_data_model import RunConfigModel


@click.option('--runconfig_yml_path', required=True, help='Path to YAML runconfig file', type=click.Path(exists=True))
@click.command()
def main(runconfig_yml_path: str | Path):
    runconfig_data = RunConfigModel.from_yaml(runconfig_yml_path)
    run_dist_s1_workflow(runconfig_data)


if __name__ == '__main__':
    main()
