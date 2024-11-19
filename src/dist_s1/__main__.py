import click

from .dist_s1_workflow import run_dist_s1_workflow
from .input_data_model import RunConfigModel


@click.option('--yaml_file', required=True, help='Path to YAML configuration file', type=click.Path(exists=True))
@click.command()
def main(yaml_file: str):
    input_data_model = RunConfigModel.from_yaml(yaml_file)
    run_dist_s1_workflow(input_data_model)


if __name__ == '__main__':
    main()
