from pathlib import Path

from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData


def main() -> None:
    run_config_path = Path('out_1/runconfig_1.yml')
    run_config_test = RunConfigData.from_yaml(run_config_path)
    run_config_test.dst_dir = Path('test_out/')
    run_config_test.product_dst_dir = Path('test_product/')
    run_config_test_path = Path('runconfig.yml')
    run_config_test.to_yaml(run_config_test_path)


if __name__ == '__main__':
    main()
