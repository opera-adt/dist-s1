from pathlib import Path

from dist_s1 import run_dist_s1_sas_workflow
from dist_s1.data_models.runconfig_model import RunConfigData


def main() -> None:
    run_config_path = Path('out_1/runconfig_1.yml')
    run_config = RunConfigData.from_yaml(run_config_path)
    run_config.dst_dir = Path('test_out/')
    run_config.product_dst_dir = Path('test_product/')
    run_config.algo_config_path = None

    run_config_test_path = Path('runconfig.yml')
    run_config.to_yaml(run_config_test_path)
    # Equivalent to running:
    # dist-s1 run_sas --run_config_path runconfig_test.yml
    run_config_test = RunConfigData.from_yaml(run_config_test_path)
    breakpoint()
    run_config_test = run_dist_s1_sas_workflow(run_config_test)

    breakpoint()
    test_output_path = run_config_test.product_data_model
    golden_output_path = run_config.product_data_model

    if test_output_path != golden_output_path:
        raise ValueError('Test output data does not match golden output data')

    else:
        print('Test output path matches golden output path')


if __name__ == '__main__':
    main()
