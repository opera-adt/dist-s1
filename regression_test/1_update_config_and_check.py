import subprocess
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

    subprocess.run(['dist-s1', 'run_sas', '--run_config_path', run_config_test_path])

    # The runconfig in the command line will be an instance with different data than the one instantiated in this code.
    test_opera_products = sorted(list(run_config_test.product_dst_dir.glob('OPERA_L3_DIST-ALERT-S1_*')))
    test_product_path = test_opera_products[-1]

    golden_output_path = sorted(list(Path('golden_dataset/').glob('OPERA_L3_DIST-ALERT-S1_*')))[-1]

    golden_output_data_model = DistS1ProductDirectory.from_product_path(golden_output_path)
    test_output_data_model = DistS1ProductDirectory.from_product_path(test_product_path)

    if golden_output_data_model != test_output_data_model:
        raise ValueError('Test output data does not match golden output data')
    else:
        print('Test output path matches golden output path')


if __name__ == '__main__':
    main()
