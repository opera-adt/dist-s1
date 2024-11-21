from dist_s1.__main__ import cli as dist_s1
from dist_s1.input_data_model import RunConfigModel


def test_dist_s1_main(cli_runner, change_local_dir, test_dir, test_data_dir):
    change_local_dir(test_dir)

    # Run the command
    runconfig_yml_path = str(test_data_dir / '10SGD_cropped' / 'runconfig.yml')
    runconfig_data = RunConfigModel.from_yaml(runconfig_yml_path)
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--runconfig_yml_path', runconfig_yml_path],
    )
    assert result.exit_code == 0
    assert runconfig_data.output_product_dir.exists()
    assert runconfig_data.output_product_dir.is_dir()
    # assert 'some_path' in result.output
