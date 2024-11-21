from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_main(cli_runner, change_local_dir, test_dir, test_data_dir, tmpdir_factory):
    change_local_dir(test_dir)

    # Read the template runconfig file and overwrite the dst_dir and the dist_s1_alert_db_dir fields to a tmpdir
    tmp_dst_dir = tmpdir_factory.mktemp('tmpdir_dst_dir')
    tmp_db_dir = tmpdir_factory.mktemp('tmpdir_db_dir')
    template_runconfig_yml_path = str(test_data_dir / '10SGD_cropped' / 'runconfig.yml')
    runconfig_data = RunConfigData.from_yaml(
        template_runconfig_yml_path,
        fields_to_overwrite={'dst_dir': str(tmp_dst_dir), 'dist_s1_alert_db_dir': str(tmp_db_dir)},
    )

    # Write the updated runconfig file to a product directory
    tmp_runconfig_yml_path = tmp_dst_dir / 'runconfig.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path)

    # Run the command using the updated runconfig file (the tmp files are cleaned up after the test)
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--runconfig_yml_path', tmp_runconfig_yml_path],
    )
    assert result.exit_code == 0
    assert runconfig_data.product_dir_data.path.exists()
    assert runconfig_data.product_dir_data.path.is_dir()
    assert runconfig_data.product_dir_data.validate_layer_paths()
    assert runconfig_data.product_dir_data.validate_layer_dtypes()
