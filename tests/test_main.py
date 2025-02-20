import shutil
from collections.abc import Callable
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_sas_main(
    cli_runner: CliRunner,
    change_local_dir: Callable,
    test_dir: Path,
    cropped_10SGD_dataset_runconfig: Path,
    test_opera_golden_dummy_dataset: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test the dist-s1 sas main function.

    This is identical to running from the test_directory:

    `dist-s1 run_sas --runconfig_yml_path test_data/cropped/sample_runconfig_10SGD_cropped.yml`

    And comparing the output product directory to the golden dummy dataset.
    """
    change_local_dir(test_dir)
    # Even though we change our local directory, this path needs to be made relative to the relevant test dir to avoid
    # Issues in CI/CD
    tmp_dir = test_dir / Path('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Check the runconfig.yml and see it is tmp2 - want to make sure this is correctly being set
    product_dst_dir = test_dir / Path('tmp2')

    runconfig_data = RunConfigData.from_yaml(cropped_10SGD_dataset_runconfig)
    runconfig_data.dst_dir = tmp_dir
    assert runconfig_data.product_dst_dir.resolve() == product_dst_dir.resolve()

    tmp_runconfig_yml_path = tmp_dir / 'runconfig.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path)

    # Run the command using the updated runconfig file (the tmp files are cleaned up after the test)
    change_dir_command = ['cd', str(test_dir)]
    dist_s1_command = ['run_sas', '--runconfig_yml_path', str(tmp_runconfig_yml_path)]
    command = change_dir_command + ['&&'] + dist_s1_command

    result = cli_runner.invoke(
        dist_s1,
        command,
    )
    # Check the product_dst_dir exists
    assert product_dst_dir.exists()

    assert result.exit_code == 0

    product_data_golden = ProductDirectoryData.from_product_path(test_opera_golden_dummy_dataset)
    out_product_data = runconfig_data.product_data_model

    assert out_product_data == product_data_golden

    shutil.rmtree(tmp_dir)
    shutil.rmtree(product_dst_dir)
