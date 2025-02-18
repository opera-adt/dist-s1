import shutil
from collections.abc import Callable
from pathlib import Path

from click.testing import CliRunner

from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_sas_main(
    cli_runner: CliRunner,
    change_local_dir: Callable,
    test_dir: Path,
    cropped_10SGD_dataset_runconfig: Path,
    test_opera_golden_dummy_dataset: Path,
) -> None:
    change_local_dir(test_dir)

    change_local_dir(test_dir)
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    runconfig_data = RunConfigData.from_yaml(cropped_10SGD_dataset_runconfig)
    runconfig_data.dst_dir = tmp_dir
    tmp_runconfig_yml_path = tmp_dir / 'runconfig.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path)

    # Run the command using the updated runconfig file (the tmp files are cleaned up after the test)
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--runconfig_yml_path', tmp_runconfig_yml_path],
    )
    assert result.exit_code == 0

    product_data_golden = ProductDirectoryData.from_product_path(test_opera_golden_dummy_dataset)
    out_product_data = runconfig_data.product_data_model

    assert out_product_data == product_data_golden

    shutil.rmtree(tmp_dir)
