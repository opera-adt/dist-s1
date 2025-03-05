import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_sas_main(
    cli_runner: CliRunner,
    test_dir: Path,
    change_local_dir: Callable[[Path], None],
    cropped_10SGD_dataset_runconfig: Path,
    test_opera_golden_dummy_dataset: Path,
) -> None:
    """Test the dist-s1 sas main function.

    This is identical to running from the test_directory:

    `dist-s1 run_sas --runconfig_yml_path test_data/cropped/sample_runconfig_10SGD_cropped.yml`

    And comparing the output product directory to the golden dummy dataset.

    Note: the hardest part is serializing the runconfig to yml and then correctly finding the generated product.
    This is because the product paths from the in-memory runconfig object are different from the ones created via yml.
    This is because the product paths have the *processing time* in them, and that is different depending on when the
    runconfig object is created.
    """
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    product_data_golden = ProductDirectoryData.from_product_path(test_opera_golden_dummy_dataset)

    # Load and modify runconfig - not the paths are relative to the test_dir
    runconfig_data = RunConfigData.from_yaml(cropped_10SGD_dataset_runconfig)
    # Memory strategy was set to high to create the golden dataset
    runconfig_data.memory_strategy = 'high'
    # Force CPU device
    runconfig_data.device = 'cpu'
    # Limit workers for CI environment
    runconfig_data.n_workers_for_despeckling = 4
    # We have a different product_dst_dir than the dst_dir called `tmp2`
    product_dst_dir = (test_dir / 'tmp2').resolve()
    runconfig_data.product_dst_dir = str(product_dst_dir)

    tmp_runconfig_yml_path = tmp_dir / 'runconfig.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path)

    # Run the command
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--runconfig_yml_path', str(tmp_runconfig_yml_path)],
        catch_exceptions=False,  # Let exceptions propagate for better debugging
    )

    product_directories = list(product_dst_dir.glob('OPERA*'))
    # Should be one and only one product directory
    assert len(product_directories) == 1

    # If we get here, check the product contents
    product_data_path = product_directories[0]
    out_product_data = ProductDirectoryData.from_product_path(product_data_path)

    # Check the product_dst_dir exists
    assert product_dst_dir.exists()
    assert result.exit_code == 0

    assert out_product_data == product_data_golden

    shutil.rmtree(tmp_dir)
    shutil.rmtree(product_dst_dir)


@pytest.mark.parametrize('device', ['best', 'cpu'])
def test_dist_s1_main_interface(
    cli_runner: CliRunner,
    test_dir: Path,
    test_data_dir: Path,
    change_local_dir: Callable[[Path], None],
    mocker: MockerFixture,
    device: str,
) -> None:
    """Tests the main dist-s1 CLI interface (not the outputs)."""
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    mocker.patch('dist_s1.localize_rtc_s1.enumerate_one_dist_s1_product', return_value=df_product)
    mocker.patch('dist_s1.localize_rtc_s1.localize_rtc_s1_ts', return_value=df_product)
    mocker.patch('dist_s1.workflows.run_dist_s1_sas_workflow', return_value=config)

    # Run the command
    result = cli_runner.invoke(
        dist_s1,
        [
            'run',
            '--mgrs_tile_id',
            '10SGD',
            '--post_date',
            '2025-01-02',
            '--track_number',
            '137',
            '--dst_dir',
            str(tmp_dir),
            '--apply_water_mask',
            'false',
            '--memory_strategy',
            'high',
            '--moderate_confidence_threshold',
            '3.5',
            '--high_confidence_threshold',
            '5.5',
            '--n_lookbacks',
            '3',
            '--product_dst_dir',
            str(tmp_dir),
            '--device',
            device,
        ],
    )
    assert result.exit_code == 0

    shutil.rmtree(tmp_dir)
