import json
import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from dist_s1.__main__ import cli as dist_s1
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData


def test_dist_s1_sas_main(
    cli_runner: CliRunner,
    test_dir: Path,
    change_local_dir: Callable[[Path], None],
    cropped_10SGD_dataset_runconfig: Path,
    test_opera_golden_cropped_dataset_dict: dict[str, Path],
) -> None:
    """Test the dist-s1 sas main function.

    This is identical to running from the test_directory:

    `dist-s1 run_sas --runconfig_yml_path test_data/cropped/sample_runconfig_10SGD_cropped.yml`

    And comparing the output product directory to the golden dummy dataset.

    Note: the hardest part is serializing the runconfig to yml and then correctly finding the generated product.
    This is because the product paths from the in-memory runconfig object are different from the ones created via yml.
    This is because the product paths have the *processing time* in them, and that is different depending on when the
    runconfig object is created.

    To generate the runconfig, run the following command from the test_dir:
    ```python
    from dist_s1.data_models.runconfig_model import RunConfigData
    import geopandas as gpd

    df = gpd.read_parquet('test_data/cropped/10SGD__137__2025-01-02_dist_s1_inputs.parquet')

    config = RunConfigData.from_product_df(df)
    config.to_yaml('run_config.yml')
    ```

    Then remove fields that are not required so they can be set to default.
    """
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    product_dst_dir = (test_dir / 'tmp2').resolve()
    if product_dst_dir.exists():
        shutil.rmtree(product_dst_dir)

    product_data_golden = DistS1ProductDirectory.from_product_path(test_opera_golden_cropped_dataset_dict['current'])

    # Load and modify runconfig - not the paths are relative to the test_dir
    runconfig_data = RunConfigData.from_yaml(cropped_10SGD_dataset_runconfig)
    # Memory strategy was set to high to create the golden dataset
    runconfig_data.algo_config.memory_strategy = 'high'
    runconfig_data.algo_config.device = 'cpu'
    runconfig_data.algo_config.n_workers_for_despeckling = 4
    runconfig_data.apply_water_mask = True
    runconfig_data.algo_config.stride_for_norm_param_estimation = 16
    runconfig_data.algo_config.low_confidence_alert_threshold = 3.5
    runconfig_data.algo_config.high_confidence_alert_threshold = 5.5

    # We have a different product_dst_dir than the dst_dir called `tmp2`
    runconfig_data.product_dst_dir = str(product_dst_dir)

    tmp_runconfig_yml_path = tmp_dir / 'runconfig.yml'
    tmp_algo_params_yml_path = tmp_dir / 'algo_params.yml'
    runconfig_data.to_yaml(tmp_runconfig_yml_path, algo_param_path=tmp_algo_params_yml_path)

    # Run the command
    result = cli_runner.invoke(
        dist_s1,
        ['run_sas', '--run_config_path', str(tmp_runconfig_yml_path)],
        catch_exceptions=False,  # Let exceptions propagate for better debugging
    )

    product_directories = list(product_dst_dir.glob('OPERA*'))
    # Should be one and only one product directory
    assert len(product_directories) == 1

    # If we get here, check the product contents
    product_data_path = product_directories[0]
    out_product_data = DistS1ProductDirectory.from_product_path(product_data_path)

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
    monkeypatch: MonkeyPatch,
    device: str,
) -> None:
    """Tests the main dist-s1 CLI interface (not the outputs)."""
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('EARTHDATA_USERNAME', 'foo')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'bar')

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2025-01-02_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    # We don't need credentials because we mock the data.
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
            '--low_confidence_alert_threshold',
            '3.5',
            '--high_confidence_alert_threshold',
            '5.5',
            '--product_dst_dir',
            str(tmp_dir),
            '--device',
            device,
            '--n_workers_for_norm_param_estimation',
            '1',  # Required for MPS/CUDA devices when device='best' resolves to GPU
        ],
    )
    assert result.exit_code == 0

    shutil.rmtree(tmp_dir)


def test_dist_s1_main_interface_external_model(
    cli_runner: CliRunner,
    test_dir: Path,
    test_data_dir: Path,
    change_local_dir: Callable[[Path], None],
    mocker: MockerFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests the main dist-s1 CLI interface with external model source.

    Note: This test only uses CPU device to avoid MPS validation issues in the mocked workflow.
    """
    device = 'cpu'  # Use CPU to avoid MPS multiprocessing validation issues
    # Store original working directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('EARTHDATA_USERNAME', 'foo')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'bar')

    # Create temporary model config and weights files
    model_cfg_path = tmp_dir / 'model_config.json'
    model_wts_path = tmp_dir / 'model_weights.pth'

    # Create dummy config file (JSON format)
    model_cfg_content = {'model_type': 'transformer', 'n_heads': 8, 'd_model': 256, 'num_layers': 6, 'max_seq_len': 4}
    with model_cfg_path.open('w') as f:
        json.dump(model_cfg_content, f)

    # Create dummy weights file (just a placeholder)
    model_wts_path.write_text('dummy_weights_content')

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2025-01-02_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    # We don't need credentials because we mock the data.
    mocker.patch('dist_s1.localize_rtc_s1.enumerate_one_dist_s1_product', return_value=df_product)
    mocker.patch('dist_s1.localize_rtc_s1.localize_rtc_s1_ts', return_value=df_product)
    mocker.patch('dist_s1.workflows.run_dist_s1_sas_workflow', return_value=config)

    # Run the command with external model source
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
            '--low_confidence_alert_threshold',
            '3.5',
            '--high_confidence_alert_threshold',
            '5.5',
            '--stride_for_norm_param_estimation',
            '16',
            '--product_dst_dir',
            str(tmp_dir),
            '--device',
            device,
            '--n_workers_for_norm_param_estimation',
            '1',  # Required for MPS/CUDA devices
            '--model_source',
            'external',
            '--model_cfg_path',
            str(model_cfg_path),
            '--model_wts_path',
            str(model_wts_path),
        ],
    )
    assert result.exit_code == 0

    # Verify the temporary files were created and exist
    assert model_cfg_path.exists()
    assert model_wts_path.exists()

    shutil.rmtree(tmp_dir)


def test_run_one_confirmation_main_interface(
    cli_runner: CliRunner,
    test_dir: Path,
    change_local_dir: Callable[[Path], None],
    unconfirmed_products_chile_fire_dir: Path,
) -> None:
    """Test the run_one_confirmation CLI interface that mirrors confirm_pair.sh.

    This test mirrors the functionality of confirm_pair.sh:
    dist-s1 run_one_confirmation \
        --prior_dist_s1_product ./chile_fire_unconfirmed/\
OPERA_L3_DIST-ALERT-S1_T19HBD_20240128T233646Z_20250806T142756Z_S1_30_v0.1 \
        --current_dist_s1_product ./chile_fire_unconfirmed/\
OPERA_L3_DIST-ALERT-S1_T19HBD_20240202T100430Z_20250806T143249Z_S1_30_v0.1 \
        --dst_dist_product_parent ./chile_fire_confirmed/
    """
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp_confirm_pair'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Get the sorted product directories to match the bash script exactly
    product_dirs = sorted(list(unconfirmed_products_chile_fire_dir.glob('OPERA*')))
    assert len(product_dirs) >= 2, 'Need at least 2 products for pair confirmation'

    # Use the first two products as prior and current (chronologically ordered)
    prior_product = product_dirs[0]
    current_product = product_dirs[1]

    # Run the CLI command
    result = cli_runner.invoke(
        dist_s1,
        [
            'run_one_confirmation',
            '--prior_dist_s1_product',
            str(prior_product),
            '--current_dist_s1_product',
            str(current_product),
            '--dst_dist_product_parent',
            str(tmp_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f'CLI command failed: {result.output}'

    # Verify that a confirmed product was created
    confirmed_products = list(tmp_dir.glob('OPERA*'))
    assert len(confirmed_products) == 1, f'Expected 1 confirmed product, got {len(confirmed_products)}'

    # Verify the confirmed product inherits the name from the current product
    expected_product_name = current_product.name
    actual_product_name = confirmed_products[0].name
    assert actual_product_name == expected_product_name, (
        f'Confirmed product name mismatch: expected {expected_product_name}, got {actual_product_name}'
    )

    shutil.rmtree(tmp_dir)


def test_run_sequential_confirmation_main_interface(
    cli_runner: CliRunner,
    test_dir: Path,
    change_local_dir: Callable[[Path], None],
    unconfirmed_products_chile_fire_dir: Path,
) -> None:
    """Test the run_sequential_confirmation CLI interface that mirrors confirm_sequence.sh.

    This test mirrors the functionality of confirm_sequence.sh:
    dist-s1 run_sequential_confirmation --unconfirmed_dist_s1_product_dir ./chile_fire_unconfirmed \
                                        --dst_dist_product_parent chile_fire_confirmed
    """
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp_confirm_sequence'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Count the number of unconfirmed products
    unconfirmed_products = list(unconfirmed_products_chile_fire_dir.glob('OPERA*'))
    expected_product_count = len(unconfirmed_products)
    assert expected_product_count > 0, 'Need at least 1 product for sequential confirmation'

    # Run the CLI command
    result = cli_runner.invoke(
        dist_s1,
        [
            'run_sequential_confirmation',
            '--unconfirmed_dist_s1_product_dir',
            str(unconfirmed_products_chile_fire_dir),
            '--dst_dist_product_parent',
            str(tmp_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f'CLI command failed: {result.output}'

    # Verify that all products were confirmed
    confirmed_products = list(tmp_dir.glob('OPERA*'))
    assert len(confirmed_products) == expected_product_count, (
        f'Expected {expected_product_count} confirmed products, got {len(confirmed_products)}'
    )

    # Verify all confirmed products exist and have expected names
    confirmed_product_names = {p.name for p in confirmed_products}
    unconfirmed_product_names = {p.name for p in unconfirmed_products}
    assert confirmed_product_names == unconfirmed_product_names, (
        f'Product name mismatch: confirmed {confirmed_product_names} vs unconfirmed {unconfirmed_product_names}'
    )

    shutil.rmtree(tmp_dir)
