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
