import json
import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import check_profiles_match, open_one_profile
from dist_s1.workflows import (
    run_burst_disturbance_workflow,
    run_despeckle_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
)


ERASE_WORKFLOW_OUTPUTS = True


def test_despeckle_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)
    config.apply_water_mask = False

    run_despeckle_workflow(config)

    dspkl_copol_paths = config.df_inputs.loc_path_copol_dspkl.tolist()
    dspkl_crosspol_paths = config.df_inputs.loc_path_crosspol_dspkl.tolist()
    dst_paths = dspkl_copol_paths + dspkl_crosspol_paths

    assert all(Path(dst_path).exists() for dst_path in dst_paths)

    burst_ids = config.df_inputs.jpl_burst_id.unique().tolist()
    for burst_id in burst_ids:
        dst_path_by_burst_id = [path for path in dst_paths if burst_id in path]
        profiles = [open_one_profile(path) for path in dst_path_by_burst_id]
        assert all(check_profiles_match(profiles[0], profile) for profile in profiles[1:])

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)


def test_burst_disturbance_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dirs_to_move = ['tv_despeckle', 'normal_params']
    for dir_name in dirs_to_move:
        src_dir = test_data_dir / '10SGD_cropped_dst' / dir_name
        dst_dir = tmp_dir / dir_name
        if Path(dst_dir).exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)
    config.apply_water_mask = False
    config.device = 'cpu'

    run_burst_disturbance_workflow(config)

    shutil.rmtree(tmp_dir)


def test_dist_s1_sas_workflow(
    test_dir: Path, test_data_dir: Path, change_local_dir: Callable, test_opera_golden_dummy_dataset: Path
) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.apply_water_mask = False
    config.device = 'cpu'

    run_dist_s1_sas_workflow(config)

    product_data = config.product_data_model
    product_data_golden = ProductDirectoryData.from_product_path(test_opera_golden_dummy_dataset)

    assert product_data == product_data_golden

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)


def test_dist_s1_workflow_interface(
    test_dir: Path,
    test_data_dir: Path,
    change_local_dir: Callable,
    mocker: MockerFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests the s1 workflow interface, not the outputs."""
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('EARTHDATA_USERNAME', 'foo')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'bar')

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)
    config.apply_water_mask = False

    # We don't need credentials because we mock the data.
    mocker.patch('dist_s1.localize_rtc_s1.enumerate_one_dist_s1_product', return_value=df_product)
    mocker.patch('dist_s1.localize_rtc_s1.localize_rtc_s1_ts', return_value=df_product)
    mocker.patch('dist_s1.workflows.run_dist_s1_sas_workflow', return_value=config)

    run_dist_s1_workflow(
        mgrs_tile_id='10SGD',
        post_date='2025-01-02',
        track_number=137,
        dst_dir=tmp_dir,
        apply_water_mask=False,
    )

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)


def test_dist_s1_workflow_interface_external_model(
    test_dir: Path,
    test_data_dir: Path,
    change_local_dir: Callable,
    mocker: MockerFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests the s1 workflow interface with external model source, not the outputs."""
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('EARTHDATA_USERNAME', 'foo')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'bar')

    # Create temporary model config and weights files
    model_cfg_path = tmp_dir / 'model_config.json'
    model_wts_path = tmp_dir / 'model_weights.pth'

    # Create dummy config file (JSON format)
    model_cfg_content = {'model_type': 'transformer', 'n_heads': 8, 'd_model': 256, 'num_layers': 6}
    with model_cfg_path.open('w') as f:
        json.dump(model_cfg_content, f)

    # Create dummy weights file (just a placeholder)
    model_wts_path.write_text('dummy_weights_content')

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)
    config.apply_water_mask = False

    # We don't need credentials because we mock the data.
    mocker.patch('dist_s1.localize_rtc_s1.enumerate_one_dist_s1_product', return_value=df_product)
    mocker.patch('dist_s1.localize_rtc_s1.localize_rtc_s1_ts', return_value=df_product)
    mocker.patch('dist_s1.workflows.run_dist_s1_sas_workflow', return_value=config)

    run_dist_s1_workflow(
        mgrs_tile_id='10SGD',
        post_date='2025-01-02',
        track_number=137,
        dst_dir=tmp_dir,
        apply_water_mask=False,
        device='cpu',  # Use CPU to avoid MPS validation issues
        n_workers_for_norm_param_estimation=1,  # Required for MPS/CUDA devices
        model_source='external',
        model_cfg_path=str(model_cfg_path),
        model_wts_path=str(model_wts_path),
    )

    # Verify the temporary files were created and exist
    assert model_cfg_path.exists()
    assert model_wts_path.exists()

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)
