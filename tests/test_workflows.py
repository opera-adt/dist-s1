import json
import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import check_profiles_match, open_one_profile
from dist_s1.workflows import (
    run_burst_disturbance_workflow,
    run_despeckle_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
    run_sequential_confirmation_of_dist_products_workflow,
)


ERASE_WORKFLOW_OUTPUTS = True


def test_despeckle_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2025-01-02_dist_s1_inputs.parquet')
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


def test_burst_disturbance_workflow(
    test_dir: Path,
    test_data_dir: Path,
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_dir = test_data_dir / '10SGD_cropped_dst' / 'tv_despeckle'
    dst_dir = tmp_dir / 'tv_despeckle'
    if Path(dst_dir).exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)
    config.apply_water_mask = False
    config.algo_config.device = 'cpu'

    run_burst_disturbance_workflow(config)

    shutil.rmtree(tmp_dir)


@pytest.mark.parametrize('current_or_prior', ['current', 'prior'])
def test_dist_s1_sas_workflow_no_confirmation(
    test_dir: Path,
    change_local_dir: Callable,
    current_or_prior: str,
    test_opera_golden_cropped_dataset_dict: dict[str, Path],
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    """Tests the dist-s1-sas workflow against a golden dataset."""
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict[current_or_prior]
    df_product = gpd.read_parquet(parquet_path)
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.apply_water_mask = True
    config.algo_config.device = 'cpu'

    run_dist_s1_sas_workflow(config)

    product_data = config.product_data_model
    golden_dataset_path = test_opera_golden_cropped_dataset_dict[current_or_prior]
    product_data_golden = DistS1ProductDirectory.from_product_path(golden_dataset_path)

    assert product_data == product_data_golden

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)


def test_dist_s1_sas_workflow_with_confirmation(
    test_dir: Path,
    change_local_dir: Callable,
    test_opera_golden_cropped_dataset_dict: dict[str, Path],
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    """Tests the dist-s1-sas workflow against a golden dataset."""
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.apply_water_mask = True
    config.algo_config.device = 'cpu'
    config.prior_dist_s1_product = test_opera_golden_cropped_dataset_dict['prior']

    run_dist_s1_sas_workflow(config)

    product_data_with_confirmation = config.product_data_model
    golden_dataset_path_confirmed = test_opera_golden_cropped_dataset_dict['confirmed']
    product_data_golden_confirmed = DistS1ProductDirectory.from_product_path(golden_dataset_path_confirmed)

    assert product_data_with_confirmation == product_data_golden_confirmed

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)


def test_dist_s1_workflow_interface(
    test_dir: Path,
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
    mocker: MockerFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests the s1 workflow interface, not the outputs."""
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv('EARTHDATA_USERNAME', 'foo')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'bar')

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)
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
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
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
    model_cfg_content = {'model_type': 'transformer', 'n_heads': 8, 'd_model': 256, 'num_layers': 6, 'max_seq_len': 4}
    with model_cfg_path.open('w') as f:
        json.dump(model_cfg_content, f)

    # Create dummy weights file (just a placeholder)
    model_wts_path.write_text('dummy_weights_content')

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)
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


def test_sequential_confirmation_workflow(
    test_dir: Path,
    change_local_dir: Callable,
    unconfirmed_products_chile_fire_dir: Path,
    confirmed_products_chile_fire_golden_dir: Path,
) -> None:
    """Test sequential confirmation workflow against golden dataset.

    This test:
    1. Takes unconfirmed products from test_data/products_without_confirmation_cropped__chile-fire_2024
    2. Runs run_sequential_confirmation_of_dist_products_workflow
    3. Compares output with golden dataset in test_data/golden_datasets/\
        products_with_confirmation_cropped__chile-fire_2024
    """
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_sequential_dir = test_dir / 'tmp' / 'confirmation_chile_sequential'
    if tmp_sequential_dir.exists():
        shutil.rmtree(tmp_sequential_dir)
    tmp_sequential_dir.mkdir(parents=True, exist_ok=True)

    # Run sequential confirmation workflow with explicit default parameters
    # Note: These are the default values from DEFAULT_* constants to ensure test stability
    run_sequential_confirmation_of_dist_products_workflow(
        directory_of_dist_s1_products=unconfirmed_products_chile_fire_dir,
        dst_dist_product_parent=tmp_sequential_dir,
        no_day_limit=30,  # DEFAULT_NO_DAY_LIMIT
        exclude_consecutive_no_dist=True,  # DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST
        percent_reset_thresh=10,  # DEFAULT_PERCENT_RESET_THRESH
        no_count_reset_thresh=7,  # DEFAULT_NO_COUNT_RESET_THRESH
        confidence_upper_lim=32000,  # DEFAULT_CONFIDENCE_UPPER_LIM
        confidence_thresh=31.5,  # DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD (3**2 * 3.5)
        metric_value_upper_lim=100.0,  # DEFAULT_METRIC_VALUE_UPPER_LIM
        tqdm_enabled=False,  # Disable progress bar for testing
    )

    # Compare each confirmed product with the corresponding golden dataset product
    golden_products = sorted(list(confirmed_products_chile_fire_golden_dir.glob('OPERA*')))
    confirmed_products = sorted(list(tmp_sequential_dir.glob('OPERA*')))

    assert len(golden_products) == len(confirmed_products), (
        f'Number of products mismatch: {len(golden_products)} golden vs {len(confirmed_products)} confirmed'
    )

    # Compare each product using DistS1ProductDirectory __eq__ method
    for golden_path, confirmed_path in zip(golden_products, confirmed_products):
        assert golden_path.name == confirmed_path.name, (
            f'Product name mismatch: {golden_path.name} vs {confirmed_path.name}'
        )

        golden_product = DistS1ProductDirectory.from_product_path(golden_path)
        confirmed_product = DistS1ProductDirectory.from_product_path(confirmed_path)

        assert golden_product == confirmed_product, f'Product comparison failed for {golden_path.name}'

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_sequential_dir)
