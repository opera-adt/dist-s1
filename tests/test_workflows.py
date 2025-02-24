import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest
from pytest_mock import MockerFixture

from dist_s1.data_models.output_models import ProductDirectoryData
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import check_profiles_match, open_one_profile
from dist_s1.workflows import (
    curate_input_burst_rtc_input_for_dist,
    curate_input_burst_rtc_s1_paths_for_normal_param_est,
    run_burst_disturbance_workflow,
    run_despeckle_workflow,
    run_dist_s1_sas_workflow,
    run_dist_s1_workflow,
    run_normal_param_estimation_workflow,
)


ERASE_WORKFLOW_OUTPUTS = True


def test_despeckle_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

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


def test_normal_params_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_tv_dir = test_data_dir / '10SGD_cropped_dst' / 'tv_despeckle'

    dst_tv_dir = tmp_dir / 'tv_despeckle'

    if Path(dst_tv_dir).exists():
        shutil.rmtree(dst_tv_dir)
    shutil.copytree(src_tv_dir, dst_tv_dir)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    run_normal_param_estimation_workflow(config)

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
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    run_burst_disturbance_workflow(config)

    shutil.rmtree(tmp_dir)


@pytest.mark.parametrize('lookback', [0, 1, 2, 3])
def test_curation_of_burst_rtc_s1_paths_for_normal_param_est(lookback: int) -> None:
    cross_pol_paths = [
        './tv_despeckle/2024-01-08/OPERA_L2_RTC-S1_T137-292318-IW1_20240108T015902Z_20240109T091413Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-01-20/OPERA_L2_RTC-S1_T137-292318-IW1_20240120T015902Z_20240120T143322Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-01/OPERA_L2_RTC-S1_T137-292318-IW1_20240201T015901Z_20240201T114629Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-13/OPERA_L2_RTC-S1_T137-292318-IW1_20240213T015901Z_20240213T091319Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-25/OPERA_L2_RTC-S1_T137-292318-IW1_20240225T015901Z_20240225T100928Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-03-08/OPERA_L2_RTC-S1_T137-292318-IW1_20240308T015901Z_20240409T075111Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-03-20/OPERA_L2_RTC-S1_T137-292318-IW1_20240320T015901Z_20240321T155238Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-01/OPERA_L2_RTC-S1_T137-292318-IW1_20240401T015902Z_20240418T135305Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-13/OPERA_L2_RTC-S1_T137-292318-IW1_20240413T015901Z_20240419T082133Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-25/OPERA_L2_RTC-S1_T137-292318-IW1_20240425T015902Z_20240427T061145Z_S1A_30_v1.0_VH_tv.tif',
    ]
    copol_paths = [path.replace('VH_tv.tif', 'VV_tv.tif') for path in cross_pol_paths]

    copol_paths_pre, crosspol_paths_pre = curate_input_burst_rtc_s1_paths_for_normal_param_est(
        copol_paths, cross_pol_paths, lookback=lookback
    )
    assert len(copol_paths_pre) == len(crosspol_paths_pre)
    assert crosspol_paths_pre == cross_pol_paths[: -lookback - 1]


@pytest.mark.parametrize('lookback', [0, 1, 2, 3])
def test_curate_input_burst_rtc_input_for_dist(lookback: int) -> None:
    cross_pol_paths = [
        './tv_despeckle/2024-01-08/OPERA_L2_RTC-S1_T137-292318-IW1_20240108T015902Z_20240109T091413Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-01-20/OPERA_L2_RTC-S1_T137-292318-IW1_20240120T015902Z_20240120T143322Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-01/OPERA_L2_RTC-S1_T137-292318-IW1_20240201T015901Z_20240201T114629Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-13/OPERA_L2_RTC-S1_T137-292318-IW1_20240213T015901Z_20240213T091319Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-02-25/OPERA_L2_RTC-S1_T137-292318-IW1_20240225T015901Z_20240225T100928Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-03-08/OPERA_L2_RTC-S1_T137-292318-IW1_20240308T015901Z_20240409T075111Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-03-20/OPERA_L2_RTC-S1_T137-292318-IW1_20240320T015901Z_20240321T155238Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-01/OPERA_L2_RTC-S1_T137-292318-IW1_20240401T015902Z_20240418T135305Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-13/OPERA_L2_RTC-S1_T137-292318-IW1_20240413T015901Z_20240419T082133Z_S1A_30_v1.0_VH_tv.tif',
        './tv_despeckle/2024-04-25/OPERA_L2_RTC-S1_T137-292318-IW1_20240425T015902Z_20240427T061145Z_S1A_30_v1.0_VH_tv.tif',
    ]
    copol_paths = [path.replace('VH_tv.tif', 'VV_tv.tif') for path in cross_pol_paths]

    copol_paths_post, crosspol_paths_post = curate_input_burst_rtc_input_for_dist(
        copol_paths, cross_pol_paths, lookback=lookback
    )
    assert len(copol_paths_post) == len(crosspol_paths_post)
    assert crosspol_paths_post == cross_pol_paths[-lookback - 1 :]
    assert copol_paths_post == copol_paths[-lookback - 1 :]


def test_dist_s1_sas_workflow(
    test_dir: Path, test_data_dir: Path, change_local_dir: Callable, test_opera_golden_dummy_dataset: Path
) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    assert tmp_dir.exists() and tmp_dir.is_dir()

    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

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
    test_opera_golden_dummy_dataset: Path,
) -> None:
    """Tests the s1 workflow interface, not the outputs."""
    change_local_dir(test_dir)
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / 'cropped' / '10SGD__137__2024-09-04_dist_s1_inputs.parquet')
    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir, apply_water_mask=False)

    mocker.patch('dist_s1.localize_rtc_s1.enumerate_one_dist_s1_product', return_value=df_product)
    mocker.patch('dist_s1.localize_rtc_s1.localize_rtc_s1_ts', return_value=df_product)
    mocker.patch('dist_s1.workflows.run_dist_s1_sas_workflow', return_value=config)

    run_dist_s1_workflow(
        mgrs_tile_id='10SGD', post_date='2025-01-02', track_number=137, dst_dir=tmp_dir, apply_water_mask=False
    )

    if ERASE_WORKFLOW_OUTPUTS:
        shutil.rmtree(tmp_dir)
