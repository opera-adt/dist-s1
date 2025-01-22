import shutil
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd

from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import check_profiles_match, open_one_profile
from dist_s1.workflows import run_despeckle_workflow, run_normal_params_workflow


def test_despeckle_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    # Ensure that validation is relative to the test directory
    change_local_dir(test_dir)
    dst_dir = test_dir / 'tmp'
    dst_dir.mkdir(parents=True, exist_ok=True)

    df_product = gpd.read_parquet(test_data_dir / '10SGD_cropped' / '10SGD__137__2024-01-08_dist_s1_inputs.parquet')
    assert dst_dir.exists() and dst_dir.is_dir()

    config = RunConfigData.from_product_df(df_product, dst_dir=dst_dir)

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

    # shutil.rmtree(dst_dir)


def test_normal_params_workflow(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    change_local_dir(test_dir)
    src_tv_dir = test_data_dir / '10SGD_cropped_dst' / 'tv_despeckle'
    dst_dir = test_dir / 'tmp'
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_tv_dir = dst_dir / 'tv_despeckle'

    if Path(dst_tv_dir).exists():
        shutil.rmtree(dst_tv_dir)
    shutil.copytree(src_tv_dir, dst_tv_dir)

    df_product = gpd.read_parquet(test_data_dir / '10SGD_cropped' / '10SGD__137__2024-01-08_dist_s1_inputs.parquet')

    config = RunConfigData.from_product_df(df_product, dst_dir=dst_dir)
    assert dst_dir.exists() and dst_dir.is_dir()

    run_normal_params_workflow(config)

    # shutil.rmtree(dst_dir)
