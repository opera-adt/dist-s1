from collections.abc import Callable
from pathlib import Path

import geopandas as gpd

from dist_s1.input_data_model import RunConfigModel


def test_input_data_model_from_cropped_dataset(test_dir: Path, test_data_dir: Path, change_local_dir: Callable):
    change_local_dir(test_dir)

    df_pre = gpd.read_parquet(test_data_dir / '10SGD_cropped' / 'pre_10SGD.parquet')
    df_post = gpd.read_parquet(test_data_dir / '10SGD_cropped' / 'post_10SGD.parquet')

    pre_rtc_copol_paths = df_pre.loc_path_vv.tolist()
    pre_rtc_crosspol_paths = df_pre.loc_path_vh.tolist()
    post_rtc_copol_paths = df_post.loc_path_vv.tolist()
    post_rtc_crosspol_paths = df_post.loc_path_vh.tolist()

    mgrs_tile_id = '10SGD'

    config = RunConfigModel(
        pre_rtc_copol=pre_rtc_copol_paths,
        pre_rtc_crosspol=pre_rtc_crosspol_paths,
        post_rtc_copol=post_rtc_copol_paths,
        post_rtc_crosspol=post_rtc_crosspol_paths,
        mgrs_tile_id=mgrs_tile_id,
    )

    # Check burst ids
    burst_ids_actual = sorted(list(config.time_series_by_burst.keys()))
    burst_ids_expected = [
        'T137-292318-IW1',
        'T137-292318-IW2',
        'T137-292319-IW1',
        'T137-292319-IW2',
        'T137-292320-IW1',
        'T137-292320-IW2',
        'T137-292321-IW1',
        'T137-292321-IW2',
        'T137-292322-IW1',
        'T137-292322-IW2',
        'T137-292323-IW1',
        'T137-292323-IW2',
        'T137-292324-IW1',
        'T137-292324-IW2',
        'T137-292325-IW1',
    ]
    assert burst_ids_actual == burst_ids_expected

    # Check Copol/crosspol ids for 1 burst - we expect each list to be sorted by date of acq
    pre_rtc_crosspol_paths = config.time_series_by_burst['T137-292319-IW2']['pre_rtc_crosspol']
    pre_rtc_crosspol_ids_actual = [p.stem for p in pre_rtc_crosspol_paths]

    pre_rtc_copol_paths = config.time_series_by_burst['T137-292319-IW2']['pre_rtc_copol']
    pre_rtc_copol_ids_actual = [p.stem for p in pre_rtc_copol_paths]

    post_rtc_crosspol_paths = config.time_series_by_burst['T137-292319-IW2']['post_rtc_crosspol']
    post_rtc_crosspol_ids_actual = [p.stem for p in post_rtc_crosspol_paths]

    post_rtc_copol_paths = config.time_series_by_burst['T137-292319-IW2']['post_rtc_copol']
    post_rtc_copol_ids_actual = [p.stem for p in post_rtc_copol_paths]

    pre_rtc_copol_ids_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240706T015904Z_20240708T000435Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240718T015904Z_20240718T072557Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240730T015904Z_20240730T064555Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240811T015904Z_20240811T064742Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240823T015904Z_20240823T084936Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240904T015904Z_20240904T150822Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240916T015905Z_20240916T114330Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240928T015905Z_20240929T005548Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20241010T015906Z_20241010T101259Z_S1A_30_v1.0_VV',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20241022T015905Z_20241022T180854Z_S1A_30_v1.0_VV',
    ]

    pre_rtc_crosspol_ids_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240706T015904Z_20240708T000435Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240718T015904Z_20240718T072557Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240730T015904Z_20240730T064555Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240811T015904Z_20240811T064742Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240823T015904Z_20240823T084936Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240904T015904Z_20240904T150822Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240916T015905Z_20240916T114330Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240928T015905Z_20240929T005548Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20241010T015906Z_20241010T101259Z_S1A_30_v1.0_VH',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20241022T015905Z_20241022T180854Z_S1A_30_v1.0_VH',
    ]

    post_rtc_copol_ids_expected = ['OPERA_L2_RTC-S1_T137-292319-IW2_20241103T015905Z_20241103T071409Z_S1A_30_v1.0_VV']
    post_rtc_crosspol_ids_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20241103T015905Z_20241103T071409Z_S1A_30_v1.0_VH'
    ]

    assert pre_rtc_crosspol_ids_actual == pre_rtc_crosspol_ids_expected
    assert pre_rtc_copol_ids_actual == pre_rtc_copol_ids_expected
    assert post_rtc_copol_ids_actual == post_rtc_copol_ids_expected
    assert post_rtc_crosspol_ids_actual == post_rtc_crosspol_ids_expected

    # Check acquisition dates for 1 burst
    pre_acq_dts = config.time_series_by_burst['T137-292319-IW2']['pre_acq_dts']
    post_acq_dts = config.time_series_by_burst['T137-292319-IW2']['post_acq_dts']

    pre_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in pre_acq_dts]
    post_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in post_acq_dts]

    pre_acq_dts_str_expected = [
        '20240706T015904',
        '20240718T015904',
        '20240730T015904',
        '20240811T015904',
        '20240823T015904',
        '20240904T015904',
        '20240916T015905',
        '20240928T015905',
        '20241010T015906',
        '20241022T015905',
    ]

    post_acq_dts_str_expected = ['20241103T015905']

    assert pre_acq_dts_str_actual == pre_acq_dts_str_expected
    assert post_acq_dts_str_actual == post_acq_dts_str_expected
