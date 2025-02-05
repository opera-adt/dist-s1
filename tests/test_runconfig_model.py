from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np

from dist_s1.data_models.runconfig_model import RunConfigData


def test_input_data_model_from_cropped_dataset(test_dir: Path, test_data_dir: Path, change_local_dir: Callable) -> None:
    change_local_dir(test_dir)

    df_product = gpd.read_parquet(test_data_dir / '10SGD_cropped' / '10SGD__137__2024-01-08_dist_s1_inputs.parquet')

    config = RunConfigData.from_product_df(df_product)

    df = config.df_inputs

    # Check burst ids
    burst_ids_actual = df.jpl_burst_id.unique().tolist()
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

    ind_burst = df.jpl_burst_id == 'T137-292319-IW2'
    ind_pre = df.input_category == 'pre'
    ind_post = df.input_category == 'post'

    pre_rtc_crosspol_paths = df[ind_pre & ind_burst].loc_path_crosspol.tolist()
    pre_rtc_copol_paths = df[ind_pre & ind_burst].loc_path_copol.tolist()

    pre_rtc_crosspol_tif_filenames_actual = [Path(p).name for p in pre_rtc_crosspol_paths]
    pre_rtc_copol_tif_filenames_actual = [Path(p).name for p in pre_rtc_copol_paths]

    post_rtc_crosspol_paths = df[ind_post & ind_burst].loc_path_crosspol.tolist()
    post_rtc_copol_paths = df[ind_post & ind_burst].loc_path_copol.tolist()

    post_rtc_crosspol_tif_filenames_actual = [Path(p).name for p in post_rtc_crosspol_paths]
    post_rtc_copol_tif_filenames_actual = [Path(p).name for p in post_rtc_copol_paths]

    pre_rtc_copol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240108T015906Z_20240109T091413Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240120T015905Z_20240120T143322Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240201T015905Z_20240201T114629Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240213T015905Z_20240213T091319Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240225T015905Z_20240225T100928Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240308T015905Z_20240409T075111Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240320T015905Z_20240321T155238Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240401T015906Z_20240418T135305Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240413T015904Z_20240419T082133Z_S1A_30_v1.0_VV.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240425T015906Z_20240427T061145Z_S1A_30_v1.0_VV.tif',
    ]

    pre_rtc_crosspol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240108T015906Z_20240109T091413Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240120T015905Z_20240120T143322Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240201T015905Z_20240201T114629Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240213T015905Z_20240213T091319Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240225T015905Z_20240225T100928Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240308T015905Z_20240409T075111Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240320T015905Z_20240321T155238Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240401T015906Z_20240418T135305Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240413T015904Z_20240419T082133Z_S1A_30_v1.0_VH.tif',
        'OPERA_L2_RTC-S1_T137-292319-IW2_20240425T015906Z_20240427T061145Z_S1A_30_v1.0_VH.tif',
    ]

    post_rtc_copol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20250102T015901Z_20250102T190143Z_S1A_30_v1.0_VV.tif'
    ]

    post_rtc_crosspol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20250102T015901Z_20250102T190143Z_S1A_30_v1.0_VH.tif'
    ]

    assert pre_rtc_crosspol_tif_filenames_actual == pre_rtc_crosspol_tif_filenames_expected
    assert pre_rtc_copol_tif_filenames_actual == pre_rtc_copol_tif_filenames_expected
    assert post_rtc_copol_tif_filenames_actual == post_rtc_copol_tif_filenames_expected
    assert post_rtc_crosspol_tif_filenames_actual == post_rtc_crosspol_tif_filenames_expected

    # Check acquisition dates for 1 burst
    pre_acq_dts = np.array(df[ind_pre & ind_burst].acq_dt.dt.to_pydatetime())
    post_acq_dts = np.array(df[ind_post & ind_burst].acq_dt.dt.to_pydatetime())

    pre_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in pre_acq_dts]
    post_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in post_acq_dts]

    pre_acq_dts_str_expected = [
        '20240108T015906',
        '20240120T015905',
        '20240201T015905',
        '20240213T015905',
        '20240225T015905',
        '20240308T015905',
        '20240320T015905',
        '20240401T015906',
        '20240413T015904',
        '20240425T015906',
    ]

    post_acq_dts_str_expected = ['20250102T015901']

    assert pre_acq_dts_str_actual == pre_acq_dts_str_expected
    assert post_acq_dts_str_actual == post_acq_dts_str_expected
