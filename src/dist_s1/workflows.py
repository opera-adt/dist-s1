from datetime import datetime
from pathlib import Path

from dist_s1.data_models.runconfig_model import RunConfigData, create_runconfig_from_input_df
from dist_s1.localize_rtc_s1 import localize_rtc_s1


def run_dist_s1_localization_workflow(mgrs_tile_id: str,
                                      post_date: str | datetime,
                                      track_number: int,
                                      post_date_buffer_days: int) -> RunConfigData:

    # Localize inputs
    rtc_s1_gdf = localize_rtc_s1(mgrs_tile_id, post_date, track_number, post_date_buffer_days = post_date_buffer_days)

    # Create runconfig
    run_config = create_runconfig_from_input_df(rtc_s1_gdf)

    return run_config


def run_dist_s1_processing_workflow(run_config: RunConfigData) -> Path:

    return Path('OPERA_L3_DIST_DIRECTORY')


def run_dist_s1_packaging_workflow(run_config: RunConfigData) -> Path:

    return Path('OPERA_L3_DIST_DIRECTORY')


def run_dist_s1_sas_workflow(run_config: RunConfigData) -> Path:

    _ = run_dist_s1_processing_workflow(run_config)
    _ = run_dist_s1_packaging_workflow(run_config)
    return run_config


def run_dist_s1_workflow(mgrs_tile_id: str,
                         post_date: str | datetime,
                         track_number: int,
                         post_date_buffer_days: int) -> Path:

    run_config = run_dist_s1_localization_workflow(mgrs_tile_id, post_date, track_number, post_date_buffer_days)
    _ = run_dist_s1_sas_workflow(run_config)

    return run_config

