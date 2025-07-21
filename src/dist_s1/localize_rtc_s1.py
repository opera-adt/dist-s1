from datetime import datetime
from pathlib import Path

import pandas as pd
from dist_s1_enumerator import enumerate_one_dist_s1_product, localize_rtc_s1_ts

from dist_s1.constants import MODEL_CONTEXT_LENGTH
from dist_s1.credentials import ensure_earthdata_credentials
from dist_s1.data_models.runconfig_model import RunConfigData


def localize_rtc_s1(
    mgrs_tile_id: str,
    post_date: str | datetime | pd.Timestamp,
    track_number: int,
    lookback_strategy: str = 'multi_window',
    post_date_buffer_days: int = 1,
    max_pre_imgs_per_burst_mw: list[int] = [5, 5],
    delta_lookback_days_mw: list[int] = [730, 365],
    input_data_dir: Path | str | None = None,
    dst_dir: Path | str | None = 'out',
    tqdm_enabled: bool = True,
) -> RunConfigData:
    """Localize RTC-S1 data and create RunConfigData.

    This function focuses on data enumeration and localization.
    Configuration and algorithm parameters should be set via assignment after creation.

    Parameters
    ----------
    mgrs_tile_id : str
        MGRS tile identifier
    post_date : str | datetime | pd.Timestamp
        Post acquisition date
    track_number : int
        Sentinel-1 track number
    lookback_strategy : str
        Strategy for looking back at historical data
    post_date_buffer_days : int
        Buffer days around post date
    max_pre_imgs_per_burst_mw : list[int]
        Max pre-images per burst for multi-window
    delta_lookback_days_mw : list[int]
        Lookback days for multi-window
    input_data_dir : Path | str | None
        Directory for input data storage
    dst_dir : Path | str | None
        Destination directory for outputs
    tqdm_enabled : bool
        Whether to show progress bars

    Returns
    -------
    RunConfigData
        Configured RunConfigData object with localized RTC inputs
    """
    df_product = enumerate_one_dist_s1_product(
        mgrs_tile_id,
        track_number=track_number,
        post_date=post_date,
        lookback_strategy=lookback_strategy,
        post_date_buffer_days=post_date_buffer_days,
        max_pre_imgs_per_burst=(MODEL_CONTEXT_LENGTH + 2),
        max_pre_imgs_per_burst_mw=[item for item in max_pre_imgs_per_burst_mw],
        delta_lookback_days_mw=delta_lookback_days_mw,
    )
    ensure_earthdata_credentials()

    if input_data_dir is None:
        input_data_dir = dst_dir
    df_product_loc = localize_rtc_s1_ts(df_product, input_data_dir, max_workers=5, tqdm_enabled=tqdm_enabled)

    runconfig = RunConfigData.from_product_df(
        df_product_loc,
        dst_dir=dst_dir,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        lookback_strategy=lookback_strategy,
    )
    return runconfig
