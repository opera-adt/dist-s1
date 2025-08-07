from datetime import datetime
from pathlib import Path

import pandas as pd
from dist_s1_enumerator import enumerate_one_dist_s1_product, localize_rtc_s1_ts

from dist_s1.credentials import ensure_earthdata_credentials
from dist_s1.data_models.data_utils import get_max_pre_imgs_per_burst_mw
from dist_s1.data_models.defaults import DEFAULT_LOOKBACK_STRATEGY
from dist_s1.data_models.runconfig_model import RunConfigData


def localize_rtc_s1(
    mgrs_tile_id: str,
    post_date: str | datetime | pd.Timestamp,
    track_number: int,
    lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
    post_date_buffer_days: int = 1,
    max_pre_imgs_per_burst_mw: tuple[int, ...] | None = None,
    delta_lookback_days_mw: tuple[int, ...] | None = None,
    input_data_dir: Path | str | None = None,
    dst_dir: Path | str | None = 'out',
    tqdm_enabled: bool = True,
    model_context_length: int = 10,
    n_anniversaries_for_mw: int = 3,
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
    max_pre_imgs_per_burst_mw : tuple[int, int]
        Max pre-images per burst for multi-window
    delta_lookback_days_mw : tuple[int, int]
        Lookback days for multi-window
    input_data_dir : Path | str | None
        Directory for input data storage
    dst_dir : Path | str | None
        Destination directory for outputs
    tqdm_enabled : bool
        Whether to show progress bars
    model_context_length : int
        Context length for model application - maximum number of pre-images to
        use to establish baseline estimates. Default is 10. If max_pre_imgs_per_burst_mw is not provided,
        it will be calculated based on model_context_length and n_anniversaries_for_mw.
    n_anniversaries_for_mw : int
        Number of anniversaries to use for multi-window. Default is 3. If delta_lookback_days_mw is not provided,
        that variable will be calculated based on n_anniversaries_for_mw.


    Returns
    -------
    RunConfigData
        Configured RunConfigData object with localized RTC inputs
    """
    if max_pre_imgs_per_burst_mw is None:
        max_pre_imgs_per_burst_mw = get_max_pre_imgs_per_burst_mw(model_context_length, n_anniversaries_for_mw)
    if delta_lookback_days_mw is None:
        delta_lookback_days_mw = tuple(365 * n for n in range(n_anniversaries_for_mw, 0, -1))

    if len(max_pre_imgs_per_burst_mw) != len(delta_lookback_days_mw):
        raise ValueError(
            'len(max_pre_imgs_per_burst_mw) must be equal to len(delta_lookback_days_mw), '
            f'but got {len(max_pre_imgs_per_burst_mw)} and {len(delta_lookback_days_mw)}'
        )
    if sum(max_pre_imgs_per_burst_mw) > model_context_length:
        raise ValueError(
            'sum(max_pre_imgs_per_burst_mw) must be less than or equal to model_context_length, '
            f'but got {sum(max_pre_imgs_per_burst_mw)} from {max_pre_imgs_per_burst_mw} and '
            f'{model_context_length} from model_context_length.'
        )

    df_product = enumerate_one_dist_s1_product(
        mgrs_tile_id,
        track_number=track_number,
        post_date=post_date,
        lookback_strategy=lookback_strategy,
        post_date_buffer_days=post_date_buffer_days,
        max_pre_imgs_per_burst=max_pre_imgs_per_burst_mw,
        delta_lookback_days=delta_lookback_days_mw,
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
