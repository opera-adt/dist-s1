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
    lookback_strategy: str = 'immediate_lookback',
    post_date_buffer_days: int = 1,
    max_pre_imgs_per_burst_mw: list[int] = [5, 5],
    delta_lookback_days_mw: list[int] = [730, 365],
    input_data_dir: Path | str | None = None,
    dst_dir: Path | str | None = 'out',
    tqdm_enabled: bool = True,
    apply_water_mask: bool = True,
    water_mask_path: Path | str | None = None,
) -> RunConfigData:
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
    # Ensure earthdata Credentials
    ensure_earthdata_credentials()

    # The function will create the out_dir if it doesn't exist
    if input_data_dir is None:
        input_data_dir = dst_dir
    df_product_loc = localize_rtc_s1_ts(df_product, input_data_dir, max_workers=5, tqdm_enabled=tqdm_enabled)
    runconfig = RunConfigData.from_product_df(
        df_product_loc,
        dst_dir,
        apply_water_mask=apply_water_mask,
        water_mask_path=water_mask_path,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
    )
    return runconfig
