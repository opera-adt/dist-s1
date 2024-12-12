from datetime import datetime

import geopandas as gpd


def localize_rtc_s1(mgrs_tile_id: str,
                    post_date: str | datetime,
                    track_number: int,
                    post_date_buffer_days: int = 1) -> gpd.GeoDataFrame:

    return gpd.GeoDataFrame()
