from pathlib import Path

import numpy as np
import rasterio
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from rasterio.crs import CRS
from rasterio.transform import array_bounds as get_array_bounds
from rasterio.warp import transform_bounds as transform_bounds_into_crs
from tile_mate import get_raster_from_tiles

from dist_s1.rio_tools import get_mgrs_profile


def get_water_mask(mgrs_tile_id: str, out_path: Path, overwrite: bool = False) -> Path:
    if out_path.exists() and not overwrite:
        return out_path
    profile_mgrs = get_mgrs_profile(mgrs_tile_id)
    height = profile_mgrs['height']
    width = profile_mgrs['width']
    transform = profile_mgrs['transform']
    mgrs_bounds_utm = get_array_bounds(height, width, transform)
    mgrs_bounds_4326 = transform_bounds_into_crs(mgrs_bounds_utm, CRS.from_epsg(4326))

    X_glad_lc, p_glad_lc = get_raster_from_tiles(mgrs_bounds_4326, tile_shortname='glad_landcover', year=2020)
    # open water classes
    water_labels = [k for k in range(203, 208)]  # These are pixels that have surface water at least 50% of the time.
    water_labels.extend(
        [
            254,  # ocean
            255,  # no data
        ]
    )
    X_water_glad = np.isin(X_glad_lc[0, ...], water_labels).astype(np.uint8)

    X_water_glad_r, p_water_glad = reproject_arr_to_match_profile(
        X_water_glad, p_glad_lc, profile_mgrs, resampling='nearest'
    )
    X_water_glad_r = X_water_glad_r[0, ...]

    with rasterio.open(out_path, 'w', **p_water_glad) as dst:
        dst.write(X_water_glad_r, 1)

    return out_path
