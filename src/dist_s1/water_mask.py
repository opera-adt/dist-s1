from pathlib import Path

import numpy as np
import rasterio
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from dem_stitcher.rio_window import read_raster_from_window
from rasterio.crs import CRS
from rasterio.transform import array_bounds as get_array_bounds
from rasterio.warp import transform_bounds as transform_bounds_into_crs
from shapely.geometry import box
from tile_mate import get_raster_from_tiles

from dist_s1.rio_tools import get_mgrs_profile


def check_water_mask_profile(water_mask_profile: dict, ref_profile: dict) -> None:
    if water_mask_profile['crs'] != ref_profile['crs']:
        raise ValueError('Water mask and disturbance array CRS do not match')
    if water_mask_profile['transform'] != ref_profile['transform']:
        raise ValueError('Water mask and disturbance array transform do not match')
    if water_mask_profile['height'] != ref_profile['height']:
        raise ValueError('Water mask and disturbance array height do not match')
    if water_mask_profile['width'] != ref_profile['width']:
        raise ValueError('Water mask and disturbance array width do not match')
    return True


def get_water_mask(mgrs_tile_id: str, out_path: Path, overwrite: bool = False) -> Path:
    if out_path.exists() and not overwrite:
        return out_path
    profile_mgrs = get_mgrs_profile(mgrs_tile_id)
    height = profile_mgrs['height']
    width = profile_mgrs['width']
    transform = profile_mgrs['transform']
    mgrs_bounds_utm = get_array_bounds(height, width, transform)
    mgrs_bounds_4326 = transform_bounds_into_crs(profile_mgrs['crs'], CRS.from_epsg(4326), *mgrs_bounds_utm)

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


def water_mask_control_flow(
    *,
    water_mask_path: Path | str | None,
    mgrs_tile_id: str,
    apply_water_mask: bool,
    dst_dir: Path,
    overwrite: bool = True,
    buffer_size_pixel: int = 10,
) -> Path | None:
    """Read and resample water mask for serialization to disk, outputing its path on filesystem.

    Parameters
    ----------
    water_mask_path : Path | str | None
        Path or url to water mask file. If none, will retrieve using `get_water_mask` which utilizes the Glad landcover
        dataset.
    mgrs_tile_id : str
        MGRS tile id of the tile to process.
    apply_water_mask : bool
        If True, will read and resample the water mask to the MGRS tile id. If False, will not do any preprocessing and
        return None.
    dst_dir : Path
        Directory to save the water mask.
    overwrite : bool, optional
        If True, will overwrite the water mask if it already exists. If False, will not overwrite the water mask if it
        already exists.
    buffer_size_pixel : int, optional
        How many additional pixels to buffer around the MGRS tile, by default 5

    Returns
    -------
    Path | None
        Path to the water mask on filesystem if `apply_water_mask` is True, otherwise None.


    Raises
    ------
    FileNotFoundError
        When water mask doesn't begin with http or s3, and doesn't exist on filesystem.
    ValueError
        When water mask indicated by `water_mask_path` doesn't contain the MGRS tile.
    """
    # This path will be used if we don't have a local water mask path or url
    out_water_mask_path = dst_dir / f'{mgrs_tile_id}_water_mask.tif'
    if not apply_water_mask:
        out_water_mask_path = None
    elif water_mask_path is None and apply_water_mask:
        if overwrite or not Path(out_water_mask_path).exists():
            _ = get_water_mask(mgrs_tile_id, out_water_mask_path, overwrite=False)
    elif isinstance(water_mask_path, str | Path) and apply_water_mask:
        if not str(water_mask_path).startswith('http') or not str(water_mask_path).startswith('s3'):
            if not water_mask_path.exists():
                raise FileNotFoundError(f'Water mask file does not exist: {water_mask_path}')
        with rasterio.open(water_mask_path) as src:
            wm_profile = src.profile
        bounds_wm = get_array_bounds(wm_profile['height'], wm_profile['width'], wm_profile['transform'])

        p_mgrs = get_mgrs_profile(mgrs_tile_id)

        mgrs_tile_bounds_utm = get_array_bounds(p_mgrs['height'], p_mgrs['width'], p_mgrs['transform'])
        bounds_wm_utm = transform_bounds_into_crs(wm_profile['crs'], p_mgrs['crs'], *bounds_wm)
        wm_geo_utm = box(*bounds_wm_utm)

        buffer_res = buffer_size_pixel * p_mgrs['transform'][0]
        mgrs_tile_geo_utm = box(*mgrs_tile_bounds_utm)

        if not wm_geo_utm.contains(mgrs_tile_geo_utm):
            raise ValueError('Water mask does not contain the mgrs tile')

        mgrs_tile_bounds_utm_buffered = mgrs_tile_geo_utm.buffer(buffer_res).bounds
        X_wm_window, p_wm_window = read_raster_from_window(
            water_mask_path,
            mgrs_tile_bounds_utm_buffered,
            window_crs=p_mgrs['crs'],
            res_buffer=0,
        )
        X_wm_window = X_wm_window[0, ...]

        X_wm_mgrs, p_wm_mgrs = reproject_arr_to_match_profile(X_wm_window, p_wm_window, p_mgrs)
        X_wm_mgrs = X_wm_mgrs[0, ...]

        p_wm_mgrs['count'] = 1
        p_wm_mgrs['dtype'] = np.uint8
        with rasterio.open(out_water_mask_path, 'w', **p_wm_mgrs) as dst:
            dst.write(X_wm_mgrs, 1)
    return out_water_mask_path
