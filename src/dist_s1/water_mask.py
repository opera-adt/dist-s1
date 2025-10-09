from pathlib import Path

import numpy as np
import rasterio
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from dem_stitcher.rio_window import read_raster_from_window
from rasterio.crs import CRS
from shapely.geometry import box
from tile_mate import get_raster_from_tiles

from dist_s1.rio_tools import get_mgrs_bounds_in_4326, get_mgrs_bounds_in_utm, get_mgrs_profile, open_one_ds


def apply_water_mask(band_src: np.ndarray, profile_src: dict, water_mask_path: Path | str | None = None) -> np.ndarray:
    X_wm, p_wm = open_one_ds(water_mask_path)
    check_water_mask_profile(p_wm, profile_src)
    band_src[X_wm == 1] = profile_src['nodata']
    return band_src


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
    if Path(out_path).exists() and not overwrite:
        return out_path
    mgrs_bounds_4326 = get_mgrs_bounds_in_4326(mgrs_tile_id)

    # The ocean mask is distance to land in km
    X_dist_to_land, p_dist = get_raster_from_tiles(mgrs_bounds_4326, tile_shortname='umd_ocean_mask')

    # open water classes
    water_labels = [2, 3, 4]  # These are pixels that are more than 1 km from land
    X_om = np.isin(X_dist_to_land[0, ...], water_labels).astype(np.uint8)

    profile_mgrs = get_mgrs_profile(mgrs_tile_id)
    X_om_r, p_om_r = reproject_arr_to_match_profile(X_om, p_dist, profile_mgrs, resampling='nearest')
    X_om_r = X_om_r[0, ...]

    p_new = p_om_r.copy()
    # Better compression for localized water mask
    p_new['blockxsize'] = 512
    p_new['blockysize'] = 512
    p_new['tiled'] = True

    with rasterio.open(out_path, 'w', **p_new) as dst:
        dst.write(X_om_r, 1)

    return out_path


def water_mask_control_flow(
    *,
    water_mask_path: Path | str | None,
    mgrs_tile_id: str,
    dst_dir: Path,
    overwrite: bool = True,
    buffer_size_pixel: int = 5,
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
        How many additional pixels to read around the water mask, size determined in pixels of water mask, by default 5

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
    # This path will be used if we don't have a local water mask path or url is provided
    out_water_mask_path = dst_dir / f'{mgrs_tile_id}_water_mask.tif'
    if water_mask_path is None:
        if overwrite or not Path(out_water_mask_path).exists():
            _ = get_water_mask(mgrs_tile_id, out_water_mask_path, overwrite=False)
    elif isinstance(water_mask_path, str | Path):
        if not str(water_mask_path).startswith('http') or not str(water_mask_path).startswith('s3'):
            if not Path(water_mask_path).exists():
                raise FileNotFoundError(f'Water mask file does not exist: {water_mask_path}')
        with rasterio.open(water_mask_path) as src:
            wm_profile = src.profile
            wm_geo = box(*src.bounds)

        if wm_profile['crs'] == CRS.from_epsg(4326):
            mgrs_bounds = get_mgrs_bounds_in_4326(mgrs_tile_id)
            mgrs_geo = box(*mgrs_bounds)
        else:
            mgrs_bounds = get_mgrs_bounds_in_utm(mgrs_tile_id)
            mgrs_geo = box(*mgrs_bounds)
        if not wm_geo.contains(mgrs_geo):
            raise ValueError('Water mask does not contain the mgrs tile')
        X_wm_window, p_wm_window = read_raster_from_window(
            water_mask_path,
            mgrs_bounds,
            window_crs=wm_profile['crs'],
            res_buffer=buffer_size_pixel,
        )
        X_wm_window = X_wm_window[0, ...]

        p_mgrs = get_mgrs_profile(mgrs_tile_id)
        X_wm_mgrs, p_wm_mgrs = reproject_arr_to_match_profile(X_wm_window, p_wm_window, p_mgrs)
        X_wm_mgrs = X_wm_mgrs[0, ...]

        p_wm_mgrs['count'] = 1
        p_wm_mgrs['dtype'] = np.uint8
        with rasterio.open(out_water_mask_path, 'w', **p_wm_mgrs) as dst:
            dst.write(X_wm_mgrs, 1)
    return out_water_mask_path
