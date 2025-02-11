from pathlib import Path

import numpy as np
import rasterio
from dist_s1_enumerator.mgrs_burst_data import get_mgrs_tile_table_by_ids
from rasterio.crs import CRS
from rasterio.profiles import default_gtiff_profile
from rasterio.transform import from_origin
from shapely import from_wkt


def check_profiles_match(prof_0: dict, prof_1: dict) -> None:
    prof_0_keys = set(prof_0.keys())
    prof_1_keys = set(prof_1.keys())
    if prof_0_keys != prof_1_keys:
        raise ValueError('Profiles have different keys')

    for key in prof_0_keys:
        if key == 'nodata':
            if np.isnan(prof_0['nodata']):
                if not np.isnan(prof_1['nodata']):
                    raise ValueError('Nodata values do not match')
            elif prof_0['nodata'] != prof_1['nodata']:
                raise ValueError('Nodata values do not match')
        elif prof_0[key] != prof_1[key]:
            raise ValueError(f'Profile key {key} does not match')
    return True


def open_one_ds(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        X = ds.read(1)
        p = ds.profile
    return X, p


def open_one_profile(path: Path) -> dict:
    with rasterio.open(path) as ds:
        p = ds.profile
    return p


def serialize_one_2d_ds(
    arr: np.ndarray, p: dict, out_path: Path, colormap: dict | None = None, tags: dict | None = None
) -> Path:
    with rasterio.open(out_path, 'w', **p) as ds:
        ds.write(arr, 1)
        if colormap is not None:
            ds.write_colormap(1, colormap)
        if tags is not None:
            ds.update_tags(**tags)
    return out_path


def get_mgrs_bounds(mgrs_tile_id: str) -> tuple[float]:
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    utm_wkt = df_mgrs['utm_wkt'].tolist()[0]
    utm_geo = from_wkt(utm_wkt)
    return utm_geo.bounds


def get_mgrs_utm_epsg(mgrs_tile_id: str) -> tuple[float]:
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    utm_epsg = df_mgrs['utm_epsg'].tolist()[0]
    utm_crs = CRS.from_epsg(utm_epsg)
    return utm_crs


def get_mgrs_profile(mgrs_tile_id: str, count: int = 1, dtype: np.dtype = np.float32, nodata: float = np.nan) -> dict:
    profile = default_gtiff_profile.copy()

    xmin, ymin, xmax, ymax = get_mgrs_bounds(mgrs_tile_id)
    transform = from_origin(xmin, ymax, 30, 30)
    utm_crs = get_mgrs_utm_epsg(mgrs_tile_id)

    profile['transform'] = transform
    profile['crs'] = utm_crs
    profile['count'] = count
    profile['dtype'] = dtype
    profile['nodata'] = nodata
    profile['width'] = 3660
    profile['height'] = 3660

    dims = [(xmax - xmin) / 30.0, (ymax - ymin) / 30.0]
    if all([3660.0 != diff for diff in dims]):
        dim_str = ', '.join(list(map(str, dims)))
        raise ValueError(f'The expected dimensions of 3660 disagree with the mgrs table: {dim_str}')
    return profile
