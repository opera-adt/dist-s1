from pathlib import Path

import numpy as np
import rasterio
from dist_s1_enumerator.mgrs_burst_data import get_mgrs_tile_table_by_ids
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.profiles import default_gtiff_profile
from rasterio.transform import from_origin
from shapely import LineString, MultiPolygon, from_wkt, union_all
from shapely.affinity import translate


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
    arr: np.ndarray, p: dict, out_path: Path, colormap: dict | None = None, tags: dict | None = None, cog: bool = False
) -> Path:
    p_out = p.copy()
    if cog:
        # Unsupported cog fields
        p_out.pop('tiled', None)
        p_out.pop('interleave', None)
        p_out.pop('blockxsize', None)
        p_out.pop('blockysize', None)
        predictor = 3 if np.issubdtype(p_out['dtype'], np.floating) else 2
        p_out.update(
            {
                'driver': 'COG',
                'compress': 'ZSTD',
                'zlevel': 9,
                'predictor': predictor,
                'blocksize': 512,
                'resampling': Resampling.average,
                'BIGTIFF': 'IF_SAFER',
            }
        )
    with rasterio.open(out_path, 'w', **p_out) as ds:
        ds.write(arr, 1)
        if colormap is not None:
            ds.write_colormap(1, colormap)
        if tags is not None:
            ds.update_tags(**tags)
    return out_path


def get_mgrs_antimeridian_crossing(mgrs_tile_id: str) -> bool:
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    if df_mgrs.shape[0] != 1:
        raise ValueError(f'The MGRS tile {mgrs_tile_id} has multiple entries in the MGRS tile table')
    antimeridian_crossing = isinstance(df_mgrs.geometry.iloc[0], MultiPolygon)
    return antimeridian_crossing


def get_mgrs_bounds_in_utm(mgrs_tile_id: str) -> tuple[float]:
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    if df_mgrs.shape[0] != 1:
        raise ValueError(f'The MGRS tile {mgrs_tile_id} has multiple entries in the MGRS tile table')
    utm_wkt = df_mgrs['utm_wkt'].tolist()[0]
    utm_geo = from_wkt(utm_wkt)
    return utm_geo.bounds


def get_mgrs_bounds_in_4326(mgrs_tile_id: str) -> tuple[float]:
    """Get the bounds of the MGRS tile in 4326.

    If the MGRS tile crosses the antimeridian, the bounds are adjusted as Polygon within -181, -179.
    """
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    if df_mgrs.shape[0] != 1:
        raise ValueError(f'The MGRS tile {mgrs_tile_id} has multiple entries in the MGRS tile table')
    mgrs_geo = df_mgrs.geometry.iloc[0]
    antimeridian_crossing = isinstance(mgrs_geo, MultiPolygon)
    if antimeridian_crossing:
        antimeridian = LineString(coordinates=((-180, 90), (-180, -90))).buffer(0.01)
        mgrs_geos_neg = [geo if geo.intersects(antimeridian) else translate(geo, xoff=-360) for geo in mgrs_geo.geoms]
        mgrs_poly = union_all(mgrs_geos_neg)
    else:
        mgrs_poly = mgrs_geo
    return mgrs_poly.bounds


def get_mgrs_utm_epsg(mgrs_tile_id: str) -> tuple[float]:
    df_mgrs = get_mgrs_tile_table_by_ids([mgrs_tile_id])
    utm_epsg = df_mgrs['utm_epsg'].tolist()[0]
    utm_crs = CRS.from_epsg(utm_epsg)
    return utm_crs


def get_mgrs_profile(
    mgrs_tile_id: str,
    count: int = 1,
    dtype: np.dtype = np.float32,
    nodata: float = np.nan,
    better_compression: bool = True,
) -> dict:
    profile = default_gtiff_profile.copy()

    xmin, ymin, xmax, ymax = get_mgrs_bounds_in_utm(mgrs_tile_id)
    transform = from_origin(xmin, ymax, 30, 30)
    utm_crs = get_mgrs_utm_epsg(mgrs_tile_id)

    profile['transform'] = transform
    profile['crs'] = utm_crs
    profile['count'] = count
    profile['dtype'] = dtype
    profile['nodata'] = nodata
    profile['width'] = 3660
    profile['height'] = 3660
    # Better default compression
    if better_compression:
        profile['blockxsize'] = 512
        profile['blockysize'] = 512
        profile['BIGTIFF'] = 'IF_SAFER'
        profile['tiled'] = True
        profile['zlevel'] = 9
        profile['interleave'] = 'band'
        profile['compress'] = 'zstd'

    dims = [(xmax - xmin) / 30.0, (ymax - ymin) / 30.0]
    if all([3660.0 != diff for diff in dims]):
        dim_str = ', '.join(list(map(str, dims)))
        raise ValueError(f'The expected dimensions of 3660 disagree with the mgrs table: {dim_str}')
    return profile
