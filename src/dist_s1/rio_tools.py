from pathlib import Path

import numpy as np
import rasterio


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


def serialize_one_ds(arr: np.ndarray, p: dict, out_path: Path) -> Path:
    with rasterio.open(out_path, 'w', **p) as ds:
        ds.write(arr, 1)
    return out_path
