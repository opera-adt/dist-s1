from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env

from dist_s1.constants import DIST_CMAP
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds


def check_water_mask(water_mask_profile: dict, disturbance_profile: dict) -> None:
    if water_mask_profile['crs'] != disturbance_profile['crs']:
        raise ValueError('Water mask and disturbance array CRS do not match')
    if water_mask_profile['transform'] != disturbance_profile['transform']:
        raise ValueError('Water mask and disturbance array transform do not match')
    if water_mask_profile['height'] != disturbance_profile['height']:
        raise ValueError('Water mask and disturbance array height do not match')
    if water_mask_profile['width'] != disturbance_profile['width']:
        raise ValueError('Water mask and disturbance array width do not match')
    return True


def apply_water_mask(band_src: np.ndarray, profile_src: dict, water_mask_path: Path | str | None = None) -> np.ndarray:
    X_wm, p_wm = open_one_ds(water_mask_path)
    check_water_mask(p_wm, profile_src)
    band_src[X_wm == 1] = profile_src['nodata']
    return band_src


def convert_geotiff_to_png(
    geotiff_path: Path,
    out_png_path: Path,
    output_height: int = None,
    output_width: int = None,
    colormap: dict | None = None,
    water_mask_path: Path | str | None = None,
) -> None:
    with rasterio.open(geotiff_path) as ds:
        band = ds.read(1)
        profile_src = ds.profile
        if colormap is None:
            colormap = ds.colormap(1) if ds.count == 1 else None

        if water_mask_path is not None:
            band = apply_water_mask(band, profile_src, water_mask_path)

        output_height = output_height or band.shape[0]
        output_width = output_width or band.shape[1]

        if (output_height, output_width) != band.shape:
            band = ds.read(1, out_shape=(output_height, output_width), resampling=Resampling.nearest)

        band = band.astype(np.float32)
        band = (255 * (band - band.min()) / (band.max() - band.min())).astype(np.uint8)

        profile = {'driver': 'PNG', 'height': output_height, 'width': output_width, 'count': 1, 'dtype': band.dtype}
        # Dummy crs and transform to avoid warnings
        profile.update({'crs': profile_src['crs'], 'transform': profile_src['transform']})

        serialize_one_2d_ds(band, profile, out_png_path, colormap=colormap)


def package_disturbance_tifs(run_config: RunConfigData) -> None:
    X_dist, p_dist = open_one_ds(run_config.final_unformatted_tif_paths['alert_status_path'])
    X_dist_delta0, p_dist_delta0 = open_one_ds(run_config.final_unformatted_tif_paths['alert_delta0_path'])
    X_metric, p_metric = open_one_ds(run_config.final_unformatted_tif_paths['metric_status_path'])

    if run_config.apply_water_mask:
        X_dist = apply_water_mask(X_dist, p_dist, run_config.water_mask_path)
        X_dist_delta0 = apply_water_mask(X_dist_delta0, p_dist_delta0, run_config.water_mask_path)
        X_metric = apply_water_mask(X_metric, p_metric, run_config.water_mask_path)

    product_data = run_config.product_data_model

    serialize_one_2d_ds(X_dist, p_dist, product_data.layer_path_dict['DIST-GEN-STATUS'], colormap=DIST_CMAP)
    serialize_one_2d_ds(
        X_dist_delta0, p_dist_delta0, product_data.layer_path_dict['DIST-GEN-STATUS-ACQ'], colormap=DIST_CMAP
    )
    serialize_one_2d_ds(X_metric, p_metric, product_data.layer_path_dict['GEN-METRIC'])


def generate_browse_image(run_config: RunConfigData) -> None:
    product_data = run_config.product_data_model
    with Env(GDAL_PAM_ENABLED='NO'):
        convert_geotiff_to_png(
            run_config.final_unformatted_tif_paths['alert_status_path'],
            product_data.layer_path_dict['browse'],
            colormap=DIST_CMAP,
            water_mask_path=run_config.water_mask_path,
        )
