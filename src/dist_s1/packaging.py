from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env

import dist_s1
from dist_s1.constants import DIST_CMAP
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds
from dist_s1.water_mask import check_water_mask_profile


def apply_water_mask(band_src: np.ndarray, profile_src: dict, water_mask_path: Path | str | None = None) -> np.ndarray:
    X_wm, p_wm = open_one_ds(water_mask_path)
    check_water_mask_profile(p_wm, profile_src)
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


def update_tags_with_opera_ids(tags: dict) -> dict:
    input_keys = ['pre_rtc_copol', 'post_rtc_copol', 'post_rtc_crosspol', 'pre_rtc_crosspol']
    for key in input_keys:
        value = tags.pop(key)
        if 'crosspol' in key:
            continue
        else:
            opera_ids = [path.name for path in value]
            opera_ids = [opera_id.replace('_VV.tif', '') for opera_id in opera_ids]
            # pre_rtc_copol -> pre_opera_ids, etc.
            new_key = key.replace('_copol', '_opera_ids')
            tags[new_key] = opera_ids
    return tags


def update_tag_types(tags: dict) -> dict:
    for key, value in tags.items():
        if isinstance(value, Path):
            tags[key] = str(value)
        elif isinstance(value, list | tuple):
            tags[key] = ','.join(list(map(str, value)))
    return tags


def package_disturbance_tifs(run_config: RunConfigData) -> None:
    X_dist, p_dist = open_one_ds(run_config.final_unformatted_tif_paths['alert_status_path'])
    X_dist_delta0, p_dist_delta0 = open_one_ds(run_config.final_unformatted_tif_paths['alert_delta0_path'])
    X_metric, p_metric = open_one_ds(run_config.final_unformatted_tif_paths['metric_status_path'])

    tags = run_config.get_public_attributes()
    tags['version'] = dist_s1.__version__
    tags = update_tags_with_opera_ids(tags)
    tags = update_tag_types(tags)

    if run_config.apply_water_mask:
        X_dist = apply_water_mask(X_dist, p_dist, run_config.water_mask_path)
        X_dist_delta0 = apply_water_mask(X_dist_delta0, p_dist_delta0, run_config.water_mask_path)
        X_metric = apply_water_mask(X_metric, p_metric, run_config.water_mask_path)

    product_data = run_config.product_data_model

    serialize_one_2d_ds(X_dist, p_dist, product_data.layer_path_dict['DIST-GEN-STATUS'], colormap=DIST_CMAP, tags=tags)
    serialize_one_2d_ds(
        X_dist_delta0,
        p_dist_delta0,
        product_data.layer_path_dict['DIST-GEN-STATUS-ACQ'],
        colormap=DIST_CMAP,
        tags=tags,
    )
    serialize_one_2d_ds(X_metric, p_metric, product_data.layer_path_dict['GEN-METRIC'], colormap=DIST_CMAP, tags=tags)


def generate_browse_image(run_config: RunConfigData) -> None:
    product_data = run_config.product_data_model
    with Env(GDAL_PAM_ENABLED='NO'):
        convert_geotiff_to_png(
            run_config.final_unformatted_tif_paths['alert_status_path'],
            product_data.layer_path_dict['browse'],
            colormap=DIST_CMAP,
            water_mask_path=run_config.water_mask_path,
        )
