from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env

import dist_s1
from dist_s1.constants import (
    BASE_DATE_FOR_CONFIRMATION,
    DISTLABEL2VAL,
    DIST_STATUS_CMAP,
    TIF_LAYERS,
    TIF_LAYER_DTYPES,
    TIF_LAYER_NODATA_VALUES,
)
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds
from dist_s1.water_mask import apply_water_mask


def generate_dist_indicator(
    dist_status_arr: np.ndarray, ind_val: int = 1, dtype: np.dtype = np.uint8, dst_nodata_value: int | float = 255
) -> np.ndarray:
    X_count = np.zeros_like(dist_status_arr, dtype=dtype)
    X_count[dist_status_arr == 255] = dst_nodata_value
    X_count[(dist_status_arr > 0) & (dist_status_arr < 255)] = ind_val
    return X_count


def update_profile(src_profile: dict, dtype: np.dtype, nodata_value: int | float) -> dict:
    profile = src_profile.copy()
    profile['dtype'] = dtype
    profile['nodata'] = nodata_value
    return profile


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
            opera_ids = [opera_id.replace('_VV.tif', '').replace('_HH.tif', '') for opera_id in opera_ids]
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


def generate_default_dist_arrs_from_metric_and_alert_status(
    X_metric: np.ndarray,
    X_status_arr: np.ndarray,
    acq_date: pd.Timestamp,
) -> dict[np.ndarray]:
    # GEN-DIST-COUNT
    X_count = generate_dist_indicator(
        X_status_arr, dtype=np.uint8, dst_nodata_value=TIF_LAYER_NODATA_VALUES['GEN-DIST-COUNT']
    )

    # GEN-DIST-PERC
    X_perc = generate_dist_indicator(
        X_status_arr, ind_val=100, dtype=np.uint8, dst_nodata_value=TIF_LAYER_NODATA_VALUES['GEN-DIST-PERC']
    )

    # GEN-DIST-DUR
    X_dur = generate_dist_indicator(
        X_status_arr, dtype=np.int16, dst_nodata_value=TIF_LAYER_NODATA_VALUES['GEN-DIST-DUR']
    )

    # GEN-DIST-DATE - everything is pd.Timestamp
    date_encoded = (acq_date.to_pydatetime() - BASE_DATE_FOR_CONFIRMATION).days
    X_date = generate_dist_indicator(
        X_status_arr, dtype=np.int16, dst_nodata_value=TIF_LAYER_NODATA_VALUES['GEN-DIST-DATE'], ind_val=date_encoded
    )

    # GEN-DIST-LAST-DATE - last date of valid observation
    X_last_date = np.full_like(X_status_arr, -1, dtype=np.int16)
    X_last_date[X_status_arr != 255] = date_encoded

    # GEN-DIST-CONF
    X_conf = np.full_like(X_metric, TIF_LAYER_NODATA_VALUES['GEN-DIST-CONF'], dtype=np.float32)
    dist_labels = [DISTLABEL2VAL[key] for key in ['first_low_conf_disturbance', 'first_high_conf_disturbance']]
    new_disturbed_mask = np.isin(X_status_arr, dist_labels)
    X_conf[new_disturbed_mask] = X_metric[new_disturbed_mask]

    valid_data_mask = ~np.isnan(X_metric)
    X_conf[~new_disturbed_mask & valid_data_mask] = 0

    # GEN-DIST-STATUS-ACQ
    X_status_acq = X_status_arr.copy()

    # GEN-METRIC-MAX
    X_metric_max = X_metric.copy()

    out_arr_dict = {
        'GEN-DIST-STATUS': X_status_arr.astype(TIF_LAYER_DTYPES['GEN-DIST-STATUS']),
        'GEN-METRIC': X_metric.astype(TIF_LAYER_DTYPES['GEN-METRIC']),
        'GEN-DIST-COUNT': X_count.astype(TIF_LAYER_DTYPES['GEN-DIST-COUNT']),
        'GEN-DIST-PERC': X_perc.astype(TIF_LAYER_DTYPES['GEN-DIST-PERC']),
        'GEN-DIST-DUR': X_dur.astype(TIF_LAYER_DTYPES['GEN-DIST-DUR']),
        'GEN-DIST-DATE': X_date.astype(TIF_LAYER_DTYPES['GEN-DIST-DATE']),
        'GEN-DIST-LAST-DATE': X_last_date.astype(TIF_LAYER_DTYPES['GEN-DIST-LAST-DATE']),
        'GEN-DIST-CONF': X_conf.astype(TIF_LAYER_DTYPES['GEN-DIST-CONF']),
        'GEN-DIST-STATUS-ACQ': X_status_acq.astype(TIF_LAYER_DTYPES['GEN-DIST-STATUS-ACQ']),
        'GEN-METRIC-MAX': X_metric_max.astype(TIF_LAYER_DTYPES['GEN-METRIC-MAX']),
    }
    return out_arr_dict


def get_product_tags(run_config: RunConfigData) -> dict:
    tags = run_config.get_public_attributes(include_algo_config_params=True)
    tags['version'] = dist_s1.__version__
    tags = update_tags_with_opera_ids(tags)
    tags = update_tag_types(tags)
    return tags


def package_disturbance_tifs_no_confirmation(run_config: RunConfigData) -> None:
    product_data = run_config.product_data_model_no_confirmation

    X_dist, p_dist = open_one_ds(run_config.final_unformatted_tif_paths['alert_status_path'])
    X_metric, p_metric = open_one_ds(run_config.final_unformatted_tif_paths['metric_status_path'])

    out_arr_dict = generate_default_dist_arrs_from_metric_and_alert_status(X_metric, X_dist, run_config.min_acq_date)
    cmap_dict = {'GEN-DIST-STATUS': DIST_STATUS_CMAP, 'GEN-DIST-STATUS-ACQ': DIST_STATUS_CMAP}
    cmap_dict.update({layer_name: None for layer_name in TIF_LAYERS if layer_name not in cmap_dict})

    # array, profile, path, colormap
    serialization_inputs = [
        (
            out_arr_dict[layer_name],  # array
            update_profile(p_dist, TIF_LAYER_DTYPES[layer_name], TIF_LAYER_NODATA_VALUES[layer_name]),  # profile
            product_data.layer_path_dict[layer_name],  # path
            cmap_dict[layer_name],  # colormap
        )
        for layer_name in TIF_LAYERS
    ]

    tags = get_product_tags(run_config)

    if run_config.apply_water_mask:
        serialization_inputs = [
            (apply_water_mask(arr, prof, run_config.water_mask_path), prof, path, cmap)
            for (arr, prof, path, cmap) in serialization_inputs
        ]

    for arr, prof, path, cmap in serialization_inputs:
        serialize_one_2d_ds(arr, prof, path, colormap=cmap, tags=tags, cog=True)


def generate_browse_image(product_data: DistS1ProductDirectory, water_mask_path: Path | str | None = None) -> None:
    with Env(GDAL_PAM_ENABLED='NO'):
        convert_geotiff_to_png(
            product_data.layer_path_dict['GEN-DIST-STATUS'],
            product_data.layer_path_dict['browse'],
            colormap=DIST_STATUS_CMAP,
            water_mask_path=water_mask_path,
        )
