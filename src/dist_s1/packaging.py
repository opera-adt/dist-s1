from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env

import dist_s1
from dist_s1.constants import BASE_DATE, DIST_CMAP
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds
from dist_s1.water_mask import apply_water_mask


def generate_count_disturbed_no_confirmation(
    dist_status_arr: np.ndarray, count_value: int = 1, dtype: np.dtype = np.uint8, nodata_value: int | float = 255
) -> np.ndarray:
    X_count = np.zeros_like(dist_status_arr, dtype=dtype)
    X_count[dist_status_arr == 255] = nodata_value
    X_count[dist_status_arr > 0] = count_value
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


def package_disturbance_tifs_no_confirmation(run_config: RunConfigData) -> None:
    X_dist, p_dist = open_one_ds(run_config.final_unformatted_tif_paths['alert_status_path'])
    X_metric, p_metric = open_one_ds(run_config.final_unformatted_tif_paths['metric_status_path'])

    # GEN-DIST-COUNT
    X_count = generate_count_disturbed_no_confirmation(X_dist, dtype=np.uint8, nodata_value=255)
    # GEN-DIST-PERC
    X_perc = generate_count_disturbed_no_confirmation(X_dist, count_value=100, dtype=np.uint8, nodata_value=255)
    # GEN-DIST-DUR
    X_dur = generate_count_disturbed_no_confirmation(X_dist, dtype=np.int16, nodata_value=-1)
    # GEN-DIST-DATE - everything is pd.Timestamp
    date_encoded = (run_config.min_acq_date.to_pydatetime() - BASE_DATE).days
    X_date = generate_count_disturbed_no_confirmation(X_dist, dtype=np.int16, nodata_value=-1, count_value=date_encoded)
    # GEN-DIST-LAST-DATE - last date of valid observation
    X_last_date = X_dur.copy()
    X_last_date[X_dist != 255] = date_encoded
    # GEN-DIST-CONF
    X_conf = X_metric.copy()
    X_conf[X_conf < 3.5] = 0
    # GEN-DIST-STATUS-ACQ
    X_status_acq = X_dist.copy()
    # GEN-METRIC-MAX
    X_metric_max = X_metric.copy()

    product_data = run_config.product_data_model
    serialization_inputs = [
        (X_dist, p_dist, product_data.layer_path_dict['GEN-DIST-STATUS'], DIST_CMAP),
        (X_metric, p_metric, product_data.layer_path_dict['GEN-METRIC'], None),
        (X_count, update_profile(p_dist, np.uint8, 255), product_data.layer_path_dict['GEN-DIST-COUNT'], None),
        (X_perc, update_profile(p_dist, np.uint8, 255), product_data.layer_path_dict['GEN-DIST-PERC'], None),
        (X_dur, update_profile(p_dist, np.int16, -1), product_data.layer_path_dict['GEN-DIST-DUR'], None),
        (X_date, update_profile(p_dist, np.int16, -1), product_data.layer_path_dict['GEN-DIST-DATE'], None),
        (X_last_date, update_profile(p_dist, np.int16, -1), product_data.layer_path_dict['GEN-DIST-LAST-DATE'], None),
        (X_conf, update_profile(p_metric, np.int16, -1), product_data.layer_path_dict['GEN-DIST-CONF'], None),
        (
            X_status_acq,
            update_profile(p_dist, np.uint8, 255),
            product_data.layer_path_dict['GEN-DIST-STATUS-ACQ'],
            DIST_CMAP,
        ),
        (
            X_metric_max,
            update_profile(p_metric, np.float32, np.nan),
            product_data.layer_path_dict['GEN-METRIC-MAX'],
            None,
        ),
    ]

    tags = run_config.get_public_attributes()
    tags['version'] = dist_s1.__version__
    tags = update_tags_with_opera_ids(tags)
    tags = update_tag_types(tags)

    if run_config.apply_water_mask:
        serialization_inputs = [
            (apply_water_mask(arr, prof, run_config.water_mask_path), prof, path, cmap)
            for (arr, prof, path, cmap) in serialization_inputs
        ]

    for arr, prof, path, cmap in serialization_inputs:
        serialize_one_2d_ds(arr, prof, path, colormap=cmap, tags=tags)


def package_conf_db_disturbance_tifs(run_config: RunConfigData) -> None:
    print('Packaging CONF DB disturbance tifs')
    # Map from field name in run_config to output key in product_data.layer_path_dict
    disturbance_layers = [
        {'key': 'dist_status_path', 'label': 'GEN-DIST-STATUS'},
        {'key': 'alert_delta0_path', 'label': 'GEN-DIST-STATUS-ACQ'},
        {'key': 'dist_max_path', 'label': 'GEN-METRIC-MAX'},
        {'key': 'dist_conf_path', 'label': 'GEN-DIST-CONF'},
        {'key': 'dist_date_path', 'label': 'GEN-DIST-DATE'},
        {'key': 'dist_count_path', 'label': 'GEN-DIST-COUNT'},
        {'key': 'dist_perc_path', 'label': 'GEN-DIST-PERC'},
        {'key': 'dist_dur_path', 'label': 'GEN-DIST-DUR'},
        {'key': 'dist_last_date_path', 'label': 'GEN-DIST-LAST-DATE'},
        {'key': 'metric_status_path', 'label': 'GEN-METRIC'},
    ]

    X_dict = {}
    p_dict = {}
    label_dict = {}

    for item in disturbance_layers:
        key = item['key']
        label = item['label']

        path = run_config.final_unformatted_tif_paths[key]
        X, p = open_one_ds(path)

        if run_config.apply_water_mask:
            X = apply_water_mask(X, p, run_config.water_mask_path)

        X_dict[key] = X
        p_dict[key] = p
        label_dict[key] = label

    tags = run_config.get_public_attributes()
    tags['version'] = dist_s1.__version__
    tags = update_tags_with_opera_ids(tags)
    tags = update_tag_types(tags)

    product_data = run_config.product_data_model

    for item in disturbance_layers:
        key = item['key']
        label = item['label']

        X = X_dict[key]
        p = p_dict[key]
        out_path = product_data.layer_path_dict[label]
        if 'STATUS' in label:
            colormap = DIST_CMAP
        else:
            colormap = None
        print('Exporting', out_path)

        serialize_one_2d_ds(X, p, out_path, colormap=colormap, tags=tags)


def generate_browse_image(run_config: RunConfigData) -> None:
    if not run_config.confirmation:
        final_tif2plot = 'alert_status_path'
    else:
        final_tif2plot = 'dist_status_path'
    product_data = run_config.product_data_model
    with Env(GDAL_PAM_ENABLED='NO'):
        convert_geotiff_to_png(
            run_config.final_unformatted_tif_paths[final_tif2plot],
            product_data.layer_path_dict['browse'],
            colormap=DIST_CMAP,
            water_mask_path=run_config.water_mask_path,
        )
