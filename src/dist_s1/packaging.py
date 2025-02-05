from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from dist_s1.constants import DIST_CMAP
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds


def convert_geotiff_to_png(
    geotiff_path: Path,
    out_png_path: Path,
    output_height: int = None,
    output_width: int = None,
    colormap: dict | None = None,
) -> None:
    with rasterio.open(geotiff_path) as ds:
        band = ds.read(1)
        profile_src = ds.profile
        if colormap is None:
            colormap = ds.colormap(1) if ds.count == 1 else None

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

    product_data = run_config.product_data_model

    serialize_one_2d_ds(X_dist, p_dist, product_data.layer_path_dict['DIST-GEN-STATUS'], colormap=DIST_CMAP)
    serialize_one_2d_ds(
        X_dist_delta0, p_dist_delta0, product_data.layer_path_dict['DIST-GEN-STATUS-ACQ'], colormap=DIST_CMAP
    )
    serialize_one_2d_ds(X_metric, p_metric, product_data.layer_path_dict['GEN-METRIC'])


def generate_browse_image(run_config: RunConfigData) -> None:
    product_data = run_config.product_data_model
    convert_geotiff_to_png(
        run_config.final_unformatted_tif_paths['alert_status_path'],
        product_data.layer_path_dict['browse'],
        colormap=DIST_CMAP,
    )
