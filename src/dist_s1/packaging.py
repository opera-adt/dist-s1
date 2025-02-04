# import numpy as np
# from pathlib import Path

# from dist_s1.constants import DIST_CMAP
# from dist_s1.rio_tools import convert_geotiff_to_png, serialize_one_2d_ds


# def generate_disturbance_map(metric: np.ndarray, metric_p: dict, dist_out_path: str, metric_name: str) -> None:
#     dist_arr = np.zeros(metric.shape[-2:], dtype=np.uint8)
#     nan_mask = np.isnan(metric[0, :, :])
#     dist_arr[nan_mask] = 255  # Set NaN values to 255 (nodata)

#     if metric_name == 'transformer':
#         dist_arr[metric[0, :, :] >= 2.5] = 1
#         dist_arr[metric[0, :, :] >= 4.5] = 4
#     elif metric_name == 'cusum_prob_max':
#         dist_arr[metric[0, :, :] >= 0.9] = 1
#         dist_arr[metric[0, :, :] >= 0.95] = 4

#     p_dist = metric_p.copy()
#     p_dist['dtype'] = 'uint8'
#     p_dist['nodata'] = 255

#     serialize_one_2d_ds(dist_arr, p_dist, dist_out_path, colormap=DIST_CMAP)
#     dest_png_file = Path(dist_out_path).with_name(f"browse_{Path(dist_out_path).stem}.png")

#     convert_geotiff_to_png(
#         src_geotiff_filename=dist_out_path,
#         dest_png_filename=dest_png_file
#     )
