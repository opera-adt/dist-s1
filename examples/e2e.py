from pathlib import Path

from dist_s1 import run_dist_s1_workflow


# Parameters for DIST-S1 submission
mgrs_tile_id = '11SLT'  # MGRS tile ID
post_date = '2025-01-21'  # date of recent pass of Sentinel-1
track_number = 71  # Sentinel-1 track number
dst_dir = Path('../notebooks/los-angeles')  # directory to save the intermediate and output DIST-S1 product
memory_strategy = 'high'  # can be high or low depending on memory availability/GPU setup
product_dst_dir = Path('../notebooks/los-angeles')  # directory to save the final products
apply_water_mask = True  # apply water mask to the data
n_lookbacks = 3  # number of lookbacks to use for change confirmation within SAS
water_mask_path = Path('../notebooks/los-angeles/water_mask.tif')  # path to an existing water mask file


# Run the workflow
run_dist_s1_workflow(
    mgrs_tile_id,
    post_date,
    track_number,
    post_date_buffer_days=1,
    dst_dir=dst_dir,
    memory_strategy=memory_strategy,
    product_dst_dir=product_dst_dir,
    apply_water_mask=apply_water_mask,
    n_lookbacks=n_lookbacks,
    water_mask_path=water_mask_path,
)
