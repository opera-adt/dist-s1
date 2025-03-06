dist-s1 run \
    --mgrs_tile_id '11SLT' \
    --post_date '2025-01-21' \
    --track_number 71 \
    --dst_dir '../notebooks/los-angeles' \
    --memory_strategy 'low' \
    --device 'cpu' \
    --n_workers_for_norm_param_estimation 5 \
    --batch_size_for_despeckling 100