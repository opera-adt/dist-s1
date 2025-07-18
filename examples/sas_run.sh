dist-s1 run_sas_prep --mgrs_tile_id '11SLT' \
    --post_date '2025-01-21' \
    --track_number 71 \
    --dst_dir '../notebooks/los-angeles' \
    --memory_strategy 'high' \
    --moderate_confidence_threshold 3.5 \
    --high_confidence_threshold 5.5 \
    --post_date_buffer_days 1 \
    --apply_water_mask true \
    --product_dst_dir '../notebooks/los-angeles' \
    --algo_config_path algo_config.yml \
    --run_config_path run_config.yml
dist-s1 run_sas --run_config_path run_config.yml 