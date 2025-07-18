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
    --model_source 'transformer_original' \
    --use_date_encoding true \
    --model_dtype 'float16' \
    --n_workers_for_norm_param_estimation 4 \
    --batch_size_for_norm_param_estimation 32 \
    --stride_for_norm_param_estimation 8 \
    --algo_config_path algo_config.yml \
    --run_config_path run_config.yml && \
dist-s1 run_sas --run_config_path run_config.yml 