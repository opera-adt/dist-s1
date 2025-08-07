dist-s1 run_sas_prep --mgrs_tile_id '11SLT' \
    --post_date '2025-01-21' \
    --track_number 71 \
    --dst_dir '../notebooks/los-angeles' \
    --memory_strategy 'high' \
    --low_confidence_alert_threshold 3.5 \
    --high_confidence_alert_threshold 5.5 \
    --apply_water_mask true \
    --device 'cpu' \
    --product_dst_dir '../notebooks/los-angeles' \
    --model_source 'transformer_original' \
    --use_date_encoding false \
    --model_dtype 'float32' \
    --n_workers_for_norm_param_estimation 4 \
    --batch_size_for_norm_param_estimation 32 \
    --stride_for_norm_param_estimation 4 \
    --algo_config_path algo_config.yml \
    --run_config_path run_config.yml && \
dist-s1 run_sas --run_config_path run_config.yml 