## AlgoConfigData

| Attribute | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `device` | `str` | best | No | Device to use for model inference. `best` will use the best available device. |
| `memory_strategy` | `str | None` | high | No | Memory strategy to use for model inference. `high` will use more memory, `low` will use less. Utilizing more memory will improve runtime performance. |
| `tqdm_enabled` | `bool` | True | No | Whether to enable tqdm progress bars. |
| `n_workers_for_norm_param_estimation` | `int` | 8 | No | Number of workers for norm parameter estimation from the baseline. Utilizing more workers will improve runtime performance and utilize more memory. Does not work with model compilation or MPS/GPU devices. |
| `batch_size_for_norm_param_estimation` | `int` | 32 | No | Batch size for norm parameter estimation from the baseline. Utilizing a larger batch size will improve runtime performance and utilize more memory. |
| `stride_for_norm_param_estimation` | `int` | 16 | No | Stride for norm parameter estimation from the baseline. Utilizing a larger stride will improve metric accuracy and utilize more memory.Memory usage scales inverse quadratically with stride. That is, If stride=16 consumes N bytes of memory, then stride=4 consumes 16N bytes of memory. |
| `apply_logit_to_inputs` | `bool` | True | No | Whether to apply logit transform to the input data. |
| `n_workers_for_despeckling` | `int` | 8 | No | Number of workers for despeckling. Utilizing more workers will improve runtime performance and utilize more memory. |
| `lookback_strategy` | `str` | multi_window | No | Lookback strategy to use for data curation of the baseline. `multi_window` will use a multi-window lookback strategy and is default for OEPRA DIST-S1, `immediate_lookback` will use an immediate lookback strategy using acquisitions preceding the post-date. `immediate_lookback` is not supported yet. |
| `post_date_buffer_days` | `int` | 1 | No | Buffer days around post-date for data collection to create acqusition image to compare baseline to. |
| `model_compilation` | `bool` | False | No | Whether to compile the model for CPU or GPU. False, use the model as is. True, load the model and compile for CPU or GPU optimizations. |
| `max_pre_imgs_per_burst_mw` | `tuple[int, ...] | None` | None | Yes | Max number of pre-images per burst within each window. If `None`, the value will be calculated based on the model context length and the number of anniversaries. Specifically, the value will be context_length // n_anniversaries with remainder added to the first window. |
| `delta_lookback_days_mw` | `tuple[int, ...] | None` | None | Yes | Delta lookback days for each window relative to post-image acquisition date. If `None`, the value will be calculated based on the number of anniversaries (default is 3). |
| `low_confidence_alert_threshold` | `float` | 3.5 | No | Low confidence alert threshold for detecting disturbance between baseline and post-image. |
| `high_confidence_alert_threshold` | `float` | 5.5 | No | High confidence alert threshold for detecting disturbance between baseline and post-image. |
| `no_day_limit` | `int` | 30 | No | Number of days to limit confirmation process logic to. Confirmation must occur within first observance of disturbance and `no_day_limit` days after first disturbance. |
| `exclude_consecutive_no_dist` | `int` | True | No | Boolean activation of consecutive no disturbance tracking during confirmation. True will apply this logic: after 2 no disturbances within product sequence, the disturbance must finish or be reset. False will not apply this logic. |
| `percent_reset_thresh` | `int` | 10 | No | Precentage number threshold to reset disturbance. Values below `percent_reset_thresh` will reset disturbance. |
| `no_count_reset_thresh` | `int` | 7 | No | If the number of non-disturbed observations `prevnocount` is above `nocount_reset_thresh` disturbance will reset. |
| `max_obs_num_year` | `int` | 253 | No | Max observation number per year. If observations exceeds this number, then the confirmation must conclude and be reset. |
| `confidence_upper_lim` | `int` | 32000 | No | Confidence upper limit for confirmation. Confidence is an accumulation of the metric over time. |
| `confirmation_confidence_threshold` | `float` | 31.5 | No | This is the threshold for the confirmation process to determine if a disturbance is confirmed. |
| `metric_value_upper_lim` | `float` | 100.0 | No | Metric upper limit set during confirmation |
| `model_source` | `str` | transformer_optimized | No | Model source. If `external`, use externally supplied paths for weights and config. Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models. |
| `model_cfg_path` | `Path | str | None` | None | Yes | Path to model config file. If `external`, use externally supplied path. Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models. |
| `model_wts_path` | `Path | str | None` | None | Yes | Path to model weights file. If `external`, use externally supplied path. Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models. |
| `apply_despeckling` | `bool` | True | No | Whether to apply despeckling to the input data. |
| `interpolation_method` | `str` | bilinear | No | Interpolation method to use for despeckling. `nearest` will use nearest neighbor interpolation, `bilinear` will use bilinear interpolation, and `none` will not apply despeckling. |
| `model_dtype` | `str` | float32 | No | Data type for model inference. Note: bfloat16 is only supported on GPU devices. |
| `use_date_encoding` | `bool` | False | No | Whether to use acquisition date encoding in model application (currently not supported) |
| `n_anniversaries_for_mw` | `int` | 3 | No | Number of anniversaries to use for multi-window |
