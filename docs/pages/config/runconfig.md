## RunConfigData

| Attribute | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `pre_rtc_copol` | `list[Path | str]` | No default | Yes | List of paths to pre-rtc copolarization data. |
| `pre_rtc_crosspol` | `list[Path | str]` | No default | Yes | List of paths to pre-rtc crosspolarization data. |
| `post_rtc_copol` | `list[Path | str]` | No default | Yes | List of paths to post-rtc copolarization data. |
| `post_rtc_crosspol` | `list[Path | str]` | No default | Yes | List of paths to post-rtc crosspolarization data. |
| `prior_dist_s1_product` | `DistS1ProductDirectory | None` | None | Yes | Path to prior DIST-S1 product. Can accept str, Path, or DistS1ProductDirectory. If None, no prior product is used and confirmation is not performed. |
| `mgrs_tile_id` | `str` | No default | Yes | MGRS tile ID. Required to kick-off disturbance processing. |
| `dst_dir` | `Path | str` | `out` | No | No description available |
| `input_data_dir` | `Path | str | None` | None | Yes | Input data directory. If None, defaults to dst_dir. |
| `water_mask_path` | `Path | str | None` | None | Yes | Path to water mask. If None, no water mask is used. |
| `apply_water_mask` | `bool` | True | No | Whether to apply water mask to the input data. If True, water mask is applied to the input data. If no water mask path is provided, the tiles to generate the water mask are localized and formatted for use. |
| `check_input_paths` | `bool` | True | No | Whether to check if the input paths exist. If True, the input paths are checked. Used during testing. |
| `product_dst_dir` | `Path | str | None` | None | Yes | Path to product directory. If None, defaults to dst_dir. |
| `bucket` | `str | None` | None | Yes | Bucket to use for product storage. If None, no bucket is used. |
| `bucket_prefix` | `str | None` | None | Yes | Bucket prefix to use for product storage. If None, no bucket prefix is used. |
| `algo_config` | `AlgoConfigData` | No default | Yes | Algorithm configuration parameters. |
| `algo_config_path` | `Path | str | None` | None | Yes | Path to external algorithm config file. If None, no external algorithm config is used. |
