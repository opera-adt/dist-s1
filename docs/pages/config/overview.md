# Configuration Overview

The Hybrid Science Data System (HySDS) interface (i.e. `dist-s1 run_sas`) assumes all the requisite OPERA RTC-S1 data is localized to the runtime environment and a runconfig.yml has been provided with relevant local paths. To run this entrypoint is simply:
```bash
dist-s1 run_sas --run_config_path run_config.yml
```
For a complete example (in which a `run_config` is generated in the process) use:
```bash
dist-s1 run_sas_prep --mgrs_tile_id '11SLT' \
    --post_date '2025-01-21' \
    --track_number 71 \
    --dst_dir '../../notebooks/los-angeles' \
    --memory_strategy 'high' \
    --low_confidence_alert_threshold 3.5 \
    --high_confidence_alert_threshold 5.5 \
    --apply_water_mask true \
    --device 'cpu' \
    --product_dst_dir '../../notebooks/los-angeles' \
    --model_source 'transformer_original' \
    --use_date_encoding false \
    --model_dtype 'float32' \
    --n_workers_for_norm_param_estimation 4 \
    --batch_size_for_norm_param_estimation 32 \
    --stride_for_norm_param_estimation 4 \
    --algo_config_path algo_config.yml \
    --run_config_path run_config.yml && \
dist-s1 run_sas --run_config_path run_config.yml
```
Sample yml configs can be found here:

- [run_config.yml](https://github.com/opera-adt/dist-s1/blob/dev/examples/no_confirmation/_run_config.yml)
- [algo_config.yml](https://github.com/opera-adt/dist-s1/blob/dev/examples/no_confirmation/_algo_config.yml)

**Note**: There is a field `algo_config_path` in `run_config.yml` so that the `algo_config.yml` is also loaded. The algorithm config initialized with such yml parameters within `RunConfigData` data model. We have provided tables to the relevant fields for the configuration ymls here:

- [RunConfig Parameter Table](runconfig.md)
- [AlgoConfig Parameter Table](algoconfig.md)

A RunConfig and AlgoConfig expose all the different parameters that can be specified for the generation of a `DIST-S1` product.
The parameters are written as gdal tags to the final `DIST-S1` product for provenance and can be read through `rasterio` in any of the final COG layers:
```python
import rasterio

with rasterio.open(<PATH_TO_A_DIST_S1_LAYER>) as ds:
    tags = ds.tags()
```
