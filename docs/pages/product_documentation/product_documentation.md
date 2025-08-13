# DIST-S1 Product Documentation

This page provides comprehensive documentation for the DIST-S1 product layers and disturbance labels.

## Product Naming Specification

DIST-S1 products follow a standardized naming convention that encodes key metadata about the product. The `ProductNameData` model manages this naming scheme and provides validation capabilities.

### Product Name Format

Products follow this format:
```
OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{version}
```

### Example Product Name

```
OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1
```

### Token Description

| Token | Description | Example |
|-------|-------------|---------|
| `OPERA` | Fixed identifier for OPERA products | `OPERA` |
| `L3` | Product level (Level 3) | `L3` |
| `DIST-ALERT-S1` | Product type identifier | `DIST-ALERT-S1` |
| `T{mgrs_tile_id}` | MGRS tile identifier with 'T' prefix | `T10SGD` |
| `{acq_datetime}` | Acquisition datetime in ISO format | `20250102T015857Z` |
| `{proc_datetime}` | Processing datetime in ISO format | `20250806T145521Z` |
| `S1` | Sentinel-1 mission identifier | `S1` |
| `30` | Fixed resolution identifier | `30` |
| `v{version}` | Product version with 'v' prefix | `v0.1` |



## Product Structure

A DIST-S1 product is organized as a directory containing multiple Cloud-optimized GeoTIFF (COG) files, each representing different aspects of the DIST-S1 product. The product follows this directory structure:

```
<OPERA_ID>/
├── <OPERA_ID>_GEN-DIST-STATUS.tif
├── <OPERA_ID>_GEN-METRIC.tif
├── <OPERA_ID>_GEN-DIST-STATUS-ACQ.tif
├── <OPERA_ID>_GEN-METRIC-MAX.tif
├── <OPERA_ID>_GEN-DIST-CONF.tif
├── <OPERA_ID>_GEN-DIST-DATE.tif
├── <OPERA_ID>_GEN-DIST-COUNT.tif
├── <OPERA_ID>_GEN-DIST-PERC.tif
├── <OPERA_ID>_GEN-DIST-DUR.tif
└── <OPERA_ID>_GEN-DIST-LAST-DATE.tif
```

Where `<OPERA_ID>` follows the naming convention: `OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{version}`

### Example Product Structure

```
OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1/
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-STATUS.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-METRIC.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-STATUS-ACQ.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-METRIC-MAX.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-CONF.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-DATE.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-COUNT.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-PERC.tif
├── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-DUR.tif
└── OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250806T145521Z_S1_30_v0.1_GEN-DIST-LAST-DATE.tif
```

## DIST-S1 Product Layers

| Layer Name | dtype | nodata | description |
|------------|-------|--------|-------------|
| `GEN-DIST-STATUS` | `uint8` | `255` | Status of the generic disturbance classification (see the DISTLABEL2VAL table for more details on the status labels). |
| `GEN-METRIC` | `float32` | `nan` | Metric value for the generic disturbance classification. Can be viewed as number of standard devations from the mean. Value is a non-negative real number. |
| `GEN-DIST-STATUS-ACQ` | `uint8` | `255` | Status of the generic disturbance classification with respect to the latest acquisition date (see DISTLABEL2VAL table for more details on the status labels) |
| `GEN-METRIC-MAX` | `float32` | `nan` | Maximum metric value for the generic disturbance classification over all acquisition dates sincefirst disturbance. Reset to 0 when a new disturbance is detected. Value is a non-negative real number. |
| `GEN-DIST-CONF` | `float32` | `-1` | Confidence level for the generic disturbance classification. Value is a non-negative real number. Reset to 0 when a new disturbance is detected. -1 is nodata or no acquisition data available over previous dates. |
| `GEN-DIST-DATE` | `int16` | `-1` | Date of the generic disturbance classification. Value is a non-negative integer and is the number of days from 2020-12-31. -1 is nodata or no acquisition data available.over previous dates. |
| `GEN-DIST-COUNT` | `uint8` | `255` | The number of generic disturbances since first detection. Value is a non-negative integer. |
| `GEN-DIST-PERC` | `uint8` | `255` | Percentage of the generic disturbance disturbance since first detection. |
| `GEN-DIST-DUR` | `int16` | `-1` | Duration of the generic disturbance classification since first detection in days. |
| `GEN-DIST-LAST-DATE` | `int16` | `-1` | Latest generic disturbance detection. |

## Disturbance Labels

| Status Value | Description |
|--------------|-------------|
| `0` | no disturbance |
| `1` | first low conf disturbance |
| `2` | provisional low conf disturbance |
| `3` | confirmed low conf disturbance |
| `4` | first high conf disturbance |
| `5` | provisional high conf disturbance |
| `6` | confirmed high conf disturbance |
| `7` | confirmed low conf disturbance finished |
| `8` | confirmed high conf disturbance finished |
| `255` | nodata |
