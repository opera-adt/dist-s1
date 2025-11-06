# DIST-S1 Product Documentation

This page provides comprehensive documentation for the DIST-S1 product layers and disturbance labels.

## Product Naming Specification

DIST-S1 products follow a standardized naming convention that encodes key metadata about the product. The `ProductNameData` model manages this naming scheme and provides validation capabilities.

### Product Name Format

Products follow this format:
```
OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_{sensor}_30_v{version}
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
| `{sensor}` | Sentinel-1 mission identifier | `S1A`,`S1B` or `S1C` |
| `30` | Fixed resolution identifier | `30` |
| `v{version}` | Product version with 'v' prefix | `v{{ get_constant_value_macro('PRODUCT_VERSION') }}` |



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

{{ generate_constants_table(constants.TIF_LAYER_DTYPES, "DIST-S1 Product Layers", "Data Type") }}

### Status Layer Values

{{ generate_disturbance_labels_table() }}


### Layer NoData Values

{{ generate_constants_table(constants.TIF_LAYER_NODATA_VALUES, "Layer NoData Values", "NoData Value") }}