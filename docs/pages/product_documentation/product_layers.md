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
