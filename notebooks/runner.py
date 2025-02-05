from pathlib import Path

from dist_s1.workflows import (
    run_burst_disturbance_workflow,
    run_dist_s1_localization_workflow,
    run_dist_s1_packaging_workflow,
    run_dist_s1_workflow,
    run_disturbance_merge_workflow
)



## Example 0

# mgrs_tile_id = '10SGD'
# post_date = '2025-01-02'
# track_number = 137
# dst_dir = Path('out')
# memory_strategy = 'high'


## Example 1 - Los Angeles Wildfire
mgrs_tile_id = '11SLT'
post_date = '2025-01-21'
track_number = 71
dst_dir = Path('los-angeles')
memory_strategy = 'high'


run_config = run_dist_s1_localization_workflow(
    mgrs_tile_id,
    post_date,
    track_number,
    1,
    dst_dir=dst_dir,
    input_data_dir=dst_dir,
)

# run_burst_disturbance_workflow(run_config)
# run_disturbance_merge_workflow(run_config)
run_dist_s1_packaging_workflow(run_config)