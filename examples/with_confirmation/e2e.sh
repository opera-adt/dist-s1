dist-s1 run \
    --mgrs_tile_id '11SLT' \
    --post_date '2025-01-09' \
    --track_number 71 \
    --dst_dir '../../notebooks/los-angeles' \
    --device 'cpu' \
    --n_workers_for_norm_param_estimation 5 \
    --product_dst_dir './unconfirmed_products'
    
# get the latest unconfirmed product
prior_product_dir=$(find unconfirmed_products/ -maxdepth 1 -type d -name 'OPERA*' \
  | grep -E 'OPERA_L3_DIST-ALERT-S1_' \
  | sort \
  | tail -n1 | tr -d '\n')

echo "Prior product directory: $prior_product_dir"

if [[ -z "$prior_product_dir" ]]; then
  echo "No prior product directory found!" >&2
  exit 1
fi

dist-s1 run \
    --mgrs_tile_id '11SLT' \
    --post_date '2025-01-21' \
    --track_number 71 \
    --dst_dir '../../notebooks/los-angeles' \
    --device 'cpu' \
    --n_workers_for_norm_param_estimation 5 \
    --product_dst_dir './confirmed_products' \
    --prior_dist_s1_product "$prior_product_dir" 