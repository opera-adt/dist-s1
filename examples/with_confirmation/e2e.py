from pathlib import Path

from dist_s1 import run_dist_s1_workflow


def main() -> None:
    # First run - generate unconfirmed product
    mgrs_tile_id = '11SLT'  # MGRS tile ID
    post_date_initial = '2025-01-09'  # date of initial pass of Sentinel-1
    track_number = 71  # Sentinel-1 track number
    dst_dir = Path('../../notebooks/los-angeles')  # directory to save the intermediate and output DIST-S1 product
    device = 'cpu'  # can be cpu, cuda, mps or best
    n_workers_for_norm_param_estimation = 5  # number of workers for normal parameter estimation
    product_dst_dir_unconfirmed = Path('./unconfirmed_products')  # directory to save the unconfirmed products

    run_dist_s1_workflow(
        mgrs_tile_id,
        post_date_initial,
        track_number,
        post_date_buffer_days=1,
        dst_dir=dst_dir,
        device=device,
        product_dst_dir=product_dst_dir_unconfirmed,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
    )

    opera_products = list(product_dst_dir_unconfirmed.glob('OPERA_L3_DIST-ALERT-S1_*'))

    if not opera_products:
        raise RuntimeError('No OPERA products found in current directory')

    prior_product_dir = sorted(opera_products)[-1]
    print(f'Prior product directory: {prior_product_dir}')

    post_date_confirmation = '2025-01-21'  # date of confirmation pass of Sentinel-1
    product_dst_dir_confirmed = Path('./confirmed_products')  # directory to save the confirmed products

    # Run the second workflow with prior product for confirmation
    run_dist_s1_workflow(
        mgrs_tile_id,
        post_date_confirmation,
        track_number,
        post_date_buffer_days=1,
        dst_dir=dst_dir,
        device=device,
        product_dst_dir=product_dst_dir_confirmed,
        prior_dist_s1_product=prior_product_dir,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
    )


if __name__ == '__main__':
    main()
