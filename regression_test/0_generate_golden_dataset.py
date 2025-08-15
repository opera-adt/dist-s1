from pathlib import Path

from dist_s1 import run_dist_s1_workflow


def main() -> None:
    mgrs_tile_id = '11SLT'
    post_date_initial = '2025-01-09'
    track_number = 71
    product_dst_dir = Path('product_0')
    dst_dir = Path('out_0')
    run_config_path = Path('out_0/runconfig_0.yml')
    # Force device to be cpu even on Mac M1
    device = 'cpu'

    # run_dist_s1_workflow(
    #     mgrs_tile_id,
    #     post_date_initial,
    #     track_number,
    #     device=device,
    #     dst_dir=dst_dir,
    #     product_dst_dir=product_dst_dir,
    #     run_config_path=run_config_path,
    # )

    opera_products = list(product_dst_dir.glob('OPERA_L3_DIST-ALERT-S1_*'))

    if not opera_products:
        raise RuntimeError('No OPERA products found in current directory')

    prior_product_dir = sorted(opera_products)[-1]
    print(f'Prior product directory: {prior_product_dir}')

    post_date_confirmation = '2025-01-21'
    product_dst_dir = Path('golden_dataset')
    dst_dir = Path('out_1')
    run_config_path = Path('out_1/runconfig_1.yml')

    run_dist_s1_workflow(
        mgrs_tile_id,
        post_date_confirmation,
        track_number,
        device=device,
        dst_dir=dst_dir,
        product_dst_dir=product_dst_dir,
        run_config_path=run_config_path,
        prior_dist_s1_product=prior_product_dir,
    )


if __name__ == '__main__':
    main()
