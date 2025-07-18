#!/usr/bin/env python3
"""
Demonstration script showing the new confirmation property behavior.

The confirmation attribute has been removed from RunConfigData and replaced
with a property that returns `prior_dist_s1_product is not None`.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.data_models.output_models import ProductDirectoryData


def main():
    print('=== Confirmation Property Demonstration ===\n')

    # Create a basic RunConfigData instance
    config = RunConfigData(
        pre_rtc_copol=[Path('dummy1.tif')],
        pre_rtc_crosspol=[Path('dummy1_vh.tif')],
        post_rtc_copol=[Path('dummy2.tif')],
        post_rtc_crosspol=[Path('dummy2_vh.tif')],
        mgrs_tile_id='10SGD',
        check_input_paths=False,  # Don't validate dummy paths
    )

    print('1. Default state (no prior product):')
    print(f'   prior_dist_s1_product = {config.prior_dist_s1_product}')
    print(f'   confirmation = {config.confirmation}')
    print(f'   confirmation type = {type(config.confirmation)}')
    print()

    print('2. Setting prior_dist_s1_product to a product:')
    # Create a dummy product directory data
    product_dir = ProductDirectoryData(
        dst_dir=Path('dummy_product'),
        product_name='OPERA_L3_DIST-ALERT-S1_T10SGD_20250102T015857Z_20250716T154103Z_S1_30_v0.1',
    )
    config.prior_dist_s1_product = product_dir
    print(f'   prior_dist_s1_product = {config.prior_dist_s1_product.product_name}')
    print(f'   confirmation = {config.confirmation}')
    print(f'   confirmation type = {type(config.confirmation)}')
    print()

    print('3. Unsetting prior_dist_s1_product:')
    config.prior_dist_s1_product = None
    print(f'   prior_dist_s1_product = {config.prior_dist_s1_product}')
    print(f'   confirmation = {config.confirmation}')
    print(f'   confirmation type = {type(config.confirmation)}')
    print()

    print('✅ The confirmation property correctly reflects the state of prior_dist_s1_product!')
    print('✅ No explicit confirmation assignment is needed anymore!')


if __name__ == '__main__':
    main()
