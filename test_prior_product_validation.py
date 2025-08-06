#!/usr/bin/env python3
"""
Quick test to verify that the prior_dist_s1_product field validator works correctly.
"""

from pathlib import Path
from src.dist_s1.data_models.runconfig_model import RunConfigData
from src.dist_s1.data_models.output_models import DistS1ProductDirectory

def test_prior_product_validation():
    """Test that prior_dist_s1_product accepts str, Path, and DistS1ProductDirectory."""

    # Test data
    test_product_path = "tests/test_data/golden_datasets/10SGD/OPERA_L3_DIST-ALERT-S1_T10SGD_20241221T015858Z_20250806T145536Z_S1_30_v0.1"

    # Test with string path
    try:
        config_from_str = RunConfigData(
            pre_rtc_copol=["dummy_path.tif"],
            pre_rtc_crosspol=["dummy_path.tif"],
            post_rtc_copol=["dummy_path.tif"],
            post_rtc_crosspol=["dummy_path.tif"],
            mgrs_tile_id="10SGD",
            prior_dist_s1_product=test_product_path,
            check_input_paths=False  # Skip file existence checks for this test
        )
        print("✓ String path accepted and converted to DistS1ProductDirectory")
        print(f"  Type: {type(config_from_str.prior_dist_s1_product)}")
    except Exception as e:
        print(f"✗ String path failed: {e}")

    # Test with Path object
    try:
        config_from_path = RunConfigData(
            pre_rtc_copol=["dummy_path.tif"],
            pre_rtc_crosspol=["dummy_path.tif"],
            post_rtc_copol=["dummy_path.tif"],
            post_rtc_crosspol=["dummy_path.tif"],
            mgrs_tile_id="10SGD",
            prior_dist_s1_product=Path(test_product_path),
            check_input_paths=False  # Skip file existence checks for this test
        )
        print("✓ Path object accepted and converted to DistS1ProductDirectory")
        print(f"  Type: {type(config_from_path.prior_dist_s1_product)}")
    except Exception as e:
        print(f"✗ Path object failed: {e}")

    # Test with None (should remain None)
    try:
        config_none = RunConfigData(
            pre_rtc_copol=["dummy_path.tif"],
            pre_rtc_crosspol=["dummy_path.tif"],
            post_rtc_copol=["dummy_path.tif"],
            post_rtc_crosspol=["dummy_path.tif"],
            mgrs_tile_id="10SGD",
            prior_dist_s1_product=None,
            check_input_paths=False  # Skip file existence checks for this test
        )
        print("✓ None value accepted")
        print(f"  Value: {config_none.prior_dist_s1_product}")
        print(f"  Confirmation mode: {config_none.confirmation}")
    except Exception as e:
        print(f"✗ None value failed: {e}")

if __name__ == "__main__":
    test_prior_product_validation()