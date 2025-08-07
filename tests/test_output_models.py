import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio

from dist_s1.data_models.output_models import DistS1ProductDirectory


def test_product_directory_data_from_product_path(
    test_dir: Path, test_opera_golden_cropped_dataset_dict: dict[str, Path]
) -> None:
    """Tests that a copied directory with a different procesing timestamp is equal.

    Also tests if we replace a layer by a random array of the same shape and dtype, the product is not equal.
    """
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    product_dir_path = Path(test_opera_golden_cropped_dataset_dict['current'])
    product_name_dir = product_dir_path.name
    tokens = product_name_dir.split('_')
    # Change processing timestamp
    new_processing_timetamp = '20250101T000000Z'
    tokens[5] = new_processing_timetamp
    new_product_dir_name = '_'.join(tokens)

    product_new_dir_path = tmp_dir / new_product_dir_name
    if product_new_dir_path.exists():
        shutil.rmtree(product_new_dir_path)
    shutil.copytree(product_dir_path, product_new_dir_path)

    # Change tokens in all the files
    product_file_paths = list(product_new_dir_path.glob('*.tif')) + list(product_new_dir_path.glob('*.png'))
    new_product_file_paths = []
    for path in product_file_paths:
        file_name = path.name
        tokens = file_name.split('_')
        tokens[5] = new_processing_timetamp
        new_file_name = '_'.join(tokens)
        out_path = path.parent / new_file_name
        path.rename(out_path)
        new_product_file_paths.append(out_path)

    golden_data = DistS1ProductDirectory.from_product_path(product_dir_path)
    copied_data = DistS1ProductDirectory.from_product_path(product_new_dir_path)

    assert golden_data == copied_data

    gen_status_path = [p for p in new_product_file_paths if p.name.endswith('GEN-DIST-STATUS.tif')][0]
    with rasterio.open(gen_status_path) as src:
        p = src.profile
        t = src.tags()

    X = (np.random.randn(p['height'], p['width']) * 100).astype(np.uint8)
    with rasterio.open(gen_status_path, 'w', **p) as dst:
        dst.write(X, 1)
        dst.update_tags(**t)

    assert golden_data != copied_data

    shutil.rmtree(tmp_dir)


def test_generate_product_path_with_placeholders(test_dir: Path) -> None:
    """Test the generate_product_path_with_placeholders function."""
    # Test parameters
    mgrs_tile_id = '10SGD'
    acq_datetime = datetime(2024, 1, 15, 12, 0, 0)

    # Create tmp directory in test directory
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Generate product with placeholders
    product_data = DistS1ProductDirectory.generate_product_path_with_placeholders(
        mgrs_tile_id=mgrs_tile_id,
        acq_datetime=acq_datetime,
        dst_dir=tmp_dir,
        water_mask_path=None,  # No water mask for this test
        overwrite=True,
    )

    # Check that product directory was created
    assert product_data.product_dir_path.exists()
    assert product_data.product_dir_path.is_dir()

    # Check that all TIF layers were created
    for layer_name, layer_path in product_data.layer_path_dict.items():
        if layer_path.suffix == '.tif':
            assert layer_path.exists(), f'Layer {layer_name} was not created: {layer_path}'

            # Check that the file has the correct data type
            with rasterio.open(layer_path) as src:
                expected_dtype = product_data.tif_layer_dtypes[layer_name]
                actual_dtype = src.dtypes[0]
                assert actual_dtype == expected_dtype, (
                    f'Layer {layer_name} has wrong dtype: {actual_dtype} != {expected_dtype}'
                )

                # Check that the array contains only zeros
                data = src.read(1)
                assert np.all(data == 0), f'Layer {layer_name} contains non-zero values'

    # Validate the product using existing validation methods
    assert product_data.validate_layer_paths(), 'Layer paths validation failed'
    assert product_data.validate_tif_layer_dtypes(), 'Data types validation failed'

    # Test with water mask path (should raise error for invalid path)
    invalid_water_mask = tmp_dir / 'nonexistent_water_mask.tif'

    # This should raise a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DistS1ProductDirectory.generate_product_path_with_placeholders(
            mgrs_tile_id=mgrs_tile_id,
            acq_datetime=acq_datetime,
            dst_dir=tmp_dir,
            water_mask_path=invalid_water_mask,
            overwrite=True,
        )

    # Clean up
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
