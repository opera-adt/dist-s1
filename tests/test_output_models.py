import shutil
from pathlib import Path

import numpy as np
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


def test_from_product_path_s3(test_dir: Path) -> None:
    """Test loading product from public S3 bucket."""
    s3_uri = 's3://dist-s1-golden-datasets/2.0.9/golden_dataset/OPERA_L3_DIST-ALERT-S1_T11SLT_20250121T135246Z_20251215T221221Z_S1A_30_v0.1/'

    product = DistS1ProductDirectory.from_product_path(s3_uri)

    assert product.product_name == 'OPERA_L3_DIST-ALERT-S1_T11SLT_20250121T135246Z_20251215T221221Z_S1A_30_v0.1'
    assert product.mgrs_tile_id == '11SLT'
    assert product.acq_datetime.year == 2025
    assert product.acq_datetime.month == 1
    assert product.acq_datetime.day == 21

    assert product.validate_layer_paths()
    assert product.validate_tif_layer_dtypes()

    for layer in product.layers:
        assert layer in product.layer_path_dict
        layer_uri = product.layer_path_dict[layer]
        assert layer_uri.startswith('s3://')
        assert product.product_name in layer_uri

    # Download the product and verify it matches
    tmp_dir = test_dir / 'tmp_s3_download'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    product_same = product.download_to(tmp_dir)

    assert product == product_same

    # Clean up
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
