import shutil
from pathlib import Path

import numpy as np
import rasterio

from dist_s1.data_models.output_models import ProductDirectoryData


def test_product_directory_data_from_product_path(test_dir: Path, test_opera_golden_dummy_dataset: Path) -> None:
    """Tests that a copied directory with a different procesing timestamp is equal.

    Also tests if we replace a layer by a random array of the same shape and dtype, the product is not equal.
    """
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    product_dir_path = Path(test_opera_golden_dummy_dataset)
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

    golden_data = ProductDirectoryData.from_product_path(product_dir_path)
    copied_data = ProductDirectoryData.from_product_path(product_new_dir_path)

    assert golden_data == copied_data

    gen_status_path = [p for p in new_product_file_paths if p.name.endswith('DIST-GEN-STATUS.tif')][0]
    with rasterio.open(gen_status_path) as src:
        p = src.profile
        t = src.tags()

    X = (np.random.randn(p['height'], p['width']) * 100).astype(np.uint8)
    with rasterio.open(gen_status_path, 'w', **p) as dst:
        dst.write(X, 1)
        dst.update_tags(**t)

    assert golden_data != copied_data

    shutil.rmtree(tmp_dir)
