import shutil
from pathlib import Path
from unittest.mock import call

from pytest_mock import MockerFixture

from dist_s1.aws import upload_product_to_s3


def test_upload_product_to_s3(
    test_opera_golden_cropped_dataset_dict: dict[str, Path],
    test_dir: Path,
    mocker: MockerFixture,
) -> None:
    """Test upload_product_to_s3 uploads files to correct S3 locations."""
    product_directory = test_opera_golden_cropped_dataset_dict['current']

    tmp_product_dir = test_dir / 'tmp_product_test'
    if tmp_product_dir.exists():
        shutil.rmtree(tmp_product_dir)
    shutil.copytree(product_directory, tmp_product_dir)

    mock_upload = mocker.patch('dist_s1.aws.upload_file_to_s3')

    bucket = 'test-bucket'
    prefix = 'test/prefix'

    upload_product_to_s3(tmp_product_dir, bucket, prefix)

    png_files = list(tmp_product_dir.glob('*.png'))
    tif_files = list(tmp_product_dir.glob('*.tif'))
    zip_path = f'{tmp_product_dir}.zip'
    prefix_prod = f'{prefix}/{tmp_product_dir.name}'

    # Expected uploads:
    # 1. PNGs to root prefix
    # 2. Zip to root prefix
    # 3. PNGs to prefix/product_name
    # 4. TIFs to prefix/product_name
    expected_calls = []
    for png in png_files:
        expected_calls.append(call(png, bucket, prefix))
    expected_calls.append(call(zip_path, bucket, prefix))
    for png in png_files:
        expected_calls.append(call(png, bucket, prefix_prod))
    for tif in tif_files:
        expected_calls.append(call(tif, bucket, prefix_prod))

    mock_upload.assert_has_calls(expected_calls, any_order=True)
    assert mock_upload.call_count == 2 * len(png_files) + len(tif_files) + 1

    # Verify zip cleanup
    assert not Path(zip_path).exists()
