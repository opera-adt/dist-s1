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
    """Test upload_product_to_s3 function with mocked upload_file_to_s3.

    This test verifies that:
    1. upload_file_to_s3 is called for each .png file
    2. A zip file is created and uploaded
    3. The bucket and prefix are correctly passed to upload_file_to_s3
    4. The zip file is cleaned up after upload
    """
    product_directory = test_opera_golden_cropped_dataset_dict['current']

    tmp_product_dir = test_dir / 'tmp_product_test'
    if tmp_product_dir.exists():
        shutil.rmtree(tmp_product_dir)
    shutil.copytree(product_directory, tmp_product_dir)

    mock_upload_file_to_s3 = mocker.patch('dist_s1.aws.upload_file_to_s3')

    # Test parameters
    bucket = 'test-bucket'
    prefix = 'test/prefix'

    try:
        upload_product_to_s3(tmp_product_dir, bucket, prefix)

        png_files = list(tmp_product_dir.glob('*.png'))

        expected_calls = []

        for png_file in png_files:
            expected_calls.append(call(png_file, bucket, prefix, None))

        zip_path_str = f'{tmp_product_dir}.zip'
        expected_calls.append(call(zip_path_str, bucket, prefix))

        mock_upload_file_to_s3.assert_has_calls(expected_calls, any_order=True)

        expected_call_count = len(png_files) + 1  # +1 for zip
        assert mock_upload_file_to_s3.call_count == expected_call_count

        assert not Path(zip_path_str).exists()

    finally:
        if tmp_product_dir.exists():
            shutil.rmtree(tmp_product_dir)
        zip_path = Path(f'{tmp_product_dir}.zip')
        if zip_path.exists():
            zip_path.unlink()
