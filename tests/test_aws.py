import shutil
from pathlib import Path
from unittest.mock import call

import pytest
from pytest_mock import MockerFixture

from dist_s1.aws import get_opera_product_from_s3_job_id_prefix, upload_product_to_s3


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
    # 1. Zip to root prefix
    # 2. PNGs to prefix/product_name
    # 3. TIFs to prefix/product_name
    expected_calls = []
    expected_calls.append(call(zip_path, bucket, prefix))
    for png in png_files:
        expected_calls.append(call(png, bucket, prefix_prod))
    for tif in tif_files:
        expected_calls.append(call(tif, bucket, prefix_prod))

    mock_upload.assert_has_calls(expected_calls, any_order=True)
    assert mock_upload.call_count == len(png_files) + len(tif_files) + 1

    # Verify zip cleanup
    assert not Path(zip_path).exists()


def test_get_opera_product_from_s3_prefix_success(mocker: MockerFixture) -> None:
    mock_s3_client = mocker.MagicMock()
    mocker.patch('dist_s1.aws.boto3.client', return_value=mock_s3_client)

    mock_paginator = mocker.MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    opera_id = 'OPERA_L3_DIST-ALERT-S1_T36UYA_20220324T152116Z_20260309T202443Z_S1A_30_v0.1'
    mock_paginator.paginate.return_value = [
        {
            'Contents': [
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-STATUS.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-METRIC.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-STATUS-ACQ.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-METRIC-MAX.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-CONF.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-DATE.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-COUNT.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-PERC.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-DUR.tif'},
                {'Key': f'job-123/{opera_id}/{opera_id}_GEN-DIST-LAST-DATE.tif'},
            ]
        }
    ]

    result = get_opera_product_from_s3_job_id_prefix('test-bucket', 'job-123')

    assert (
        result == 's3://test-bucket/job-123/OPERA_L3_DIST-ALERT-S1_T36UYA_20220324T152116Z_20260309T202443Z_S1A_30_v0.1'
    )


def test_get_opera_product_from_s3_prefix_no_products(mocker: MockerFixture) -> None:
    mock_s3_client = mocker.MagicMock()
    mocker.patch('dist_s1.aws.boto3.client', return_value=mock_s3_client)

    mock_paginator = mocker.MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    mock_paginator.paginate.return_value = [{'Contents': [{'Key': 'job-123/some-file.txt'}]}]

    with pytest.raises(ValueError, match='No OPERA product directories found'):
        get_opera_product_from_s3_job_id_prefix('test-bucket', 'job-123')


def test_get_opera_product_from_s3_prefix_multiple_products(mocker: MockerFixture) -> None:
    mock_s3_client = mocker.MagicMock()
    mocker.patch('dist_s1.aws.boto3.client', return_value=mock_s3_client)

    mock_paginator = mocker.MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    opera_id_1 = 'OPERA_L3_DIST-ALERT-S1_T36UYA_20220324T152116Z_20260309T202443Z_S1A_30_v0.1'
    opera_id_2 = 'OPERA_L3_DIST-ALERT-S1_T36UYA_20220325T152116Z_20260309T202443Z_S1A_30_v0.1'
    mock_paginator.paginate.return_value = [
        {
            'Contents': [
                {'Key': f'job-123/{opera_id_1}/file1.tif'},
                {'Key': f'job-123/{opera_id_2}/file2.tif'},
            ]
        }
    ]

    with pytest.raises(ValueError, match='Multiple OPERA products found'):
        get_opera_product_from_s3_job_id_prefix('test-bucket', 'job-123')


def test_get_opera_product_from_s3_prefix_ignores_zip_files(mocker: MockerFixture) -> None:
    mock_s3_client = mocker.MagicMock()
    mocker.patch('dist_s1.aws.boto3.client', return_value=mock_s3_client)

    mock_paginator = mocker.MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    opera_id = 'OPERA_L3_DIST-ALERT-S1_T36UYA_20220324T152116Z_20260309T202443Z_S1A_30_v0.1'
    mock_paginator.paginate.return_value = [
        {
            'Contents': [
                {'Key': f'job-123/{opera_id}.zip'},
                {'Key': f'job-123/{opera_id}/file1.tif'},
            ]
        }
    ]

    result = get_opera_product_from_s3_job_id_prefix('test-bucket', 'job-123')

    assert result == f's3://test-bucket/job-123/{opera_id}'


def test_get_opera_product_from_s3_prefix_invalid_product_name(mocker: MockerFixture) -> None:
    mock_s3_client = mocker.MagicMock()
    mocker.patch('dist_s1.aws.boto3.client', return_value=mock_s3_client)

    mock_paginator = mocker.MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    mock_paginator.paginate.return_value = [{'Contents': [{'Key': 'job-123/OPERA_L3_DIST-INVALID-NAME/file1.tif'}]}]

    with pytest.raises(ValueError, match='Invalid OPERA product name'):
        get_opera_product_from_s3_job_id_prefix('test-bucket', 'job-123')
