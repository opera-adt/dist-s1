import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from mimetypes import guess_type
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


def get_s3_client(profile_name: str | None = None) -> boto3.client:
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client('s3')
    else:
        return boto3.client('s3')


def get_tag_set(file_name: str) -> dict:
    if file_name.endswith('.png'):
        file_type = 'browse'
    else:
        file_type = 'product'

    tag_set = {'TagSet': [{'Key': 'file_type', 'Value': file_type}]}
    return tag_set


def get_content_type(file_location: Path | str) -> str:
    content_type = guess_type(file_location)[0]
    if not content_type:
        content_type = 'application/octet-stream'
    return content_type


def upload_file_to_s3(path_to_file: Path | str, bucket: str, prefix: str = '', profile_name: str | None = None) -> None:
    file_posix_path = Path(path_to_file)

    s3_client = get_s3_client(profile_name)
    key = str(Path(prefix) / file_posix_path.name)
    extra_args = {'ContentType': get_content_type(key)}

    s3_client.upload_file(str(file_posix_path), bucket, key, extra_args)

    tag_set = get_tag_set(file_posix_path.name)

    s3_client.put_object_tagging(Bucket=bucket, Key=key, Tagging=tag_set)


def upload_file_with_error_handling(file_info: tuple[Path, str, str, str | None]) -> tuple[bool, str, str | None]:
    path_to_file, bucket, key, profile_name = file_info
    try:
        s3_client = get_s3_client(profile_name)
        extra_args = {'ContentType': get_content_type(key)}

        s3_client.upload_file(str(path_to_file), bucket, key, extra_args)

        tag_set = get_tag_set(path_to_file.name)
        s3_client.put_object_tagging(Bucket=bucket, Key=key, Tagging=tag_set)

    except Exception as e:
        error_msg = f'Failed to upload {path_to_file} to s3://{bucket}/{key}: {str(e)}'
        logging.exception(error_msg)
        return False, error_msg, str(e)
    else:
        return True, f'Successfully uploaded {path_to_file} to s3://{bucket}/{key}', None


def upload_files_to_s3_threaded(
    file_list: list[tuple[Path, str, str]], bucket: str, profile_name: str | None = None, max_workers: int = 5
) -> tuple[list[str], list[str]]:
    successful_uploads = []
    failed_uploads = []

    # Prepare file info tuples for threading
    file_info_list = []
    for file_path, s3_key, prefix in file_list:
        full_key = str(Path(prefix) / s3_key) if prefix else s3_key
        file_info_list.append((file_path, bucket, full_key, profile_name))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(upload_file_with_error_handling, file_info): file_info[0] for file_info in file_info_list
        }

        # Process completed uploads with progress bar
        with tqdm(total=len(file_info_list), desc='Uploading files to S3', unit='file') as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, message, error = future.result()
                    if success:
                        successful_uploads.append(message)
                        pbar.set_postfix_str(f'✓ {file_path.name}')
                    else:
                        failed_uploads.append(message)
                        pbar.set_postfix_str(f'✗ {file_path.name}')
                except Exception as e:
                    error_msg = f'Unexpected error uploading {file_path}: {str(e)}'
                    failed_uploads.append(error_msg)
                    logging.exception(error_msg)
                    pbar.set_postfix_str(f'✗ {file_path.name}')
                finally:
                    pbar.update(1)

    return successful_uploads, failed_uploads


def upload_product_to_s3(
    product_directory: Path | str, bucket: str, prefix: str = '', profile_name: str | None = None
) -> None:
    product_posix_path = Path(product_directory)

    for file in product_posix_path.glob('*.png'):
        upload_file_to_s3(file, bucket, prefix, profile_name)

    product_zip_path = f'{product_posix_path}.zip'
    shutil.make_archive(str(product_posix_path), 'zip', product_posix_path)
    upload_file_to_s3(product_zip_path, bucket, prefix)
    Path(product_zip_path).unlink()


def is_s3_path(path: str | Path) -> bool:
    """Check if path is S3 URI."""
    return str(path).startswith('s3://')


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    uri = uri.removeprefix('s3://')
    parts = uri.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key


def check_s3_prefix_exists(bucket: str, prefix: str) -> bool:
    """Check if S3 prefix has any objects under it.

    Uses unsigned requests for public bucket access.
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response


def check_s3_object_exists(bucket: str, key: str) -> bool:
    """Check if S3 object exists using HEAD request.

    Uses unsigned requests for public bucket access.
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


def download_file_from_s3(
    bucket: str, key: str, dst_path: Path | str, profile_name: str | None = None
) -> None:
    """Download file from S3 to local path.

    Parameters
    ----------
    bucket : str
        S3 bucket name
    key : str
        S3 object key
    dst_path : Path | str
        Local destination path
    profile_name : str | None
        AWS profile name. If None, uses unsigned requests for public buckets.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if profile_name:
        s3 = get_s3_client(profile_name)
    else:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    s3.download_file(bucket, key, str(dst_path))


def download_product_from_s3(
    s3_uri: str, dst_dir: Path | str, profile_name: str | None = None
) -> Path:
    """Download DIST-S1 product from S3 to local directory.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the product directory (s3://bucket/prefix/product_name/)
    dst_dir : Path | str
        Local destination directory
    profile_name : str | None
        AWS profile name. If None, uses unsigned requests for public buckets.

    Returns
    -------
    Path
        Path to the downloaded product directory
    """
    from pathlib import PurePosixPath

    bucket, key_prefix = parse_s3_uri(s3_uri)
    key_prefix = key_prefix.rstrip('/')

    product_name = PurePosixPath(key_prefix).name
    dst_dir = Path(dst_dir)
    product_dir = dst_dir / product_name
    product_dir.mkdir(parents=True, exist_ok=True)

    if profile_name:
        s3 = get_s3_client(profile_name)
    else:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # List all objects under the prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

    files_to_download = []
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            # Skip if it's a directory marker
            if key.endswith('/'):
                continue
            files_to_download.append(key)

    # Download all files with progress bar
    with tqdm(total=len(files_to_download), desc=f'Downloading {product_name}', unit='file') as pbar:
        for key in files_to_download:
            # Get relative path from product directory
            rel_path = PurePosixPath(key).relative_to(key_prefix)
            dst_file = product_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(dst_file))
            pbar.set_postfix_str(f'{dst_file.name}')
            pbar.update(1)

    return product_dir
