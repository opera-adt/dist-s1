import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from mimetypes import guess_type
from pathlib import Path, PurePosixPath
from typing import ParamSpec, TypeVar

import boto3
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


P = ParamSpec('P')
R = TypeVar('R')


def rasterio_anon_s3_env(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
            return func(*args, **kwargs)

    return wrapper


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


def upload_files_to_s3_threaded(
    file_list: list[tuple[Path, str, str]], bucket: str, profile_name: str | None = None, max_workers: int = 5
) -> None:
    def _upload(file_path: Path, s3_key: str, prefix: str) -> None:
        full_key = str(Path(prefix) / s3_key) if prefix else s3_key
        upload_file_to_s3(file_path, bucket, full_key, profile_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_upload, file_path, s3_key, prefix): file_path for file_path, s3_key, prefix in file_list
        }

        with tqdm(total=len(file_list), desc='Uploading files to S3', unit='file') as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                future.result()  # Raises exception if upload failed
                pbar.set_postfix_str(file_path.name)
                pbar.update(1)


def upload_product_to_s3(
    product_directory: Path | str,
    bucket: str,
    prefix: str = '',
    upload_zipped: bool = True,
) -> None:
    product_dir_path = Path(product_directory)

    for file in product_dir_path.glob('*.png'):
        upload_file_to_s3(file, bucket, prefix)

    if upload_zipped:
        product_zip_path = f'{product_dir_path}.zip'
        shutil.make_archive(str(product_dir_path), 'zip', product_dir_path)
        upload_file_to_s3(product_zip_path, bucket, prefix)
        Path(product_zip_path).unlink()

    prefix_prod = f'{prefix}/{product_dir_path.name}' if prefix else product_dir_path.name
    for file in product_dir_path.glob('*.png'):
        upload_file_to_s3(file, bucket, prefix_prod)
    for file in product_directory.glob('*.tif'):
        upload_file_to_s3(file, bucket, prefix_prod)


def is_s3_path(path: str | Path) -> bool:
    return str(path).startswith('s3://')


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    uri = uri.removeprefix('s3://')
    parts = uri.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key


def check_s3_prefix_exists(bucket: str, prefix: str) -> bool:
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response


def check_s3_object_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


def download_file_from_s3(bucket: str, key: str, dst_path: Path | str, profile_name: str | None = None) -> None:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if profile_name:
        s3 = get_s3_client(profile_name)
    else:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    s3.download_file(bucket, key, str(dst_path))


def download_product_from_s3(s3_uri: str, dst_dir: Path | str, profile_name: str | None = None) -> Path:
    bucket, key_prefix = parse_s3_uri(s3_uri)

    # Want to remove trailing slash so path is non-empty
    # Add it back so we can list files underneath the prefix
    key_prefix = key_prefix.rstrip('/')
    product_name = PurePosixPath(key_prefix).name
    key_prefix += '/'

    product_dir = Path(dst_dir) / product_name
    product_dir.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client(profile_name) if profile_name else boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Collect all file keys first to show accurate progress
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/'):
                files.append(obj['Key'])

    # Download with progress bar
    with tqdm(total=len(files), desc=f'Downloading {product_name}', unit='file') as pbar:
        for key in files:
            dst_file = product_dir / PurePosixPath(key).relative_to(key_prefix)
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(dst_file))
            pbar.set_postfix_str(dst_file.name)
            pbar.update(1)

    return product_dir
