import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from mimetypes import guess_type
from pathlib import Path

import boto3
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


def upload_file_to_s3(path_to_file: Path, bucket: str, prefix: str = '', profile_name: str | None = None) -> None:
    s3_client = get_s3_client(profile_name)
    key = str(Path(prefix) / path_to_file.name)
    extra_args = {'ContentType': get_content_type(key)}

    s3_client.upload_file(str(path_to_file), bucket, key, extra_args)

    tag_set = get_tag_set(path_to_file.name)

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
    product_directory: Path, bucket: str, prefix: str = '', profile_name: str | None = None
) -> None:
    for file in product_directory.glob('*.png'):
        upload_file_to_s3(file, bucket, prefix, profile_name)

    product_zip_path = f'{product_directory}.zip'
    shutil.make_archive(product_directory, 'zip', product_directory)
    upload_file_to_s3(product_zip_path, bucket, prefix)
    Path(product_zip_path).unlink()
