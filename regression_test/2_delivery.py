from pathlib import Path

import dist_s1
from dist_s1.aws import upload_file_to_s3, upload_files_to_s3_threaded


def upload_directory_to_s3(
    directory_path: Path, bucket: str, prefix: str, profile_name: str | None = None, max_workers: int = 5
) -> None:
    if not directory_path.exists():
        raise ValueError(f'Directory does not exist: {directory_path}')

    if not directory_path.is_dir():
        raise ValueError(f'Path is not a directory: {directory_path}')

    # Collect all files to upload
    files_to_upload: list[tuple[Path, str, str]] = []

    for file_path in directory_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(directory_path.parent)
            s3_key = str(relative_path)
            files_to_upload.append((file_path, s3_key, prefix))

    if files_to_upload:
        successful, failed = upload_files_to_s3_threaded(files_to_upload, bucket, profile_name, max_workers)


def upload_file_to_s3_with_path(file_path: Path, bucket: str, prefix: str, profile_name: str | None = None) -> None:
    if (not file_path.exists()) or (not file_path.is_file()):
        raise ValueError(f'Path is not a file: {file_path}')

    # Use the file's parent directory as part of the S3 key
    s3_prefix = str(Path(prefix) / file_path.parent) if str(file_path.parent) != '.' else prefix

    upload_file_to_s3(file_path, bucket, s3_prefix, profile_name)


def upload_data_to_s3(
    bucket: str, prefix: str, paths: tuple[str, ...], profile_name: str | None = None, max_workers: int = 5
) -> None:
    version = dist_s1.__version__
    full_prefix = f'{prefix}/{version}' if prefix else version

    all_files_to_upload: list[tuple[Path, str, str]] = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(path)
                    s3_key = str(relative_path)
                    all_files_to_upload.append((file_path, s3_key, full_prefix))
        elif path.is_file():
            s3_key = str(Path(path.parent) / path.name) if str(path.parent) != '.' else path.name
            all_files_to_upload.append((path, s3_key, full_prefix))

    if all_files_to_upload:
        successful, failed = upload_files_to_s3_threaded(all_files_to_upload, bucket, profile_name, max_workers)
    if failed:
        print(f'Failed to upload {len(failed)} files')
        print(failed)


def main() -> None:
    bucket = 'dist-s1-golden-datasets'
    prefix = None
    profile = 'saml-pub'
    max_workers = 20
    paths = (
        'golden_dataset',
        'product_0',
        'runconfig.yml',
        'algo_params.yml',
        'out_1/11SLT',
        'out_1/11SLT_water_mask.tif',
    )

    upload_data_to_s3(bucket, prefix, paths, profile, max_workers)


if __name__ == '__main__':
    main()
