from dataclasses import dataclass
from pathlib import Path

import torch.multiprocessing as mp
from distmetrics.despeckle import despeckle_one_rtc_arr_with_tv
from tqdm import tqdm

from dist_s1.rio_tools import open_one_ds, serialize_one_2d_ds


@dataclass
class ProcessingArgs:
    """Container for processing arguments to make multiprocessing more readable."""

    source_path: Path
    destination_path: Path
    interpolation_method: str
    preserve_exterior_mask: bool
    n_iter_bilinear: int
    fill_value: float | None


def despeckle_and_serialize_one_rtc_s1(
    rtc_s1_path: Path,
    dst_path: Path,
    interpolation_method: str = 'none',
    preserve_exterior_mask: bool = True,
    n_iter_bilinear: int = 10,
    fill_value: float = None,
) -> Path:
    arr, prof = open_one_ds(rtc_s1_path)
    arr_d = despeckle_one_rtc_arr_with_tv(
        arr,
        interp_method=interpolation_method,
        preserve_exterior_mask=preserve_exterior_mask,
        n_iter_bilinear=n_iter_bilinear,
        fill_value=fill_value,
    )
    serialize_one_2d_ds(arr_d, prof, dst_path)
    return dst_path


def _process_wrapper(args: ProcessingArgs) -> Path:
    """Process a single RTC S1 file with despeckling and serialization."""
    return despeckle_and_serialize_one_rtc_s1(
        rtc_s1_path=args.source_path,
        dst_path=args.destination_path,
        interpolation_method=args.interpolation_method,
        preserve_exterior_mask=args.preserve_exterior_mask,
        n_iter_bilinear=args.n_iter_bilinear,
        fill_value=args.fill_value,
    )


def despeckle_and_serialize_rtc_s1(
    rtc_s1_paths: list[Path],
    dst_paths: list[Path],
    tqdm_enabled: bool = True,
    n_workers: int = 5,
    interpolation_method: str = 'bilinear',
    preserve_exterior_mask: bool = True,
    overwrite: bool = False,
    n_iter_bilinear: int = 10,
    fill_value: float = None,
) -> list[Path]:
    dst_paths = list(map(Path, dst_paths))
    src_paths = list(map(Path, rtc_s1_paths))

    if overwrite:
        dst_paths_ = list(dst_paths)
        dst_paths = [dst_p for dst_p in dst_paths_ if not dst_p.exists()]
        src_paths = [src_p for (src_p, dst_p) in zip(src_paths, dst_paths_) if not dst_p.exists()]

    if len(dst_paths) != len(src_paths):
        raise ValueError('Number of destination paths must match number of source paths')

    # Make sure the parent directories exist
    [p.parent.mkdir(exist_ok=True, parents=True) for p in dst_paths]

    if dst_paths:
        # Create ProcessingArgs objects for multiprocessing
        processing_args = [
            ProcessingArgs(
                source_path=src_path,
                destination_path=dst_path,
                interpolation_method=interpolation_method,
                preserve_exterior_mask=preserve_exterior_mask,
                n_iter_bilinear=n_iter_bilinear,
                fill_value=fill_value,
            )
            for src_path, dst_path in zip(src_paths, dst_paths)
        ]

        # Use multiprocessing to process the files with real-time progress updates
        pool = None
        try:
            pool = mp.Pool(processes=n_workers)
            # Use imap for real-time progress updates
            dst_paths = list(
                tqdm(
                    pool.imap(_process_wrapper, processing_args),
                    total=len(src_paths),
                    disable=not tqdm_enabled,
                    desc='Despeckling and serializing RTC S1 files',
                )
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    return dst_paths
