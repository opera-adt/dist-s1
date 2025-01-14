from pathlib import Path
from dataclasses import dataclass


def despeckle_rtc_s1(paths: list[Path], dst_dir: Path) -> list[Path]:
    pass


def compute_dist_metrics(copol_paths: list[Path], crosspol_paths: list[Path], out_dir: Path, n_lookbacks: int) -> None:
    pass


def compute_disturbance(metric_paths: list[Path], out_dir: Path) -> None:
    pass
