from pathlib import Path

import numpy as np
from distmetrics.despeckle import despeckle_rtc_arrs_with_tv
from distmetrics.transformer import estimate_normal_params_of_logits, load_transformer_model
from tqdm import tqdm

from dist_s1.rio_tools import check_profiles_match, open_one_ds, serialize_one_ds


def despeckle_and_serialize_rtc_s1(
    rtc_s1_paths: list[Path], dst_paths: list[Path], batch_size: int = 100
) -> list[Path]:
    # Cast to Path
    dst_paths = list(map(Path, dst_paths))
    # Make sure the parent directories exist
    [p.parent.mkdir(exist_ok=True, parents=True) for p in dst_paths]

    n_batches = int(np.ceil(len(rtc_s1_paths) / batch_size))
    for k in tqdm(range(n_batches), desc='batch'):
        paths_subset = rtc_s1_paths[k * batch_size : (k + 1) * batch_size]
        dst_paths_subset = dst_paths[k * batch_size : (k + 1) * batch_size]

        # don't overwrite existing data
        dst_paths_subset_to_create = [dst_p for dst_p in dst_paths_subset if not dst_p.exists()]
        paths_subset_to_create = [src_p for (src_p, dst_p) in zip(paths_subset, dst_paths_subset) if not dst_p.exists()]

        # open
        if dst_paths_subset_to_create:
            data = list(map(open_one_ds, paths_subset_to_create))
            arrs, ps = zip(*data)
            # despeckle
            arrs_d = despeckle_rtc_arrs_with_tv(arrs)
            # serialize
            [serialize_one_ds(arr, prof, dst_path) for (arr, prof, dst_path) in zip(arrs_d, ps, dst_paths_subset)]

    return dst_paths


def compute_normal_params_per_burst_and_serialize(
    pre_copol_paths_dskpl_paths: list[Path],
    pre_crosspol_paths_dskpl_paths: list[Path],
    out_path_mu_copol: Path,
    out_path_mu_crosspol: Path,
    out_path_sigma_copol: Path,
    out_path_sigma_crosspol: Path,
    memory_strategy: str = 'high',
) -> Path:
    model = load_transformer_model()

    copol_data = [open_one_ds(path) for path in pre_copol_paths_dskpl_paths]
    crosspol_data = [open_one_ds(path) for path in pre_crosspol_paths_dskpl_paths]
    arrs_copol, profs_copol = zip(*copol_data)
    arrs_crosspol, profs_crosspol = zip(*crosspol_data)

    if len(arrs_copol) != len(arrs_crosspol):
        raise ValueError('Length of Copolar and crosspolar arrays do not match')
    p_ref = profs_copol[0]
    for p_copol, p_crosspol in zip(profs_copol, profs_crosspol):
        check_profiles_match(p_ref, p_copol)
        check_profiles_match(p_ref, p_crosspol)

    logits_mu, logits_sigma = estimate_normal_params_of_logits(
        model, arrs_copol, arrs_crosspol, memory_strategy=memory_strategy
    )
    logits_mu_copol, logits_mu_crosspol = logits_mu[0, ...], logits_mu[1, ...]
    logits_sigma_copol, logits_sigma_crosspol = logits_sigma[0, ...], logits_sigma[1, ...]

    serialize_one_ds(logits_mu_copol, p_ref, out_path_mu_copol)
    serialize_one_ds(logits_mu_crosspol, p_ref, out_path_mu_crosspol)
    serialize_one_ds(logits_sigma_copol, p_ref, out_path_sigma_copol)
    serialize_one_ds(logits_sigma_crosspol, p_ref, out_path_sigma_crosspol)


def compute_disturbance(metric_paths: list[Path], out_dir: Path) -> None:
    pass
