from datetime import datetime
from pathlib import Path

from dist_s1.constants import MODEL_CONTEXT_LENGTH, N_LOOKBACKS
from dist_s1.data_models.runconfig_model import RunConfigData
from dist_s1.localize_rtc_s1 import localize_rtc_s1
from dist_s1.processing import (
    aggregate_burst_disturbance_over_lookbacks_and_serialize,
    compute_burst_disturbance_for_lookback_group_and_serialize,
    compute_normal_params_per_burst_and_serialize,
    despeckle_and_serialize_rtc_s1,
)


def run_dist_s1_localization_workflow(
    mgrs_tile_id: str, post_date: str | datetime, track_number: int, post_date_buffer_days: int
) -> RunConfigData:
    # Localize inputs
    rtc_s1_gdf = localize_rtc_s1(mgrs_tile_id, post_date, track_number, post_date_buffer_days=post_date_buffer_days)

    # Create runconfig
    run_config = RunConfigData.from_product_df(rtc_s1_gdf)

    return run_config


def run_despeckle_workflow(run_config: RunConfigData) -> None:
    """Despeckle by burst/polarization and then serializes.

    Parameters
    ----------
    run_config : RunConfigData

    Notes
    -----
    - All input and output paths are in the run_config.
    """
    # Table has input copol/crosspol paths and output despeckled paths
    df_inputs = run_config.df_inputs

    # Inputs
    copol_paths = df_inputs.loc_path_copol.tolist()
    crosspol_paths = df_inputs.loc_path_crosspol.tolist()

    # Outputs
    dspkl_copol_paths = df_inputs.loc_path_copol_dspkl.tolist()
    dspkl_crosspol_paths = df_inputs.loc_path_crosspol_dspkl.tolist()

    assert len(copol_paths) == len(dspkl_copol_paths) == len(crosspol_paths) == len(dspkl_crosspol_paths)

    # The copol/crosspol paths must be in the same order
    rtc_paths = copol_paths + crosspol_paths
    dst_paths = dspkl_copol_paths + dspkl_crosspol_paths

    despeckle_and_serialize_rtc_s1(rtc_paths, dst_paths)


def run_normal_param_estimation_workflow(run_config: RunConfigData) -> None:
    """Compute normal params per burst and serialize.

    Parameters
    ----------
    run_config : RunConfigData
    """
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    for burst_id in df_inputs.jpl_burst_id.unique():
        for lookback in range(N_LOOKBACKS):
            indices_input = (df_inputs.jpl_burst_id == burst_id) & (df_inputs.input_category == 'pre')
            df_burst_input_data = df_inputs[indices_input].reset_index(drop=True)
            df_metric = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)

            copol_paths = df_burst_input_data.loc_path_copol_dspkl.tolist()
            crosspol_paths = df_burst_input_data.loc_path_crosspol_dspkl.tolist()

            # curate the paths to the correct length
            n_imgs = len(copol_paths)
            start = max(n_imgs - MODEL_CONTEXT_LENGTH - lookback - 1 + N_LOOKBACKS, 0)
            stop = min(n_imgs - lookback - 1 + N_LOOKBACKS, n_imgs - 1)
            copol_paths = copol_paths[start:stop]
            crosspol_paths = crosspol_paths[start:stop]

            output_mu_copol_l = df_metric[f'loc_path_normal_mean_delta{lookback}_copol'].tolist()
            output_mu_crosspol_l = df_metric[f'loc_path_normal_mean_delta{lookback}_crosspol'].tolist()
            output_sigma_copol_l = df_metric[f'loc_path_normal_std_delta{lookback}_copol'].tolist()
            output_sigma_crosspol_l = df_metric[f'loc_path_normal_std_delta{lookback}_crosspol'].tolist()

            assert (
                len(output_mu_copol_l)
                == len(output_mu_crosspol_l)
                == len(output_sigma_copol_l)
                == len(output_sigma_crosspol_l)
                == 1
            )

            output_mu_copol_path = output_mu_copol_l[0]
            output_mu_crosspol_path = output_mu_crosspol_l[0]
            output_sigma_copol_path = output_sigma_copol_l[0]
            output_sigma_crosspol_path = output_sigma_crosspol_l[0]

            compute_normal_params_per_burst_and_serialize(
                copol_paths,
                crosspol_paths,
                output_mu_copol_path,
                output_mu_crosspol_path,
                output_sigma_copol_path,
                output_sigma_crosspol_path,
            )


def run_burst_disturbance_workflow(run_config: RunConfigData) -> None:
    df_inputs = run_config.df_inputs
    df_burst_distmetrics = run_config.df_burst_distmetrics

    for burst_id in df_inputs.jpl_burst_id.unique():
        indices_input = df_inputs.jpl_burst_id == burst_id
        df_burst_input_data = df_inputs[indices_input].reset_index(drop=True)
        df_metric_burst = df_burst_distmetrics[df_burst_distmetrics.jpl_burst_id == burst_id].reset_index(drop=True)

        assert df_metric_burst.shape[0] == 1

        copol_paths = df_burst_input_data.loc_path_copol_dspkl.tolist()
        crosspol_paths = df_burst_input_data.loc_path_crosspol_dspkl.tolist()

        for lookback in range(N_LOOKBACKS):
            logit_mean_copol_path = df_metric_burst[f'loc_path_normal_mean_delta{lookback}_copol'].iloc[0]
            logit_mean_crosspol_path = df_metric_burst[f'loc_path_normal_mean_delta{lookback}_crosspol'].iloc[0]
            logit_sigma_copol_path = df_metric_burst[f'loc_path_normal_std_delta{lookback}_copol'].iloc[0]
            logit_sigma_crosspol_path = df_metric_burst[f'loc_path_normal_std_delta{lookback}_crosspol'].iloc[0]

            dist_path_lookback_l = df_metric_burst[f'loc_path_disturb_delta{lookback}'].tolist()
            assert len(dist_path_lookback_l) == 1
            output_dist_path = dist_path_lookback_l[0]

            n_imgs = len(copol_paths)
            start = max(n_imgs - lookback - 1, 0)
            stop = n_imgs - lookback
            copol_paths_lookback_group = copol_paths[start:stop]
            crosspol_paths_lookback_group = crosspol_paths[start:stop]
            output_metric_path = None
            if lookback == 0:
                output_metric_path = df_metric_burst[f'loc_path_metric_delta{lookback}'].iloc[0]

            # Computes the disturbance for a a single lookback group
            compute_burst_disturbance_for_lookback_group_and_serialize(
                copol_paths=copol_paths_lookback_group,
                crosspol_paths=crosspol_paths_lookback_group,
                logit_mean_copol_path=logit_mean_copol_path,
                logit_mean_crosspol_path=logit_mean_crosspol_path,
                logit_sigma_copol_path=logit_sigma_copol_path,
                logit_sigma_crosspol_path=logit_sigma_crosspol_path,
                out_dist_path=output_dist_path,
                out_metric_path=output_metric_path,
            )
        # Aggregate over lookbacks
        time_aggregated_disturbance_path = df_metric_burst['loc_path_disturb_time_aggregated'].iloc[0]
        disturbance_paths = [
            df_metric_burst[f'loc_path_disturb_delta{lookback}'].iloc[0] for lookback in range(N_LOOKBACKS)
        ]
        aggregate_burst_disturbance_over_lookbacks_and_serialize(disturbance_paths, time_aggregated_disturbance_path)


def run_dist_s1_processing_workflow(run_config: RunConfigData) -> Path:
    # Despeckle by burst
    run_despeckle_workflow(run_config)

    # Compute normal params for logit transformed data per burst
    run_normal_param_estimation_workflow(run_config)

    # Compute disturbance per burst and all possible lookbacks
    run_burst_disturbance_workflow(run_config)

    return Path('OPERA_L3_DIST_DIRECTORY')


def run_dist_s1_packaging_workflow(run_config: RunConfigData) -> Path:
    return Path('OPERA_L3_DIST_DIRECTORY')


def run_dist_s1_sas_workflow(run_config: RunConfigData) -> Path:
    _ = run_dist_s1_processing_workflow(run_config)
    _ = run_dist_s1_packaging_workflow(run_config)
    return run_config


def run_dist_s1_workflow(
    mgrs_tile_id: str, post_date: str | datetime, track_number: int, post_date_buffer_days: int
) -> Path:
    run_config = run_dist_s1_localization_workflow(mgrs_tile_id, post_date, track_number, post_date_buffer_days)
    _ = run_dist_s1_sas_workflow(run_config)

    return run_config
