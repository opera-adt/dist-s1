import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pydantic import ValidationError

from dist_s1.data_models.algoconfig_model import AlgoConfigData
from dist_s1.data_models.output_models import DistS1ProductDirectory
from dist_s1.data_models.runconfig_model import RunConfigData


def test_input_data_model_from_cropped_dataset(
    test_dir: Path, change_local_dir: Callable, test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path]
) -> None:
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    config = RunConfigData.from_product_df(df_product, dst_dir=tmp_dir)

    # Set configuration parameters via assignment
    config.apply_water_mask = False
    config.prior_dist_s1_product = None

    df = config.df_inputs

    # Check burst ids
    burst_ids_actual = df.jpl_burst_id.unique().tolist()
    burst_ids_expected = [
        'T137-292318-IW1',
        'T137-292318-IW2',
        'T137-292319-IW1',
        'T137-292319-IW2',
        'T137-292320-IW1',
        'T137-292320-IW2',
        'T137-292321-IW1',
        'T137-292321-IW2',
        'T137-292322-IW1',
        'T137-292322-IW2',
        'T137-292323-IW1',
        'T137-292323-IW2',
        'T137-292324-IW1',
        'T137-292324-IW2',
        'T137-292325-IW1',
    ]

    assert burst_ids_actual == burst_ids_expected

    ind_burst = df.jpl_burst_id == 'T137-292319-IW2'
    ind_pre = df.input_category == 'pre'
    ind_post = df.input_category == 'post'

    pre_rtc_crosspol_paths = df[ind_pre & ind_burst].loc_path_crosspol.tolist()
    pre_rtc_copol_paths = df[ind_pre & ind_burst].loc_path_copol.tolist()

    pre_rtc_crosspol_tif_filenames_actual = [Path(p).name for p in pre_rtc_crosspol_paths]
    pre_rtc_copol_tif_filenames_actual = [Path(p).name for p in pre_rtc_copol_paths]

    post_rtc_crosspol_paths = df[ind_post & ind_burst].loc_path_crosspol.tolist()
    post_rtc_copol_paths = df[ind_post & ind_burst].loc_path_copol.tolist()

    post_rtc_crosspol_tif_filenames_actual = sorted([Path(p).name for p in post_rtc_crosspol_paths])
    post_rtc_copol_tif_filenames_actual = sorted([Path(p).name for p in post_rtc_copol_paths])

    pre_rtc_copol_tif_filenames_expected = sorted(
        [
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221114T015904Z_20250222T234616Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221126T015904Z_20250228T045856Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221208T015903Z_20250301T173913Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221220T015902Z_20250302T062310Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20230101T015902Z_20250118T044147Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231109T015908Z_20231109T105731Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231121T015908Z_20231206T001302Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231203T015908Z_20231203T124004Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231215T015907Z_20231215T143038Z_S1A_30_v1.0_VV.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231227T015906Z_20231227T112953Z_S1A_30_v1.0_VV.tif',
        ]
    )

    pre_rtc_crosspol_tif_filenames_expected = sorted(
        [
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221114T015904Z_20250222T234616Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221126T015904Z_20250228T045856Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221208T015903Z_20250301T173913Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20221220T015902Z_20250302T062310Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20230101T015902Z_20250118T044147Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231109T015908Z_20231109T105731Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231121T015908Z_20231206T001302Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231203T015908Z_20231203T124004Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231215T015907Z_20231215T143038Z_S1A_30_v1.0_VH.tif',
            'OPERA_L2_RTC-S1_T137-292319-IW2_20231227T015906Z_20231227T112953Z_S1A_30_v1.0_VH.tif',
        ]
    )

    post_rtc_copol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20250102T015901Z_20250102T190143Z_S1A_30_v1.0_VV.tif'
    ]

    post_rtc_crosspol_tif_filenames_expected = [
        'OPERA_L2_RTC-S1_T137-292319-IW2_20250102T015901Z_20250102T190143Z_S1A_30_v1.0_VH.tif'
    ]

    assert pre_rtc_crosspol_tif_filenames_actual == pre_rtc_crosspol_tif_filenames_expected
    assert pre_rtc_copol_tif_filenames_actual == pre_rtc_copol_tif_filenames_expected
    assert post_rtc_copol_tif_filenames_actual == post_rtc_copol_tif_filenames_expected
    assert post_rtc_crosspol_tif_filenames_actual == post_rtc_crosspol_tif_filenames_expected

    # Check acquisition dates for 1 burst
    pre_acq_dts = np.array(df[ind_pre & ind_burst].acq_dt.dt.to_pydatetime())
    post_acq_dts = np.array(df[ind_post & ind_burst].acq_dt.dt.to_pydatetime())

    pre_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in pre_acq_dts]
    post_acq_dts_str_actual = [dt.strftime('%Y%m%dT%H%M%S') for dt in post_acq_dts]

    pre_acq_dts_str_expected = [
        '20221114T015904',
        '20221126T015904',
        '20221208T015903',
        '20221220T015902',
        '20230101T015902',
        '20231109T015908',
        '20231121T015908',
        '20231203T015908',
        '20231215T015907',
        '20231227T015906',
    ]
    post_acq_dts_str_expected = ['20250102T015901']

    assert pre_acq_dts_str_actual == pre_acq_dts_str_expected
    assert post_acq_dts_str_actual == post_acq_dts_str_expected

    shutil.rmtree(tmp_dir)


def test_confirmation_property_behavior(
    test_dir: Path,
    test_opera_golden_cropped_dataset_dict: Path,
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    """Test that confirmation property correctly reflects prior_dist_s1_product state."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    golden_dataset_path = test_opera_golden_cropped_dataset_dict['current']
    product_dir = DistS1ProductDirectory.from_product_path(golden_dataset_path)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    # Test 1: confirmation is False when prior_dist_s1_product is None (default)
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.check_input_paths = False  # Bypass file path validation
    config.apply_water_mask = False
    # Default state: prior_dist_s1_product is None, so confirmation should be False
    assert config.prior_dist_s1_product is None
    assert config.confirmation is False

    # Test 2: confirmation is True when prior_dist_s1_product is set
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
        prior_dist_s1_product=product_dir,
    )
    config.check_input_paths = False  # Bypass file path validation
    config.apply_water_mask = False
    # With prior_dist_s1_product set, confirmation should be True
    assert config.prior_dist_s1_product == product_dir
    assert config.confirmation is True

    # Test 3: confirmation changes when prior_dist_s1_product is modified
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.check_input_paths = False
    config.apply_water_mask = False

    # Initially confirmation should be False
    assert config.confirmation is False

    # Setting prior_dist_s1_product should make confirmation True
    config.prior_dist_s1_product = product_dir
    assert config.confirmation is True

    # Unsetting prior_dist_s1_product should make confirmation False again
    config.prior_dist_s1_product = None
    assert config.confirmation is False

    shutil.rmtree(tmp_dir)


def test_lookback_strategy_validation(
    test_dir: Path, change_local_dir: Callable, test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path]
) -> None:
    """Test that lookback_strategy only accepts 'multi_window' and 'immediate_lookback' values."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    # Test 1: Valid lookback_strategy values should succeed
    valid_strategies = ['multi_window']
    for strategy in valid_strategies:
        config = RunConfigData.from_product_df(
            df_product,
            dst_dir=tmp_dir,
            lookback_strategy=strategy,
        )
        config.apply_water_mask = False
        config.prior_dist_s1_product = None
        assert config.algo_config.lookback_strategy == strategy

    # Test 2: Invalid lookback_strategy values should fail
    invalid_strategies = ['invalid_strategy', 'single_window', 'delayed_lookback', 'multi', 'immediate']
    for strategy in invalid_strategies:
        with pytest.raises(ValidationError, match='String should match pattern'):
            RunConfigData.from_product_df(
                df_product,
                dst_dir=tmp_dir,
                lookback_strategy=strategy,
            )

    # Test 3: Default value should be 'multi_window'
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.apply_water_mask = False
    config.prior_dist_s1_product = None
    assert config.algo_config.lookback_strategy == 'multi_window'

    shutil.rmtree(tmp_dir)


def test_device_resolution(
    test_dir: Path, change_local_dir: Callable, test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path]
) -> None:
    """Test that device='best' gets properly resolved to the actual available device."""
    import torch

    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    # Test that device='best' gets resolved to an actual device
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
    )
    config.apply_water_mask = False
    config.prior_dist_s1_product = None

    # Set n_workers to 1 first to avoid validation errors with GPU devices
    config.algo_config.n_workers_for_norm_param_estimation = 1
    config.algo_config.device = 'best'

    # Verify that 'best' was resolved to an actual device
    assert config.algo_config.device in ['cpu', 'cuda', 'mps'], (
        f"Device should be one of ['cpu', 'cuda', 'mps'], got {config.algo_config.device}"
    )

    # Test that explicit device values work correctly
    for device in ['cpu', 'cuda', 'mps']:
        try:
            config = RunConfigData.from_product_df(
                df_product,
                dst_dir=tmp_dir,
            )
            config.apply_water_mask = False
            config.prior_dist_s1_product = None
            if device in ['cuda', 'mps']:
                config.algo_config.n_workers_for_norm_param_estimation = 1  # Required for GPU devices
            config.algo_config.device = device
            assert config.algo_config.device == device
        except ValidationError as e:
            # It's okay for cuda/mps to fail if not available
            if device == 'cuda' and not torch.cuda.is_available():
                assert 'CUDA is not available' in str(e)
            elif device == 'mps' and not torch.backends.mps.is_available():
                assert 'MPS is not available' in str(e)
            else:
                raise

    shutil.rmtree(tmp_dir)


def test_algorithm_config_from_yaml(
    test_dir: Path,
    test_algo_config_path: Path,
    runconfig_yaml_template: str,
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    """Test that algorithm parameters are properly loaded from external YAML file."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create a main runconfig YAML file that references the algorithm config
    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    main_config_path = tmp_dir / 'test_main_config.yml'
    main_config_content = runconfig_yaml_template.format(
        pre_rtc_copol=df_product[df_product.input_category == 'pre'].loc_path_copol.tolist(),
        pre_rtc_crosspol=df_product[df_product.input_category == 'pre'].loc_path_crosspol.tolist(),
        post_rtc_copol=df_product[df_product.input_category == 'post'].loc_path_copol.tolist(),
        post_rtc_crosspol=df_product[df_product.input_category == 'post'].loc_path_crosspol.tolist(),
        dst_dir=tmp_dir,
        algo_config_path=test_algo_config_path,
        additional_params='',
    )
    with Path.open(main_config_path, 'w') as f:
        f.write(main_config_content)

    # Load configuration and verify that algorithm parameters were applied
    config = RunConfigData.from_yaml(str(main_config_path))

    # Verify that the algorithm parameters were actually applied
    assert config.algo_config.interpolation_method == 'bilinear'
    assert config.algo_config.low_confidence_alert_threshold == 4.2
    assert config.algo_config.high_confidence_alert_threshold == 6.8
    assert config.algo_config.device == 'cpu'
    assert config.algo_config.apply_despeckling is False
    assert config.algo_config.apply_logit_to_inputs is False
    assert config.algo_config.memory_strategy == 'low'
    assert config.algo_config.batch_size_for_norm_param_estimation == 64

    shutil.rmtree(tmp_dir)


def test_algorithm_config_validation_errors(
    test_dir: Path,
    test_algo_config_invalid_path: Path,
    runconfig_yaml_template: str,
    change_local_dir: Callable,
    test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path],
) -> None:
    """Test that validation errors are properly raised when invalid algorithm parameter values are provided."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Direct AlgoConfigData loading should fail with invalid parameters
    with pytest.raises(ValidationError, match=r'(device|interpolation_method|memory_strategy)'):
        AlgoConfigData.from_yaml(test_algo_config_invalid_path)

    # Test 2: RunConfigData loading should also fail when using an invalid algorithm config
    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    main_config_path = tmp_dir / 'test_main_config.yml'
    main_config_content = runconfig_yaml_template.format(
        pre_rtc_copol=df_product[df_product.input_category == 'pre'].loc_path_copol.tolist(),
        pre_rtc_crosspol=df_product[df_product.input_category == 'pre'].loc_path_crosspol.tolist(),
        post_rtc_copol=df_product[df_product.input_category == 'post'].loc_path_copol.tolist(),
        post_rtc_crosspol=df_product[df_product.input_category == 'post'].loc_path_crosspol.tolist(),
        dst_dir=tmp_dir,
        algo_config_path=test_algo_config_invalid_path,
        additional_params='',
    )
    with Path.open(main_config_path, 'w') as f:
        f.write(main_config_content)

    # Should raise ValidationError when trying to load RunConfigData with invalid algorithm config
    with pytest.raises(ValidationError, match=r'(device|interpolation_method|memory_strategy)'):
        RunConfigData.from_yaml(str(main_config_path))

    # Test 3: Verify specific field validation messages using match patterns
    # Test individual invalid field values by creating minimal config objects

    # Test invalid device - matches "device" field and the invalid value
    with pytest.raises(ValidationError, match=r'device'):
        AlgoConfigData(device='invalid_device')

    # Test invalid interpolation_method - matches field name
    with pytest.raises(ValidationError, match=r'interpolation_method'):
        AlgoConfigData(interpolation_method='invalid_method')

    # Test invalid memory_strategy - matches field name
    with pytest.raises(ValidationError, match=r'memory_strategy'):
        AlgoConfigData(memory_strategy='invalid_strategy')

    shutil.rmtree(tmp_dir)


def test_model_dtype_device_compatibility_warning(
    test_dir: Path, change_local_dir: Callable, test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path]
) -> None:
    """Test that warnings are issued when bfloat16 is used with non-GPU devices."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    # Test 1: bfloat16 with CPU should issue warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        config = RunConfigData.from_product_df(
            df_product,
            dst_dir=tmp_dir,
        )
        config.apply_water_mask = False
        config.prior_dist_s1_product = None
        config.algo_config.model_dtype = 'bfloat16'
        config.algo_config.device = 'cpu'

        # Check that warning was issued
        warning_messages = [str(warning.message) for warning in w]
        dtype_warnings = [
            msg for msg in warning_messages if 'bfloat16' in msg and 'only supported on GPU devices' in msg
        ]
        assert len(dtype_warnings) > 0, 'Expected warning for bfloat16 with CPU device'

    # Test 2: bfloat16 with MPS should issue warning
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            config = RunConfigData.from_product_df(
                df_product,
                dst_dir=tmp_dir,
            )
            config.apply_water_mask = False
            config.prior_dist_s1_product = None
            config.algo_config.n_workers_for_norm_param_estimation = 1  # Required for MPS
            config.algo_config.model_dtype = 'bfloat16'
            config.algo_config.device = 'mps'

            # Check that warning was issued
            warning_messages = [str(warning.message) for warning in w]
            dtype_warnings = [
                msg for msg in warning_messages if 'bfloat16' in msg and 'only supported on GPU devices' in msg
            ]
            assert len(dtype_warnings) > 0, 'Expected warning for bfloat16 with MPS device'
    except ValidationError as e:
        # It's okay if MPS is not available
        if 'MPS is not available' in str(e):
            pass
        else:
            raise

    # Test 3: bfloat16 with CUDA should NOT issue warning
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            config = RunConfigData.from_product_df(
                df_product,
                dst_dir=tmp_dir,
            )
            config.apply_water_mask = False
            config.prior_dist_s1_product = None
            config.algo_config.n_workers_for_norm_param_estimation = 1  # Required for CUDA
            config.algo_config.model_dtype = 'bfloat16'
            config.algo_config.device = 'cuda'

            # Check that NO warning was issued for dtype compatibility
            warning_messages = [str(warning.message) for warning in w]
            dtype_warnings = [
                msg for msg in warning_messages if 'bfloat16' in msg and 'only supported on GPU devices' in msg
            ]
            assert len(dtype_warnings) == 0, 'Should not have warning for bfloat16 with CUDA device'
    except ValidationError as e:
        # It's okay if CUDA is not available
        if 'CUDA is not available' in str(e):
            pass
        else:
            raise

    # Test 4: float32 with any device should NOT issue warning
    for device in ['cpu', 'mps']:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                config = RunConfigData.from_product_df(
                    df_product,
                    dst_dir=tmp_dir,
                )
                config.apply_water_mask = False
                config.prior_dist_s1_product = None
                if device in ['mps', 'cuda']:
                    config.algo_config.n_workers_for_norm_param_estimation = 1  # Required for GPU devices
                config.algo_config.model_dtype = 'float32'
                config.algo_config.device = device

                # Check that NO warning was issued for dtype compatibility
                warning_messages = [str(warning.message) for warning in w]
                dtype_warnings = [
                    msg for msg in warning_messages if 'bfloat16' in msg and 'only supported on GPU devices' in msg
                ]
                assert len(dtype_warnings) == 0, f'Should not have warning for float32 with {device} device'
        except ValidationError as e:
            # It's okay if the device is not available
            if 'is not available' in str(e):
                pass
            else:
                raise

    # Test 5: Test with AlgoConfigData directly
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = AlgoConfigData(model_dtype='bfloat16', device='cpu')

        # Check that warning was issued
        warning_messages = [str(warning.message) for warning in w]
        dtype_warnings = [
            msg for msg in warning_messages if 'bfloat16' in msg and 'only supported on GPU devices' in msg
        ]
        assert len(dtype_warnings) > 0, 'Expected warning for bfloat16 with CPU device in AlgoConfigData'

    shutil.rmtree(tmp_dir)


def test_model_path_validation(test_dir: Path, change_local_dir: Callable) -> None:
    """Test that validation errors are properly raised when model paths don't exist."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create a real config file for testing
    real_config_path = tmp_dir / 'real_config.json'
    real_config_path.write_text('{"model_type": "test"}')

    # Create a real weights file for testing
    real_weights_path = tmp_dir / 'real_weights.pth'
    real_weights_path.write_text('fake_weights_content')

    # Test 1: Non-existent model_cfg_path should raise ValidationError
    with pytest.raises(ValidationError, match=r'Model config path does not exist'):
        AlgoConfigData(model_cfg_path='/non/existent/config.json')

    # Test 2: Non-existent model_wts_path should raise ValidationError
    with pytest.raises(ValidationError, match=r'Model weights path does not exist'):
        AlgoConfigData(model_wts_path='/non/existent/weights.pth')

    # Test 3: Both non-existent paths should raise ValidationError
    with pytest.raises(ValidationError, match=r'Model config path does not exist'):
        AlgoConfigData(model_cfg_path='/non/existent/config.json', model_wts_path='/non/existent/weights.pth')

    # Test 4: Directory instead of file for model_cfg_path should raise ValidationError
    with pytest.raises(ValidationError, match=r'Model config path is not a file'):
        AlgoConfigData(model_cfg_path=str(tmp_dir))

    # Test 5: Directory instead of file for model_wts_path should raise ValidationError
    with pytest.raises(ValidationError, match=r'Model weights path is not a file'):
        AlgoConfigData(model_wts_path=str(tmp_dir))

    # Test 6: Valid paths should work (no ValidationError)
    config = AlgoConfigData(model_cfg_path=str(real_config_path), model_wts_path=str(real_weights_path))
    assert config.model_cfg_path == real_config_path
    assert config.model_wts_path == real_weights_path

    # Test 7: None values should be allowed (no ValidationError)
    config_with_none = AlgoConfigData(model_cfg_path=None, model_wts_path=None)
    assert config_with_none.model_cfg_path is None
    assert config_with_none.model_wts_path is None

    # Test 8: String paths should be converted to Path objects
    config_with_strings = AlgoConfigData(model_cfg_path=str(real_config_path), model_wts_path=str(real_weights_path))
    assert isinstance(config_with_strings.model_cfg_path, Path)
    assert isinstance(config_with_strings.model_wts_path, Path)

    shutil.rmtree(tmp_dir)


def test_algo_config_path_programmatic_loading(
    test_dir: Path, change_local_dir: Callable, test_10SGD_dist_s1_inputs_parquet_dict: dict[str, Path]
) -> None:
    """Test that algorithm config is automatically loaded when algo_config_path is provided programmatically."""
    change_local_dir(test_dir)

    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom algorithm config file with non-default parameters
    algo_config_content = """
algo_config:
  device: cpu
  memory_strategy: low
  low_confidence_alert_threshold: 4.5
  high_confidence_alert_threshold: 7.0
  apply_despeckling: false
  apply_logit_to_inputs: false
  n_workers_for_despeckling: 2
  batch_size_for_norm_param_estimation: 64
  stride_for_norm_param_estimation: 8
  interpolation_method: nearest
  tqdm_enabled: false
"""

    algo_config_path = tmp_dir / 'test_algo_config.yml'
    with algo_config_path.open('w') as f:
        f.write(algo_config_content)

    parquet_path = test_10SGD_dist_s1_inputs_parquet_dict['current']
    df_product = gpd.read_parquet(parquet_path)

    # Test that algorithm config is loaded automatically when creating RunConfigData programmatically
    config = RunConfigData.from_product_df(
        df_product,
        dst_dir=tmp_dir,
        apply_water_mask=False,
    )

    # Set algo_config_path to trigger automatic loading
    config.algo_config_path = algo_config_path

    # Verify that the algorithm parameters from the config file were correctly applied
    assert config.algo_config.device == 'cpu'
    assert config.algo_config.memory_strategy == 'low'
    assert config.algo_config.low_confidence_alert_threshold == 4.5
    assert config.algo_config.high_confidence_alert_threshold == 7.0
    assert config.algo_config.apply_despeckling is False
    assert config.algo_config.apply_logit_to_inputs is False
    assert config.algo_config.n_workers_for_despeckling == 2
    assert config.algo_config.batch_size_for_norm_param_estimation == 64
    assert config.algo_config.stride_for_norm_param_estimation == 8
    assert config.algo_config.interpolation_method == 'nearest'
    assert config.algo_config.tqdm_enabled is False

    # Test 2: Verify that non-existent algo_config_path raises ValidationError
    with pytest.raises(ValidationError, match=r'Algorithm config path does not exist'):
        config = RunConfigData.from_product_df(
            df_product,
            dst_dir=tmp_dir,
            apply_water_mask=False,
        )
        config.algo_config_path = tmp_dir / 'non_existent_config.yml'

    # Test 3: Verify that directory instead of file raises ValidationError
    with pytest.raises(ValidationError, match=r'Algorithm config path is not a file'):
        config = RunConfigData.from_product_df(
            df_product,
            dst_dir=tmp_dir,
            apply_water_mask=False,
        )
        config.algo_config_path = tmp_dir

    # Test 4: Test that algo_config_path can be provided during object creation
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')

        config = RunConfigData.from_product_df(
            df_product,
            dst_dir=tmp_dir,
            apply_water_mask=False,
        )
        # Set algo_config_path directly (this should trigger the validator)
        config.algo_config_path = algo_config_path

        # Verify at least one parameter was loaded correctly
        assert config.algo_config.low_confidence_alert_threshold == 4.5

    shutil.rmtree(tmp_dir)
