import multiprocessing as mp
import warnings
from datetime import datetime
from pathlib import Path, PosixPath

import geopandas as gpd
import pandas as pd
import torch
import yaml
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id
from dist_s1_enumerator.data_models import dist_s1_loc_input_schema
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from distmetrics import get_device
from pandera.pandas import check_input
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from yaml import Dumper

from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_DESPECKLING,
    DEFAULT_APPLY_LOGIT_TO_INPUTS,
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_CHECK_INPUT_PATHS,
    DEFAULT_CONF_THRESH,
    DEFAULT_CONF_UPPER_LIM,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_DEVICE,
    DEFAULT_DST_DIR,
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    DEFAULT_INTERPOLATION_METHOD,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_MAX_OBS_NUM_YEAR,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_MEMORY_STRATEGY,
    DEFAULT_METRIC_VALUE_UPPER_LIM,
    DEFAULT_MODEL_COMPILATION,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODERATE_CONFIDENCE_THRESHOLD,
    DEFAULT_NODAYLIMIT,
    DEFAULT_N_WORKERS_FOR_DESPECKLING,
    DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_TQDM_ENABLED,
    DEFAULT_USE_DATE_ENCODING,
)
from dist_s1.data_models.output_models import ProductDirectoryData, ProductNameData
from dist_s1.water_mask import water_mask_control_flow


def posix_path_encoder(dumper: Dumper, data: PosixPath) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))


def none_encoder(dumper: Dumper, _: None) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:null', '')


yaml.add_representer(PosixPath, posix_path_encoder)
yaml.add_representer(type(None), none_encoder)


def get_opera_id(opera_rtc_s1_tif_path: Path | str) -> str:
    stem = Path(opera_rtc_s1_tif_path).stem
    tokens = stem.split('_')
    opera_id = '_'.join(tokens[:-1])
    return opera_id


def get_burst_id(opera_rtc_s1_path: Path | str) -> str:
    opera_rtc_s1_path = Path(opera_rtc_s1_path)
    tokens = opera_rtc_s1_path.name.split('_')
    return tokens[3]


def get_track_number(opera_rtc_s1_path: Path | str) -> str:
    burst_id = get_burst_id(opera_rtc_s1_path)
    track_number_str = burst_id.split('-')[0]
    track_number = int(track_number_str[1:])
    return track_number


def get_acquisition_datetime(opera_rtc_s1_path: Path) -> datetime:
    tokens = opera_rtc_s1_path.name.split('_')
    try:
        return pd.Timestamp(tokens[4], tz='UTC')
    except ValueError:
        raise ValueError(f"Datetime token in filename '{opera_rtc_s1_path.name}' is not correctly formatted.")


def check_filename_format(filename: str, polarization: str) -> None:
    if polarization not in ['crosspol', 'copol']:
        raise ValueError(f"Polarization '{polarization}' is not valid; must be in ['crosspol', 'copol']")

    tokens = filename.split('_')
    if len(tokens) != 10:
        raise ValueError(f"File '{filename}' does not have 10 tokens")
    if tokens[0] != 'OPERA':
        raise ValueError(f"File '{filename}' first token is not 'OPERA'")
    if tokens[1] != 'L2':
        raise ValueError(f"File '{filename}' second token is not 'L2'")
    if tokens[2] != 'RTC-S1':
        raise ValueError(f"File '{filename}' third token is not 'RTC-S1'")
    if polarization == 'copol' and not (filename.endswith('_VV.tif') or filename.endswith('_HH.tif')):
        raise ValueError(f"File '{filename}' should end with '_VV.tif' or '_HH.tif' because it is copolarization")
    elif polarization == 'crosspol' and not (filename.endswith('_VH.tif') or filename.endswith('_HV.tif')):
        raise ValueError(f"File '{filename}' should end with '_VH.tif' or '_HV.tif' because it is crosspolarization")
    return True


def check_dist_product_filename_format(filename: str) -> None:
    valid_suffixes = (
        'GEN-DIST-STATUS.tif',
        'GEN-METRIC-MAX.tif',
        'GEN-DIST-CONF.tif',
        'GEN-DIST-DATE.tif',
        'GEN-DIST-COUNT.tif',
        'GEN-DIST-PERC.tif',
        'GEN-DIST-DUR.tif',
        'GEN-DIST-LAST-DATE.tif',
        'GEN-DIST-STATUS.tif',
    )

    tokens = filename.split('_')
    if len(tokens) != 10:
        raise ValueError(f"File '{filename}' does not have 10 tokens")
    if tokens[0] != 'OPERA':
        raise ValueError(f"File '{filename}' first token is not 'OPERA'")
    if tokens[1] != 'L3':
        raise ValueError(f"File '{filename}' second token is not 'L3'")
    if tokens[2] != 'DIST-ALERT-S1':
        raise ValueError(f"File '{filename}' third token is not 'DIST-ALERT-S1'")
    if not any(filename.endswith(suffix) for suffix in valid_suffixes):
        raise ValueError(f"Filename '{filename}' must be a valid DIST-ALERT-S1 product: {valid_suffixes}")
    return True


class AlgoConfigData(BaseModel):
    """Base class containing algorithm configuration parameters."""

    device: str = Field(
        default=DEFAULT_DEVICE,
        pattern='^(best|cuda|mps|cpu)$',
    )
    memory_strategy: str | None = Field(
        default=DEFAULT_MEMORY_STRATEGY,
        pattern='^(high|low)$',
    )
    tqdm_enabled: bool = Field(default=DEFAULT_TQDM_ENABLED)
    n_workers_for_norm_param_estimation: int = Field(
        default=DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
    )
    # Batch size for transformer model.
    batch_size_for_norm_param_estimation: int = Field(
        default=DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
    )
    # Stride for transformer model.
    stride_for_norm_param_estimation: int = Field(
        default=DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
        le=16,
    )
    n_workers_for_despeckling: int = Field(
        default=DEFAULT_N_WORKERS_FOR_DESPECKLING,
        ge=1,
    )
    lookback_strategy: str = Field(
        default=DEFAULT_LOOKBACK_STRATEGY,
        pattern='^(multi_window|immediate_lookback)$',
    )
    # Flag to enable optimizations. False, load the model and use it.
    # True, load the model and compile for CPU or GPU
    model_compilation: bool = Field(default=DEFAULT_MODEL_COMPILATION)
    max_pre_imgs_per_burst_mw: list[int] = Field(
        default=DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
        description='Max number of pre-images per burst within each window',
    )
    delta_lookback_days_mw: list[int] = Field(
        default=DEFAULT_DELTA_LOOKBACK_DAYS_MW,
        description='Delta lookback days for each window relative to post-image acquisition date',
    )
    # This is where default thresholds are set!
    moderate_confidence_threshold: float = Field(default=DEFAULT_MODERATE_CONFIDENCE_THRESHOLD, ge=0.0, le=15.0)
    high_confidence_threshold: float = Field(default=DEFAULT_HIGH_CONFIDENCE_THRESHOLD, ge=0.0, le=15.0)
    nodaylimit: int = Field(default=DEFAULT_NODAYLIMIT)
    max_obs_num_year: int = Field(default=DEFAULT_MAX_OBS_NUM_YEAR, description='Max observation number per year')
    conf_upper_lim: int = Field(default=DEFAULT_CONF_UPPER_LIM, description='Confidence upper limit')
    conf_thresh: float = Field(default=DEFAULT_CONF_THRESH, description='Confidence threshold')
    metric_value_upper_lim: float = Field(default=DEFAULT_METRIC_VALUE_UPPER_LIM, description='Metric upper limit')
    base_date_for_confirmation: datetime = Field(
        default=datetime(2020, 12, 31),
        description='Reference date used to calculate the number of days in confirmation process',
    )
    # model_source == "external" means use externally supplied paths for weights and config
    # otherwise use distmetrics.model_load.ALLOWED_MODELS for other models
    model_source: str | None = DEFAULT_MODEL_SOURCE
    model_cfg_path: Path | str | None = None
    model_wts_path: Path | str | None = None
    # Use logit transform
    apply_logit_to_inputs: bool = Field(default=DEFAULT_APPLY_LOGIT_TO_INPUTS)
    # Use despeckling
    apply_despeckling: bool = Field(default=DEFAULT_APPLY_DESPECKLING)
    interpolation_method: str = Field(
        default=DEFAULT_INTERPOLATION_METHOD,
        pattern='^(nearest|bilinear|none)$',
    )
    # Model data type for inference
    model_dtype: str = Field(
        default=DEFAULT_MODEL_DTYPE,
        pattern='^(float32|bfloat16|float)$',
        description='Data type for model inference. Note: bfloat16 is only supported on GPU devices.',
    )
    # Use date encoding in processing
    use_date_encoding: bool = Field(
        default=DEFAULT_USE_DATE_ENCODING,
        description='Whether to use acquisition date encoding in processing',
    )
    # Validate assignments to all fields
    model_config = ConfigDict(validate_assignment=True)

    @field_validator('model_source', mode='before')
    def validate_model_source(cls, v: str | None) -> str:
        if v is None:
            return 'transformer_optimized'
        return v

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> 'AlgoConfigData':
        """Load algorithm configuration from a YAML file."""
        yaml_file = Path(yaml_file)
        with yaml_file.open() as file:
            data = yaml.safe_load(file)
            if 'algo_config' in data:
                algo_data = data['algo_config']
            else:
                algo_data = data.get('algorithm_config', data)

        # Create instance and warn about loaded parameters
        obj = cls(**algo_data)

        # Issue warnings for all parameters that were loaded from file
        for field_name, field_value in algo_data.items():
            if hasattr(obj, field_name):
                warnings.warn(
                    f"Algorithm parameter '{field_name}' set to {field_value} from external config file: {yaml_file}",
                    UserWarning,
                    stacklevel=2,
                )

        return obj

    @field_validator('memory_strategy')
    def validate_memory_strategy(cls, memory_strategy: str) -> str:
        if memory_strategy not in ['high', 'low']:
            raise ValueError("Memory strategy must be in ['high', 'low']")
        return memory_strategy

    @field_validator('device', mode='before')
    def validate_device(cls, device: str) -> str:
        """Validate and set the device. None or 'none' will be converted to the default device."""
        if device == 'best':
            device = get_device()
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available even though device is set to cuda')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is not available even though device is set to mps')
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Device '{device}' must be one of: cpu, cuda, mps")
        return device

    @field_validator(
        'n_workers_for_despeckling',
        'n_workers_for_norm_param_estimation',
    )
    def validate_n_workers(cls, n_workers: int, info: ValidationInfo) -> int:
        if n_workers > mp.cpu_count():
            warnings.warn(
                f'{info.field_name} ({n_workers}) is greater than the number of CPUs ({mp.cpu_count()}), using latter.',
                UserWarning,
            )
            n_workers = mp.cpu_count()
        return n_workers

    @field_validator('moderate_confidence_threshold')
    def validate_moderate_threshold(cls, moderate_threshold: float, info: ValidationInfo) -> float:
        """Validate that moderate_confidence_threshold is less than high_confidence_threshold."""
        high_threshold = info.data.get('high_confidence_threshold')
        if high_threshold is not None and moderate_threshold >= high_threshold:
            raise ValueError(
                f'moderate_confidence_threshold ({moderate_threshold}) must be less than '
                f'high_confidence_threshold ({high_threshold})'
            )
        return moderate_threshold

    @field_validator('model_dtype')
    def validate_model_dtype(cls, model_dtype: str) -> str:
        """Validate that model_dtype is a supported data type."""
        valid_dtypes = ['float32', 'bfloat16', 'float']
        if model_dtype not in valid_dtypes:
            raise ValueError(f"model_dtype '{model_dtype}' must be one of: {valid_dtypes}")
        return model_dtype

    @field_validator('model_cfg_path', mode='before')
    def validate_model_cfg_path(cls, model_cfg_path: Path | str | None) -> Path | None:
        """Validate that model_cfg_path exists if provided."""
        if model_cfg_path is None:
            return None
        model_cfg_path = Path(model_cfg_path) if isinstance(model_cfg_path, str) else model_cfg_path
        if not model_cfg_path.exists():
            raise ValueError(f'Model config path does not exist: {model_cfg_path}')
        if not model_cfg_path.is_file():
            raise ValueError(f'Model config path is not a file: {model_cfg_path}')
        return model_cfg_path

    @field_validator('model_wts_path', mode='before')
    def validate_model_wts_path(cls, model_wts_path: Path | str | None) -> Path | None:
        """Validate that model_wts_path exists if provided."""
        if model_wts_path is None:
            return None
        model_wts_path = Path(model_wts_path) if isinstance(model_wts_path, str) else model_wts_path
        if not model_wts_path.exists():
            raise ValueError(f'Model weights path does not exist: {model_wts_path}')
        if not model_wts_path.is_file():
            raise ValueError(f'Model weights path is not a file: {model_wts_path}')
        return model_wts_path

    @model_validator(mode='after')
    def validate_model_compilation_device_compatibility(self) -> 'AlgoConfigData':
        """Validate that model_compilation is not True when device is 'mps'."""
        if self.model_compilation is True and self.device == 'mps':
            raise ValueError('model_compilation cannot be True when device is set to mps')
        return self

    @model_validator(mode='after')
    def validate_model_dtype_device_compatibility(self) -> 'AlgoConfigData':
        """Warn when bfloat16 is used with non-GPU devices."""
        if self.model_dtype == 'bfloat16' and self.device not in ['cuda']:
            warnings.warn(
                f"model_dtype 'bfloat16' is only supported on GPU devices. "
                f"Current device is '{self.device}'. "
                f"Consider using 'float32' or 'float' for CPU/MPS devices.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def to_yml(self, yaml_file: str | Path) -> None:
        """Save algorithm configuration to a YAML file."""
        config_dict = self.model_dump()
        yml_dict = {'algo_config': config_dict}

        # Write to YAML file
        yaml_file = Path(yaml_file)
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)


class RunConfigData(AlgoConfigData):
    pre_rtc_copol: list[Path | str]
    pre_rtc_crosspol: list[Path | str]
    post_rtc_copol: list[Path | str]
    post_rtc_crosspol: list[Path | str]
    prior_dist_s1_product: ProductDirectoryData | None = None
    mgrs_tile_id: str
    dst_dir: Path | str = DEFAULT_DST_DIR
    water_mask_path: Path | str | None = None
    apply_water_mask: bool = Field(default=DEFAULT_APPLY_WATER_MASK)
    check_input_paths: bool = DEFAULT_CHECK_INPUT_PATHS
    product_dst_dir: Path | str | None = None
    bucket: str | None = None
    bucket_prefix: str | None = None
    # Path to external algorithm config file
    algo_config_path: Path | str | None = None

    # Private attributes that are associated to properties
    _burst_ids: list[str] | None = None
    _df_inputs: pd.DataFrame | None = None
    _df_prior_dist_products: pd.DataFrame | None = None
    _df_burst_distmetrics: pd.DataFrame | None = None
    _df_mgrs_burst_lut: gpd.GeoDataFrame | None = None
    _product_name: ProductNameData | None = None
    _product_data_model: ProductDirectoryData | None = None
    _min_acq_date: datetime | None = None
    _processing_datetime: datetime | None = None
    # Validate assignments to all fields
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, yaml_file: str, fields_to_overwrite: dict | None = None) -> 'RunConfigData':
        """Load configuration from a YAML file and initialize RunConfigModel."""
        with Path.open(yaml_file) as file:
            data = yaml.safe_load(file)
            runconfig_data = data['run_config']
        if fields_to_overwrite is not None:
            runconfig_data.update(fields_to_overwrite)

        # Handle algorithm config if specified
        algo_config_path = runconfig_data.get('algo_config_path')
        if algo_config_path is not None:
            algo_config = AlgoConfigData.from_yaml(algo_config_path)

            # Override algorithm parameters with those from external file
            # Only override if the parameter is not explicitly set in the main config
            algo_dict = algo_config.model_dump()
            for field_name, field_value in algo_dict.items():
                if field_name not in runconfig_data:
                    runconfig_data[field_name] = field_value
                    warnings.warn(
                        f"Algorithm parameter '{field_name}' inherited from external config: {algo_config_path}",
                        UserWarning,
                        stacklevel=2,
                    )

        obj = cls(**runconfig_data)
        return obj

    @field_validator('pre_rtc_copol', 'pre_rtc_crosspol', 'post_rtc_copol', 'post_rtc_crosspol', mode='before')
    def convert_to_paths(cls, values: list[Path | str], info: ValidationInfo) -> list[Path]:
        """Convert all values to Path objects."""
        paths = [Path(value) if isinstance(value, str) else value for value in values]
        if info.data.get('check_input_paths', True):
            bad_paths = []
            for path in paths:
                if not path.exists():
                    bad_paths.append(path)
            if bad_paths:
                bad_paths_str = 'The following paths do not exist: ' + ', '.join(str(path) for path in bad_paths)
                raise ValueError(bad_paths_str)
        return paths

    @field_validator('dst_dir', mode='before')
    def validate_dst_dir(cls, dst_dir: Path | str | None, info: ValidationInfo) -> Path:
        dst_dir = Path(dst_dir) if isinstance(dst_dir, str) else dst_dir
        if dst_dir.exists() and not dst_dir.is_dir():
            raise ValidationError(f"Path '{dst_dir}' exists but is not a directory")
        dst_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir.resolve()

    @field_validator('product_dst_dir', mode='before')
    def validate_product_dst_dir(cls, product_dst_dir: Path | str | None, info: ValidationInfo) -> Path:
        if product_dst_dir is None:
            product_dst_dir = Path(info.data['dst_dir'])
        elif isinstance(product_dst_dir, str):
            product_dst_dir = Path(product_dst_dir)
        if product_dst_dir.exists() and not product_dst_dir.is_dir():
            raise ValidationError(f"Path '{product_dst_dir}' exists but is not a directory")
        product_dst_dir.mkdir(parents=True, exist_ok=True)
        return product_dst_dir.resolve()

    @field_validator('pre_rtc_crosspol', 'post_rtc_crosspol')
    def check_matching_lengths_copol_and_crosspol(
        cls: type['RunConfigData'], rtc_crosspol: list[Path], info: ValidationInfo
    ) -> list[Path]:
        """Ensure pre_rtc_copol and pre_rtc_crosspol have the same length."""
        key = 'pre_rtc_copol' if info.field_name == 'pre_rtc_crosspol' else 'post_rtc_copol'
        rtc_copol = info.data.get(key)
        if rtc_copol is not None and len(rtc_copol) != len(rtc_crosspol):
            raise ValueError("The lists 'pre_rtc_copol' and 'pre_rtc_crosspol' must have the same length.")
        return rtc_crosspol

    @field_validator('pre_rtc_copol', 'pre_rtc_crosspol', 'post_rtc_copol', 'post_rtc_crosspol')
    def check_filename_format(cls, values: Path, field: ValidationInfo) -> None:
        """Check the filename format to ensure correct structure and tokens."""
        for file_path in values:
            check_filename_format(file_path.name, field.field_name.split('_')[-1])
        return values

    @field_validator('mgrs_tile_id')
    def validate_mgrs_tile_id(cls, mgrs_tile_id: str) -> str:
        """Validate that mgrs_tile_id is present in the lookup table."""
        df_mgrs_burst = get_lut_by_mgrs_tile_ids(mgrs_tile_id)
        if df_mgrs_burst.empty:
            raise ValueError('The MGRS tile specified is not processed by DIST-S1')
        return mgrs_tile_id

    @property
    def confirmation(self) -> bool:
        """Returns True if prior_dist_s1_product is not None."""
        return self.prior_dist_s1_product is not None

    @property
    def processing_datetime(self) -> datetime:
        if self._processing_datetime is None:
            self._processing_datetime = datetime.now()
        return self._processing_datetime

    @property
    def min_acq_date(self) -> datetime:
        if self._min_acq_date is None:
            self._min_acq_date = min(
                get_acquisition_datetime(opera_rtc_s1_path) for opera_rtc_s1_path in self.post_rtc_copol
            )
        return self._min_acq_date

    @property
    def product_name(self) -> ProductNameData:
        if self._product_name is None:
            self._product_name = ProductNameData(
                mgrs_tile_id=self.mgrs_tile_id,
                acq_date_time=self.min_acq_date,
                processing_date_time=self.processing_datetime,
            )
        return self._product_name.name()

    @property
    def product_data_model(self) -> ProductDirectoryData:
        if self._product_data_model is None:
            product_name = self.product_name
            # Use dst_dir if product_dst_dir is None
            dst_dir = (
                Path(self.product_dst_dir).resolve()
                if self.product_dst_dir is not None
                else Path(self.dst_dir).resolve()
            )
            self._product_data_model = ProductDirectoryData(
                dst_dir=dst_dir,
                product_name=product_name,
            )
        return self._product_data_model

    def get_public_attributes(self) -> dict:
        config_dict = {k: v for k, v in self.model_dump().items() if not k.startswith('_')}
        config_dict.pop('check_input_paths', None)
        return config_dict

    def to_yaml(self, yaml_file: str | Path, algo_param_path: str | Path | None = None) -> None:
        """Save configuration to a YAML file.

        Parameters
        ----------
        yaml_file : str | Path
            Path to save the main configuration YAML file
        algo_param_path : str | Path | None, default None
            If provided, save algorithm parameters to this separate YAML file and reference it
            in the main config. If None, save all parameters in one file.
        """
        yaml_file = Path(yaml_file)

        if algo_param_path is not None:
            algo_param_path = Path(algo_param_path)

            # Get algorithm parameters (fields defined in AlgoConfigData)
            algo_field_names = set(AlgoConfigData.model_fields.keys())

            # Get all public attributes
            config_dict = self.get_public_attributes()

            # Separate algorithm and run config parameters
            algo_dict = {k: v for k, v in config_dict.items() if k in algo_field_names}
            run_dict = {k: v for k, v in config_dict.items() if k not in algo_field_names}

            # Save algorithm config to specified file
            algo_config = AlgoConfigData(**algo_dict)
            algo_config.to_yml(algo_param_path)

            # Add reference to algorithm config file in main config
            run_dict['algo_config_path'] = str(algo_param_path)
            yml_dict = {'run_config': run_dict}
        else:
            # Save everything in one file
            config_dict = self.get_public_attributes()
            yml_dict = {'run_config': config_dict}

        # Write main configuration to YAML file
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)

    @classmethod
    @check_input(dist_s1_loc_input_schema, obj_getter=0, lazy=True)
    def from_product_df(
        cls,
        product_df: gpd.GeoDataFrame,
        dst_dir: Path | str | None = DEFAULT_DST_DIR,
        apply_water_mask: bool = DEFAULT_APPLY_WATER_MASK,
        water_mask_path: Path | str | None = None,
        max_pre_imgs_per_burst_mw: list[int] | None = None,
        delta_lookback_days_mw: list[int] | None = None,
        lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
        prior_dist_s1_product: ProductDirectoryData | None = None,
        # Removed algorithm parameters that can be assigned later:
        # device, interpolation_method, apply_despeckling, apply_logit_to_inputs
    ) -> 'RunConfigData':
        """Transform input table from dist-s1-enumerator into RunConfigData object.

        Algorithm parameters should be assigned via attributes after creation.
        """
        df_pre = product_df[product_df.input_category == 'pre'].reset_index(drop=True)
        df_post = product_df[product_df.input_category == 'post'].reset_index(drop=True)
        if max_pre_imgs_per_burst_mw is None:
            max_pre_imgs_per_burst_mw = DEFAULT_MAX_PRE_IMGS_PER_BURST_MW
        if delta_lookback_days_mw is None:
            delta_lookback_days_mw = DEFAULT_DELTA_LOOKBACK_DAYS_MW
        runconfig_data = RunConfigData(
            pre_rtc_copol=df_pre.loc_path_copol.tolist(),
            pre_rtc_crosspol=df_pre.loc_path_crosspol.tolist(),
            post_rtc_copol=df_post.loc_path_copol.tolist(),
            post_rtc_crosspol=df_post.loc_path_crosspol.tolist(),
            mgrs_tile_id=df_pre.mgrs_tile_id.iloc[0],
            dst_dir=dst_dir,
            apply_water_mask=apply_water_mask,
            water_mask_path=water_mask_path,
            max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
            delta_lookback_days_mw=delta_lookback_days_mw,
            lookback_strategy=lookback_strategy,
            prior_dist_s1_product=prior_dist_s1_product,
            # Algorithm parameters removed - set via assignment
        )
        return runconfig_data

    @property
    def df_tile_dist(self) -> pd.DataFrame:
        if self._df_tile_dist is None:
            pd.DataFrame(
                {
                    'delta_lookback': [0, 1, 2],
                }
            )
        return self._df_tile_dist

    @property
    def product_directory(self) -> Path:
        return Path(self.product_data_model.product_dir_path)

    @property
    def final_unformatted_tif_paths(self) -> dict:
        # We are going to have a directory without metadata, colorbar, tags, etc.
        product_no_confirmation_dir = self.dst_dir / 'product_without_confirmation'
        product_no_confirmation_dir.mkdir(parents=True, exist_ok=True)
        final_unformatted_tif_paths = {
            'alert_status_path': product_no_confirmation_dir / 'alert_status.tif',
            'metric_status_path': product_no_confirmation_dir / 'metric_status.tif',
            # cofirmation db fields
            'dist_status_path': product_no_confirmation_dir / 'dist_status.tif',
            'dist_max_path': product_no_confirmation_dir / 'dist_max.tif',
            'dist_conf_path': product_no_confirmation_dir / 'dist_conf.tif',
            'dist_date_path': product_no_confirmation_dir / 'dist_date.tif',
            'dist_count_path': product_no_confirmation_dir / 'dist_count.tif',
            'dist_perc_path': product_no_confirmation_dir / 'dist_perc.tif',
            'dist_dur_path': product_no_confirmation_dir / 'dist_dur.tif',
            'dist_last_date_path': product_no_confirmation_dir / 'dist_last_date.tif',
        }

        return final_unformatted_tif_paths

    @property
    def df_burst_distmetrics(self) -> pd.DataFrame:
        if self._df_burst_distmetrics is None:
            df_inputs = self.df_inputs
            df_post = df_inputs[df_inputs.input_category == 'post'].reset_index(drop=True)
            df_distmetrics = (
                df_post.groupby('jpl_burst_id')
                .agg({'opera_id': 'first', 'acq_dt': 'first', 'acq_date_for_mgrs_pass': 'first'})
                .reset_index(drop=False)
            )

            # Metric Paths
            metric_dir = self.dst_dir / 'metric_burst'
            metric_dir.mkdir(parents=True, exist_ok=True)
            df_distmetrics['loc_path_metric'] = df_distmetrics.opera_id.map(
                lambda id_: f'{metric_dir}/metric_{id_}.tif'
            )
            # Dist Alert Intermediate by Burst
            dist_alert_dir = self.dst_dir / 'dist_alert_burst'
            dist_alert_dir.mkdir(parents=True, exist_ok=True)
            df_distmetrics['loc_path_dist_alert_burst'] = df_distmetrics.opera_id.map(
                lambda id_: f'{dist_alert_dir}/dist_alert_{id_}.tif'
            )
            self._df_burst_distmetrics = df_distmetrics

        return self._df_burst_distmetrics

    @property
    def df_inputs(self) -> pd.DataFrame:
        if self._df_inputs is None:
            data_pre = [
                {'input_category': 'pre', 'loc_path_copol': path_copol, 'loc_path_crosspol': path_crosspol}
                for path_copol, path_crosspol in zip(self.pre_rtc_copol, self.pre_rtc_crosspol)
            ]
            data_post = [
                {'input_category': 'post', 'loc_path_copol': path_copol, 'loc_path_crosspol': path_crosspol}
                for path_copol, path_crosspol in zip(self.post_rtc_copol, self.post_rtc_crosspol)
            ]
            data = data_pre + data_post
            df = pd.DataFrame(data)
            df['opera_id'] = df.loc_path_copol.apply(get_opera_id)
            df['jpl_burst_id'] = df.loc_path_copol.apply(get_burst_id).astype(str)
            df['track_number'] = df.loc_path_copol.apply(get_track_number)
            df['acq_dt'] = df.loc_path_copol.apply(get_acquisition_datetime)
            df['pass_id'] = df.acq_dt.apply(extract_pass_id)
            df = append_pass_data(df, [self.mgrs_tile_id])
            df['dst_dir'] = self.dst_dir

            # despeckle_paths
            def get_despeckle_path(row: pd.Series, polarization: str = 'copol') -> str:
                loc_path = row.loc_path_copol if polarization == 'copol' else row.loc_path_crosspol
                loc_path = str(loc_path).replace('.tif', '_tv.tif')
                acq_pass_date = row.acq_date_for_mgrs_pass
                filename = Path(loc_path).name
                out_path = self.dst_dir / 'tv_despeckle' / acq_pass_date / filename
                return str(out_path)

            df['loc_path_copol_dspkl'] = df.apply(get_despeckle_path, polarization='copol', axis=1)
            df['loc_path_crosspol_dspkl'] = df.apply(get_despeckle_path, polarization='crosspol', axis=1)

            df = df.sort_values(by=['jpl_burst_id', 'acq_dt']).reset_index(drop=True)
            self._df_inputs = df
        return self._df_inputs.copy()

    @property
    def df_prior_dist_products(self) -> pd.DataFrame:
        VALID_SUFFIXES = (
            '_GEN-DIST-STATUS.tif',
            '_GEN-METRIC-MAX.tif',
            '_GEN-DIST-CONF.tif',
            '_GEN-DIST-DATE.tif',
            '_GEN-DIST-COUNT.tif',
            '_GEN-DIST-PERC.tif',
            '_GEN-DIST-DUR.tif',
            '_GEN-DIST-LAST-DATE.tif',
        )

        if self._df_prior_dist_products is None:
            if not self.prior_dist_s1_product:
                self._df_prior_dist_products = pd.DataFrame()
                return self._df_prior_dist_products.copy()

            # Normalize paths
            paths = [Path(p) for p in self.prior_dist_s1_product]

            # Group by base name (everything before the DIST suffix)
            grouped = {}
            for path in paths:
                for suffix in VALID_SUFFIXES:
                    if path.name.endswith(suffix):
                        key = path.name.replace(suffix, '')
                        if key not in grouped:
                            grouped[key] = {}
                        grouped[key][suffix] = path
                        break

            # Build rows for DataFrame
            rows = []
            for key, files in grouped.items():
                if all(suffix in files for suffix in VALID_SUFFIXES):
                    row = {suffix: str(files[suffix]) for suffix in VALID_SUFFIXES}
                    row['product_key'] = key
                    rows.append(row)
                else:
                    missing = [s for s in VALID_SUFFIXES if s not in files]
                    raise ValueError(f'Missing files for {key}: {missing}')

            # Rename columns to user-friendly names
            column_mapping = {
                '_GEN-DIST-STATUS.tif': 'path_dist_status',
                '_GEN-METRIC-MAX.tif': 'path_dist_max',
                '_GEN-DIST-CONF.tif': 'path_dist_conf',
                '_GEN-DIST-DATE.tif': 'path_dist_date',
                '_GEN-DIST-COUNT.tif': 'path_dist_count',
                '_GEN-DIST-PERC.tif': 'path_dist_perc',
                '_GEN-DIST-DUR.tif': 'path_dist_dur',
                '_GEN-DIST-LAST-DATE.tif': 'path_dist_last_date',
            }

            df = pd.DataFrame(rows)
            df = df.rename(columns=column_mapping)
            df = df.sort_values(by='product_key').reset_index(drop=True)
            self._df_prior_dist_products = df
            return self._df_prior_dist_products.copy()

    @model_validator(mode='after')
    def handle_water_mask_control_flow(self) -> 'RunConfigData':
        """Trigger water mask control flow when apply_water_mask is True."""
        if self.apply_water_mask and self.water_mask_path is None:
            self.water_mask_path = water_mask_control_flow(
                water_mask_path=self.water_mask_path,
                mgrs_tile_id=self.mgrs_tile_id,
                apply_water_mask=self.apply_water_mask,
                dst_dir=self.dst_dir,
                overwrite=True,
            )
        return self

    @model_validator(mode='after')
    def handle_device_specific_validations(self) -> 'RunConfigData':
        """Handle device-specific validations and adjustments."""
        # Device-specific validations
        if self.device in ['cuda', 'mps'] and self.n_workers_for_norm_param_estimation > 1:
            raise ValueError(
                f'CUDA and MPS devices do not support multiprocessing. '
                f'When device="{self.device}", n_workers_for_norm_param_estimation must be 1, '
                f'but got {self.n_workers_for_norm_param_estimation}. '
                f'Either set device="cpu" to use multiprocessing or set n_workers_for_norm_param_estimation=1.'
            )
        if self.device in ['cuda', 'mps'] and self.model_compilation:
            raise ValueError(
                f'Model compilation with CUDA/MPS devices requires single-threaded processing. '
                f'When device="{self.device}" and model_compilation=True, '
                f'n_workers_for_norm_param_estimation must be 1, '
                f'but got {self.n_workers_for_norm_param_estimation}. '
                f'Either set device="cpu", model_compilation=False, or n_workers_for_norm_param_estimation=1.'
            )
        return self
