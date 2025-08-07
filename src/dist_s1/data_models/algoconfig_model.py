import warnings
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from distmetrics import get_device
from distmetrics.model_load import get_model_context_length
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_serializer, field_validator, model_validator

from dist_s1.data_models.data_utils import get_max_pre_imgs_per_burst_mw
from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_DESPECKLING,
    DEFAULT_APPLY_LOGIT_TO_INPUTS,
    DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_CONFIDENCE_UPPER_LIM,
    DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_DEVICE,
    DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_INTERPOLATION_METHOD,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_MAX_OBS_NUM_YEAR,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_MEMORY_STRATEGY,
    DEFAULT_METRIC_VALUE_UPPER_LIM,
    DEFAULT_MODEL_CFG_PATH,
    DEFAULT_MODEL_COMPILATION,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_WTS_PATH,
    DEFAULT_NO_COUNT_RESET_THRESH,
    DEFAULT_NO_DAY_LIMIT,
    DEFAULT_N_ANNIVERSARIES_FOR_MW,
    DEFAULT_N_WORKERS_FOR_DESPECKLING,
    DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_PERCENT_RESET_THRESH,
    DEFAULT_POST_DATE_BUFFER_DAYS,
    DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_TQDM_ENABLED,
    DEFAULT_USE_DATE_ENCODING,
)


class AlgoConfigData(BaseModel):
    """Base class containing algorithm configuration parameters."""

    device: str = Field(
        default=DEFAULT_DEVICE,
        pattern='^(best|cuda|mps|cpu)$',
        description='Device to use for model inference. `best` will use the best available device.',
    )
    memory_strategy: str | None = Field(
        default=DEFAULT_MEMORY_STRATEGY,
        pattern='^(high|low)$',
        description='Memory strategy to use for model inference. `high` will use more memory, `low` will use less. '
        'Utilizing more memory will improve runtime performance.',
    )
    tqdm_enabled: bool = Field(
        default=DEFAULT_TQDM_ENABLED,
        description='Whether to enable tqdm progress bars.',
    )
    n_workers_for_norm_param_estimation: int = Field(
        default=DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
        description='Number of workers for norm parameter estimation from the baseline. '
        'Utilizing more workers will improve runtime performance and utilize more memory. '
        'Does not work with model compilation or MPS/GPU devices.',
    )
    batch_size_for_norm_param_estimation: int = Field(
        default=DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
        description='Batch size for norm parameter estimation from the baseline. '
        'Utilizing a larger batch size will improve runtime performance and utilize more memory.',
    )
    stride_for_norm_param_estimation: int = Field(
        default=DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
        ge=1,
        le=16,
        description='Stride for norm parameter estimation from the baseline. '
        'Utilizing a larger stride will improve metric accuracy and utilize more memory.'
        'Memory usage scales inverse quadratically with stride. That is, '
        'If stride=16 consumes N bytes of memory, then stride=4 consumes 16N bytes of memory.',
    )
    apply_logit_to_inputs: bool = Field(
        default=DEFAULT_APPLY_LOGIT_TO_INPUTS,
        description='Whether to apply logit transform to the input data.',
    )
    n_workers_for_despeckling: int = Field(
        default=DEFAULT_N_WORKERS_FOR_DESPECKLING,
        ge=1,
        description='Number of workers for despeckling. '
        'Utilizing more workers will improve runtime performance and utilize more memory.',
    )
    lookback_strategy: str = Field(
        default=DEFAULT_LOOKBACK_STRATEGY,
        pattern='^multi_window$',
        description='Lookback strategy to use for data curation of the baseline. '
        '`multi_window` will use a multi-window lookback strategy and is default for OEPRA DIST-S1, '
        '`immediate_lookback` will use an immediate lookback strategy using acquisitions preceding the post-date. '
        '`immediate_lookback` is not supported yet.',
    )
    post_date_buffer_days: int = Field(
        default=DEFAULT_POST_DATE_BUFFER_DAYS,
        ge=0,
        le=7,
        description='Buffer days around post-date for data collection to create acqusition image to compare baseline '
        'to.',
    )
    model_compilation: bool = Field(
        default=DEFAULT_MODEL_COMPILATION,
        description='Whether to compile the model for CPU or GPU. '
        'False, use the model as is. '
        'True, load the model and compile for CPU or GPU optimizations.',
    )
    max_pre_imgs_per_burst_mw: tuple[int, ...] | None = Field(
        default=DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
        description='Max number of pre-images per burst within each window',
    )
    delta_lookback_days_mw: tuple[int, ...] | None = Field(
        default=DEFAULT_DELTA_LOOKBACK_DAYS_MW,
        description='Delta lookback days for each window relative to post-image acquisition date',
    )
    low_confidence_alert_threshold: float = Field(
        default=DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
        ge=0.0,
        le=15.0,
        description='Low confidence alert threshold for detecting disturbance between baseline and post-image.',
    )
    high_confidence_alert_threshold: float = Field(
        default=DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
        ge=0.0,
        le=15.0,
        description='High confidence alert threshold for detecting disturbance between baseline and post-image.',
    )
    no_day_limit: int = Field(
        default=DEFAULT_NO_DAY_LIMIT,
        description='Number of days to limit confirmation process logic to. Confirmation must occur within first '
        'observance of disturbance and `no_day_limit` days after first disturbance.',
    )
    exclude_consecutive_no_dist: int = Field(
        default=DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
        description='Boolean activation of consecutive no disturbance tracking during confirmation. '
        'True will apply this logic: '
        'after 2 no disturbances within product sequence, the disturbance must finish or be reset. '
        'False will not apply this logic.',
    )
    percent_reset_thresh: int = Field(
        default=DEFAULT_PERCENT_RESET_THRESH,
        description='Precentage number threshold to reset disturbance. Values below `percent_reset_thresh` '
        'will reset disturbance.',
    )
    no_count_reset_thresh: int = Field(
        default=DEFAULT_NO_COUNT_RESET_THRESH,
        description='If the number of non-disturbed observations `prevnocount` is above `nocount_reset_thresh` '
        'disturbance will reset.',
    )
    max_obs_num_year: int = Field(
        default=DEFAULT_MAX_OBS_NUM_YEAR,
        description='Max observation number per year. If observations exceeds this number, then the confirmation must '
        'conclude and be reset.',
    )
    confidence_upper_lim: int = Field(
        default=DEFAULT_CONFIDENCE_UPPER_LIM,
        description='Confidence upper limit for confirmation. Confidence is an accumulation of the metric over time.',
    )
    confirmation_confidence_threshold: float = Field(
        default=DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
        description='This is the threshold for the confirmation process to determine if a disturbance is confirmed.',
    )
    metric_value_upper_lim: float = Field(
        default=DEFAULT_METRIC_VALUE_UPPER_LIM, description='Metric upper limit set during confirmation'
    )
    model_source: str = Field(
        default=DEFAULT_MODEL_SOURCE,
        description='Model source. If `external`, use externally supplied paths for weights and config. '
        'Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models.',
    )
    model_cfg_path: Path | str | None = Field(
        default=DEFAULT_MODEL_CFG_PATH,
        description='Path to model config file. If `external`, use externally supplied path. '
        'Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models.',
    )
    model_wts_path: Path | str | None = Field(
        default=DEFAULT_MODEL_WTS_PATH,
        description='Path to model weights file. If `external`, use externally supplied path. '
        'Otherwise, use distmetrics.model_load.ALLOWED_MODELS for other models.',
    )
    apply_despeckling: bool = Field(
        default=DEFAULT_APPLY_DESPECKLING,
        description='Whether to apply despeckling to the input data.',
    )
    interpolation_method: str = Field(
        default=DEFAULT_INTERPOLATION_METHOD,
        pattern='^(nearest|bilinear|none)$',
        description='Interpolation method to use for despeckling. `nearest` will use nearest neighbor interpolation, '
        '`bilinear` will use bilinear interpolation, and `none` will not apply despeckling.',
    )
    model_dtype: str = Field(
        default=DEFAULT_MODEL_DTYPE,
        pattern='^(float32|bfloat16|float)$',
        description='Data type for model inference. Note: bfloat16 is only supported on GPU devices.',
    )
    use_date_encoding: bool = Field(
        default=DEFAULT_USE_DATE_ENCODING,
        description='Whether to use acquisition date encoding in model application (currently not supported)',
    )
    _model_context_length: int | None = None
    n_anniversaries_for_mw: int = Field(
        default=DEFAULT_N_ANNIVERSARIES_FOR_MW,
        description='Number of anniversaries to use for multi-window',
    )

    # Validate assignments to all fields
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> 'AlgoConfigData':
        """Load algorithm configuration from a YAML file."""
        yaml_file = Path(yaml_file)
        with yaml_file.open() as file:
            data = yaml.safe_load(file)
            algo_data = data.get('algo_config')
            if algo_data is None:
                algo_data = data.get('algorithm_config', data)

        obj = cls(**algo_data)
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

    @field_validator('max_pre_imgs_per_burst_mw', 'delta_lookback_days_mw', mode='before')
    def convert_list_to_tuple(cls, v: list[int] | None) -> tuple[int, ...] | None:
        """Convert lists to tuples for YAML compatibility."""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_serializer('max_pre_imgs_per_burst_mw', 'delta_lookback_days_mw')
    def serialize_tuple_as_list(self, v: tuple[int, ...] | None) -> list[int] | None:
        """Serialize tuples as lists for YAML compatibility."""
        if v is not None:
            return list(v)
        return None

    @field_validator('low_confidence_alert_threshold')
    def validate_low_confidence_alert_threshold(
        cls, low_confidence_alert_threshold: float, info: ValidationInfo
    ) -> float:
        """Validate that low_confidence_alert_threshold is less than high_confidence_alert_threshold."""
        high_threshold = info.data.get('high_confidence_alert_threshold')
        if high_threshold is not None and low_confidence_alert_threshold >= high_threshold:
            raise ValueError(
                f'low_confidence_alert_threshold ({low_confidence_alert_threshold}) must be less than '
                f'high_confidence_alert_threshold ({high_threshold})'
            )
        return low_confidence_alert_threshold

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

    @property
    def model_context_length(self) -> int:
        if self._model_context_length is None:
            self._model_context_length = get_model_context_length(self.model_source, self.model_cfg_path)
        return self._model_context_length

    @model_validator(mode='after')
    def handle_device_specific_validations(self) -> 'AlgoConfigData':
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

    @model_validator(mode='after')
    def calculate_max_pre_imgs_per_burst_mw(self) -> 'AlgoConfigData':
        """Calculate max_pre_imgs_per_burst_mw if not provided."""
        if self.max_pre_imgs_per_burst_mw is None:
            self.max_pre_imgs_per_burst_mw = get_max_pre_imgs_per_burst_mw(
                self.model_context_length, self.n_anniversaries_for_mw
            )
        return self

    @model_validator(mode='after')
    def calculate_delta_lookback_days_mw(self) -> 'AlgoConfigData':
        """Calculate delta_lookback_days_mw if not provided."""
        if self.delta_lookback_days_mw is None:
            self.delta_lookback_days_mw = tuple(365 * n for n in range(self.n_anniversaries_for_mw, 0, -1))
        return self

    def to_yml(self, yaml_file: str | Path) -> None:
        """Save algorithm configuration to a YAML file."""
        config_dict = self.model_dump()
        yml_dict = {'algo_config': config_dict}

        yaml_file = Path(yaml_file)
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)
