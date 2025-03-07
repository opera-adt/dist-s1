import multiprocessing as mp
import warnings
from datetime import datetime
from pathlib import Path, PosixPath

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import yaml
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id
from dist_s1_enumerator.data_models import dist_s1_loc_input_schema
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from distmetrics.transformer import get_device
from pandera import check_input
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator
from yaml import Dumper

from dist_s1.data_models.output_models import ProductDirectoryData, ProductNameData
from dist_s1.water_mask import water_mask_control_flow


def posix_path_encoder(dumper: Dumper, data: PosixPath) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))


def none_encoder(dumper: Dumper, _: None) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:null', '')


yaml.add_representer(PosixPath, posix_path_encoder)
yaml.add_representer(type(None), none_encoder)


def generate_burst_dist_paths(
    row: pd.Series,
    *,
    top_level_data_dir: Path,
    dst_dir_name: str,
    lookback: int | None = 0,
    path_token: str | None = None,
    polarization_token: str | None = None,
    date_lut: dict[str, list[pd.Timestamp]] | None = None,
    n_lookbacks: int | None = None,
) -> Path:
    if path_token is None:
        path_token = dst_dir_name
    data_dir = top_level_data_dir / dst_dir_name
    if lookback is not None:
        lookback_dir = data_dir / f'Delta_{lookback}'
    else:
        lookback_dir = data_dir
    lookback_dir.mkdir(parents=True, exist_ok=True)
    burst_id = row.jpl_burst_id
    if date_lut is not None and lookback is not None:
        acq_date = date_lut[burst_id][n_lookbacks - lookback - 1]
    else:
        acq_date = row.acq_dt
    acq_date_str = acq_date.date().strftime('%Y-%m-%d')
    fn = f'{path_token}_{burst_id}_{acq_date_str}.tif'
    if lookback is not None:
        fn = fn.replace('.tif', f'_delta{lookback}.tif')
    if polarization_token is not None:
        fn = fn.replace(f'{path_token}', f'{path_token}_{polarization_token}')
    out_path = lookback_dir / fn
    return out_path


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
    if (polarization == 'copol') and not filename.endswith('_VV.tif'):
        raise ValueError(f"File '{filename}' should end with '_VV.tif' because it is copolarization")
    elif (polarization == 'crosspol') and not filename.endswith('_VH.tif'):
        raise ValueError(f"File '{filename}' should end with '_VH.tif' because it is crosspolarization")
    return True


class RunConfigData(BaseModel):
    pre_rtc_copol: list[Path | str]
    pre_rtc_crosspol: list[Path | str]
    post_rtc_copol: list[Path | str]
    post_rtc_crosspol: list[Path | str]
    mgrs_tile_id: str
    dst_dir: Path | str = Path('out')
    water_mask_path: Path | str | None = None
    apply_water_mask: bool = Field(default=True)
    check_input_paths: bool = True
    device: str = Field(
        default='best',
        pattern='^(best|cuda|mps|cpu)$',
    )
    memory_strategy: str | None = Field(
        default='low',
        pattern='^(high|low)$',
    )
    tqdm_enabled: bool = Field(default=True)
    batch_size_for_despeckling: int = Field(
        default=25,
        ge=1,
    )
    n_workers_for_norm_param_estimation: int = Field(
        default=8,
        ge=1,
    )
    n_workers_for_despeckling: int = Field(
        default=8,
        ge=1,
    )
    n_lookbacks: int = Field(default=3, ge=1, le=3)
    # This is where default thresholds are set!
    moderate_confidence_threshold: float = Field(default=3.5, ge=0.0, le=15.0)
    high_confidence_threshold: float = Field(default=5.5, ge=0.0, le=15.0)
    product_dst_dir: Path | str | None = None
    bucket: str | None = None
    bucket_prefix: str = ''

    # Private attributes that are associated to properties
    _burst_ids: list[str] | None = None
    _df_inputs: pd.DataFrame | None = None
    _df_burst_distmetric: pd.DataFrame | None = None
    _df_mgrs_burst_lut: gpd.GeoDataFrame | None = None
    _product_name: ProductNameData | None = None
    _product_data_model: ProductDirectoryData | None = None
    _min_acq_date: datetime | None = None
    _processing_datetime: datetime | None = None

    @classmethod
    def from_yaml(cls, yaml_file: str, fields_to_overwrite: dict | None = None) -> 'RunConfigData':
        """Load configuration from a YAML file and initialize RunConfigModel."""
        with Path.open(yaml_file) as file:
            data = yaml.safe_load(file)
            runconfig_data = data['run_config']
        if fields_to_overwrite is not None:
            runconfig_data.update(fields_to_overwrite)
        obj = cls(**runconfig_data)
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

    @field_validator('n_workers_for_despeckling', 'n_workers_for_norm_param_estimation')
    def validate_n_workers(cls, n_workers: int, info: ValidationInfo) -> int:
        if n_workers > mp.cpu_count():
            warnings.warn(
                f'{info.field_name} ({n_workers}) is greater than the number of CPUs ({mp.cpu_count()}), using latter.',
                UserWarning,
            )
            n_workers = mp.cpu_count()
        return n_workers

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

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Save configuration to a YAML file."""
        # Get only the non-private attributes (those that don't start with _)
        config_dict = self.get_public_attributes()
        yml_dict = {'run_config': config_dict}

        # Write to YAML file
        yaml_file = Path(yaml_file)
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)

    @classmethod
    @check_input(dist_s1_loc_input_schema, obj_getter=0, lazy=True)
    def from_product_df(
        cls,
        product_df: gpd.GeoDataFrame,
        dst_dir: Path | str | None = Path('out'),
        apply_water_mask: bool = True,
        water_mask_path: Path | str | None = None,
    ) -> 'RunConfigData':
        """Transform input table from dist-s1-enumerator into RunConfigData object.

        Additional runconfig parameters should be assigned via attributes.
        """
        df_pre = product_df[product_df.input_category == 'pre'].reset_index(drop=True)
        df_post = product_df[product_df.input_category == 'post'].reset_index(drop=True)
        runconfig_data = RunConfigData(
            pre_rtc_copol=df_pre.loc_path_copol.tolist(),
            pre_rtc_crosspol=df_pre.loc_path_crosspol.tolist(),
            post_rtc_copol=df_post.loc_path_copol.tolist(),
            post_rtc_crosspol=df_post.loc_path_crosspol.tolist(),
            mgrs_tile_id=df_pre.mgrs_tile_id.iloc[0],
            dst_dir=dst_dir,
            apply_water_mask=apply_water_mask,
            water_mask_path=water_mask_path,
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
        pre_product_dir = self.dst_dir / 'pre_product'
        pre_product_dir.mkdir(parents=True, exist_ok=True)
        final_unformatted_tif_paths = {
            'alert_status_path': pre_product_dir / 'alert_status.tif',
            'metric_status_path': pre_product_dir / 'metric_status.tif',
        }
        for lookback in range(self.n_lookbacks):
            final_unformatted_tif_paths[f'alert_delta{lookback}_path'] = pre_product_dir / f'alert_delta{lookback}.tif'

        return final_unformatted_tif_paths

    @property
    def df_burst_distmetrics(self) -> pd.DataFrame:
        if self._df_burst_distmetric is None:
            normal_param_dir = self.dst_dir / 'normal_params'
            normal_param_dir.mkdir(parents=True, exist_ok=True)

            df_inputs = self.df_inputs.copy()
            df_post = df_inputs[df_inputs.input_category == 'post'].reset_index(drop=True)
            burst_ids = df_post.jpl_burst_id.unique()
            df_dist_by_burst = pd.DataFrame({'jpl_burst_id': burst_ids})

            df_date = df_inputs.groupby('jpl_burst_id')['acq_dt'].apply(np.maximum.reduce).reset_index(drop=False)
            df_dist_by_burst = pd.merge(df_dist_by_burst, df_date, on='jpl_burst_id', how='left')

            # Get the N_LOOKBACKS most recent dates before the current acquisition
            df_pre = df_inputs[df_inputs.input_category == 'pre'].reset_index(drop=True)
            df_pre.sort_values(by=['jpl_burst_id', 'acq_dt'], inplace=True, ascending=True)
            df_date_pre = df_pre.groupby('jpl_burst_id')['acq_dt'].apply(
                lambda x: sorted(x.nlargest(self.n_lookbacks).tolist())
            )
            burst2predates = df_date_pre.to_dict()

            # Distribution Paths
            for lookback in range(self.n_lookbacks):
                for normal_param_token in ['mean', 'std']:
                    for polarization_token in ['copol', 'crosspol']:
                        df_dist_by_burst[
                            f'loc_path_normal_{normal_param_token}_delta{lookback}_{polarization_token}'
                        ] = df_dist_by_burst.apply(
                            generate_burst_dist_paths,
                            top_level_data_dir=self.dst_dir,
                            dst_dir_name='normal_params',
                            path_token=normal_param_token,
                            polarization_token=polarization_token,
                            lookback=lookback,
                            date_lut=burst2predates,
                            axis=1,
                            n_lookbacks=self.n_lookbacks,
                        )

            # Metrics Paths
            df_dist_by_burst['loc_path_metric_delta0'] = df_dist_by_burst.apply(
                generate_burst_dist_paths,
                top_level_data_dir=self.dst_dir,
                dst_dir_name='metrics',
                path_token='distmetric',
                lookback=0,
                date_lut=None,
                axis=1,
                n_lookbacks=self.n_lookbacks,
            )

            # Disturbance Paths for Each Lookback
            for lookback in range(self.n_lookbacks):
                df_dist_by_burst[f'loc_path_disturb_delta{lookback}'] = df_dist_by_burst.apply(
                    generate_burst_dist_paths,
                    top_level_data_dir=self.dst_dir,
                    dst_dir_name='disturbance',
                    path_token='disturb',
                    lookback=lookback,
                    date_lut=None,
                    axis=1,
                    n_lookbacks=self.n_lookbacks,
                )

            # Disturbance Paths Time Aggregated
            df_dist_by_burst['loc_path_disturb_time_aggregated'] = df_dist_by_burst.apply(
                generate_burst_dist_paths,
                top_level_data_dir=self.dst_dir,
                dst_dir_name='disturbance/time_aggregated',
                path_token='disturb',
                lookback=None,
                date_lut=None,
                axis=1,
                n_lookbacks=self.n_lookbacks,
            )

            self._df_burst_distmetric = df_dist_by_burst

        return self._df_burst_distmetric

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

    def model_post_init(self, __context: ValidationInfo) -> None:
        # Water mask control flow
        self.water_mask_path = water_mask_control_flow(
            water_mask_path=self.water_mask_path,
            mgrs_tile_id=self.mgrs_tile_id,
            apply_water_mask=self.apply_water_mask,
            dst_dir=self.dst_dir,
            overwrite=True,
        )

        # Device-specific validations
        if self.device in ['cuda', 'mps'] and self.n_workers_for_norm_param_estimation > 1:
            warnings.warn(
                'CUDA and mps do not support multiprocessing; setting n_workers_for_norm_param_estimation to 1',
                UserWarning,
            )
            self.n_workers_for_norm_param_estimation = 1
