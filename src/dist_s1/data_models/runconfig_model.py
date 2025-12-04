from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from dist_s1_enumerator.tabular_models import dist_s1_loc_input_schema
from pandera.pandas import check_input
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from dist_s1.data_models.algoconfig_model import AlgoConfigData
from dist_s1.data_models.data_utils import (
    check_filename_format,
    extract_rtc_metadata_from_path,
    get_acquisition_datetime,
    get_burst_id,
    get_opera_id,
    get_opera_id_without_proccessing_time,
    get_polarization_from_row,
    get_sensor,
    get_track_number,
)
from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_CHECK_INPUT_PATHS,
    DEFAULT_DST_DIR,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_MODEL_CONTEXT_LENGTH_MAXIMUM,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_N_ANNIVERSARIES_FOR_MW,
    DEFAULT_POST_DATE_BUFFER_DAYS,
    DEFAULT_SRC_WATER_MASK_PATH,
)
from dist_s1.data_models.output_models import DistS1ProductDirectory, ProductNameData
from dist_s1.water_mask import water_mask_control_flow


class RunConfigData(BaseModel):
    check_input_paths: bool = Field(
        default=DEFAULT_CHECK_INPUT_PATHS,
        description='Whether to check if the input paths exist. If True, the input paths are checked. '
        'Used during testing.',
    )
    pre_rtc_copol: list[Path | str] = Field(..., description='List of paths to pre-rtc copolarization data.')
    pre_rtc_crosspol: list[Path | str] = Field(..., description='List of paths to pre-rtc crosspolarization data.')
    post_rtc_copol: list[Path | str] = Field(..., description='List of paths to post-rtc copolarization data.')
    post_rtc_crosspol: list[Path | str] = Field(..., description='List of paths to post-rtc crosspolarization data.')
    prior_dist_s1_product: DistS1ProductDirectory | str | Path | None = Field(
        default=None,
        description='Path to prior DIST-S1 product. Can accept str, Path, or DistS1ProductDirectory. '
        'If None, no prior product is used and confirmation is not performed.',
    )
    mgrs_tile_id: str = Field(..., description='MGRS tile ID. Required to kick-off disturbance processing.')
    dst_dir: Path | str = DEFAULT_DST_DIR
    input_data_dir: Path | str | None = Field(
        default=DEFAULT_INPUT_DATA_DIR,
        description='Input data directory. If None, defaults to dst_dir.',
    )
    src_water_mask_path: Path | str | None = Field(
        default=DEFAULT_SRC_WATER_MASK_PATH,
        description='Path to water mask. If None and apply_water_mask is False, '
        'no water mask is used. If None and apply_water_mask is True, '
        'the tiles to generate the water mask over MGRS area are localized and formatted for use.',
    )
    apply_water_mask: bool = Field(
        default=DEFAULT_APPLY_WATER_MASK,
        description='Whether to apply water mask to the input data. If True, water mask is applied to the input data. '
        'If no water mask path is provided, the tiles to generate the water mask over MGRS area are localized and '
        'formatted for use.',
    )
    product_dst_dir: Path | str | None = Field(
        default=None,
        description='Path to product directory. If None, defaults to dst_dir.',
    )
    bucket: str | None = Field(
        default=None,
        description='Bucket to use for product storage. If None, no bucket is used.',
    )
    bucket_prefix: str | None = Field(
        default=None,
        description='Bucket prefix to use for product storage. If None, no bucket prefix is used.',
    )
    # Algorithm configuration
    algo_config: AlgoConfigData = Field(
        default_factory=AlgoConfigData,
        description='Algorithm configuration parameters.',
    )
    # Path to external algorithm config file
    algo_config_path: Path | str | None = Field(
        default=None,
        description='Path to external algorithm config file. If None, no external algorithm config is used.',
    )

    # Private attributes that are associated to properties
    _burst_ids: list[str] | None = None
    _df_copol_data: pd.DataFrame | None = None
    _df_crosspol_data: pd.DataFrame | None = None
    _df_inputs: pd.DataFrame | None = None
    _df_prior_dist_products: pd.DataFrame | None = None
    _df_burst_distmetrics: pd.DataFrame | None = None
    _df_mgrs_burst_lut: gpd.GeoDataFrame | None = None
    _product_name: ProductNameData | None = None
    _product_data_model: DistS1ProductDirectory | None = None
    _product_data_model_no_confirmation: DistS1ProductDirectory | None = None
    _min_acq_date: datetime | None = None
    _processing_datetime: datetime | None = None
    _algo_config_loaded: bool = False
    _processed_water_mask_path: Path | str | None = None
    # Validate assignments to all fields
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'RunConfigData':
        """Load configuration from a YAML file and initialize RunConfigModel."""
        with Path.open(yaml_file) as file:
            data = yaml.safe_load(file)
        # Nested or flat run config
        run_params = data['run_config'] if 'run_config' in data.keys() else data
        config = cls(**run_params)
        if 'algo_config_path' in run_params.keys():
            config.algo_config = AlgoConfigData.from_yaml(run_params['algo_config_path'])
        return config

    @field_validator('pre_rtc_copol', 'pre_rtc_crosspol', 'post_rtc_copol', 'post_rtc_crosspol', mode='after')
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
        return dst_dir

    @field_validator('product_dst_dir', mode='before')
    def validate_product_dst_dir(cls, product_dst_dir: Path | str | None, info: ValidationInfo) -> Path:
        if product_dst_dir is None:
            product_dst_dir = Path(info.data['dst_dir'])
        elif isinstance(product_dst_dir, str):
            product_dst_dir = Path(product_dst_dir)
        if product_dst_dir.exists() and not product_dst_dir.is_dir():
            raise ValidationError(f"Path '{product_dst_dir}' exists but is not a directory")
        product_dst_dir.mkdir(parents=True, exist_ok=True)
        return product_dst_dir

    @field_validator('input_data_dir', mode='before')
    def validate_input_data_dir(cls, input_data_dir: Path | str | None) -> Path | None:
        """Convert string to Path and validate if provided."""
        if input_data_dir is None:
            return None
        input_data_dir = Path(input_data_dir) if isinstance(input_data_dir, str) else input_data_dir
        if not input_data_dir.exists():
            raise ValueError(f'Input data directory does not exist: {input_data_dir}')
        if not input_data_dir.is_dir():
            raise ValueError(f'Input data directory is not a directory: {input_data_dir}')
        return input_data_dir

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

    @field_validator('algo_config_path', mode='before')
    def validate_algo_config_path(cls, algo_config_path: Path | str | None) -> Path | None:
        """Validate that algo_config_path exists if provided."""
        if algo_config_path is None:
            return None
        algo_config_path = Path(algo_config_path) if isinstance(algo_config_path, str) else algo_config_path
        if not algo_config_path.exists():
            raise ValueError(f'Algorithm config path does not exist: {algo_config_path}')
        if not algo_config_path.is_file():
            raise ValueError(f'Algorithm config path is not a file: {algo_config_path}')
        return algo_config_path

    @field_validator('prior_dist_s1_product', mode='before')
    def validate_prior_dist_s1_product(
        cls, prior_dist_s1_product: DistS1ProductDirectory | Path | str | None
    ) -> DistS1ProductDirectory | None:
        """Convert string or Path to DistS1ProductDirectory using from_product_path if needed."""
        if (prior_dist_s1_product is None) or (isinstance(prior_dist_s1_product, str) and prior_dist_s1_product == ''):
            return None
        if isinstance(prior_dist_s1_product, DistS1ProductDirectory):
            return prior_dist_s1_product
        elif isinstance(prior_dist_s1_product, str | Path):
            return DistS1ProductDirectory.from_product_path(prior_dist_s1_product)

    @property
    def confirmation(self) -> bool:
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
    def sensor(self) -> str:
        return get_sensor(self.post_rtc_copol[0])

    @property
    def product_name(self) -> ProductNameData:
        if self._product_name is None:
            self._product_name = ProductNameData(
                mgrs_tile_id=self.mgrs_tile_id,
                acq_date_time=self.min_acq_date,
                processing_date_time=self.processing_datetime,
                sensor=self.sensor,
            )
        return self._product_name.name()

    @property
    def product_data_model(self) -> DistS1ProductDirectory:
        if self._product_data_model is None:
            product_name = self.product_name
            dst_dir = Path(self.product_dst_dir) if self.product_dst_dir is not None else Path(self.dst_dir)
            self._product_data_model = DistS1ProductDirectory(
                dst_dir=dst_dir,
                product_name=product_name,
            )
        return self._product_data_model

    @property
    def df_copol_data(self) -> pd.DataFrame:
        if self._df_copol_data is None:
            data_pre = [
                {**extract_rtc_metadata_from_path(path_copol), 'input_category': 'pre'}
                for path_copol in self.pre_rtc_copol
            ]
            data_post = [
                {**extract_rtc_metadata_from_path(path_copol), 'input_category': 'post'}
                for path_copol in self.post_rtc_copol
            ]
            data = data_pre + data_post
            df = pd.DataFrame(data)
            df.rename(columns={'path': 'loc_path_copol'}, inplace=True)
            df = df.sort_values(by=['jpl_burst_id', 'acq_dt'], ascending=True)
            self._df_copol_data = df
        return self._df_copol_data

    @property
    def df_crosspol_data(self) -> pd.DataFrame:
        if self._df_crosspol_data is None:
            data_pre = [
                {**extract_rtc_metadata_from_path(path_crosspol), 'input_category': 'pre'}
                for path_crosspol in self.pre_rtc_crosspol
            ]
            data_post = [
                {**extract_rtc_metadata_from_path(path_crosspol), 'input_category': 'post'}
                for path_crosspol in self.post_rtc_crosspol
            ]
            data = data_pre + data_post
            df = pd.DataFrame(data)
            df.rename(columns={'path': 'loc_path_crosspol'}, inplace=True)
            df = df.sort_values(by=['jpl_burst_id', 'acq_dt'], ascending=True)
            self._df_crosspol_data = df

        return self._df_crosspol_data

    @property
    def product_data_model_no_confirmation(self) -> DistS1ProductDirectory:
        if self._product_data_model_no_confirmation is None:
            product_name = self.product_name
            dst_dir = self.dst_dir / 'product_without_confirmation'
            self._product_data_model_no_confirmation = DistS1ProductDirectory(
                dst_dir=dst_dir,
                product_name=product_name,
            )
        return self._product_data_model_no_confirmation

    def get_public_attributes(self, include_algo_config_params: bool = False) -> dict:
        config_dict = {k: v for k, v in self.model_dump().items() if not k.startswith('_')}
        config_dict.pop('check_input_paths', None)
        config_dict.pop('algo_config', None)
        config_dict['sensor'] = self.sensor
        if include_algo_config_params:
            config_dict.update(self.algo_config.model_dump())
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

        if algo_param_path is None:
            algo_param_path = yaml_file.parent / 'algo_params.yml'
        algo_param_path = Path(algo_param_path)

        config_dict = {k: v for k, v in self.model_dump().items() if not k.startswith('_')}
        config_dict['algo_config_path'] = str(algo_param_path)
        config_dict.pop('check_input_paths', None)
        config_dict.pop('algo_config')

        yml_dict = {'run_config': config_dict}
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)
        self.algo_config.to_yml(algo_param_path)
        self.algo_config_path = algo_param_path

    @classmethod
    @check_input(dist_s1_loc_input_schema, obj_getter=0, lazy=True)
    def from_product_df(
        cls,
        product_df: gpd.GeoDataFrame,
        dst_dir: Path | str | None = DEFAULT_DST_DIR,
        apply_water_mask: bool = DEFAULT_APPLY_WATER_MASK,
        water_mask_path: Path | str | None = None,
        max_pre_imgs_per_burst_mw: list[int] | None = None,
        prior_dist_s1_product: DistS1ProductDirectory | None = None,
        model_source: str = DEFAULT_MODEL_SOURCE,
        model_cfg_path: Path | str | None = None,
        lookback_strategy: str = 'multi_window',
        model_context_length: int | None = None,
        delta_lookback_days_mw: list[int] | None = None,
        n_anniversaries_for_mw: int = DEFAULT_N_ANNIVERSARIES_FOR_MW,
        device: str = 'cpu',  # to avoid annoying validation errors on GPU devices
    ) -> 'RunConfigData':
        """Transform input table from dist-s1-enumerator into RunConfigData object.

        Algorithm parameters should be assigned via attributes after creation.
        """
        df_pre = product_df[product_df.input_category == 'pre'].reset_index(drop=True)
        df_post = product_df[product_df.input_category == 'post'].reset_index(drop=True)

        # Create algorithm config with provided algorithm parameters
        algo_config = AlgoConfigData(
            max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
            delta_lookback_days_mw=delta_lookback_days_mw,
            lookback_strategy=lookback_strategy,
            post_date_buffer_days=DEFAULT_POST_DATE_BUFFER_DAYS,
            model_context_length=model_context_length,
            model_source=model_source,
            model_cfg_path=model_cfg_path,
            n_anniversaries_for_mw=n_anniversaries_for_mw,
            device=device,
        )

        runconfig_data = RunConfigData(
            pre_rtc_copol=df_pre.loc_path_copol.tolist(),
            pre_rtc_crosspol=df_pre.loc_path_crosspol.tolist(),
            post_rtc_copol=df_post.loc_path_copol.tolist(),
            post_rtc_crosspol=df_post.loc_path_crosspol.tolist(),
            mgrs_tile_id=df_pre.mgrs_tile_id.iloc[0],
            dst_dir=dst_dir,
            apply_water_mask=apply_water_mask,
            src_water_mask_path=water_mask_path,
            prior_dist_s1_product=prior_dist_s1_product,
            algo_config=algo_config,
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
            df = pd.merge(
                self.df_copol_data[['opera_id', 'loc_path_copol', 'input_category']],
                self.df_crosspol_data[['opera_id', 'loc_path_crosspol', 'input_category']],
                on=['opera_id', 'input_category'],
                how='inner',
            )
            df = df[['opera_id', 'loc_path_copol', 'loc_path_crosspol', 'input_category']]
            df['opera_id'] = df.loc_path_copol.apply(get_opera_id)
            df['jpl_burst_id'] = df.loc_path_copol.apply(get_burst_id).astype(str)
            df['track_number'] = df.loc_path_copol.apply(get_track_number)
            df['acq_dt'] = df.loc_path_copol.apply(get_acquisition_datetime)
            df['polarizations'] = df.apply(get_polarization_from_row, axis=1)
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

    @computed_field
    @property
    def water_mask_path(self) -> Path | None:
        """Get the water mask path, processing if needed when apply_water_mask is True."""
        if self.apply_water_mask:
            return self.processed_water_mask_path
        return self.src_water_mask_path

    @property
    def processed_water_mask_path(self) -> Path | None:
        """Get the processed water mask path, generating it if needed."""
        if not self.apply_water_mask:
            return None

        if self._processed_water_mask_path is not None:
            return self._processed_water_mask_path

        processed_path = water_mask_control_flow(
            water_mask_path=self.src_water_mask_path,
            mgrs_tile_id=self.mgrs_tile_id,
            dst_dir=self.dst_dir,
            overwrite=True,
        )
        self._processed_water_mask_path = processed_path
        return processed_path

    @model_validator(mode='after')
    def handle_input_data_dir_default(self) -> 'RunConfigData':
        """Set input_data_dir to dst_dir if None."""
        if self.input_data_dir is None:
            self.input_data_dir = Path(self.dst_dir)
        return self

    @model_validator(mode='after')
    def validate_burst_ids_in_mgrs_tile(self) -> 'RunConfigData':
        """Validate that the jpl_burst_ids are in the specified MGRS tile."""
        df_mgrs_burst = get_lut_by_mgrs_tile_ids(self.mgrs_tile_id)
        if df_mgrs_burst.empty:
            raise ValueError('The MGRS tile specified is not processed by DIST-S1')
        provided_jpl_burst_ids = set(self.df_inputs.jpl_burst_id.unique())
        allowed_jpl_burst_ids = set(df_mgrs_burst.jpl_burst_id.unique())
        supplied_jpl_burst_ids_not_in_mgrs_tile = provided_jpl_burst_ids - allowed_jpl_burst_ids
        if len(supplied_jpl_burst_ids_not_in_mgrs_tile) > 0:
            raise ValueError(
                'The following jpl burst IDs are not in the specified MGRS tile: '
                f'{supplied_jpl_burst_ids_not_in_mgrs_tile}'
            )
        return self

    @model_validator(mode='after')
    def ensure_consistent_polarizations_per_burst(self) -> 'RunConfigData':
        """Ensure that each burst has consistent polarizations across all acquisitions."""
        df = self.df_inputs
        df_burst_grouped = df.groupby('jpl_burst_id')['polarizations'].nunique()
        inconsistent_bursts = df_burst_grouped[df_burst_grouped > 1]
        if len(inconsistent_bursts) > 0:
            raise ValueError(
                f'The following bursts have inconsistent polarizations across acquisitions: '
                f'{inconsistent_bursts.index.tolist()}'
            )
        return self

    @model_validator(mode='after')
    def validate_input_data(self) -> 'RunConfigData':
        """Validate the input data across pre-/post-acquisition sets and across polarizations."""
        if self.df_inputs.empty:
            raise ValueError('The input data DataFrame is empty')
        pre_jpl_burst_ids = self.df_inputs[self.df_inputs.input_category == 'pre'].jpl_burst_id.unique().tolist()
        post_jpl_burst_ids = self.df_inputs[self.df_inputs.input_category == 'post'].jpl_burst_id.unique().tolist()

        pre_jpl_burst_ids_not_in_post = set(pre_jpl_burst_ids) - set(post_jpl_burst_ids)
        post_jpl_burst_ids_not_in_pre = set(post_jpl_burst_ids) - set(pre_jpl_burst_ids)
        if len(pre_jpl_burst_ids_not_in_post) > 0:
            raise ValueError(
                'The following jpl burst IDs are in pre-set not but not in post-set: '
                + ', '.join(pre_jpl_burst_ids_not_in_post)
            )
        if len(post_jpl_burst_ids_not_in_pre) > 0:
            raise ValueError(
                'The following jpl burst IDs are in post-set but not in pre-set: '
                + ', '.join(post_jpl_burst_ids_not_in_pre)
            )
        if (
            self.df_copol_data[self.df_copol_data.input_category == 'pre'].shape[0]
            != self.df_crosspol_data[self.df_crosspol_data.input_category == 'pre'].shape[0]
        ):
            raise ValueError('The number of baseline/pre-image set of copol and crosspol data is not the same')
        if (
            self.df_copol_data[self.df_copol_data.input_category == 'post'].shape[0]
            != self.df_crosspol_data[self.df_crosspol_data.input_category == 'post'].shape[0]
        ):
            raise ValueError(
                'The number of recent acquisition/post-image set of copol and crosspol data is not the same'
            )
        return self

    @model_validator(mode='after')
    def validate_dates_across_inputs(self) -> 'RunConfigData':
        # The dataframes should be sorted by jpl_burst_id and acq_dt
        copol_dates = self.df_copol_data.acq_dt.dt.date
        crosspol_dates = self.df_crosspol_data.acq_dt.dt.date
        # The length of these two dataframes must be the same for this comparison to make sense
        if len(copol_dates) != len(crosspol_dates):
            raise ValueError('The number of copol and crosspol data is not the same')
        if (copol_dates != crosspol_dates).any():
            copol_ids_without_crosspol = self.df_copol_data[copol_dates != crosspol_dates].opera_id.tolist()
            crosspol_ids_without_copol = self.df_crosspol_data[copol_dates != crosspol_dates].opera_id.tolist()
            msg_copol = f'The following copol products do not have crosspol dates: {copol_ids_without_crosspol}.'
            msg_crosspol = f'The following crosspol products do not have copol dates: {crosspol_ids_without_copol}.'
            raise ValueError(
                'There are discrepancies between copol and crosspol data:\n' + msg_copol + '\n' + msg_crosspol
            )
        return self

    @model_validator(mode='after')
    def handle_algo_config_loading(self) -> 'RunConfigData':
        """
        Load the algo config from the yaml file if it is not already loaded.

        This must occur before any validation of inputs that require algo config!
        This includes `validate_model_context_length`, `validate_unique_inputs`, and
        `validate_single_pass_for_post_data`.
        """
        if self.algo_config_path is not None and self._algo_config_loaded is False:
            algo_config_data = AlgoConfigData.from_yaml(self.algo_config_path)
            self.algo_config.__dict__.update(algo_config_data.model_dump())
            self._algo_config_loaded = True
        return self

    @model_validator(mode='after')
    def validate_model_context_length(self) -> 'RunConfigData':
        context_length = self.algo_config.model_context_length
        if context_length > DEFAULT_MODEL_CONTEXT_LENGTH_MAXIMUM:
            raise ValueError(
                f'The model context length is greater than maximum allowed:{DEFAULT_MODEL_CONTEXT_LENGTH_MAXIMUM}'
            )
        df_inputs = self.df_inputs
        df_pre = df_inputs[df_inputs.input_category == 'pre'].reset_index(drop=True)
        df_pre_by_burst = df_pre.groupby('jpl_burst_id')[['acq_dt']].nunique().reset_index(drop=False)
        bad_bursts = df_pre_by_burst[df_pre_by_burst.acq_dt > context_length]['jpl_burst_id'].tolist()
        if len(bad_bursts) > 0:
            raise ValueError(
                f'The following bursts have more than model ({self.algo_config.model_source}) context length of '
                f'({context_length}) pre-images: '
                f'{", ".join(bad_bursts)}'
            )
        return self

    @model_validator(mode='after')
    def validate_unique_inputs(self) -> 'RunConfigData':
        df_inputs = self.df_inputs.copy()
        df_inputs['dedup_id'] = df_inputs.opera_id.map(get_opera_id_without_proccessing_time)
        duplicated_ind = df_inputs[['dedup_id']].duplicated()
        if duplicated_ind.any():
            duplicated_opera_ids = df_inputs[duplicated_ind].opera_id.tolist()
            raise ValueError(f'The following products are duplicated: {", ".join(duplicated_opera_ids)}')
        return self

    @model_validator(mode='after')
    def validate_single_pass_for_post_data(self) -> 'RunConfigData':
        df_post = self.df_inputs[self.df_inputs.input_category == 'post'].reset_index(drop=True)
        min_acq_date = df_post.acq_dt.min()
        max_acq_date = df_post.acq_dt.max()
        if (min_acq_date - max_acq_date).total_seconds() > (60 * 20):  # more than 20 minutes difference
            raise ValueError(
                'The minimum acquisition date is more than 20 minutes greaterthan the maximum acquisition date:'
                f'{min_acq_date} - {max_acq_date}'
            )
        return self

    def __setattr__(self, name: str, value: object) -> None:
        if name == 'src_water_mask_path':
            super().__setattr__('_processed_water_mask_path', None)
        super().__setattr__(name, value)

    @field_serializer('prior_dist_s1_product')
    def serialize_prior_dist_s1_product(self, prior_dist_s1_product: DistS1ProductDirectory | Path | str | None) -> str:
        if prior_dist_s1_product is None:
            return prior_dist_s1_product
        return str(prior_dist_s1_product)
