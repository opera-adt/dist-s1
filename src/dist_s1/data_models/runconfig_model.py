from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from dist_s1_enumerator.tabular_models import dist_s1_loc_input_schema
from pandera.pandas import check_input
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from dist_s1.data_models.algoconfig_model import AlgoConfigData
from dist_s1.data_models.data_utils import (
    check_filename_format,
    get_acquisition_datetime,
    get_burst_id,
    get_opera_id,
    get_track_number,
)
from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_CHECK_INPUT_PATHS,
    DEFAULT_DST_DIR,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_POST_DATE_BUFFER_DAYS,
)
from dist_s1.data_models.output_models import DistS1ProductDirectory, ProductNameData
from dist_s1.water_mask import water_mask_control_flow


class RunConfigData(BaseModel):
    pre_rtc_copol: list[Path | str] = Field(..., description='List of paths to pre-rtc copolarization data.')
    pre_rtc_crosspol: list[Path | str] = Field(..., description='List of paths to pre-rtc crosspolarization data.')
    post_rtc_copol: list[Path | str] = Field(..., description='List of paths to post-rtc copolarization data.')
    post_rtc_crosspol: list[Path | str] = Field(..., description='List of paths to post-rtc crosspolarization data.')
    prior_dist_s1_product: DistS1ProductDirectory | None = Field(
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
    water_mask_path: Path | str | None = Field(
        default=None,
        description='Path to water mask. If None, no water mask is used.',
    )
    apply_water_mask: bool = Field(
        default=DEFAULT_APPLY_WATER_MASK,
        description='Whether to apply water mask to the input data. If True, water mask is applied to the input data. '
        'If no water mask path is provided, the tiles to generate the water mask are localized and formatted for use.',
    )
    check_input_paths: bool = Field(
        default=DEFAULT_CHECK_INPUT_PATHS,
        description='Whether to check if the input paths exist. If True, the input paths are checked. '
        'Used during testing.',
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
        return input_data_dir.resolve()

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
        if prior_dist_s1_product is None:
            return None
        if isinstance(prior_dist_s1_product, DistS1ProductDirectory):
            return prior_dist_s1_product
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
    def product_name(self) -> ProductNameData:
        if self._product_name is None:
            self._product_name = ProductNameData(
                mgrs_tile_id=self.mgrs_tile_id,
                acq_date_time=self.min_acq_date,
                processing_date_time=self.processing_datetime,
            )
        return self._product_name.name()

    @property
    def product_data_model(self) -> DistS1ProductDirectory:
        if self._product_data_model is None:
            product_name = self.product_name
            dst_dir = (
                Path(self.product_dst_dir).resolve()
                if self.product_dst_dir is not None
                else Path(self.dst_dir).resolve()
            )
            self._product_data_model = DistS1ProductDirectory(
                dst_dir=dst_dir,
                product_name=product_name,
            )
        return self._product_data_model

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
        lookback_strategy: str = 'multi_window',
        prior_dist_s1_product: DistS1ProductDirectory | None = None,
    ) -> 'RunConfigData':
        """Transform input table from dist-s1-enumerator into RunConfigData object.

        Algorithm parameters should be assigned via attributes after creation.
        """
        df_pre = product_df[product_df.input_category == 'pre'].reset_index(drop=True)
        df_post = product_df[product_df.input_category == 'post'].reset_index(drop=True)
        if max_pre_imgs_per_burst_mw is None:
            from dist_s1.data_models.defaults import DEFAULT_MAX_PRE_IMGS_PER_BURST_MW

            max_pre_imgs_per_burst_mw = DEFAULT_MAX_PRE_IMGS_PER_BURST_MW
        if delta_lookback_days_mw is None:
            from dist_s1.data_models.defaults import DEFAULT_DELTA_LOOKBACK_DAYS_MW

            delta_lookback_days_mw = DEFAULT_DELTA_LOOKBACK_DAYS_MW

        # Create algorithm config with provided algorithm parameters
        algo_config = AlgoConfigData(
            max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
            delta_lookback_days_mw=delta_lookback_days_mw,
            lookback_strategy=lookback_strategy,
            post_date_buffer_days=DEFAULT_POST_DATE_BUFFER_DAYS,
        )

        runconfig_data = RunConfigData(
            pre_rtc_copol=df_pre.loc_path_copol.tolist(),
            pre_rtc_crosspol=df_pre.loc_path_crosspol.tolist(),
            post_rtc_copol=df_post.loc_path_copol.tolist(),
            post_rtc_crosspol=df_post.loc_path_crosspol.tolist(),
            mgrs_tile_id=df_pre.mgrs_tile_id.iloc[0],
            dst_dir=dst_dir,
            apply_water_mask=apply_water_mask,
            water_mask_path=water_mask_path,
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
    def handle_input_data_dir_default(self) -> 'RunConfigData':
        """Set input_data_dir to dst_dir if None."""
        if self.input_data_dir is None:
            self.input_data_dir = self.dst_dir
        return self

    @model_validator(mode='after')
    def handle_algo_config_loading(self) -> 'RunConfigData':
        if self.algo_config_path is not None and self._algo_config_loaded is False:
            algo_config_data = AlgoConfigData.from_yaml(self.algo_config_path)
            self.algo_config.__dict__.update(algo_config_data.model_dump())
            self._algo_config_loaded = True
        return self
