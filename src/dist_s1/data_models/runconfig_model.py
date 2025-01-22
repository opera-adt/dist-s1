from datetime import datetime
from pathlib import Path, PosixPath

import geopandas as gpd
import pandas as pd
import yaml
from dist_s1_enumerator.asf import append_pass_data, extract_pass_id
from dist_s1_enumerator.data_models import dist_s1_loc_input_schema
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from pandera import check_input
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from yaml import Dumper

from dist_s1.constants import N_LOOKBACKS
from dist_s1.data_models.output_models import ProductDirectoryData, ProductNameData


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
    if lookback is not None:
        acq_date = date_lut[burst_id][N_LOOKBACKS - lookback - 1]
    else:
        acq_date = row.acq_dt
    acq_date_str = acq_date.date().strftime('%Y-%m-%d')
    fn = f'{path_token}_{burst_id}_{acq_date_str}.tif'
    if lookback is not None:
        fn = fn.replace('.tif', f'_delta{lookback}.tif')
    if polarization_token is not None:
        fn = fn.replace('.tif', f'_{polarization_token}.tif')
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
    dst_dir: Path | str | None = None
    water_mask: Path | str | None = None
    check_input_paths: bool = True

    # Private attributes that are associated to properties
    _burst_ids: list[str] | None = None
    _df_inputs: pd.DataFrame | None = None
    _df_burst_distmetric: pd.DataFrame | None = None
    _df_mgrs_burst_lut: gpd.GeoDataFrame | None = None
    _product_name: ProductNameData | None = None
    _product_dir_data: ProductDirectoryData | None = None
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
        return cls(**runconfig_data)

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

    @field_validator('dst_dir', mode='before')
    def validate_dst_dir(cls, dst_dir: Path | str | None, info: ValidationInfo) -> Path:
        if dst_dir is None:
            dst_dir = Path.cwd()
        dst_dir = Path(dst_dir) if isinstance(dst_dir, str) else dst_dir
        if dst_dir.exists() and not dst_dir.is_dir():
            raise ValidationError(f"Path '{dst_dir}' exists but is not a directory")
        dst_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir

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
    def product_dir_data(self) -> ProductDirectoryData:
        if self._product_dir_data is None:
            product_name = self.product_name
            self._product_dir_data = ProductDirectoryData(
                dst_dir=self.dst_dir,
                product_name=product_name,
            )
        return self._product_dir_data

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Save configuration to a YAML file."""
        # Get only the non-private attributes (those that don't start with _)
        config_dict = {k: v for k, v in self.model_dump().items() if not k.startswith('_')}
        config_dict.pop('check_input_paths', None)
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
        water_mask: Path | str | None = None,
    ) -> 'RunConfigData':
        df_pre = product_df[product_df.input_category == 'pre'].reset_index(drop=True)
        df_post = product_df[product_df.input_category == 'post'].reset_index(drop=True)
        runconfig_data = RunConfigData(
            pre_rtc_copol=df_pre.loc_path_copol.tolist(),
            pre_rtc_crosspol=df_pre.loc_path_crosspol.tolist(),
            post_rtc_copol=df_post.loc_path_copol.tolist(),
            post_rtc_crosspol=df_post.loc_path_crosspol.tolist(),
            mgrs_tile_id=df_pre.mgrs_tile_id.iloc[0],
            dst_dir=dst_dir,
            water_mask=water_mask,
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
    def df_burst_distmetrics(self) -> pd.DataFrame:
        if self._df_burst_distmetric is None:
            normal_param_dir = self.dst_dir / 'normal_params'
            normal_param_dir.mkdir(parents=True, exist_ok=True)

            df_inputs = self.df_inputs.copy()
            df_post = df_inputs[df_inputs.input_category == 'post'].reset_index(drop=True)
            burst_ids = df_post.jpl_burst_id.unique()
            df_dist_by_burst = pd.DataFrame({'jpl_burst_id': burst_ids})

            df_date = df_inputs.groupby('jpl_burst_id')['acq_dt'].apply(max).reset_index(drop=False)
            df_dist_by_burst = pd.merge(df_dist_by_burst, df_date, on='jpl_burst_id', how='left')

            # Get the N_LOOKBACKS most recent dates before the current acquisition
            df_pre = df_inputs[df_inputs.input_category == 'pre'].reset_index(drop=True)
            df_pre.sort_values(by=['jpl_burst_id', 'acq_dt'], inplace=True, ascending=True)
            df_date_pre = df_pre.groupby('jpl_burst_id')['acq_dt'].apply(
                lambda x: sorted(x.nlargest(N_LOOKBACKS).tolist())
            )
            burst2predates = df_date_pre.to_dict()
            df_date = df_inputs.groupby('jpl_burst_id')['acq_dt'].apply(
                lambda x: sorted(x.nlargest(N_LOOKBACKS).tolist())
            )
            burst2postdates = df_date.to_dict()

            # Distribution Paths
            for lookback in range(N_LOOKBACKS):
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
                        )

            # Metrics Paths
            df_dist_by_burst['loc_path_metric_delta0'] = df_dist_by_burst.apply(
                generate_burst_dist_paths,
                top_level_data_dir=self.dst_dir,
                dst_dir_name='metrics',
                path_token='distmetric',
                lookback=0,
                date_lut=burst2postdates,
                axis=1,
            )

            # Disturbance Paths for Each Lookback
            for lookback in range(N_LOOKBACKS):
                df_dist_by_burst[f'loc_path_disturb_delta{lookback}'] = df_dist_by_burst.apply(
                    generate_burst_dist_paths,
                    top_level_data_dir=self.dst_dir,
                    dst_dir_name='disturbance',
                    path_token='disturb',
                    lookback=lookback,
                    date_lut=burst2postdates,
                    axis=1,
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
