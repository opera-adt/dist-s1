from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath

import geopandas as gpd
import yaml
from dist_s1_enumerator.mgrs_burst_data import get_lut_by_mgrs_tile_ids
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from .output_models import ProductDirectoryData, ProductNameData


def posix_path_encoder(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))


def none_encoder(dumper, _):
    return dumper.represent_scalar('tag:yaml.org,2002:null', '')


yaml.add_representer(PosixPath, posix_path_encoder)
yaml.add_representer(type(None), none_encoder)


def get_burst_id(opera_rtc_s1_path: Path) -> str:
    tokens = opera_rtc_s1_path.name.split('_')
    return tokens[3]


def get_acquisition_datetime(opera_rtc_s1_path: Path) -> datetime:
    tokens = opera_rtc_s1_path.name.split('_')
    try:
        return datetime.strptime(tokens[4], '%Y%m%dT%H%M%SZ')
    except ValueError:
        raise ValueError(f"Datetime token in filename '{opera_rtc_s1_path.name}' is not correctly formatted.")


class RunConfigData(BaseModel):
    pre_rtc_copol: list[Path | str]
    pre_rtc_crosspol: list[Path | str]
    post_rtc_copol: list[Path | str]
    post_rtc_crosspol: list[Path | str]
    mgrs_tile_id: str
    dst_dir: Path | str | None = None
    water_mask: Path | str | None = None

    # Private attributes that are associated to properties
    _time_series_by_burst: dict[str, dict[str, list[Path] | list[datetime]]] | None = None
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
    @classmethod
    def convert_to_paths(cls, values: list[Path | str]) -> list[Path]:
        """Convert all values to Path objects."""
        return [Path(value) if isinstance(value, str) else value for value in values]

    @field_validator('pre_rtc_crosspol', 'post_rtc_crosspol')
    @classmethod
    def check_matching_lengths_copol_and_crosspol(
        cls: type['RunConfigData'], rtc_crosspol: list[Path], info: ValidationInfo
    ) -> list[Path]:
        """Ensure pre_rtc_copol and pre_rtc_crosspol have the same length."""
        key = 'pre_rtc_copol' if info.field_name == 'pre_rtc_crosspol' else 'post_rtc_copol'
        rtc_copol = info.data.get(key)
        if rtc_copol is not None and len(rtc_copol) != len(rtc_crosspol):
            raise ValidationError("The lists 'pre_rtc_copol' and 'pre_rtc_crosspol' must have the same length.")
        return rtc_crosspol

    @classmethod
    def check_filename_format(cls, file_path: Path, field) -> None:
        """Check the filename format to ensure correct structure and tokens."""
        filename = file_path.name
        tokens = filename.split('_')
        if len(tokens) != 10 or tokens[0] != 'OPERA' or tokens[1] != 'L2' or tokens[2] != 'RTC-S1':
            raise ValidationError(f"File '{filename}' in {field.name} does not match the required format.")
        if 'copol' in field.name and not filename.endswith('_VV.tif'):
            raise ValidationError(f"File in {field.name} should end with '_VV.tif'")
        elif 'crosspol' in field.name and not filename.endswith('_VH.tif'):
            raise ValidationError(f"File in {field.name} should end with '_VH.tif'")

    @field_validator('mgrs_tile_id')
    @classmethod
    def validate_mgrs_tile_id(cls, mgrs_tile_id: str) -> str:
        """Validate that mgrs_tile_id is present in the lookup table."""
        df_mgrs_burst = get_lut_by_mgrs_tile_ids(mgrs_tile_id)
        if df_mgrs_burst.empty:
            raise ValidationError('The MGRS tile specified is not processed by DIST-S1')
        return mgrs_tile_id

    @field_validator('dst_dir', mode='before')
    @classmethod
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

    # @field_validator('water_mask', mode='after')
    # @classmethod
    # def validate_water_mask(self) -> Path:
    #     """Validate that water_mask exists and contains the MGRS tile."""
    #     if self.water_mask is None:
    #         return None
    #     wm_path = Path(self.water_mask)
    #     if not wm_path.exists():
    #         raise ValidationError(f"Water mask file '{wm_path}' does not exist")
    #     with rasterio.open(wm_path) as ds:
    #         bounds = ds.bounds
    #     water_geo = box(*bounds)
    #     df_mgrs_tiles = get_mgrs_table()
    #     df_mgrs_tiles = df_mgrs_tiles[df_mgrs_tiles.mgrs_tile_id == self.mgrs_tile_id].reset_index(drop=True)
    #     tile_geo = df_mgrs_tiles.geometry.iloc[0]
    #     containment = water_geo.contains(tile_geo)
    #     if not containment:
    #         raise ValidationError(f"Water mask file '{wm_path}' does not contain the MGRS tile {self.mgrs_tile_id}")
    #     return wm_path
    @property
    def product_dir_data(self) -> ProductDirectoryData:
        if self._product_dir_data is None:
            product_name = self.product_name
            self._product_dir_data = ProductDirectoryData(
                dst_dir=self.dst_dir,
                product_name=product_name,
            )
        return self._product_dir_data

    @property
    def time_series_by_burst(self) -> dict[str, dict[str, list[Path] | list[datetime]]]:
        if self._time_series_by_burst is not None:
            return self._time_series_by_burst

        organized_data: dict[str, dict[str, list[Path] | list[datetime]]] = defaultdict(
            lambda: {
                'pre_rtc_copol': [],
                'pre_rtc_crosspol': [],
                'post_rtc_copol': [],
                'post_rtc_crosspol': [],
                'pre_acq_dts': [],
                'post_acq_dts': [],
            }
        )

        for field_name in ['pre_rtc_copol', 'pre_rtc_crosspol', 'post_rtc_copol', 'post_rtc_crosspol']:
            file_list = sorted(getattr(self, field_name))
            for file_path in file_list:
                burst_id = get_burst_id(file_path)
                organized_data[burst_id][field_name].append(file_path)

        # Check all the pre/post copol and crosspol have the same length
        for burst_id in organized_data.keys():
            if len(organized_data[burst_id]['pre_rtc_copol']) != len(organized_data[burst_id]['pre_rtc_crosspol']):
                raise ValueError(
                    f'Pre-acquisition data for burst {burst_id} do not have the same length between copol and crosspol.'
                )
            if len(organized_data[burst_id]['post_rtc_copol']) != len(organized_data[burst_id]['post_rtc_crosspol']):
                raise ValueError(
                    f'Post-acquisition data for burst {burst_id} do '
                    'not have the same length between copol and crosspol.'
                )

        # Set acq_dt and ensure all dates are consistent for copol and crosspol (per burst)
        for burst_id in organized_data.keys():
            # Set pre and post acq dts using copol
            organized_data[burst_id]['pre_acq_dts'] = [
                get_acquisition_datetime(file_path) for file_path in organized_data[burst_id]['pre_rtc_copol']
            ]
            organized_data[burst_id]['post_acq_dts'] = [
                get_acquisition_datetime(file_path) for file_path in organized_data[burst_id]['post_rtc_copol']
            ]

            # Check the acq_dts of crosspol are consistent with copol
            pre_crosspol_dts = [
                get_acquisition_datetime(file_path) for file_path in organized_data[burst_id]['pre_rtc_crosspol']
            ]
            post_crosspol_dts = [
                get_acquisition_datetime(file_path) for file_path in organized_data[burst_id]['post_rtc_crosspol']
            ]
            if pre_crosspol_dts != organized_data[burst_id]['pre_acq_dts']:
                error_msg = (
                    f'Pre-acquisition dates for burst {burst_id} do not match between copol and crosspol. '
                    f'copol: {organized_data[burst_id]["pre_acq_dts"]}, and crosspol: {pre_crosspol_dts}'
                )
                raise ValueError(error_msg)
            if post_crosspol_dts != organized_data[burst_id]['post_acq_dts']:
                error_msg = (
                    f'Post-acquisition dates for burst {burst_id} do not match between copol and crosspol.'
                    f'copol: {organized_data[burst_id]["post_acq_dts"]}, and crosspol: {post_crosspol_dts}'
                )
                raise ValueError(error_msg)

        self._time_series_by_burst = dict(organized_data)

        # Check the data is indeed contained in the MGRS tile
        burst_ids_input = list(organized_data.keys())
        df_mgrs_burst = get_lut_by_mgrs_tile_ids(self.mgrs_tile_id)
        burst_ids_in_mgrs_tile = df_mgrs_burst.jpl_burst_id.tolist()
        bids_outside_of_mgrs_tile = [bid for bid in burst_ids_input if bid not in burst_ids_in_mgrs_tile]
        if bids_outside_of_mgrs_tile:
            bids_outside_str = ', '.join(bids_outside_of_mgrs_tile)
            raise ValueError(f'Some of the burst ids are not within the MGRS tile {bids_outside_str}')

        # Check that all bursts are within the same acquisition group
        df_pass = df_mgrs_burst[df_mgrs_burst.jpl_burst_id.isin(burst_ids_input)].reset_index(drop=True)
        group_ids = df_pass.acq_group_id_within_mgrs_tile.unique()
        if len(group_ids) > 1:
            raise ValueError('Multiple Acquisition Groups within the MGRS tile were specified in input')

        return self._time_series_by_burst

    def to_yaml(self, yaml_file: str | Path) -> None:
        """Save configuration to a YAML file."""
        config_dict = self.model_dump()
        config_dict.pop('_time_series_by_burst', None)
        config_dict.pop('_df_mgrs_burst_lut', None)
        yml_dict = {'run_config': config_dict}

        # Write to YAML file
        yaml_file = Path(yaml_file)
        with yaml_file.open('w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False, indent=4, sort_keys=False)


def create_runconfig_from_input_df(input_df: gpd.GeoDataFrame) -> RunConfigData:
    return RunConfigData()
