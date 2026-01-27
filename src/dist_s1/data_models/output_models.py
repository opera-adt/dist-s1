import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import ClassVar
from warnings import warn

import numpy as np
import rasterio
from pydantic import BaseModel, Field, field_validator, model_validator

from dist_s1.aws import (
    check_s3_object_exists,
    check_s3_prefix_exists,
    download_product_from_s3,
    is_s3_path,
    parse_s3_uri,
    rasterio_anon_s3_env,
)
from dist_s1.constants import (
    EXPECTED_FORMAT_STRING,
    MAX_FLOAT_LAYER_DIFF,
    MAX_INT_LAYER_DIFF,
    PRODUCT_VERSION,
    TIF_LAYERS,
    TIF_LAYER_DTYPES,
    TIF_LAYER_NODATA_VALUES,
)
from dist_s1.data_models.data_utils import compare_dist_s1_product_tag, get_acquisition_datetime
from dist_s1.rio_tools import get_mgrs_profile
from dist_s1.water_mask import apply_water_mask


PRODUCT_TAGS_FOR_EQUALITY = [
    'pre_rtc_opera_ids',
    'post_rtc_opera_ids',
    'low_confidence_alert_threshold',
    'high_confidence_alert_threshold',
    'model_source',
    'prior_dist_s1_product',
    'sensor',
]
REQUIRED_PRODUCT_TAGS = PRODUCT_TAGS_FOR_EQUALITY + ['version']


@dataclass
class LayerComparisonResult:
    layer_name: str
    is_equal: bool
    max_difference: float
    tolerance_used: float
    arrays_close: bool
    metadata_matches: bool
    percent_mismatch: float
    percent_nodata_mismatch: float
    error_message: str = ''


@dataclass
class ProductComparisonResult:
    is_equal: bool
    mgrs_matches: bool
    datetime_matches: bool
    layer_results: dict[str, LayerComparisonResult]

    @property
    def failed_layers(self) -> list[str]:
        return [name for name, result in self.layer_results.items() if not result.is_equal]


class ProductNameData(BaseModel):
    mgrs_tile_id: str = Field(description='MGRS (Military Grid Reference System) tile identifier')
    acq_date_time: datetime = Field(description='Acquisition datetime of the Sentinel-1 data')
    processing_date_time: datetime = Field(description='Processing datetime when the product was generated')
    sensor: str = Field(description='Sensor identifier', pattern=r'^S1[ABC]$')

    def __str__(self) -> str:
        tokens = [
            'OPERA',
            'L3',
            'DIST-ALERT-S1',
            f'T{self.mgrs_tile_id}',
            self.acq_date_time.strftime('%Y%m%dT%H%M%SZ'),
            self.processing_date_time.strftime('%Y%m%dT%H%M%SZ'),
            self.sensor,
            '30',
            f'v{PRODUCT_VERSION}',
        ]
        return '_'.join(tokens)

    def name(self) -> str:
        return f'{self}'

    @classmethod
    def validate_product_name(cls, product_name: str) -> bool:
        """
        Validate if a string matches the OPERA L3 DIST-ALERT-S1 product name format.

        Expected format:
        OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{version}
        """
        tokens = product_name.split('_')

        # Check if we have the correct number of tokens first
        if len(tokens) != 9:
            return False

        conditions = [
            tokens[0] != 'OPERA',
            tokens[1] != 'L3',
            tokens[2] != 'DIST-ALERT-S1',
            not tokens[3].startswith('T'),  # MGRS tile ID
            tokens[6] not in ['S1A', 'S1B', 'S1C'],
            tokens[7] != '30',
            not tokens[8].startswith('v'),  # Version
        ]

        # If any condition is True, validation fails
        if any(conditions):
            return False

        # Validate datetime formats
        datetime.strptime(tokens[4], '%Y%m%dT%H%M%SZ')  # Acquisition datetime
        datetime.strptime(tokens[5], '%Y%m%dT%H%M%SZ')  # Processing datetime

        return True


class ProductFileData(BaseModel):
    path: Path

    @classmethod
    def from_product_path(cls, product_path: str) -> 'ProductFileData':
        """Instantiate from a file path."""
        return cls(path=product_path)

    def compare(
        self, other: 'ProductFileData', rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = True
    ) -> tuple[bool, str]:
        """Compare two GeoTIFF files for equality.

        Parameters
        ----------
        other : ProductFileData
            GeoTIFF file to compare against.
        rtol : float, optional
            Relative tolerance for numpy.allclose, default 1e-05.
        atol : float, optional
            Absolute tolerance for numpy.allclose, default 1e-08.
        equal_nan : bool, optional
            Whether NaN values should be considered equal, default True.

        Returns
        -------
        tuple (bool, str)
            - True if the files match, False otherwise.
            - A message describing differences if files do not match.
        """
        # Check if both files exist
        if not self.path.exists():
            return False, f'File not found: {self.path}'
        if not other.path.exists():
            return False, f'File not found: {other.path}'

        with rasterio.open(self.path) as src_self, rasterio.open(other.path) as src_other:
            # Compare image dimensions
            if src_self.shape != src_other.shape:
                return False, f'Shape mismatch: {src_self.shape} != {src_other.shape}'

            # Read raster data
            data_self = src_self.read()
            data_other = src_other.read()

            # Find pixel mismatches
            diff_mask = ~np.isclose(data_self, data_other, rtol=rtol, atol=atol, equal_nan=equal_nan)
            mismatch_count = np.sum(diff_mask)

            if mismatch_count > 0:
                max_diff = np.max(np.abs(data_self - data_other))
                min_diff = np.min(np.abs(data_self - data_other))
                return False, (
                    f'Pixel mismatch count: {mismatch_count}\nMax difference: {max_diff}\nMin difference: {min_diff}'
                )

            # Compare metadata (tags)
            tags_self = src_self.tags()
            tags_other = src_other.tags()
            mismatched_tags = {
                key: (tags_self[key], tags_other[key])
                for key in [key for key in tags_self.keys() if key in PRODUCT_TAGS_FOR_EQUALITY]
                if not compare_dist_s1_product_tag(key, tags_self[key], tags_other[key])
            }

            if mismatched_tags:
                return False, f'Metadata mismatch in tags: {mismatched_tags}'

        return True, 'Files match perfectly.'


class DistS1ProductDirectory(BaseModel):
    product_name: str
    dst_dir: Path | str
    tif_layer_dtypes: ClassVar[dict[str, str]] = dict(TIF_LAYER_DTYPES)

    @property
    def product_dir_path(self) -> Path | str:
        if is_s3_path(str(self.dst_dir)):
            bucket, key = parse_s3_uri(str(self.dst_dir))
            key = key.rstrip('/')
            product_key = f'{key}/{self.product_name}' if key else self.product_name
            return f's3://{bucket}/{product_key}'
        return self.dst_dir / self.product_name

    def __str__(self) -> str:
        return str(self.product_dir_path)

    @field_validator('product_name')
    def validate_product_name(cls, product_name: str) -> str:
        if not ProductNameData.validate_product_name(product_name):
            raise ValueError(f'Invalid product name: {product_name}; should match: {EXPECTED_FORMAT_STRING}')
        return product_name

    @field_validator('dst_dir')
    def validate_dst_dir(cls, dst_dir: Path | str) -> Path | str:
        if is_s3_path(dst_dir):
            return str(dst_dir)
        return Path(dst_dir)

    @model_validator(mode='after')
    def validate_product_directory(self) -> Path:
        if is_s3_path(str(self.dst_dir)):
            return self

        product_dir = self.product_dir_path
        if product_dir.exists() and not product_dir.is_dir():
            raise ValueError(f'Path {product_dir} exists but is not a directory')
        if not product_dir.exists():
            product_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def layers(self) -> list[str]:
        return list(TIF_LAYERS)

    @property
    def layer_path_dict(self) -> dict[str, Path | str]:
        if is_s3_path(str(self.dst_dir)):
            base_uri = str(self.product_dir_path)
            layer_dict = {layer: f'{base_uri}/{self.product_name}_{layer}.tif' for layer in self.layers}
            layer_dict['browse'] = f'{base_uri}/{self.product_name}_BROWSE.png'
            return layer_dict
        layer_dict = {layer: self.product_dir_path / f'{self.product_name}_{layer}.tif' for layer in self.layers}
        layer_dict['browse'] = self.product_dir_path / f'{self.product_name}_BROWSE.png'
        return layer_dict

    @property
    def acq_datetime(self) -> datetime:
        return get_acquisition_datetime(self.product_dir_path)

    @property
    def mgrs_tile_id(self) -> str:
        tokens = self.product_name.split('_')
        return tokens[3][1:]

    def validate_layer_paths(self) -> bool:
        failed_layers = []
        if is_s3_path(str(self.dst_dir)):
            for layer, path_or_uri in self.layer_path_dict.items():
                bucket, key = parse_s3_uri(path_or_uri)
                if not check_s3_object_exists(bucket, key):
                    warn(f'Layer {layer} does not exist: {path_or_uri}', UserWarning)
                    failed_layers.append(layer)
        else:
            for layer, path in self.layer_path_dict.items():
                if not path.exists():
                    warn(f'Layer {layer} does not exist at path: {path}', UserWarning)
                    failed_layers.append(layer)
        return len(failed_layers) == 0

    @rasterio_anon_s3_env
    def validate_tif_layer_dtypes(self) -> bool:
        failed_layers = []
        for layer, path_or_uri in self.layer_path_dict.items():
            if layer not in TIF_LAYERS:
                continue
            try:
                with rasterio.open(str(path_or_uri)) as src:
                    if src.dtypes[0] != TIF_LAYER_DTYPES[layer]:
                        warn(
                            f'Layer {layer} has incorrect dtype: {src.dtypes[0]}; should be: {TIF_LAYER_DTYPES[layer]}',
                            UserWarning,
                        )
                        failed_layers.append(layer)
            except Exception as e:
                warn(f'Failed to validate layer {layer}: {e}', UserWarning)
                failed_layers.append(layer)
        return len(failed_layers) == 0

    @rasterio_anon_s3_env
    def compare_products(self, other: 'DistS1ProductDirectory') -> ProductComparisonResult:
        """Compare two ProductDirectoryData instances with detailed results.

        Parameters
        ----------
        other : ProductDirectoryData
            The other instance to compare against

        Returns
        -------
        ProductComparisonResult
            Detailed comparison results
        """
        tokens_self = self.product_name.split('_')
        tokens_other = other.product_name.split('_')

        mgrs_self = tokens_self[3][1:]
        mgrs_other = tokens_other[3][1:]
        mgrs_matches = mgrs_self == mgrs_other

        acq_dt_self = datetime.strptime(tokens_self[4], '%Y%m%dT%H%M%SZ')
        acq_dt_other = datetime.strptime(tokens_other[4], '%Y%m%dT%H%M%SZ')
        datetime_matches = acq_dt_self == acq_dt_other

        layer_results = {}
        for layer in self.layers:
            path_self = self.layer_path_dict[layer]
            path_other = other.layer_path_dict[layer]

            with rasterio.open(path_self) as src_self, rasterio.open(path_other) as src_other:
                data_self = src_self.read(1)
                data_other = src_other.read(1)

                layer_dtype = self.tif_layer_dtypes[layer]
                tolerance = (
                    MAX_FLOAT_LAYER_DIFF if np.issubdtype(np.dtype(layer_dtype), np.floating) else MAX_INT_LAYER_DIFF
                )

                nodata_value = TIF_LAYER_NODATA_VALUES[layer]

                if np.issubdtype(np.dtype(layer_dtype), np.floating):
                    nodata_mask_self = np.isnan(data_self) if np.isnan(nodata_value) else data_self == nodata_value
                    nodata_mask_other = np.isnan(data_other) if np.isnan(nodata_value) else data_other == nodata_value
                else:
                    nodata_mask_self = data_self == nodata_value
                    nodata_mask_other = data_other == nodata_value

                nodata_masks_consistent = np.array_equal(nodata_mask_self, nodata_mask_other)
                if not nodata_masks_consistent:
                    warn(f'Layer {layer}: nodata masks are inconsistent between compared products', UserWarning)

                valid_mask_self = ~nodata_mask_self
                valid_mask_other = ~nodata_mask_other
                both_valid = valid_mask_self & valid_mask_other

                diff = np.zeros_like(data_self, dtype=np.float64)

                diff[both_valid] = np.abs(data_self[both_valid] - data_other[both_valid])

                one_nodata_self = nodata_mask_self & valid_mask_other
                one_nodata_other = nodata_mask_other & valid_mask_self

                diff[one_nodata_self] = np.abs(data_other[one_nodata_self])
                diff[one_nodata_other] = np.abs(data_self[one_nodata_other])

                max_diff = np.max(diff) if diff.size > 0 else 0.0

                arrays_close = max_diff <= tolerance

                total_pixels = data_self.size
                mismatched_valid_pixels = np.sum(diff[both_valid] > tolerance) if np.any(both_valid) else 0
                nodata_mismatch_pixels = np.sum(one_nodata_self | one_nodata_other)

                percent_mismatch = (mismatched_valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
                percent_nodata_mismatch = (nodata_mismatch_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

                tags_self = src_self.tags()
                tags_other = src_other.tags()
                metadata_matches = all(
                    compare_dist_s1_product_tag(key, tags_self[key], tags_other[key])
                    for key in PRODUCT_TAGS_FOR_EQUALITY
                )

                layer_equal = arrays_close and metadata_matches and nodata_masks_consistent

                layer_results[layer] = LayerComparisonResult(
                    layer_name=layer,
                    is_equal=layer_equal,
                    max_difference=max_diff,
                    tolerance_used=tolerance,
                    arrays_close=arrays_close,
                    metadata_matches=metadata_matches,
                    percent_mismatch=percent_mismatch,
                    percent_nodata_mismatch=percent_nodata_mismatch,
                    error_message='' if layer_equal else f'Max diff {max_diff:.6e} > tolerance {tolerance:.6e}',
                )

        # Overall equality
        overall_equal = mgrs_matches and datetime_matches and all(r.is_equal for r in layer_results.values())

        return ProductComparisonResult(
            is_equal=overall_equal,
            mgrs_matches=mgrs_matches,
            datetime_matches=datetime_matches,
            layer_results=layer_results,
        )

    def __eq__(self, other: 'DistS1ProductDirectory') -> bool:
        """Compare two ProductDirectoryData instances for equality.

        Checks that:
        1. The MGRS tile IDs match
        2. The acquisition datetimes match
        3. All TIF layers have numerically close data using layer-specific tolerances
        4. Nodata masks are consistent between layers

        Parameters
        ----------
        other : ProductDirectoryData
            The other instance to compare against

        Returns
        -------
        bool
            True if the instances are considered equal, False otherwise
        """
        result = self.compare_products(other)

        if not result.mgrs_matches:
            warn(f'MGRS tile IDs do not match: {self.mgrs_tile_id} != {other.mgrs_tile_id}', UserWarning)
        if not result.datetime_matches:
            warn(f'Acquisition datetimes do not match: {self.acq_datetime} != {other.acq_datetime}', UserWarning)
        for layer_result in result.layer_results.values():
            if not layer_result.is_equal:
                warn(
                    f'Layer {layer_result.layer_name}: {layer_result.error_message} with max diff'
                    f' {layer_result.max_difference:.6e}, {layer_result.percent_mismatch:.2f}% pixels mismatched,'
                    f' {layer_result.percent_nodata_mismatch:.2f}% nodata mismatched',
                    UserWarning,
                )

        return result.is_equal

    @classmethod
    def from_product_path(cls, product_dir_path: Path | str) -> 'DistS1ProductDirectory':
        if is_s3_path(product_dir_path):
            return cls._from_s3_product_path(str(product_dir_path))
        return cls._from_local_product_path(Path(product_dir_path))

    @classmethod
    def _from_local_product_path(cls, product_dir_path: Path) -> 'DistS1ProductDirectory':
        """Load from local filesystem."""
        if not product_dir_path.exists() or not product_dir_path.is_dir():
            raise ValueError(f'Product directory does not exist or is not a directory: {product_dir_path}')

        product_name = product_dir_path.name
        if not ProductNameData.validate_product_name(product_name):
            raise ValueError(f'Invalid product name: {product_name}')

        obj = cls(product_name=product_name, dst_dir=product_dir_path.parent)

        if not obj.validate_layer_paths():
            raise ValueError(f'Product directory missing required layers: {product_dir_path}')
        if not obj.validate_tif_layer_dtypes():
            raise ValueError(f'Product directory contains layers with incorrect dtypes: {product_dir_path}')

        return obj

    @classmethod
    def _from_s3_product_path(cls, s3_uri: str) -> 'DistS1ProductDirectory':
        """Load from S3 location."""
        bucket, key_prefix = parse_s3_uri(s3_uri)

        if not check_s3_prefix_exists(bucket, key_prefix):
            raise ValueError(f'S3 product directory does not exist: {s3_uri}')

        product_name = PurePosixPath(key_prefix).name
        if not ProductNameData.validate_product_name(product_name):
            raise ValueError(f'Invalid product name: {product_name}')

        parent_key = str(PurePosixPath(key_prefix).parent)
        s3_parent_uri = f's3://{bucket}/{parent_key}'

        obj = cls(product_name=product_name, dst_dir=s3_parent_uri)

        if not obj.validate_layer_paths():
            raise ValueError(f'S3 product missing required layers: {s3_uri}')
        if not obj.validate_tif_layer_dtypes():
            raise ValueError(f'S3 product has incorrect dtypes: {s3_uri}')

        return obj

    def download_to(self, dst_dir: Path | str, profile_name: str | None = None) -> 'DistS1ProductDirectory':
        if not is_s3_path(str(self.dst_dir)):
            raise ValueError('download_to only works with S3 data. Current product is already local.')

        downloaded_dir = download_product_from_s3(str(self.product_dir_path), dst_dir, profile_name)
        return self.from_product_path(downloaded_dir)

    def copy_to(self, dst_dir: Path | str, profile_name: str | None = None) -> 'DistS1ProductDirectory':
        if is_s3_path(str(self.dst_dir)):
            return self.download_to(dst_dir, profile_name)

        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_product_dir = self.product_dir_path
        dst_product_dir = dst_dir / self.product_name

        if dst_product_dir.exists():
            shutil.rmtree(dst_product_dir)

        shutil.copytree(src_product_dir, dst_product_dir)
        return self.from_product_path(dst_product_dir)
