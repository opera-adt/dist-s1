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
from dist_s1.data_models.data_utils import (
    compare_dist_s1_product_tag,
    get_acquisition_datetime,
    validate_dist_s1_product_name,
)
from dist_s1.rio_tools import serialize_one_2d_ds


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


def _get_nodata_mask(data: np.ndarray, nodata_value: float | int, layer_dtype: str) -> np.ndarray:
    if np.issubdtype(np.dtype(layer_dtype), np.floating):
        return np.isnan(data) if np.isnan(nodata_value) else data == nodata_value
    return data == nodata_value


@dataclass
class TagComparisonResult:
    layer_name: str
    all_match: bool
    matching_tags: list[str]
    mismatched_tags: dict[str, tuple[str, str]]

    @property
    def num_mismatched(self) -> int:
        return len(self.mismatched_tags)


@dataclass
class ValidDataComparisonResult:
    layer_name: str
    masks_consistent: bool
    percent_nodata_mismatch: float
    nodata_mismatch_pixels: int
    total_pixels: int


@dataclass
class RasterComparisonResult:
    layer_name: str
    is_equal: bool
    max_difference: float
    tolerance_used: float
    percent_mismatch: float
    mismatched_valid_pixels: int
    total_valid_pixels: int


@dataclass
class ProductLayerComparisonResult:
    layer_name: str
    tags_match: bool
    valid_areas_match: bool
    values_match: bool
    tag_result: TagComparisonResult
    valid_area_result: ValidDataComparisonResult
    value_result: RasterComparisonResult

    @property
    def is_equal(self) -> bool:
        return self.tags_match and self.valid_areas_match and self.values_match


@dataclass
class ProductComparisonResult:
    is_equal: bool
    mgrs_matches: bool
    datetime_matches: bool
    layer_results: dict[str, ProductLayerComparisonResult]

    @property
    def failed_layers(self) -> list[str]:
        return [name for name, result in self.layer_results.items() if not result.is_equal]


def _ensure_product_directory(product: 'DistS1ProductDirectory | Path | str') -> 'DistS1ProductDirectory':
    if isinstance(product, DistS1ProductDirectory):
        return product
    return DistS1ProductDirectory.from_product_path(product)


@rasterio_anon_s3_env
def compare_product_layer_tags(
    product1: 'DistS1ProductDirectory | Path | str',
    product2: 'DistS1ProductDirectory | Path | str',
    layer: str = 'GEN-DIST-STATUS',
    tags_to_compare: list[str] | None = None,
    ignore_tags: list[str] | None = None,
) -> TagComparisonResult:
    prod1 = _ensure_product_directory(product1)
    prod2 = _ensure_product_directory(product2)

    if tags_to_compare is None:
        tags_to_compare = PRODUCT_TAGS_FOR_EQUALITY

    if ignore_tags is None:
        ignore_tags = []

    path1 = prod1.layer_path_dict[layer]
    path2 = prod2.layer_path_dict[layer]

    matching_tags = []
    mismatched_tags = {}

    with rasterio.open(str(path1)) as src1, rasterio.open(str(path2)) as src2:
        tags1 = src1.tags()
        tags2 = src2.tags()

        for key in tags_to_compare:
            if key in ignore_tags:
                continue
            if compare_dist_s1_product_tag(key, tags1[key], tags2[key]):
                matching_tags.append(key)
            else:
                mismatched_tags[key] = (tags1[key], tags2[key])

    return TagComparisonResult(
        layer_name=layer,
        all_match=len(mismatched_tags) == 0,
        matching_tags=matching_tags,
        mismatched_tags=mismatched_tags,
    )


@rasterio_anon_s3_env
def compare_valid_areas_in_layer(
    product1: 'DistS1ProductDirectory | Path | str',
    product2: 'DistS1ProductDirectory | Path | str',
    layer: str,
) -> ValidDataComparisonResult:
    prod1 = _ensure_product_directory(product1)
    prod2 = _ensure_product_directory(product2)

    path1 = prod1.layer_path_dict[layer]
    path2 = prod2.layer_path_dict[layer]

    with rasterio.open(str(path1)) as src1, rasterio.open(str(path2)) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)

        nodata_value = TIF_LAYER_NODATA_VALUES[layer]
        layer_dtype = prod1.tif_layer_dtypes[layer]

        nodata_mask1 = _get_nodata_mask(data1, nodata_value, layer_dtype)
        nodata_mask2 = _get_nodata_mask(data2, nodata_value, layer_dtype)

        masks_consistent = np.array_equal(nodata_mask1, nodata_mask2)

        valid_mask1 = ~nodata_mask1
        valid_mask2 = ~nodata_mask2

        one_nodata1 = nodata_mask1 & valid_mask2
        one_nodata2 = nodata_mask2 & valid_mask1
        nodata_mismatch_pixels = int(np.sum(one_nodata1 | one_nodata2))

        total_pixels = data1.size
        percent_nodata_mismatch = (nodata_mismatch_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        return ValidDataComparisonResult(
            layer_name=layer,
            masks_consistent=masks_consistent,
            percent_nodata_mismatch=percent_nodata_mismatch,
            nodata_mismatch_pixels=nodata_mismatch_pixels,
            total_pixels=total_pixels,
        )


@rasterio_anon_s3_env
def compare_product_raster(
    product1: 'DistS1ProductDirectory | Path | str',
    product2: 'DistS1ProductDirectory | Path | str',
    layer: str,
    tolerance: float | None = None,
) -> RasterComparisonResult:
    prod1 = _ensure_product_directory(product1)
    prod2 = _ensure_product_directory(product2)

    path1 = prod1.layer_path_dict[layer]
    path2 = prod2.layer_path_dict[layer]

    with rasterio.open(str(path1)) as src1, rasterio.open(str(path2)) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)

        layer_dtype = prod1.tif_layer_dtypes[layer]
        if tolerance is None:
            is_float = np.issubdtype(np.dtype(layer_dtype), np.floating)
            tolerance = MAX_FLOAT_LAYER_DIFF if is_float else MAX_INT_LAYER_DIFF

        nodata_value = TIF_LAYER_NODATA_VALUES[layer]

        nodata_mask1 = _get_nodata_mask(data1, nodata_value, layer_dtype)
        nodata_mask2 = _get_nodata_mask(data2, nodata_value, layer_dtype)

        valid_mask1 = ~nodata_mask1
        valid_mask2 = ~nodata_mask2
        both_valid = valid_mask1 & valid_mask2

        total_valid_pixels = int(np.sum(both_valid))

        if total_valid_pixels == 0:
            return RasterComparisonResult(
                layer_name=layer,
                is_equal=True,
                max_difference=0.0,
                tolerance_used=tolerance,
                percent_mismatch=0.0,
                mismatched_valid_pixels=0,
                total_valid_pixels=0,
            )

        diff = np.abs(data1[both_valid] - data2[both_valid])
        max_diff = float(np.max(diff))
        mismatched_valid_pixels = int(np.sum(diff > tolerance))
        percent_mismatch = (mismatched_valid_pixels / total_valid_pixels) * 100

        is_equal = max_diff <= tolerance

        return RasterComparisonResult(
            layer_name=layer,
            is_equal=is_equal,
            max_difference=max_diff,
            tolerance_used=tolerance,
            percent_mismatch=percent_mismatch,
            mismatched_valid_pixels=mismatched_valid_pixels,
            total_valid_pixels=total_valid_pixels,
        )


@rasterio_anon_s3_env
def compare_product_layer(
    product1: 'DistS1ProductDirectory | Path | str',
    product2: 'DistS1ProductDirectory | Path | str',
    layer: str,
    tolerance: float | None = None,
    tags_to_compare: list[str] | None = None,
) -> ProductLayerComparisonResult:
    tag_result = compare_product_layer_tags(product1, product2, layer, tags_to_compare)
    valid_area_result = compare_valid_areas_in_layer(product1, product2, layer)
    value_result = compare_product_raster(product1, product2, layer, tolerance)

    return ProductLayerComparisonResult(
        layer_name=layer,
        tags_match=tag_result.all_match,
        valid_areas_match=valid_area_result.masks_consistent,
        values_match=value_result.is_equal,
        tag_result=tag_result,
        valid_area_result=valid_area_result,
        value_result=value_result,
    )


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
        return validate_dist_s1_product_name(product_name)


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
        tokens_self = self.product_name.split('_')
        tokens_other = other.product_name.split('_')

        mgrs_self = tokens_self[3][1:]
        mgrs_other = tokens_other[3][1:]
        mgrs_matches = mgrs_self == mgrs_other

        acq_dt_self = datetime.strptime(tokens_self[4], '%Y%m%dT%H%M%SZ')
        acq_dt_other = datetime.strptime(tokens_other[4], '%Y%m%dT%H%M%SZ')
        datetime_matches = acq_dt_self == acq_dt_other

        layer_results = {layer: compare_product_layer(self, other, layer) for layer in self.layers}

        overall_equal = mgrs_matches and datetime_matches and all(r.is_equal for r in layer_results.values())

        return ProductComparisonResult(
            is_equal=overall_equal,
            mgrs_matches=mgrs_matches,
            datetime_matches=datetime_matches,
            layer_results=layer_results,
        )

    def __eq__(self, other: 'DistS1ProductDirectory') -> bool:
        result = self.compare_products(other)

        if not result.mgrs_matches:
            warn(f'MGRS tile IDs do not match: {self.mgrs_tile_id} != {other.mgrs_tile_id}', UserWarning)
        if not result.datetime_matches:
            warn(f'Acquisition datetimes do not match: {self.acq_datetime} != {other.acq_datetime}', UserWarning)
        for layer_result in result.layer_results.values():
            if not layer_result.tags_match:
                tag_result = layer_result.tag_result
                mismatch_details = ', '.join(
                    f'{k}: {v1!r} != {v2!r}' for k, (v1, v2) in tag_result.mismatched_tags.items()
                )
                warn(
                    f'Layer {layer_result.layer_name}: {tag_result.num_mismatched} tags do not match: '
                    f'{mismatch_details}',
                    UserWarning,
                )
            if not layer_result.valid_areas_match:
                warn(
                    f'Layer {layer_result.layer_name}: valid areas do not match, '
                    f'{layer_result.valid_area_result.percent_nodata_mismatch:.2f}% nodata mismatch',
                    UserWarning,
                )
            if not layer_result.values_match:
                val_result = layer_result.value_result
                warn(
                    f'Layer {layer_result.layer_name}: values do not match, '
                    f'max diff {val_result.max_difference:.6e} > tolerance {val_result.tolerance_used:.6e}, '
                    f'{val_result.percent_mismatch:.2f}% valid pixels mismatched',
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

    def add_prior_product_path(self, prior_product_path: str | Path) -> None:
        if is_s3_path(str(self.dst_dir)):
            raise NotImplementedError(
                'Adding tags to s3 products is not currently supported as it require '
                'writing to the s3-stored product and appropriate permissions.'
            )

        prior_product_str = Path(prior_product_path).name

        for layer in self.layers:
            layer_path = self.layer_path_dict[layer]
            with rasterio.open(str(layer_path), 'r') as src:
                t = src.tags()
                p = src.profile
                X = src.read(1)
                if layer in ['GEN-DIST-STATUS', 'GEN-DIST-STATUS-ACQ']:
                    cmap = src.colormap(1)
                else:
                    cmap = None
            t['prior_dist_s1_product'] = prior_product_str
            serialize_one_2d_ds(X, p, layer_path, colormap=cmap, tags=t, cog=True)
