from datetime import datetime
from pathlib import Path
from warnings import warn

import rasterio
from pydantic import BaseModel, field_validator


PRODUCT_VERSION = '0.0.1'
LAYERS = ['DIST-STATUS', 'DIST-STATUS-ACQ', 'GEN-METRIC', 'DATE-FIRST', 'DATE-LATEST', 'N-OBS', 'N-DIST']
LAYER_DTYPES = {
    'DIST-STATUS': 'uint8',
    'DIST-STATUS-ACQ': 'uint8',
    'GEN-METRIC': 'float32',
    'DATE-FIRST': 'uint32',
    'DATE-LATEST': 'uint32',
    'N-OBS': 'uint32',
    'N-DIST': 'uint32',
}
EXPECTED_FORMAT_STRING = (
    'OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{PRODUCT_VERSION}'
)


class ProductNameData(BaseModel):
    mgrs_tile_id: str
    acq_date_time: datetime
    processing_date_time: datetime

    def __str__(self):
        tokens = [
            'OPERA',
            'L3',
            'DIST-ALERT-S1',
            f'T{self.mgrs_tile_id}',
            self.acq_date_time.strftime('%Y%m%dT%H%M%SZ'),
            self.processing_date_time.strftime('%Y%m%dT%H%M%SZ'),
            'S1',
            '30',
            f'v{PRODUCT_VERSION}',
        ]
        return '_'.join(tokens)

    def name(self):
        return f'{self}'

    @classmethod
    def validate_product_name(cls, product_name: str) -> bool:
        """
        Validates if a string matches the OPERA L3 DIST-ALERT-S1 product name format.

        Expected format:
        OPERA_L3_DIST-ALERT-S1_T{mgrs_tile_id}_{acq_datetime}_{proc_datetime}_S1_30_v{version}
        """
        try:
            tokens = product_name.split('_')

            # Check if we have the correct number of tokens first
            if len(tokens) != 9:
                return False

            conditions = [
                tokens[0] != 'OPERA',
                tokens[1] != 'L3',
                tokens[2] != 'DIST-ALERT-S1',
                not tokens[3].startswith('T'),  # MGRS tile ID
                tokens[6] != 'S1',
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

        except (ValueError, IndexError):
            return False


class ProductDirectoryData(BaseModel):
    product_name: str
    dst_dir: Path | str
    layers: list[str] = LAYERS

    @property
    def path(self) -> Path:
        path = self.dst_dir / self.product_name
        path = path.relative_to(Path.cwd())
        return path

    @field_validator('product_name')
    @classmethod
    def validate_product_name(cls, product_name: str) -> str:
        if not ProductNameData.validate_product_name(product_name):
            raise ValueError(f'Invalid product name: {product_name}; should match: {EXPECTED_FORMAT_STRING}')
        return product_name

    @field_validator('dst_dir')
    @classmethod
    def validate_product_directory(cls, dst_dir: Path | str, values: dict) -> Path:
        if isinstance(dst_dir, str):
            dst_dir = Path(dst_dir)
        product_dir = dst_dir / values.data['product_name']
        if product_dir.exists() and not product_dir.is_dir():
            raise ValueError(f'Path {product_dir} exists but is not a directory')
        if not product_dir.exists():
            product_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir

    @property
    def layer_path_dict(self) -> dict[str, Path]:
        return {layer: self.path / f'{self.product_name}_{layer}.tif' for layer in self.layers}

    def validate_layer_paths(self) -> bool:
        failed_layers = []
        for layer, path in self.layer_path_dict.items():
            if not path.exists():
                warn(f'Layer {layer} does not exist at path: {path}', UserWarning)
                failed_layers.append(layer)
        return len(failed_layers) == 0

    def validate_layer_dtypes(self) -> bool:
        failed_layers = []
        for layer, path in self.layer_path_dict.items():
            with rasterio.open(path) as src:
                if src.dtypes[0] != LAYER_DTYPES[layer]:
                    warn(
                        f'Layer {layer} has incorrect dtype: {src.dtypes[0]}; should be: {LAYER_DTYPES[layer]}',
                        UserWarning,
                    )
                    failed_layers.append(layer)
        return len(failed_layers) == 0
