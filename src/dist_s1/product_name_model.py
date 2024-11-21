from datetime import datetime

from pydantic import BaseModel

from .constants import PRODUCT_VERSION


class ProductNameModel(BaseModel):
    mgrs_tile_id: str
    acq_date_time: datetime
    processing_date_time: datetime

    def __str__(self):
        tokens = [
            'OPERA',
            'L3',
            'DIST-ALERT-S1',
            f'T{self.mgrs_tile_id}',
            self.acq_date_time.strftime('%Y%m%dT%h%m%sZ'),
            self.processing_date_time.strftime('%Y%m%dT%h%m%sZ'),
            'S1',
            '30',
            f'v{PRODUCT_VERSION}',
        ]
        return '_'.join(tokens)

    def get_product_name(self):
        return f'{self}'

    def get_tif_layer_name(self, layer_name: str):
        return f'{self}_{layer_name}.tif'
