import shutil
from pathlib import Path

import pytest
import rasterio

from dist_s1.rio_tools import get_mgrs_profile
from dist_s1.water_mask import water_mask_control_flow


def test_good_water_mask_path(test_dir: Path, good_water_mask_path_for_17SLR: Path) -> None:
    """Apply the water mask control flow to a water mask that is buffered by .25 degrees around the MGRS tile."""
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    water_mask_path = water_mask_control_flow(
        water_mask_path=good_water_mask_path_for_17SLR,
        mgrs_tile_id='17SLR',
        apply_water_mask=True,
        dst_dir=tmp_dir,
    )
    assert water_mask_path.exists()
    assert water_mask_path.is_file()
    assert water_mask_path.suffix == '.tif'
    assert water_mask_path.name == '17SLR_water_mask.tif'

    mgrs_profile = get_mgrs_profile('17SLR')

    with rasterio.open(water_mask_path) as src:
        water_mask_profile = src.profile

    assert water_mask_profile['count'] == 1
    assert str(water_mask_profile['dtype']) == 'uint8'
    assert water_mask_profile['crs'] == mgrs_profile['crs']
    assert water_mask_profile['transform'] == mgrs_profile['transform']
    assert water_mask_profile['width'] == mgrs_profile['width']
    assert water_mask_profile['height'] == mgrs_profile['height']

    shutil.rmtree(tmp_dir)


def test_bad_water_mask_path(test_dir: Path, bad_water_mask_path_for_17SLR: Path) -> None:
    """Apply the water mask control flow to a water mask that is eroded by -.25 degrees around the MGRS tile."""
    tmp_dir = test_dir / 'tmp'

    with pytest.raises(ValueError):
        water_mask_control_flow(
            water_mask_path=bad_water_mask_path_for_17SLR,
            mgrs_tile_id='17SLR',
            apply_water_mask=True,
            dst_dir=tmp_dir,
        )


def test_bad_file_path(test_dir: Path) -> None:
    """Apply the water mask control flow to a file that does not exist."""
    tmp_dir = test_dir / 'tmp'

    with pytest.raises(FileNotFoundError):
        water_mask_control_flow(
            water_mask_path=tmp_dir / 'bad_file.tif',
            mgrs_tile_id='17SLR',
            apply_water_mask=True,
            dst_dir=tmp_dir,
        )


def test_antimeridian_water_mask(test_dir: Path, antimeridian_water_mask_path_for_01VCK: Path) -> None:
    """Apply the water mask control flow to a water mask file that crosses the antimeridian."""
    tmp_dir = test_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    water_mask_path = water_mask_control_flow(
        water_mask_path=antimeridian_water_mask_path_for_01VCK,
        mgrs_tile_id='01VCK',
        apply_water_mask=True,
        dst_dir=tmp_dir,
    )
    assert water_mask_path.exists()
    assert water_mask_path.is_file()
    assert water_mask_path.suffix == '.tif'
    assert water_mask_path.name == '01VCK_water_mask.tif'
