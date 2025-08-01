from dist_s1.constants import TIF_LAYERS, TIF_LAYER_DTYPES, TIF_LAYER_NODATA_VALUES


def test_tif_layer_nodata_values() -> None:
    layer_nodata_vals = set(TIF_LAYER_NODATA_VALUES.keys())
    layer_dtypes = set(TIF_LAYER_DTYPES.keys())
    layers = set(TIF_LAYERS)
    assert layer_nodata_vals == layer_dtypes == layers
