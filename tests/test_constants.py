from dist_s1.constants import DISTVAL2LABEL, TIF_LAYERS, TIF_LAYER_DTYPES, TIF_LAYER_NODATA_VALUES


def test_tif_layer_nodata_values() -> None:
    keys_nodata_values = set(TIF_LAYER_NODATA_VALUES.keys())
    keys_dtypes = set(TIF_LAYER_DTYPES.keys())
    label_keys = set(DISTVAL2LABEL.keys())
    layers = set(TIF_LAYERS)
    assert keys_nodata_values == keys_dtypes == label_keys == layers
