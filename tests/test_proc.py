import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dist_s1.constants import DISTLABEL2VAL
from dist_s1.processing import aggregate_disturbance_over_lookbacks, aggregate_disturbance_over_time


def label_array_in_center(arr: np.ndarray, center_size: int, label: int = 1) -> np.ndarray:
    c = center_size
    m, n = arr.shape
    sx = (n - c) // 2
    ex = sx + c
    sy = (m - c) // 2
    ey = sy + c

    arr[sy:ey, sx:ex] = label
    return arr


@pytest.mark.parametrize('n_looks', [1, 2, 3])
def test_aggregate_disturbance_over_time_consistent_disturbances(n_looks: int) -> None:
    # Test where all the data in the list is the same
    sample_dist_data = [np.zeros((10, 10)) for _ in range(3)]
    sample_dist_data = [label_array_in_center(X, 4, label=1) for X in sample_dist_data]
    sample_dist_data = [label_array_in_center(X, 2, label=2) for X in sample_dist_data]

    X_agg_expected = np.zeros((10, 10))
    if n_looks == 1:
        high_conf_label = DISTLABEL2VAL['first_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['first_moderate_conf_disturbance']
    elif n_looks == 2:
        high_conf_label = DISTLABEL2VAL['provisional_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['provisional_moderate_conf_disturbance']
    elif n_looks == 3:
        high_conf_label = DISTLABEL2VAL['confirmed_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['confirmed_moderate_conf_disturbance']

    X_agg_expected = label_array_in_center(X_agg_expected, 4, label=moderate_conf_label)
    X_agg_expected = label_array_in_center(X_agg_expected, 2, label=high_conf_label)

    X_agg = aggregate_disturbance_over_time(sample_dist_data[:n_looks])
    assert_array_equal(X_agg, X_agg_expected)


@pytest.mark.parametrize('n_looks', [1, 2, 3])
def test_aggregate_disturbance_over_time_with_min(n_looks: int) -> None:
    # Test where all the data in the list is the same
    sample_dist_data = [np.zeros((10, 10)) for _ in range(3)]
    sample_dist_data = [label_array_in_center(X, 4, label=1) for X in sample_dist_data]
    # The min will ignore and label center as 1
    sample_dist_data[0] = label_array_in_center(sample_dist_data[0], 2, label=2)

    X_agg_expected = np.zeros((10, 10))
    X_agg_expected = np.zeros((10, 10))
    if n_looks == 1:
        high_conf_label = DISTLABEL2VAL['first_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['first_moderate_conf_disturbance']
    elif n_looks == 2:
        moderate_conf_label = DISTLABEL2VAL['provisional_moderate_conf_disturbance']
    elif n_looks == 3:
        moderate_conf_label = DISTLABEL2VAL['confirmed_moderate_conf_disturbance']
    if n_looks > 1:
        X_agg_expected = label_array_in_center(X_agg_expected, 4, label=moderate_conf_label)
    else:
        X_agg_expected = label_array_in_center(X_agg_expected, 4, label=moderate_conf_label)
        X_agg_expected = label_array_in_center(X_agg_expected, 2, label=high_conf_label)

    X_agg = aggregate_disturbance_over_time(sample_dist_data[:n_looks])
    assert_array_equal(X_agg, X_agg_expected)


@pytest.mark.parametrize('n_looks', [2, 3])
def test_aggregate_disturbance_over_time_with_nodata(n_looks: int) -> None:
    # Test where all the data in the list is the same
    sample_dist_data = [np.zeros((10, 10)) for _ in range(3)]
    sample_dist_data = [label_array_in_center(X, 4, label=1) for X in sample_dist_data]
    sample_dist_data = [label_array_in_center(X, 2, label=2) for X in sample_dist_data]
    # Diagonal of nodata in the first array
    for k in range(10):
        sample_dist_data[0][k, k] = 255

    X_agg_expected = np.zeros((10, 10))
    if n_looks == 1:
        high_conf_label = DISTLABEL2VAL['first_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['first_moderate_conf_disturbance']
    elif n_looks == 2:
        high_conf_label = DISTLABEL2VAL['provisional_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['provisional_moderate_conf_disturbance']
    elif n_looks == 3:
        high_conf_label = DISTLABEL2VAL['confirmed_high_conf_disturbance']
        moderate_conf_label = DISTLABEL2VAL['confirmed_moderate_conf_disturbance']

    X_agg_expected = label_array_in_center(X_agg_expected, 4, label=moderate_conf_label)
    X_agg_expected = label_array_in_center(X_agg_expected, 2, label=high_conf_label)
    for k in range(10):
        X_agg_expected[k, k] = 255

    X_agg = aggregate_disturbance_over_time(sample_dist_data[:n_looks])
    assert_array_equal(X_agg, X_agg_expected)


def test_aggregate_disturbance_over_lookbacks_1() -> None:
    """Testing the prioritization of higher indices over lower indices."""
    X_delta_l = [np.zeros((10, 10)) for _ in range(3)]
    X_delta_l[0] = label_array_in_center(X_delta_l[0], 4, label=1)
    X_delta_l[1] = label_array_in_center(X_delta_l[1], 3, label=2)
    X_delta_l[2] = label_array_in_center(X_delta_l[2], 2, label=3)
    X_agg = aggregate_disturbance_over_lookbacks(X_delta_l)

    X_agg_expected = np.zeros((10, 10))
    X_agg_expected = label_array_in_center(X_agg_expected, 4, label=1)
    X_agg_expected = label_array_in_center(X_agg_expected, 3, label=2)
    X_agg_expected = label_array_in_center(X_agg_expected, 2, label=3)

    assert_array_equal(X_agg, X_agg_expected)


def test_aggregate_disturbance_over_lookbacks_2() -> None:
    """Testing the prioritization of higher indices over lower indices."""
    X_delta_l = [np.zeros((10, 10)) for _ in range(3)]
    X_delta_l[0] = label_array_in_center(X_delta_l[0], 2, label=1)
    X_delta_l[1] = label_array_in_center(X_delta_l[1], 3, label=2)
    X_delta_l[2] = label_array_in_center(X_delta_l[2], 4, label=3)
    X_delta_l[1][0, 0] = 10

    X_agg_expected = np.zeros((10, 10))
    X_agg_expected = label_array_in_center(X_agg_expected, 4, label=3)
    X_agg_expected[0, 0] = 10

    X_agg = aggregate_disturbance_over_lookbacks(X_delta_l)
    assert_array_equal(X_agg, X_agg_expected)
