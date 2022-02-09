import numpy as np
from ineqpy import inequality


def test_gini_2d():
    x = np.array([[57], [63], [81], [79], [88], [57], [42], [3], [77], [89]])
    w = np.array([[2], [5], [2], [9], [5], [7], [4], [5], [9], [9]])
    obtained = inequality.gini(income=x, weights=w)
    expected = 0.2134389018024818
    assert obtained==expected


def test_gini_1d():
    x = np.array([57, 63, 81, 79, 88, 57, 42, 3, 77, 89])
    w = np.array([2, 5, 2, 9, 5, 7, 4, 5, 9, 9])
    obtained = inequality.gini(income=x, weights=w)
    expected = 0.2134389018024818
    assert obtained==expected


def test_gini_1d_0_w():
    x = np.array([2,       2])
    w = np.array([1000000, 1])
    obtained = inequality.gini(income=x, weights=w)
    expected = 0
    assert obtained==expected


def test_gini_1d_0_series():
    x = np.array([2, 2])
    # w = np.array([1000000, 1])
    obtained = inequality.gini(income=x)
    expected = 0
    assert obtained==expected


def test_gini_1d_1_series():
    x = np.array([0, 1])
    # w = np.array([1000000, 1])
    obtained = inequality.gini(income=x)
    expected = 1
    assert obtained==expected


def test_gini_1d_1_w():
    x = np.array([0, 1])
    w = np.array([1, 1])
    obtained = inequality.gini(income=x, weights=w)
    expected = 1
    assert obtained==expected


def test_atkinson_2d():
    x = np.array([[57], [63], [81], [79], [88], [57], [42], [3], [77], [89]])
    w = np.array([[2], [5], [2], [9], [5], [7], [4], [5], [9], [9]])
    obtained = inequality.atkinson(income=x, weights=w)
    expected = 0.06537929778911322
    assert obtained==expected


def test_atkinson_1d():
    x = np.array([57, 63, 81, 79, 88, 57, 42, 3, 77, 89])
    w = np.array([2, 5, 2, 9, 5, 7, 4, 5, 9, 9])
    obtained = inequality.atkinson(income=x, weights=w)
    expected = 0.06537929778911322
    assert obtained==expected


def test_atkinson_1d_1_w():
    x = np.array([1, 1])
    w = np.array([1, 1])
    obtained = inequality.atkinson(income=x, weights=w)
    expected = 0
    assert obtained==expected


def test_theil_1d_1_w():
    # TODO check this
    x = np.array([1, 1])
    w = np.array([1, 1])
    obtained = inequality.theil(income=x, weights=w)
    expected = 0
    assert obtained==expected

def test_ratio_1d():
    x = np.array([57, 63, 81, 79, 88, 57, 42, 3, 77, 89])
    w = np.array([2, 5, 2, 9, 5, 7, 4, 5, 9, 9])
    obtained = inequality.ratio_top_rest(income=x, weights=w)
    expected = 0.2654955253563142
    assert obtained == expected

def test_ratio_2d():
    x = np.array([[57], [63], [81], [79], [88], [57], [42], [3], [77], [89]])
    w = np.array([[2], [5], [2], [9], [5], [7], [4], [5], [9], [9]])
    obtained = inequality.ratio_top_rest(income=x, weights=w)
    expected = 0.2654955253563142
    assert obtained == expected

def test_ratio_1d_0_w():
    x = np.array([2,       2])
    w = np.array([1000000, 1])
    obtained = inequality.ratio_top_rest(income=x, weights=w)
    expected = 2000000 / 2
    assert obtained == expected


def test_ratio_1d_0_series():
    x = np.array([2, 2])
    # w = np.array([1, 1])
    obtained = inequality.ratio_top_rest(income=x)
    expected = 1
    assert obtained == expected
