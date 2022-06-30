from unittest import expectedFailure
import numpy as np
from ineqpy import inequality
import pytest


def test_concentration_0d():
    x = np.array([100])

    obtained = inequality.concentration(income=x)

    assert np.isnan(obtained)

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

def test_theil_1d_series():
    """ Testing theil with no weights. Every value is the same """
    x = np.repeat(5, 10)
    
    obtained = inequality.theil(income=x)
    expected = 0

    np.testing.assert_almost_equal(obtained, expected)

def test_theil_1d_series_2():
    x = np.arange(1, 10)

    obtained = inequality.theil(income=x)
    expected = 0.1473838569435545

    np.testing.assert_almost_equal(obtained, expected)

def test_theil_1d_1_w():
    # TODO check this
    x = np.array([1, 1])
    w = np.array([1, 1])
    obtained = inequality.theil(income=x, weights=w)
    expected = 0
    assert obtained==expected

def test_ratio_equality():
    x = np.array([1, 9])
    w = np.array([9, 1])
    obtained = inequality.top_rest(income=x, weights=w)
    assert obtained == 1.0

def test_ratio_equality_fracc():
    x = np.array([1, 9])
    w = np.array([.9, .1])
    obtained = inequality.top_rest(income=x, weights=w)
    assert obtained == 1.0

def test_ratio_0d():
    x = np.array([100])
    obtained = inequality.top_rest(income=x)

    assert np.isnan(obtained)

def test_ratio_1d():
    x = np.array([57, 63, 81, 79, 88, 42, 3, 77, 89])
    w = np.array([9, 5, 2, 9, 5, 4, 5, 9, 9])
    obtained = inequality.top_rest(income=x, weights=w)
    expected = pytest.approx(0.15323043465128208)
    assert obtained == expected

def test_ratio_2d():
    x = np.array([[57], [63], [81], [79], [88], [42], [3], [77], [89]])
    w = np.array([[9], [5], [2], [9], [5], [4], [5], [9], [9]])
    obtained = inequality.top_rest(income=x, weights=w)
    expected = pytest.approx(0.15323043465128208)
    assert obtained == expected


@pytest.mark.parametrize('n', range(15, 20))
def test_ratio_weighted_eq_unweighted(n):
    # Generating a random list of between 10 and 100 items
    x = np.random.randint(1, 100, n)
    w = np.random.randint(1, 5, n)

    # Weight should be the same as repeating the number multiple times
    xw = []
    for xi, wi in zip(x,w):
        xw += [xi]*wi  # Create a list that contains

    xw = np.array(xw)

    assert len(xw) == np.sum(w)

    weighted = inequality.top_rest(income=x, weights=w)
    unweighted = inequality.top_rest(income=xw)
    assert pytest.approx(weighted) == unweighted

def test_ratio_unweighted():
    x = np.array([
       11, 67, 93, 68, 80, 71,  0, 65, 45, 73, 56, 38, 18, 24, 94, 72, 56,
       37, 26, 34, 49, 30, 30, 31, 10,  0, 77,  6, 64, 75, 56, 79, 46, 87,
       39, 73, 63,  3, 49, 52, 94,  0, 68, 86, 42, 84, 58,  5, 45, 62, 49,
       97, 77, 94, 66, 84, 42, 39,  7, 24, 65, 52, 59, 52, 38, 27, 85, 43,
       26,  6, 93, 24, 48, 42, 50, 58, 89, 79, 94, 50,  2, 46, 82, 98, 69,
       9,  50, 33, 86, 77, 25, 39, 61, 78, 47, 29, 43, 20, 56, 35])
    obtained = inequality.top_rest(x)
    expected = 0.22203712517848642
    assert pytest.approx(obtained) == expected


def test_hoover_index_series():
    """ Testing hoover with no weights (default all ones) """
    x = np.arange(10)
    obtained = inequality.hoover(x)
    expected = 4.0

    np.testing.assert_almost_equal(obtained, expected)

def test_hoover_index():
    x = np.arange(10)
    w = np.ones(10)
    obtained = inequality.hoover(x, w)
    expected = 4
    np.testing.assert_almost_equal(obtained, expected)
