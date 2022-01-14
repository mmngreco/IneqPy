from ineqpy import _statistics
from ineqpy import utils
import scipy.stats as sc
import numpy as np
import numpy.testing as nptest
import pytest


def gen_inputs(n_tuples=100):
    for _ in range(n_tuples):
        (x, w) = utils.generate_data_to_test((3, 7))
        repeated_x, repeated_w = utils.repeat_data_from_weighted(x, w)
        # repeated_w is a vector of ones.
        yield x, w, repeated_x


INPUTS = gen_inputs()


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_mean(x, w, r_x):
    real = np.mean(r_x)
    obtained = _statistics.mean(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_variance(x, w, r_x):
    real = np.var(r_x)
    obtained = _statistics.var(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_kurt(x, w, r_x):
    real = sc.kurtosis(r_x) + 3
    obtained = _statistics.kurt(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_skew(x, w, r_x):
    real = sc.skew(r_x)
    obtained = _statistics.skew(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_coef_variation(x, w, r_x):
    real = np.var(r_x) ** 0.5 / abs(np.mean(r_x))
    obtained = _statistics.coef_variation(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", INPUTS)
def test_percentile(x, w, r_x):
    p = 50
    real = np.percentile(r_x, p, interpolation="lower")
    obtained = _statistics.percentile(x, w, percentile=p)
    mssg = msg_assert(real, obtained, r_x, x, w)
    nptest.assert_almost_equal(obtained, real, mssg)


def msg_assert(real, obtained, r_x, x, w):
    if abs(real - obtained) > 1e-6:
        return "\nr_x = {}\nx = {}\nw = {}".format(str(r_x), str(x), str(w))
