from ineqpy import statistics
from ineqpy import utils
import scipy.stats as sc
import numpy as np
import numpy.testing as nptest
import pytest


def gen_inputs(n_tuples=100):
    for _ in range(n_tuples):
        (x, w) = utils.generate_data_to_test((3, 7))

        # NOBUG: _ is `repeated_w` which is a vector of ones.
        repeated_x, _ = utils.repeat_data_from_weighted(x, w)
        yield x, w, repeated_x


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_mean(x, w, r_x):
    real = np.mean(r_x)
    obtained = statistics.mean(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_variance(x, w, r_x):
    real = np.var(r_x)
    obtained = statistics.var(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_kurt(x, w, r_x):
    real = sc.kurtosis(r_x) + 3
    obtained = statistics.kurt(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_skew(x, w, r_x):
    real = sc.skew(r_x)
    obtained = statistics.skew(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_coef_variation(x, w, r_x):
    real = np.var(r_x) ** 0.5 / abs(np.mean(r_x))
    obtained = statistics.coef_variation(x, w)
    nptest.assert_almost_equal(obtained, real)


@pytest.mark.parametrize("x,w,r_x", gen_inputs())
def test_percentile(x, w, r_x):
    p = 50
    real = np.percentile(r_x, p, interpolation="lower")
    obtained = statistics.percentile(x, w, p=p)
    nptest.assert_almost_equal(
        obtained, real, err_msg=msg(real, obtained, r_x, x, w)
    )


def msg(real, obtained, r_x, x, w):
    if abs(real - obtained) > 1e-6:
        return "\nr_x = {}\nx = {}\nw = {}".format(str(r_x), str(x), str(w))
