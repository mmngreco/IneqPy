from ineqpy import _statistics
import scipy.stats as sc
import numpy as np
import pandas as pd
import numpy.testing as nptest
import pandas.testing as pdtest
import unittest


def data_weighted():
    N = np.random.randint(20, 100)
    x = np.random.randint(0, 1000, N)
    w = np.random.randint(1, 9, N)
    repeated_x = np.array([])
    repeated_w = np.array([])

    for xi, wi in zip(x, w):
        repeated_x = np.append(repeated_x, np.repeat(xi, wi))
        repeated_w = np.append(repeated_w, np.ones(wi))
    return (x, w), (repeated_x, repeated_w)

def data():
    N = np.random.randint(20,1000,1)
    x = np.random.randn(N, 4)
    w = abs(np.random.randn(N))
    return x, w

class TestStatistics(unittest.TestCase):

    def test_statistics(self):

        for i in range(100):
            (x, w), (repeated_x, repeated_w) = data_weighted()

            with self.subTest(name='mean', i=i):
                real = np.mean(repeated_x)
                obtained = _statistics.mean(x, w)
                nptest.assert_almost_equal(obtained, real)

            with self.subTest(name='variance', i=i):
                real = np.var(repeated_x)
                obtained = _statistics.var(x, w)
                # assert
                nptest.assert_almost_equal(obtained, real)

            with self.subTest(name='kurtosis', i=i):
                real = sc.kurtosis(repeated_x) + 3
                obtained = _statistics.kurt(x, w)
                # assert
                nptest.assert_almost_equal(obtained, real)

            with self.subTest(name='skewness', i=i):
                real = sc.skew(repeated_x)
                obtained = _statistics.skew(x, w)
                nptest.assert_almost_equal(obtained, real)

            with self.subTest(name='coef_variation', i=i):
                real = np.var(repeated_x) ** 0.5 / abs(np.mean(repeated_x))
                obtained = _statistics.coef_variation(x,w)
                nptest.assert_almost_equal(obtained, real)

            with self.subTest(name='quantile', i=i):
                q = 75
                x,w = _statistics.misc._sort_values(x,w)
                repeated_x, repeated_w = _statistics.misc._sort_values(repeated_x, repeated_w)
                real = pd.Series(repeated_x).quantile(q/100, interpolation='midpoint')
                q_np = np.percentile(repeated_x, q, interpolation='midpoint')
                obtained = _statistics.quantile(x,w, q=q/100)
                nptest.assert_almost_equal(obtained, real)

if __name__ == '__main__':
    unittest.main()