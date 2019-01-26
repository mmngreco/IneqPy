import numpy as np
import ineqpy


def test_api():
    # todo improve this test.
    # only checks that all methods works.
    svy = ineqpy.api.Survey
    data = np.random.randint(0, 100, (int(1e3), 3))
    w = np.random.randint(1, 10, int(1e3))
    data = np.c_[data, w]
    columns = list("abcw")

    df = svy(data=data, columns=columns, weights="w")
    df.weights
    df.mean("a")
    df.var("a")
    df.skew("a")
    df.kurt("a")
    df.gini("a")
    df.atkinson("a")
    df.theil("a")
    df.percentile("a")
