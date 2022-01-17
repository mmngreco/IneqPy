"""Low level desciptive statistics.

References
----------
1. http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
2. https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
   #Weighted_sample_variance
3. https://en.wikipedia.org/wiki/Algorithms%5Ffor%5Fcalculating%5Fvariance
   #Weighted_incremental_algorithm
"""

import numpy as np
from numba import guvectorize

from . import utils


def c_moment(variable=None, weights=None, order=2, param=None, ddof=0):
    """Calculate central momment.

    Calculate the central moment of `x` with respect to `param` of order `n`,
    given the weights `w`.

    Parameters
    ----------
    variable : 1d-array
        Variable
    weights : 1d-array
        Weights
    order : int, optional
        Moment order, 2 by default (variance)
    param : int or array, optional
        Parameter for which the moment is calculated, the default is None,
        implies use the mean.
    ddof : int, optional
        Degree of freedom, zero by default.

    Returns
    -------
    central_moment : float

    Notes
    -----
    - The cmoment of order 1 is 0
    - The cmoment of order 2 is the variance.

    Source : https://en.wikipedia.org/wiki/Moment_(mathematics)

    Todo
    ----
    Implement : https://en.wikipedia.org/wiki/L-moment#cite_note-wang:96-6
    """
    # return np.sum((x-c)^n*counts) / np.sum(counts)
    variable = variable.copy()
    weights = utils.not_empty_weights(weights, like=variable)

    if param is None:
        param = mean(variable=variable, weights=weights)
    elif not isinstance(param, (np.ndarray, int, float)):
        raise NotImplementedError

    return np.sum((variable - param) ** order * weights) / (
        np.sum(weights) - ddof
    )


def percentile(
    variable, weights, percentile=50, interpolation="lower"
) -> float:
    """Calculate the percentile.

    Parameters
    ----------
    variable : str or array
    weights :  str or array
    percentile : int or list
        Percentile level, if pass 50 we get the median.
    interpolation : {'lower', 'higher', 'midpoint'}, optional
        Select interpolation method.

    Returns
    -------
    percentile : float
    """
    sorted_idx = np.argsort(variable)
    cum_weights = np.cumsum(weights[sorted_idx])
    lower_percentile_idx = np.searchsorted(
        cum_weights, (percentile / 100.0) * cum_weights[-1]
    )

    if interpolation == "midpoint":
        res = np.interp(
            lower_percentile_idx + 0.5,
            np.arange(len(variable)),
            variable[sorted_idx],
        )
    elif interpolation == "lower":
        res = variable[sorted_idx[lower_percentile_idx]]
    elif interpolation == "higher":
        res = variable[sorted_idx[lower_percentile_idx + 1]]
    else:
        raise NotImplementedError

    return float(res)


def std_moment(variable=None, weights=None, param=None, order=3, ddof=0):
    """Calculate the standarized moment.

    Calculate the standarized moment of order `c` for the variable` x` with
    respect to `c`.

    Parameters
    ----------
    variable : 1d-array
       Random Variable
    weights : 1d-array, optional
       Weights or probability
    order : int, optional
       Order of Moment, three by default
    param : int or float or array, optional
       Central trend, default is the mean.
    ddof : int, optional
        Degree of freedom.

    Returns
    -------
    std_moment : float
       Returns the standardized `n` order moment.

    References
    ----------
    - https://en.wikipedia.org/wiki/Moment_(mathematics)
      #Significance_of_the_moments
    - https://en.wikipedia.org/wiki/Standardized_moment

    Todo
    ----
    It is the general case of the raw and central moments. Review
    implementation.

    """
    if param is None:
        param = mean(variable=variable, weights=weights)

    res = c_moment(
        variable=variable, weights=weights, order=order, param=param, ddof=ddof
    )
    res /= var(variable=variable, weights=weights, ddof=ddof) ** (order / 2)
    return res


def mean(variable=None, weights=None):
    """Calculate the mean of `variable` given `weights`.

    Parameters
    ----------
    variable : array-like or str
        Variable on which the mean is estimated.
    weights : array-like or str
        Weights of the `x` variable.

    Returns
    -------
    mean : array-like or float
    """
    # if pass a DataFrame separate variables.
    variable = variable.copy()
    weights = utils.not_empty_weights(weights, like=variable)
    variable, weights = utils._clean_nans_values(variable, weights)
    return np.average(a=variable, weights=weights, axis=0)


def var(variable=None, weights=None, ddof=0):
    """Calculate the population variance of ``variable`` given `weights`.

    Parameters
    ----------
    variable : 1d-array or pd.Series or pd.DataFrame
        Variable on which the quasivariation is estimated
    weights : 1d-array or pd.Series or pd.DataFrame
        Weights of the `variable`.

    Returns
    -------
    variance : 1d-array or pd.Series or float
        Estimation of quasivariance of `variable`

    References
    ----------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)

    Notes
    -----
    If stratificated sample must pass with groupby each strata.
    """
    return c_moment(variable=variable, weights=weights, order=2, ddof=ddof)


def coef_variation(variable=None, weights=None):
    """Calculate the coefficient of variation.

    Calculate the coefficient of variation of a `variable` given weights. The
    coefficient of variation is the square root of the variance of the incomes
    divided by the mean income. It has the advantages of being mathematically
    tractable and is subgroup decomposable, but is not bounded from above.

    Parameters
    ----------
    variable : array-like or str
    weights : array-like or str

    Returns
    -------
    coefficient_variation : float

    References
    ----------
    Coefficient of variation. (2017, May 5). In Wikipedia, The Free
    Encyclopedia. Retrieved 15:03, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Coefficient_of_variation
    """
    # todo complete docstring
    return var(variable=variable, weights=weights) ** 0.5 / abs(
        mean(variable=variable, weights=weights)
    )


def kurt(variable=None, weights=None):
    """Calculate the asymmetry coefficient.

    Parameters
    ----------
    variable : 1d-array
    weights : 1d-array

    Returns
    -------
    kurt : float
        Kurtosis coefficient.

    References
    ----------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)

    Notes
    -----
    It is an alias of the standardized fourth-order moment.
    """
    return std_moment(variable=variable, weights=weights, order=4)


def skew(variable=None, weights=None):
    """Return the asymmetry coefficient of a sample.

    Parameters
    ----------
    variable : array-like, str
    weights : array-like, str

    Returns
    -------
    skew : float

    References
    ----------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)

    Notes
    -----
    It is an alias of the standardized third-order moment.

    """
    return std_moment(variable=variable, weights=weights, order=3)


@guvectorize(
    "float64[:], float64[:], int64, float64[:]",
    "(n),(n),()->()",
    nopython=True,
    cache=True,
)
def wvar(x, w, kind, out):
    """Calculate weighted variance of X.

    Calculates the weighted variance of x according to a kind of weights.

    Parameters
    ----------
    x : np.ndarray
        Main variable.
    w : np.ndarray
        Weigths.
    kind : int
        Has three modes to calculate de variance, you can control that with
        this argument, the values and the output are the next:
        * 1. population variance
        * 2. sample frequency variance
        * 3. sample reliability variance.
    out : np.ndarray

    Returns
    -------
    weighted_variance : float

    References
    ----------
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #Weighted_incremental_algorithm
    """
    wSum = wSum2 = mean = S = 0

    for i in range(len(x)):  # Alternatively "for x, w in zip(data, weights):"
        wSum = wSum + w[i]
        wSum2 = wSum2 + w[i] * w[i]
        meanOld = mean
        mean = meanOld + (w[i] / wSum) * (x[i] - meanOld)
        S = S + w[i] * (x[i] - meanOld) * (x[i] - mean)

    if kind == 1:
        # population_variance
        out[0] = S / wSum
    elif kind == 2:
        # Bessel's correction for weighted samples
        # Frequency weights
        # sample_frequency_variance
        out[0] = S / (wSum - 1)
    elif kind == 3:
        # Reliability weights
        # sample_reliability_variance
        out[0] = S / (wSum - wSum2 / wSum)


@guvectorize(
    "float64[:], float64[:], float64[:], int64, float64[:]",
    "(n),(n),(n),()->()",
    nopython=True,
    cache=True,
)
def wcov(x, y, w, kind, out):
    """Compute weighted covariance between x and y.

    Compute the weighted covariance between two variables, we can chose which
    kind of covariance returns.

    Parameters
    ----------
    x : np.array
        Main variable.
    y : np.array
        Second variable.
    w : np.array
        Weights.
    kind : int
        Kind of weighted covariance is returned:
            1 : population variance
            2 : sample frequency variance
            3 : sample reliability variance.
    out : np.array

    Returns
    -------
    weighted_covariance = float

    References
    ----------
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    """
    meanx = meany = 0
    wsum = wsum2 = 0
    C = 0
    for i in range(len(x)):
        wsum += w[i]
        wsum2 += w[i] * w[i]
        dx = x[i] - meanx
        meanx += (w[i] / wsum) * dx
        meany += (w[i] / wsum) * (y[i] - meany)
        C += w[i] * dx * (y[i] - meany)

    if kind == 1:
        # population_covar
        out[0] = C / wsum


@guvectorize(
    "float64[:], float64[:], float64[:]",
    "(n),(n)->()",
    nopython=True,
    cache=True,
)
def online_kurtosis(x, w, out):
    """Online kurtosis."""
    n = 0
    mean = 0
    M2 = 0
    M3 = 0
    M4 = 0

    for i in range(len(x)):
        n1 = w[i]
        n = n + w[i]
        delta = x[i] - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        mean = mean + w[i] * delta_n / n
        M4 = (
            M4
            + term1 * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * M2
            - 4 * delta_n * M3
        )
        M3 = M3 + term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 = M2 + term1

    out[0] = (n * M4) / (M2 * M2) - 3


@guvectorize(
    "float64[:], float64[:], int64, float64[:]",
    "(n),(n),()->()",
    nopython=True,
    cache=True,
)
def Mk(x, w, k, out):
    """Calculate Mk."""
    w_sum = wx_sum = 0

    for i in range(len(x)):
        wx_sum += w[i] * (x[i] ** k)
        w_sum += w[i]

    out[0] = wx_sum / w_sum
