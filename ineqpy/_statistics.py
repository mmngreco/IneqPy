import numpy as np
import pandas as pd
from . import utils


def c_moment(variable=None, weights=None, order=2, param=None, ddof=0):
    """Calculate the central moment of `x` with respect to `param` of order `n`,
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

    TODO
    ----

    Implement: https://en.wikipedia.org/wiki/L-moment#cite_note-wang:96-6

    """
    # return np.sum((x-c)^n*counts) / np.sum(counts)
    variable = variable.copy()
    weights = utils._check_weights(weights, as_of=variable)

    if param is None:
        param = mean(variable=variable, weights=weights)
    elif not isinstance(param, (np.ndarray, int, float)):
        raise NotImplementedError

    return np.sum((variable - param) ** order * weights) / \
           (np.sum(weights) - ddof)


# def quantile(variable=None, weights=None, q=0.5, interpolate=True):
#     """Calculate the value of a quantile given a variable and his weights.
#
#     Parameters
#     ----------
#     variable : str or array
#     weights :  str or array
#     q : float
#         Quantile level, if pass 0.5 means median.
#     interpolate : bool
#
#     Returns
#     -------
#     quantile : float or pd.Series
#
#     """
#     # Fixme : Doesn't work properly
#
#     w = utils._check_weights(weights, as_of=variable)
#     x, w = utils._sort_values(variable, w)
#     cum_weights = weights.cumsum(0)# - 0.5 * weights
#     #cum_weights -= cum_weights[0]
#     #cum_weights /= cum_weights[-1]
#     q = np.array(q) * cum_weights[-1]
#
#     if interpolate:
#         res = np.interp(q, cum_weights, variable)
#     else:
#         res = variable[cum_weights < q][-1]
#     return res


def percentile(variable, weights, percentile=50):
    """
    Parameters
    ----------
    variable : str or array
    weights :  str or array
    percentile : int or list
        Percentile level, if pass 50 we get the median.

    Returns
    -------
    percentile : float
    """

    sorted_idx = np.argsort(variable)
    cum_weights = np.cumsum(weights[sorted_idx])
    percentile_idx = np.searchsorted(
            cum_weights,
            (percentile / 100.) * cum_weights[-1]
    )
    return variable[sorted_idx[percentile_idx]]

def std_moment(variable=None, weights=None, param=None, order=3, ddof=0):
    """Calculate the standardized moment of order `c` for the variable` x` with
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
    - https://en.wikipedia.org/wiki/Moment_(mathematics)#Significance_of_the_moments
    - https://en.wikipedia.org/wiki/Standardized_moment

    TODO
    ----
    It is the general case of the raw and central moments. Review
    implementation.

    """
    if param is None:
        param = mean(variable=variable, weights=weights)

    res = c_moment(variable=variable, weights=weights, order=order, param=param,
                   ddof=ddof)
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
    data : pandas.DataFrame
        Is possible pass a DataFrame with variable and weights, then you must
        pass as `variable` and `weights` the column name stored in `data`.

    Returns
    -------
    mean : array-like or float
    """
    # if pass a DataFrame separate variables.
    variable = variable.copy()
    weights = utils._check_weights(weights, as_of=variable)
    variable, weights = utils._clean_nans_values(variable, weights)
    return np.average(a=variable, weights=weights, axis=0)


def var(variable=None, weights=None, ddof=0):
    """Calculate the population variance of `variable` given `weights`.

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
    ---------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&oldid=778996402

    Notes
    -----
    If stratificated sample must pass with groupby each strata.
    """
    return c_moment(variable=variable, weights=weights, order=2, ddof=ddof)


def coef_variation(variable=None, weights=None):
    """Calculate the coefficient of variation of a `variable` given weights.
    The coefficient of variation is the square root of the variance of the
    incomes divided by the mean income. It has the advantages of being
    mathematically tractable and is subgroup decomposable, but is not bounded
    from above.

    Parameters
    ----------
    data : pandas.DataFrame
    variable : array-like or str
    weights : array-like or str

    Returns
    -------
    coefficient_variation : float

    References
    ----------
    Coefficient of variation. (2017, May 5). In Wikipedia, The Free Encyclopedia.
    Retrieved 15:03, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Coefficient_of_variation&oldid=778842331
    """
    # todo complete docstring
    return var(variable=variable, weights=weights) ** 0.5 / \
           abs(mean(variable=variable, weights=weights))


def kurt(variable=None, weights=None):
    """Calculate the asymmetry coefficient

    Parameters
    ---------
    variable : 1d-array
    w : 1d-array

    Returns
    -------
    kurt : float
        Kurtosis coefficient.

    References
    ---------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&oldid=778996402


    Notes
    -----
    It is an alias of the standardized fourth-order moment.

    """
    return std_moment(variable=variable, weights=weights, order=4)


def skew(variable=None, weights=None):
    """Returns the asymmetry coefficient of a sample.

    Parameters
    ---------
    data : pandas.DataFrame

    variable : array-like, str
    weights : array-like, str

    Returns
    -------
    skew : float

    References
    ---------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&oldid=778996402


    Notes
    -----
    It is an alias of the standardized third-order moment.

    """
    return std_moment(variable=variable, weights=weights, order=3)


