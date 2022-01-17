"""Descriptive statistics.

This module contains main descriptive statistics like: mean, variance, etc.

"""

from . import _statistics as stat
from . import utils


def c_moment(
    variable=None, weights=None, data=None, order=2, param=None, ddof=0
):
    """Calculate central momment.

    Calculate the central moment of `x` with respect to `param` of order `n`,
    given the weights `w`.

    Parameters
    ----------
    variable : 1d-array
        Variable
    weights : 1d-array
        Weights
    data : pandas.DataFrame
        Contains all variables needed.
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
    Implement: https://en.wikipedia.org/wiki/L-moment#cite_note-wang:96-6

    """
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.c_moment(variable, weights, order, param, ddof)


def percentile(
    variable=None, weights=None, data=None, p=50, interpolate="lower"
):
    """Calculate the value of a quantile given a variable and his weights.

    Parameters
    ----------
    variable : str or array
    weights :  str or array
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
    q : float
        Quantile level, if pass 0.5 means median.
    interpolate : bool

    Returns
    -------
    percentile : float or pd.Series

    """
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.percentile(variable, weights, p, interpolate)


def std_moment(
    variable=None, weights=None, data=None, param=None, order=3, ddof=0
):
    """Calculate standarized momment.

    Calculate the standardized moment of order `c` for the variable` x` with
    respect to `c`.

    Parameters
    ----------
    variable : 1d-array
       Random Variable
    weights : 1d-array, optional
       Weights or probability
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
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
    - https://en.wikipedia.org/wiki/Standardized_moment

    Todo
    ----
    It is the general case of the raw and central moments. Review
    implementation.

    """
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.std_moment(variable, weights, param, order, ddof)


def mean(variable=None, weights=None, data=None):
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
    if data is not None:
        variable, weights = utils.extract_values(data, variable, weights)
    return stat.mean(variable, utils.not_empty_weights(weights, variable))


def density(variable=None, weights=None, groups=None, data=None):
    """Density in percentage.

    Calculates density in percentage. This make division of variable inferring
    width in groups as max - min.

    Parameters
    ----------
    variable : numpy.array or pandas.DataFrame
        Main variable.
    weights : numpy.array or pandas.DataFrame
        Weights of main variable.
    groups : numpy.array or pandas.DataFrame
        Label that show which group each element belongs to.
    data : pd.DataFrame, optional
        Object that contains all variables needed.


    Returns
    -------
    density : array-like

    References
    ----------
    Histogram. (2017, May 9). In Wikipedia, The Free Encyclopedia. Retrieved
    14:47, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Histogram&oldid=779516918
    """
    variable, weights = utils.extract_values(data, variable, weights)
    if groups is not None:
        groups = data[groups].values
    return stat.density(variable, weights, groups)


def var(variable=None, weights=None, data=None, ddof=0):
    """Calculate the variance.

    Calculate the population variance of `variable` given `weights`.

    Parameters
    ----------
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
    variable : 1d-array or pd.Series or pd.DataFrame
        Variable on which the quasivariation is estimated
    weights : 1d-array or pd.Series or pd.DataFrame
        Weights of the `variable`.
    data : pd.DataFrame
        Object that contains all variables needed.
    ddof : int
        Degree of freedom.

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
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.var(variable, weights, ddof)


def coef_variation(variable=None, weights=None, data=None):
    """Calculate the coefficient of variation.

    Calculate the coefficient of variation of a `variable` given weights.
    The coefficient of variation is the square root of the variance of the
    incomes divided by the mean income. It has the advantages of being
    mathematically tractable and is subgroup decomposable, but is not bounded
    from above.

    Parameters
    ----------
    variable : array-like or str
    weights : array-like or str
    data : pandas.DataFrame

    Returns
    -------
    coefficient_variation : float

    References
    ----------
    Coefficient of variation. (2017, May 5). In Wikipedia, The Free
    Encyclopedia. Retrieved 15:03, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Coefficient_of_variation
    """
    # TODO complete docstring
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.coef_variation(variable, weights)


def kurt(variable=None, weights=None, data=None):
    """Calculate the Kurtosis coefficient.

    Parameters
    ----------
    variable : 1d-array
    weights : 1d-array
    data : pandas.DataFrame
        Object which stores ``variable`` and ``weights``.

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
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.kurt(variable, weights)


def skew(variable=None, weights=None, data=None):
    """Return the asymmetry coefficient of a sample.

    Parameters
    ----------
    data : pandas.DataFrame
    variable : array-like, str
    weights : array-like, str
    data : pandas.DataFrame
        Object which stores ``variable`` and ``weights``.

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
    variable, weights = utils.extract_values(data, variable, weights)
    return stat.skew(variable, weights)
