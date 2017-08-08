import numpy as np
import pandas as pd
from . import _statistics
from . import utils
from utils.msic import _to_df


def c_moment(data=None, variable=None, weights=None, order=2, param=None,
             ddof=0):
    """Calculate the central moment of `x` with respect to `param` of order `n`,
    given the weights `w`.

    Parameters
    ----------
    data :
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.c_moment(variable, weights, order, param, ddof)


def percentile(data=None, variable=None, weights=None, p=50, interpolate=True):
    """Calculate the value of a quantile given a variable and his weights.

    Parameters
    ----------
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
    variable : str or array
    weights :  str or array
    q : float
        Quantile level, if pass 0.5 means median.
    interpolate : bool

    Returns
    -------
    quantile : float or pd.Series

    """
    variable, weights = utils._extract_values(data, variable, weights)

    return _.statistics.percentile(variable, weights, p, interpolate)


def std_moment(data=None, variable=None, weights=None, param=None, order=3,
               ddof=0):
    """Calculate the standardized moment of order `c` for the variable` x` with
    respect to `c`.

    Parameters
    ----------
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.std_moment(data, variable, weights, param, order, ddof)


def mean(data=None, variable=None, weights=None):
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.mean(variable, weights)


def density(data=None, variable=None, weights=None, groups=None):
    """Calculates density in percentage. This make division of variable inferring
    width in groups as max - min.

    Parameters
    ----------
    data : pd.DataFrame, optional
        pandas.DataFrame that contains all variables needed.
    variable :
    weights :
    groups :

    Returns
    -------
    density : array-like

    References
    ----------
    Histogram. (2017, May 9). In Wikipedia, The Free Encyclopedia. Retrieved
    14:47, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Histogram&oldid=779516918
    """
    variable, weights = utils._extract_values(data, variable, weights)
    if groups:
        groups = data[groups].values
    return _statistics.density(variable, weights, groups)


def var(data=None, variable=None, weights=None, ddof=0):
    """Calculate the population variance of `variable` given `weights`.

    Parameters
    ----------
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.var(variable, weights, ddof)


def coef_variation(data=None, variable=None, weights=None):
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
    # TODO complete docstring
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.coef_variation(variable, weights)



def kurt(data=None, variable=None, weights=None):
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.kurt(variable, weights)


def skew(data=None, variable=None, weights=None):
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
    variable, weights = utils._extract_values(data, variable, weights)
    return _statistics.skew(variable, weights)
