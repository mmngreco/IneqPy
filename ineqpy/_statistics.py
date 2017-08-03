import numpy as np
import pandas as pd
from .utils import misc


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
    weights = misc._check_weights(weights, as_of=variable)

    if param is None:
        param = mean(variable=variable, weights=weights)
    elif not isinstance(param, (np.ndarray, int, float)):
        raise NotImplementedError

    return np.sum((variable - param) ** order * weights) / \
           (np.sum(weights) - ddof)


def quantile(variable=None, weights=None, q=0.5, interpolate=True):
    """Calculate the value of a quantile given a variable and his weights.

    Parameters
    ----------
    variable : str or array
    weights :  str or array
    q : float
        Quantile level, if pass 0.5 means median.
    interpolate : bool

    Returns
    -------
    quantile : float or pd.Series

    """

    if isinstance(q, list):
        kw = {'variable': variable,
              'weights': weights,
              'interpolate': interpolate}
        res_join = [quantile(**kw, q=qi) for qi in q]
        return dict(zip(q, res_join))

    variable = variable.copy()
    weights = misc._check_weights(weights, as_of=variable)
    weights = weights * (1 / weights.sum()) # normalize
    variable, weights = misc._sort_values(variable, weights)
    F = weights.cumsum(0)

    if interpolate:
        res = np.interp(q, F, variable)
    else:
        res = variable[F <= q][-1]
    return res


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
    weights = misc._check_weights(weights, as_of=variable)
    variable, weights = misc._clean_nans_values(variable, weights)
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


def quasivariance_hat_group(variable=None, weights=None, group=None):
    """Sample variance of `variable`, calculated as the second-order central
    moment.

    Parameters
    ---------
    data : pd.DataFrame, optional
        pd.DataFrame that contains all variables needed.
    variable : array or str
        variable `x` apply the statistic. If `data` is None then must pass this
        argument as array, else as string name in `data`
    weights : array or str
        weights can be interpreted as frequency, probability,
        density function of `x`, each element in `x`. If `data` is None then
        must pass this argument as array, else as string name in `data`
    group : array or str
        group is a categorical variable to calculate the statistical by each
        group. If `data` is None then must pass this argument as array, else as
        string name in `data`



    Returns
    -------
    shat2_group : array or pd.Series

    References
    ---------
    Moment (mathematics). (2017, May 6). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:40, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&oldid=778996402

    Notes
    -----
    This function is useful to calculate the variance of the mean.

    TODO
    ----
    Review function
    """

    if data is None:
        data = misc._to_df(x=variable, weights=weights)
        variable = 'x'
        weights = 'weights'

    def sd(df):
        return c_moment(data=df, variable=variable, weights=weights, param=mean(x))

    return data.groupby(group).apply(sd)