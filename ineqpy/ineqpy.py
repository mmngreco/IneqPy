#!/usr/bin/env python

"""A PYTHON PACKAGE TO QUANTITATIVE ANALYSIS OF INEQUALITY.

This package provide an easy way to realize a quantitative analysis of
inequality, also make easy work with stratified data, in this module you can
find statistics and inequality indicators to this task.

Todo
----
- Rethinking this module as Class.
- https://en.wikipedia.org/wiki/Income_inequality_metrics

"""
import pandas as pd
import numpy as np

# TODO implementar L-moments
# def legendre_pol(x):
#  """
#  https://en.wikipedia.org/wiki/Legendre_polynomials
#  https://es.wikipedia.org/wiki/Polinomios_de_Legendre
#  https://en.wikipedia.org/wiki/Binomial_coefficient
#  http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/lmoment.htm
#  """
#  return None


def _to_df(*args, **kwargs):
    res = None
    if args != ():
        res = pd.DataFrame([*args]).T

    if kwargs is not None:
        if res is not None:
            res = pd.concat([res,
                             pd.DataFrame.from_dict(kwargs, orient='columns')],
                            axis=1)
        else:
            res = pd.DataFrame.from_dict(kwargs, orient='columns')
    return res


def _apply_to_df(func, df, x, weights, *args, **kwargs):
    """This function generalize main arguments as Series of a pd.Dataframe.

    Parameters
    ---------
    func : function
        Function to convert his arguments in Series of an Dataframe.
    df : pandas.Dataframe
        DataFrame whats contains the Series `x_name` and `w_name`
    x_name : str
        Name of the column in `df`
    weights_name : str
        Name of the column in `df

    Returns
    -------
    return : func return
        It's depends of func output type

    """
    return func(df[x], df[weights], *args, **kwargs)


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
    # return np.sum((x-c)^n*counts) / np.sum(counts)
    if data is not None:
        data = data.copy()
        variable = data[variable]
        weights = data[weights] if weights is not None else None
    else:
        variable = variable.copy()
        weights = weights.copy() if weights is not None else None

    if param is None:
        param = mean(variable=variable, weights=weights)
    elif not isinstance(param, (np.ndarray, int, float)):
        raise NotImplementedError
    if weights is None:
        weights = np.repeat([1], len(variable))
    return np.sum((variable - param) ** order * weights) / \
           (np.sum(weights) - ddof)


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
    if data is None:
        data = pd.DataFrame(np.c_[variable, weights, groups], columns=list('vwg'))
        variable, weights, groups = list('vwg')
    if groups is not None:
        den = data.groupby(groups)\
                  .apply(lambda df: df[weights].sum() /
                                    (df[variable].max()-df[variable].min()))
        den /= den.sum()
    else:
        den = data[weights].sum() / (data[variable].max() - data[variable].min())
    return den


def quantile(data=None, variable=None, weights=None, q=0.5, interpolate=True):
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

    if isinstance(q, list):
        kw = {'variable': variable,
              'weights': weights,
              'data': data,
              'interpolate': interpolate}
        res_join = [quantile(**kw, q=qi) for qi in q]
        return pd.Series(res_join, index=q)

    if data is not None:
        data = data.copy()
        name = variable
        variable = data[name].values
        weights = np.ones(variable.shape) if weights is None else \
                  data[weights].values
    else:
        variable = variable.copy()
        weights = weights.copy() if weights is not None else \
                  np.ones(variable.shape)

    weights /= weights.sum()
    order = np.argsort(variable, axis=0)
    weights = weights[order]
    F = weights.cumsum(0)
    variable = variable[order]

    if interpolate:
        res = np.interp(q, F, variable)
    else:
        res = variable[F <= q][-1]
    return res


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
    if param is None:
        param = mean(variable=variable, weights=weights)
    res = c_moment(data=data, variable=variable, weights=weights, order=order, param=param,
                   ddof=ddof)
    res /= variance(data=data, variable=variable, weights=weights, ddof=ddof) ** (order / 2)
    return res


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
    if data is not None:
        data = data.copy()
        variable = data[variable].values
        weights = data[weights].values if weights is not None else None
    else:
        variable = variable.copy()
        weights = weights.copy()

    if np.any(np.isnan(variable)):
        idx = ~np.isnan(variable)
        variable = variable[idx]
        if weights is not None:
            weights = weights[idx]
    return np.average(a=variable, weights=weights, axis=0)


def variance(data=None, variable=None, weights=None, ddof=0):
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
    return c_moment(data=data, variable=variable, weights=weights, order=2, ddof=ddof)

def coefficient_variation(data=None, variable=None, weights=None):
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
    if data is not None:
        variable = data[variable].values
        weights = data[weights].values if weights is not None else np.ones(len(variable))

    return mean(variable=variable, weights=weights) / \
           variance(variable=variable, weights=weights) ** 0.5


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
    if data is not None:
        variable = data[variable].values
        if weights is not None:
            weights = data[weights].values
    return std_moment(variable=variable, weights=weights, order=4)


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
    if data is not None:
        variable = data[variable].values
        if weights is not None:
            weights = data[weights].values
    return std_moment(variable=variable, weights=weights, order=3)


def quasivariance_hat_group(data=None, variable=None, weights=None, group=None):
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
        data = _to_df(x=variable, weights=weights)
        variable = 'x'
        weights = 'weights'

    def sd(df):
        return c_moment(data=df, variable=variable, weights=weights, param=mean(x))

    return data.groupby(group).apply(sd)


def variance_hat_group(data=None, variable='x', weights='w', group='h'):
    """Data a DataFrame calculates the sample variance for each stratum. The
    objective of this function is to make it easy to calculate the moments of
    the distribution that follows an estimator, eg. Can be used to calculate
    the variance that follows the mean.

    Parameters
    ---------

    data : pandas.DataFrame
        Dataframe containing the series needed for the calculation
    x : str
    weights : str
        Name of the weights `w` in the DataFrame
    group : str
        Name of the stratum variable `h` in the DataFrame

    Returns
    -------
    vhat_h : pandas.Series
        A series with the values of the variance of each `h` stratum.

    Notes
    -----

    TODO
    ----
    Review improvements.

    Examples
    --------

    >>> # Computes the variance of the mean
    >>> data = pd.DataFrame(data=[renta, peso, estrato],
                            columns=["renta", "peso", "estrato"])
    >>> v = variance_hat_group(data)
    >>> v
    stratum
    1                700.917.728,64
    2              9.431.897.980,96
    3            317.865.839.789,10
    4            741.304.873.092,88
    5            535.275.436.859,10
    6            225.573.783.240,68
    7            142.048.272.010,63
    8             40.136.989.131,06
    9             18.501.808.022,56
    dtype: float64

    >>> # the value of de variance of the mean:
    >>> v_total = v.sum() / peso.sum() ** 2
        24662655225.947945
    """
    if data is None:
        data = _to_df(x=variable, weights=weights, group=group)
        variable = 'x'
        weights = 'weights'
        group = 'group'

    def v(df):
        """Calculate the variance of each stratum `h`.

        Parameters
        ---------
        df : pandas.DataFrame
            Dataframe containing the data.

        Returns
        -------
        vhat : float
            Value of the population variance for the stratum `h`.

        Notes
        -----
        Source:
        .. math:: r`N_h ^2 \cdot fpc \cdot \frac{ \hatS ^2 _h }{n_h}`

        """
        xi = df[variable].copy().values
        Nh = df[weights].sum()
        fpc = 1 - (len(df) / Nh)
        ddof = 1 if len(df) > 1 else 0
        shat2h = c_moment(variable=xi, order=2, ddof=ddof)
        return (Nh ** 2) * fpc * shat2h / len(df)

    return data.groupby(group).apply(v)


def moment_group(data=None, variable='x', weights='w', group='h', order=2):
    """Calculates the asymmetry of each `h` stratum.

    Parameters
    ----------
    variable : array or str
    weights : array or str
    group : array or str
    data : pd.DataFrame, optional
    order : int, optional

    Returns
    -------
    moment_of_order : float

    TODO
    ----
    Review calculations, it does not appear to be correct.
    Attempt to make a generalization of vhat_group, for any estimator.

    .. warning:: Actually Does Not Work!


    """
    if data is None:
        data = _to_df(x=variable, weights=weights, group=group)
        variable = 'x'
        weights = 'weights'
        group = 'group'

    def mh(df):
        x = df[variable].copy().values
        weights = np.repeat([1], len(df))
        Nh = df.loc[:, weights].sum()
        fpc = 1 - (len(df) / Nh)
        ddof = 1 if len(df) > 1 else 0
        stdm = std_moment(variable=x, weights=weights, order=order, ddof=ddof)
        return (Nh ** order) * fpc * stdm / len(df)
    return data.groupby(group).apply(mh)

'''Inequality functions'''

def concentration(data=None, income=None, weights=None, sort=True):
    """This function calculate the concentration index, according to the
    notation used in [Jenkins1988]_ you can calculate the:
    C_x = 2 / x · cov(x, F_x)
    if x = g(x) then C_x becomes C_y
    when there are taxes:

    y = g(x) = x - t(x)

    Parameters
    ----------
    income : array-like
    weights : array-like
    data : pandas.DataFrame
    sort : bool

    Returns
    -------
    concentration : array-like

    References
    ----------
    Jenkins, S. (1988). Calculating income distribution indices
    from micro-data. National Tax Journal. http://doi.org/10.2307/41788716
    """
    # TODO complete docstring

    # check if DataFrame is passed, if yes then extract variables else make a copy
    if data is not None:
        data = data.copy()
        income = data[income].values
        if weights is not None:
            weights = data[weights].values
    else:
        income = income.copy()
        weights = weights.copy() if weights is not None else np.ones(len(income))

    # if sort is true then sort the variables.
    if sort:
        idx_sort = np.argsort(income)
        income = income[idx_sort]
        weights = weights[idx_sort]
    # main calc
    f_x = weights / weights.sum()
    F_x = f_x.cumsum()
    mu = np.sum(income * f_x)
    cov = np.cov(income, F_x, rowvar=False, aweights=f_x)[0,1]
    return 2 * cov / mu


def lorenz(data=None, income=None, weights=None):
    """In economics, the Lorenz curve is a graphical representation of the 
    distribution of income or of wealth. It was developed by Max O. Lorenz in 
    1905 for representing inequality of the wealth distribution. This function 
    compute the lorenz curve and returns a DF with two columns of axis x and y.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas.DataFrame that contains data.
    income : str or 1d-array, optional
        Population or wights, if a DataFrame is passed then `income` should be a
        name of the column of DataFrame, else can pass a pandas.Series or array.
    weights : str or 1d-array
        Income, monetary variable, if a DataFrame is passed then `y`is a name
        of the series on this DataFrame, however, you can pass a pd.Series or
        np.array.

    Returns
    -------
    lorenz : pandas.Dataframe
        Lorenz distribution in a Dataframe with two columns, labeled x and y,
        that corresponds to plots axis.
    
    References
    ----------
    Lorenz curve. (2017, February 11). In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:34, May 15, 2017, from 
    https://en.wikipedia.org/w/index.php?title=Lorenz_curve&oldid=764853675
    """

    if data is not None:
        data = data.copy()
        income = data[income].values
        weights = data[weights].values
    else:
        income = income.copy()
        weights = weights.copy() if weights is not None else np.ones(len(income))

    total_income = income * weights
    idx_sort = np.argsort(weights)
    weights = weights[idx_sort].cumsum() / weights.sum()
    weights = weights.reshape(len(weights), 1)
    total_income = total_income[idx_sort].cumsum() / total_income.sum()
    total_income = total_income.reshape(len(total_income), 1)
    res = pd.DataFrame(np.c_[weights, total_income], columns=['x', 'y'])
    return res


def gini(data=None, income=None, weights=None, sort=True):
    """The Gini coefficient (sometimes expressed as a Gini ratio or a 
    normalized Gini index) is a measure of statistical dispersion intended to 
    represent the income or wealth distribution of a nation's residents, and is 
    the most commonly used measure of inequality. It was developed by Corrado 
    Gini.
    The Gini coefficient measures the inequality among values of a frequency 
    distribution (for example, levels of income). A Gini coefficient of zero 
    expresses perfect equality, where all values are the same (for example, 
    where everyone has the same income). A Gini coefficient of 1 (or 100%) 
    expresses maximal inequality among values (e.g., for a large number of 
    people, where only one person has all the income or consumption, and all 
    others have none, the Gini coefficient will be very nearly one).

    Parameters
    ---------
    data : pandas.DataFrame
        DataFrame that contains the data.
    income : str or np.array, optional
        Name of the monetary variable `x` in` df`
    weights : str or np.array, optional
        Name of the series containing the weights `x` in` df`
    sorted : bool, optional
        If the DataFrame is previously ordered by the variable `x`, it's must 
        pass True, but False by default.

    Returns
    -------
    gini : float
        Gini Index Value.

    Notes
    -----
    The calculation is done following (discrete probability distribution):
    G = 1 - [∑_i^n f(y_i)·(S_{i-1} + S_i)]
    where:
    - y_i = Income
    - S_i = ∑_{j=1}^i y_i · f(y_i)

    Reference
    ---------
    - Gini coefficient. (2017, May 8). In Wikipedia, The Free Encyclopedia. 
      Retrieved 14:30, May 15, 2017, from 
      https://en.wikipedia.org/w/index.php?title=Gini_coefficient&oldid=779424616
    
    - Jenkins, S. (1988). Calculating income distribution indices 
    from micro-data. National Tax Journal. http://doi.org/10.2307/41788716

    TODO
    ----
    - Implement statistical deviation calculation, VAR (GINI)

    """
    return concentration(data=data, income=income, weights=weights, sort=sort)


def atkinson(data=None, income=None, weights=None, e=0.5):
    """More precisely labelled a family of income inequality measures, the 
    theoretical range of Atkinson values is 0 to 1, with 0 being a state of 
    equal distribution.
    An intuitive interpretation of this index is possible: Atkinson values can 
    be used to calculate the proportion of total income that would be required 
    to achieve an equal level of social welfare as at present if incomes were 
    perfectly distributed. 
    
    For example, an Atkinson index value of 0.20 suggests
    that we could achieve the same level of social welfare with only 
    1 – 0.20 = 80% of income. The theoretical range of Atkinson values is 0 to 1,
    with 0 being a state of equal distribution.

    Parameters
    ---------
    income : array or str
        If `data` is none `income` must be an 1D-array, when `data` is a
        pd.DataFrame, you must pass the name of income variable as string.
    weights : array or str, optional
        If `data` is none `weights` must be an 1D-array, when `data` is a
        pd.DataFrame, you must pass the name of weights variable as string.
    e : int, optional
        Epsilon parameter interpreted by atkinson index as inequality adversion,
        must be between 0 and 1.
    data : pd.DataFrame, optional
        data is a pd.DataFrame that contains the variables.

    Returns
    -------
    atkinson : float

    Reference
    ---------
    Atkinson index. (2017, March 12). In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:35, May 15, 2017, from 
    https://en.wikipedia.org/w/index.php?title=Atkinson_index&oldid=769991852

    TODO
    ----
    - Implement: CALCULATING INCOME DISTRIBUTION INDICES FROM MICRO-DATA
      http://www.jstor.org/stable/41788716
    - The results has difference with stata, maybe have a bug.
    """
    if (income is None) and (data is None):
        raise ValueError('Must pass at least one of both `income` or `df`')

    if data is not None:
        data = data.copy()
        income = data[income].values
        weights = data[weights].values
    else:
        income = income.copy()
        weights = weights.copy() if weights is not None else None

    # not-null condition
    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        if weights is not None:
            weights = weights[mask]

    # not-empty condition
    if len(income) == 0:
        return 0

    N = len(income)  # observations

    # not-empty wights
    if weights is None:
        weights = np.ones(N)

    # auxiliar variables: mean and distribution
    mu = mean(variable=income, weights=weights)
    f_i = weights / sum(weights)  # density function

    # main calc
    if e == 1:
        atkinson = 1 - np.power(np.e, np.sum(f_i * np.log(income) - np.log(mu)))
    elif (0 <= e) or (e < 1):
        atkinson = 1 - np.power(np.sum(f_i * np.power(income / mu, 1 - e)),
                                1 / (1 - e))
    else:
        assert (e < 0) or (e > 1), "Not valid e value,  0 ≤ e ≤ 1"
        atkinson = None
    return atkinson


def atkinson_group(data=None, income=None, weights=None, group=None, e=0.5):
    r"""The Atkinson index (also known as the Atkinson measure or Atkinson 
    inequality measure) is a measure of income inequality developed by 
    British economist Anthony Barnes Atkinson. The measure is useful in 
    determining which end of the distribution contributed most to the observed 
    inequality.The index is subgroup decomposable. This means that overall 
    inequality in the population can be computed as the sum of the corresponding
    Atkinson indices within each group, and the Atkinson index of the group mean
    incomes.

    Parameters
    ---------
    income : str or np.array
        Income variable, you can pass name of variable in `df` or array-like
    weights : str or np.array
        probability or weights, you can pass name of variable in `df` or
        array-like
    groups : str or np.array
        stratum, name of stratum in `df` or array-like
    e : int, optional
        Value of epsilon parameter
    data : pd.DataFrame, optional
        DataFrame that's contains the previous data.

    Returns
    -------
    atkinson_by_group : float

    Reference
    ---------
    Atkinson index. (2017, March 12). In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:52, May 15, 2017, from 
    https://en.wikipedia.org/w/index.php?title=Atkinson_index&oldid=769991852

    TODO
    ----
    - Review function, has different results with stata.
    """
    if weights is None:
        if data is None:
            weights = np.repeat([1], len(income))
        else:
            weights = np.repeat([1], len(data))

    if data is None:
        data = _to_df(income=income, weights=weights, group=group)
        income = 'income'
        weights = 'weights'
        group = 'group'

    N = len(data)

    def a_h(df):
        """
        Funtion alias to calculate atkinson from a DataFrame
        """
        if df is None:
            raise ValueError

        res = atkinson(income=df[income].values, weights=df[weights].values, e=e) * len(df) / N
        return res

    # main calc:
    if data is not None:
        data = data.copy()
        atk_by_group = data.groupby(group).apply(a_h)
        mu_by_group = data.groupby(group).apply(lambda dw: mean(dw[income],
                                                                dw[weights]))

        return atk_by_group.sum() + atkinson(income=mu_by_group.values)
    else:
        raise NotImplementedError


def kakwani(data=None, tax=None, income_pre_tax=None, weights=None):
    """The Kakwani (1977) index of tax progressivity is defined as twice the 
    area between the concentration curves for taxes and pre-tax income, 
    or equivalently, the concentration index for t(x) minus the Gini index for 
    x, i.e.
    
    K = C(t) - G(x)
      = (2/t) cov [t(x), F(x)] - (2/x) cov [x, F(x)].

    Parameters
    ----------
    data : pandas.DataFrame
        This variable is a DataFrame that contains all data required in
        columns.
    tax_variable : array-like or str
        This variable represent tax payment of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    income_pre_tax : array-like or str
        This variable represent income of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    weights : array-like or str
        This variable represent weights of each person, if pass array-like
        then data must be None, else you pass str-name column in `data`.

    Returns
    -------
    kakwani : float
    
    References
    ----------
    Jenkins, S. (1988). Calculating income distribution indices from micro-data. 
    National Tax Journal. http://doi.org/10.2307/41788716
    """
    if weights is None:
        if data is None:
            weights = np.repeat([1], len(tax))
        else:
            weights = np.repeat([1], len(data))

    if data is None:
        data = _to_df(income_pre_tax=income_pre_tax,
                      tax=tax,
                      weights=weights)
        income_pre_tax = 'income_pre_tax'
        tax = 'tax'
        weights = 'weights'

    # main calc
    c_t = concentration(data=data, income=tax, weights=weights, sort=True)
    g_y = concentration(data=data, income=income_pre_tax, weights=weights,
                        sort=True)
    return c_t - g_y


def reynolds_smolensky(data=None, income_pre_tax=None, income_post_tax=None,
                       weights=None):
    """The Reynolds-Smolensky (1977) index of the redistributive effect of 
    taxes, which can also be interpreted as an index of progressivity 
    (Lambert 1985), is defined as:
    
    L = Gx - Gy 
      = [2/x]cov[x,F(x)] - [2/ybar] cov [y, F(y)]. 

    Parameters
    ----------
    data : pandas.DataFrame
        This variable is a DataFrame that contains all data required in it's
        columns.
    income_pre_tax : array-like or str
        This variable represent tax payment of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    income_post_tax : array-like or str
        This variable represent income of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    weights : array-like or str
        This variable represent weights of each person, if pass array-like
        then data must be None, else you pass str-name column in `data`.

    Returns
    -------
    reynolds_smolensky : float
    
    References
    ----------
    Jenkins, S. (1988). Calculating income distribution indices from micro-data.
    National Tax Journal. http://doi.org/10.2307/41788716
    """
    if weights is None:
        if data is None:
            weights = np.repeat([1], len(income_pre_tax))
        else:
            weights = np.repeat([1], len(data))
    g_y = concentration(data=data, income=income_post_tax, weights=weights)
    g_x = concentration(data=data, income=income_pre_tax, weights=weights)
    return g_x - g_y

# todo to complete
# def suits():
#     return


def theil(data=None, income=None, weights=None):
    """The Theil index is a statistic primarily used to measure economic 
    inequality and other economic phenomena. It is a special case of the 
    generalized entropy index. It can be viewed as a measure of redundancy, 
    lack of diversity, isolation, segregation, inequality, non-randomness, and 
    compressibility. It was proposed by econometrician Henri Theil. 

    Parameters
    ----------
    data : pandas.DataFrame
        This variable is a DataFrame that contains all data required in it's
        columns.
    income : array-like or str
        This variable represent tax payment of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    weights : array-like or str
        This variable represent weights of each person, if pass array-like
        then data must be None, else you pass str-name column in `data`.

    Returns
    -------
    theil : float
    
    References
    ----------
    Theil index. (2016, December 17). In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:17, May 15, 2017, from 
    https://en.wikipedia.org/w/index.php?title=Theil_index&oldid=755407818

    """

    if weights is None:
        if data is None:
            weights = np.repeat([1], len(income))
        else:
            weights = np.repeat([1], len(data))

    if data is not None:
        data = data.copy()
        income = data[income].values
        weights = data[weights].values
    else:
        income = income.copy()
        weights = weights.copy()

    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        weights = weights[mask]

    # variables needed
    mu = mean(variable=income, weights=weights)
    f_i = weights / np.sum(weights)
    # main calc
    theil = np.sum((f_i * income / mu) * np.log(income / mu))
    return theil


def avg_tax_rate(data=None, total_tax=None, total_base=None, weights=None):
    """This function compute the average tax rate given a base income and a
    total tax.

    Parameters
    ----------
    total_base : str or numpy.array
    total_tax : str or numpy.array
    data : pd.DataFrame

    Returns
    -------
    avg_tax_rate : float or pd.Series
        Is the ratio between mean the tax income and base of income.

    Reference
    ---------
    Panel de declarantes de IRPF 1999-2007: Metodología, estructura y variables. 
    (2011). Panel de declarantes de IRPF 1999-2007: 
    Metodología, estructura y variables. Documentos.
    """
    has_list_of_names = False
    n_cols = 1
    base_name = None
    tax_name = None

    if isinstance(total_base, (list, np.ndarray)):
        has_list_of_names = True
        n_cols = len(total_base)

    if data is not None:
        data = data.copy()
        base_name = total_base
        tax_name = total_tax
    else:
        total_base = total_base.copy()
        total_tax = total_tax.copy()
        weights = weights.copy() if weights is not None else None

    numerator = mean(data=data, variable=total_tax , weights=weights)
    denominator = mean(data=data, variable=total_base, weights=weights)
    res = numerator / denominator

    if not has_list_of_names:
        names = [tax_name + '_' + base_name]
    else:
        names = [tax_name[i] + '_' + b for i, b in enumerate(base_name)]

    res = pd.Series(res, index=names)
    return res