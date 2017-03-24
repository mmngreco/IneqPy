#!/usr/bin/env python

"""A PYTHON PACKAGE TO QUANTITATIVE ANALYSIS OF INEQUALITY.

This package provide an easy way to realize a quantitative analysis of
inequality, also make easy work with stratified data, in this module you can
find statistics and inequality indicators to this task.

Todo
----
- Rethinking this module as Class.

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


def cmoment(variable, weights=None, order=2, param=None, ddof=0):
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
    if param is None:
        param = mean(variable, weights)
    elif not isinstance(param, (np.ndarray, int, float)):
        raise NotImplementedError
    if weights is None:
        weights = np.repeat([1], len(variable))
    return np.sum((variable - param) ** order * weights) / \
           (np.sum(weights) - ddof)


def quantile(variable, weights, q, data=None, interpolate=True):
    """Calculate the value of a quantile given a variable and his weights.

    Parameters
    ----------
    variable : str or array
    weights :  str or array
    q : float
    data : pandas.DataFrame
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
        res_join = [quantile(q=qi, **kw) for qi in q]
        return pd.Series(res_join, index=q)

    if data is not None:
        name = variable
        variable = data[name].values
        if weights is None:
            weights = np.ones(variable.shape)
        else:
            weights = data[weights].values
    if weights is None:
        weights = np.ones(variable.shape)

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


def std_moment(variable, weights=None, param=None, order=3, ddof=0):
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

    Notes
    -----

    Source:

    - https://en.wikipedia.org/wiki/Moment_(mathematics)#Significance_of_the_moments
    - https://en.wikipedia.org/wiki/Standardized_moment


    TODO
    ----

    It is the general case of the raw and central moments. Review
    implementation.

    """
    if weights is None:
        weights = np.repeat([1], len(x))
    if param is None:
        param = xbar(x, weights)
    # m = np.subtract(x, c)
    # m = np.power(m, n) * w / np.sum(w)
    # m = np.sum(m)
    # m = np.divide(m, np.power(var(x, w, ddof=ddof), n / 2))
    # return m
    res = cmoment(x, weights, order, param=param, ddof=ddof)
    res /= var(x, weights, ddof=ddof) ** (order / 2)
    return res


def xbar(x, weights=None, data=None):
    """Calculate the mean of `x` given weights `w`.

    Parameters
    ----------

    x : 1d-array or pd.Series or pd.DataFrame
        Variable on which the mean is estimated
    w : 1d-array or pd.Series or pd.DataFrame, optional
        Weights of the `x` variable of a dimension

    Returns
    -------

    xbar : 1d-array or pd.Series or float
    """
    # todo need the same for weights ?
    if data is not None:
        x = data[x].values
        if weights is not None:
            weights = data[weights].values

    if np.any(np.isnan(x)):
        idx = ~np.isnan(x)
        x = x[idx]
        if weights is not None:
            weights = weights[idx]

    return np.average(x, weights=weights, axis=0)


def var(x, weights=None, data=None, ddof=0):
    """Calculate the population variance of `x` given
    weights `w`, for a homogeneous population.


    Parameters
    ----------

    x : 1d-array or pd.Series or pd.DataFrame
        Variable on which the quasivariation is estimated
    w : 1d-array or pd.Series or pd.DataFrame
        Weights of the `x` variable of a dimension

    Returns
    -------

    Shat2 : 1d-array or pd.Series or float
        Estimation of quasivariance of `x`

    Notes
    -----

    If stratificated sample must pass with groupby each strata.
    """
    if data is not None:
        x = data[x].values
        if weights is not None:
            weights = data[weights].values
    if weights is None:
        weights = np.repeat([1], len(x))
    return cmoment(x, weights=weights, order=2, ddof=ddof)


def kurt(x, weights, data=None):
    """Calculate the asymmetry coefficient

    Parameters
    ---------

    x : 1d-array
    w : 1d-array

    Returns
    -------

    kurt : float
        Kurtosis coefficient.

    Notes
    -----

    It is an alias of the standardized fourth-order moment.

    """
    if data is not None:
        x = data[x].values
        if weights is not None:
            weights = data[weights].values
    return stdmoment(x=x, weights=weights, order=4)


def skew(x, weights, data=None):
    """Returns the asymmetry coefficient of a sample.

    Parameters
    ---------
    x : 1d-array
    w : 1d-array

    Returns
    -------
    skew : float

    Notes
    -----
    It is an alias of the standardized third-order moment.

    """
    if data is not None:
        x = data[x].values
        if weights is not None:
            weights = data[weights].values
    return stdmoment(x=x, weights=weights, order=3)


def shat2_group(x, weights, group, data=None):
    """Sample variance of `x_name`, calculated as the second-order central
    moment.

    Parameters
    ---------
    x : array or str
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
    data : pd.DataFrame, optional
        pd.DataFrame has all variables needed.
        order


    Returns
    -------

    shat2_group : array or pd.Series

    Notes
    -----

    This function is useful to calculate the variance of the mean.

    TODO
    ----

    Review function
    """

    if data is None:
        data = _to_df(x=x, weights=weights)
        x = 'x'
        weights = 'weights'

    def sd(df):
        x = df.loc[:, x].copy().values
        weights = np.repeat([1], len(df))
        return cmoment(x, weights, 2, param=xbar(x))
    return data.groupby(group).apply(sd)


def vhat_group(x='x', weights='w', group='h', data=None):
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
    >>> v = vhat_group(data,x_name='income')
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
        data = _to_df(x=x, weights=weights, group=group)
        x = 'x'
        weights = 'weights'
        group = 'group'

    def v(df):
        """Calculate the variance of each stratum `h`

        Parameters
        ---------
        df : pandas.DataFrame
            Dataframe containing the data

        Returns
        -------

        vhat : float
            Value of the population variance for the stratum `h`

        Notes
        -----

        Source:
        .. math:: r`N_h ^2 \cdot fpc \cdot \frac{ \hatS ^2 _h }{n_h}`

        """
        xi = df[x].copy().values
        Nh = df[weights].sum()
        fpc = 1 - (len(df) / Nh)
        ddof = 1 if len(df) > 1 else 0
        shat2h = cmoment(x=xi, order=2, ddof=ddof)
        return (Nh ** 2) * fpc * shat2h / len(df)
    return data.groupby(group).apply(v)


def moment_group(x='x', weights='w', group='h', data=None, order=2):
    """Calculates the asymmetry of each `h` stratum.

    Parameters
    ----------

    x : array or str
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
        data = _to_df(x=x, weights=weights, group=group)
        x = 'x'
        weights = 'weights'
        group = 'group'

    def mh(df):
        x = df.loc[:, x].copy().values
        weights = np.repeat([1], len(df))
        Nh = df.loc[:, weights].sum()
        fpc = 1 - (len(df) / Nh)
        ddof = 1 if len(df) > 1 else 0
        stdm = stdmoment(x=x, weights=weights, order=order, ddof=ddof)
        return (Nh ** order) * fpc * stdm / len(df)
    return data.groupby(group).apply(mh)

'''Inequality functions'''


def concentration(income, weights=None, sort=True, data=None):
    """This function calculate the concentration index, according to the
    notation used in [Jenkins1988]_ you can calculate the
    :math: C_x = 2 / x · cov(x, F_x)
    if x = g(x) then C_x becomes C_y
    when there are taxes:
    y = g(x) = x - t(x)

    Parameters
    ----------
    income :
    weights :
    data :
    sort :

    Returns
    -------

    """
    # todo complete docstring

    if data is not None:
        income = data[income].values
        if weights is not None:
            weights = data[weights].values

    if weights is None:
        weights = np.repeat(1, len(income))

    if sort:
        idx_sort = np.argsort(income)
        income = income[idx_sort]
        weights = weights[idx_sort]

    f_x = weights / weights.sum()
    F_x = f_x.cumsum()
    mu = np.sum(income * f_x)
    cov = np.cov(income, F_x, rowvar=False, aweights=f_x)[0,1]
    return 2 * cov / mu


def lorenz(income, weights, data=None):
    """This function compute the lorenz curve and returns a DF with two columns
    of axis x and y.

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

    """

    if data is not None:
        income = data[income].values
        weights = data[weights].values
    total_income = income * weights
    idx_sort = np.argsort(weights)
    weights = weights[idx_sort].cumsum() / weights.sum()
    weights = weights.reshape(len(weights), 1)
    total_income = total_income[idx_sort].cumsum() / total_income.sum()
    total_income = total_income.reshape(len(total_income), 1)
    res = pd.DataFrame(np.c_[weights, total_income], columns=['x', 'y'])
    return res


def gini(income, weights=None, data=None, sort=True):
    """Compute de index Gini.

    Parameters
    ---------
    data : pandas.DataFrame
        DataFrame that contains the data.
    income : str or np.array, optional
        Name of the monetary variable `x` in` df`
    weights : str or np.array, optional
        Name of the series containing the weights `x` in` df`
    sorted : bool, optional
        If the DataFrame is previously ordered by the variable `x`, it's must pass True, but False by default.

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
    - https://en.wikipedia.org/wiki/Gini_coefficient
    - CALCULATING INCOME DISTRIBUTION INDICES FROM MICRO-DATA - STEPHEN JENKINS

    TODO
    ----
    - Implement statistical deviation calculation, VAR (GINI)
    - Clear comments
    - Rename output

    """
    # another aproach
    # x = df[income]
    # f_x = df[weights]
    # f_x /= f_x.sum()
    # si = x * f_x
    # si = si.cumsum()
    # si_1 = si.shift(1)
    # sn = si.iloc[-1]
    # g = (1 - np.divide(np.sum(f_x * (si_1 + si)), sn))
    # return G, G2, G3, G4
    return concentration(income=income, weights=weights, sort=sort, data=data)


def atkinson(income, weights=None, e=0.5, data=None):
    """Calculate the coefficient of atkinson

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
    Source: https://en.wikipedia.org/wiki/Atkinson_index

    TODO
    ----
    - Implement: CALCULATING INCOME DISTRIBUTION INDICES FROM MICRO-DATA
      http://www.jstor.org/stable/41788716

    .. warning:: The results has difference with stata, maybe have a bug.
    """
    if (income is None) and (data is None):
        raise ValueError('Must pass at least one of both `income` or `df`')

    if data is not None:
        income = data[income].values
        weights = data[weights].values

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
        weights = np.repeat(1, N)

    mu = xbar(income, weights)
    f_i = weights / sum(weights)  # density function
    # another aproach
    # e value condition
    # if e == 1:
    #     Ee = np.power(np.e, np.sum(f_i * np.log(income)))
    # elif (0 <= e) or (e < 1):
    #     Ee = np.power(np.sum(f_i * np.power(income, 1 - e)), 1 / (1 - e))
    # else:
    #     assert (e < 0) or (e > 1), "Not valid e value,  0 ≤ e ≤ 1"
    #     Ee = None
    #     return None
    # atkinson = (mu - Ee) / mu
    if e == 1:
        atkinson = 1 - np.power(np.e, np.sum(f_i * np.log(income) - np.log(mu)))
    elif (0 <= e) or (e < 1):
        atkinson = 1 - np.power(np.sum(f_i * np.power(income / mu, 1 - e)),
                                1 / (1 - e))
    else:
        assert (e < 0) or (e > 1), "Not valid e value,  0 ≤ e ≤ 1"
        atkinson = None
    return atkinson


def atkinson_group(income, weights, group, data=None, e=0.5):
    """

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

    Notes
    -----

    Source: https://en.wikipedia.org/wiki/Atkinson_index

    TODO
    ----
    Review function, has different results with stata.

    Examples
    --------

    """
    if weights is None:
        if data is None:
            weights = np.reapeat(1, len(income))
        else:
            weights = np.reapeat(1, len(data))

    if data is None:
        data = _to_df(income=income, weights=weights, group=group)
        income = 'income'
        weights = 'weights'
        group = 'group'
    N = len(data)

    def a_h(df):
        '''
        Funtion alias to calculate atkinson from a DataFrame
        '''
        if df is None:
            raise ValueError

        res = atkinson(income=df[income].values, weights=df[weights].values,
                       e=e) * len(df) / N
        return res

    if data is not None:
        atk_by_group = data.groupby(group).apply(a_h)
        mu_by_group = data.groupby(group).apply(lambda dw: xbar(dw[income],
                                                                dw[weights]))

        return atk_by_group.sum() + atkinson(income=mu_by_group.values)
    else:
        raise NotImplementedError


def kakwani(tax, income_after_tax, weights, data):

    if weights is None:
        if data is None:
            weights = np.reapeat(1, len(tax))
        else:
            weights = np.reapeat(1, len(data))

    if data is None:
        data = _to_df(income_after_tax=income_after_tax,
                      tax=tax,
                      weights=weights)
        income_after_tax = 'income_after_tax'
        tax = 'tax'
        weights = 'weights'
    c_t = concentration(income=tax, weights=weights, sort=True, data=data)
    g_y = concentration(income=income_after_tax, weights=weights, sort=True,
                        data=data)
    return c_t - g_y


def reynolds_smolensky(income_before_tax, income_after_tax, weights, data=None):

    if weights is None:
        if data is None:
            weights = np.repeat(1, len(income_before_tax))
        else:
            weights = np.repeat(1, len(data))

    if data is not None:
        income_after_tax = data[income_after_tax].values
        income_before_tax = data[income_before_tax].values
        weights = data[weights].values
    g_y = concentration(income=income_after_tax, weights=weights, data=data)
    g_x = concentration(income=income_before_tax, weights=weights, data=data)
    return g_x - g_y

# todo
# def suits():
#     return


def theil(income, weights=None, data=None):
    """This function calculates the theil index

    Parameters
    ----------
    income :
    weights :
    data :

    Returns
    -------
    theil : float
    """

    if weights is None:
        if data is None:
            weights = np.repeat(1, len(income))
        else:
            weights = np.repeat(1, len(data))

    if data is not None:
        income = data[income].values
        weights = data[weights].values

    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        weights = weights[mask]

    mu = xbar(income, weights)
    f_i = weights / np.sum(weights)
    t = np.sum((f_i * income / mu) * np.log(income / mu))
    return t


def avg_tax_rate(base, tax, weights=None, data=None):
    """This function cumpute the average tax rate from base income and total tax

    :param base:
    :param tax:
    :param data:
    :return:


    Notes
    -----

    Siguiendo la metodología iniciada en Picos, Pérez y González (2011) como en
    la muestra de declarantes de 2008, 2009, 2010 y 2011. En el Gráfico 6 se
    recogen los valores medios para cinco definiciones de tipos medios
    calculados individualmente:

    tm1: cociente entre el resultado de aplicar las escalas del impuesto a las
         bases liquidables y la base liquidable, que refleja el efecto de las
         escalas del impuesto.
    tm2: cociente entre el resultado de aplicar las escalas del impuesto a las
         bases liquidables y la base imponible, que añade a lo anterior el
         efecto de las reducciones aplicadas en dicha base.
    tm3: cociente entre el resultado de aplicar las escalas del impuesto a las
         bases liquidables y la renta del periodo, que añade a lo anterior el
         efecto de la reducción por rendimientos del trabajo.
    tm4: cociente entre la cuota íntegra y la renta del periodo, que añade a lo
         anterior el efecto del mínimo personal y familiar.
    tm5: cociente entre la cuota resultante de la autoliquidación3 y la renta
         del periodo, que añade a lo anterior el efecto de las deducciones en
         cuota.
    """
    # todo introduce weights calc to be correct average
    if data is None:
        res = [xbar(t, weights) / xbar(b, weights) for t, b in zip(tax, base)]
    else:
        num_df = data[tax].apply(lambda x: xbar(x, data[weights])).values
        den_df = data[base].apply(lambda x: xbar(x, data[weights])).values
        res = num_df / den_df
        if type(base) == str:
            names = base + '_' + tax
        else:
            names = [tax[i] + '_' + b for i, b in enumerate(base)]
        res = pd.Series(res, index=names)
    return res