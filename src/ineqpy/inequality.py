#!/usr/bin/env python

"""Analysis of inequality.

This package provide an easy way to realize a quantitative analysis of
grouped, also make easy work with stratified data, in this module you can
find statistics and grouped indicators to this task.

Todo
----
- Rethinking this module as Class.
- https://en.wikipedia.org/wiki/Income_inequality_metrics

"""
import numpy as np
import pandas as pd
from .statistics import mean
from . import utils


__all__ = [
    "atkinson",
    "avg_tax_rate",
    "concentration",
    "gini",
    "kakwani",
    "lorenz",
    "reynolds_smolensky",
    "theil",
    "top_rest",
    "hoover",
]


def concentration(income, weights=None, data=None, sort=True):
    """Calculate concentration's index.

    This function calculate the concentration index, according to the notation
    used in [Jenkins1988]_ you can calculate the:

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
        If true, will sort the values.

    Returns
    -------
    concentration : array-like

    References
    ----------
    Jenkins, S. (1988). Calculating income distribution indices
    from micro-data. National Tax Journal. http://doi.org/10.2307/41788716
    """
    # TODO complete docstring

    # check if DataFrame is passed, if yes then extract variables else make a
    # copy
    income, weights = utils.extract_values(data, income, weights)
    if weights is None:
        weights = utils.not_empty_weights(weights, like=income)

    if income.ndim == 0:
        income = np.array([income])
    elif income.ndim == 2:
        income = np.squeeze(income, axis=1)

    if weights.ndim == 0:
        weights = np.array([weights])
    elif weights.ndim == 2:
        weights = np.squeeze(weights, axis=1)

    # Small shortcut to avoid warnings below
    if income.size <= 1:
        return np.nan

    # if sort is true then sort the variables.
    if sort:
        income, weights = utils._sort_values(income, weights)

    # main calc
    f_x = np.atleast_1d(utils.normalize(weights))
    F_x = f_x.cumsum(axis=0)
    mu = np.sum(income * f_x)
    cov = np.cov(income, F_x, rowvar=False, aweights=f_x)[0, 1]
    return 2 * cov / mu


def lorenz(income, weights=None, data=None):
    """Calculate Lorent's curve.

    In economics, the Lorenz curve is a graphical representation of the
    distribution of income or of wealth. It was developed by Max O. Lorenz in
    1905 for representing grouped of the wealth distribution. This function
    compute the lorenz curve and returns a DF with two columns of axis x and y.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas.DataFrame that contains data.
    income : str or 1d-array, optional
        Population or wights, if a DataFrame is passed then `income` should be
        a name of the column of DataFrame, else can pass a pandas.Series or
        array.
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
        income, weights = utils.extract_values(data, income, weights)
    if weights is None:
        weights = utils.not_empty_weights(weights, like=income)

    total_income = income * weights
    idx_sort = np.argsort(income)

    weights = weights[idx_sort].cumsum() / weights.sum()
    weights = weights.reshape(len(weights), 1)

    total_income = total_income[idx_sort].cumsum() / total_income.sum()
    total_income = total_income.reshape(len(total_income), 1)

    # to pandas
    data = np.hstack([weights, total_income])
    columns = ["Equality", "Income"]
    index = pd.Index(weights.round(3).squeeze())
    res = pd.DataFrame(data=data, columns=columns, index=index)
    res.index.name = "x"

    return res


def gini(income, weights=None, data=None, sort=True):
    """Calculate Gini's index.

    The Gini coefficient (sometimes expressed as a Gini ratio or a
    normalized Gini index) is a measure of statistical dispersion intended to
    represent the income or wealth distribution of a nation's residents, and is
    the most commonly used measure of grouped. It was developed by Corrado
    Gini.
    The Gini coefficient measures the grouped among values of a frequency
    distribution (for example, levels of income). A Gini coefficient of zero
    expresses perfect equality, where all values are the same (for example,
    where everyone has the same income). A Gini coefficient of 1 (or 100%)
    expresses maximal grouped among values (e.g., for a large number of
    people, where only one person has all the income or consumption, and all
    others have none, the Gini coefficient will be very nearly one).

    Parameters
    ----------
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


def atkinson(income, weights=None, data=None, e=0.5) -> float:
    """Calculate atkinson index.

    More precisely labelled a family of income grouped measures, the
    theoretical range of Atkinson values is 0 to 1, with 0 being a state of
    equal distribution.

    An intuitive interpretation of this index is possible: Atkinson values can
    be used to calculate the proportion of total income that would be required
    to achieve an equal level of social welfare as at present if incomes were
    perfectly distributed.

    For example, an Atkinson index value of 0.20 suggests that we could achieve
    the same level of social welfare with only 1 – 0.20 = 80% of income. The
    theoretical range of Atkinson values is 0 to 1, with 0 being a state of
    equal distribution.

    Parameters
    ----------
    income : array or str
        If `data` is none `income` must be an 1D-array, when `data` is a
        pd.DataFrame, you must pass the name of income variable as string.
    weights : array or str, optional
        If `data` is none `weights` must be an 1D-array, when `data` is a
        pd.DataFrame, you must pass the name of weights variable as string.
    e : int, optional
        Epsilon parameter interpreted by atkinson index as grouped adversion,
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
    https://en.wikipedia.org/w/index.php?title=Atkinson_index

    TODO
    ----
    - Implement: CALCULATING INCOME DISTRIBUTION INDICES FROM MICRO-DATA
      http://www.jstor.org/stable/41788716
    - The results has difference with stata, maybe have a bug.
    """
    if (income is None) and (data is None):
        raise ValueError("Must pass at least one of both `income` or `df`")

    income, weights = utils.extract_values(data, income, weights)
    weights = utils.not_empty_weights(weights, income)

    # not-null condition
    income, weights = utils.not_null_condition(income, weights)

    # not-empty condition
    if len(income) == 0:
        return 0

    # auxiliar variables: mean and distribution
    mu = mean(variable=income, weights=weights)
    f_i = np.atleast_1d(weights / sum(weights))  # density function

    # main calc
    if e == 1:
        return 1 - np.power(np.e, np.sum(f_i * np.log(income) - np.log(mu)))
    elif e >= 0 or e < 1:
        return 1 - np.power(
            np.sum(f_i * np.power(income / mu, 1 - e)), 1 / (1 - e)
        )
    else:
        assert (e < 0) or (e > 1), "Not valid e value,  0 ≤ e ≤ 1"
        return np.nan


def kakwani(tax, income_pre_tax, weights=None, data=None):
    """Calculate Kakwani's index.

    The Kakwani (1977) index of tax progressivity is defined as twice the
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
    Jenkins, S. (1988). Calculating income distribution indices from
    micro-data. National Tax Journal. http://doi.org/10.2307/41788716
    """
    # main calc
    c_t = concentration(data=data, income=tax, weights=weights, sort=True)
    g_y = concentration(
        data=data, income=income_pre_tax, weights=weights, sort=True
    )
    return c_t - g_y


def reynolds_smolensky(
    income_pre_tax, income_post_tax, weights=None, data=None
):
    """Calculate Reynolds-Smolensky's index.

    The Reynolds-Smolensky (1977) index of the redistributive effect of
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
    Jenkins, S. (1988). Calculating income distribution indices from
    micro-data. National Tax Journal. http://doi.org/10.2307/41788716
    """
    g_y = concentration(data=data, income=income_post_tax, weights=weights)
    g_x = concentration(data=data, income=income_pre_tax, weights=weights)
    return g_x - g_y


def theil(income, weights=None, data=None):
    """Calculate Theil's index.

    The Theil index is a statistic primarily used to measure economic
    grouped and other economic phenomena. It is a special case of the
    generalized entropy index. It can be viewed as a measure of redundancy,
    lack of diversity, isolation, segregation, grouped, non-randomness, and
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
    if data is not None:
        income, weights = utils.extract_values(data, income, weights)
    else:
        income = income.copy()

        if weights is None:
            weights = utils.not_empty_weights(weights, like=income)
        else:
            weights = weights.copy()
    income, weights = utils.not_null_condition(income, weights)

    # variables needed
    mu = mean(variable=income, weights=weights)
    f_i = utils.normalize(weights)
    return np.sum((f_i * income / mu) * np.log(income / mu))


def avg_tax_rate(total_tax, total_base, weights=None, data=None):
    """Calculate average tax rate.

    This function compute the average tax rate given a base income and a total
    tax.

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
    Panel de declarantes de IRPF 1999-2007: Metodología, estructura y
    variables. (2011).
    Panel de declarantes de IRPF 1999-2007: Metodología, estructura y
    variables. Documentos.
    """
    if (
        isinstance(total_base, (np.ndarray))
        or not isinstance(total_base, (list))
        and not isinstance(total_base, (str))
    ):
        n_cols = total_base.shape[1]
    elif isinstance(total_base, list):
        n_cols = len(total_base)
    else:
        n_cols = 1
    numerator = mean(data=data, variable=total_tax, weights=weights)
    denominator = mean(data=data, variable=total_base, weights=weights)
    # main calc
    res = numerator / denominator

    if data is not None:
        base_name = total_base
        tax_name = total_tax
    else:
        base_name = ["base" % i for i in range(n_cols)]
        tax_name = ["tax_%s" % i for i in range(n_cols)]

    names = ["_".join([t, b]) for t, b in zip(tax_name, base_name)]
    res = pd.Series(res, index=names)
    return res


def top_rest(income, weights=None, data=None, top_percentage=10.0):
    """Calculate the 10:90 Ratio.

    Calculates the quotient between the number of contributions from the top
    10% of contributors divided by the number contributions made by the other
    90%. The ratio is 1 if the total contributions by the top contributors are
    equal to the cotnributions made by the rest; less than zero if the top 10%
    contributes less than the rest; and greater that 1 if the top 10%
    contributes more than the other ninety percent.

    Parameters
    ----------
    income : array-like or str
        This variable represent tax payment of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    weights : array-like or str
        This variable represent weights of each person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
        All-ones by default
    data : pandas.DataFrame
        This variable is a DataFrame that contains all data required in it's
        columns.
    top_percentage : float
        The richest x percent to consider. (10 percent by default)
        It must be a number between 0 and 100

    Returns
    -------
    ratio : float

    References
    ----------
    Participation Inequality in Wikis: A Temporal Analysis Using WikiChron.
    Serrano, Abel & Arroyo, Javier & Hassan, Samer. (2018).
    DOI: 10.1145/3233391.3233536.
    """
    if data is not None:
        income, weights = utils.extract_values(data, income, weights)
    else:
        income = income.copy()
        weights = np.ones_like(income) if weights is None else weights.copy()

    # Small shortcut to avoid divide by zero below
    if income.size <= 1:
        return np.nan

    income, weights = utils._sort_values(income, weights)

    # variables needed
    weights = utils.normalize(weights)
    cumw = np.cumsum(weights)
    ftosearch = 1 - top_percentage / 100
    k = np.searchsorted(cumw, ftosearch, side='right')
    f_i = np.atleast_1d(income*weights)

    t = np.sum(f_i[k:])
    r = np.sum(f_i[:k])

    # Correction
    if k > 0:
        error = (ftosearch - cumw[k-1]) * income[k]
        t -= error
        r += error

    return t / r


def hoover(income, weights=None, data=None):
    """Calculate Hoover index.

    The Hoover index, also known as the Robin Hood index or the Schutz index,
    is a measure of income metrics. It is equal to the portion of the total
    community income that would have to be redistributed (taken from the richer
    half of the population and given to the poorer half) for there to be income
    uniformity.

    Formula:

        H = 1/2 sum_i( |xi - mu| ) / sum_i(xi)

    Parameters
    ----------
    income : array-like or str
        This variable represent tax payment of person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    weights : array-like or str
        This variable represent weights of each person, if pass array-like
        then data must be None, else you pass str-name column in `data`.
    data : pandas.DataFrame
        This variable is a DataFrame that contains all data required in it's
        columns.

    Returns
    -------
    hoover : float

    References
    ----------
    Hoover index : https://en.wikipedia.org/wiki/Hoover_index
    """
    if data is not None:
        income, weights = utils.extract_values(data, income, weights)
    else:
        income = income.copy()
        if weights is None:
            weights = utils.not_empty_weights(weights, like=income)
        else:
            weights = weights.copy()

    income, weights = utils.not_null_condition(income, weights)

    # variables needed
    mu = mean(variable=income, weights=weights)
    f_i = utils.normalize(weights)
    xi = f_i * income

    # main calc
    h = np.sum(abs(xi - mu)) * 0.5 / sum(xi)

    return h
