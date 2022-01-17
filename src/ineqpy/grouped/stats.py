"""Stats' module."""
from .. import utils
from .._statistics import c_moment, mean, std_moment


def variance_hat_group(data=None, variable="x", weights="w", group="h"):
    """Calculate variance.

    Data a DataFrame calculates the sample variance for each stratum. The
    objective of this function is to make it easy to calculate the moments of
    the distribution that follows an estimator, eg. Can be used to calculate
    the variance that follows the mean.

    Parameters
    ----------
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

    Todo
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
        data = utils._to_df(x=variable, weights=weights, group=group)
        variable = "x"
        weights = "weights"
        group = "group"

    def v(df):
        r"""Calculate the variance of each stratum `h`.

        Parameters
        ----------
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


def moment_group(data=None, variable="x", weights="w", group="h", order=2):
    """Calculate the asymmetry of each `h` stratum.

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
        data = utils._to_df(x=variable, weights=weights, group=group)
        variable = "x"
        weights = "weights"
        group = "group"

    def mh(df, weights=weights):
        x = df[variable].copy().values
        weights = utils.not_empty_weights(weights, x)
        Nh = df.loc[:, weights].sum()
        fpc = 1 - (len(df) / Nh)
        ddof = 1 if len(df) > 1 else 0
        stdm = std_moment(variable=x, weights=weights, order=order, ddof=ddof)
        return (Nh ** order) * fpc * stdm / len(df)

    return data.groupby(group).apply(mh)


def quasivariance_hat_group(
    data=None, variable=None, weights=None, group=None
):
    """Calculate quasivariance.

    Sample variance of `variable`, calculated as the second-order central
    moment.

    Parameters
    ----------
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
    ----------
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
        data = utils._to_df(x=variable, weights=weights)
        variable = "x"
        weights = "weights"

    def sd(df):
        x = variable
        return c_moment(variable=x, weights=weights, param=mean(x))

    return data.groupby(group).apply(sd)
