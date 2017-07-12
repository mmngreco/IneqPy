import numpy as np

from ineqpy import c_moment, std_moment
from utils.msic import _to_df


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