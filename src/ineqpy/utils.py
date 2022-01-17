"""Useful functions that make easier develop other functions."""

import numpy as np
import pandas as pd


def _to_df(*args, **kwargs) -> pd.DataFrame:
    res = pd.DataFrame()

    if args != ():
        res = pd.DataFrame([*args]).T

    if kwargs is not None:
        df = pd.DataFrame.from_dict(kwargs, orient="columns")
        if res.empty:
            res = df
        else:
            res = pd.concat([res, df], axis=1)

    return res


def _apply_to_df(func, df, x, weights, *args, **kwargs):
    """Generalize main arguments as Series of a pd.Dataframe.

    Parameters
    ----------
    func : function
        Function to convert his arguments in Series of an Dataframe.
    df : pandas.Dataframe
        DataFrame whats contains the Series `x_name` and `w_name`.
    x_name : str
        Name of the column in `df`.
    weights_name : str
        Name of the column in `df`.

    Returns
    -------
    return : func return
        It's depends of func output type.
    """
    return func(df[x], df[weights], *args, **kwargs)


def not_empty_weights(weights, like):
    """Create weights.

    Create normalized weight if it's None use like to create it.

    Parameters
    ----------
    income, like : array-like

    Returns
    -------
    weights : array-like
        Filtered array-like.

    See Also
    --------
    normalize
    """
    if weights is not None:
        return normalize(weights.copy())

    return np.ones_like(like)


def not_null_condition(income, weights):
    """Filter not null condition.

    If a negative value is found in the incomes it will dropped.

    Parameters
    ----------
    income, weights : array-like

    Returns
    -------
    income, weights : array-like
        Filtered array-like.
    """
    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        if weights is not None:
            weights = weights[mask]

    return income, weights


def _sort_values(values, partner):
    idx_sort = np.argsort(values, axis=0).squeeze()
    values = values[idx_sort]
    partner = partner[idx_sort]
    return values, partner


def _clean_nans_values(this, pair):
    if np.any(np.isnan(this)):
        idx = ~np.isnan(this)
        this = this[idx]
        pair = pair[idx]
    return this, pair


def normalize(this):
    """Normalize data by the sum.

    Parameters
    ----------
    this : array-like

    Returns
    -------
    out : array-like
    """
    return this / np.sum(this)


def extract_values(data, variable, weights):
    """Extract values.

    Parameters
    ----------
    data : pandas.DataFrame
    variable : str
    weights : str

    Returns
    -------
    variable, weights : array-like
    """
    if data is not None:
        variable = data.loc[:, variable].values
        weights = not_empty_weights(
            data.loc[:, weights].values, like=variable
        )
    return variable, weights


def repeat_data_from_weighted(x, w):
    """Generate data data (not sampled) from weights.

    Parameters
    ----------
    x, w : array-like

    Returns
    -------
    repeated_x, repeated_w : np.array
    """
    if isinstance(w[0], float):
        raise NotImplementedError

    repeated_x = np.array([])
    repeated_w = np.array([])

    for xi, wi in zip(x, w):
        repeated_x = np.append(repeated_x, np.repeat(xi, wi))
        repeated_w = np.append(repeated_w, np.ones(wi))

    return repeated_x, repeated_w


def generate_data_to_test(n_sample_range=(20, 100)):
    """Generate sampled data for testing.

    Parameters
    ----------
    n_sample_range : tuple[int, int]
        It's a shape, lenght and columns.

    Returns
    -------
    income, weights : np.array
    """
    N_sample = np.random.randint(*n_sample_range)
    weighted_x = np.random.randint(0, 1000, N_sample)
    weights = np.random.randint(1, 9, N_sample)
    return weighted_x, weights
