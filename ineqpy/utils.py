"""Useful functions that make easier develop other functions.
"""

import numpy as np
import pandas as pd


def _to_df(*args, **kwargs):
    res = None
    if args != ():
        res = pd.DataFrame([*args]).T

    if kwargs is not None:
        if res is not None:
            res = pd.concat(
                [res, pd.DataFrame.from_dict(kwargs, orient="columns")], axis=1
            )
        else:
            res = pd.DataFrame.from_dict(kwargs, orient="columns")
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


def not_empty_weights(weights, as_of):
    if not (weights is None):
        w = normalize(weights.copy())
    else:
        w = np.ones(len(as_of))
    return w


def not_null_condition(income, weights):

    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        if weights is not None:
            weights = weights[mask]

    return income, weights


def _sort_values(values, partner):
    idx_sort = np.argsort(values)
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
    return this / np.sum(this)


def extract_values(data, variable, weights):
    if data is not None:
        variable = data.loc[:, variable].values
        weights = not_empty_weights(
            data.loc[:, weights].values, as_of=variable
        )
    return variable, weights


def repeat_data_from_weighted(x, w):
    if isinstance(w[0], float):
        raise NotImplementedError

    repeated_x = np.array([])
    repeated_w = np.array([])

    for xi, wi in zip(x, w):
        repeated_x = np.append(repeated_x, np.repeat(xi, wi))
        repeated_w = np.append(repeated_w, np.ones(wi))

    return repeated_x, repeated_w


def generate_data_to_test(n_sample_range=(20, 100)):
    N_sample = np.random.randint(*n_sample_range)
    weighted_x = np.random.randint(0, 1000, N_sample)
    weights = np.random.randint(1, 9, N_sample)
    return weighted_x, weights
