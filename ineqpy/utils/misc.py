import pandas as pd
import numpy as np

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


def _check_weights(weights, as_of):
    return weights.copy() if weights is not None else np.ones(len(as_of))


def _not_null_condition(income, weights):
    income = income.copy()
    weights = weights.copy()

    if np.any(income <= 0):
        mask = income > 0
        income = income[mask]
        if weights is not None:
            weights = weights[mask]

    return income, weights


def _sort_values(by, pair):
    idx_sort = np.argsort(by)
    by = by[idx_sort]
    pair = pair[idx_sort]
    return by, pair

def _clean_nans_values(this, pair):
    if np.any(np.isnan(this)):
        idx = ~np.isnan(this)
        this = this[idx]
        pair = pair[idx]
    return this, pair
