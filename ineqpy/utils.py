import numpy as np
import pandas as pd


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
    w = normalize_weights(
            weights.copy() if weights is not None else np.ones(len(as_of))
    )
    return w


def _not_null_condition(income, weights):
    income = income.copy()
    weights = weights.copy()

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


def generate_data_to_test(n_sample_range=(20,100)):
    N_sample = np.random.randint(*n_sample_range)
    weighted_x = np.random.randint(0, 1000, N_sample)
    weights = np.random.randint(1, 9, N_sample)
    repeated_x = np.array([])
    repeated_w = np.array([])

    for xi, wi in zip(weighted_x, weights):
        repeated_x = np.append(repeated_x, np.repeat(xi, wi))
        repeated_w = np.append(repeated_w, np.ones(wi))

    return (weighted_x, weights), (repeated_x, repeated_w)


def normalize_weights(weights):
    return weights / np.sum(weights)


def _extract_values(data, variable, weights):
    variable = data[variable].values
    weights = data[weights].values
    return variable, weights
