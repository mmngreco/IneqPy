"""inequality module."""
import pandas as pd
from typing import Union
import numpy as np

from .. import inequality as ineq
from .. import _statistics as stats
from .. import utils


def atkinson_group(
    data: pd.DataFrame = None,
    income: Union[str, pd.DataFrame, np.ndarray] = None,
    weights: Union[str, pd.DataFrame, np.ndarray] = None,
    group: Union[str, pd.DataFrame, np.ndarray] = None,
    e: float = 0.5,
):
    r"""Calculate atkinson index.

    The Atkinson index (also known as the Atkinson measure or Atkinson
    grouped measure) is a measure of income grouped developed by British
    economist Anthony Barnes Atkinson. The measure is useful in determining
    which end of the distribution contributed most to the observed grouped.The
    index is subgroup decomposable. This means that overall grouped in the
    population can be computed as the sum of the corresponding Atkinson indices
    within each group, and the Atkinson index of the group mean incomes.

    Parameters
    ----------
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
    if (weights is None) and (data is None):
        weights = utils.not_empty_weights(weights, income)

    if data is None:
        data = utils._to_df(income=income, weights=weights, group=group)
        income = "income"
        weights = "weights"
        group = "group"

    N = data.shape[0]

    def a_h(df):
        """Funtion alias to calculate atkinson from a DataFrame."""
        if df is None:
            raise ValueError

        inc = df[income].values
        w = df[weights].values
        atk = ineq.atkinson(income=inc, weights=w, e=e)
        out = atk * (len(df) / N)

        return out

    # main calc:
    data = data.copy()
    groupped = data.groupby(group)
    atk_by_group = groupped.apply(a_h)
    mu_by_group = groupped.apply(lambda d: stats.mean(d[income], d[weights]))
    out = atk_by_group.sum() + ineq.atkinson(income=mu_by_group.values)

    return out
