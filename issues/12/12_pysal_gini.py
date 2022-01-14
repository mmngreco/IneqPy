import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import ineqpy
import asciiplotlib as apl

from scipy.stats import lognorm
from pysal.explore.inequality.gini import Gini


def get_gini(size):
    y = lognorm.rvs(s=1, size=size)

    # comparision
    gini_ineqpy = ineqpy.gini(income=y)
    gini_pysal = Gini(y).g
    gini_diff = abs(gini_ineqpy - gini_pysal)
    return n, gini_ineqpy, gini_pysal, gini_diff


def plot_line(x, y):
    fig = apl.figure()
    fig.plot(x, y, label=y.name, width=80, height=20)
    fig.show()

# makes calculation for each series size
store_res = [get_gini(size=n) for n in [100, 1000, 5000, 10000, 20000, 50000]]
# present results
df = pd.DataFrame(store_res, columns=["n","ineqpy", "pysal", "diff"])
print(df)

# plotting in plain text
x = df.iloc[:, 0]
for icol in [1,2,3]:
    y = df.iloc[:, icol]
    print(y.name)
    plot_line(x, y)


