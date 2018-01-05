import numpy as np
from pygsl import statistics as gsl_stat
from scipy import stats as sp_stat

import ineqpy as ineq
from ineqpy import _statistics as ineq_stat

# Generate random data
x, w = ineq.utils.generate_data_to_test((60,90))
# Replicating weights
x_rep, w_rep = ineq.utils.repeat_data_from_weighted(x, w)
svy = ineq.api.Survey

print(
"""
We generate random weighted data to show how ineqpy works. The variables 
simulate being:
    x : Income
    w : Weights

```python
>>> x, w = ineq.utils.generate_data_to_test((60,90))
```

To test with classical statistics we generate:
    x_rep : Income values replicated w times each one.
    w_rep : Ones column.

```python
>>> x_rep, w_rep = ineq.utils.repeat_data_from_weighted(x, w)
```

Additional information:
    np : numpy package
    sp : scipy package
    pd : pandas package
    gsl_stat : GNU Scientific Library written in C.
    ineq : IneqPy
"""
)


print(
"""

STATISTICS
==========

MEAN CALCULATIONS
-----------------

"""
)

print('```python')
print('>>> np.mean(x_rep)'.ljust(24), '=', np.mean(x_rep))
print('>>> ineq.mean(x, w)'.ljust(24), '=', ineq.mean(x, w))
print('>>> gsl_stat.wmean(w, x)'.ljust(24), '=', gsl_stat.wmean(w, x))
print('```')

# %timeit ineq.mean(None, x, w)
# %timeit gsl_stat.wmean(w, x)
# %timeit ineq_stat.mean(x, w)

print(
"""

VARIANCE CALCULATIONS
---------------------

"""
)

np_var = np.var(x_rep)
inq_var = ineq.var(x, w)
wvar_1 = ineq_stat.wvar(x, w, 1)  # population variance
wvar_2 = ineq_stat.wvar(x, w, 2)  # sample frequency variance
gsl_wvar = gsl_stat.wvariance(w, x)
wvar_3 = ineq_stat.wvar(x, w, 3)  # sample reliability variance

print('```python')
print('>>> np.var(x_rep)'.ljust(32), '=', np_var)
print('>>> ineq.var(x, w)'.ljust(32), '=', inq_var)
print('>>> ineq_stat.wvar(x, w, kind=1)'.ljust(32), '=', wvar_1)
print('>>> ineq_stat.wvar(x, w, kind=2)'.ljust(32), '=', wvar_2)
print('>>> gsl_stat.wvariance(w, x)'.ljust(32), '=', gsl_wvar)
print('>>> ineq_stat.wvar(x, w, kind=3)'.ljust(32), '=', wvar_3)
print('```')

print(
"""

COVARIANCE CALCULATIONS
-----------------------

"""
)

np_cov = np.cov(x_rep, x_rep)
ineq_wcov1 = ineq_stat.wcov(x, x, w, 1)
ineq_wcov2 = ineq_stat.wcov(x, x, w, 2)
ineq_wcov3 = ineq_stat.wcov(x, x, w, 3)

print('```python')
print('>>> np.cov(x_rep, x_rep)'.ljust(35), '= ', np_cov)
print('>>> ineq_stat.wcov(x, x, w, kind=1)'.ljust(35), '= ', ineq_wcov1)
print('>>> ineq_stat.wcov(x, x, w, kind=2)'.ljust(35), '= ', ineq_wcov2)
print('>>> ineq_stat.wcov(x, x, w, kind=3)'.ljust(35), '= ', ineq_wcov3)
print('```')
print(
"""

SKEWNESS CALCULATIONS
---------------------

"""
)

gsl_wskew = gsl_stat.wskew(w, x)
sp_skew =  sp_stat.skew(x_rep)
ineq_skew =  ineq.skew(x, w)

print('```python')
print('>>> gsl_stat.wskew(w, x)'.ljust(24), '= ', gsl_wskew)
print('>>> sp_stat.skew(x_rep)'.ljust(24), '= ', sp_skew)
print('>>> ineq.skew(x, w)'.ljust(24), '= ', ineq_skew)
print('```')

# %timeit gsl_stat.wskew(w, x)
# %timeit sp_stat.skew(x_rep)
# %timeit ineq.skew(None, x, w)

print(
"""

KURTOSIS CALCULATIONS
---------------------

"""
)

sp_kurt = sp_stat.kurtosis(x_rep)
gsl_wkurt = gsl_stat.wkurtosis(w, x)
ineq_kurt = ineq.kurt(x, w) - 3
print('```python')
print('>>> sp_stat.kurtosis(x_rep)'.ljust(28), '= ', sp_kurt)
print('>>> gsl_stat.wkurtosis(w, x)'.ljust(28), '= ', gsl_wkurt)
print('>>> ineq.kurt(x, w) - 3'.ljust(28), '= ', ineq_kurt)
print('```')
# %timeit sp_stat.kurtosis(x_rep)
# %timeit gsl_stat.wkurtosis(w, x)
# %timeit ineq.kurt(None, x, w) - 3

print(
"""
PERCENTILES CALCULATIONS
------------------------

"""
)
q = 50
ineq_perc_50 = ineq_stat.percentile(x, w, q)
np_perc_50 = np.percentile(x_rep, q)
print('```python')
print('>>> ineq_stat.percentile(x, w, %s)'.ljust(34) % q, '= ', ineq_perc_50)
print('>>> np.percentile(x_rep, %s)'.ljust(34) % q, '= ', np_perc_50)

q = 25
ineq_perc_25 = ineq_stat.percentile(x, w, q)
np_perc_25 = np.percentile(x_rep, q)
print('>>> ineq_stat.percentile(x, w, %s)'.ljust(34) % q, '= ', ineq_perc_25)
print('>>> np.percentile(x_rep, %s)'.ljust(34) % q, '= ', np_perc_25)

q = 75
ineq_perc_75 = ineq_stat.percentile(x, w, q)
np_perc_75 = np.percentile(x_rep, q)
print('>>> ineq_stat.percentile(x, w, %s)'.ljust(34) % q, '= ', ineq_perc_75)
print('>>> np.percentile(x_rep, %s)'.ljust(34) % q, '= ', np_perc_75)

q = 10
ineq_perc_10 = ineq_stat.percentile(x, w, q)
np_perc_10 = np.percentile(x_rep, q)
print('>>> ineq_stat.percentile(x, w, %s)'.ljust(34) % q, '= ', ineq_perc_10)
print('>>> np.percentile(x_rep, %s)'.ljust(34) % q, '= ', np_perc_10)

q = 90
ineq_perc_90 = ineq_stat.percentile(x, w, q)
np_perc_90 = np.percentile(x_rep, q)
print('>>> ineq_stat.percentile(x, w, %s)'.ljust(34) % q, '= ', ineq_perc_90)
print('>>> np.percentile(x_rep, %s)'.ljust(34) % q, '= ', np_perc_90)
print('```')

print(
"""
Another way to use this is through the API module as shown below:

API MODULE
==========

"""
)

data = np.c_[x, w]
columns = list('xw')

df = svy(data=data, columns=columns, weights='w')
print('```python')
print(">>> data = svy(data=data, columns=columns, weights='w')")
print(">>> data.head()")
print(df.head())
print('')
print('>>> data.weights =', df.weights)
print('```')

main_var = 'x'
# df.mean(main_var)
# df.var(main_var)
# df.skew(main_var)
# df.kurt(main_var)
# df.gini(main_var)
# df.atkinson(main_var)
# df.theil(main_var)
# df.percentile(main_var)

print('```python')
print('>>> df.mean(main_var)'.ljust(27), '=', df.mean(main_var))
print('>>> df.percentile(main_var)'.ljust(27), '=', df.percentile(main_var))
print('>>> df.var(main_var)'.ljust(27), '=', df.var(main_var))
print('>>> df.skew(main_var)'.ljust(27), '=', df.skew(main_var))
print('>>> df.kurt(main_var)'.ljust(27), '=', df.kurt(main_var))
print('>>> df.gini(main_var)'.ljust(27), '=', df.gini(main_var))
print('>>> df.atkinson(main_var)'.ljust(27), '=', df.atkinson(main_var))
print('>>> df.theil(main_var)'.ljust(27), '=', df.theil(main_var))
print('```')