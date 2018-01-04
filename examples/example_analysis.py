import numpy as np
from IPython.extensions.autoreload import reload
from pygsl import statistics as gsl_stat
from scipy import stats as sp_stat

import ineqpy as ineq
from ineqpy import _statistics as ineq_stat

reload(ineq)
reload(ineq_stat)

# Generate random data
x, w = ineq.utils.generate_data_to_test((10,100))
np.c_[x, w]

# Replicating weights
x_rep, w_rep = ineq.utils.repeat_data_from_weighted(x, w)
np.c_[x_rep, w_rep]


# ==========
# STATISTICS
# ==========

# MEAN
# ====
np.mean(x_rep)
ineq.mean(x, w)
gsl_stat.wmean(w, x)

# %timeit ineq.mean(None, x, w)
# %timeit gsl_stat.wmean(w, x)
# %timeit ineq_stat.mean(x, w)

# VARIANCE
# ========
np.var(x_rep)
ineq.var(x, w)
ineq_stat.wvar(x, w, 1)  # population variance
ineq_stat.wvar(x, w, 2)  # sample frequency variance
# next both are equal
gsl_stat.wvariance(w, x)
ineq_stat.wvar(x, w, 3)  # sample reliability variance

# COVARIANCE
# ==========
np.cov(x_rep, x_rep)
ineq_stat.wcov(x,x,w, 1)
ineq_stat.wcov(x,x,w, 2)
ineq_stat.wcov(x,x,w, 3)

# SKEWNESS
# ========
gsl_stat.wskew(w, x)
sp_stat.skew(x_rep)
ineq.skew(x, w)

# %timeit gsl_stat.wskew(w, x)
# %timeit sp_stat.skew(x_rep)
# %timeit ineq.skew(None, x, w)

# KURTOSIS
# ========
sp_stat.kurtosis(x_rep)
gsl_stat.wkurtosis(w, x)
ineq.kurt(x, w) - 3

# %timeit sp_stat.kurtosis(x_rep)
# %timeit gsl_stat.wkurtosis(w, x)
# %timeit ineq.kurt(None, x, w) - 3

# PERCENTILES
# ===========
q = 50
ineq_stat.percentile(x, w, q)
np.percentile(x_rep, q)

q = 25
ineq_stat.percentile(x, w, q)
np.percentile(x_rep, q)

q = 75
ineq_stat.percentile(x, w, q)
np.percentile(x_rep, q)

q = 10
ineq_stat.percentile(x, w, q)
np.percentile(x_rep, q)

q = 90
ineq_stat.percentile(x, w, q)
np.percentile(x_rep, q)
