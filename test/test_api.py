import numpy as np

svy = ineqpy.api.Survey
data = np.random.randint(0,100, (int(1e3),3))
w = np.random.randint(1,10, int(1e3))
data = np.c_[data, w]
columns = list('abcw')

df = svy(data, columns=columns, weights='w')

df
df.weights
df.mean('a')
df.var('a')
df.skew('a')
df.kurt('a')
df.gini('a')
df.atkinson('a')
df.theil('a')
df.percentile('a')
