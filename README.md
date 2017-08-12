[![Build Status](https://travis-ci.org/mmngreco/IneqPy.svg?branch=master)](https://travis-ci.org/mmngreco/IneqPy)

IneqPy Package
==============

This package provides statistics to do a properly quantitative analysis of
inequality. Among the estimators provided by this package you can find:


| Main Statistics                   | Inequality Indicators |
| :--------------                   | :-------------------- |
| Weighted Mean                     | Weighted Gini         |
| Weighted Variance                 | Weighted Atkinson     |
| Weighted Coefficient of variation | Weighted Theil        |
| Weighted Kurtosis                 | Weighted Kakwani      |
| Weighted Skewness                 | Weighted Lorenz curve |


-----------
First-steps
-----------

- Installation
- Examples


Install
-------

```bash
git clone https://github.com/mmngreco/IneqPy.git
cd IneqPy
pip install .
```

--------
Examples
--------

Some examples of how use this package:
Data of example:

```python
>>> import pandas as pd
>>> import numpy as np
>>> import ineqpy
>>> d

             renta   factor
0        -13004.12   1.0031
89900    141656.97   1.4145
179800     1400.38   4.4122
269700   415080.96   1.3295
359600    69165.22   1.3282
449500     9673.83  19.4605
539400    55057.72   1.2923
629300     -466.73   1.0050
719200     3431.86   2.2861
809100      423.24   1.1552
899000        0.00   1.0048
988900     -344.41   1.0028
1078800   56254.09   1.2752
1168700   60543.33   2.0159
1258600    2041.70   2.7381
1348500     581.38   7.9426
1438400   55646.05   1.2818
1528300       0.00   1.0281
1618200   69650.24   1.2315
1708100   -2770.88   1.0035
1798000    4088.63   1.1256
1887900       0.00   1.0251
1977800   10662.63  28.0409
2067700    3281.95   1.1670

```

----------------------
Descriptive statistics
----------------------

```python

    ineqpy.mean(x=d.renta, weights=d.factor)
    20444.700666031338
    ineqpy.variance(x=d.renta, weights=d.factor)
    2982220948.7413292
    x, w = d.renta.values, d.factor.values
```

> Note that the standardized moment for order `n`, retrieve the value in that
  column:


| `n` | value     |
|:---:|:---------:|
| 1   | 0         |
| 2   | 1         |
| 3   | Skew      |
| 4   | Kurtosis  |


A helpful table of interpretation of the moments

```python
>>> ineqpy.std_moment(x, w, 1)  # = 0
2.4624948200717338e-17
>>> ineqpy.std_moment(x, w, 2)  # = 1
1.0
>>> ineqpy.std_moment(x, w, 3)  # = skew
5.9965055750379426
>>> ineqpy.skew(x, w)
5.9965055750379426
>>> ineqpy.std_moment(x, w, 4)  # = kurtosis
42.319928851703004
>>> ineqpy.kurt(x, w)
42.319928851703004
```
---------------------
Inequality estimators
---------------------

```python
# pass a pandas.DataFrame and inputs as strings
>>> ineqpy.gini(df=d, income='renta', weights='factor')
0.76739136365917116
# you can pass arrays too
>>> ineqpy.gini(income=d.renta.values, weights=d.factor.values)
0.76739136365917116
```
