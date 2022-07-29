![Build Status](https://github.com/mmngreco/ineqpy/actions/workflows/python-package.yml/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1419582.svg)](https://doi.org/10.5281/zenodo.1419582)


# IneqPy's Package

This package provides statistics to carry on inequality's analysis. Among the
estimators provided by this package you can find:


| Main Statistics                   | Inequality Indicators |
| :--------------                   | :-------------------- |
| Weighted Mean                     | Weighted Gini         |
| Weighted Variance                 | Weighted Atkinson     |
| Weighted Coefficient of variation | Weighted Theil        |
| Weighted Kurtosis                 | Weighted Kakwani      |
| Weighted Skewness                 | Weighted Lorenz curve |


## Installation

```bash
pip install ineqpy
# or from github's repo
pip install git+https://github.com/mmngreco/IneqPy.git
```

## What you can find

Some examples of how use this package:

```python
>>> import pandas as pd
>>> import numpy as np
>>> import ineqpy
>>> d = load_data()  # dataframe
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

### Descriptive statistics

```python
>>> ineqpy.mean(variable=d.renta, weights=d.factor)
20444.700666031338
>>> ineqpy.var(variable=d.renta, weights=d.factor)
2982220948.7413292
>>> x, w = d.renta.values, d.factor.values
```

> Note that the standardized moment for order `n`, retrieve the value in that
> column:
>
> | `n` | value     |
> |:---:|:---------:|
> | 1   | 0         |
> | 2   | 1         |
> | 3   | Skew      |
> | 4   | Kurtosis  |

A helpful table of interpretation of the moments

```python
>>> ineqpy.std_moment(variable=x, weights=w, order=1)  # ~= 0
2.4624948200717338e-17
>>> ineqpy.std_moment(variable=x, weights=w, order=2)  # = 1
1.0
>>> ineqpy.std_moment(variable=x, weights=w, order=3)  # = skew
5.9965055750379426
>>> ineqpy.skew(variable=x, weights=w)
5.9965055750379426
>>> ineqpy.std_moment(variable=x, weights=w, order=4)  # = kurtosis
42.319928851703004
>>> ineqpy.kurt(variable=x, weights=w)
42.319928851703004
```

### Inequality's estimators

```python
# pass a pandas.DataFrame and inputs as strings
>>> ineqpy.gini(data=d, income='renta', weights='factor')
0.76739136365917116
# you can pass arrays too
>>> ineqpy.gini(income=d.renta.values, weights=d.factor.values)
0.76739136365917116
```

### More examples and comparison with other packages:

We generate random weighted data to show how ineqpy works. The variables
simulate being:

    x : Income
    w : Weights

To test with classical statistics we generate:

    x_rep : Income values replicated w times each one.
    w_rep : Ones column.

Additional information:

    np : numpy package
    sp : scipy package
    pd : pandas package
    gsl_stat : GNU Scientific Library written in C.
    ineq : IneqPy


#### Mean

```python
>>> np.mean(x_rep)       = 488.535714286
>>> ineq.mean(x, w)      = 488.535714286
>>> gsl_stat.wmean(w, x) = 488.5357142857143
```

#### Variance

```python
>>> np.var(x_rep)                = 63086.1364796
>>> ineq.var(x, w)               = 63086.1364796
>>> ineq_stat.wvar(x, w, kind=1) = 63086.1364796
>>> ineq_stat.wvar(x, w, kind=2) = 63247.4820972
>>> gsl_stat.wvariance(w, x)     = 63993.161585889124
>>> ineq_stat.wvar(x, w, kind=3) = 63993.1615859
```

#### Covariance

```python
>>> np.cov(x_rep, x_rep)            =  [[ 63247.48209719  63247.48209719]
 [ 63247.48209719  63247.48209719]]
>>> ineq_stat.wcov(x, x, w, kind=1) =  63086.1364796
>>> ineq_stat.wcov(x, x, w, kind=2) =  4.94065645841e-324
>>> ineq_stat.wcov(x, x, w, kind=3) =  9.88131291682e-324
```

#### Skewness

```python
>>> gsl_stat.wskew(w, x) =  -0.05742668111416989
>>> sp_stat.skew(x_rep)  =  -0.058669605967865954
>>> ineq.skew(x, w)      =  -0.0586696059679
```

#### Kurtosis

```python
>>> sp_stat.kurtosis(x_rep)  =  -0.7919389201857214
>>> gsl_stat.wkurtosis(w, x) =  -0.8540884810553052
>>> ineq.kurt(x, w) - 3      =  -0.791938920186
```

#### Percentiles

```python
>>> ineq_stat.percentile(x, w, 25) =  293
>>> np.percentile(x_rep, 25)       =  293.0

>>> ineq_stat.percentile(x, w, 50) =  526
>>> np.percentile(x_rep, 50)       =  526.0

>>> ineq_stat.percentile(x, w, 90) =  839
>>> np.percentile(x_rep, 90)       =  839.0
```

Another way to use this is through the API module as shown below:

## API's module

Using API's module:

```python
>>> data = Survey(data=data, columns=columns, weights='w')
>>> data.df.head()
     x  w
0  111  3
1  711  4
2  346  4
3  667  1
4  944  1
```

### Statistics

```python
>>> data.weights = w
>>> df.mean(main_var)       = 488.535714286
>>> df.percentile(main_var) = 526
>>> df.var(main_var)        = 63086.1364796
>>> df.skew(main_var)       = -0.0586696059679
>>> df.kurt(main_var)       = 2.20806107981
>>> df.gini(main_var)       = 0.298494329293
>>> df.atkinson(main_var)   = 0.0925853855635
>>> df.theil(main_var)      = 0.156137490566
```
