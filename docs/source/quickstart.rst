
==========
Quickstart
==========

We generate random weighted data to show how ineqpy works. The variables 
simulate being:

.. code::

    x : Income
    w : Weights

.. code-block:: python

   >>> x, w = ineq.utils.generate_data_to_test((60,90))


To test with classical statistics we generate:

.. code::

    x_rep : Income values replicated w times each one.
    w_rep : Ones column.

.. code-block:: python

   >>> x_rep, w_rep = ineq.utils.repeat_data_from_weighted(x, w)


Additional information:

.. code::

    np : numpy package
    sp : scipy package
    pd : pandas package
    gsl_stat : GNU Scientific Library written in C.
    ineq : IneqPy



STATISTICS
==========

MEAN
----

.. code-block:: python

   >>> np.mean(x_rep)       = 527.663398693
   >>> ineq.mean(x, w)      = 527.663398693
   >>> gsl_stat.wmean(w, x) = 527.6633986928105



VARIANCE
--------


.. code-block:: python

   >>> np.var(x_rep)                = 112609.556634
   >>> ineq.var(x, w)               = 112609.556634
   >>> ineq_stat.wvar(x, w, kind=1) = 112609.556634
   >>> ineq_stat.wvar(x, w, kind=2) = 112978.768295
   >>> gsl_stat.wvariance(w, x)     = 114731.76842139623
   >>> ineq_stat.wvar(x, w, kind=3) = 114731.768421



COVARIANCE
----------

.. code-block:: python

   >>> np.cov(x_rep, x_rep)            =  [[ 112978.7682953  112978.7682953]
   [ 112978.7682953  112978.7682953]]
   >>> ineq_stat.wcov(x, x, w, kind=1) =  112609.556634
   >>> ineq_stat.wcov(x, x, w, kind=2) =  4.94065645841e-324
   >>> ineq_stat.wcov(x, x, w, kind=3) =  9.88131291682e-324



SKEWNESS
--------


.. code-block:: python

   >>> gsl_stat.wskew(w, x) =  -0.0285099856045751
   >>> sp_stat.skew(x_rep)  =  -0.02931970907039857
   >>> ineq.skew(x, w)      =  -0.0293197090704



KURTOSIS
--------


.. code-block:: python

   >>> sp_stat.kurtosis(x_rep)  =  -1.5386564632396265
   >>> gsl_stat.wkurtosis(w, x) =  -1.5922178801295013
   >>> ineq.kurt(x, w) - 3      =  -1.53865646324


PERCENTILES
-----------


.. code-block:: python

   >>> ineq_stat.percentile(x, w, 50) =  494
   >>> np.percentile(x_rep, 50)       =  494.0
   >>> ineq_stat.percentile(x, w, 25) =  229
   >>> np.percentile(x_rep, 25)       =  229.0
   >>> ineq_stat.percentile(x, w, 75) =  849
   >>> np.percentile(x_rep, 75)       =  849.0
   >>> ineq_stat.percentile(x, w, 10) =  70
   >>> np.percentile(x_rep, 10)       =  70.0
   >>> ineq_stat.percentile(x, w, 90) =  962
   >>> np.percentile(x_rep, 90)       =  962.0


Another way to use this is through the API module as shown below:

API MODULE
==========


.. code-block:: python

   >>> data = svy(data=data, columns=columns, weights='w')
   >>> data.head()
        x  w
   0  943  2
   1  271  8
   2  974  5
   3  509  5
   4  887  8

   >>> data.weights = w


.. code-block:: python

   >>> df.mean(main_var)       = 527.663398693
   >>> df.percentile(main_var) = 494
   >>> df.var(main_var)        = 112609.556634
   >>> df.skew(main_var)       = -0.0293197090704
   >>> df.kurt(main_var)       = 1.46134353676
   >>> df.gini(main_var)       = 0.369087636611
   >>> df.atkinson(main_var)   = 0.137253740458
   >>> df.theil(main_var)      = 0.237290929519

