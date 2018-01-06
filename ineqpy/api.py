"""This module extend pandas.DataFrames with the main functions from statistics and 
inequality modules.
"""

from . import _statistics
from . import inequality
from . import statistics
from . import utils
from .statistics import mean

from functools import partial
from types import MethodType

import inspect
import numpy as np
import pandas as pd


class Convey(pd.DataFrame):

    def __init__(self, data=None, index=None, columns=None, weights=None,
                 group=None, **kw):
        super(Convey, self).__init__(data=data, index=index, columns=columns,
                                     **kw)
        self.weights = weights
        self.group = group
        self._attach_method(statistics, self)
        self._attach_method(inequality, self)

    @property
    def _constructor(self):
        return Survey

    @staticmethod
    def _attach_method(module, instance):
        # get methods names contained in module
        res_names = list()
        res_methods = list()
        method_name_list = inspect.getmembers(module, inspect.isfunction)
        for method_name, func in method_name_list:
            # if method_name.startswith('_'): continue  # avoid private methods
            func = getattr(module, method_name)  # get function
            if 'weights' in inspect.signature(
                    func).parameters:  # replace weights variable
                func = partial(func, weights=instance.weights)
            # func = partial(func, data=instance.data)
            func = MethodType(func, instance)
            res_methods.append(func)
            res_names.append(method_name)
            setattr(instance, method_name, func)

    _constructor_sliced = pd.Series


class Survey(pd.DataFrame):

    def __init__(self, data=None, index=None, columns=None, weights=None,
                 group=None, **kw):
        super(Survey, self).__init__(data=data, index=index, columns=columns,
                                     **kw)
        self.weights = weights
        self.group = group

    @property
    def _constructor(self):
        return Survey

    _constructor_sliced = pd.Series


    def c_moment(self, variable=None, weights=None, order=2, param=None,
                 ddof=0):
        """Calculate the central moment of `x` with respect to `param` of order
        `n`, given the weights `w`.

        Parameters
        ----------
        data :
        variable : 1d-array
            Variable
        weights : 1d-array
            Weights
        order : int, optional
            Moment order, 2 by default (variance)
        param : int or array, optional
            Parameter for which the moment is calculated, the default is None,
            implies use the mean.
        ddof : int, optional
            Degree of freedom, zero by default.

        Returns
        -------
        central_moment : float

        Notes
        -----
        - The cmoment of order 1 is 0
        - The cmoment of order 2 is the variance.

        Source : https://en.wikipedia.org/wiki/Moment_(mathematics)

        TODO
        ----

        Implement: https://en.wikipedia.org/wiki/L-moment#cite_note-wang:96-6

        """
        if weights is None:
            weights = self.weights
        return statistics.c_moment(variable, weights, self, order, param, ddof)

    def percentile(self, variable=None, weights=None, p=50,
                   interpolate='lower'):
        """Calculate the value of a quantile given a variable and his weights.

        Parameters
        ----------
        data : pd.DataFrame, optional
            pd.DataFrame that contains all variables needed.
        variable : str or array
        weights :  str or array
        q : float
            Quantile level, if pass 0.5 means median.
        interpolate : bool

        Returns
        -------
        percentile : float or pd.Series

        """
        if weights is None:
            weights = self.weights
        return statistics.percentile(variable, weights, self, p, interpolate)

    def std_moment(self, variable=None, weights=None, param=None, order=3,
                   ddof=0):
        """Calculate the standardized moment of order `c` for the variable` x`
        with respect to `c`.

        Parameters
        ----------
        data : pd.DataFrame, optional
            pd.DataFrame that contains all variables needed.
        variable : 1d-array
           Random Variable
        weights : 1d-array, optional
           Weights or probability
        order : int, optional
           Order of Moment, three by default
        param : int or float or array, optional
           Central trend, default is the mean.
        ddof : int, optional
            Degree of freedom.

        Returns
        -------
        std_moment : float
           Returns the standardized `n` order moment.

        References
        ----------
        - https://en.wikipedia.org/wiki/Moment_(mathematics)#Significance_
          of_the_moments
        - https://en.wikipedia.org/wiki/Standardized_moment

        TODO
        ----
        It is the general case of the raw and central moments. Review
        implementation.

        """
        if weights is None:
            weights = self.weights
        return statistics.std_moment(variable, weights, self, param, order,
                                      ddof)

    def mean(self, variable=None, weights=None):
        """Calculate the mean of `variable` given `weights`.

        Parameters
        ----------
        variable : array-like or str
            Variable on which the mean is estimated.
        weights : array-like or str
            Weights of the `x` variable.
        data : pandas.DataFrame
            Is possible pass a DataFrame with variable and weights, then you
            must pass as `variable` and `weights` the column name stored in
            `data`.

        Returns
        -------
        mean : array-like or float
        """
        # if pass a DataFrame separate variables.
        if weights is None:
            weights = self.weights
        return statistics.mean(variable, weights, self)

    def density(self, variable=None, weights=None, groups=None):
        """Calculates density in percentage. This make division of variable
        inferring width in groups as max - min.

        Parameters
        ----------
        data : pd.DataFrame, optional
            pandas.DataFrame that contains all variables needed.
        variable :
        weights :
        groups :

        Returns
        -------
        density : array-like

        References
        ----------
        Histogram. (2017, May 9). In Wikipedia, The Free Encyclopedia.
        Retrieved:
        https://en.wikipedia.org/w/index.php?title=Histogram&oldid=779516918
        """
        if weights is None:
            weights = self.weights
        return statistics.density(variable, weights, groups, self)

    def var(self, variable=None, weights=None, ddof=0):
        """Calculate the population variance of `variable` given `weights`.

        Parameters
        ----------
        data : pd.DataFrame, optional
            pd.DataFrame that contains all variables needed.
        variable : 1d-array or pd.Series or pd.DataFrame
            Variable on which the quasivariation is estimated
        weights : 1d-array or pd.Series or pd.DataFrame
            Weights of the `variable`.


        Returns
        -------
        variance : 1d-array or pd.Series or float
            Estimation of quasivariance of `variable`

        References
        ---------
        Moment (mathematics). (2017, May 6). In Wikipedia, The Free
        Encyclopedia.
        Retrieved 14:40, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&oldid=
        778996402

        Notes
        -----
        If stratificated sample must pass with groupby each strata.
        """
        if weights is None:
            weights = self.weights
        return statistics.var(variable, weights, self, ddof)

    def coef_variation(self, variable=None, weights=None):
        """Calculate the coefficient of variation of a `variable` given weights.
        The coefficient of variation is the square root of the variance of the
        incomes divided by the mean income. It has the advantages of being
        mathematically tractable and is subgroup decomposable, but is not
        bounded from above.

        Parameters
        ----------
        data : pandas.DataFrame
        variable : array-like or str
        weights : array-like or str

        Returns
        -------
        coefficient_variation : float

        References
        ----------
        Coefficient of variation. (2017, May 5). In Wikipedia, The Free
        Encyclopedia.
        Retrieved 15:03, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Coefficient_of_variation&
        oldid=778842331
        """
        # TODO complete docstring
        if weights is None:
            weights = self.weights
        return statistics.coef_variation(variable, weights, self)

    def kurt(self, variable=None, weights=None):
        """Calculate the asymmetry coefficient

        Parameters
        ---------
        variable : 1d-array
        w : 1d-array

        Returns
        -------
        kurt : float
            Kurtosis coefficient.

        References
        ---------
        Moment (mathematics). (2017, May 6). In Wikipedia, The Free
        Encyclopedia.
        Retrieved 14:40, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&
        oldid=778996402

        Notes
        -----
        It is an alias of the standardized fourth-order moment.
        """
        if weights is None:
            weights = self.weights
        return statistics.kurt(variable, weights, self)

    def skew(self, variable=None, weights=None):
        """Returns the asymmetry coefficient of a sample.

        Parameters
        ---------
        data : pandas.DataFrame

        variable : array-like, str
        weights : array-like, str

        Returns
        -------
        skew : float

        References
        ---------
        Moment (mathematics). (2017, May 6). In Wikipedia, The Free
        Encyclopedia.
        Retrieved 14:40, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Moment_(mathematics)&
        oldid=778996402

        Notes
        -----
        It is an alias of the standardized third-order moment.

        """
        if weights is None:
            weights = self.weights
        return statistics.skew(variable, weights, self)

    # INEQUALITY
    # ----------

    def concentration(self, income=None, weights=None, sort=True):
        """This function calculate the concentration index, according to the
        notation used in [Jenkins1988]_ you can calculate the:
        C_x = 2 / x · cov(x, F_x)
        if x = g(x) then C_x becomes C_y
        when there are taxes:

        y = g(x) = x - t(x)

        Parameters
        ----------
        income : array-like
        weights : array-like
        data : pandas.DataFrame
        sort : bool

        Returns
        -------
        concentration : array-like

        References
        ----------
        Jenkins, S. (1988). Calculating income distribution indices
        from micro-data. National Tax Journal. http://doi.org/10.2307/41788716
        """
        # TODO complete docstring
        if weights is None:
            weights = self.weights
        return inequality.concentration(income, weights, self, sort)

    def lorenz(self, income=None, weights=None):
        """In economics, the Lorenz curve is a graphical representation of the
        distribution of income or of wealth. It was developed by Max O. Lorenz
        in 1905 for representing grouped of the wealth distribution. This
        function compute the lorenz curve and returns a DF with two columns of
        axis x and y.

        Parameters
        ----------
        data : pandas.DataFrame
            A pandas.DataFrame that contains data.
        income : str or 1d-array, optional
            Population or wights, if a DataFrame is passed then `income` should
            be a name of the column of DataFrame, else can pass a pandas.Series
            or array.
        weights : str or 1d-array
            Income, monetary variable, if a DataFrame is passed then `y`is a
            name of the series on this DataFrame, however, you can pass a
            pd.Series or np.array.

        Returns
        -------
        lorenz : pandas.Dataframe
            Lorenz distribution in a Dataframe with two columns, labeled x and
            y, that corresponds to plots axis.

        References
        ----------
        Lorenz curve. (2017, February 11). In Wikipedia, The Free Encyclopedia.
        Retrieved 14:34, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Lorenz_curve&oldid=764853675
        """
        if weights is None:
            weights = self.weights
        return inequality.lorenz(income, weights, self)

    def gini(self, income=None, weights=None, sort=True):
        """The Gini coefficient (sometimes expressed as a Gini ratio or a
        normalized Gini index) is a measure of statistical dispersion intended
        to represent the income or wealth distribution of a nation's residents,
        and is the most commonly used measure of grouped. It was developed by
        Corrado Gini.
        The Gini coefficient measures the grouped among values of a frequency
        distribution (for example, levels of income). A Gini coefficient of
        zero expresses perfect equality, where all values are the same (for
        example, where everyone has the same income). A Gini coefficient of 1
        (or 100%) expresses maximal grouped among values (e.g., for a large
        number of people, where only one person has all the income or
        consumption, and all others have none, the Gini coefficient will be
        very nearly one).

        Parameters
        ---------
        data : pandas.DataFrame
            DataFrame that contains the data.
        income : str or np.array, optional
            Name of the monetary variable `x` in` df`
        weights : str or np.array, optional
            Name of the series containing the weights `x` in` df`
        sorted : bool, optional
            If the DataFrame is previously ordered by the variable `x`, it's
            must pass True, but False by default.

        Returns
        -------
        gini : float
            Gini Index Value.

        Notes
        -----
        The calculation is done following (discrete probability distribution):
        G = 1 - [∑_i^n f(y_i)·(S_{i-1} + S_i)]
        where:
        - y_i = Income
        - S_i = ∑_{j=1}^i y_i · f(y_i)

        Reference
        ---------
        - Gini coefficient. (2017, May 8). In Wikipedia, The Free Encyclopedia.
          Retrieved 14:30, May 15, 2017, from
          https://en.wikipedia.org/w/index.php?title=Gini_coefficient&oldid=779424616

        - Jenkins, S. (1988). Calculating income distribution indices
        from micro-data. National Tax Journal. http://doi.org/10.2307/41788716

        TODO
        ----
        - Implement statistical deviation calculation, VAR (GINI)

        """
        if weights is None:
            weights = self.weights
        return inequality.gini(income, weights, self, sort)

    def atkinson(self, income=None, weights=None, e=0.5):
        """More precisely labelled a family of income grouped measures, the
        theoretical range of Atkinson values is 0 to 1, with 0 being a state of
        equal distribution.
        An intuitive interpretation of this index is possible: Atkinson values
        can be used to calculate the proportion of total income that would be
        required to achieve an equal level of social welfare as at present if
        incomes were perfectly distributed.

        For example, an Atkinson index value of 0.20 suggests
        that we could achieve the same level of social welfare with only
        1 – 0.20 = 80% of income. The theoretical range of Atkinson values is 0
        to 1, with 0 being a state of equal distribution.

        Parameters
        ---------
        income : array or str
            If `data` is none `income` must be an 1D-array, when `data` is a
            pd.DataFrame, you must pass the name of income variable as string.
        weights : array or str, optional
            If `data` is none `weights` must be an 1D-array, when `data` is a
            pd.DataFrame, you must pass the name of weights variable as string.
        e : int, optional
            Epsilon parameter interpreted by atkinson index as grouped
            adversion, must be a number between 0 to 1.
        data : pd.DataFrame, optional
            data is a pd.DataFrame that contains the variables.

        Returns
        -------
        atkinson : float

        Reference
        ---------
        Atkinson index. (2017, March 12). In Wikipedia, The Free Encyclopedia.
        Retrieved 14:35, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Atkinson_index&oldid=769991852

        TODO
        ----
        - Implement: CALCULATING INCOME DISTRIBUTION INDICES FROM MICRO-DATA
          http://www.jstor.org/stable/41788716
        - The results has difference with stata, maybe have a bug.
        """
        if weights is None:
            weights = self.weights
        return inequality.atkinson(income, weights, self, e)

    def kakwani(self, tax=None, income_pre_tax=None, weights=None):
        """The Kakwani (1977) index of tax progressivity is defined as twice the
        area between the concentration curves for taxes and pre-tax income,
        or equivalently, the concentration index for t(x) minus the Gini index
        for x, i.e.

        K = C(t) - G(x)
          = (2/t) cov [t(x), F(x)] - (2/x) cov [x, F(x)].

        Parameters
        ----------
        data : pandas.DataFrame
            This variable is a DataFrame that contains all data required in
            columns.
        tax_variable : array-like or str
            This variable represent tax payment of person, if pass array-like
            then data must be None, else you pass str-name column in `data`.
        income_pre_tax : array-like or str
            This variable represent income of person, if pass array-like
            then data must be None, else you pass str-name column in `data`.
        weights : array-like or str
            This variable represent weights of each person, if pass array-like
            then data must be None, else you pass str-name column in `data`.

        Returns
        -------
        kakwani : float

        References
        ----------
        Jenkins, S. (1988). Calculating income distribution indices from
        micro-data. National Tax Journal. http://doi.org/10.2307/41788716
        """
        # main calc
        if weights is None:
            weights = self.weights
        return inequality.kakwani(tax, income_pre_tax, weights, self)

    def reynolds_smolensky(self, income_pre_tax=None, income_post_tax=None,
                           weights=None):
        """The Reynolds-Smolensky (1977) index of the redistributive effect of
        taxes, which can also be interpreted as an index of progressivity
        (Lambert 1985), is defined as:

        L = Gx - Gy
          = [2/x]cov[x,F(x)] - [2/ybar] cov [y, F(y)].

        Parameters
        ----------
        data : pandas.DataFrame
            This variable is a DataFrame that contains all data required in it's
            columns.
        income_pre_tax : array-like or str
            This variable represent tax payment of person, if pass array-like
            then data must be None, else you pass str-name column in `data`.
        income_post_tax : array-like or str
            This variable represent income of person, if pass array-like
            then data must be None, else you pass str-name column in `data`.
        weights : array-like or str
            This variable represent weights of each person, if pass array-like
            then data must be None, else you pass str-name column in `data`.

        Returns
        -------
        reynolds_smolensky : float

        References
        ----------
        Jenkins, S. (1988). Calculating income distribution indices from
        micro-data. National Tax Journal. http://doi.org/10.2307/41788716
        """
        if weights is None:
            weights = self.weights
        return inequality.reynolds_smolensky(income_pre_tax,
                                             income_post_tax, weights, self)

    def theil(self, income=None, weights=None):
        """The Theil index is a statistic primarily used to measure economic
        grouped and other economic phenomena. It is a special case of the
        generalized entropy index. It can be viewed as a measure of redundancy,
        lack of diversity, isolation, segregation, grouped, non-randomness, and
        compressibility. It was proposed by econometrician Henri Theil.

        Parameters
        ----------
        data : pandas.DataFrame
            This variable is a DataFrame that contains all data required in it's
            columns.
        income : array-like or str
            This variable represent tax payment of person, if pass array-like
            then data must be None, else you pass str-name column in `data`.
        weights : array-like or str
            This variable represent weights of each person, if pass array-like
            then data must be None, else you pass str-name column in `data`.

        Returns
        -------
        theil : float

        References
        ----------
        Theil index. (2016, December 17). In Wikipedia, The Free Encyclopedia.
        Retrieved 14:17, May 15, 2017, from
        https://en.wikipedia.org/w/index.php?title=Theil_index&oldid=755407818

        """
        if weights is None:
            weights = self.weights
        return inequality.theil(income, weights, self)

    def avg_tax_rate(self, total_tax=None, total_base=None, weights=None):
        """This function compute the average tax rate given a base income and a
        total tax.

        Parameters
        ----------
        total_base : str or numpy.array
        total_tax : str or numpy.array
        data : pd.DataFrame

        Returns
        -------
        avg_tax_rate : float or pd.Series
            Is the ratio between mean the tax income and base of income.

        Reference
        ---------
        Panel de declarantes de IRPF 1999-2007: Metodología, estructura y variables.
        (2011). Panel de declarantes de IRPF 1999-2007:
        Metodología, estructura y variables. Documentos.
        """
        if weights is None:
            weights = self.weights
        return inequality.avg_tax_rate(total_tax, total_base, weights, self)
