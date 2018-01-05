import ineqpy as ineq
import numpy as np
from ineqpy import _statistics as ineq_stats

def main():
    x, w = ineq.utils.generate_data_to_test((1e6 - 1, 1e6))

    np_mean = np.average(x, weights=w)
    mean = ineq.mean(None, x, w)
    print('Numpy Mean =', np_mean)
    print('IneqPy Mean =', mean)

    print('-' * 20)

    var = ineq.var(None, x, w)
    var2 = ineq_stats.wvar(x, w, 1)
    var3 = ineq_stats.wvar(x, w, 2)
    var4 = ineq_stats.wvar(x, w, 3)
    print('IneqPy var =', var)
    print('IneqPy var (population) =', var2)
    print('IneqPy var (sample freq w) =', var3)
    print('IneqPy var (sample reliability w) =', var4)

    print('-' * 20)

    gini = ineq.gini(None, x, w)
    print('IneqPy Gini =', gini)

    print('-' * 20)

    atk = ineq.atkinson(None, x, w)
    print('IneqPy Atkinson = ', atk)


if __name__ == '__main__':
    main()
