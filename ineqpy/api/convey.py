import pandas as pd
from .. import statistics
from .. import inequality
from functools import partial


class Survey(pd.DataFrame):

    def __init__(self, data, index, columns, weights=None, group=None):
        super().__init__(data, index, columns)
        self.weights = weights
        self.group = group


class Convey(Survey):

    @classmethod
    def wrapper(cls):
        for method in dir(statistics):
            if method.startswith('_'): continue
            if hasattr(getattr(my_module, method), 'weights'):
                func = partial(getattr(my_module, method), weights=self.weights)
            setattr(cls, method, func)

        for method in dir(inequality):
            if method.startswith('_'): continue
            if hasattr(getattr(my_module, method), 'weights'):
                func = partial(getattr(my_module, method), weights=self.weights)
            setattr(cls, method, func)

