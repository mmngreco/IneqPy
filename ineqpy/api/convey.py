import pandas as pd
from .. import statistics
from .. import inequality
from functools import partial
from types import MethodType
import inspect


def _attach_method(module, instance):
    # get methods names contained in module
    method_name_list = inspect.getmembers(module, inspect.isfunction)
    for method_name, func in method_name_list:
        # if method_name.startswith('_'): continue  # avoid private methods
        func = getattr(module, method_name)  # get function
        if 'weights' in inspect.signature(func).parameters:  # replace weights variable
            func = partial(func, weights=instance.weights)
        func = MethodType(func, instance)
        setattr(instance, method_name, func)


class Survey(object):

    def __init__(self, data, index=None, columns=None, weights=None, group=None):
        self.data = pd.DataFrame(data=data, index=index, columns=columns)
        self.weights = weights
        self.group = group
        _attach_method(statistics, self)
        _attach_method(inequality, self)

