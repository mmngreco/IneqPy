'''
IneqPy
'''
from .inequality import *
from .statistics import *
from . import api
from . import utils

__author__ = "Maximiliano Greco"
__version__ = "0.0.2"
__maintainer__ = "Maximiliano Greco"
__email__ = "mmngreco@gmail.com"
__status__ = "Production"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
