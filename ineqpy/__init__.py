'''
IneqPy
'''
from .inequality import *
from .statistics import *
from . import api
from . import utils
from ._version import get_versions

__version__ = get_versions()['version']
__author__ = "Maximiliano Greco"
__maintainer__ = "Maximiliano Greco"
__email__ = "mmngreco@gmail.com"
__status__ = "Production"

del get_versions
