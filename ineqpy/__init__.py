'''
IneqPy: A python package for inequality analysis.
'''

from .inequality import *
from . import inequality
from .statistics import *
from . import statistics
from . import grouped
from . import api
from . import utils
from ._version import get_versions

__version__ = get_versions()['version']
__author__ = "Maximiliano Greco"
__maintainer__ = "Maximiliano Greco"
__email__ = "mmngreco@gmail.com"
__status__ = "Production"

del get_versions
