"""
IneqPy: A python package for inequality analysis.
"""

from ineqpy.inequality import *
from ineqpy import inequality
from ineqpy.statistics import *
from ineqpy import statistics
from ineqpy import grouped
from ineqpy import api
from ineqpy import utils
from ineqpy._version import get_versions

__version__ = get_versions()["version"]
__author__ = "Maximiliano Greco"
__maintainer__ = "Maximiliano Greco"
__email__ = "mmngreco@gmail.com"
__status__ = "Production"

del get_versions
