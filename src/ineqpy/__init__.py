"""IneqPy: A python package for inequality analysis."""

from ineqpy import inequality
from ineqpy import statistics
from ineqpy import grouped
from ineqpy import api
from ineqpy import utils
from ineqpy._version import get_versions

__version__ = get_versions()["version"]

del get_versions

__all__ = ["inequality", "statistics", "grouped", "api", "utils"]
