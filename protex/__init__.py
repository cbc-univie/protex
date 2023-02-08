"""protex
Proton exchange using SAMS and openMM for ionic liquids.
"""

# Handle versioneer
#from ._version import get_versions

# Add imports here
from .protex import *

#versions = get_versions()
#__version__ = versions["version"]
#__git_revision__ = versions["full-revisionid"]
#del get_versions, versions
try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

import logging

# format logging message
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
# set logging level
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%Y:%H:%M", level=logging.INFO)
