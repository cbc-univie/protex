"""
protex
Proton exchange using SAMS and openMM for ionic liquids
"""

# Add imports here
from .protex import *

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

import logging

# format logging message
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
# set logging level
logging.basicConfig(format=FORMAT, datefmt="%d-%m-%Y:%H:%M", level=logging.INFO)
