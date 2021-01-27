"""
protex
Proton exchange using SAMS and openMM for ionic liquids
"""

# Add imports here
from .protex import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
