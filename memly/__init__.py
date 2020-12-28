"""
Extensible analysis tool for lipid bilayer simulations.
"""

# Add imports here
from .analysis import Analysis
from .membrane import Membrane

import logging
logging.basicConfig(level=logging.WARNING)

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


def say_hello():
    """say hello"""
    print("Hello!")


def say_goodbye():
    """say goodbye"""
    print("Goodbye!")
