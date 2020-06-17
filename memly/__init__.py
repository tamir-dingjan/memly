"""
Extensible analysis tool for lipid bilayer simulations.
"""

# Add imports here
from memly.analysis import Analysis
from memly.membrane import Membrane


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


def say_hello():
    """say hello"""
    print("Hello!")
