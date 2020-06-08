"""
Unit and regression test for the memly package.
"""

# Import package, test suite, and other packages as needed
import memly
import pytest
import sys

def test_memly_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "memly" in sys.modules
