"""
Unit and regression test for the protex package.
"""

# Import package, test suite, and other packages as needed
import protex
import pytest
import sys

def test_protex_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "protex" in sys.modules
