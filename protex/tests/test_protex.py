"""
Unit and regression test for the protex package.
"""

import sys

import pytest

# Import package, test suite, and other packages as needed
import protex


def test_protex_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "protex" in sys.modules
