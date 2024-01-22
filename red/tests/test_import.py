"""
Check that we can import red.
"""

import sys

import red


def test_red_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "red" in sys.modules
