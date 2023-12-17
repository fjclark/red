"""
Check that we can import deea.
"""

import sys

import deea


def test_deea_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "deea" in sys.modules
