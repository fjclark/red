"""
Unit and regression test for the validation module.
"""

import numpy as np
import pytest

from red._exceptions import InvalidInputError
from red._validation import check_data


def test_not_numpy_array():
    """Test that an exception is raised if data is not a numpy array."""
    data = [1, 2, 3]
    with pytest.raises(InvalidInputError):
        check_data(data)


def test_wrong_number_of_dimensions():
    """Test that an exception is raised if data has the wrong number of dimensions."""
    data = np.array([[[1, 2, 3], [4, 5, 6]]])
    with pytest.raises(InvalidInputError):
        check_data(data)


def test_one_d_not_allowed():
    """Test that an exception is raised if data is one dimensional and one_dim_allowed is False."""
    data = np.array([1, 2, 3])
    check_data(data, one_dim_allowed=True)
    with pytest.raises(InvalidInputError):
        check_data(data, one_dim_allowed=False)


def check_shape():
    """Check that the shape of the data is (n_chains, n_samples)."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert check_data(data) is None
    # Swap the dimensions.
    data = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(InvalidInputError):
        check_data(data)


def test_reshape_1d():
    """Test that a one dimensional array is reshaped to (1, n_samples)."""
    data = np.array([1, 2, 3])
    assert check_data(data, one_dim_allowed=True).shape == (1, 3)


def test_n_chains_greater_than_n_samples():
    """Test that an exception is raised if n_chains > n_samples."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.warns(RuntimeWarning):
        check_data(data)


def test_1_run_2_dim():
    """
    Test that validation still recognises that data is 1D if there
    are two dimensions but only one run.
    """
    with pytest.raises(InvalidInputError):
        check_data(np.array([[1, 2, 3]]), one_dim_allowed=False)
