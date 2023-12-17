"""Functions for data validation."""

import numpy as np

from .exceptions import InvalidInputError


def check_data(data: np.ndarray, one_dim_allowed: bool = False) -> np.ndarray:
    """
    Assert that data passed is a numpy array where
    the first dimension is the number of chains and
    the second dimension is the number of samples.
    If the array is one dimensional, add a second
    dimension with length 1.

    Parameters
    ----------
    data : np.ndarray
        The data to check. Expected shape is (n_chains, n_samples).
    one_dim_allowed : bool, optional
        If True, allow data to be one dimensional, by default False.
        Maximum dimensionality allowed is 2.

    Returns
    -------
    np.ndarray
        Data with shape (n_chains, n_samples).
    """
    # Check that data is a numpy array.
    if not isinstance(data, np.ndarray):
        raise InvalidInputError("Data must be a numpy array.")

    # Check that data has right number of dimensions.
    dims_allowed = [1, 2] if one_dim_allowed else [2]
    # Check for case when n_dim is 2 but only one run.
    n_dims = data.ndim
    if n_dims == 2 and data.shape[0] == 1:
        n_dims = 1
    if n_dims not in dims_allowed:
        raise InvalidInputError(f"Data must have {dims_allowed} dimensions.")

    # If the data is two dimensional, check that the first dimension is not
    # smaller than the second dimension (i.e. there are more samples than
    # chains).
    if n_dims == 2:
        n_chains, n_samples = data.shape
        if n_chains > n_samples:
            raise InvalidInputError(
                "Data must have shape (n_chains, n_samples) where n_chains >= n_samples."
            )

    # If the array is one dimensional, reshape it to (1, n_samples).
    if n_dims == 1:
        data = data.reshape(1, -1)

    return data
