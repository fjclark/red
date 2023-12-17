"""Functions for computing variance."""

import numpy as np

from ._validation import check_data
from .exceptions import AnalysisError, InvalidInputError


def replicated_batch_means_variance(data: np.ndarray, batch_size: int) -> float:
    """
    Estimate the variance of a time series using the replicated batch means method.
    See section 3.1 in Statist. Sci. 36(4): 518-529 (November 2021).
    DOI: 10.1214/20-STS812 .

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape (n_chains, n_samples).

    batch_size : int
        The batch size to use.

    Returns
    -------
    float
        The estimated variance.
    """
    data = check_data(data, one_dim_allowed=True)

    # If array is

    # Check that batch_size is valid.
    n_chains, n_samples = data.shape
    if batch_size < 1 or batch_size > n_samples:
        raise InvalidInputError(
            f"batch_size must be between 1 and n_samples = {n_samples} (inclusive), but got {batch_size}."
        )

    # Compute the number of batches.
    n_batches = n_samples // batch_size

    # Compute the mean of each batch.
    batch_means = np.mean(
        data[:, : n_batches * batch_size].reshape(n_chains, n_batches, batch_size),
        axis=2,
    )

    # Compute the variance of the batch means.
    batch_means_variance = np.var(batch_means, ddof=1)

    # Multiply by the batch size.
    batch_means_variance *= batch_size

    return batch_means_variance


def lugsail_variance(data: np.ndarray, n_pow: float = 1 / 3) -> float:
    """
    Estimate the variance of a time series using the lugsail method.
    See section 3.2 in Statist. Sci. 36(4): 518-529 (November 2021).
    DOI: 10.1214/20-STS812 .

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape (n_chains, n_samples).

    n_pow : float, optional, default=1/3
        The batch size is computed as floor(n_samples**n_pow). Recommended
        choices are 1/3 or 1/2.

    Returns
    -------
    float
        The estimated variance.
    """
    # Check that the data is valid.
    data = check_data(data, one_dim_allowed=True)

    # Check that n_pow is valid.
    if n_pow <= 0 or n_pow > 1:
        raise InvalidInputError(
            f"n_pow must be between 0 and 1 (inclusive), but got {n_pow}."
        )

    # Get the two batch sizes.
    n_chains, n_samples = data.shape
    batch_size_large = int(np.floor(n_samples**n_pow))
    batch_size_small = int(np.floor(batch_size_large / 3))

    # Make sure that the batch sizes are valid.
    if batch_size_large == batch_size_small or batch_size_small < 1:
        raise AnalysisError(
            "The batch sizes computed using n_pow are too small. Try a larger value of n_pow."
        )

    # Compute the variance of the batch means.
    variance_large_batch = replicated_batch_means_variance(data, batch_size_large)
    variance_small_batch = replicated_batch_means_variance(data, batch_size_small)

    # Compute the lugsail variance.
    lugsail_variance = 2 * variance_large_batch - variance_small_batch

    return lugsail_variance


def inter_run_variance(data: np.ndarray) -> float:
    """
    Compute the variance based on the inter-run differences
    between means.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape (n_chains, n_samples).

    Returns
    -------
    float
        The estimated variance.
    """
    # Check that the data is valid.
    data = check_data(data, one_dim_allowed=False)

    # Compute the inter-run variance.
    inter_run_variance = np.var(np.mean(data, axis=1), ddof=1)

    # Multiply by the number of samples per run.
    _, n_samples = data.shape
    inter_run_variance *= n_samples

    return inter_run_variance


def intra_run_variance(data: np.ndarray) -> float:
    """
    Compute the average intra-run variance estimate.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape (n_chains, n_samples).

    Returns
    -------
    float
        The mean intra-run variance estimate.
    """
    # Check that the data is valid.
    data = check_data(data, one_dim_allowed=True)

    # Compute the intra-run variance estimates.
    intra_run_variance = np.var(data, axis=1, ddof=1)

    # Compute the mean intra-run variance estimate.
    mean_intra_run_variance = np.mean(intra_run_variance)

    return mean_intra_run_variance
