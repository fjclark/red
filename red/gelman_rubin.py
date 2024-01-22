"""Compute the Gelman-Rubin diagnostic."""

import numpy as _np

from ._validation import check_data
from .variance import inter_run_variance, intra_run_variance, lugsail_variance


def gelman_rubin(data: _np.ndarray) -> float:
    """
    Compute the Gelman-Rubin diagnostic according to
    equation 4 in  Statist. Sci. 36(4): 518-529
    (November 2021). DOI: 10.1214/20-STS812

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    Returns
    -------
    float
        The Gelman-Rubin diagnostic.
    """
    # Check that the data is valid.
    data = check_data(data, one_dim_allowed=False)
    _, n_samples = data.shape

    # Compute the variance estimates.
    intra_run_variance_est = intra_run_variance(data)
    inter_run_variance_est = inter_run_variance(data)

    # Get combined variance estimate.
    combined_variance_est = (
        n_samples - 1
    ) / n_samples * intra_run_variance_est + inter_run_variance_est / n_samples

    # Compute GR diagnostic.
    gelman_rubin_diagnostic = _np.sqrt(combined_variance_est / intra_run_variance_est)

    return gelman_rubin_diagnostic


def stable_gelman_rubin(data: _np.ndarray, n_pow: float = 1 / 3) -> float:
    """
    Compute the stable Gelman-Rubin diagnostic according to
    equation 7 in  Statist. Sci. 36(4): 518-529
    (November 2021). DOI: 10.1214/20-STS812. This is applicable to
    a single run.
    """
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)
    _, n_samples = data.shape

    # Compute the variance estimates.
    intra_run_variance_est = intra_run_variance(data)
    lugsail_variance_est = lugsail_variance(data, n_pow=n_pow)

    # Get combined variance estimate.
    combined_variance_est = (
        n_samples - 1
    ) / n_samples * intra_run_variance_est + lugsail_variance_est / n_samples

    # Compute GR diagnostic.
    gelman_rubin_diagnostic = _np.sqrt(combined_variance_est / intra_run_variance_est)

    return gelman_rubin_diagnostic
