"""Compute the Gelman-Rubin diagnostic."""

import numpy as _np
import numpy.typing as _npt

from ._validation import check_data as _check_data
from .variance import (
    inter_run_variance as _inter_run_variance,
)
from .variance import (
    intra_run_variance as _intra_run_variance,
)
from .variance import (
    lugsail_variance as _lugsail_variance,
)


def gelman_rubin(data: _npt.NDArray[_np.float64]) -> float:
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
    data = _check_data(data, one_dim_allowed=False)
    _, n_samples = data.shape

    # Compute the variance estimates.
    intra_run_variance_est = _intra_run_variance(data)
    inter_run_variance_est = _inter_run_variance(data)

    # Get combined variance estimate.
    combined_variance_est = (
        n_samples - 1
    ) / n_samples * intra_run_variance_est + inter_run_variance_est / n_samples

    # Compute GR diagnostic.
    gelman_rubin_diagnostic = _np.sqrt(combined_variance_est / intra_run_variance_est)

    return float(gelman_rubin_diagnostic)


def stable_gelman_rubin(data: _npt.NDArray[_np.float64], n_pow: float = 1 / 3) -> float:
    """
    Compute the stable Gelman-Rubin diagnostic according to
    equation 7 in  Statist. Sci. 36(4): 518-529
    (November 2021). DOI: 10.1214/20-STS812. This is applicable to
    a single run.
    """
    # Validate the data.
    data = _check_data(data, one_dim_allowed=True)
    _, n_samples = data.shape

    # Compute the variance estimates.
    intra_run_variance_est = _intra_run_variance(data)
    lugsail_variance_est = _lugsail_variance(data, n_pow=n_pow)

    # Get combined variance estimate.
    combined_variance_est = (
        n_samples - 1
    ) / n_samples * intra_run_variance_est + lugsail_variance_est / n_samples

    # Compute GR diagnostic.
    gelman_rubin_diagnostic = _np.sqrt(combined_variance_est / intra_run_variance_est)

    return float(gelman_rubin_diagnostic)
