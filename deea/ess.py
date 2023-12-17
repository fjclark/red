"""Functions to calculate the statistical inefficiency and effective sample size."""

import numpy as np

from ._validation import check_data
from .variance import inter_run_variance, intra_run_variance, lugsail_variance


def statistical_inefficiency_inter_variance(data: np.ndarray) -> float:
    """
    Compute the statistical inefficiency of a time series by dividing
    the inter-run variance estimate by the intra-run variance estimate.
    More than one run is required.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    Returns
    -------
    float
        The statistical inefficiency.
    """
    g = inter_run_variance(data) / intra_run_variance(data)
    # Ensure that the statistical inefficiency is at least 1.
    return max(g, 1)


def statistical_inefficiency_lugsail_variance(
    data: np.ndarray, n_pow: float = 1 / 3
) -> float:
    """
    Compute the statistical inefficiency of a time series by dividing
    the lugsail replicated batch means variance estimate by the
    intra-run variance estimate. This is applicable to a single run.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3.

    Returns
    -------
    float
        The statistical inefficiency.
    """
    g = lugsail_variance(data, n_pow=n_pow) / intra_run_variance(data)
    # Ensure that the statistical inefficiency is at least 1.
    return max(g, 1)


def ess_inter_variance(data: np.ndarray) -> float:
    """
    Compute the effective sample size of a time series by dividing
    the total number of samples by the statistical inefficiency, where
    the statistical inefficiency is calculated using the ratio of the
    inter-run and intra-run variance estimates.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    Returns
    -------
    float
        The effective sample size.
    """
    data = check_data(data, one_dim_allowed=False)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_inter_variance(data)


def ess_lugsail_variance(data: np.ndarray, n_pow: float = 1 / 3) -> float:
    """
    Compute the effective sample size of a time series by dividing
    the total number of samples by the statistical inefficiency, where
    the statistical inefficiency is calculated using the ratio of the
    lugsail replicated batch means and intra-run variance estimates.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3.

    Returns
    -------
    float
        The effective sample size.
    """
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_lugsail_variance(data, n_pow=n_pow)
