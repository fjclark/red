"""Functions to calculate the statistical inefficiency and effective sample size."""

from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
import numpy.typing as _npt

from ._validation import check_data
from .sse import get_sse_series_init_seq as _get_sse_series_init_seq
from .sse import get_sse_series_window as _get_sse_series_window
from .variance import inter_run_variance, intra_run_variance, lugsail_variance


def convert_sse_series_to_ess_series(
    data: _npt.NDArray[_np.float64], sse_series: _npt.NDArray[_np.float64]
) -> _npt.NDArray[_np.float64]:
    """
    Convert a series of squared standard errors to a series of effective sample sizes.

    Parameters
    ----------
    sse_series : np.ndarray
        The squared standard error series.

    uncor_vars : np.ndarray
        The uncorrelated variances.

    Returns
    -------
    np.ndarray
        The effective sample size series.
    """
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)

    # Now get the uncorrelated variances.
    uncor_vars = _np.zeros_like(sse_series)

    for i in range(len(sse_series)):
        # Get "biased", rather than n - 1, variance.
        uncor_vars[i] = data[:, i:].var()  # type: ignore

    return uncor_vars / sse_series  # type: ignore


def get_ess_series_init_seq(
    data: _npt.NDArray[_np.float64],
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
    """
    Compute a series of effective sample sizes for a time series as data
    is discarded from the beginning of the time series. The autocorrelation
    is computed using the sequence estimator specified.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    sequence_estimator : str, optional
        The initial sequence estimator to use. Can be "positive", "initial_positive",
        "initial_monotone", or "initial_convex". The default is "initial_convex". "positive"
        corresponds to truncating the auto-covariance function at the first negative value, as is
        done in pymbar. The other methods correspond to the methods described in Geyer, 1992:
        https://www.jstor.org/stable/2246094.

    min_max_lag_time : int, optional, default=3
        The minimum maximum lag time to use when estimating the statistical inefficiency.

    max_max_lag_time : int, optional, default=None
        The maximum maximum lag time to use when calculating the auto-correlation function.
        If None, the maximum lag time will be the length of the time series.

    smooth_lag_times : bool, optional, default=False
        Whether to smooth out the max lag times by a) converting them to a monotinically
        decreasing sequence and b) linearly interpolating between points where the sequence
        changes. This may be useful when the max lag times are noisy.

    frac_padding : float, optional, default=0.1
        The fraction of the end of the timeseries to avoid calculating the variance
        for. For example, if frac_padding = 0.1, the variance will be calculated
        for the first 90% of the time series. This helps to avoid noise in the
        variance when there are few data points.

    Returns
    -------
    np.ndarray
        The effective sample size series.

    np.ndarray
        The maximum lag times used.
    """
    sse_series, max_lag_times = _get_sse_series_init_seq(
        data,
        sequence_estimator=sequence_estimator,
        min_max_lag_time=min_max_lag_time,
        max_max_lag_time=max_max_lag_time,
        smooth_lag_times=smooth_lag_times,
        frac_padding=frac_padding,
    )

    ess_series = convert_sse_series_to_ess_series(data, sse_series)

    return ess_series, max_lag_times


def get_ess_series_window(
    data: _npt.NDArray[_np.float64],
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**0.5),
    window_size: _Optional[int] = None,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
    """
    Compute a series of effective sample sizes for a time series as data
    is discarded from the beginning of the time series. The squared standard
    error is computed using the window size and kernel specified.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    kernel : callable, optional, default=numpy.bartlett
        A function that takes a window size and returns a window function.

    window_size_fn : callable, optional, default=lambda x: round(x**0.5)
        A function that takes the length of the time series and returns the window size
        to use. If this is not None, window_size must be None.

    window_size : int, optional, default=None
        The size of the window to use, defined in terms of time lags in the
        forwards direction. If this is not None, window_size_fn must be None.

    Returns
    -------
    np.ndarray
        The squared standard error series.

    np.ndarray
        The window sizes used.
    """
    sse_series, max_lag_times = _get_sse_series_window(
        data, kernel=kernel, window_size_fn=window_size_fn, window_size=window_size
    )

    ess_series = convert_sse_series_to_ess_series(data, sse_series)

    return ess_series, max_lag_times


def statistical_inefficiency_inter_variance(data: _npt.NDArray[_np.float64]) -> float:
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
    data: _npt.NDArray[_np.float64], n_pow: float = 1 / 3
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


def ess_inter_variance(data: _npt.NDArray[_np.float64]) -> float:
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
    n_runs: int = data.shape[0]
    n_samples: int = data.shape[1]
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_inter_variance(data)


def ess_lugsail_variance(data: _npt.NDArray[_np.float64], n_pow: float = 1 / 3) -> float:
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
    n_runs: int = data.shape[0]
    n_samples: int = data.shape[1]
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_lugsail_variance(data, n_pow=n_pow)
