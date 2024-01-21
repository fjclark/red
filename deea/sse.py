"""Functions to calculate the squared standard error series."""

from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np

from ._validation import check_data
from .variance import (
    get_variance_series_initial_sequence,
    get_variance_series_window,
    inter_run_variance,
    intra_run_variance,
    lugsail_variance,
)


def get_sse_series_init_seq(
    data: _np.ndarray,
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Compute a series of squared standard errors for a time series as data
    is discarded from the beginning of the time series. The squared standard
    error is computed using the sequence estimator specified.

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
        The squared standard error series.

    np.ndarray
        The maximum lag times used.
    """
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # Compute the variance estimates.
    var_series, max_lag_times = get_variance_series_initial_sequence(
        data,
        sequence_estimator=sequence_estimator,
        min_max_lag_time=min_max_lag_time,
        max_max_lag_time=max_max_lag_time,
        smooth_lag_times=smooth_lag_times,
        frac_padding=frac_padding,
    )

    # Compute the squared standard error series by dividing the variance series by
    # the total number of samples.
    tot_samples = _np.arange(n_samples, n_samples - len(var_series), -1) * n_runs
    sse_series = var_series / tot_samples

    return sse_series, max_lag_times


def get_sse_series_window(
    data: _np.ndarray,
    kernel: _Callable[[int], _np.ndarray] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**1 / 2),
    window_size: _Optional[int] = None,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Compute a series of squared standard errors for a time series as data
    is discarded from the beginning of the time series. The squared standard
    error is computed using the window size and kernel specified.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    kernel : callable, optional, default=numpy.bartlett
        A function that takes a window size and returns a window function.

    window_size_fn : callable, optional, default=lambda x: round(x**1 / 2)
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
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # Compute the variance estimates.
    var_series, window_sizes = get_variance_series_window(
        data, kernel=kernel, window_size_fn=window_size_fn, window_size=window_size
    )

    # Compute the squared standard error series by dividing the variance series by
    # the total number of samples.
    tot_samples = _np.arange(n_samples, n_samples - len(var_series), -1) * n_runs
    sse_series = var_series / tot_samples

    return sse_series, window_sizes


def sse(data: _np.ndarray, method: str = "lugsail", n_pow: float = 1 / 3) -> float:
    """
    Compute the SSE of a time series by dividing the lugsail
    replicated batch means variance estimate by the total number
    of samples. This is applicable to a single run.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    method : str, optional
        The method to use to compute the variance estimate.
        This can be 'lugsail', 'inter', or 'intra'. The default
        is 'lugsail'. 'lugsail' estimates the SSE using the
        lugsail replicated batch means limiting variance estimate,
        'inter' uses the inter-run limiting variance estimate, and
        'intra' uses the intra-run limiting variance estimate. One
        dimensional data is only allowed when method is 'lugsail'
        or 'intra'.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3.

    Returns
    -------
    float
        The SSE.
    """
    # Check that the method is valid.
    valid_methods = ["lugsail", "inter", "intra"]
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, but got {method}.")

    # Validate the data.
    one_dim_allowed = method in ["lugsail", "intra"]
    data = check_data(data, one_dim_allowed=one_dim_allowed)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples

    # Compute the variance estimates.
    if method == "lugsail":
        lim_var_est = lugsail_variance(data, n_pow=n_pow)
    elif method == "inter":
        lim_var_est = inter_run_variance(data)
    else:  # method == "intra"
        lim_var_est = intra_run_variance(data)

    # Compute the SSE.
    sse = lim_var_est / total_samples

    return sse
