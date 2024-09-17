"""Functions to calculate the squared standard error series."""

from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
import numpy.typing as _npt

from ._validation import check_data
from .variance import get_variance_series_initial_sequence, get_variance_series_window


def get_sse_series_init_seq(
    data: _npt.NDArray[_np.float64],
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
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
    data: _npt.NDArray[_np.float64],
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**0.5),
    window_size: _Optional[int] = None,
    frac_padding: float = 0.1,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
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

    window_size_fn : callable, optional, default=lambda x: round(x**0.5)
        A function that takes the length of the time series and returns the window size
        to use. If this is not None, window_size must be None.

    window_size : int, optional, default=None
        The size of the window to use, defined in terms of time lags in the
        forwards direction. If this is not None, window_size_fn must be None.

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
        The window sizes used.
    """
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # Compute the variance estimates.
    var_series, window_sizes = get_variance_series_window(
        data,
        kernel=kernel,
        window_size_fn=window_size_fn,
        window_size=window_size,
        frac_padding=frac_padding,
    )

    # Compute the squared standard error series by dividing the variance series by
    # the total number of samples.
    tot_samples = _np.arange(n_samples, n_samples - len(var_series), -1) * n_runs
    sse_series = var_series / tot_samples

    return sse_series, window_sizes
