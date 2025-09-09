"""
Functions to calculate the variance of a time series, accounting for autocorrelation.

Methods implemented:

- Initial sequence methods (see Geyer, 1992: https://www.jstor.org/stable/2246094)
- Window estimators (see summary in see Geyer, 1992: https://www.jstor.org/stable/2246094)

Did not implement overlapping batch means (see Meketon and Schmeiser, 1984:
https://repository.lib.ncsu.edu/bitstream/handle/1840.4/7707/1984_0041.pdf?sequence=1), as this
is equivalent to using a Bartlett window.

"""

from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union
from typing import cast as _cast
from warnings import warn as _warn

import numba as _numba
import numpy as _np
import numpy.typing as _npt
from statsmodels.tsa.stattools import acovf as _acovf

from ._exceptions import AnalysisError, InvalidInputError
from ._validation import check_data as _check_data

####### Private functions #######
# No need to thoroughly validate input as this is done in the public functions.


@_numba.njit(cache=True)  # type: ignore
def _compute_autocovariance_no_fft(
    data: _npt.NDArray[_np.float64], max_lag: int
) -> _npt.NDArray[_np.float64]:
    """
    Calculate the auto-covariance as a function of lag time for a time series.
    Avoids using statsmodel's acovf function as using numpy's dot function and jit
    gives a substantial speedup.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    max_lag : int, optional, default=None
        The maximum lag time to use when calculating the auto-correlation function.
        If None, the maximum lag time will be the length of the time series.
        The default is None.

    Returns
    -------
    numpy.ndarray
        The auto-correlation function of the time series.
    """
    # Don't use statsmodel's acovf as we can get a substantial speedup by using
    # numpy's dot function and jit.
    n_samples = data.shape[0]

    # Initialise the auto-correlation function.
    auto_cov = _np.zeros(max_lag + 1)

    # Calculate the auto-correlation function.
    auto_cov[0] = data.dot(data)
    for t in range(1, max_lag + 1):
        auto_cov[t] = data[t:].dot(data[:-t])
    auto_cov /= n_samples  # "Biased" estimate, rather than n - 1.

    return auto_cov


def _compute_autocovariance_fft(
    data: _npt.NDArray[_np.float64], max_lag: int
) -> _npt.NDArray[_np.float64]:
    """
    Calculate the autocovariance using the FFT method, as implemented in statsmodels.
    Note that we can speed this up for large arrays by rewriting to directly use numpy's fft
    function and using jit with rocket-fft https://github.com/styfenschaer/rocket-fft.
    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    max_lag : int, optional, default=None
        The maximum lag time to use when calculating the auto-correlation function.
        If None, the maximum lag time will be the length of the time series.
        The default is None.

    Returns
    -------
    numpy.ndarray
    The auto-correlation function of the time series.
    """
    autocov_fn = _acovf(data, adjusted=False, nlag=max_lag, fft=True, demean=False)
    autocov_fn = autocov_fn.astype(_np.float64, copy=False)
    return _cast(_npt.NDArray[_np.float64], autocov_fn)


def _get_autocovariance(
    data: _npt.NDArray[_np.float64],
    max_lag: _Union[None, int] = None,
    mean: _Union[None, float] = None,
    fft: bool = False,
) -> _npt.NDArray[_np.float64]:
    """
    Calculate the auto-covariance as a function of lag time for a time series.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    max_lag : int, optional, default=None
        The maximum lag time to use when calculating the auto-correlation function.
        If None, the maximum lag time will be the length of the time series.
        The default is None.

    mean: float, optional, default=None
        The mean of the time series. If None, the mean will be calculated from the
        time series. This is useful when the mean has been calculated from an
        ensemble of time series.

    fft: bool, optional, default=False
        Whether to use the FFT method to calculate the auto-covariance. The FFT method is faster
        for large arrays and slower for shorter arrays.

    Returns
    -------
    numpy.ndarray
        The auto-correlation function of the time series.
    """
    # Copy the data so we don't modify the original.
    data = data.copy()

    # Get the length of the time series.
    n_samples: int = data.shape[0]

    # If max_lag_time is None, set it according to the length of the time series.
    if max_lag is None:
        max_lag = n_samples - 1

    # If mean is None, calculate it from the time series.
    valid_mean = mean if mean is not None else data.mean()

    # Subtract the mean from the data.
    data -= valid_mean

    # FFT is faster for large arrays and slower for shorter arrays.
    compute_autocov_fn = _compute_autocovariance_fft if fft else _compute_autocovariance_no_fft

    return compute_autocov_fn(data, max_lag)


@_numba.njit(cache=True)  # type: ignore
def _get_gamma_cap(
    autocov_series: _npt.NDArray[_np.float64],
) -> _npt.NDArray[_np.float64]:
    """
    Compute the capitial gamma function from the auto-covariance function.

    Parameters
    ----------
    autocov_series : numpy.ndarray
        The auto-covariance function of a time series.

    Returns
    -------
    numpy.ndarray
        The capital gamma function of the time series.
    """
    # Get the length of the time series.
    n_samples = autocov_series.shape[0]
    max_gamma = round(_np.floor(n_samples / 2))

    # Check that max_gamma is valid.
    if max_gamma < 1:
        raise AnalysisError(
            "The length of the time series is too short to compute the capital gamma function."
        )

    # Initialise the gamma function.
    gamma = _np.zeros(max_gamma)

    # Calculate the gamma function.
    for m in range(max_gamma):
        gamma[m] = autocov_series[2 * m] + autocov_series[(2 * m) + 1]

    return gamma


@_numba.njit(cache=True)  # type: ignore
def _get_initial_positive_sequence(
    gamma_cap: _npt.NDArray[_np.float64],
    min_max_lag_time: int = 3,
) -> _npt.NDArray[_np.float64]:
    """ "
    Get the initial positive sequence from the capital gamma function of a time series.
    See Geyer, 1992: https://www.jstor.org/stable/2246094.

    Parameters
    ----------
    gamma_cap : numpy.ndarray
        The capital gamma function of a time series.

    min_max_lag_time : int, optional
        The minimum maximum lag time to use when estimating the statistical inefficiency.
        The default is 3.

    Returns
    -------
    numpy.ndarray
        The initial positive sequence.
    """
    # Make a copy of gamma_cap so we don't modify the original.
    # gamma_cap = _deepcopy(gamma_cap)
    gamma_cap = gamma_cap.copy()

    # Truncate so that cap gamma is positive.
    for t in range(gamma_cap.shape[0]):
        if gamma_cap[t] < 0 and t > min_max_lag_time:
            return gamma_cap[:t]

    # All elements are positive.
    return gamma_cap


@_numba.njit(cache=True)  # type: ignore
def _get_initial_monotone_sequence(
    gamma_cap: _npt.NDArray[_np.float64],
    min_max_lag_time: int = 3,
) -> _npt.NDArray[_np.float64]:
    """
    Get the initial monotone sequence from the capital gamma function of a time series.
    See Geyer, 1992: https://www.jstor.org/stable/2246094.

    Parameters
    ----------
    gamma_cap : numpy.ndarray
        The capital gamma function of a time series.

    min_max_lag_time : int, optional
        The minimum maximum lag time to use when estimating the statistical inefficiency.

    Returns
    -------
    numpy.ndarray
        The initial monotone sequence.
    """
    # Make a copy of gamma_cap so we don't modify the original.
    gamma_cap = gamma_cap.copy()

    # Get the initial positive sequence.
    gamma_cap = _get_initial_positive_sequence(gamma_cap, min_max_lag_time=min_max_lag_time)

    # Now reduce to the initial monotone sequence.
    for t in range(gamma_cap.shape[0] - 1):
        if gamma_cap[t] < gamma_cap[t + 1]:
            gamma_cap[t + 1] = gamma_cap[t]

    return gamma_cap


@_numba.njit(cache=True)  # type: ignore
def _get_initial_convex_sequence(
    gamma_cap: _npt.NDArray[_np.float64],
    min_max_lag_time: int = 3,
) -> _npt.NDArray[_np.float64]:
    """
    Get the initial convex sequence from the capital gamma function of a time series.
    See Geyer, 1992: https://www.jstor.org/stable/2246094.

    Parameters
    ----------
    gamma_cap : numpy.ndarray
        The capital gamma function of a time series.

    min_max_lag_time : int, optional
        The minimum maximum lag time to use when estimating the statistical inefficiency.

    Returns
    -------
    numpy.ndarray
        The initial convex sequence.

    References
    ----------
    Adapted from https://github.com/cjgeyer/mcmc/blob/morph/package/mcmc/src/initseq.c,
    MIT License.
    YEAR: 2005, 2009, 2010, 2012
    COPYRIGHT HOLDER: Charles J. Geyer and Leif T. Johnson
    """
    # Make a copy of gamma_cap so we don't modify the original.
    gamma_con = gamma_cap.copy()

    # Get initial monotone sequence.
    gamma_con = _get_initial_monotone_sequence(gamma_con, min_max_lag_time=min_max_lag_time)

    # Get the length of gamma_con.
    len_gamma = gamma_con.shape[0]

    # Get a list of the first value followed by the differences.
    for j in range(len_gamma - 1, 0, -1):
        gamma_con[j] -= gamma_con[j - 1]

    # Now reduce to the initial convex sequence. Use the PAVA algorithm.
    pooled_values = _np.zeros(len_gamma)
    value_counts = _np.zeros(len_gamma, dtype=_np.int32)
    nstep = 0

    # Iterate over the elements in gamma_cap_diff.
    for j in range(1, len_gamma):
        pooled_values[nstep] = gamma_con[j]
        value_counts[nstep] = 1
        nstep += 1

        # While the average of the last two pooled values is decreasing,
        # combine them into one pool and decrement the step counter.
        while (
            nstep > 1
            and pooled_values[nstep - 1] / value_counts[nstep - 1]
            < pooled_values[nstep - 2] / value_counts[nstep - 2]
        ):
            pooled_values[nstep - 2] += pooled_values[nstep - 1]
            value_counts[nstep - 2] += value_counts[nstep - 1]
            nstep -= 1

    j = 1

    # Iterate over the steps.
    for jstep in range(nstep):
        # Calculate the average of the pooled values.
        mean_pooled_value = pooled_values[jstep] / value_counts[jstep]

        # Distribute the average pooled value over the cap gamma values in the step.
        for _ in range(value_counts[jstep]):
            gamma_con[j] = gamma_con[j - 1] + mean_pooled_value
            j += 1

    return _cast(_npt.NDArray[_np.float64], gamma_con)


def _get_autocovariance_window(
    data: _npt.NDArray[_np.float64],
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size: int = 10,
) -> _npt.NDArray[_np.float64]:
    """
    Calculate the autocovariance of a time series using window estimators.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    kernel : callable, optional, default=numpy.bartlett
        A function that takes a window size and returns a window function.
        The default is numpy.bartlett.

    window_size : int, optional, default=10
        The size of the window to use, defined in terms of time lags in the
        forwards direction.

    Returns
    -------
    np.ndarray
        The autocovariance of the time series as a function of lag time.
    """
    n_runs, n_samples = data.shape
    if n_samples < window_size:
        raise InvalidInputError("Window size is greater than the length of the time series.")

    # Get the window function. Need to truncate as numpy functions return
    # symmetric windows and we only want the forwards part.
    window = kernel(2 * window_size + 1)[window_size:]

    # Get the mean autocovariance as a function of lag time across all runs,
    # using the shared mean. Do not use FFT as we are usually calculating the autocovariance
    # for relatively small arrays where the FFT method would be slower.
    autocov = _np.mean(
        [
            _get_autocovariance(data[run], max_lag=window_size, mean=data.mean(), fft=False)
            for run in range(n_runs)
        ],
        axis=0,
    )

    # Get the windowed autocovariance.
    windowed_autocov = autocov[: window_size + 1] * window

    # Cast to satisfy mypy
    return _cast(_npt.NDArray[_np.float64], windowed_autocov)


def _smoothen_max_lag_times(
    max_lag_times: _npt.NDArray[_np.float64],
) -> _npt.NDArray[_np.int64]:
    """
    Smoothen a list of maximum lag times by a) converting them to a monotinically
    decreasing sequence and b) linearly interpolating between points where the sequence
    changes. This may be useful when the max lag times are noisy.

    Parameters
    ----------
    max_lag_times : numpy.ndarray
        The maximum lag times to smoothen.

    Returns
    -------
    numpy.ndarray
        The smoothened maximum lag times.
    """
    # Get a monotinically decreasing sequence.
    max_lag_times_monotonic = _get_initial_monotone_sequence(max_lag_times, min_max_lag_time=0)

    # Get the indices where the sequence changes.
    change_indices = _np.where(max_lag_times_monotonic[:-1] != max_lag_times_monotonic[1:])[0]

    # Get the indices immediately after the change.
    change_indices = _np.concatenate((_np.array([0]), change_indices + 1))

    # Get the values of the sequence at the change indices.
    change_values = max_lag_times_monotonic[change_indices]

    # Now linearly interpolate between these points.
    max_lag_times_to_use = _np.interp(
        _np.arange(max_lag_times_monotonic.shape[0]), change_indices, change_values
    )

    # Round the values.
    max_lag_times_to_use = _np.round(max_lag_times_to_use).astype(_np.int64)

    # Cast to int64 to satisfy mypy
    return _cast(_npt.NDArray[_np.int64], max_lag_times_to_use)


####### Public functions #######


def get_variance_initial_sequence(
    data: _npt.NDArray[_np.float64],
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    autocov: _Optional[_npt.NDArray[_np.float64]] = None,
) -> _Tuple[float, int, _npt.NDArray[_np.float64]]:
    """
    Calculate the variance of a time series using initial sequence methods.
    See Geyer, 1992: https://www.jstor.org/stable/2246094.

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

    autocov : numpy.ndarray, optional, default=None
        The auto-covariance function of the time series. If None, this will be calculated
        from the time series.

    Returns
    -------
    float
        The estimated variance of the time series, accounting for correlation.

    int
        The maximum lag time used when calculating the auto-correlated variance.

    numpy.ndarray
        The auto-covariance function of the time series.
    """
    # Validate the data.
    data = _check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # Check that sequence_estimator is valid.
    implemented_sequence_estimators = [
        "positive",
        "initial_positive",
        "initial_monotone",
        "initial_convex",
    ]
    if sequence_estimator not in implemented_sequence_estimators:
        raise InvalidInputError(
            f"Sequence_estimator must be one of {implemented_sequence_estimators}."
        )

    # Check that the minimum maximum lag time is valid.
    if min_max_lag_time < 0:
        raise InvalidInputError("Minimum maximum lag time must be greater than or equal to 0.")

    if min_max_lag_time > n_samples - 1:
        raise InvalidInputError(
            "Minimum maximum lag time must be less than or equal to the number of samples minus 1."
        )

    # Make sure that the maximum lag time is valid.
    if max_max_lag_time is not None:
        if max_max_lag_time < 0:
            raise InvalidInputError("Maximum lag time must be greater than or equal to 0.")

        if max_max_lag_time > n_samples - 1:
            raise InvalidInputError(
                "Maximum lag time must be less than or equal to the number of samples minus 1. "
                f"Maximum lag time is {max_max_lag_time} and number of samples is {n_samples}."
            )

        if max_max_lag_time < min_max_lag_time:
            raise InvalidInputError(
                "Maximum lag time must be greater than or equal to the minimum maximum lag time."
            )

        if autocov is not None:
            if max_max_lag_time > autocov.shape[0] - 1:
                raise InvalidInputError(
                    "Maximum lag time must be less than or equal to the length of the"
                    "autocovariance function minus 1."
                )

    # Check that autocov_series is valid.
    if autocov is not None:
        if not isinstance(autocov, _np.ndarray):
            raise InvalidInputError("Autocovariance must be a numpy.ndarray.")

        if autocov.ndim != 1:
            raise InvalidInputError("Autocovariance must be one-dimensional.")

        if autocov.shape[0] < 2:
            raise InvalidInputError("Autocovariance must have at least two elements.")

    # Get the uncorrected variance estimate.
    var = data.var()
    if var == 0:
        raise AnalysisError(
            "Variance of data is zero. Cannot compute variance. "
            "Check that you have input the correct data."
        )

    if autocov is None:
        # Get the mean autocovariance as a function of lag time across all runs,
        # using the shared mean. Use FFT as we are usually calculating the autocovariance
        # for large arrays.
        autocov_valid = _np.mean(
            [
                _get_autocovariance(
                    data[run],
                    mean=data.mean(),
                    max_lag=max_max_lag_time,
                    fft=True,
                )
                for run in range(n_runs)
            ],
            axis=0,
        )
    else:
        # Create autocov_valid to satisfy the type checker.
        autocov_valid = autocov

    # If using the positive estimator, truncate the autocovariance at the
    # first negative value, if this exists.
    if sequence_estimator == "positive":
        sub_zero_idxs = _np.where(autocov_valid < 0)[0]
        truncate_idx = sub_zero_idxs[0] if sub_zero_idxs.size > 0 else len(autocov_valid)
        # Limit the truncate in
        autocov_valid = autocov_valid[:truncate_idx]
        var_cor = autocov_valid.sum() * 2 - var
        # Ensure that the variance can't be less than the uncorrelated value.
        var_cor = max(var_cor, var)
        max_lag_time_used = truncate_idx - 1
        return var_cor, max_lag_time_used, autocov_valid

    # Otherwise, get the gamma function. Avoid recalculating if
    # it has already been provided.
    gamma_cap = _get_gamma_cap(autocov_valid)
    variance_fns = {
        "initial_positive": _get_initial_positive_sequence,
        "initial_monotone": _get_initial_monotone_sequence,
        "initial_convex": _get_initial_convex_sequence,
    }
    gamma_cap = variance_fns[sequence_estimator](gamma_cap, min_max_lag_time=min_max_lag_time)
    var_cor = gamma_cap.sum() * 2 - var

    # Make sure that the variance is not negative.
    var_cor = max(var_cor, var)

    # Get the maximum lag time.
    max_lag_time_used = gamma_cap.shape[0] * 2 - 1

    return var_cor, max_lag_time_used, autocov_valid


def get_variance_series_initial_sequence(
    data: _npt.NDArray[_np.float64],
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
    """
    Repeatedly calculate the variance of a time series while discarding increasing
    numbers of samples from the start of the time series. The variance is calculated
    using initial sequence methods. See Geyer, 1992: https://www.jstor.org/stable/2246094.

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
    numpy.ndarray
        The variance of the time series as a function of the number of discarded samples.

    numpy.ndarray
        The maximum lag time used when calculating the auto-correlated variance.
    """
    # Check that the data is valid.
    data = _check_data(data, one_dim_allowed=True)
    n_samples = data.shape[1]

    # Check that percent_padding is valid.
    if frac_padding < 0 or frac_padding >= 1:
        raise InvalidInputError("Percent padding must be >= 0 and < 1.")

    if frac_padding > 0.5:
        _warn(
            "Percent padding is greater than 0.5. You are evaluating less than half of the data.",
            stacklevel=2,
        )

    # Calculate the maximum index to use when discarding samples.
    max_index = n_samples - 1 - min_max_lag_time if min_max_lag_time else n_samples - 2

    # Needs to be a max of n_samples - 2 to allow the gamma function to be calculated.
    max_index = min(max_index, n_samples - 2)

    # See if we need to truncate the max index even further based on the percent padding.
    if frac_padding > 0:
        frac_padding_max_index = round(n_samples * (1 - frac_padding))
        max_index = min(max_index, frac_padding_max_index)

    # Calculate the variance at each index, and also the maximum lag time used
    # and the autocovariance function.
    variance_series = _np.zeros(max_index + 1)
    max_lag_times_used = _np.zeros(max_index + 1, dtype=int)
    # A list of autocovariance functions.
    autocov_series = []

    for index in range(max_index + 1):
        variance, max_lag_time_used, autocov = get_variance_initial_sequence(
            data[:, index:],
            sequence_estimator=sequence_estimator,
            min_max_lag_time=min_max_lag_time,
            max_max_lag_time=max_max_lag_time,
        )
        variance_series[index] = variance
        max_lag_times_used[index] = max_lag_time_used
        autocov_series.append(autocov)

        # If we are smoothing the lags, set the max lag time to the
        # maximum lag time used for the current index. This saves
        # some time computing the full autocovariance function.
        if smooth_lag_times:
            max_max_lag_time = max_lag_time_used
            # If it's the same length as the time series, subtract 1
            # so that it works on the next iteration.
            if max_max_lag_time == n_samples - index - 1:
                max_max_lag_time -= 1

    if smooth_lag_times:
        # Get the smoothened max lag times.
        max_lag_times_to_use_smooth = _smoothen_max_lag_times(max_lag_times_used)

        # Truncate the autocovariance functions at the smoothened max lag times.
        autocov_series = [
            autocov[: max_lag_times_to_use_smooth[index] + 1]
            for index, autocov in enumerate(autocov_series)
        ]

        # Recalculate the variance series.
        variance_series = _np.zeros(max_index + 1)
        max_lag_times_used = _np.zeros(max_index + 1, dtype=int)

        for index in range(max_index + 1):
            variance, max_lag_time_used, _ = get_variance_initial_sequence(
                data[:, index:],
                sequence_estimator=sequence_estimator,
                min_max_lag_time=min_max_lag_time,
                max_max_lag_time=max_lag_times_to_use_smooth[index],
                autocov=autocov_series[index],
            )
            variance_series[index] = variance
            max_lag_times_used[index] = max_lag_time_used

    return variance_series, max_lag_times_used


def get_variance_window(
    data: _npt.NDArray[_np.float64],
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size: int = 10,
) -> float:
    """
    Calculate the variance of a time series using window estimators.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    kernel : callable, optional, default=numpy.bartlett
        A function that takes a window size and returns a window function.

    window_size : int, optional, default=10
        The size of the window to use, defined in terms of time lags in the
        forwards direction.

    Returns
    -------
    float
        The estimated variance of the time series.
    """
    # Check that the data is valid.
    data = _check_data(data, one_dim_allowed=True)
    n_samples = data.shape[1]

    # Check that the window estimator is valid.
    if not callable(kernel):
        raise InvalidInputError("Window estimator must be a callable function.")

    # Check that the window size is valid.
    if window_size < 1:
        raise InvalidInputError("Window size must be greater than or equal to 1.")

    if window_size > n_samples - 1:
        raise InvalidInputError(
            "Window size must be less than or equal to the number of samples minus 1."
        )

    # Get the uncorrected variance estimate.
    var = data.var()
    if var == 0:
        raise AnalysisError(
            "Variance of data is zero. Cannot compute statistical inefficiency. "
            "Check that you have input the correct data."
        )

    # Get the windowed autocovariance.
    autocov = _get_autocovariance_window(data, kernel, window_size)

    # Account for correlation in the forwards and backwards directions.
    corr_var = autocov.sum() * 2 - var

    # Make sure that the variance is not less than the uncorrelated value.
    return float(max(corr_var, var))


def get_variance_series_window(
    data: _npt.NDArray[_np.float64],
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**0.5),
    window_size: _Optional[int] = None,
    frac_padding: float = 0.1,
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
    """
    Repeatedly calculate the variance of a time series while discarding increasing
    numbers of samples from the start of the time series. The variance is calculated
    using window estimators.

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
    numpy.ndarray
        The variance of the time series as a function of the number of discarded samples.

    numpy.ndarray
        The window size used at each index.
    """
    # Check that the data is valid.
    data = _check_data(data, one_dim_allowed=True)
    n_samples = data.shape[1]

    # Check that only one of window_size_fn and window_size is not None.
    if window_size_fn is not None and window_size is not None:
        raise InvalidInputError("Only one of window_size_fn and window_size can be not None.")

    if window_size_fn is None and window_size is None:
        raise InvalidInputError("One of window_size_fn and window_size must be not None.")

    if window_size_fn is not None:
        # Check that the window size function is valid.
        if not callable(window_size_fn):
            raise InvalidInputError("Window size function must be a callable function.")

    # Check that frac_padding is valid.
    if frac_padding < 0 or frac_padding >= 1:
        raise InvalidInputError("Percent padding must be >= 0 and < 1.")

    if frac_padding > 0.5:
        _warn(
            "Percent padding is greater than 0.5. You are evaluating less than half of the data.",
            stacklevel=2,
        )

    # Calculate the maximum index to use when discarding samples.
    max_index = n_samples - 1 - window_size if window_size is not None else n_samples - 2

    # See if we need to truncate the max index even further based on the percent padding.
    if frac_padding > 0:
        frac_padding_max_index = round(n_samples * (1 - frac_padding))
        max_index = min(max_index, frac_padding_max_index)

    # Calculate the variance at each index and store the window size used.
    variance_series = _np.zeros(max_index + 1)
    window_size_series = _np.zeros(max_index + 1, dtype=int)

    for index in range(max_index + 1):
        window_size = window_size_fn(n_samples - index) if window_size_fn else window_size
        variance_series[index] = get_variance_window(
            data[:, index:],
            kernel=kernel,
            window_size=window_size,  # type: ignore
        )
        window_size_series[index] = window_size

    return variance_series, window_size_series


def replicated_batch_means_variance(data: _npt.NDArray[_np.float64], batch_size: int) -> float:
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
    data = _check_data(data, one_dim_allowed=True)

    # Check that batch_size is valid.
    n_chains, n_samples = data.shape
    if batch_size < 1 or batch_size > n_samples:
        raise InvalidInputError(
            f"batch_size must be between 1 and n_samples = {n_samples} (inclusive),"
            f" but got {batch_size}."
        )

    # Compute the number of batches.
    n_batches = n_samples // batch_size

    # Compute the mean of each batch.
    batch_means = _np.mean(
        data[:, : n_batches * batch_size].reshape(n_chains, n_batches, batch_size),
        axis=2,
    )

    # Compute the variance of the batch means.
    batch_means_variance = _np.var(batch_means, ddof=1)

    # Multiply by the batch size.
    batch_means_variance *= batch_size

    return float(batch_means_variance)


def lugsail_variance(data: _npt.NDArray[_np.float64], n_pow: float = 1 / 3) -> float:
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
    data = _check_data(data, one_dim_allowed=True)

    # Check that n_pow is valid.
    if n_pow <= 0 or n_pow > 1:
        raise InvalidInputError(f"n_pow must be between 0 and 1 (inclusive), but got {n_pow}.")

    # Get the two batch sizes.
    _, n_samples = data.shape
    batch_size_large = int(_np.floor(n_samples**n_pow))  # type: ignore
    batch_size_small = int(_np.floor(batch_size_large / 3))  # type: ignore

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


def inter_run_variance(data: _npt.NDArray[_np.float64]) -> float:
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
    data = _check_data(data, one_dim_allowed=False)

    # Compute the inter-run variance.
    inter_run_variance = _np.var(_np.mean(data, axis=1), ddof=1)

    # Multiply by the number of samples per run.
    _, n_samples = data.shape
    inter_run_variance *= n_samples

    return float(inter_run_variance)


def intra_run_variance(data: _npt.NDArray[_np.float64]) -> float:
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
    data = _check_data(data, one_dim_allowed=True)

    # Compute the intra-run variance estimates.
    intra_run_variance = _np.var(data, axis=1, ddof=1)

    # Compute the mean intra-run variance estimate.
    mean_intra_run_variance = _np.mean(intra_run_variance)

    return float(mean_intra_run_variance)
