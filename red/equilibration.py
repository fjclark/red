"""Functions for selecting the equilibration time of a time series."""

from pathlib import Path as _Path
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
from matplotlib import gridspec as _gridspec
from scipy.stats import ttest_rel as _ttest_rel

from ._exceptions import EquilibrationNotDetectedError, InvalidInputError
from ._validation import check_data
from .ess import convert_sse_series_to_ess_series
from .plot import plot_equilibration_min_sse, plot_equilibration_paired_t_test
from .sse import get_sse_series_init_seq, get_sse_series_window


def detect_equilibration_init_seq(
    data: _npt.NDArray[_np.float64],
    times: _Optional[_npt.NDArray[_np.float64]] = None,
    method: str = "min_sse",
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
    plot: bool = False,
    plot_name: _Union[str, _Path] = "equilibration_sse_init_seq.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
    plot_max_lags: bool = True,
) -> _Tuple[_Union[float, int], float, float]:
    r"""
    Detect the equilibration time of a time series by finding the minimum
    squared standard error (SSE), or maximum effective sample size (ESS)
    of the time series, using initial sequence estimators of the variance.
    This is done by computing the SSE at each time point, discarding all
    samples before the time point. The index of the time point with
    the minimum SSE or maximum ESS is taken to be the point of equilibration.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) or (n_samples,).

    times : np.ndarray, optional
        The times at which the data was sampled. If this is
        not provided, the indices of the data will be used.

    method : str, optional, default="min_sse"
        The method to use to select the equilibration time. This can be
        "min_sse" or "max_ess".

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

    plot : bool, optional
        Whether to plot the SSE curve. The default is False.

    plot_name : str | Path, optional
        The name of the plot file. The default is 'equilibration_sse_init_seq.png'.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    plot_max_lags : bool, optional, default=True
        Whether to plot the maximum lag times used to estimate the variance.

    Returns
    -------
    equil_time: float | int
        The time (or index, if no times are supplied) at which
        the time series is equilibrated.

    equil_g: float
        The statistical inefficiency at the equilibration point.

    equil_ess: float
        The effective sample size at the equilibration point.
    """
    # Check that data is valid.
    data = check_data(data, one_dim_allowed=True)
    _, n_samples = data.shape

    # Check that method is valid.
    valid_methods = ["min_sse", "max_ess"]
    method = method.lower()
    if method not in valid_methods:
        raise InvalidInputError(f"method must be one of {valid_methods}, but got {method}.")

    # If times is None, units of time are indices.
    if times is None:
        time_units = "index"
        # Convert times to indices.
        times_valid: _npt.NDArray[_Union[_np.int64, _np.float64]] = _np.arange(
            n_samples, dtype=_np.int64
        )
    else:
        # To satisfy type checking.
        times_valid = times

    # Get the SSE timeseries.
    sse_vals, max_lag_times = get_sse_series_init_seq(
        data=data,
        sequence_estimator=sequence_estimator,
        min_max_lag_time=min_max_lag_time,
        max_max_lag_time=max_max_lag_time,
        smooth_lag_times=smooth_lag_times,
        frac_padding=frac_padding,
    )

    # Get the corresponding times (or indices).
    sse_times = times_valid[: len(sse_vals)]

    # Convert the SSE to 1/ESS if requested (divide by uncorrelated variance).
    if method == "max_ess":
        ess_vals = convert_sse_series_to_ess_series(data=data, sse_series=sse_vals)
        sse_vals = 1 / ess_vals

    # Get the index of the minimum SSE.
    equil_idx = _np.argmin(sse_vals)
    equil_time = sse_times[equil_idx]

    # Now compute the effective sample size at this time point.
    equil_data = data[:, equil_idx:]
    equil_ess = 1 / sse_vals[equil_idx]
    if method == "min_sse":  # Has not yet been multiplied by the uncorrelated variance.
        equil_ess *= equil_data.var()
    equil_g = equil_data.size / equil_ess
    equil_ess = equil_data.size / equil_g

    if plot:
        # Create a figure.
        fig = _plt.figure(figsize=(6, 4))
        gridspec_obj = _gridspec.GridSpec(1, 1, figure=fig)

        # Plot the ESS.
        plot_equilibration_min_sse(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=data,
            sse_series=sse_vals,
            max_lag_series=max_lag_times if plot_max_lags else None,
            data_times=times_valid,
            sse_times=sse_times,
            time_units=time_units,
            data_y_label=data_y_label,
            variance_y_label="ESS"
            if method == "max_ess"
            else r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$",
        )

        fig.savefig(str(plot_name), dpi=300, bbox_inches="tight")

    # Return the equilibration index, limiting variance estimate, and SSE.
    return equil_time, equil_g, equil_ess


def detect_equilibration_window(
    data: _npt.NDArray[_np.float64],
    times: _Optional[_npt.NDArray[_np.float64]] = None,
    method: str = "min_sse",
    kernel: _Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**0.5),
    window_size: _Optional[int] = None,
    frac_padding: float = 0.1,
    plot: bool = False,
    plot_name: _Union[str, _Path] = "equilibration_sse_window.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
    plot_window_size: bool = True,
) -> _Tuple[_Union[float, int], float, float]:
    r"""
    Detect the equilibration time of a time series by finding the minimum
    squared standard error (SSE) or maximum effective sample size (ESS)
    of the time series, using window estimators of the variance. This is
    done by computing the SSE at each time point, discarding all samples
    before the time point. The index of the time point with the minimum
    SSE is taken to be the point of equilibration.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) or (n_samples,). If the method
        is 'lugsail', the data may have only one run, but
        otherwise there must be at least two runs.

    times : np.ndarray, optional
        The times at which the data was sampled. If this is
        not provided, the indices of the data will be used.

    method : str, optional, default="min_sse"
        The method to use to select the equilibration time. This can be
        "min_sse" or "max_ess".

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

    plot : bool, optional
        Whether to plot the ESS curve. The default is False.

    plot_name : str | Path, optional
        The name of the plot file. The default is 'equilibration_sse_window.png'.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    plot_window_size : bool, optional, default=True
        Whether to plot the window size used to estimate the variance.

    Returns
    -------
    equil_time: float | int
        The time (or index, if no times are supplied) at which
        the time series is equilibrated.

    equil_g: float
        The statistical inefficiency at the equilibration point.

    equil_ess: float
        The effective sample size at the equilibration point.
    """
    # Check that data is valid.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # If times is None, units of time are indices.
    if times is None:
        time_units = "index"
        # Convert times to indices.
        times_valid: _npt.NDArray[_Union[_np.int64, _np.float64]] = _np.arange(
            n_samples, dtype=_np.int64
        )
    else:
        # To satisfy type checking.
        times_valid = times

    # Get the SSE timeseries
    sse_vals, window_sizes = get_sse_series_window(
        data=data,
        kernel=kernel,
        window_size_fn=window_size_fn,
        window_size=window_size,
        frac_padding=frac_padding,
    )

    # Get the corresponding times (or indices).
    sse_times = times_valid[: len(sse_vals)]  # type: ignore

    # Convert the SSE to 1/ESS if requested (divide by uncorrelated variance).
    if method == "max_ess":
        ess_vals = convert_sse_series_to_ess_series(data=data, sse_series=sse_vals)
        sse_vals = 1 / ess_vals

    # Get the index of the minimum SSE.
    equil_idx = _np.argmin(sse_vals)
    equil_time = sse_times[equil_idx]

    # Now compute the effective sample size at this time point.
    equil_data = data[:, equil_idx:]
    equil_ess = 1 / sse_vals[equil_idx]
    if method == "min_sse":  # Has not yet been multiplied by the uncorrelated variance.
        equil_ess *= equil_data.var()
    equil_g = equil_data.size / equil_ess
    equil_ess = equil_data.size / equil_g

    if plot:
        # Create a figure.
        fig = _plt.figure(figsize=(6, 4))
        gridspec_obj = _gridspec.GridSpec(1, 1, figure=fig)

        # Plot the ESS.
        plot_equilibration_min_sse(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=data,
            sse_series=sse_vals,
            data_times=times_valid,
            sse_times=sse_times,
            time_units=time_units,
            data_y_label=data_y_label,
            window_size_series=window_sizes if plot_window_size else None,
            variance_y_label="ESS"
            if method == "max_ess"
            else r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$",
        )

        fig.savefig(str(plot_name), dpi=300, bbox_inches="tight")

    # Return the equilibration time (index), statistical inefficiency, and ESS.
    return equil_time, equil_g, equil_ess


def get_paired_t_p_timeseries(
    data: _npt.NDArray[_np.float64],
    times: _Optional[_npt.NDArray[_np.float64]] = None,
    fractional_block_size: float = 0.125,
    fractional_test_end: float = 0.5,
    initial_block_size: float = 0.1,
    final_block_size: float = 0.5,
    t_test_sidedness: str = "two-sided",
) -> _Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_Union[_np.int64, _np.float64]]]:
    """
    Get a timeseries of the p-values from a paired t-test on the differences
    between sample means between intial and final portions of the data. The timeseries
    is obtained by repeatedly discarding more data from the time series between
    calculations of the p-value.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    times : np.ndarray, optional
        The times at which the data was sampled. If this is
        not provided, the indices of the data will be used.

    fractional_block_size : float, optional
        The fraction of data to discard between repeats. The default is 0.125.

    fractional_test_end : float, optional
        The fraction of the time series to use in the final test. The default is 0.5.

    initial_block_size : float, optional
        The fraction of the truncated time series to use for the "before" portion
        of the paired t-test. The default is 0.1.

    final_block_size : float, optional
        The fraction of the truncated time series to use for the "after" portion
        of the paired t-test. The default is 0.5.

    t_test_sidedness : str, optional
        The sidedness of the paired t-test. This can be either 'two-sided', 'less',
        or 'greater'. The default is 'two-sided'.

    Returns
    -------
    np.ndarray
        The p-values of the paired t-test.

    np.ndarray
        The times at which the p-values were calculated.
    """
    # Check that the data is valid.
    data = check_data(data, one_dim_allowed=False)
    n_runs, n_samples = data.shape

    # Convert times to indices if necessary.
    times_valid: _npt.NDArray[_Union[_np.float64, _np.int64]] = (
        _np.arange(n_samples, dtype=_np.int64) if times is None else times
    )

    # Check that times is match the number of samples.
    if n_samples != len(times_valid):
        raise InvalidInputError("Times must have the same length as the number of samples.")

    # Check that user inputs are valid.
    if fractional_block_size <= 0 or fractional_block_size > 1:
        raise InvalidInputError("fractional_block_size must be between 0 and 1.")

    if fractional_test_end <= 0 or fractional_test_end > 1:
        raise InvalidInputError("fractional_test_end must be between 0 and 1.")

    if round(fractional_test_end / fractional_block_size) < 2:
        raise InvalidInputError(
            "fractional_test_end must be at least twice the fractional_block_size."
        )

    # Check that fractional test end is a multiple of fractional block size.
    if round((fractional_test_end / fractional_block_size) % 1.0, 3) != 0:
        raise InvalidInputError("fractional_test_end must be a multiple of fractional_block_size.")

    if initial_block_size <= 0 or initial_block_size > 1:
        raise InvalidInputError("initial_block_size must be between 0 and 1.")

    if final_block_size <= 0 or final_block_size > 1:
        raise InvalidInputError("final_block_size must be between 0 and 1.")

    t_test_sidedness = t_test_sidedness.lower()
    if t_test_sidedness not in ["two-sided", "less", "greater"]:
        raise InvalidInputError(
            "t_test_sidedness must be either 'two-sided', 'less', or 'greater'."
        )

    # Calculate the number of repeats.
    n_repeats = round(fractional_test_end / fractional_block_size) + 1  # + 1 for the initial block

    # Calculate the number of samples to discard between repeats.
    n_discard = round(n_samples * fractional_block_size)

    # Calculate the p values with their indices.
    p_vals = _np.zeros(n_repeats, dtype=_np.float64)
    p_val_indices = _np.zeros(n_repeats, dtype=int)
    time_vals = _np.zeros(n_repeats, dtype=times_valid.dtype)

    # Loop over and calculate the p values.
    for i in range(n_repeats):
        # Truncate the data.
        idx = n_discard * i
        truncated_data = data[:, idx:]
        # Get the number of samples in the truncated data.
        n_truncated_samples = truncated_data.shape[1]
        # Get the initial and final blocks.
        initial_block = truncated_data[:, : round(n_truncated_samples * initial_block_size)].mean(
            axis=1
        )
        final_block = truncated_data[:, -round(n_truncated_samples * final_block_size) :].mean(
            axis=1
        )
        # Compute the paired t-test.
        p_vals[i] = _ttest_rel(initial_block, final_block, alternative=t_test_sidedness)[1]
        p_val_indices[i] = idx
        time_vals[i] = times_valid[idx]

    return p_vals, time_vals


def detect_equilibration_paired_t_test(
    data: _npt.NDArray[_np.float64],
    times: _Optional[_npt.NDArray[_np.float64]] = None,
    p_threshold: float = 0.05,
    fractional_block_size: float = 0.125,
    fractional_test_end: float = 0.5,
    initial_block_size: float = 0.1,
    final_block_size: float = 0.5,
    t_test_sidedness: str = "two-sided",
    plot: bool = False,
    plot_name: _Union[str, _Path] = "equilibration_paired_t_test.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> _Union[_np.int64, _np.float64]:
    r"""
    Detect the equilibration time of a time series by performing a paired
    t-test between initial and final portions of the time series. This is repeated
    , discarding more data from the time series between repeats. If the p-value
    is greater than the threshold, there is no significant evidence that the data is
    no equilibrated and the timeseries is taken to be equilibrated at this time
    point. This test may be useful when we care only about systematic bias in the
    data, and do not care about detecting inter-run differences.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    times : np.ndarray, optional
        The times at which the data was sampled. If this is
        not provided, the indices of the data will be used.

    p_threshold : float, optional
        The p-value threshold to use. The default is 0.05.

    fractional_block_size : float, optional
        The fraction of data to discard between repeats. The default is 0.125.

    fractional_test_end : float, optional
        The fraction of the time series to use in the final test. The default is 0.5.

    initial_block_size : float, optional
        The fraction of the truncated time series to use for the "before" portion
        of the paired t-test. The default is 0.1.

    final_block_size : float, optional
        The fraction of the truncated time series to use for the "after" portion
        of the paired t-test. The default is 0.5.

    t_test_sidedness : str, optional
        The sidedness of the paired t-test. This can be either 'two-sided', 'less',
        or 'greater'. The default is 'two-sided'.

    plot : bool, optional
        Whether to plot the p-values. The default is False.

    plot_name : str | Path, optional
        The name of the plot file. The default is 'equilibration_paired_t_test.png'.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    Returns
    -------
    np.float64 | np.int64
        The time (or index, if no times are supplied) at which
        the time series is equilibrated.
    """
    # Validate data.
    data = check_data(data, one_dim_allowed=False)
    n_runs, n_samples = data.shape

    # If times is None, units of time are indices.
    if times is None:
        time_units = "index"
        # Convert times to indices.
        times = _np.arange(n_samples, dtype=_np.int64)

    # Check that user options (not checked in get_paired_t_p_timeseries) are valid.
    if p_threshold <= 0 or p_threshold > 0.7:
        raise InvalidInputError("p_threshold must be between 0 and 0.7.")

    # Get the p value timeseries and n_discard.
    p_vals, times_used = get_paired_t_p_timeseries(
        data=data,
        times=times,
        fractional_block_size=fractional_block_size,
        fractional_test_end=fractional_test_end,
        initial_block_size=initial_block_size,
        final_block_size=final_block_size,
        t_test_sidedness=t_test_sidedness,
    )

    # Get the index of the first p value that is greater than the threshold.
    meets_threshold = p_vals > p_threshold
    if not any(meets_threshold):
        raise EquilibrationNotDetectedError(
            f"No p values are greater than the threshold of {p_threshold}."
        )
    equil_time: _Union[_np.float64, _np.int64] = times_used[_np.argmax(meets_threshold)]

    # Plot the p values.
    if plot:
        fig = _plt.figure(figsize=(6, 4))
        gridspec_obj = _gridspec.GridSpec(1, 1, figure=fig)
        plot_equilibration_paired_t_test(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=data,
            p_values=p_vals,
            data_times=times,
            p_times=times_used,
            p_threshold=p_threshold,
            time_units=time_units,
            data_y_label=data_y_label,
        )

        fig.savefig(str(plot_name), dpi=300, bbox_inches="tight")

    return equil_time
