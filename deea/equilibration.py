"""Functions for selecting the equilibration time of a time series."""

from pathlib import Path
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import ttest_rel

from ._exceptions import AnalysisError, EquilibrationNotDetectedError, InvalidInputError
from ._validation import check_data
from .ess import (
    ess_chodera,
    ess_inter_variance,
    ess_lugsail_variance,
    statistical_inefficiency_chodera,
    statistical_inefficiency_inter_variance,
    statistical_inefficiency_lugsail_variance,
)
from .plot import plot_equilibration_max_ess, plot_equilibration_paired_t_test


def get_ess_series(
    data: np.ndarray,
    times: _Optional[np.ndarray] = None,
    method: str = "lugsail",
    min_ess: int = 1,
    n_pow: float = 1 / 3,
) -> _Tuple[np.ndarray, np.ndarray]:
    """
    Return a list of effective sample sizes from a (set of) timeseries.
    The ESS values are obtained by computing the ESS at each time point,
    discarding all samples before the time point. The index of the time
    point with the maximum ESS is taken to be the point of equilibration.

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

    method : str, optional
        The method to use to compute the ESS. This can be either
        'lugsail' or 'inter'. The default is 'lugsail'. 'lugsail'
        uses the lugsail replicated batch means variance estimate
        and 'inter' uses the inter-run variance estimate to compute
        the ESS.

    min_ess : int, optional, default=0
        The minimum ESS to accept. If the maximum ESS is less than
        this value, an EquilibrationNotDetectedError is raised.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3. If the method
        is 'inter', this parameter is ignored.

    Returns
    -------
    np.ndarray
        The ESS values.

    np.ndarray
        The times at which the ESS values were calculated.
    """
    # Check that method is valid.
    method = method.lower()
    if method not in ["lugsail", "inter", "chodera"]:
        raise InvalidInputError("method must be 'lugsail', 'inter', or 'chodera'.")

    # Check that the data is valid.
    one_dim_allowed = method in ["lugsail", "chodera"]
    data = check_data(data, one_dim_allowed=one_dim_allowed)
    n_runs, n_samples = data.shape

    # Convert times to indices if necessary.
    if times is None:
        times = np.arange(n_samples)

    # Check that min_ess is valid.
    if min_ess <= 0 or min_ess > n_samples * n_runs:
        raise InvalidInputError(
            "min_ess must be between 0 and the total number of samples."
        )

    # Check that times is match the number of samples.
    if n_samples != len(times):
        raise InvalidInputError(
            "Times must have the same length as the number of samples."
        )

    # Exclude the last 5 % of data from the ESS calculation
    # to ensure that we don't throw away too much data.
    final_idx = round(n_samples * 0.95)
    n_discarded = n_samples - final_idx
    # Make sure that this corresponds to at least 5 samples.
    if n_discarded < 5:
        raise AnalysisError("The time series is too short.")

    # Compute the ESS at each time point.
    ess_vals = np.zeros(n_samples - n_discarded)
    time_vals = np.zeros(n_samples - n_discarded)

    for i in range(n_samples - n_discarded):
        # Get the truncated data.
        truncated_data = data[:, i:]
        # Compute the ESS.
        if method == "lugsail":
            ess_vals[i] = ess_lugsail_variance(truncated_data, n_pow=n_pow)
        elif method == "chodera":
            ess_vals[i] = ess_chodera(truncated_data)
        else:  # method == 'inter'
            ess_vals[i] = ess_inter_variance(truncated_data)
        # Get the time.
        time_vals[i] = times[i]

    return ess_vals, time_vals


def detect_equilibration_max_ess(
    data: np.ndarray,
    times: _Optional[np.ndarray] = None,
    method: str = "lugsail",
    min_ess: int = 1,
    n_pow: float = 1 / 3,
    plot: bool = False,
    plot_name: _Union[str, Path] = "equilibration_ess.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> _Tuple[int, float, float]:
    r"""
    Detect the equilibration time of a time series by finding the maximum
    effective sample size (ESS) of the time series. This is done by
    computing the ESS at each time point, discarding all samples before
    the time point. The index of the time point with the maximum ESS is taken
    to be the point of equilibration.

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

    method : str, optional
        The method to use to compute the ESS. This can be either
        'lugsail' or 'inter'. The default is 'lugsail'. 'lugsail'
        uses the lugsail replicated batch means variance estimate
        and 'inter' uses the inter-run variance estimate to compute
        the ESS.

    min_ess : int, optional, default=0
        The minimum ESS to accept. If the maximum ESS is less than
        this value, an EquilibrationNotDetectedError is raised.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3. If the method
        is 'inter', this parameter is ignored.

    plot : bool, optional
        Whether to plot the ESS curve. The default is False.

    plot_name : str | Path, optional
        The name of the plot file. The default is 'equilibration_ess.png'.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    Returns
    -------
    int
        The time point at which the time series is equilibrated.

    float
        The statistical inefficiency at the equilibration point.

    float
        The effective sample size at the equilibration point.
    """
    # Check that data is valid.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # If times is None, units of time are indices.
    if times is None:
        time_units = "index"
        # Convert times to indices.
        times = np.arange(n_samples)

    # Get the ESS timeseries
    ess_vals, ess_times = get_ess_series(
        data=data, times=times, method=method, min_ess=min_ess, n_pow=n_pow
    )

    # Get the index of the maximum ESS.
    equil_idx = np.argmax(ess_vals)
    equil_time = ess_times[equil_idx]
    equil_ess = ess_vals[equil_idx]

    # Check that the ESS is greater than the minimum ESS.
    if equil_ess < min_ess:
        raise EquilibrationNotDetectedError(
            f"The maximum ESS of {equil_ess} is less than the minimum ESS of {min_ess}."
        )

    # Now compute the statistical inefficiency at this time point.
    equil_data = data[:, equil_idx:]
    if method == "lugsail":
        equil_g = statistical_inefficiency_lugsail_variance(equil_data, n_pow=n_pow)
    elif method == "chodera":
        equil_g = statistical_inefficiency_chodera(equil_data)
    else:  # method == 'inter'
        equil_g = statistical_inefficiency_inter_variance(equil_data)

    if plot:
        # Create a figure.
        fig = plt.figure(figsize=(6, 4))
        gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

        # Plot the ESS.
        plot_equilibration_max_ess(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=data,
            ess_series=ess_vals,
            data_times=times,
            ess_times=ess_times,
            time_units=time_units,
            data_y_label=data_y_label,
        )

        fig.savefig(plot_name, dpi=600, bbox_inches="tight")

    # Return the equilibration index, statistical inefficiency, and ESS.
    return equil_time, equil_g, equil_ess


def get_paired_t_p_timeseries(
    data: np.ndarray,
    times: _Optional[np.ndarray] = None,
    fractional_block_size: float = 0.125,
    fractional_test_end: float = 0.5,
    initial_block_size: float = 0.1,
    final_block_size: float = 0.5,
    t_test_sidedness: str = "two-sided",
) -> _Tuple[np.ndarray, np.ndarray]:
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
    if times is None:
        times = np.arange(n_samples)

    # Check that times is match the number of samples.
    if n_samples != len(times):
        raise InvalidInputError(
            "Times must have the same length as the number of samples."
        )

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
        raise InvalidInputError(
            "fractional_test_end must be a multiple of fractional_block_size."
        )

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
    n_repeats = (
        round(fractional_test_end / fractional_block_size) + 1
    )  # + 1 for the initial block

    # Calculate the number of samples to discard between repeats.
    n_discard = round(n_samples * fractional_block_size)

    # Calculate the p values with their indices.
    p_vals = np.zeros(n_repeats)
    p_val_indices = np.zeros(n_repeats, dtype=int)
    time_vals = np.zeros(n_repeats)

    # Loop over and calculate the p values.
    for i in range(n_repeats):
        # Truncate the data.
        idx = n_discard * i
        truncated_data = data[:, idx:]
        # Get the number of samples in the truncated data.
        n_truncated_samples = truncated_data.shape[1]
        # Get the initial and final blocks.
        initial_block = truncated_data[
            :, : round(n_truncated_samples * initial_block_size)
        ].mean(axis=1)
        final_block = truncated_data[
            :, -round(n_truncated_samples * final_block_size) :
        ].mean(axis=1)
        # Compute the paired t-test.
        p_vals[i] = ttest_rel(initial_block, final_block, alternative=t_test_sidedness)[
            1
        ]
        p_val_indices[i] = idx
        time_vals[i] = times[idx]

    return p_vals, time_vals


def detect_equilibration_paired_t_test(
    data: np.ndarray,
    times: _Optional[np.ndarray] = None,
    p_threshold: float = 0.05,
    fractional_block_size: float = 0.125,
    fractional_test_end: float = 0.5,
    initial_block_size: float = 0.1,
    final_block_size: float = 0.5,
    t_test_sidedness: str = "two-sided",
    plot: bool = False,
    plot_name: _Union[str, Path] = "equilibration_paired_t_test.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> int:
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
    int
        The time point at which the time series is equilibrated.
    """
    # Validate dtata.
    data = check_data(data, one_dim_allowed=False)
    n_runs, n_samples = data.shape

    # If times is None, units of time are indices.
    if times is None:
        time_units = "index"
        # Convert times to indices.
        times = np.arange(n_samples)

    # Check that user options (not checked in get_paired_t_p_timeseries) are valid.
    if p_threshold <= 0 or p_threshold > 0.7:
        raise InvalidInputError("p_threshold must be between 0 and 0.7.")

    # Get the p value timeseries and n_discard.
    indices = np.arange(n_samples)
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
    equil_time = times_used[np.argmax(meets_threshold)]

    # Plot the p values.
    if plot:
        fig = plt.figure(figsize=(6, 4))
        gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)
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

        fig.savefig(plot_name, dpi=600, bbox_inches="tight")

    return equil_time
