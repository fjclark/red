"""Plotting functions."""

from typing import List as _List

import matplotlib.pyplot as plt
from matplotlib import figure, gridspec
from matplotlib.axes import Axes

plt.style.use("ggplot")
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as np
from matplotlib.lines import Line2D

from ._exceptions import InvalidInputError
from ._validation import check_data


def plot_timeseries(
    ax: Axes,
    data: np.ndarray,
    times: np.ndarray,
    show_ci: bool = True,
    n_blocks: int = 100,
    time_units: str = "ns",
    y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> None:
    r"""
    Plot the (multi-run) time series data.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.

    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples), or (n_samples,) if there
        is only one run.

    times : np.ndarray
        The times at which the data was sampled. This
        should have shape (n_samples,).

    show_ci : bool, optional
        Whether to show the 95%  t-based confidence interval.

    n_blocks : int, optional
        The number of blocks to use for block averaging. This
        makes trends clearer. If 0, no block averaging is
        performed.

    time_units : str, optional
        The units of time. The default is "ns".

    y_label : str, optional
        The y-axis label. The default is "$\Delta G$ / kcal mol$^{-1}$".
    """
    # Check that data is valid.
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape

    # Check that times is valid.
    if not isinstance(times, np.ndarray):
        raise InvalidInputError("Times must be a numpy array.")

    if times.ndim != 1:
        raise InvalidInputError("Times must be one dimensional.")

    if times.shape[0] != n_samples:
        raise InvalidInputError(
            "Times must have the same length as the number of samples."
        )

    if n_blocks < 0 or n_blocks > n_samples:
        raise InvalidInputError(
            "n_blocks must be greater than or equal to 0 and less than or equal to the number of samples."
        )

    if n_blocks == 0:
        n_blocks = n_samples

    # Block-average the data. Need to trim the data so that it is divisible by
    # the number of blocks.
    block_size = n_samples // n_blocks
    n_samples_end = n_samples - (n_samples % n_blocks)
    times = times[:n_samples_end]
    data = data[:, :n_samples_end]
    data = data.reshape(n_runs, n_blocks, block_size).mean(axis=2)  # type: ignore
    times = times.reshape(n_blocks, block_size).mean(axis=1)

    # Decide the transparency of the individual run lines.
    alpha_runs = 1.0 if n_runs == 1 else 0.5

    # Plot the data.
    for i in range(n_runs):
        label = None if n_runs == 1 else f"Run {i + 1}"
        ax.plot(times, data[i, :], alpha=alpha_runs, label=label)

    # If we have more than one run, plot the mean.
    if n_runs > 1:
        ax.plot(times, data.mean(axis=0), color="black", label="Mean")

    ## Plot the confidence interval.
    # if show_ci and n_runs > 1:
    # means = data.mean(axis=0)
    # conf_int = (
    # _stats.t.interval(
    # 0.95,
    # n_runs - 1,
    # means,
    # scale=_stats.sem(data),
    # )[1]
    # - means
    # )  # 95 % C.I.

    ## Plot the confidence interval.
    # ax.fill_between(
    # times, means - conf_int, means + conf_int, alpha=0.3, color="grey"
    # )

    # Only show the legend if there is more than one run.
    if n_runs > 1:
        ax.legend()

    # Set the axis labels.
    ax.set_xlabel(f"Time / {time_units}")
    ax.set_ylabel(y_label)


def plot_p_values(
    ax: Axes,
    p_values: np.ndarray,
    times: np.ndarray,
    p_threshold: float = 0.05,
    time_units: str = "ns",
    threshold_times: _Optional[np.ndarray] = None,
) -> None:
    """
    Plot the p-values of the paired t-test.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.

    p_values : np.ndarray
        The p-values of the paired t-test.

    times : np.ndarray
        The times at which the data was sampled.

    p_threshold : float, optional
        The p-value threshold to use. The default is 0.05.

    time_units : str, optional
        The units of time. The default is "ns".

    threshold_times : np.ndarray, optional
        The times to plot the p-value threshold at. If None, this is
        set to the times at which the data was sampled. Useful when
        using this plot underneath a time series plot.
    """
    # Check that p_values is valid.
    if not isinstance(p_values, np.ndarray) or not isinstance(times, np.ndarray):
        raise InvalidInputError("p_values and times must be numpy arrays.")

    if p_values.ndim != 1:
        raise InvalidInputError("p_values must be one dimensional.")

    if p_values.shape[0] != times.shape[0]:
        raise InvalidInputError(
            "p_values must have the same length as the number of samples."
        )

    if threshold_times is None:
        threshold_times = times

    # Plot the p-values.
    ax.scatter(times, p_values, color="black", label="p-value")

    # Plot the p-value threshold.
    ax.plot(
        threshold_times,
        np.full(threshold_times.shape, p_threshold),
        color="black",
        linestyle="-",
        linewidth=0.5,
        label="p-value threshold",
    )

    # Shade the region where the p-value is less than the threshold.
    ax.fill_between(
        threshold_times,
        0,
        p_threshold,
        alpha=0.3,
        # Black
        color="black",
    )

    # Plot a vertical dashed line at the first time point where the p-value is
    # greater than the threshold.
    ax.axvline(
        x=times[p_values > p_threshold][0],
        color="black",
        linestyle="--",
        label=f"Equilibration Time = {times[p_values > p_threshold][0]:.3g} {time_units}",
    )

    ax.legend()

    # Set the axis labels.
    ax.set_xlabel(f"Time / {time_units}")
    ax.set_ylabel("$p$-value")


def plot_ess(
    ax: Axes, ess: np.ndarray, times: np.ndarray, time_units: str = "ns"
) -> None:
    """
    Plot the ESS estimate against time.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.

    ess : np.ndarray
        The ESS estimate.

    times : np.ndarray
        The times at which the data was sampled.

    time_units : str, optional
        The units of time. The default is "ns".
    """
    # Check that ess is valid.
    if not isinstance(ess, np.ndarray) or not isinstance(times, np.ndarray):
        raise InvalidInputError("ess and times must be numpy arrays.")

    if ess.ndim != 1:
        raise InvalidInputError("ess must be one dimensional.")

    if ess.shape[0] != times.shape[0]:
        raise InvalidInputError(
            "ess must have the same length as the number of samples."
        )

    # Plot the ESS.
    ax.plot(times, ess, color="black")

    # Plot a vertical dashed line at the maximum ESS.
    max_ess = ess.max()
    ax.axvline(
        x=times[ess.argmax()],
        color="black",
        linestyle="--",
        label=f"Equilibration Time = {times[ess.argmax()]:.3g} {time_units}",
    )

    ax.legend()

    # Set the axis labels.
    ax.set_xlabel(f"Time / {time_units}")
    ax.set_ylabel("ESS")


def plot_sse(
    ax: Axes,
    sse: np.ndarray,
    max_lags: _Optional[np.ndarray],
    window_sizes: _Optional[np.ndarray],
    times: np.ndarray,
    time_units: str = "ns",
    variance_y_label: str = r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$",
    reciprocal: bool = True,
) -> _Tuple[_List[Line2D], _List[str]]:
    """
    Plot the squared standard error (SSE) estimate against time.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.

    sse : np.ndarray
        The SSE estimate.

    max_lags : np.ndarray, optional, default=None
        The maximum lag times.

    window_sizes : np.ndarray, optional, default=None
        The window sizes.

    times : np.ndarray
        The times at which the data was sampled.

    time_units : str, optional
        The units of time. The default is "ns".

    variance_y_label : str, optional
        The y-axis label for the variance. The default is
        r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$".

    reciprocal : bool, optional, default=True
        Whether to plot the reciprocal of the SSE.

    Returns
    -------
    handles : List[Line2D]
        The handles for the legend.

    labels : List[str]
        The labels for the legend.
    """
    # Check that sse is valid.
    if not isinstance(sse, np.ndarray) or not isinstance(times, np.ndarray):
        raise InvalidInputError("sse and times must be numpy arrays.")

    if sse.ndim != 1:
        raise InvalidInputError("sse must be one dimensional.")

    if sse.shape[0] != times.shape[0]:
        raise InvalidInputError(
            "sse must have the same length as the number of samples."
        )

    if max_lags is not None and window_sizes is not None:
        raise InvalidInputError(
            "Only one of max_lags and window_sizes can be supplied."
        )

    # Plot the SSE.
    to_plot = 1 / sse if reciprocal else sse
    ax.plot(times, to_plot, color="black", label="1/SSE")

    # If lags is not None, plot the lag times on a different y axis.
    if max_lags is not None or window_sizes is not None:
        label = "Max Lag Index" if window_sizes is None else "Window Size"
        to_plot = max_lags if window_sizes is None else window_sizes
        ax2 = ax.twinx()
        # Get the second colour from the colour cycle.
        ax2.set_prop_cycle(color=[plt.rcParams["axes.prop_cycle"].by_key()["color"][1]])
        ax2.plot(times, to_plot, alpha=0.8, label=label)
        ax2.set_ylabel(label)

    # Plot a vertical dashed line at the minimum SSE.
    ax.axvline(
        x=times[sse.argmin()],
        color="black",
        linestyle="--",
        label=f"Equilibration Time = {times[sse.argmin()]:.3g} {time_units}",
    )

    # Combine the legends from both axes.
    handles, labels = ax.get_legend_handles_labels()
    if max_lags is not None or window_sizes is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

    ax.legend(handles, labels)

    # Set the axis labels.
    ax.set_xlabel(f"Time / {time_units}")
    ax.set_ylabel(variance_y_label)

    return handles, labels


def plot_equilibration_paired_t_test(
    fig: figure.Figure,
    subplot_spec: gridspec.SubplotSpec,
    data: np.ndarray,
    p_values: np.ndarray,
    data_times: np.ndarray,
    p_times: np.ndarray,
    p_threshold: float = 0.05,
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> _Tuple[Axes, Axes]:
    r"""
    Plot the p-values of the paired t-test against time, underneath the
    time series data.

    Parameters
    ----------
    fig : plt.Figure
        The figure to plot on.

    gridspec_obj : plt.GridSpec
        The gridspec to use for the plot.

    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples), or (n_samples,) if there
        is only one run.

    p_values : np.ndarray
        The p-values of the paired t-test.

    data_times : np.ndarray
        The times at which the data was sampled.

    p_times : np.ndarray
        The times at which the paired t-test was performed.

    p_threshold : float, optional
        The p-value threshold to use. The default is 0.05.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    Returns
    -------
    ax_top : Axes
        The axes for the time series data.

    ax_bottom : Axes
        The axes for the p-values.
    """
    # We need to split the gridspec into two subplots, one for the time series data (above)
    # and one for the p-values (below). Share x-axis but not y-axis.
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.05)
    ax_top = fig.add_subplot(gs0[0])
    ax_bottom = fig.add_subplot(gs0[1], sharex=ax_top)

    # Plot the time series data on the top axis.
    plot_timeseries(ax_top, data, data_times)
    # Add dashed vertical line at the equilibration time.
    ax_top.axvline(
        x=p_times[p_values > p_threshold][0],
        color="black",
        linestyle="--",
    )

    # Plot the p-values on the bottom axis.
    plot_p_values(
        ax_bottom,
        p_values,
        p_times,
        p_threshold=p_threshold,
        threshold_times=data_times,
    )

    # Set the axis labels.
    ax_top.set_xlabel(f"Time / {time_units}")
    ax_top.set_ylabel(data_y_label)

    # Move the legends to the side of the plot.
    ax_top.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    ax_bottom.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Hide the x tick labels for the top axis.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", length=0)
    ax_top.set_xlabel("")

    return ax_top, ax_bottom


def plot_equilibration_max_ess(
    fig: figure.Figure,
    subplot_spec: gridspec.SubplotSpec,
    data: np.ndarray,
    ess_series: np.ndarray,
    data_times: np.ndarray,
    ess_times: np.ndarray,
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
) -> _Tuple[Axes, Axes]:
    r"""
    Plot the effective sample size (ESS) estimates against time,
    underneath the time series data.

    Parameters
    ----------
    fig : plt.Figure
        The figure to plot on.

    gridspec_obj : plt.GridSpec
        The gridspec to use for the plot.

    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples), or (n_samples,) if there
        is only one run.

    ess_series : np.ndarray
        The ESS series.

    data_times : np.ndarray
        The times at which the data was sampled.

    ess_times : np.ndarray
        The times at which the ESS was computed.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    Returns
    -------
    ax_top : Axes
        The axes for the time series data.

    ax_bottom : Axes
        The axes for the p-values.
    """
    # We need to split the gridspec into two subplots, one for the time series data (above)
    # and one for the p-values (below). Share x-axis but not y-axis.
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.05)
    ax_top = fig.add_subplot(gs0[0])
    ax_bottom = fig.add_subplot(gs0[1], sharex=ax_top)

    # Plot the time series data on the top axis.
    plot_timeseries(ax_top, data, data_times)
    # Add dashed vertical line at the equilibration time.
    ax_top.axvline(
        x=data_times[ess_series.argmax()],
        color="black",
        linestyle="--",
    )

    # Plot the ess on the bottom axis.
    plot_ess(ax_bottom, ess_series, ess_times)

    # Set the axis labels.
    ax_top.set_xlabel(f"Time / {time_units}")
    ax_top.set_ylabel(data_y_label)

    # Move the legends to the side of the plot.
    ax_top.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    ax_bottom.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Hide the x tick labels for the top axis.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", length=0)
    ax_top.set_xlabel("")

    return ax_top, ax_bottom


def plot_equilibration_min_sse(
    fig: figure.Figure,
    subplot_spec: gridspec.SubplotSpec,
    data: np.ndarray,
    sse_series: np.ndarray,
    data_times: np.ndarray,
    sse_times: np.ndarray,
    max_lag_series: _Optional[np.ndarray] = None,
    window_size_series: _Optional[np.ndarray] = None,
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
    variance_y_label=r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$",
    reciprocal: bool = True,
) -> _Tuple[Axes, Axes]:
    r"""
    Plot the (reciprocal of the) squared standard error (SSE)
    estimates against time, underneath the time series data.

    Parameters
    ----------
    fig : plt.Figure
        The figure to plot on.

    gridspec_obj : plt.GridSpec
        The gridspec to use for the plot.

    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples), or (n_samples,) if there
        is only one run.

    sse_series : np.ndarray
        The SSE series.

    data_times : np.ndarray
        The times at which the data was sampled.

    sse_times : np.ndarray
        The times at which the ESS was computed.

    max_lag_series : np.ndarray, optional
        The lag series. If None, the lag times are not
        plotted. If supplied, they are plotted on the
        bottom axis.

    window_size_series : np.ndarray, optional
        The window size series. If None, the window sizes
        are not plotted. If supplied, they are plotted on
        the bottom axis.

    time_units : str, optional
        The units of time. The default is "ns".

    data_y_label : str, optional
        The y-axis label for the time series data. The default is
        "$\Delta G$ / kcal mol$^{-1}$".

    variance_y_label : str, optional
        The y-axis label for the variance. The default is
        r"$\frac{1}{\sigma^2(\Delta G)}$ / kcal$^{-2}$ mol$^2$".

    reciprocal : bool, optional, default=True
        Whether to plot the reciprocal of the SSE.

    Returns
    -------
    ax_top : Axes
        The axes for the time series data.

    ax_bottom : Axes
        The axes for the p-values.
    """
    data = check_data(data, one_dim_allowed=True)
    n_runs, _ = data.shape

    # We need to split the gridspec into two subplots, one for the time series data (above)
    # and one for the p-values (below). Share x-axis but not y-axis.
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, hspace=0.05)
    ax_top = fig.add_subplot(gs0[0])
    ax_bottom = fig.add_subplot(gs0[1], sharex=ax_top)

    # Plot the time series data on the top axis.
    plot_timeseries(ax_top, data, data_times)
    # Add dashed vertical line at the equilibration time.
    ax_top.axvline(
        x=sse_times[sse_series.argmin()],
        color="black",
        linestyle="--",
    )

    # Plot the sse on the bottom axis.
    sse_handles, sse_labels = plot_sse(
        ax_bottom,
        sse_series,
        max_lag_series,
        window_size_series,
        sse_times,
        variance_y_label=variance_y_label,
        reciprocal=reciprocal,
    )

    # Set the axis labels.
    ax_top.set_xlabel(f"Time / {time_units}")
    ax_top.set_ylabel(data_y_label)

    # Move the legends to the side of the plot.
    if n_runs > 1:
        ax_top.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    is_second_axis = max_lag_series is not None or window_size_series is not None
    side_shift_bottom = 1.15 if is_second_axis else 1.05
    # Remove the 1/SSE label if there isn't a second axis.
    if not is_second_axis:
        sse_labels.pop(0)
        sse_handles.pop(0)
    ax_bottom.legend(
        sse_handles,
        sse_labels,
        bbox_to_anchor=(side_shift_bottom, 0.5),
        loc="center left",
    )

    # Hide the x tick labels for the top axis.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", length=0)
    ax_top.set_xlabel("")

    return ax_top, ax_bottom
