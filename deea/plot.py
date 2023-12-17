"""Plotting functions."""

import matplotlib.pyplot as plt
from matplotlib import figure, gridspec
from matplotlib.axes import Axes

plt.style.use("ggplot")
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as np
import scipy.stats as _stats

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

    # Decide the thickness of the individual run lines.
    alpha_runs = 1.0 if n_runs == 1 else 0.3

    # Plot the data.
    for i in range(n_runs):
        label = None if n_runs == 1 else f"Run {i + 1}"
        ax.plot(times, data[i, :], alpha=alpha_runs, label=label)

    # If we have more than one run, plot the mean.
    if n_runs > 1:
        ax.plot(times, data.mean(axis=0), color="black", label="Mean")

    # Plot the confidence interval.
    if show_ci and n_runs > 1:
        means = data.mean(axis=0)
        conf_int = (
            _stats.t.interval(
                0.95,
                n_runs - 1,
                means,
                scale=_stats.sem(data),
            )[1]
            - means
        )  # 95 % C.I.

        # Plot the confidence interval.
        ax.fill_between(
            times, means - conf_int, means + conf_int, alpha=0.3, color="grey"
        )

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
