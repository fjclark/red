"""Tests for the plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import gridspec

from red.equilibration import get_paired_t_p_timeseries

from .._exceptions import InvalidInputError
from ..equilibration import get_sse_series_init_seq
from ..ess import get_ess_series_init_seq
from ..plot import (
    plot_equilibration_min_sse,
    plot_equilibration_paired_t_test,
    plot_p_values,
    plot_sse,
    plot_timeseries,
)
from . import example_times, example_timeseries

# Set to True to save the plots.
SAVE_PLOTS = True


def test_plot_timeseries(example_timeseries, example_times):
    """Test plotting a timeseries."""
    # Create a figure.
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()

        # Plot the timeseries.
        plot_timeseries(ax, example_timeseries, example_times)
        # fig.savefig("test_plot_timeseries.png")

        # Check that it works for a single run.
        fig, ax = plt.subplots()
        plot_timeseries(ax, example_timeseries[0], example_times)

    if SAVE_PLOTS:
        fig.savefig("test_plot_timeseries_single.png")

    # Try with n_blocks == 0.
    plot_timeseries(ax, example_timeseries, example_times, n_blocks=0)

    # Test that invalid input raises an error.
    with pytest.raises(InvalidInputError):
        plot_timeseries(ax, example_timeseries, example_times, n_blocks=-1)

    with pytest.raises(InvalidInputError):
        plot_timeseries(ax, example_timeseries, list(example_times))

    with pytest.raises(InvalidInputError):
        plot_timeseries(ax, example_timeseries, example_times[:-2])

    with pytest.raises(InvalidInputError):
        plot_timeseries(ax, example_timeseries, example_timeseries)


def test_plot_sse(example_timeseries, example_times):
    """Test plotting the sum of squared errors."""
    # Create a figure.
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()

        # Compute the SSE for the example timeseries.
        sse_vals, lag_times = get_sse_series_init_seq(
            data=example_timeseries, smooth_lag_times=True
        )
        times = example_times[: len(sse_vals)]

        # Plot the SSE.
        plot_sse(ax=ax, sse=sse_vals, max_lags=lag_times, window_sizes=None, times=times)

    if SAVE_PLOTS:
        fig.savefig("test_plot_sse.png")

    # Check that invalid input raises an error.
    with pytest.raises(InvalidInputError):
        plot_sse(ax, sse_vals, max_lags=None, window_sizes=None, times=times[:-2])

    with pytest.raises(InvalidInputError):
        # Make sse_vals a 2D array.
        plot_sse(
            ax,
            np.array([sse_vals, sse_vals]),
            max_lags=lag_times,
            window_sizes=lag_times,
            times=times,
        )

    with pytest.raises(InvalidInputError):
        plot_sse(
            ax,
            list(sse_vals),
            max_lags=None,
            window_sizes=None,
            times=times,
        )

    with pytest.raises(InvalidInputError):
        plot_sse(
            ax,
            sse_vals,
            max_lags=lag_times,
            window_sizes=lag_times,
            times=times,
        )


def test_plot_equilibration_min_sse(example_timeseries, example_times):
    """Test plotting the equilibration detection based on the minimum SSE."""
    # Take mean to speed things up, but plot the original data to
    # test this works.
    example_timeseries_mean = example_timeseries.mean(axis=0)

    # Create a figure.
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(6, 4))
        gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

        # Compute the SSE for the example timeseries.
        sse_vals, _ = get_sse_series_init_seq(data=example_timeseries_mean, smooth_lag_times=True)
        times = example_times[: len(sse_vals)]

        # Plot the equilibration detection.
        plot_equilibration_min_sse(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=example_timeseries,
            sse_series=sse_vals,
            max_lag_series=None,
            data_times=example_times,
            sse_times=times,
        )

    if SAVE_PLOTS:
        fig.savefig("test_plot_equilibration_min_sse.png", bbox_inches="tight")


def test_plot_equilibration_max_ess(example_timeseries, example_times):
    """Test plotting the equilibration detection based on the maximum SSE."""
    # Take mean to speed things up, but plot the original data to
    # test this works.
    example_timeseries_mean = example_timeseries.mean(axis=0)

    # Create a figure.
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(6, 4))
        gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

        # Compute the ESS for the example timeseries.
        ess_vals, max_lag_series = get_ess_series_init_seq(
            data=example_timeseries_mean, smooth_lag_times=True
        )
        times = example_times[: len(ess_vals)]

        # Plot the equilibration detection.
        plot_equilibration_min_sse(
            fig=fig,
            subplot_spec=gridspec_obj[0],
            data=example_timeseries,
            sse_series=1 / ess_vals,
            max_lag_series=max_lag_series,
            data_times=example_times,
            variance_y_label="ESS",
            sse_times=times,
        )

    if SAVE_PLOTS:
        fig.savefig("test_plot_equilibration_max_ess.png", bbox_inches="tight")


def test_plot_p_values(example_timeseries, example_times):
    """Test plotting the p-values of the paired t-test."""
    # Create a figure.
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()
        _, n_samples = example_timeseries.shape

        # Compute the p-values for the example timeseries.
        p_values, times_used = get_paired_t_p_timeseries(
            example_timeseries, example_times, t_test_sidedness="two-sided"
        )

        # Plot the p-values.
        plot_p_values(ax, p_values, times_used)

    if SAVE_PLOTS:
        fig.savefig("test_plot_p_values.png")

    # Check that invalid input raises an error.
    with pytest.raises(InvalidInputError):
        plot_p_values(ax, p_values, example_times[:-2])

    with pytest.raises(InvalidInputError):
        plot_p_values(ax, list(p_values), example_timeseries)

    with pytest.raises(InvalidInputError):
        # Make p_values a 2D array.
        plot_p_values(ax, np.array([p_values, p_values]), example_timeseries)


def test_plot_equilibration_paired_t_test(example_timeseries, example_times):
    """Test plotting the equilibration detection based on the paired t-test."""
    # Create a figure.
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(6, 4))
        gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

    # Compute the p-values for the example timeseries.
    p_values, p_times = get_paired_t_p_timeseries(example_timeseries, example_times)

    # Plot the equilibration detection.
    plot_equilibration_paired_t_test(
        fig, gridspec_obj[0], example_timeseries, p_values, example_times, p_times
    )

    if SAVE_PLOTS:
        fig.savefig("test_plot_equilibration_paired_t_test.png", bbox_inches="tight")
