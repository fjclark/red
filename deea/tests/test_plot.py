"""Tests for the plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import gridspec

from deea.equilibration import get_ess_series, get_paired_t_p_timeseries

from .._exceptions import InvalidInputError
from ..equilibration import get_sse_series, get_sse_series_init_seq
from ..plot import (
    plot_equilibration_max_ess,
    plot_equilibration_min_sse,
    plot_equilibration_paired_t_test,
    plot_ess,
    plot_p_values,
    plot_sse,
    plot_timeseries,
)
from . import example_times, example_timeseries

# Set to True to save the plots.
SAVE_PLOTS = False


def test_plot_timeseries(example_timeseries, example_times):
    """Test plotting a timeseries."""
    # Create a figure.
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
    """Test plotting the equilibration detection based on minimum SSE."""
    # Create a figure.
    fig = plt.figure(figsize=(6, 4))
    gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

    # Compute the SSE for the example timeseries with
    # the lugsail variance estimator.
    sse_vals, times_used = get_sse_series(
        example_timeseries, example_times, method="inter"
    )

    # Plot the equilibration detection.
    plot_equilibration_min_sse(
        fig, gridspec_obj[0], example_timeseries, sse_vals, example_times, times_used
    )

    if SAVE_PLOTS:
        fig.savefig("test_plot_equilibration_min_sse.png", bbox_inches="tight")


def test_plot_ess(example_timeseries, example_times):
    """Test plotting the effective sample size."""
    # Create a figure.
    fig, ax = plt.subplots()
    n_runs, n_samples = example_timeseries.shape

    # Compute the ESS for the example timeseries with
    # the lugsail variance estimator.
    ess_vals, times = get_ess_series(
        example_timeseries, times=example_times, method="lugsail"
    )

    # Plot the ESS.
    plot_ess(ax, ess_vals, times)

    if SAVE_PLOTS:
        fig.savefig("test_plot_ess.png")

    # Check that invalid input raises an error.
    with pytest.raises(InvalidInputError):
        plot_ess(ax, ess_vals, example_times[:-2])

    with pytest.raises(InvalidInputError):
        plot_ess(ax, ess_vals, list(example_timeseries))

    with pytest.raises(InvalidInputError):
        # Make ess_vals a 2D array.
        plot_ess(ax, np.array([ess_vals, ess_vals]), example_timeseries)


def test_plot_p_values(example_timeseries, example_times):
    """Test plotting the p-values of the paired t-test."""
    # Create a figure.
    fig, ax = plt.subplots()
    n_runs, n_samples = example_timeseries.shape

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


def test_plot_equilibration_max_ess(example_timeseries, example_times):
    """Test plotting the equilibration detection based on maximum ESS."""
    # Create a figure.
    fig = plt.figure(figsize=(6, 4))
    gridspec_obj = gridspec.GridSpec(1, 1, figure=fig)

    # Compute the ESS for the example timeseries with
    # the lugsail variance estimator.
    ess_vals, times_used = get_ess_series(
        example_timeseries, example_times, method="inter"
    )

    # Plot the equilibration detection.
    plot_equilibration_max_ess(
        fig, gridspec_obj[0], example_timeseries, ess_vals, example_times, times_used
    )

    if SAVE_PLOTS:
        fig.savefig("test_plot_equilibration_max_ess.png", bbox_inches="tight")


def test_plot_equilibration_paired_t_test(example_timeseries, example_times):
    """Test plotting the equilibration detection based on the paired t-test."""
    # Create a figure.
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
