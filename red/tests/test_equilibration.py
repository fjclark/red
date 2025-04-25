"""Tests for equilibration detection functions."""

from pathlib import Path

import numpy as np
import pytest

from red.equilibration import (
    detect_equilibration_init_seq,
    detect_equilibration_paired_t_test,
    detect_equilibration_window,
    get_paired_t_p_timeseries,
)

from .._exceptions import EquilibrationNotDetectedError, InvalidInputError
from . import example_times, example_timeseries, gaussian_noise, tmpdir


def test_detect_equilibration_init_seq(example_timeseries, example_times, tmpdir):
    """
    Test equilibration detection based on the minimum sum of squared errors,
    using initial sequence methods to estimate the variance.
    """
    # Use the mean time to make this faster.
    example_timeseries = example_timeseries.mean(axis=0)

    # Compute the equilibration index.
    equil_idx, equil_g, equil_ess = detect_equilibration_init_seq(data=example_timeseries)

    # Make sure that the index is a numpy int64
    assert isinstance(equil_idx, np.int64)

    # Check that the equilibration index is correct.
    assert equil_idx == 398
    assert equil_g == pytest.approx(4.292145845594654, abs=1e-4)
    assert equil_ess == pytest.approx(518.85468949889, abs=1e-4)

    # Try also supplying the times. Plot in a temporary directory.
    tmp_output = Path(tmpdir) / "test_plots_min_sse_init_seq"
    # tmp_output = "./test_plots_min_sse_init_seq"
    equil_time, equil_g, equil_ess = detect_equilibration_init_seq(
        data=example_timeseries,
        times=example_times,
        method="min_sse",
        plot=True,
        plot_name=tmp_output,
        sequence_estimator="initial_convex",
        smooth_lag_times=True,  # To speed up the test.
    )
    assert equil_time == 0.3312
    assert equil_g == pytest.approx(4.206494270439359, abs=1e-4)
    assert equil_ess == pytest.approx(525.8535630357488, abs=1e-4)
    # Check that test_plots_min_sse.png exists.
    assert tmp_output.with_suffix(".png").exists()


def test_detect_equilibration_init_seq_max_ess(example_timeseries, example_times):
    """
    Test equilibration detection based on the maximum effective sample size,
    using initial sequence methods to estimate the variance.
    """
    # Compute the equilibration index.
    equil_idx, equil_g, equil_ess = detect_equilibration_init_seq(
        data=example_timeseries, method="max_ess", smooth_lag_times=True
    )

    # Make sure that the index is a numpy int64
    assert isinstance(equil_idx, np.int64)

    # Check that the equilibration index is correct.
    assert equil_idx == 486
    assert equil_g == pytest.approx(6.8237419281169265, abs=1e-4)
    assert equil_ess == pytest.approx(1567.321875982989, abs=1e-4)


def test_detect_equilibration_init_seq_raises(example_timeseries):
    """
    Test that invalid inputs raise errors.
    """
    with pytest.raises(InvalidInputError):
        detect_equilibration_init_seq(
            method="non_existent",
            data=example_timeseries,
            sequence_estimator="positive",
        )


def test_detect_equilibration_window(example_timeseries, example_times, tmpdir):
    """
    Test equilibration detection based on the minimum sum of squared errors,
    using window methods to estimate the variance.
    """
    # Use the mean time to make this faster.
    example_timeseries = example_timeseries.mean(axis=0)

    # Compute the equilibration index.
    equil_idx, equil_g, equil_ess = detect_equilibration_window(data=example_timeseries)

    # Check that the equilibration index is correct.
    assert equil_idx == 398
    assert equil_g == pytest.approx(3.840409864530417, abs=1e-4)
    assert equil_ess == pytest.approx(579.8860222103675, abs=1e-4)

    # Make sure that the index is a numpy int64
    assert isinstance(equil_idx, np.int64)

    # Try also supplying the times. Plot in a temporary directory.
    tmp_output = Path(tmpdir) / "test_plots_min_sse_window"
    # tmp_output = "./test_plots_min_sse_window"
    equil_time, equil_g, equil_ess = detect_equilibration_window(
        data=example_timeseries,
        times=example_times,
        plot=True,
        plot_name=tmp_output,
        plot_window_size=True,
    )
    assert equil_time == 0.3192
    assert equil_g == pytest.approx(3.840409864530417, abs=1e-4)
    assert equil_ess == pytest.approx(579.8860222103675, abs=1e-4)
    # Check that test_plots_min_sse.png exists.
    assert tmp_output.with_suffix(".png").exists()


@pytest.mark.parametrize(
    "equil_fn, equil_fn_args",
    [
        (detect_equilibration_init_seq, {"method": "min_sse"}),
        (detect_equilibration_window, {"method": "min_sse"}),
    ],
)
def test_correct_return_type_no_times(equil_fn, equil_fn_args, example_timeseries):
    """
    Test that we return a numpy int64 for the equilibration index when
    no times are supplied. This ensures that the index can be directly used
    for truncation of the timeseries.
    """
    equil_idx, _, _ = equil_fn(data=example_timeseries, **equil_fn_args)
    assert isinstance(equil_idx, np.int64)


@pytest.mark.parametrize(
    "equil_fn, equil_fn_args",
    [
        (detect_equilibration_init_seq, {"method": "min_sse"}),
        (detect_equilibration_window, {"method": "min_sse"}),
    ],
)
def test_correct_return_type_with_times(equil_fn, equil_fn_args, example_timeseries, example_times):
    """
    Test that we return a float for the equilibration index when
    times are supplied. This ensures that the index can be directly used
    for truncation of the timeseries.
    """
    equil_idx, _, _ = equil_fn(data=example_timeseries, times=example_times, **equil_fn_args)
    assert isinstance(equil_idx, np.float64)


def test_detect_equilibration_window_max_ess(example_timeseries, example_times):
    """
    Test equilibration detection based on the maximum effective sample size,
    using window methods to estimate the variance.
    """
    # Compute the equilibration index.
    equil_idx, equil_g, equil_ess = detect_equilibration_window(
        data=example_timeseries, method="max_ess"
    )

    # Check that the equilibration index is correct.
    assert equil_idx == 369
    assert equil_g == pytest.approx(3.864939192439241, abs=1e-4)
    assert equil_ess == pytest.approx(2918.545270276546, abs=1e-4)


def test_compare_pymbar(example_timeseries):
    """Check that we get the same result as pymbar when equivalent
    methods are used."""
    example_timeseries = example_timeseries.mean(axis=0)

    # Results below were obtained with:
    # ( equil_idx_chod,
    #     equil_g_chod,
    #     equil_ess_chod,
    # ) = pymbar.timeseries.detect_equilibration(example_timeseries, fast=False, nskip=1)
    equil_idx_chod = 877
    equil_g_chod = 4.111825
    # equil_ess_chod = 425.35858
    equil_idx, equil_g, _ = detect_equilibration_init_seq(
        example_timeseries, method="max_ess", sequence_estimator="positive"
    )
    assert equil_idx == equil_idx_chod
    assert equil_g == pytest.approx(equil_g_chod, abs=1e-4)


def test_detect_equilibration_paired_t(gaussian_noise, example_timeseries):
    """
    Test equilibration detection based on the paired t-test.
    """
    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise[0])

    # Check the index for the correlated timeseries.
    idx = detect_equilibration_paired_t_test(example_timeseries)
    assert idx == 328
    assert isinstance(idx, np.int64)

    # Try stupid inputs and check that we get input errors.
    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, p_threshold=0)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, p_threshold=1)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, fractional_block_size=0)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, fractional_block_size=1)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, t_test_sidedness="asd")

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, fractional_test_end=0)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(
            gaussian_noise, fractional_test_end=0.3, initial_block_size=0.5
        )
    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(
            gaussian_noise, fractional_test_end=0.3, initial_block_size=0.3
        )
    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, initial_block_size=1.2)

    with pytest.raises(InvalidInputError):
        detect_equilibration_paired_t_test(gaussian_noise, final_block_size=1.2)

    # Generate steeply decaying timeseries and check that it fails to equilibrate.
    # Generate 5 very similar exponential decays.
    steeply_decaying_timeseries = np.exp(-np.arange(1000))
    steeply_decaying_timeseries = np.tile(steeply_decaying_timeseries, (5, 1))
    with pytest.raises(EquilibrationNotDetectedError):
        detect_equilibration_paired_t_test(steeply_decaying_timeseries)


def test_plots_paired_t(example_timeseries, example_times, tmpdir):
    """Check that the plots are created."""
    tmp_output = Path(tmpdir) / "test_plots_paired_t"
    detect_equilibration_paired_t_test(
        example_timeseries, example_times, plot=True, plot_name=tmp_output
    )
    assert tmp_output.with_suffix(".png").exists()


def test_p_values(example_timeseries, example_times):
    """Tests on the p-values series generator which can't be run indirectly
    through the equilibration detection functions."""

    with pytest.raises(InvalidInputError):
        get_paired_t_p_timeseries(example_timeseries, times=example_times[:-2])

    # Check that this works on indices if no times passed.
    _, times = get_paired_t_p_timeseries(example_timeseries)
    assert times[-1] == 1312
