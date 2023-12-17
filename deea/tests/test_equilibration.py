"""Tests for equilibration detection functions."""

from pathlib import Path

import numpy as np
import pytest

from deea.equilibration import (
    detect_equilibration_max_ess,
    detect_equilibration_paired_t_test,
    get_ess_series,
    get_paired_t_p_timeseries,
)

from .._exceptions import (
    AnalysisError,
    EquilibrationNotDetectedError,
    InvalidInputError,
)
from . import example_times, example_timeseries, gaussian_noise, tmpdir


def test_detect_equilibration_max_ess_variance(gaussian_noise, example_timeseries):
    """
    Test equilibration detection based on maximum effective sample size estimated
    from the ratio of the inter-run to intra-run variance.
    """
    # Compute the equilibration index.
    idx, g, ess = detect_equilibration_max_ess(gaussian_noise, method="inter")

    # Check that the equilibration index is correct.
    assert idx < 6000
    assert g == pytest.approx(1, abs=2)
    assert ess == pytest.approx(50_000, abs=40_000)

    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_max_ess(gaussian_noise[0], method="inter")

    # Check that the equilibration index is correct for the correlated timeseries.
    idx, g, ess = detect_equilibration_max_ess(example_timeseries)
    assert idx == 431
    assert g == pytest.approx(2.69474, abs=0.00001)
    assert ess == pytest.approx(4070.8931, abs=0.0001)

    with pytest.raises(InvalidInputError):
        idx, g, ess = detect_equilibration_max_ess(gaussian_noise, method="asd")

    with pytest.raises(InvalidInputError):
        idx, g, ess = detect_equilibration_max_ess(gaussian_noise, min_ess=0)

    # Feed in very short timeseries and check that it fails.
    with pytest.raises(AnalysisError):
        idx, g, ess = detect_equilibration_max_ess(gaussian_noise[:, :10])


def test_detect_equilibration_max_ess_lugsail(gaussian_noise, example_timeseries):
    """
    Test equilibration detection based on maximum effective sample size estimated
    from the ratio of the lugsail to intra-run variance.
    """
    # Compute the equilibration index.
    idx, g, ess = detect_equilibration_max_ess(gaussian_noise, method="lugsail")

    # Check that the equilibration index is correct.
    assert idx < 5000
    assert g == pytest.approx(1, abs=2)
    assert ess == pytest.approx(50_000, abs=40_000)

    # Check result for a single run.
    idx, g, ess = detect_equilibration_max_ess(gaussian_noise[0], method="lugsail")
    assert idx == pytest.approx(0, abs=3000)
    assert g == pytest.approx(1, abs=2)
    assert ess == pytest.approx(10_000, abs=3000)

    # Check that the equilibration index is correct for the correlated timeseries.
    idx, g, ess = detect_equilibration_max_ess(example_timeseries)
    assert idx == 431
    assert g == pytest.approx(2.69474, abs=0.00001)
    assert ess == pytest.approx(4070.8931, abs=0.0001)

    # Make a dummy timeseries which will cause the ESS to be small and detection to fail.
    # Do this by concatentating arrays of zeros and ones so that the first half of the
    # timeseries is all zeros and the second half is all ones.
    dummy_timeseries = np.concatenate(
        (np.zeros_like(example_timeseries), np.ones_like(example_timeseries))
    )
    dummy_timeseries = np.array(dummy_timeseries)
    with pytest.raises(EquilibrationNotDetectedError):
        idx, g, ess = detect_equilibration_max_ess(dummy_timeseries)


def test_detect_equilibration_paired_t(gaussian_noise, example_timeseries):
    """
    Test equilibration detection based on the paired t-test.
    """
    # Compute the equilibration index.
    idx = detect_equilibration_paired_t_test(gaussian_noise)
    assert idx == 0

    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise[0])

    # Check the index for the correlated timeseries.
    idx = detect_equilibration_paired_t_test(example_timeseries)
    assert idx == 328

    # Try stupid inputs and check that we get input errors.
    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, p_threshold=0)

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, p_threshold=1)

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(
            gaussian_noise, fractional_block_size=0
        )

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(
            gaussian_noise, fractional_block_size=1
        )

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, t_test_sidedness="asd")

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, fractional_test_end=0)

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(
            gaussian_noise, fractional_test_end=0.3, initial_block_size=0.5
        )
    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(
            gaussian_noise, fractional_test_end=0.3, initial_block_size=0.3
        )
    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, initial_block_size=1.2)

    with pytest.raises(InvalidInputError):
        idx = detect_equilibration_paired_t_test(gaussian_noise, final_block_size=1.2)

    # Generate steeply decaying timeseries and check that it fails to equilibrate.
    # Generate 5 very similar exponential decays.
    steeply_decaying_timeseries = np.exp(-np.arange(1000))
    steeply_decaying_timeseries = np.tile(steeply_decaying_timeseries, (5, 1))
    with pytest.raises(EquilibrationNotDetectedError):
        idx = detect_equilibration_paired_t_test(steeply_decaying_timeseries)


def test_plots_paired_t(example_timeseries, example_times, tmpdir):
    """Check that the plots are created."""
    tmp_output = Path(tmpdir) / "test_plots_paired_t"
    detect_equilibration_paired_t_test(
        example_timeseries, example_times, plot=True, plot_name=tmp_output
    )
    assert tmp_output.with_suffix(".png").exists()


def test_plots_max_ess(example_timeseries, example_times, tmpdir):
    """Check that the plots are created."""
    tmp_output = Path(tmpdir) / "test_plots_max_ess"
    detect_equilibration_max_ess(
        example_timeseries, example_times, plot=True, plot_name=tmp_output
    )
    assert tmp_output.with_suffix(".png").exists()


def test_ess_series(example_timeseries, example_times):
    """Tests on the ESS series generator which can't be run indirectly
    through the equilibration detection functions."""

    with pytest.raises(InvalidInputError):
        ess_vals, times = get_ess_series(example_timeseries, times=example_times[:-2])

    # Check that this works on indices if no times passed.
    ess_vals, times = get_ess_series(example_timeseries)
    assert times[-1] == 2493


def test_p_values(example_timeseries, example_times):
    """Tests on the p-values series generator which can't be run indirectly
    through the equilibration detection functions."""

    with pytest.raises(InvalidInputError):
        p_vals, times = get_paired_t_p_timeseries(
            example_timeseries, times=example_times[:-2]
        )

    # Check that this works on indices if no times passed.
    p_vals, times = get_paired_t_p_timeseries(example_timeseries)
    assert times[-1] == 1312
