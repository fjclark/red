"""
Unit and regression test for the variance module.
"""

import pytest

from deea._exceptions import AnalysisError, InvalidInputError
from deea.variance import (
    inter_run_variance,
    intra_run_variance,
    lugsail_variance,
    replicated_batch_means_variance,
)

from . import example_times, example_timeseries, gaussian_noise, gaussian_noise_offset


def test_replicated_batch_means_variance(gaussian_noise):
    """Test the replicated batch means variance."""
    # Compute the replicated batch means variance.
    variance = replicated_batch_means_variance(gaussian_noise, 1)

    # Check that the variance is correct.
    assert variance == pytest.approx(0.1, abs=0.01)

    # Make the batch 100 times larger and check that the variance estimate does not change.
    variance = replicated_batch_means_variance(gaussian_noise, 10)
    assert variance == pytest.approx(0.1, abs=0.01)

    # Check that it works for a single run.
    variance = replicated_batch_means_variance(gaussian_noise[0], 1)
    assert variance == pytest.approx(0.1, abs=0.01)

    # Make sure errors are raised if the batch size is too small or large.
    with pytest.raises(InvalidInputError):
        replicated_batch_means_variance(gaussian_noise, 0)

    with pytest.raises(InvalidInputError):
        replicated_batch_means_variance(gaussian_noise, 100_000)


def test_lugsail_variance(gaussian_noise):
    """Test the lugsail variance."""
    # Compute the lugsail variance.
    variance = lugsail_variance(gaussian_noise)

    # Check that the variance is correct.
    assert variance == pytest.approx(0.1, abs=0.02)

    # Check that it works for a single run.
    variance = lugsail_variance(gaussian_noise[0])
    assert variance == pytest.approx(
        0.1, abs=0.05
    )  # Wide tolerance due to higher variance from single run.

    # Make sure errors are raised if n_pow is unreasonable.
    with pytest.raises(InvalidInputError):
        lugsail_variance(gaussian_noise, 0)

    with pytest.raises(InvalidInputError):
        lugsail_variance(gaussian_noise, 100_000)

    with pytest.raises(AnalysisError):
        variance = lugsail_variance(gaussian_noise, 0.1)


def test_inter_run_variance(gaussian_noise):
    """Test the inter-run variance."""
    # Compute the inter-run variance.
    variance = inter_run_variance(gaussian_noise)

    # Check that the variance is correct.
    assert variance == pytest.approx(
        0.1, abs=0.5
    )  # Wide tolerance due to high variance.

    # Check that it raises an invalid input error if the data is one dimensional.
    with pytest.raises(InvalidInputError):
        inter_run_variance(gaussian_noise[0])


def test_intra_run_variance(gaussian_noise):
    """Test the intra-run variance."""
    # Compute the intra-run variance.
    variance = intra_run_variance(gaussian_noise)

    # Check that the variance is correct.
    assert variance == pytest.approx(0.1, abs=0.01)

    # Check that it runs for a single run.
    variance = intra_run_variance(gaussian_noise[0])
    assert variance == pytest.approx(0.1, abs=0.01)
