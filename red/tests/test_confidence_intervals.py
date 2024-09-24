"""Test for confidence interval calculation functions."""

import pytest

from red.confidence_intervals import get_conf_int_initial_sequence

from . import example_timeseries, gaussian_noise, gaussian_noise_widely_offset


def test_conf_int_initial_sequence_gaussian_noise(gaussian_noise):
    """Test the confidence interval calculation with gaussian noise."""
    # Calculate the confidence interval. We have 50_000 points with mean 0
    # and variance 0.1, with no correlation. So conf int should be 1.96 * sqrt(0.1 / 50_000).
    # = 0.002772
    conf_int = get_conf_int_initial_sequence(gaussian_noise, conf_level_two_sided=0.95)
    assert conf_int == pytest.approx(0.002772, abs=1e-4)


def test_conf_int_initial_sequence_gaussian_noise_widely_offset(gaussian_noise_widely_offset):
    """Test the confidence interval calculation with gaussian noise with widely offset means."""
    # Calculate the confidence interval. We have 50_000 points with mean -2, -1, 0, 1, 2
    # and variance 0.1, with no correlation within each run. However, each sample is effectively
    # completely correlated, so we effectively have 5 samples with variance 2. So 95 % CI should be
    # 2.776 * sqrt(2 / 5) = 1.7557. We'll underestimate slightly, but should be ~ 1.7.
    conf_int = get_conf_int_initial_sequence(
        gaussian_noise_widely_offset, conf_level_two_sided=0.95
    )
    assert conf_int == pytest.approx(
        1.7557, abs=1e-1
    )  # Fairly wide tolerance to account for underestimation.


def test_conf_int_initial_sequence_example_timeseries(example_timeseries):
    """Regression test for the example timeseries."""
    conf_int = get_conf_int_initial_sequence(example_timeseries)
    assert conf_int == pytest.approx(3.388047, abs=1e-4)
