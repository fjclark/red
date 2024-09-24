"""Tests for functions to compute effective sample size and statistical inefficiency."""

import numpy as np
import pytest

from .._exceptions import InvalidInputError
from ..ess import (
    ess_inter_variance,
    ess_lugsail_variance,
    get_ess_series_init_seq,
    get_ess_series_window,
    statistical_inefficiency_inter_variance,
    statistical_inefficiency_lugsail_variance,
)
from . import example_times, example_timeseries, gaussian_noise, gaussian_noise_offset


def test_get_ess_series_init_seq(example_timeseries):
    """Test the get_ess_series_init_seq function."""
    # Get the effective sample size series.
    ess_series, _ = get_ess_series_init_seq(example_timeseries, smooth_lag_times=True)

    assert ess_series[0] == pytest.approx(452.70227776104, abs=1e-2)
    assert ess_series[1] == pytest.approx(456.25279094474644, abs=1e-2)
    assert ess_series[-1] == pytest.approx(405.7192903828652, abs=1e-2)

    # Create artificial dataset with 5 runs of widely spaced gaussian noise.
    # Should give an effective sample size of ~5.
    data = np.array([np.random.normal(loc, 1, 100) for loc in [0, 3, 6, 9, 12]])
    ess_series, _ = get_ess_series_init_seq(data, smooth_lag_times=True)
    assert ess_series[0] == pytest.approx(5, abs=0.5)


def test_get_ess_series_window(example_timeseries):
    """Test the get_ess_series_window function."""
    # Get the effective sample size series.
    ess_series, _ = get_ess_series_window(example_timeseries)

    assert ess_series[0] == pytest.approx(1946.3449653482178, abs=1e-2)
    assert ess_series[1] == pytest.approx(1950.275554403233, abs=1e-2)
    assert ess_series[-1] == pytest.approx(629.1193764166628, abs=1e-2)

    # Create artificial dataset with 5 runs of widely spaced gaussian noise.
    # Should give an effective sample size of ~5.
    data = np.array([np.random.normal(loc, 1, 100) for loc in [0, 3, 6, 9, 12]])
    ess_series, _ = get_ess_series_window(data)
    # However, window method misses a lot of auto-correlation, so ess estimate
    # is larger.
    assert ess_series[0] == pytest.approx(54.357400316107, abs=0.8)


def test_statistical_inefficiency_inter_variance(gaussian_noise, example_timeseries):
    """Test the statistical inefficiency using the inter-run variance."""
    # Compute the statistical inefficiency.
    si = statistical_inefficiency_inter_variance(gaussian_noise)

    # Check that the statistical inefficiency is correct.
    assert si == pytest.approx(1, abs=5)

    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        statistical_inefficiency_inter_variance(gaussian_noise[0])

    # Check that statistical inefficiency is higher for the correlated timeseries.
    si = statistical_inefficiency_inter_variance(example_timeseries)
    assert si == pytest.approx(19.67766, abs=0.0001)


def test_statistical_inefficiency_lugsail_variance(gaussian_noise, example_timeseries):
    """Test the statistical inefficiency using the lugsail variance."""
    # Compute the statistical inefficiency.
    si = statistical_inefficiency_lugsail_variance(gaussian_noise)
    assert si == pytest.approx(1, abs=2)

    # Check that it works for a single run.
    si = statistical_inefficiency_lugsail_variance(gaussian_noise[0])
    assert si == pytest.approx(1, abs=2)

    # Check that statistical inefficiency is higher for the correlated timeseries.
    si = statistical_inefficiency_lugsail_variance(example_timeseries)
    assert si == pytest.approx(4.0319668, abs=0.0001)


def test_effective_sample_size_inter_variance(gaussian_noise, example_timeseries):
    """Test the effective sample size using the inter-run variance."""
    # Compute the effective sample size.
    ess = ess_inter_variance(gaussian_noise)

    # Check that the effective sample size is correct.
    # 50,000 is the maximum possible value. Set massive tolerances
    # as this is noisy.
    assert ess == pytest.approx(50_000, abs=40_000)

    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        ess_inter_variance(gaussian_noise[0])

    # Check that effective sample size is lower for the correlated timeseries.
    ess = ess_inter_variance(example_timeseries)
    assert ess == pytest.approx(700, abs=1000)


def test_effective_sample_size_lugsail_variance(gaussian_noise, example_timeseries):
    """Test the effective sample size using the lugsail variance."""
    # Compute the effective sample size.
    ess = ess_lugsail_variance(gaussian_noise)

    # Check that the effective sample size is correct.
    assert ess == pytest.approx(50_000, abs=20_000)

    # Check that it works for a single run.
    ess = ess_lugsail_variance(gaussian_noise[0])
    assert ess == pytest.approx(10_000, abs=5000)

    # Check that effective sample size is lower for the correlated timeseries.
    ess = ess_lugsail_variance(example_timeseries)
    assert ess == pytest.approx(3255.2351, abs=0.001)
