"""Tests for the Gelman-Rubin convergence diagnostic."""

import pytest

from .._exceptions import InvalidInputError
from ..gelman_rubin import gelman_rubin, stable_gelman_rubin
from . import example_timeseries, gaussian_noise, gaussian_noise_offset


def test_gelman_rubin_gaussian(gaussian_noise):
    """Test the Gelman-Rubin diagnostic."""
    # Compute the Gelman-Rubin diagnostic.
    gr = gelman_rubin(gaussian_noise)

    # Check that the diagnostic is correct.
    assert gr == pytest.approx(1, abs=0.001)
    # Check that it fails for a single run.
    with pytest.raises(InvalidInputError):
        gelman_rubin(gaussian_noise[0])


def test_stable_gelman_rubin_gaussian(gaussian_noise):
    """Test the stable Gelman-Rubin diagnostic."""
    # Compute the stable Gelman-Rubin diagnostic.
    gr = stable_gelman_rubin(gaussian_noise)

    # Check that the diagnostic is correct.
    assert gr == pytest.approx(1, abs=0.001)

    # Check that it works for a single run.
    gr = stable_gelman_rubin(gaussian_noise[0])
    assert gr == pytest.approx(1, abs=0.001)


def test_compare_gr_to_stable_gr(gaussian_noise_offset, example_timeseries):
    """Compare the Gelman-Rubin diagnostic to the stable Gelman-Rubin diagnostic."""
    # Compare GR diagnostics for offset Gaussian noise.
    gr = gelman_rubin(gaussian_noise_offset)
    stable_gr = stable_gelman_rubin(gaussian_noise_offset)
    assert stable_gr <= gr
    assert gr == pytest.approx(1.1, abs=0.05)
    assert stable_gr == pytest.approx(1, abs=0.005)

    # Compare GR diagnostics for example timeseries.
    # This is always the same, so we can use tight tolerances.
    gr = gelman_rubin(example_timeseries)
    stable_gr = stable_gelman_rubin(example_timeseries)
    assert stable_gr <= gr
    assert gr == pytest.approx(1.00355134, abs=0.000001)
    assert stable_gr == pytest.approx(1.00057735, abs=0.000001)
