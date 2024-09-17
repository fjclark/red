"""
Unit and regression test for the variance module.
"""

import numpy as np
import numpy.testing as npt
import pytest
from statsmodels.tsa.stattools import acovf

from red._exceptions import AnalysisError, InvalidInputError
from red.variance import (
    _get_autocovariance,
    _get_autocovariance_window,
    _get_gamma_cap,
    _smoothen_max_lag_times,
    get_variance_initial_sequence,
    get_variance_series_initial_sequence,
    get_variance_series_window,
    get_variance_window,
    inter_run_variance,
    intra_run_variance,
    lugsail_variance,
    replicated_batch_means_variance,
)

from . import example_times, example_timeseries, gaussian_noise, gaussian_noise_offset


def test_get_autocovariance(example_timeseries):
    """Test the _get_autocovariance function."""
    example_timeseries = example_timeseries.mean(axis=0)
    example_timeseries_demeaned = example_timeseries - example_timeseries.mean()

    # Compute the autocovariance function with no truncation.
    autocovariance = _get_autocovariance(example_timeseries)
    autocovariance_statsmodels = acovf(example_timeseries_demeaned, fft=False)
    assert autocovariance == pytest.approx(autocovariance_statsmodels, abs=0.01)

    # Test with max lag time.
    autocovariance = _get_autocovariance(example_timeseries, max_lag=10)
    autocovariance_statsmodels = acovf(example_timeseries_demeaned, fft=False, nlag=10)
    assert autocovariance == pytest.approx(autocovariance_statsmodels, abs=0.01)
    assert len(autocovariance) == 11

    # Test with different mean.
    autocovariance = _get_autocovariance(example_timeseries, mean=50)  # Actual mean 54.9
    assert autocovariance[:3] == pytest.approx(
        np.array([342.65922098, 132.30372161, 120.22912133]), abs=0.01
    )


def test_gamma_cap(example_timeseries):
    """Test the _get_gamma_cap function."""
    example_timeseries = example_timeseries.mean(axis=0)

    # Compute the autocovariance function with no truncation.
    autocovariance = _get_autocovariance(example_timeseries)

    # Compute the gamma function.
    gamma_cap = _get_gamma_cap(autocovariance)

    # Check that we get the right lengths of results.
    assert len(gamma_cap) == len(autocovariance) // 2

    # Check that the gamma function result has not regressed.
    assert gamma_cap[:3] == pytest.approx(
        np.array([427.04712998, 193.80629572, 183.46922874]), abs=0.01
    )
    assert gamma_cap[-3:] == pytest.approx(np.array([0.54119574, 0.04336649, 0.68390851]), abs=0.01)

    # Check that an analysis error is raised if input sequence too short.
    with pytest.raises(AnalysisError):
        _get_gamma_cap(example_timeseries[:1])


def test_variance_initial_sequence(example_timeseries):
    """Test the get_variance_initial_sequence function."""
    # Take mean of timeseries.
    example_timeseries = example_timeseries.mean(axis=0)

    # Get variances with different methods.
    methods = [
        "positive",
        "initial_positive",
        "initial_monotone",
        "initial_convex",
    ]
    res = {
        method: get_variance_initial_sequence(np.array([example_timeseries]), method)
        for method in methods
    }

    # Geyer results obtained from R package mcmc.
    geyer_res = {
        "initial_positive": 27304.776049686046,
        "initial_monotone": 24216.32288181667,
        # Actual MCMC result for initial convex is 21898.6178149005, but very close.
        "initial_convex": 21904.973713096544,
    }

    # Check that the results are correct.
    for method in geyer_res:
        assert res[method][0] == pytest.approx(geyer_res[method], abs=0.01)

    # Check that the results decrease in the correct order.
    assert (
        res["initial_positive"][0]
        > res["positive"][0]
        > res["initial_monotone"][0]
        > res["initial_convex"][0]
    )


def test_variance_initial_sequence_multiple_runs(example_timeseries):
    """Regression test for the get_variance_initial_sequence function
    with multiple runs."""
    # Expected results.
    expected = {
        "positive": 49723.19149920802,
        "initial_positive": 52014.07839470464,
        "initial_monotone": 43587.36425899603,
        "initial_convex": 39009.608609724724,
    }
    for method in expected:
        var = get_variance_initial_sequence(example_timeseries, method)[0]
        assert var == pytest.approx(expected[method], abs=0.01)


def test_get_variance_series_initial_sequence(example_timeseries):
    """Test the get_variance_series_initial_sequence function."""
    # Take mean of timeseries.
    example_timeseries = example_timeseries.mean(axis=0)

    var_seq = get_variance_series_initial_sequence(
        example_timeseries, sequence_estimator="initial_positive", min_max_lag_time=0
    )[0]
    assert var_seq[0] == pytest.approx(27304.77604968605, abs=0.01)
    assert var_seq[1] == pytest.approx(27121.16749972886, abs=0.01)
    assert var_seq[-1] == pytest.approx(819.4266803627752, abs=0.01)


def test_get_variance_initial_sequence_raises(example_timeseries):
    """Test that get_variance_initial_sequence raises the errors
    if the input is not valid."""
    with pytest.raises(InvalidInputError):
        get_variance_series_initial_sequence(
            example_timeseries, sequence_estimator="not_implemented", min_max_lag_time=1
        )

    with pytest.raises(InvalidInputError):
        get_variance_series_initial_sequence(
            example_timeseries,
            sequence_estimator="initial_positive",
            min_max_lag_time=-1,
        )

    with pytest.raises(InvalidInputError):
        get_variance_initial_sequence(
            example_timeseries,
            sequence_estimator="initial_positive",
            min_max_lag_time=10_000,
        )

    for max_max_lag_time in [-1, 10_000, 3]:
        with pytest.raises(InvalidInputError):
            get_variance_initial_sequence(
                example_timeseries,
                sequence_estimator="initial_positive",
                min_max_lag_time=5,
                max_max_lag_time=max_max_lag_time,
            )

    for autocov in [4, np.array([[1, 2, 3]]), np.array([1])]:
        with pytest.raises(InvalidInputError):
            get_variance_initial_sequence(
                example_timeseries,
                sequence_estimator="initial_positive",
                autocov=autocov,
            )

    with pytest.raises(AnalysisError):
        get_variance_initial_sequence(
            np.zeros((10)),
            sequence_estimator="initial_positive",
        )

    with pytest.raises(InvalidInputError):
        get_variance_initial_sequence(
            example_timeseries,
            sequence_estimator="initial_positive",
            max_max_lag_time=4,
            autocov=np.array([1, 2, 3]),
        )


def test_get_variance_series_initial_sequence_raises(example_timeseries):
    """Test that get_variance_series_initial_sequence raises the errors
    if the input is not valid."""
    for frac_pad in [-0.1, 1.1]:
        with pytest.raises(InvalidInputError):
            get_variance_series_initial_sequence(
                example_timeseries,
                sequence_estimator="initial_positive",
                frac_padding=frac_pad,
            )

    with pytest.warns(UserWarning):
        get_variance_series_initial_sequence(
            example_timeseries, sequence_estimator="initial_positive", frac_padding=0.6
        )


def test_smoothen_lags():
    """Test the _smoothen_max_lag_times function."""
    rough_lags = np.array([10, 11, 6, 5, 4, 4, 4, 3])
    smooth_lags = _smoothen_max_lag_times(rough_lags)
    # Compare the arrays of lists using pytest. Expect answer is
    # monotonic decreasing and interpolated between change points.
    npt.assert_array_equal(smooth_lags, np.array([10, 8, 6, 5, 4, 4, 3, 3]))


def test_get_variance_series_initial_sequence_smooth_lags(example_timeseries):
    """Regression test for the get_variance_series_initial_sequence function
    with smoothened lags."""
    # Take mean of timeseries.
    example_timeseries = example_timeseries.mean(axis=0)

    var_seq = get_variance_series_initial_sequence(
        example_timeseries, sequence_estimator="initial_positive", smooth_lag_times=True
    )[0]
    assert var_seq[0] == pytest.approx(27304.77604968605, abs=0.01)
    assert var_seq[1] == pytest.approx(27118.973970536812, abs=0.01)
    assert var_seq[20] == pytest.approx(22105.33941568437, abs=0.01)
    assert var_seq[-1] == pytest.approx(690.7015384062945, abs=0.01)

    # Try with a sequence where the max lag will be the end of the sequence.
    smooth_seq = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    var_seq = get_variance_series_initial_sequence(
        smooth_seq, sequence_estimator="initial_positive", smooth_lag_times=True
    )[0]
    assert var_seq[0] == pytest.approx(1, abs=0.01)


def test_get_autocovariance_window():
    """Check that an error is raised if the window size is too large."""
    with pytest.raises(InvalidInputError):
        _get_autocovariance_window(np.array([[1, 2, 3]]), window_size=4)


def test_get_variance_window(example_timeseries):
    """Test the get_variance_window function."""
    # Take mean of timeseries.
    example_timeseries = np.array([example_timeseries.mean(axis=0)])

    # First, try with a window size of 1 and check that we get the same result as
    # np.var.
    variance = get_variance_window(example_timeseries, window_size=1)
    assert variance == pytest.approx(np.var(example_timeseries), abs=0.01)

    # Now try with a window size of 5.
    variance = get_variance_window(example_timeseries, window_size=5)
    assert variance == pytest.approx(721.2492651825239, abs=0.01)


def test_get_variance_window_raises(example_timeseries):
    """Check that errors are raised if the input is not valid."""
    with pytest.raises(InvalidInputError):
        get_variance_window(example_timeseries, kernel=4)

    for window_size in [-1, 0, 10_000]:
        with pytest.raises(InvalidInputError):
            get_variance_window(example_timeseries, window_size=window_size)

    with pytest.raises(AnalysisError):
        get_variance_window(np.zeros((10, 10)), window_size=1)


def test_get_variance_window_multiple_runs(example_timeseries):
    """Regression test for the get_variance_window function with multiple runs."""
    # Expected results.
    expected = {
        1: 1345.5038988334522,
        5: 2220.1371817657955,
        10: 3182.492111191889,
        100: 14670.359227619572,
    }
    for window_size in expected:
        var = get_variance_window(example_timeseries, window_size=window_size)
        assert var == pytest.approx(expected[window_size], abs=0.01)


def test_get_variance_series_window(example_timeseries):
    """Test the get_variance_series_window function."""
    # Take mean of timeseries.
    example_timeseries = example_timeseries.mean(axis=0)

    var_seq = get_variance_series_window(example_timeseries, window_size=1, window_size_fn=None)[0]
    assert var_seq[0] == pytest.approx(318.6396575172195, abs=0.01)
    assert var_seq[1] == pytest.approx(318.3955761337024, abs=0.01)
    assert var_seq[-1] == pytest.approx(243.41472988044396, abs=0.01)

    # Now try with the window size function.
    var_seq = get_variance_series_window(
        example_timeseries, window_size=None, window_size_fn=lambda x: round(x**0.5)
    )[0]
    assert var_seq[0] == pytest.approx(4262.823818412766, abs=0.01)
    assert var_seq[1] == pytest.approx(4243.115237432978, abs=0.01)
    assert var_seq[-1] == pytest.approx(580.7484698310258, abs=0.01)


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
        lugsail_variance(gaussian_noise, 0.1)


def test_get_variance_series_window_raises(example_timeseries):
    """Check that get_variance_series_window raises the errors
    if the input is not valid."""
    with pytest.raises(InvalidInputError):
        get_variance_series_window(
            example_timeseries, window_size=1, window_size_fn=lambda x: x**0.5
        )

    with pytest.raises(InvalidInputError):
        get_variance_series_window(example_timeseries, window_size=None, window_size_fn=None)

    with pytest.raises(InvalidInputError):
        get_variance_series_window(example_timeseries, window_size=None, window_size_fn=4)

    for frac_pad in [-0.1, 1.1]:
        with pytest.raises(InvalidInputError):
            get_variance_series_window(example_timeseries, frac_padding=frac_pad)

    with pytest.warns(UserWarning):
        get_variance_series_window(example_timeseries, frac_padding=0.6)


def test_inter_run_variance(gaussian_noise):
    """Test the inter-run variance."""
    # Compute the inter-run variance.
    variance = inter_run_variance(gaussian_noise)

    # Check that the variance is correct.
    assert variance == pytest.approx(0.1, abs=0.5)  # Wide tolerance due to high variance.

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
