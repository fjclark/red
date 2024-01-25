"""Tests for the red.sse module, which computes the squared standard error series."""

import pytest

from red.sse import get_sse_series_init_seq, get_sse_series_window

from . import example_timeseries


def test_get_sse_series_init_seq(example_timeseries):
    """Test the get_sse_series_init_seq function."""
    # Get the squared standard error series.
    sse_series, _ = get_sse_series_init_seq(example_timeseries, smooth_lag_times=True)

    assert sse_series[0] == pytest.approx(2.972160655979027, abs=1e-4)
    assert sse_series[1] == pytest.approx(2.948013765819789, abs=1e-4)
    assert sse_series[-1] == pytest.approx(3.042444011451404, abs=1e-4)


def test_get_sse_series_window(example_timeseries):
    """Test the get_sse_series_window function."""
    # Get the squared standard error series.
    sse_series, _ = get_sse_series_window(example_timeseries)

    assert sse_series[0] == pytest.approx(0.6912977518313307, abs=1e-4)
    assert sse_series[1] == pytest.approx(0.6896663937370537, abs=1e-4)
    assert sse_series[-1] == pytest.approx(1.962073132107981, abs=1e-4)
