"""
Setup fixtures for the tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def example_timeseries():
    # This is not nice, but the suggested setuptools approach does not work for python
    # 3.9 (worked fine for 3.12).
    timeseries_path = Path(__file__).parent.parent / "data" / "example_timeseries.npy"
    yield np.load(timeseries_path)


@pytest.fixture(scope="session")
def example_times():
    # This is not nice, but the suggested setuptools approach does not work for python
    # 3.9 (worked fine for 3.12).
    times_path = Path(__file__).parent.parent / "data" / "example_times.npy"
    yield np.load(times_path)


# Generate 5 runs of 1000 samples of Gaussian noise.
@pytest.fixture(scope="session")
def gaussian_noise():
    N_SAMPLES = 10_000
    N_RUNS = 5
    RUN_MEANS = [0.0 for i in range(N_RUNS)]
    VARIANCE = 0.1  # Shared variance.
    noise = np.zeros((N_RUNS, N_SAMPLES))
    for i, mean in enumerate(RUN_MEANS):
        noise[i] = np.random.normal(mean, np.sqrt(VARIANCE), N_SAMPLES)
    yield noise


# As above but with offset means.
@pytest.fixture(scope="session")
def gaussian_noise_offset():
    N_SAMPLES = 10_000
    N_RUNS = 5
    RUN_MEANS = [0.1 * i for i in range(N_RUNS)]  # Different means.
    VARIANCE = 0.1  # Shared variance.
    noise = np.zeros((N_RUNS, N_SAMPLES))
    for i, mean in enumerate(RUN_MEANS):
        noise[i] = np.random.normal(mean, np.sqrt(VARIANCE), N_SAMPLES)
    yield noise


# As above but with very widely offset means.
@pytest.fixture(scope="session")
def gaussian_noise_widely_offset():
    N_SAMPLES = 10_000
    N_RUNS = 5
    RUN_MEANS = [-2 + 1 * i for i in range(N_RUNS)]  # Different means.
    VARIANCE = 0.1  # Shared variance.
    noise = np.zeros((N_RUNS, N_SAMPLES))
    for i, mean in enumerate(RUN_MEANS):
        noise[i] = np.random.normal(mean, np.sqrt(VARIANCE), N_SAMPLES)
    yield noise


# Create temporary directory for tests.
@pytest.fixture(scope="session")
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
