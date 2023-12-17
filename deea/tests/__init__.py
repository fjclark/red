"""
Setup fixtures for the tests.
"""

import tempfile
from importlib import resources

import numpy as np
import pytest


@pytest.fixture(scope="session")
def example_timeseries():
    timeseries_bytes = (
        resources.files("deea.data").joinpath("example_timeseries.npy").read_bytes()
    )
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(timeseries_bytes)
        tmp.seek(0)
        timeseries = np.load(tmp.name)
    yield timeseries


@pytest.fixture(scope="session")
def example_times():
    times_bytes = (
        resources.files("deea.data").joinpath("example_times.npy").read_bytes()
    )
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(times_bytes)
        tmp.seek(0)
        times = np.load(tmp.name)
    yield times


# Generate 5 runs of 1000 samples of Gaussian noise.
@pytest.fixture(scope="session")
def gaussian_noise():
    N_SAMPLES = 10_000
    N_RUNS = 5
    RUN_MEANS = [0.0 for i in range(N_RUNS)]  # Different means.
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


# Create temporary directory for tests.
@pytest.fixture(scope="session")
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
