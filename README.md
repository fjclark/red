deea
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/fjclark/deea/workflows/CI/badge.svg)](https://github.com/fjclark/deea/actions?query=workflow%3ACI)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fff40e5573f847399bee98eef495f8c6)](https://app.codacy.com/gh/fjclark/deea/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/gh/fjclark/deea/branch/main/graph/badge.svg)](https://codecov.io/gh/fjclark/deea/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Detection of Equilibration by Ensemble Analysis

A Python package for detecting equilibration in timeseries data where an initial transient is followed by a stationary distribution. Two main methods are implemented, which differ in the way they account for autocorrelation: 

  - `detect_equilibration_init_seq`: This uses the initial sequence methods of Geyer ([Geyer, 1992](https://www.jstor.org/stable/2246094)) to determine the truncation point of the sum of autocovariances.
  - `detect_equilibration_window`: This uses window methods (see [Geyer](https://www.jstor.org/stable/2246094) again) when calculating the 
autocorrelation. 

For both, the equilibration point can be determined either according to the minimum of the squared standard error (the default), or the maximum effective sample size, by specifying `method="min_sse"` or `method="max_ess"`.

### Installation

```bash
git clone https://github.com/fjclark/deea.git
cd deea
pip install -e .
```

### Usage

```python
import deea

# Load your timeseries of interest.
# This should be a 2D numpy array with shape (n_runs, n_samples),
# or a 1D numpy array with shape (n_samples).
my_timeseries = ...

# Detect equilibration based on the minimum squared standard error
# using Geyer's initial convex sequence method to account for
# autocorrelation. idx is the index of the first sample after
# equilibration, g is the statistical inefficiency of the equilibrated 
# sample, and ess is the effective sample size of the equilibrated sample.
idx, g, ess = deea.detect_equilibration_init_seq(my_timeseries, method="min_sse", plot=True)

# Alternatively, use the window method to account for autocorrelation.
# By default, this uses a Bartlett kernel and a window size of round(n_samples**0.5).
idx, g, ess = deea.detect_equilibration_window(my_timeseries, method="min_sse", plot=True)

# We can also determine equilibration in the same way as in pymbar.timeseries.detect_equilibration.
idx, g, ess = deea.detect_equilibration_init_seq(my_timeseries, method="max_ess", sequence_estimator="positive")
```

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
