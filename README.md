deea
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/fjclark/deea/workflows/CI/badge.svg)](https://github.com/fjclark/deea/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fjclark/deea/branch/main/graph/badge.svg)](https://codecov.io/gh/fjclark/deea/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Detection of Equilibration by Ensemble Analysis

A lightweight Python package for detecting equilibration in timeseries data where an initial transient is followed by a stationary distribution. Currently, two equilibration detection methods are implemented:

- Maximum effective sample size based on the lugsail replicated batch means method. See [here](https://projecteuclid.org/journals/statistical-science/volume-36/issue-4/Revisiting-the-GelmanRubin-Diagnostic/10.1214/20-STS812.full) and [here](https://academic.oup.com/biomet/article/109/3/735/6395353).

- Paired t-test. This checks for significant differences between the first 10 %, and last 50 % of the data. The test is repeated while sequentially removing more initial data. The first time point at which no significant difference is found is taken as the equilibration time.


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
# This should be a 2D numpy array with shape (n_runs, n_samples)
my_timeseries = ...

# Detect equilibration based on the maximum effective sample size 
# based on the lugsail replicated batch means method.
# idx is the index of the first sample after equilibration, g is the
# statistical inefficiency, and ess is the effective sample size.
idx, g, ess = deea.detect_equilibration_max_ess(my_timeseries, method="lugsail", plot=True)

# Detect equilibration using a paired t-test.
# idx is the index of the first sample after equilibration.
idx = deea.detect_equilibration_ttest(my_timeseries, plot=True)
```

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
