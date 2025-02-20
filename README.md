
<p align="center">
  <img src="https://github.com/fjclark/red/assets/90148170/5b0cf397-f902-4a43-9323-6414aa408d1a" width="500">
</p>

<h2 align="center">Robust Equilibration Detection</h2>

<p align="center">
  <a href="https://github.com/fjclark/red/actions?query=workflow%3ACI">
    <img alt="ci" src="https://github.com/fjclark/red/workflows/CI/badge.svg" />
  </a>
  <a href="https://app.codacy.com/gh/fjclark/red/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
    <img alt="Codacy Badge" src="https://app.codacy.com/project/badge/Grade/fff40e5573f847399bee98eef495f8c6" />
  </a>
  <a href="https://codecov.io/gh/fjclark/red/branch/main">
    <img alt="codecov" src="https://codecov.io/gh/fjclark/red/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" />
  </a>
  <a href="https://mypy-lang.org/">
    <img alt="Checked with mypy" src="https://www.mypy-lang.org/static/mypy_badge.svg" />
  </a>
</p>

---


A Python package for detecting equilibration in time series where an initial transient is followed by a stationary distribution. Two main approaches are implemented, which differ in the way they account for autocorrelation:

  - `detect_equilibration_init_seq`: This uses the initial sequence methods of Geyer ([Geyer, 1992](https://www.jstor.org/stable/2246094)) to determine the truncation point of the sum of autocovariances. Chodera's method ([Chodera, 2016](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b00784)) of simply truncating the autocovariance series at the first negative value is also implemented.
  - `detect_equilibration_window`: This uses window methods (see [Geyer](https://www.jstor.org/stable/2246094) again) when calculating the
autocorrelation. Setting the window size to 1 will give you White's original Marginal Standard Error Rule ([White, 1997](https://journals.sagepub.com/doi/abs/10.1177/003754979706900601)).

For both, the equilibration point can be determined either according to the minimum of the squared standard error (the default), or the maximum effective sample size, by specifying `method="min_sse"` or `method="max_ess"`.

For testing and more details, please see the associated publication: **Clark, F.; Cole, D. J.; Michel, J. Robust Automated Truncation Point Selection for Molecular Simulations. J. Chem. Theory Comput. 2024. https://doi.org/10.1021/acs.jctc.4c01359.**

### Installation

The easiest way to install `red` is using `conda` (or `mamba`) (note that the conda-forge/ PyPI name is `red-molsim` to avoid conflicts):

```bash
conda install -c conda-forge red-molsim
```

Alternatively, you can install `red` from the Python Package Index (PyPI) using `pip`:

```bash
pip install red-molsim
```

### Usage

**Warning**: `red` will work with multi-run data, but has only been thoroughly tested with single-run data. Using multi-run data is likely to be more robust, but we have not verified this.

#### Equilibration Detection

```python
import red

# Load your timeseries of interest.
# This should be a 2D numpy array with shape (n_runs, n_samples),
# or a 1D numpy array with shape (n_samples).
my_timeseries = ...

# Detect equilibration based on the minimum squared standard error using
# using the window method with a Bartlett kernel with a window size of
# round(n_samples**0.5) to account for autocorrelation. idx is the index
# of the first sample after equilibration, g is the statistical
# inefficiency of the equilibrated sample, and ess is the effective sample
# size of the equilibrated sample.
idx, g, ess = red.detect_equilibration_window(my_timeseries,
                                              method="min_sse",
                                              plot=True)

# Alternatively, use Geyer's initial convex sequence method to account
# for autocorrelation.
idx, g, ess = red.detect_equilibration_init_seq(my_timeseries,
                                                method="min_sse",
                                                plot=True)

# We can also determine equilibration in the same way as in
# pymbar.timeseries.detect_equilibration(my_timeseries, fast=False)
idx, g, ess = red.detect_equilibration_init_seq(my_timeseries,
                                                method="max_ess",
                                                sequence_estimator="positive")
```

#### Uncertainty Quantification

```python
# Estimate the 95 % confidence interval, accounting for autocorrelation using Geyer's initial
# convex sequence method.
ci_95 = red.get_conf_int_init_seq(my_timeseries, alpha_two_tailed=0.05)

```

For more examples, see the [documentation](https://fjclark.github.io/red/latest/examples/).

### Copyright

Copyright (c) 2023, Finlay Clark


### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1, with several ideas (Makefile, documentation) borrowed from Simon Boothroyd's super helpful [python-template](https://github.com/SimonBoothroyd/python-template).
