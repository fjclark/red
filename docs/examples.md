# Examples

???+ Note "Recommended methods"
    For **equilibration detection**, we recommend starting with the [`detect_equilibration_window`][red.equilibration.detect_equilibration_window] function with "min_sse" method and a window size function of `round(n_samples**0.5)` (the defaults):
    ```python
    idx, g, ess = red.detect_equilibration_window(my_timeseries, plot=True)
    my_truncated_timeseries = my_timeseries[idx:]
    ```

    For **uncertainty_estimation**, we recommend using Geyer's initial convex sequence method ([Geyer, 1992](https://www.jstor.org/stable/2246094)), which is the default for [`get_variance_initial_sequence`][red.variance.get_variance_initial_sequence]:
    ```python
    var, max_lag, acovf = red.get_variance_initial_sequence(my_timeseries)
    std_err = np.sqrt(var)
    ```

This package provides a range of methods for selecting the truncation point of a time series, where the aim is to remove an initial "unequilibrated" portion of the data. Functionality is also provided for estimating the variance of the mean of a time series, which is useful for uncertainty estimation. For details, please see the [theory](theory.md) page.

For all examples, `my_timeseries` should be a numpy array with shape `(n_samples,)`, or `(n_samples, n_repeats)` if you have multiple repeats of the same simulation.

???+ Warning
    These methods only been thoroughtly tested on-single run data. Using multi-run data is likely to be more robust, but we have not verified this.

## Detecting Equilibration

### Initial Sequence Methods

To use any of Geyer's initial sequence methods ([Geyer, 1992](https://www.jstor.org/stable/2246094)), you can specify the "sequence_estimator" to be "initial_positive" (the least strict), "initial_monotone", or "initial_convex" (the strictest):

```python
idx, g, ess = red.detect_equilibration_init_seq(my_timeseries, sequence_estimator="initial_convex", plot=True)
my_truncated_timeseries = my_timeseries[idx:]
```
To use Chodera's method of simply truncating the autocovariance series at the first negative value ([Chodera, 2016](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b00784)), you can specify the "sequence estimator" to be "positive".

### Window Methods

When using window methods, you can either specify a fixed window size, or a window size function which computes the window size as a function of the number of data points (which decreases as the truncation point increases). These are specified via `window_size` and `window_size_fn`, respectively (one must be specified and the other must be `None`). The default window size function is `lambda x: round(x**0.5)` - explicitly:

```python
idx, g, ess = red.detect_equilibration_window(my_timeseries, window_size=None, window_size_fn=lambda x: round(x**0.5), plot=True)
# This is equivalent to:
idx, g, ess = red.detect_equilibration_window(my_timeseries, plot=True)
```

To use a window size of 10:

```python
idx, g, ess = red.detect_equilibration_window(my_timeseries, window_size=10, window_size_fn = None, plot=True)
```

You can also play with the kernel function used in the window method by specifying the `kernel` argument. You should supply the function directly - the default is `np.bartlett`.

### The Original Marginal Standard Error Rule

To use White's original Marginal Standard Error Rule ([White, 1997](https://journals.sagepub.com/doi/abs/10.1177/003754979706900601)), you can use the window method with a window size of 1:

```python
idx, g, ess = red.detect_equilibration_window(my_timeseries, window_size=1, window_size_fn=None, plot=True)
```

### Maximum Effective Sample Size and Chodera's Method

To select the truncation point according to the maximum effective sample size (instead of the minimum squared standard error), you can specify the `method` argument to be "max_ess". To use Chodera's method ([Chodera, 2016](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b00784)) as implemented in `pymbar.timeseries`, you can specify the `sequence_estimator` to be "positive":

```python
idx, g, ess = red.detect_equilibration_init_seq(my_timeseries, method="max_ess", sequence_estimator="positive", plot=True)
# Equivalent to pymbar.timeseries.detect_equilibration(my_timeseries, fast=False)
```

### Plotting

To save a plot showing the (block-averaged) time series and variance of the mean/ effective sample size against truncation time, simply specify `plot=True` and, optionally, specify a name for the plot with `plot_name`. This works for either of the equilbration detection functions.

```python
idx, g, ess = red.detect_equilibration_window(my_timeseries, plot=True, plot_name="my_equilibration_plot.png")
```

## Estimating Uncertainty

To calculate uncertainty, we recommend using Geyer's initial convex sequence method ([Geyer, 1992](https://www.jstor.org/stable/2246094)), which is the default for [`get_conf_int_init_seq`][red.confidence_intervals.get_conf_int_init_seq]. For example, to estimate a 95 % confidence interval:

```python
ci_95 = red.get_conf_int_init_seq(my_timeseries, sequence_estimator="initial_convex", alpha_two_tailed=0.05)
```

This function has a similar interface to [`detect_equilibration_init_seq`][red.equilibration.detect_equilibration_init_seq], and you can specify the "sequence_estimator" in the same way. Note that this assumes we have a reasonable effective sample size, and hence that the means are approximately normally distributed by the central limit theorem.

???+ Warning
    Estimates of uncertainty based on single runs are very likely to be underestimates. Estimating uncertainty from deviations between repeat runs is more robust.
