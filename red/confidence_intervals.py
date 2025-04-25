"""Convenience function for computing 95 % confidence intervals."""

from typing import Optional as _Optional
from warnings import warn as _warn

import numpy as _np
import numpy.typing as _npt
from scipy.stats import t as _t

from .variance import get_variance_initial_sequence as _get_variance_initial_sequence


def get_conf_int_init_seq(
    data: _npt.NDArray[_np.float64],
    alpha_two_tailed: float = 0.05,
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
) -> float:
    """
    Calculate the confidence interval for the mean of a time
    series using initial sequence methods. See Geyer, 1992:
    https://www.jstor.org/stable/2246094.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    alpha_two_tailed : float, optional
        The two-tailed significance level to use. The default is 0.05.

    sequence_estimator : str, optional
        The initial sequence estimator to use. Can be "positive", "initial_positive",
        "initial_monotone", or "initial_convex". The default is "initial_convex". "positive"
        corresponds to truncating the auto-covariance function at the first negative value, as is
        done in pymbar. The other methods correspond to the methods described in Geyer, 1992:
        https://www.jstor.org/stable/2246094.

    min_max_lag_time : int, optional, default=3
        The minimum maximum lag time to use when estimating the statistical inefficiency.

    max_max_lag_time : int, optional, default=None
        The maximum maximum lag time to use when calculating the auto-correlation function.
        If None, the maximum lag time will be the length of the time series.

    Returns
    -------
    float
        The standard error of the mean.
    """
    # Get the correlated estimate of the variance.
    var_cor, max_lag, acovf = _get_variance_initial_sequence(
        data=data,
        sequence_estimator=sequence_estimator,
        min_max_lag_time=min_max_lag_time,
        max_max_lag_time=max_max_lag_time,
    )

    # Get the standard error of the mean.
    sem = _np.sqrt(var_cor / data.size)

    # Get the effective sample size.
    g = var_cor / acovf[0]
    ess = data.size / g

    # Raise a warning for low effective sample size.
    if ess < 50:  # Arbitrary, but matches pymbar timeseries
        _warn(
            f"Effective sample size is low: {ess}. Confidence intervals may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get the 95 % confidence interval.
    t_val = _t.ppf(1 - alpha_two_tailed / 2, ess - 1)

    ci = t_val * sem

    return float(ci)
