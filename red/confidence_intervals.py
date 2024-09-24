"""Convenience function for computing 95 % confidence intervals."""

from typing import Optional as _Optional

import numpy as _np
import numpy.typing as _npt
from scipy.stats import t as _t

from .variance import get_variance_initial_sequence as _get_variance_initial_sequence


def get_conf_int_initial_sequence(
    data: _npt.NDArray[_np.float64],
    conf_level_two_sided: float = 0.95,
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

    conf_level_two_sided : float, optional
        The two-sided confidence level. The default is 0.95.

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

    # Get the 95 % confidence interval.
    t_val = _t.ppf(1 - (1 - conf_level_two_sided) / 2, ess - 1)
    ci = t_val * sem

    return ci  # type: ignore[no-any-return]
