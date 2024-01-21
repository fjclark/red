"""Functions to calculate the statistical inefficiency and effective sample size."""

from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
import pymbar

from ._validation import check_data
from .sse import get_sse_series_init_seq as _get_sse_series_init_seq
from .sse import get_sse_series_window as _get_sse_series_window
from .variance import inter_run_variance, intra_run_variance, lugsail_variance


def convert_sse_series_to_ess_series(
    data: _np.ndarray, sse_series: _np.ndarray
) -> _np.ndarray:
    """
    Convert a series of squared standard errors to a series of effective sample sizes.

    Parameters
    ----------
    sse_series : np.ndarray
        The squared standard error series.

    uncor_vars : np.ndarray
        The uncorrelated variances.

    Returns
    -------
    np.ndarray
        The effective sample size series.
    """
    # Validate the data.
    data = check_data(data, one_dim_allowed=True)

    # Now get the uncorrelated variances.
    uncor_vars = _np.zeros_like(sse_series)

    for i in range(len(sse_series)):
        # Get "biased", rather than n - 1, variance.
        uncor_vars[i] = data[:, i:].var()  # type: ignore

    return uncor_vars / sse_series  # type: ignore


def get_ess_series_init_seq(
    data: _np.ndarray,
    sequence_estimator: str = "initial_convex",
    min_max_lag_time: int = 3,
    max_max_lag_time: _Optional[int] = None,
    smooth_lag_times: bool = False,
    frac_padding: float = 0.1,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Compute a series of effective sample sizes for a time series as data
    is discarded from the beginning of the time series. The autocorrelation
    is computed using the sequence estimator specified.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

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

    smooth_lag_times : bool, optional, default=False
        Whether to smooth out the max lag times by a) converting them to a monotinically
        decreasing sequence and b) linearly interpolating between points where the sequence
        changes. This may be useful when the max lag times are noisy.

    frac_padding : float, optional, default=0.1
        The fraction of the end of the timeseries to avoid calculating the variance
        for. For example, if frac_padding = 0.1, the variance will be calculated
        for the first 90% of the time series. This helps to avoid noise in the
        variance when there are few data points.

    Returns
    -------
    np.ndarray
        The effective sample size series.

    np.ndarray
        The maximum lag times used.
    """
    sse_series, max_lag_times = _get_sse_series_init_seq(
        data,
        sequence_estimator=sequence_estimator,
        min_max_lag_time=min_max_lag_time,
        max_max_lag_time=max_max_lag_time,
        smooth_lag_times=smooth_lag_times,
        frac_padding=frac_padding,
    )

    ess_series = convert_sse_series_to_ess_series(data, sse_series)

    return ess_series, max_lag_times


def get_ess_series_window(
    data: _np.ndarray,
    kernel: _Callable[[int], _np.ndarray] = _np.bartlett,  # type: ignore
    window_size_fn: _Optional[_Callable[[int], int]] = lambda x: round(x**1 / 2),
    window_size: _Optional[int] = None,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Compute a series of effective sample sizes for a time series as data
    is discarded from the beginning of the time series. The squared standard
    error is computed using the window size and kernel specified.

    Parameters
    ----------
    data : numpy.ndarray
        A time series of data with shape (n_samples,).

    kernel : callable, optional, default=numpy.bartlett
        A function that takes a window size and returns a window function.

    window_size_fn : callable, optional, default=lambda x: round(x**1 / 2)
        A function that takes the length of the time series and returns the window size
        to use. If this is not None, window_size must be None.

    window_size : int, optional, default=None
        The size of the window to use, defined in terms of time lags in the
        forwards direction. If this is not None, window_size_fn must be None.

    Returns
    -------
    np.ndarray
        The squared standard error series.

    np.ndarray
        The window sizes used.
    """
    sse_series, max_lag_times = _get_sse_series_window(
        data, kernel=kernel, window_size_fn=window_size_fn, window_size=window_size
    )

    ess_series = convert_sse_series_to_ess_series(data, sse_series)

    return ess_series, max_lag_times


def statistical_inefficiency_inter_variance(data: _np.ndarray) -> float:
    """
    Compute the statistical inefficiency of a time series by dividing
    the inter-run variance estimate by the intra-run variance estimate.
    More than one run is required.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    Returns
    -------
    float
        The statistical inefficiency.
    """
    g = inter_run_variance(data) / intra_run_variance(data)
    # Ensure that the statistical inefficiency is at least 1.
    return max(g, 1)


def statistical_inefficiency_lugsail_variance(
    data: _np.ndarray, n_pow: float = 1 / 3
) -> float:
    """
    Compute the statistical inefficiency of a time series by dividing
    the lugsail replicated batch means variance estimate by the
    intra-run variance estimate. This is applicable to a single run.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3.

    Returns
    -------
    float
        The statistical inefficiency.
    """
    g = lugsail_variance(data, n_pow=n_pow) / intra_run_variance(data)
    # Ensure that the statistical inefficiency is at least 1.
    return max(g, 1)


def statistical_inefficiency_chodera(data: _np.ndarray) -> float:
    """
    Compute the statistical inefficiency of a time series using the
    Chodera method. This is applicable to a single run.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) or (n_samples,).

    Returns
    -------
    float
        The statistical inefficiency.
    """
    data = check_data(data, one_dim_allowed=True)
    if data.shape[0] == 1:
        return pymbar.timeseries.statistical_inefficiency(data[0], fast=False)
    else:
        return pymbar.timeseries.statistical_inefficiency(data.mean(axis=0), fast=False)


def ess_inter_variance(data: _np.ndarray) -> float:
    """
    Compute the effective sample size of a time series by dividing
    the total number of samples by the statistical inefficiency, where
    the statistical inefficiency is calculated using the ratio of the
    inter-run and intra-run variance estimates.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    Returns
    -------
    float
        The effective sample size.
    """
    data = check_data(data, one_dim_allowed=False)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_inter_variance(data)


def ess_lugsail_variance(data: _np.ndarray, n_pow: float = 1 / 3) -> float:
    """
    Compute the effective sample size of a time series by dividing
    the total number of samples by the statistical inefficiency, where
    the statistical inefficiency is calculated using the ratio of the
    lugsail replicated batch means and intra-run variance estimates.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) and must have at least two
        runs.

    n_pow : float, optional
        The power to use in the lugsail variance estimate. This
        should be between 0 and 1. The default is 1/3.

    Returns
    -------
    float
        The effective sample size.
    """
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples
    return total_samples / statistical_inefficiency_lugsail_variance(data, n_pow=n_pow)


def ess_chodera(data: _np.ndarray) -> float:
    """
    Compute the effective sample size of a time series by dividing
    the total number of samples by the statistical inefficiency, where
    the statistical inefficiency is calculated using the Chodera method.

    Parameters
    ----------
    data : np.ndarray
        The time series data. This should have shape
        (n_runs, n_samples) or (n_samples,).

    Returns
    -------
    float
        The effective sample size.
    """
    data = check_data(data, one_dim_allowed=True)
    n_runs, n_samples = data.shape
    total_samples = n_runs * n_samples
    # return total_samples / statistical_inefficiency_chodera(data)
    if n_runs > 1:
        data = data.mean(axis=0)
    return total_samples / statistical_inefficiency_geyer(data, method="con")


################## DELETE BELOW #####################
def statistical_inefficiency_geyer(A_n, method="con"):
    """Compute the statistical inefficiency of a timeseries using the methods of Geyer.

    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    method : str, optional, default='con'
        The method to use; matches notation from `initseq` from `mcmc` R package by Geyer.
        'pos' : initial positive sequence (IPS) estimator
        'dec' : initial monotone sequence (IMS) estimator
        'con' : initial convex sequence (ICS) estimator

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.

    Notes
    -----
    Implementation based on the `initseq` method from the `mcmc` R package by Geyer.

    References
    ----------
    [1] Geyer, CJ. Practical Markov chain Monte Carlo. Statistical Science 7(4):473-511, 1992.

    Examples
    --------

    Compute statistical inefficiency of timeseries data with known correlation time using different methods.

    >>> from pymbar import testsystems
    >>> A_n = testsystems.correlated_timeseries_example(N=100000, tau=10.0)
    >>> g_IPS = statisticalInefficiency_geyer(A_n, method='pos')
    >>> g_IMS = statisticalInefficiency_geyer(A_n, method='dec')
    >>> g_ICS = statisticalInefficiency_geyer(A_n, method='con')

    """

    if method not in ["pos", "dec", "con"]:
        raise Exception(
            "Unknown method '%s'; must be one of ['pos', 'dec', 'con']" % method
        )

    # Create numpy copies of input arguments.
    A_n = _np.array(A_n)

    # Subtract off sample mean.
    A_n -= A_n.mean()

    # Get the length of the timeseries.
    N = A_n.size

    # Compute sample variance.
    gamma_zero = (A_n**2).sum() / N

    # Compute sequential covariance pairs.
    gamma_pos = list()
    for i in range(int(_np.floor(N / 2))):
        lag1 = 2 * i
        gam1 = (A_n[0 : (N - lag1)] * A_n[lag1:N]).sum() / N
        lag2 = lag1 + 1
        gam2 = (A_n[0 : (N - lag2)] * A_n[lag2:N]).sum() / N

        # Terminate if sum is no longer positive.
        if (gam1 + gam2) < 0.0:
            break

        # Otherwise, store the consecutive sum.
        gamma_pos.append(gam1 + gam2)

    # Number of nonnegative values in array.
    ngamma = len(gamma_pos)

    # Compute IPS gamma sequence.
    gamma_pos = _np.array(gamma_pos)

    # Compute IMS gamma sequence.
    gamma_dec = _np.array(gamma_pos)
    for i in range(ngamma - 1):
        if gamma_dec[i] < gamma_dec[i + 1]:
            gamma_dec[i] = gamma_dec[i + 1]

    # Compute ICS gamma sequence.
    gamma_con = _np.array(gamma_dec)
    for i in range(ngamma - 1, 0, -1):
        gamma_con[i] -= gamma_con[i - 1]

    # Pool adjacent violators (PAVA) algorithm.
    puff = _np.zeros([ngamma], _np.float64)
    nuff = _np.zeros([ngamma], _np.int32)
    nstep = 0
    for j in range(1, ngamma):
        puff[nstep] = gamma_con[j]
        nuff[nstep] = 1
        nstep += 1
        while (nstep > 1) and (
            (puff[nstep - 1] / nuff[nstep - 1]) < (puff[nstep - 2] / nuff[nstep - 2])
        ):
            puff[nstep - 2] += puff[nstep - 1]
            nuff[nstep - 2] += nuff[nstep - 1]
            nstep -= 1

    j = 1
    for jstep in range(nstep):
        muff = puff[jstep] / nuff[jstep]
        for k in range(nuff[jstep]):
            gamma_con[j] = gamma_con[j - 1] + muff
            j += 1

    # Compute sample variance estimates.
    var_pos = (2 * gamma_pos.sum() - gamma_zero) / N
    var_dec = (2 * gamma_dec.sum() - gamma_zero) / N
    var_con = (2 * gamma_con.sum() - gamma_zero) / N

    # Compute statistical inefficiencies from sample mean var = var(A_n) / (N/g)
    # g = var / (var(A_n)/N)
    var_uncorr = gamma_zero / N
    g_pos = var_pos / var_uncorr
    g_dec = var_dec / var_uncorr
    g_con = var_con / var_uncorr

    # DEBUG
    # print "pos dec con : %12.3f %12.3f %12.3f" % (g_pos, g_dec, g_con)

    # Select appropriate g.
    if method == "pos":
        g = g_pos
    elif method == "dec":
        g = g_dec
    elif method == "con":
        g = g_con

    # g must be at least unity
    if g < 1.0:
        g = 1.0

    # Return the computed statistical inefficiency.
    return g


def statistical_inefficiency_geyer_indices(A_n, method="con"):
    """Compute the statistical inefficiency of a timeseries using the methods of Geyer.

    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    method : str, optional, default='con'
        The method to use; matches notation from `initseq` from `mcmc` R package by Geyer.
        'pos' : initial positive sequence (IPS) estimator
        'dec' : initial monotone sequence (IMS) estimator
        'con' : initial convex sequence (ICS) estimator

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.

    Notes
    -----
    Implementation based on the `initseq` method from the `mcmc` R package by Geyer.

    References
    ----------
    [1] Geyer, CJ. Practical Markov chain Monte Carlo. Statistical Science 7(4):473-511, 1992.

    Examples
    --------

    Compute statistical inefficiency of timeseries data with known correlation time using different methods.

    >>> from pymbar import testsystems
    >>> A_n = testsystems.correlated_timeseries_example(N=100000, tau=10.0)
    >>> g_IPS = statisticalInefficiency_geyer(A_n, method='pos')
    >>> g_IMS = statisticalInefficiency_geyer(A_n, method='dec')
    >>> g_ICS = statisticalInefficiency_geyer(A_n, method='con')

    """

    if method not in ["pos", "dec", "con"]:
        raise Exception(
            "Unknown method '%s'; must be one of ['pos', 'dec', 'con']" % method
        )

    # Create numpy copies of input arguments.
    A_n = _np.array(A_n)

    # Subtract off sample mean.
    A_n -= A_n.mean()

    # Get the length of the timeseries.
    N = A_n.size

    # Compute sample variance.
    gamma_zero = (A_n**2).sum() / N

    # Compute sequential covariance pairs.
    gamma_pos = list()
    for i in range(int(_np.floor(N / 2))):
        lag1 = 2 * i
        gam1 = (A_n[0 : (N - lag1)] * A_n[lag1:N]).sum() / N
        lag2 = lag1 + 1
        gam2 = (A_n[0 : (N - lag2)] * A_n[lag2:N]).sum() / N

        # Terminate if sum is no longer positive.
        if (gam1 + gam2) < 0.0:
            break

        # Otherwise, store the consecutive sum.
        gamma_pos.append(gam1 + gam2)

    # Number of nonnegative values in array.
    ngamma = len(gamma_pos)

    # Compute IPS gamma sequence.
    gamma_pos = _np.array(gamma_pos)

    # Compute IMS gamma sequence.
    gamma_dec = _np.array(gamma_pos)
    for i in range(ngamma - 1):
        if gamma_dec[i] < gamma_dec[i + 1]:
            gamma_dec[i] = gamma_dec[i + 1]

    # Compute ICS gamma sequence.
    gamma_con = _np.array(gamma_dec)
    for i in range(ngamma - 1, 0, -1):
        gamma_con[i] -= gamma_con[i - 1]

    # Pool adjacent violators (PAVA) algorithm.
    puff = _np.zeros([ngamma], _np.float64)
    nuff = _np.zeros([ngamma], _np.int32)
    nstep = 0
    for j in range(1, ngamma):
        puff[nstep] = gamma_con[j]
        nuff[nstep] = 1
        nstep += 1
        while (nstep > 1) and (
            (puff[nstep - 1] / nuff[nstep - 1]) < (puff[nstep - 2] / nuff[nstep - 2])
        ):
            puff[nstep - 2] += puff[nstep - 1]
            nuff[nstep - 2] += nuff[nstep - 1]
            nstep -= 1

    j = 1
    for jstep in range(nstep):
        muff = puff[jstep] / nuff[jstep]
        for k in range(nuff[jstep]):
            gamma_con[j] = gamma_con[j - 1] + muff
            j += 1

    # Compute sample variance estimates.
    var_pos = (2 * gamma_pos.sum() - gamma_zero) / N
    var_dec = (2 * gamma_dec.sum() - gamma_zero) / N
    var_con = (2 * gamma_con.sum() - gamma_zero) / N

    # Compute statistical inefficiencies from sample mean var = var(A_n) / (N/g)
    # g = var / (var(A_n)/N)
    var_uncorr = gamma_zero / N
    g_pos = var_pos / var_uncorr
    g_dec = var_dec / var_uncorr
    g_con = var_con / var_uncorr

    # DEBUG
    # print "pos dec con : %12.3f %12.3f %12.3f" % (g_pos, g_dec, g_con)

    # Select appropriate g.
    if method == "pos":
        g = g_pos
    elif method == "dec":
        g = g_dec
    elif method == "con":
        g = g_con

    # g must be at least unity
    if g < 1.0:
        g = 1.0

    # Return the computed statistical inefficiency.
    return (g,)


def statistical_inefficiency_multiscale(A_n, B_n=None, fast=False, mintime=3):
    """
    Compute the (cross) statistical inefficiency of (two) timeseries using multiscale method from Chodera.

    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    B_n : np.ndarray, float, optional, default=None
        B_n[n] is nth value of timeseries B.  Length is deduced from vector.
        If supplied, the cross-correlation of timeseries A and B will be estimated instead of the
        autocorrelation of timeseries A.
    fast : bool, optional, default=False
        f True, will use faster (but less accurate) method to estimate correlation
        time, described in Ref. [1] (default: False).
    mintime : int, optional, default=3
        minimum amount of correlation function to compute (default: 3)
        The algorithm terminates after computing the correlation time out to mintime when the
        correlation function first goes negative.  Note that this time may need to be increased
        if there is a strong initial negative peak in the correlation function.

    Returns
    -------
    g : np.ndarray,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.

    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.

    Examples
    --------

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> from pymbar import testsystems
    >>> A_n = testsystems.correlated_timeseries_example(N=100000, tau=5.0)
    >>> g = statisticalInefficiency_multiscale(A_n, fast=True)

    """

    # Create numpy copies of input arguments.
    A_n = _np.array(A_n)

    if B_n is not None:
        B_n = _np.array(B_n)
    else:
        B_n = _np.array(A_n)

    # Get the length of the timeseries.
    N = A_n.size

    # Be sure A_n and B_n have the same dimensions.
    if A_n.shape != B_n.shape:
        raise Exception("A_n and B_n must have same dimensions.")

    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0

    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()

    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(_np.float64) - mu_A
    dB_n = B_n.astype(_np.float64) - mu_B

    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean()  # standard estimator to ensure C(0) = 1

    # Trap the case where this covariance is zero, and we cannot proceed.
    if sigma2_AB == 0:
        raise ParameterError(
            "Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency"
        )

    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while t < N - 1:
        # compute normalized fluctuation correlation function at time t
        C = _np.sum(dA_n[0 : (N - t)] * dB_n[t:N] + dB_n[0 : (N - t)] * dA_n[t:N]) / (
            2.0 * float(N - t) * sigma2_AB
        )
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break

        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t) / float(N)) * float(increment)

        # Increment t and the amount by which we increment t.
        t += increment

        # Increase the interval if "fast mode" is on.
        if fast:
            increment += 1

    # g must be at least unity
    if g < 1.0:
        g = 1.0

    # Return the computed statistical inefficiency.
    return g, t
