# Theory

???+ Info
    To see examples of these methods in action, see the [examples](examples.md) page.


## Equilibration Detection

This package provides a range of methods for selecting the truncation point of a time series, where the aim is to remove an initial "unequilibrated" portion of the data. These methods differ in the way they account for autocorrelation in the data, and whether they select the truncation point based on the minimum of the squared standard error or the maximum effective sample size. When using the minimum of the squared standard error, we can think of these as generalisations of the Marginal Standard Error Rule (MSER, [White, 1997](https://journals.sagepub.com/doi/abs/10.1177/003754979706900601)). Formally, the generalised MSER is:


\begin{align}
    n_0* &= \operatorname*{arg min}_{N > n_0 > 0}\left[\widehat{\mathrm{Var}}(\langle A \rangle_{[n_{0},N]})\right],
     \label{eqn:mser_correlated}
\end{align}

which means that we select the "optimal" truncation point, $n_0*$, so that it minimises $\widehat{\mathrm{Var}}(\langle A \rangle_{[n_{0},N]})$. $\widehat{\mathrm{Var}}(\langle A \rangle_{[n_{0},N]})$ is the estimated variance of the mean of some observable $A(\mathbf{x})$ over the range $[n_{0},N]$, where $n_0$ is the number of your first data point, and $N$ is the number of your final data point. The naive estimator of the variance of the mean is:

\begin{align}
    \widehat{\mathrm{Var}}_{\mathrm{Naive}}(\langle A \rangle_{[n_{0},N]}) &= \frac{1}{N_{n_0}}\sum^{N_{n_0}-1}_{t=-(N_{n_0}-1)}\hat{\gamma}_{t,[n_{0},N]}
    \label{eqn:var_naive_unsplit}\\
    &= \frac{1}{N_{n_0}}\left(\hat{\gamma}_{0,[n_{0},N]} +  2\sum^{N_{n_0}-1}_{t=1}\hat{\gamma}_{t,[n_{0},N]}\right),
    \label{eqn:var_naive}
\end{align}

where $N_{n_0} = N - n_0 + 1$ and the autocovariance terms are estimated as:

\begin{align}
    \hat{\gamma}_{t,[n_0, N]} &= \frac{1}{N_{n_0}} \sum^{N-t}_{n=n_0} (A(\mathbf{x}_n) - \langle A \rangle_{[n_{0},N]})(A(\mathbf{x}_{n+t}) - \langle A \rangle_{[n_{0},N]}).
    \label{eqn:autocov_estimate}
\end{align}

The key issue is that we can't add up all the autocovariance terms, as our estimate of the variance would become very noisy. The methods discussed calculate this sum differently. We'll start with the methods which most rigorously account for correlation by including the most terms, then move through the spectrum of methods to end with the original MSER, which does not account for correlation at all and only includes the $\hat{\gamma}_{0,[n_0, N]}$ term.

### Initial sequence methods fully account for autocorrelation

Geyer's initial sequence methods ([Geyer, 1992](https://www.jstor.org/stable/2246094)) apply certain rules to the sum of autocovariance terms to ensure that they make sense of Markov chains. The initial sequence methods are the most rigorous in terms of accounting for autocorrelation, in that they include the most terms from the autocovariance sum. Geyer's methods, in order of increasing strictness, are "initial positive" < "initial monotone" < "Initial convex". Chodera proposed simply truncating the sum at the first negative value ([Chodera, 2016](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5b00784)) - we include this method with the initial sequence methods. These are implemented in the [`detect_equilibration_init_seq`][red.equilibration.detect_equilibration_init_seq] function.

### The window method partially accounts for autocorrelation

The window method ([Geyer, 1992](https://www.jstor.org/stable/2246094)) weights each of the terms in the autocovariance sum according to some "window" function. This method includes more or less terms of the autocovariance series depending on the window size. The window can be of fixed size, or change with $N_{n_0}$. These are implemented in the [`detect_equilibration_window`][red.equilibration.detect_equilibration_window] function.

### MSER: The method which ignores autocorrelation

White's original Marginal Standard Error Rule ([White, 1997](https://journals.sagepub.com/doi/abs/10.1177/003754979706900601)) simply truncates the autocovariance sum at the first term, $\hat{\gamma}_{0,[n_0, N]}$, so that autocorrelation is ignored. You can think of it as a special case of the window method with a window size of 1.

### ESS

Selecting the truncation point based on the maximum effective sample size amounts to using the formula:

\begin{align}
    n_0* &= \operatorname*{arg min}_{N > n_0 > 0}\left[\frac{\widehat{\mathrm{Var}}_{\mathrm{Trajs}}(\langle A \rangle_{[n_{0},N]})}{\widehat{\mathrm{Var}}_{[n_{0},N]}(A(\mathrm{\mathbf{x}}))}  \right]
    \label{eqn:max_ess_algo}
\end{align}

which is very similar to the generalised MSER formula. We've found that we generally get very similar results using each approach, but that the min SSE approach sometimes avoids issues with one or two very different samples at the start of the timeseries (where the ESS approach selects a truncation point that includes these samples).

### Which method should I use?

???+ Note "Recommended method"
    For **equilibration detection** we recommend starting with the window method with window size $\sqrt{N_{n_0}}$, which is the default for [`detect_equilibration_window`][red.equilibration.detect_equilibration_window].

We've tested the above generalised MSER methods on synthetic single-run data modelled on absolute binding free energy calculations for a range of systems. In general, the methods which more thoroughly accounted for autocorrelation we more liable to choose late and variable truncation points, while the methods which less thoroughly accounted for autocorrelation were liable to truncate early. We found that the original MSER was generally the worst-performing method, and that the window method with a window size of $\sqrt{N_{n_0}}$ was the best-performing method, striking a balance between accounting for autocorrelation and not truncating too late/ variably. However, the best method for your data may vary, so we recommend trying a few different methods on a few test time series to get a feel for the best method for you.

## Uncertainty Estimation

???+ Note "Recommended method"
    For **uncertainty estimation** we recommend using Geyer's initial convex sequence method, which is the default for [`get_conf_int_init_seq`][red.confidence_intervals.get_conf_int_init_seq].

Estimating the uncertainty in the mean of a time series is a different problem to finding the optimum truncation point. In this case, you want to rigorously account for autocorrelation to avoid underestimating the uncertainty. Therefore, we recommend Geyer's initial convex sequence method ([Geyer, 1992](https://www.jstor.org/stable/2246094)) as the strictest of the initial sequence methods. Note that selecting your equilibration point based on the minimum SSE (or maximum ESS) before, then estimating the uncertainty on the truncated data biases your uncertainty estimates downwards. This is especially true if you use a method which thoroughly accounts for autocorrelation for both equilibration detection and uncertainty estimation.
