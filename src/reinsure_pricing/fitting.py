"""
fitting.py
----------
Distribution fitting for reinsurance frequency and severity data.

Fits frequency and severity distributions to historical loss data
using Maximum Likelihood Estimation (MLE). Selects the best-fitting
distribution using the Akaike Information Criterion (AIC).

AIC = 2k - 2 * log(L)

where k is the number of parameters and L is the maximised likelihood.
Lower AIC indicates a better fit. AIC penalises model complexity,
preventing overfitting when comparing distributions with different
numbers of parameters.

Usage:
    from reinsure_pricing.fitting import fit_frequency, fit_severity

    freq = fit_frequency(claim_counts=[98, 134, 112, 145, 89])
    sev  = fit_severity(losses=[234000, 891000, 1200000, 456000])

    print(freq)
    print(sev)
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from typing import Optional

from reinsure_pricing.frequency import PoissonFrequency, NegativeBinomialFrequency
from reinsure_pricing.severity import LognormalSeverity, GammaSeverity, ParetoSeverity


@dataclass
class FitResult:
    """
    Result of a distribution fitting procedure.

    Parameters
    ----------
    distribution : object
        The fitted distribution object ready to use in simulation.
    distribution_name : str
        Name of the fitted distribution.
    aic : float
        Akaike Information Criterion — lower is better.
    log_likelihood : float
        Maximised log-likelihood at the fitted parameters.
    n_params : int
        Number of parameters in the fitted distribution.
    n_observations : int
        Number of data points used for fitting.
    params : dict
        Fitted parameter values.
    """
    distribution:      object
    distribution_name: str
    aic:               float
    log_likelihood:    float
    n_params:          int
    n_observations:    int
    params:            dict

    def __str__(self) -> str:
        lines = [
            f"Best fit   : {self.distribution_name}",
            f"Parameters : {', '.join(f'{k}={v:.4f}' for k,v in self.params.items())}",
            f"AIC        : {self.aic:.2f}",
            f"Log-L      : {self.log_likelihood:.2f}",
            f"n          : {self.n_observations}",
        ]
        return "\n".join(lines)


@dataclass
class FittingComparison:
    """
    Comparison of all candidate distributions fitted to the same data.

    Parameters
    ----------
    best : FitResult
        The best-fitting distribution by AIC.
    all_results : list of FitResult
        All fitted distributions sorted by AIC (best first).
    data_summary : dict
        Basic summary statistics of the input data.
    """
    best:         FitResult
    all_results:  list
    data_summary: dict

    def summary(self) -> str:
        """Return a formatted comparison table of all fitted distributions."""
        lines = [
            f"{'─' * 65}",
            f"{'DISTRIBUTION FITTING COMPARISON':^65}",
            f"{'─' * 65}",
            f"{'Distribution':<25} {'AIC':>10} {'Log-L':>10} {'Params':>8} {'Best':>6}",
            f"{'─' * 65}",
        ]
        for r in self.all_results:
            best_marker = " ✓" if r == self.best else ""
            lines.append(
                f"{r.distribution_name:<25} {r.aic:>10.2f} "
                f"{r.log_likelihood:>10.2f} {r.n_params:>8} {best_marker:>6}"
            )
        lines.append(f"{'─' * 65}")
        lines.append(f"\nBest fit: {self.best.distribution_name}")
        lines.append(
            f"Parameters: "
            f"{', '.join(f'{k}={v:.4f}' for k,v in self.best.params.items())}"
        )
        lines.append(f"\nData summary:")
        for k, v in self.data_summary.items():
            lines.append(f"  {k:<20}: {v:.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Frequency fitting
# ─────────────────────────────────────────────

def _fit_poisson(counts: np.ndarray) -> FitResult:
    """
    Fit a Poisson distribution to claim count data via MLE.

    For Poisson, the MLE of lambda is simply the sample mean.
    The log-likelihood is:
        log L = sum(k * log(lambda) - lambda - log(k!))
    """
    lambda_mle = float(counts.mean())
    log_l      = float(stats.poisson.logpmf(counts, mu=lambda_mle).sum())
    aic        = 2 * 1 - 2 * log_l  # 1 parameter

    return FitResult(
        distribution      = PoissonFrequency(lambda_=lambda_mle),
        distribution_name = "Poisson",
        aic               = aic,
        log_likelihood    = log_l,
        n_params          = 1,
        n_observations    = len(counts),
        params            = {"lambda": lambda_mle},
    )


def _fit_negative_binomial(counts: np.ndarray) -> FitResult:
    """
    Fit a Negative Binomial distribution to claim count data via MLE.

    Uses scipy's nbinom parameterisation internally, then converts
    to the actuarial (mu, phi) convention:
        mu  = n(1-p)/p
        phi = 1/n

    The optimisation minimises the negative log-likelihood.
    """
    mu_init  = float(counts.mean())
    var_init = float(counts.var())

    # Method of moments initial estimate for phi
    phi_init = max((var_init - mu_init) / mu_init**2, 0.01)

    def neg_log_l(params):
        mu, phi = params
        if mu <= 0 or phi <= 0:
            return 1e10
        n = 1.0 / phi
        p = n / (n + mu)
        return -float(stats.nbinom.logpmf(counts, n=n, p=p).sum())

    result = minimize(
        neg_log_l,
        x0=[mu_init, phi_init],
        bounds=[(1e-6, None), (1e-6, None)],
        method="L-BFGS-B",
    )

    mu_mle, phi_mle = result.x
    log_l = -result.fun
    aic   = 2 * 2 - 2 * log_l  # 2 parameters

    return FitResult(
        distribution      = NegativeBinomialFrequency(mu=mu_mle, phi=phi_mle),
        distribution_name = "Negative Binomial",
        aic               = aic,
        log_likelihood    = log_l,
        n_params          = 2,
        n_observations    = len(counts),
        params            = {"mu": mu_mle, "phi": phi_mle},
    )


def fit_frequency(claim_counts: list | np.ndarray) -> FittingComparison:
    """
    Fit frequency distributions to historical claim count data.

    Tests Poisson and Negative Binomial distributions and selects
    the best fit by AIC. Returns both the best fit and a full
    comparison of all candidates.

    Parameters
    ----------
    claim_counts : list or np.ndarray
        Integer claim counts, one per accident year.
        Example: [98, 134, 112, 145, 89, 121, 103]

    Returns
    -------
    FittingComparison
        Contains the best-fitting distribution and a full comparison
        table. Use .best.distribution to get the fitted object ready
        for use in MonteCarloEngine.

    Examples
    --------
    >>> comp = fit_frequency([98, 134, 112, 145, 89])
    >>> print(comp.summary())
    >>> engine = MonteCarloEngine(comp.best.distribution, severity, treaty)
    """
    counts = np.asarray(claim_counts, dtype=int)

    if len(counts) < 5:
        raise ValueError(
            f"At least 5 years of claim count data required for fitting. "
            f"Got {len(counts)}."
        )
    if np.any(counts < 0):
        raise ValueError("Claim counts must be non-negative.")

    results = [
        _fit_poisson(counts),
        _fit_negative_binomial(counts),
    ]

    results.sort(key=lambda r: r.aic)
    best = results[0]

    data_summary = {
        "mean":               float(counts.mean()),
        "variance":           float(counts.var()),
        "dispersion (var/mean)": float(counts.var() / counts.mean()),
        "min":                float(counts.min()),
        "max":                float(counts.max()),
        "n_years":            float(len(counts)),
    }

    return FittingComparison(
        best=best,
        all_results=results,
        data_summary=data_summary,
    )


# ─────────────────────────────────────────────
# Severity fitting
# ─────────────────────────────────────────────

def _fit_lognormal(losses: np.ndarray) -> FitResult:
    """
    Fit a Lognormal distribution to individual loss data via MLE.

    For lognormal, the MLEs are:
        mu    = mean(log(X))
        sigma = std(log(X))
    """
    log_x  = np.log(losses)
    mu     = float(log_x.mean())
    sigma  = float(log_x.std(ddof=1))
    log_l  = float(stats.lognorm.logpdf(losses, s=sigma, scale=np.exp(mu)).sum())
    aic    = 2 * 2 - 2 * log_l

    return FitResult(
        distribution      = LognormalSeverity(mu=mu, sigma=sigma),
        distribution_name = "Lognormal",
        aic               = aic,
        log_likelihood    = log_l,
        n_params          = 2,
        n_observations    = len(losses),
        params            = {"mu": mu, "sigma": sigma},
    )


def _fit_gamma(losses: np.ndarray) -> FitResult:
    """
    Fit a Gamma distribution to individual loss data via MLE.

    Uses scipy's gamma.fit() with floc=0 to fix the location
    parameter at zero. Converts scipy's (a, scale) to the
    actuarial (mean, CV) convention.
    """
    a, _, scale = stats.gamma.fit(losses, floc=0)
    mean        = float(a * scale)
    cv          = float(1.0 / np.sqrt(a))
    log_l       = float(stats.gamma.logpdf(losses, a=a, scale=scale).sum())
    aic         = 2 * 2 - 2 * log_l

    return FitResult(
        distribution      = GammaSeverity(mean=mean, cv=cv),
        distribution_name = "Gamma",
        aic               = aic,
        log_likelihood    = log_l,
        n_params          = 2,
        n_observations    = len(losses),
        params            = {"mean": mean, "cv": cv},
    )


def _fit_pareto(losses: np.ndarray) -> FitResult:
    x_m   = float(losses.min())
    n     = len(losses)
    alpha = float(n / np.sum(np.log(losses / x_m)))

    # Invalidate if alpha ≤ 1.5 — produces near-infinite variance
    # and is not actuarially useful for severity modelling
    if alpha <= 1.5:
        return FitResult(
            distribution      = None,
            distribution_name = "Pareto",
            aic               = 1e10,
            log_likelihood    = -1e10,
            n_params          = 2,
            n_observations    = n,
            params            = {"alpha": alpha, "x_m": x_m},
        )

    log_l = float(
        n * np.log(alpha) + n * alpha * np.log(x_m)
        - (alpha + 1) * np.sum(np.log(losses))
    )
    aic = 2 * 2 - 2 * log_l

    return FitResult(
        distribution      = ParetoSeverity(alpha=alpha, x_m=x_m),
        distribution_name = "Pareto",
        aic               = aic,
        log_likelihood    = log_l,
        n_params          = 2,
        n_observations    = n,
        params            = {"alpha": alpha, "x_m": x_m},
    )


def fit_severity(losses: list | np.ndarray,
                 threshold: Optional[float] = None) -> FittingComparison:
    """
    Fit severity distributions to individual loss data.

    Tests Lognormal, Gamma, and Pareto distributions and selects
    the best fit by AIC. Returns both the best fit and a full
    comparison of all candidates.

    Parameters
    ----------
    losses : list or np.ndarray
        Individual loss amounts. Must be strictly positive.
        Example: [234000, 891000, 1200000, 456000, 678000]
    threshold : float, optional
        If provided, only losses above this threshold are used for
        fitting. Useful when fitting to losses above a deductible
        or retention level. Default is None (use all losses).

    Returns
    -------
    FittingComparison
        Contains the best-fitting distribution and a full comparison
        table. Use .best.distribution to get the fitted object ready
        for use in MonteCarloEngine.

    Examples
    --------
    >>> comp = fit_severity([234000, 891000, 1200000, 456000])
    >>> print(comp.summary())
    >>> engine = MonteCarloEngine(frequency, comp.best.distribution, treaty)

    >>> # Fit only to losses above 100K
    >>> comp = fit_severity(losses, threshold=100_000)
    """
    losses = np.asarray(losses, dtype=float)

    if threshold is not None:
        losses = losses[losses > threshold]
        if len(losses) == 0:
            raise ValueError(
                f"No losses above threshold {threshold:,.0f}. "
                f"Lower the threshold or provide more data."
            )

    if len(losses) < 10:
        raise ValueError(
            f"At least 10 individual losses required for fitting. "
            f"Got {len(losses)}."
        )
    if np.any(losses <= 0):
        raise ValueError("All loss amounts must be strictly positive.")

    results = [
        _fit_lognormal(losses),
        _fit_gamma(losses),
        _fit_pareto(losses),
    ]

    # Filter out invalid fits (e.g. Pareto with alpha ≤ 1)
    valid = [r for r in results if r.distribution is not None]
    if not valid:
        raise ValueError("No valid distribution fit found for the provided data.")

    valid.sort(key=lambda r: r.aic)
    best = valid[0]
    
    data_summary = {
        "mean":       float(losses.mean()),
        "std":        float(losses.std()),
        "cv":         float(losses.std() / losses.mean()),
        "skewness":   float(stats.skew(losses)),
        "min":        float(losses.min()),
        "median":     float(np.median(losses)),
        "max":        float(losses.max()),
        "n_losses":   float(len(losses)),
    }

    return FittingComparison(
        best=best,
        all_results=valid,
        data_summary=data_summary,
    )