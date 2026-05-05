"""
bootstrap.py
------------
Bootstrapped confidence intervals for reinsurance risk measures.

Bootstrapping resamples the simulated ceded loss distribution with
replacement to estimate the sampling uncertainty of each risk measure.
This answers the question: how stable are our risk measure estimates
given the number of simulations run?

A wide confidence interval means the estimate is sensitive to the
specific random draws — more simulations are needed. A narrow interval
means the estimate is stable and can be trusted.

Usage:
    from reinsure_pricing.bootstrap import bootstrap_risk_measures

    results  = engine.run()
    boot     = bootstrap_risk_measures(results, treaty_limit=treaty.limit)
    print(boot.summary())
"""

import numpy as np
from dataclasses import dataclass
from reinsure_pricing.simulation import SimulationResults
from reinsure_pricing.risk_measures import RiskMeasures, compute_risk_measures


@dataclass
class BootstrappedInterval:
    """
    A single bootstrapped confidence interval for one risk measure.

    Parameters
    ----------
    point_estimate : float
        The original estimate from the full simulation.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    confidence_level : float
        Confidence level of the interval (e.g. 0.95 for 95%).
    n_bootstrap : int
        Number of bootstrap resamples used.
    """
    point_estimate:   float
    ci_lower:         float
    ci_upper:         float
    confidence_level: float
    n_bootstrap:      int

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_width(self) -> float:
        """
        Width as a fraction of the point estimate.
        Useful for comparing stability across measures of different scales.
        """
        return self.width / self.point_estimate if self.point_estimate > 0 else 0.0

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        return (f"{self.point_estimate:,.0f} "
                f"({pct}% CI: {self.ci_lower:,.0f} — {self.ci_upper:,.0f})")


@dataclass
class BootstrappedRiskMeasures:
    """
    Bootstrapped confidence intervals for the full suite of risk measures.

    Each field contains a BootstrappedInterval with the point estimate
    and confidence interval for that measure.

    Parameters
    ----------
    ecl : BootstrappedInterval
    std : BootstrappedInterval
    var_95 : BootstrappedInterval
    var_99 : BootstrappedInterval
    var_995 : BootstrappedInterval
    tvar_95 : BootstrappedInterval
    tvar_99 : BootstrappedInterval
    tvar_995 : BootstrappedInterval
    prob_attachment : BootstrappedInterval
    prob_exhaustion : BootstrappedInterval
    confidence_level : float
    n_bootstrap : int
    n_simulations : int
    """
    ecl:              BootstrappedInterval
    std:              BootstrappedInterval
    var_95:           BootstrappedInterval
    var_99:           BootstrappedInterval
    var_995:          BootstrappedInterval
    tvar_95:          BootstrappedInterval
    tvar_99:          BootstrappedInterval
    tvar_995:         BootstrappedInterval
    prob_attachment:  BootstrappedInterval
    prob_exhaustion:  BootstrappedInterval
    confidence_level: float
    n_bootstrap:      int
    n_simulations:    int

    def summary(self) -> str:
        """Return a formatted table of all bootstrapped confidence intervals."""
        pct = int(self.confidence_level * 100)
        lines = [
            f"{'─' * 70}",
            f"{'BOOTSTRAPPED RISK MEASURES':^70}",
            f"{'n_simulations = '+str(self.n_simulations):^70}",
            f"{'n_bootstrap = '+str(self.n_bootstrap)+' | CI = '+str(pct)+'%':^70}",
            f"{'─' * 70}",
            f"{'Measure':<22} {'Estimate':>12} {'CI Lower':>12} {'CI Upper':>12} {'Rel Width':>10}",
            f"{'─' * 70}",
        ]
        for label, bi in [
            ("ECL",              self.ecl),
            ("Std Deviation",    self.std),
            ("VaR 95%",          self.var_95),
            ("VaR 99%",          self.var_99),
            ("VaR 99.5%",        self.var_995),
            ("TVaR 95%",         self.tvar_95),
            ("TVaR 99%",         self.tvar_99),
            ("TVaR 99.5%",       self.tvar_995),
            ("Prob Attachment",  self.prob_attachment),
            ("Prob Exhaustion",  self.prob_exhaustion),
        ]:
            if label.startswith("Prob"):
                lines.append(
                    f"{label:<22} {bi.point_estimate:>11.1%} "
                    f"{bi.ci_lower:>11.1%} "
                    f"{bi.ci_upper:>11.1%} "
                    f"{bi.relative_width:>10.1%}"
                )
            else:
                lines.append(
                    f"{label:<22} {bi.point_estimate:>12,.0f} "
                    f"{bi.ci_lower:>12,.0f} "
                    f"{bi.ci_upper:>12,.0f} "
                    f"{bi.relative_width:>10.1%}"
                )
        lines.append(f"{'─' * 70}")
        lines.append(
            "Rel Width = CI width / point estimate. "
            "< 5% = stable. > 10% = consider more simulations."
        )
        return "\n".join(lines)


def _bootstrap_statistic(losses: np.ndarray,
                         stat_fn,
                         n_bootstrap: int,
                         rng: np.random.Generator,
                         confidence_level: float) -> BootstrappedInterval:
    """
    Compute a bootstrapped confidence interval for a single statistic.

    Parameters
    ----------
    losses : np.ndarray
        The simulated ceded loss array to resample from.
    stat_fn : callable
        Function that takes a loss array and returns a scalar statistic.
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : np.random.Generator
        Random generator for reproducibility.
    confidence_level : float
        Confidence level for the interval.

    Returns
    -------
    BootstrappedInterval
        Point estimate and confidence interval for the statistic.
    """
    n = len(losses)
    point_estimate = stat_fn(losses)

    # Draw all bootstrap samples at once for efficiency
    # Shape: (n_bootstrap, n_simulations)
    indices  = rng.integers(0, n, size=(n_bootstrap, n))
    samples  = losses[indices]

    # Apply statistic to each bootstrap sample
    boot_stats = np.apply_along_axis(stat_fn, 1, samples)

    alpha   = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrappedInterval(
        point_estimate=float(point_estimate),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_risk_measures(
        results: SimulationResults,
        treaty_limit: float,
        n_bootstrap: int = 1_000,
        confidence_level: float = 0.95,
        random_state: int = 42,
) -> BootstrappedRiskMeasures:
    """
    Compute bootstrapped confidence intervals for all risk measures.

    Resamples the simulated ceded loss distribution with replacement
    n_bootstrap times, computing each risk measure on every resample.
    The spread of the bootstrap distribution gives the confidence interval.

    Parameters
    ----------
    results : SimulationResults
        Output from MonteCarloEngine.run().
    treaty_limit : float
        Treaty limit — used to compute exhaustion probability.
    n_bootstrap : int
        Number of bootstrap resamples. 1,000 is sufficient for 95% CIs.
        Use 5,000 for more precise intervals. Default is 1,000.
    confidence_level : float
        Confidence level for the intervals. Default is 0.95 (95%).
    random_state : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    BootstrappedRiskMeasures
        Dataclass containing bootstrapped intervals for all risk measures.

    Notes
    -----
    Memory usage: n_bootstrap × n_simulations floats are held in memory
    during computation. For n_bootstrap=1,000 and n_simulations=100,000
    this is ~800MB. If memory is a concern, reduce n_bootstrap or
    n_simulations, or set chunk_size in a future version.

    For VaR and TVaR at high confidence levels (99.5%), the bootstrap
    distribution may be noisy if n_simulations is small (< 50,000)
    because only a handful of observations fall in the tail.
    """
    losses = results.ceded_losses
    rng    = np.random.default_rng(random_state)

    def var(alpha):
        return lambda x: float(np.quantile(x, alpha))

    def tvar(alpha):
        def _tvar(x):
            v    = np.quantile(x, alpha)
            tail = x[x > v]
            return float(tail.mean()) if len(tail) > 0 else float(v)
        return _tvar

    def prob_attach(x):
        return float((x > 0).mean())

    def prob_exhaust(x):
        return float((x >= treaty_limit).mean())

    print(f"Running {n_bootstrap} bootstrap resamples "
          f"(n_simulations={len(losses):,})...")

    return BootstrappedRiskMeasures(
        ecl             = _bootstrap_statistic(losses, np.mean,        n_bootstrap, rng, confidence_level),
        std             = _bootstrap_statistic(losses, np.std,         n_bootstrap, rng, confidence_level),
        var_95          = _bootstrap_statistic(losses, var(0.95),      n_bootstrap, rng, confidence_level),
        var_99          = _bootstrap_statistic(losses, var(0.99),      n_bootstrap, rng, confidence_level),
        var_995         = _bootstrap_statistic(losses, var(0.995),     n_bootstrap, rng, confidence_level),
        tvar_95         = _bootstrap_statistic(losses, tvar(0.95),     n_bootstrap, rng, confidence_level),
        tvar_99         = _bootstrap_statistic(losses, tvar(0.99),     n_bootstrap, rng, confidence_level),
        tvar_995        = _bootstrap_statistic(losses, tvar(0.995),    n_bootstrap, rng, confidence_level),
        prob_attachment = _bootstrap_statistic(losses, prob_attach,    n_bootstrap, rng, confidence_level),
        prob_exhaustion = _bootstrap_statistic(losses, prob_exhaust,   n_bootstrap, rng, confidence_level),
        confidence_level=confidence_level,
        n_bootstrap     =n_bootstrap,
        n_simulations   =len(losses),
    )