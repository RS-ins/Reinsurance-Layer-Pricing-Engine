"""
risk_measures.py
----------------
Risk measure computation for reinsurance layer pricing.

Provides a comprehensive suite of risk metrics computed from the
empirical ceded loss distribution produced by the Monte Carlo engine.

The RiskMeasures dataclass collects all metrics in a single object
for convenient reporting and downstream use in pricing and notebooks.

The compute_risk_measures() function is the main entry point — it
takes a SimulationResults object and treaty limit and returns a
fully populated RiskMeasures instance.

Risk measures implemented:
    - Expected Ceded Loss (ECL)
    - Standard Deviation
    - Coefficient of Variation (CV)
    - Skewness
    - VaR at 95%, 99%, 99.5%
    - TVaR at 95%, 99%, 99.5%
    - Probability of Attachment
    - Probability of Exhaustion
"""

import numpy as np
from dataclasses import dataclass
from reinsure_pricing.simulation import SimulationResults


@dataclass
class RiskMeasures:
    """
    Container for the full suite of risk measures for a reinsurance layer.

    All monetary values are in the same currency as the simulation inputs.
    All probabilities are expressed as floats between 0 and 1.

    Parameters
    ----------
    expected_ceded_loss : float
        Mean annual ceded loss across all simulated years (pure premium).
    std_deviation : float
        Standard deviation of annual ceded losses. Measures volatility
        of the layer — high std relative to ECL signals an unstable layer.
    var_95 : float
        Value at Risk at 95% confidence — the 95th percentile of the
        ceded loss distribution.
    var_99 : float
        Value at Risk at 99% confidence — the 99th percentile.
    var_995 : float
        Value at Risk at 99.5% confidence — the 99.5th percentile.
        Used in Solvency II SCR calculations.
    tvar_95 : float
        Tail Value at Risk at 95% — average loss in the worst 5% of years.
    tvar_99 : float
        Tail Value at Risk at 99% — average loss in the worst 1% of years.
        Used as the capital proxy in the technical pricing formula.
    tvar_995 : float
        Tail Value at Risk at 99.5% — average loss in the worst 0.5% of years.
    prob_attachment : float
        Fraction of simulated years where the layer was triggered
        (ceded loss > 0). Indicates how frequently the reinsurer
        can expect to receive a claim from the cedant.
    prob_exhaustion : float
        Fraction of simulated years where the full treaty limit was
        consumed. High exhaustion probability indicates a heavily
        exposed layer commanding a higher premium.
    coefficient_of_variation : float
        Ratio of standard deviation to ECL (std / ECL). A scale-free
        measure of volatility — useful for comparing layers of
        different sizes. Higher CV indicates a more volatile layer.
    skewness : float
        Third standardised central moment of the ceded loss distribution.
        Positive skewness (typical for reinsurance layers) indicates
        a right-skewed distribution with a heavy upper tail.
    """

    expected_ceded_loss: float
    std_deviation: float
    var_95: float
    var_99: float
    var_995: float
    tvar_95: float
    tvar_99: float
    tvar_995: float
    prob_attachment: float
    prob_exhaustion: float
    coefficient_of_variation: float
    skewness: float

    def summary(self) -> str:
        """Return a formatted table of all risk measures."""
        lines = [
            f"{'─' * 45}",
            f"{'RISK MEASURES':^45}",
            f"{'─' * 45}",
            f"Expected Ceded Loss   : {self.expected_ceded_loss:>15,.0f}",
            f"Std Deviation         : {self.std_deviation:>15,.0f}",
            f"Coeff of Variation    : {self.coefficient_of_variation:>15.3f}",
            f"Skewness              : {self.skewness:>15.3f}",
            f"{'─' * 45}",
            f"VaR  95%              : {self.var_95:>15,.0f}",
            f"VaR  99%              : {self.var_99:>15,.0f}",
            f"VaR  99.5%            : {self.var_995:>15,.0f}",
            f"{'─' * 45}",
            f"TVaR 95%              : {self.tvar_95:>15,.0f}",
            f"TVaR 99%              : {self.tvar_99:>15,.0f}",
            f"TVaR 99.5%            : {self.tvar_995:>15,.0f}",
            f"{'─' * 45}",
            f"Prob of Attachment    : {self.prob_attachment:>15.1%}",
            f"Prob of Exhaustion    : {self.prob_exhaustion:>15.1%}",
            f"{'─' * 45}",
        ]
        return "\n".join(lines)


def compute_risk_measures(results: SimulationResults,
                          treaty_limit: float) -> RiskMeasures:
    """
    Compute the full suite of risk measures from simulation results.

    This is the main entry point for risk measure computation. It
    extracts the ceded loss array from the SimulationResults object
    and computes all metrics in a single pass.

    Parameters
    ----------
    results : SimulationResults
        Output from MonteCarloEngine.run(). Contains the full array
        of simulated annual ceded losses.
    treaty_limit : float
        Maximum ceded loss per year under the treaty. Used to compute
        the probability of exhaustion. For XL treaties this is the
        per-occurrence limit times a reasonable multiplier, or simply
        the treaty limit. For Stop-Loss this is the cap.

    Returns
    -------
    RiskMeasures
        Fully populated dataclass containing all risk metrics,
        ready for reporting, pricing, or sensitivity analysis.

    Notes
    -----
    Skewness is computed as the third standardised central moment:
        skewness = E[((X - mu) / sigma)^3]

    A positive skewness (typical for reinsurance layers) confirms
    the distribution has a heavy right tail — extreme years are
    significantly worse than average years.

    Coefficient of variation returns 0.0 if ECL is zero to avoid
    division by zero in degenerate cases (e.g. retention above
    all simulated losses).
    """

    losses = results.ceded_losses
    ecl    = results.expected_ceded_loss
    std    = results.std_ceded_loss

    return RiskMeasures(
        expected_ceded_loss      = ecl,
        std_deviation            = std,
        var_95                   = results.var(0.95),
        var_99                   = results.var(0.99),
        var_995                  = results.var(0.995),
        tvar_95                  = results.tvar(0.95),
        tvar_99                  = results.tvar(0.99),
        tvar_995                 = results.tvar(0.995),
        prob_attachment          = results.prob_attachment(),
        prob_exhaustion          = results.prob_exhaustion(treaty_limit),
        coefficient_of_variation = std / ecl if ecl > 0 else 0.0,

        # Third standardised central moment — measures right-skewness
        # of the ceded loss distribution. Returns 0.0 if std is zero.
        skewness = float(
            np.mean(((losses - ecl) / std) ** 3) if std > 0 else 0.0
        ),
    )