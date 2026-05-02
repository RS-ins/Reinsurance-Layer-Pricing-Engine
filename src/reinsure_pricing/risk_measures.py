import numpy as np
from dataclasses import dataclass
from reinsure_pricing.simulation import SimulationResults


@dataclass
class RiskMeasures:
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
    """Compute full suite of risk measures from simulation results."""

    losses = results.ceded_losses
    ecl    = results.expected_ceded_loss
    std    = results.std_ceded_loss

    return RiskMeasures(
        expected_ceded_loss   = ecl,
        std_deviation         = std,
        var_95                = results.var(0.95),
        var_99                = results.var(0.99),
        var_995               = results.var(0.995),
        tvar_95               = results.tvar(0.95),
        tvar_99               = results.tvar(0.99),
        tvar_995              = results.tvar(0.995),
        prob_attachment       = results.prob_attachment(),
        prob_exhaustion       = results.prob_exhaustion(treaty_limit),
        coefficient_of_variation = std / ecl if ecl > 0 else 0.0,
        skewness              = float(
            np.mean(((losses - ecl) / std) ** 3) if std > 0 else 0.0
        ),
    )