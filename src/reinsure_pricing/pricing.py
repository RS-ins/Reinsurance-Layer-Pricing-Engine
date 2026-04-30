from dataclasses import dataclass
from reinsure_pricing.simulation import SimulationResults


@dataclass
class PricingResult:
    expected_ceded_loss: float
    risk_margin: float
    expense_loading: float
    technical_premium: float
    loss_ratio_at_technical: float
    rate_on_line: float
    treaty_limit: float

    def summary(self) -> str:
        lines = [
            f"Expected Ceded Loss  : {self.expected_ceded_loss:>15,.0f}",
            f"Risk Margin          : {self.risk_margin:>15,.0f}",
            f"Expense Loading      : {self.expense_loading:>15,.0f}",
            f"─" * 40,
            f"Technical Premium    : {self.technical_premium:>15,.0f}",
            f"Rate on Line         : {self.rate_on_line:>15.2%}",
            f"Loss Ratio           : {self.loss_ratio_at_technical:>15.2%}",
        ]
        return "\n".join(lines)


class TechnicalPricer:
    """
    Computes technical premium from simulation results.

    technical_premium = ECL + risk_margin + expense_loading

    where:
        ECL          = expected ceded loss (mean of simulated ceded losses)
        risk_margin  = tvar_multiplier * (TVaR - ECL)
        expense_load = expense_ratio * technical_premium  (solved analytically)
    """

    def __init__(self, tvar_multiplier: float = 0.5,
                 expense_ratio: float = 0.15,
                 tvar_alpha: float = 0.99):
        if not 0 <= tvar_multiplier <= 1:
            raise ValueError("tvar_multiplier must be between 0 and 1")
        if not 0 <= expense_ratio < 1:
            raise ValueError("expense_ratio must be between 0 and 1")
        self.tvar_multiplier = tvar_multiplier
        self.expense_ratio = expense_ratio
        self.tvar_alpha = tvar_alpha

    def price(self, results: SimulationResults,
              treaty_limit: float) -> PricingResult:

        ecl = results.expected_ceded_loss
        tvar = results.tvar(self.tvar_alpha)

        risk_margin = self.tvar_multiplier * (tvar - ecl)

        # Solve for premium: P = ECL + risk_margin + expense_ratio * P
        # P * (1 - expense_ratio) = ECL + risk_margin
        technical_premium = (ecl + risk_margin) / (1 - self.expense_ratio)
        expense_loading = technical_premium * self.expense_ratio

        return PricingResult(
            expected_ceded_loss=ecl,
            risk_margin=risk_margin,
            expense_loading=expense_loading,
            technical_premium=technical_premium,
            loss_ratio_at_technical=ecl / technical_premium,
            rate_on_line=technical_premium / treaty_limit,
            treaty_limit=treaty_limit,
        )