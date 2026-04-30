from dataclasses import dataclass


@dataclass
class PricingResult:
    expected_ceded_loss: float
    expense_loading: float
    profit_loading: float
    capital_load: float
    technical_premium: float
    rate_on_line: float
    treaty_limit: float

    def summary(self) -> str:
        lines = [
            f"Expected Ceded Loss  : {self.expected_ceded_loss:>15,.0f}",
            f"Expense Loading      : {self.expense_loading:>15,.0f}",
            f"Profit Loading       : {self.profit_loading:>15,.0f}",
            f"Capital Load         : {self.capital_load:>15,.0f}",
            f"{'─' * 40}",
            f"Technical Premium    : {self.technical_premium:>15,.0f}",
            f"Rate on Line         : {self.rate_on_line:>15.2%}",
        ]
        return "\n".join(lines)


class TechnicalPricer:
    """
    Computes technical premium from simulation results.

    Technical Premium = ECL
                      + expense_load * ECL
                      + profit_load  * ECL
                      + cost_of_capital * max(TVaR99 - ECL, 0)
    """

    def __init__(self,
                 expected_ceded_loss: float,
                 tvar_99: float,
                 expense_load: float = 0.05,
                 profit_load: float = 0.08,
                 cost_of_capital: float = 0.10):

        if expected_ceded_loss < 0:
            raise ValueError("expected_ceded_loss must be non-negative")
        if not 0 <= expense_load < 1:
            raise ValueError("expense_load must be between 0 and 1")
        if not 0 <= profit_load < 1:
            raise ValueError("profit_load must be between 0 and 1")
        if not 0 <= cost_of_capital < 1:
            raise ValueError("cost_of_capital must be between 0 and 1")

        self.expected_ceded_loss = expected_ceded_loss
        self.tvar_99 = tvar_99
        self.expense_load = expense_load
        self.profit_load = profit_load
        self.cost_of_capital = cost_of_capital

    def technical_premium(self) -> float:
        ecl = self.expected_ceded_loss
        expense  = self.expense_load * ecl
        profit   = self.profit_load * ecl
        capital  = self.cost_of_capital * max(self.tvar_99 - ecl, 0)
        return ecl + expense + profit + capital

    def rate_on_line(self, treaty_limit: float) -> float:
        return self.technical_premium() / treaty_limit

    def price(self, treaty_limit: float) -> PricingResult:
        ecl      = self.expected_ceded_loss
        expense  = self.expense_load * ecl
        profit   = self.profit_load * ecl
        capital  = self.cost_of_capital * max(self.tvar_99 - ecl, 0)
        premium  = ecl + expense + profit + capital

        return PricingResult(
            expected_ceded_loss=ecl,
            expense_loading=expense,
            profit_loading=profit,
            capital_load=capital,
            technical_premium=premium,
            rate_on_line=premium / treaty_limit,
            treaty_limit=treaty_limit,
        )