"""
pricing.py
----------
Technical premium computation for reinsurance layers.

The TechnicalPricer takes the output of a Monte Carlo simulation and
computes a risk-adjusted technical premium using a cost-of-capital
loading approach consistent with Solvency II principles.

The premium is built up from four components:

    Technical Premium = ECL
                      + expense_load  × ECL
                      + profit_load   × ECL
                      + cost_of_capital × max(TVaR99 - ECL, 0)

where:
    ECL              = Expected Ceded Loss (pure premium)
    expense_load     = proportional load for acquisition and admin costs
    profit_load      = target profit margin as a fraction of ECL
    cost_of_capital  = required return on the risk capital held to
                       support the unexpected loss (TVaR99 - ECL)

The capital load is applied only to the unexpected loss — the portion
of the tail that exceeds the ECL and requires actual capital support.
This reflects the economic reality that the reinsurer must hold capital
against adverse deviations, not against the expected loss which is
covered by the pure premium.
"""

from dataclasses import dataclass


@dataclass
class PricingResult:
    """
    Container for the output of a technical pricing calculation.

    Stores all premium components separately so the user can inspect
    how the technical premium was built up and identify which loading
    drives the result.

    Parameters
    ----------
    expected_ceded_loss : float
        The pure premium — mean simulated ceded loss.
    expense_loading : float
        Absolute expense load in currency units (expense_load × ECL).
    profit_loading : float
        Absolute profit load in currency units (profit_load × ECL).
    capital_load : float
        Absolute capital load in currency units
        (cost_of_capital × max(TVaR99 - ECL, 0)).
    technical_premium : float
        Sum of all four components — the final risk-adjusted premium.
    rate_on_line : float
        Technical premium as a fraction of the treaty limit.
        Standard reinsurance market metric for comparing layer pricing.
    treaty_limit : float
        Treaty limit used to compute the rate on line.
    """

    expected_ceded_loss: float
    expense_loading: float
    profit_loading: float
    capital_load: float
    technical_premium: float
    rate_on_line: float
    treaty_limit: float

    def summary(self) -> str:
        """Return a formatted breakdown of all premium components."""
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
    Computes a risk-adjusted technical premium from simulation results.

    The pricing formula follows a cost-of-capital approach:

        Premium = ECL
                + expense_load  × ECL          (expense recovery)
                + profit_load   × ECL          (target profit margin)
                + cost_of_capital × max(TVaR99 - ECL, 0)  (capital charge)

    The capital charge compensates the reinsurer for holding risk capital
    against unexpected losses. The unexpected loss is proxied by the
    difference between TVaR99 and ECL — the portion of the tail that
    exceeds the expected loss and cannot be funded from premium alone.

    Parameters
    ----------
    expected_ceded_loss : float
        Mean simulated ceded loss from MonteCarloEngine.run().
        This is the pure premium — the minimum the reinsurer must
        charge to break even on average. Must be non-negative.
    tvar_99 : float
        TVaR at 99% confidence from SimulationResults.tvar_99.
        Used as the capital proxy — represents the average loss in
        the worst 1% of years, which defines the capital requirement.
    expense_load : float
        Proportional expense load as a fraction of ECL.
        Covers acquisition costs, brokerage, and administration.
        Default is 0.05 (5%).
    profit_load : float
        Target profit margin as a fraction of ECL.
        Default is 0.08 (8%).
    cost_of_capital : float
        Required return on risk capital as a fraction.
        Typically set to the reinsurer's internal hurdle rate.
        Default is 0.10 (10%), consistent with typical Solvency II
        cost-of-capital assumptions.
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
        """
        Compute the technical premium.

        Sums all four components:
            ECL + expense loading + profit loading + capital load

        Returns
        -------
        float
            The risk-adjusted technical premium in currency units.
        """
        ecl     = self.expected_ceded_loss
        expense = self.expense_load * ecl
        profit  = self.profit_load * ecl

        # Capital load: required return on the unexpected loss only.
        # max(..., 0) ensures no negative capital load if TVaR < ECL
        # (can happen with very short tails or few simulations).
        capital = self.cost_of_capital * max(self.tvar_99 - ecl, 0)

        return ecl + expense + profit + capital

    def rate_on_line(self, treaty_limit: float) -> float:
        """
        Rate on Line (ROL) — technical premium as a fraction of treaty limit.

        The standard reinsurance market metric for comparing the price
        of layers with different limits. A ROL of 10% means the
        reinsurer charges 10% of the limit as premium.

        Parameters
        ----------
        treaty_limit : float
            The maximum ceded loss under the treaty — the limit for XL
            or the cap for Stop-Loss.

        Returns
        -------
        float
            Technical premium divided by treaty limit.
        """
        return self.technical_premium() / treaty_limit

    def price(self, treaty_limit: float) -> PricingResult:
        """
        Compute the full pricing breakdown and return a PricingResult.

        Parameters
        ----------
        treaty_limit : float
            Treaty limit used to compute the rate on line.

        Returns
        -------
        PricingResult
            Dataclass containing all premium components and the
            rate on line, ready for reporting or further analysis.
        """
        ecl     = self.expected_ceded_loss
        expense = self.expense_load * ecl
        profit  = self.profit_load * ecl
        capital = self.cost_of_capital * max(self.tvar_99 - ecl, 0)
        premium = ecl + expense + profit + capital

        return PricingResult(
            expected_ceded_loss=ecl,
            expense_loading=expense,
            profit_loading=profit,
            capital_load=capital,
            technical_premium=premium,
            rate_on_line=premium / treaty_limit,
            treaty_limit=treaty_limit,
        )